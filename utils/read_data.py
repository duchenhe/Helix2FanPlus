import os
import struct
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pydicom
import tqdm


HEADER_AND_PIXEL_TAGS = [
    "Manufacturer",
    "Rows",
    "Columns",
    "RescaleIntercept",
    "RescaleSlope",
    "PixelData",
    0x70311001,
    0x70311002,
    0x70291002,
    0x70291006,
    0x70311033,
    0x70311003,
    0x70311031,
    0x70411001,
]
# 这里列出的 tag 足以支持：
# 1. 原始投影像素读取
# 2. helical -> fan 重排所需的关键采集几何恢复
# 同时避免解析整份 DICOM 带来的额外开销。


def unpack_tag(data, tag):
    # 这个数据集的若干私有 tag 用 4-byte float 存储，需要手动解包。
    return struct.unpack("f", data[tag].value)[0]


def _resolve_vendor_flag(flag, manufacturer, auto_value):
    # 允许显式覆盖，也允许根据厂商走默认策略。
    if isinstance(flag, bool):
        return flag

    flag = str(flag).lower()
    if flag == "true":
        return True
    if flag == "false":
        return False
    if flag != "auto":
        raise ValueError(f"Unsupported flag value: {flag}")
    return auto_value(manufacturer)


def _needs_detector_row_flip(manufacturer):
    """这个数据集里 Siemens 与 GE 的探测器行方向定义不同。"""
    manufacturer = (manufacturer or "").upper()
    return manufacturer.startswith("SIEMENS")


def _read_projection_dataset(path, flip_detector_rows):
    """单次读取文件，同时保留几何解析所需 tag 和对应像素数据。

    这里刻意只读取最小必要标签集：
    1. 减少 pydicom 对整份 DICOM 的解析开销
    2. 避免后续为了读 header 又把同一个文件重复打开
    """
    dataset = pydicom.dcmread(path, specific_tags=HEADER_AND_PIXEL_TAGS)
    proj_array = np.frombuffer(dataset.PixelData, "H").astype(np.float32)
    proj_array = proj_array.reshape([dataset.Columns, dataset.Rows], order="F")
    proj_array *= dataset.RescaleSlope
    proj_array += dataset.RescaleIntercept

    if flip_detector_rows:
        proj_array = proj_array[:, ::-1]

    return dataset, proj_array


def read_projections(folder, indices, detector_row_flip="auto", n_jobs=1):
    """读取 DICOM-CT-PD 投影，并按厂商约定处理探测器行方向。

    注意这里的数组布局是:
    - 维度 0: 投影视角
    - 维度 1: detector v / column
    - 维度 2: detector u / row
    这与后续 rebin 函数中的索引约定保持一致。
    """
    # DICOM-CT-PD 数据通常以大量小文件形式存放，读取时首先固定文件顺序。
    file_names = sorted([f for f in os.listdir(folder) if f.endswith(".dcm")])
    if not file_names:
        raise ValueError(f"No DICOM files found in {folder}")

    file_paths = [os.path.join(folder, file_name) for file_name in file_names[indices]]
    if not file_paths:
        raise ValueError("No DICOM files selected after applying the projection index range.")

    # 先用极轻量 header 读取厂商信息，决定是否需要翻转探测器行方向。
    first_dataset = pydicom.dcmread(file_paths[0], stop_before_pixels=True, specific_tags=["Manufacturer"])
    manufacturer = getattr(first_dataset, "Manufacturer", "")
    flip_detector_rows = _resolve_vendor_flag(detector_row_flip, manufacturer, _needs_detector_row_flip)

    # 第一帧单独读取，用来确定最终数组形状；剩余帧再统一走迭代读取。
    first_dataset, first_projection = _read_projection_dataset(file_paths[0], flip_detector_rows)
    datasets = [None] * len(file_paths)
    datasets[0] = first_dataset

    # 预分配完整输出数组，避免边读边扩容带来的重复拷贝。
    raw_projections = np.empty(
        (len(file_paths), first_dataset.Columns, first_dataset.Rows),
        dtype=np.float32,
    )
    raw_projections[0] = first_projection

    if len(file_paths) == 1:
        return datasets, raw_projections, bool(flip_detector_rows)

    # 对“大量小文件”的 DICOM 读取，线程太多通常会因为文件系统争用变慢。
    # 这里把读取并发限制得比重排更保守，是实测后得到的折中。
    max_workers = min(max(1, int(n_jobs)), 2)
    read_fn = lambda path: _read_projection_dataset(path, flip_detector_rows)

    if max_workers == 1:
        iterator = map(read_fn, file_paths[1:])
    else:
        executor = ThreadPoolExecutor(max_workers=max_workers)
        iterator = executor.map(read_fn, file_paths[1:])

    try:
        for i, (dataset, projection) in enumerate(
            tqdm.tqdm(iterator, total=len(file_paths) - 1, desc="Loading projection data"),
            start=1,
        ):
            datasets[i] = dataset
            raw_projections[i] = projection
    finally:
        if max_workers > 1:
            executor.shutdown(wait=True)

    return datasets, raw_projections, bool(flip_detector_rows)


def _build_geometry(headers, indices, resolved_detector_row_flip):
    """从 DICOM 头里提取后续 rebin 与重建真正会用到的几何量。

    这里恢复的是“采集几何”而不是病人坐标系下的 volume pose。
    例如 angles / z_positions / dso / dsd 等量，都是为投影重排服务的。
    """
    # 原始角度定义与后续 fan-beam 重建坐标系不同，这里统一转换成单调展开后的内部约定。
    angles = np.array([unpack_tag(d, 0x70311001) for d in headers], dtype=np.float32) + (np.pi / 2)
    angles = -np.unwrap(angles) - np.pi
    z_positions = np.array([unpack_tag(d, 0x70311002) for d in headers], dtype=np.float32)

    manufacturer = getattr(headers[0], "Manufacturer", "")
    nu = int(headers[0].Rows)
    nv = int(headers[0].Columns)
    du = float(unpack_tag(headers[0], 0x70291002))
    dv = float(unpack_tag(headers[0], 0x70291006))
    # 当前项目固定使用 1 mm 的 rebinned z 采样间隔。
    dv_rebinned = 1.0
    det_central_element = np.array(struct.unpack("2f", headers[0][0x70311033].value), dtype=np.float32)
    dso = float(unpack_tag(headers[0], 0x70311003))
    dsd = float(unpack_tag(headers[0], 0x70311031))
    ddo = float(dsd - dso)
    # 原始数据里并不总是直接给出每转床移，因此这里根据总 z 位移和总旋转圈数反推 pitch。
    pitch = float((z_positions[-1] - z_positions[0]) / ((angles[-1] - angles[0]) / (2 * np.pi)))
    nz_rebinned = int(abs(z_positions[-1] - z_positions[0]) / dv_rebinned)
    hu_factor = float(headers[0][0x70411001].value)
    # rotview 表示一整圈 2pi 范围内等价 fan angle 的数量，
    # 后续 helical -> fan 重排需要依赖这个周期结构。
    rotview = int(len(angles) / ((angles[-1] - angles[0]) / (2 * np.pi)))

    return {
        "indices": [indices.start, indices.stop],
        "nu": nu,
        "nv": nv,
        "du": du,
        "dv": dv,
        "dv_rebinned": dv_rebinned,
        "det_central_element": det_central_element.tolist(),
        "dso": dso,
        "dsd": dsd,
        "ddo": ddo,
        "manufacturer": manufacturer,
        "resolved_detector_row_flip": resolved_detector_row_flip,
        "pitch": pitch,
        "nz_rebinned": nz_rebinned,
        "rotview": rotview,
        "hu_factor": hu_factor,
        "angles": angles.tolist(),
        "z_positions": z_positions.tolist(),
    }


def read_dicom(args):
    """读取投影数据，并返回原始投影和几何 metadata。

    返回的 geometry 会直接写入 TIFF metadata，供后续 fan-beam 重建阶段复用。
    """
    indices = slice(args.idx_proj_start, args.idx_proj_stop)
    headers, raw_projections, resolved_detector_row_flip = read_projections(
        args.path_dicom,
        indices,
        detector_row_flip="auto",
        n_jobs=getattr(args, "n_jobs", 1),
    )
    geometry = _build_geometry(headers, indices, resolved_detector_row_flip)
    return raw_projections, geometry
