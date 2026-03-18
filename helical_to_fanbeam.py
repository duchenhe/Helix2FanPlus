import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import tqdm
from joblib import Parallel, delayed

from utils.helper import save_to_tiff_stack_with_metadata
from utils.read_data import read_dicom
from utils.rebinning_functions import (
    build_curved_to_flat_cache,
    build_fan_rebin_cache,
    rebin_curved_to_flat_detector,
    rebin_curved_to_flat_detector_single_angle,
    rebin_helical_to_fan_beam_trajectory,
    rebin_helical_to_fan_beam_trajectory_single_angle,
)


def _attach_geometry(args, geometry):
    """把读取阶段恢复出的几何量附加到 argparse namespace。"""
    # 把 DICOM 头里解析出的几何量挂回 args，后续重排函数统一从 args 读取。
    for key, value in geometry.items():
        setattr(args, key, value)


def _split_indices(length, n_jobs):
    """把角度维均匀切成若干静态块。"""
    # 并行时按角度维做静态分块，避免任务粒度过细导致调度开销变大。
    n_chunks = max(1, min(int(n_jobs), int(length)))
    return [chunk for chunk in np.array_split(np.arange(length, dtype=np.int32), n_chunks) if len(chunk) > 0]


def _rebin_curved_chunk(args, raw_projections, proj_flat_detector, angle_indices, cache):
    """线程工作函数：处理一段曲面到平面的视角块。"""
    # 每个线程负责一段投影视角；输出直接写入共享数组，避免额外拼接拷贝。
    for i_angle in angle_indices:
        proj_flat_detector[int(i_angle)] = rebin_curved_to_flat_detector_single_angle(
            args,
            raw_projections,
            int(i_angle),
            cache=cache,
        )


def _parallel_rebin_curved_to_flat(args, raw_projections):
    """多线程执行 curved-detector -> flat-detector 重排。"""
    # 曲面到平面的映射与视角无关，所以索引/权重只预计算一次。
    cache = build_curved_to_flat_cache(args)
    angle_chunks = _split_indices(raw_projections.shape[0], args.n_jobs)
    proj_flat_detector = np.empty_like(raw_projections, dtype=np.float32)
    Parallel(n_jobs=args.n_jobs, prefer="threads", require="sharedmem")(
        delayed(_rebin_curved_chunk)(
            args,
            raw_projections,
            proj_flat_detector,
            angle_chunk,
            cache,
        )
        for angle_chunk in tqdm.tqdm(angle_chunks, desc="Rebin curved to flat detector")
    )
    return proj_flat_detector


def _rebin_fan_chunk(args, proj_flat_detector, proj_fan_geometry, s_angles, cache):
    """线程工作函数：处理一段 fan-angle 的 SSRB 重排。"""
    # 每个线程处理若干 rebinned fan angle，同样直接写入目标数组。
    for s_angle in s_angles:
        proj_fan_geometry[int(s_angle)] = rebin_helical_to_fan_beam_trajectory_single_angle(
            args,
            proj_flat_detector,
            int(s_angle),
            cache=cache,
        )


def _parallel_rebin_helical_to_fan(args, proj_flat_detector):
    """多线程执行 helical cone-beam -> 2D fan-beam 重排。"""
    # SSRB 里与几何相关的中间量可复用，因此统一缓存后再并行。
    cache = build_fan_rebin_cache(args)
    s_angle_chunks = _split_indices(args.rotview, args.n_jobs)
    proj_fan_geometry = np.empty((args.rotview, args.nu, args.nz_rebinned), dtype=np.float32)
    Parallel(n_jobs=args.n_jobs, prefer="threads", require="sharedmem")(
        delayed(_rebin_fan_chunk)(
            args,
            proj_flat_detector,
            proj_fan_geometry,
            s_angle_chunk,
            cache,
        )
        for s_angle_chunk in tqdm.tqdm(s_angle_chunks, desc="Rebin helical to fan-beam geometry")
    )
    return proj_fan_geometry


def run(args):
    """执行完整的重排主流程。

    流程包括：
    1. 读取 DICOM-CT-PD 原始投影及几何参数
    2. 曲面探测器重排到虚拟平面探测器
    3. 螺旋锥束轨迹重排为逐层 2D fan-beam 轨迹
    4. 把结果以 TIFF/NIfTI 形式写出，供后续重建脚本使用
    """
    print(f"Processing scan {args.scan_id}.")

    # 1. 读取 DICOM-CT-PD 原始投影，并从私有 tag 中恢复后续重排所需的扫描几何。
    raw_projections, geometry = read_dicom(args)
    _attach_geometry(args, geometry)

    if args.save_all:
        save_path = Path(args.path_out) / f"{args.scan_id}_curved_helix_projections.tif"
        save_to_tiff_stack_with_metadata(raw_projections, save_path, metadata=vars(args))

    # 2. 先把 scanner 的曲面探测器采样重排到虚拟平面探测器。
    # 这样做的目的不是改变物理投影，而是为了适配后续 fan-beam 重建工具常见的 flat detector 假设。
    if args.no_multiprocessing or args.n_jobs == 1:
        proj_flat_detector = rebin_curved_to_flat_detector(args, raw_projections)
    else:
        proj_flat_detector = _parallel_rebin_curved_to_flat(args, raw_projections)

    if args.save_all:
        save_path = Path(args.path_out) / f"{args.scan_id}_flat_helix_projections.tif"
        save_to_tiff_stack_with_metadata(proj_flat_detector, save_path, metadata=vars(args))
        sitk.WriteImage(
            sitk.GetImageFromArray(proj_flat_detector),
            f"{args.path_out}/{args.scan_id}_flat_helix_projections.nii.gz",
        )

    # 3. 按参考文献 `@article{Frédéric Noo_1999}` 的 single-slice rebinning 思路，
    #    把螺旋轨迹投影重排为 2D fan-beam 轨迹。
    # 每个输出 fan angle 对应原始螺旋轨迹上同一方位角的一串 source 位置，再沿 z 方向做重采样。
    if args.no_multiprocessing or args.n_jobs == 1:
        proj_fan_geometry = rebin_helical_to_fan_beam_trajectory(args, proj_flat_detector)
    else:
        proj_fan_geometry = _parallel_rebin_helical_to_fan(args, proj_flat_detector)

    save_path = Path(args.path_out) / f"{args.scan_id}_flat_fan_projections.tif"
    save_to_tiff_stack_with_metadata(proj_fan_geometry, save_path, metadata=vars(args))
    sitk.WriteImage(
        sitk.GetImageFromArray(proj_fan_geometry),
        f"{args.path_out}/{args.scan_id}_flat_fan_projections.nii.gz",
    )

    print(f"Finished. Results saved at {save_path.resolve()}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_dicom", type=str, required=True, help="Local path of helical projection data.")
    parser.add_argument("--path_out", type=str, default="out", help="Output path of rebinned data.")
    parser.add_argument("--scan_id", type=str, default="scan_001", help="Custom scan ID.")
    parser.add_argument(
        "--idx_proj_start",
        type=int,
        default=0,
        help="First index of helical projections that are processed.",
    )
    parser.add_argument(
        "--idx_proj_stop",
        type=int,
        default=48590,
        help="Last index of helical projections that are processed.",
    )
    parser.add_argument("--save_all", dest="save_all", action="store_true", help="Save intermediate results.")
    parser.add_argument(
        "--no_multiprocessing",
        dest="no_multiprocessing",
        action="store_true",
        help="Switch off multithreading and run serially.",
    )
    parser.add_argument("--n_jobs", type=int, default=8, help="Number of CPU threads used for rebinning.")
    run(parser.parse_args())
