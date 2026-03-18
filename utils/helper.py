import json

import tifffile


def load_tiff_stack_with_metadata(file):
    """读取 TIFF 体数据及其附带的 JSON metadata。

    本项目把几何参数直接写在 TIFF 的 `ImageDescription` 字段中，
    因此这里会同时返回像素数组和解析后的 metadata 字典。
    """
    if not (file.name.endswith(".tif") or file.name.endswith(".tiff")):
        raise FileNotFoundError("File has to be tif.")

    with tifffile.TiffFile(file) as tif:
        data = tif.asarray()
        metadata = tif.pages[0].tags["ImageDescription"].value

    # 优先按标准 JSON 解析；如果历史文件里使用了 Python dict 风格字符串，
    # 则退化到一次简单的引号修正，以兼容旧输出。
    try:
        metadata = json.loads(metadata)
    except json.JSONDecodeError:
        metadata = metadata.replace("'", '"')
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            print("The tiff file you try to open does not seem to have metadata attached.")
            metadata = None
    except KeyError:
        metadata = None

    return data, metadata


def save_to_tiff_stack(array, file):
    """保存不带 metadata 的 TIFF stack。"""
    if not file.parent.is_dir():
        file.parent.mkdir(parents=True, exist_ok=True)
    if not (file.name.endswith(".tif") or file.name.endswith(".tiff")):
        raise FileNotFoundError("File has to be tif.")
    tifffile.imwrite(file, array)


def save_to_tiff_stack_with_metadata(array, file, metadata):
    """保存带 metadata 的 TIFF stack。

    metadata 会序列化为 JSON 并写入 TIFF description，便于后续重建直接复用。
    """
    if not file.parent.is_dir():
        file.parent.mkdir(parents=True, exist_ok=True)
    if not (file.name.endswith(".tif") or file.name.endswith(".tiff")):
        raise FileNotFoundError("File has to be tif.")
    metadata_json = json.dumps(metadata, ensure_ascii=False)
    tifffile.imwrite(file, array, description=metadata_json)


def save_to_json(params, file):
    """把参数字典保存为 UTF-8 JSON。"""
    if not file.parent.is_dir():
        file.parent.mkdir(parents=True, exist_ok=True)
    with open(file, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
