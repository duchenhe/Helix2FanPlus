import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from tqdm import tqdm

from utils.helper import load_tiff_stack_with_metadata, save_to_json, save_to_tiff_stack
from utils.iterative_reconstruction import (
    build_fanbeam_operator,
    prepare_sinogram_batch,
    reconstruct_batch,
)
from utils.recon_utils import (
    build_recon_geometry,
    build_recon_tag,
    require_projection_metadata,
    resolve_device,
    resolve_flip_u_setting,
    save_preview_figure,
    set_image_geometry,
)


def _reconstruct_volume(args, projections, metadata, flip_u_for_recon, device):
    """按 batch 组织 slice，并调用统一的 fan-beam 重建后端。

    这里的输入 `projections` 仍然保持项目内部的存储约定:
    - 维度 0: angle
    - 维度 1: detector
    - 维度 2: slice

    在送入 torch-radon 前，会在 `prepare_sinogram_batch` 中转换成
    `[batch, angle, detector]` 的张量布局。
    """
    method = str(args.method).lower()
    batch_size = max(1, min(int(args.batch_size), projections.shape[2]))
    radon = build_fanbeam_operator(
        metadata,
        det_count=projections.shape[1],
        image_size=args.image_size,
        voxel_size=args.voxel_size,
        device=device,
    )

    # 逐 batch 重建的目的主要有两点：
    # 1. 控制 GPU 显存占用
    # 2. 避免完全逐 slice 循环带来的 Python 调度开销
    recon_batches = []
    batch_iterator = range(0, projections.shape[2], batch_size)
    desc = f"Reconstructing slices ({method})"

    for start_idx in tqdm(batch_iterator, desc=desc):
        stop_idx = min(start_idx + batch_size, projections.shape[2])
        sinogram_batch = prepare_sinogram_batch(
            projections,
            start_idx,
            stop_idx,
            flip_u_for_recon=flip_u_for_recon,
            voxel_size=args.voxel_size,
            device=device,
        )

        # 重建算子本身是线性的，不依赖 autograd；这里显式关闭梯度以减少额外开销。
        with torch.no_grad():
            recon_batch = reconstruct_batch(
                radon=radon,
                sinograms=sinogram_batch,
                method=method,
                image_size=args.image_size,
                det_count=projections.shape[1],
                num_iters=args.num_iters,
                relaxation=args.relaxation,
                num_subsets=args.num_subsets,
                fbp_filter=args.fbp_filter,
                enforce_nonnegativity=args.enforce_nonnegativity,
            )

        recon_batches.append(recon_batch.detach().cpu().numpy())

    # 每个 batch 的输出形状都是 [batch, H, W]，最终沿 slice 维拼回完整体数据。
    return np.concatenate(recon_batches, axis=0).astype(np.float32, copy=False)


def _convert_to_hu(recon, metadata):
    """按 DICOM-CT-PD 的 water attenuation coefficient 把线衰减系数换算为 HU。

    输出采用 int16，便于与常见 CT 体数据的 HU 存储习惯保持一致。
    """
    hu_factor = float(metadata.get("hu_factor", 0.0186))
    recon_hu = np.rint(1000.0 * ((recon - hu_factor) / hu_factor)).astype(np.int16)
    return recon_hu


def _save_reconstruction_outputs(path_out, scan_id, recon_tag, voxel_size, recon_hu, spacing, origin, direction):
    """保存重建结果。

    当前主输出聚焦在 HU 结果：
    - NIfTI: 用于三维查看和后续处理
    - TIFF: 便于快速检查与跨工具读取
    """
    recon_hu_image = sitk.GetImageFromArray(recon_hu)
    set_image_geometry(recon_hu_image, spacing, origin, direction)
    sitk.WriteImage(
        recon_hu_image,
        str(path_out / f"{scan_id}_recon_HU_{recon_tag}_{voxel_size}.nii.gz"),
    )

    save_to_tiff_stack(recon_hu, path_out / f"{scan_id}_recon_{recon_tag}.tif")


def _build_output_metadata(args, metadata, spacing, origin, direction, flip_u_for_recon, device):
    """整理并扩展输出 metadata，便于结果追踪和复现实验配置。"""
    metadata_out = dict(metadata)
    metadata_out.update(
        {
            "recon_method": str(args.method).lower(),
            "recon_num_iters": int(args.num_iters),
            "recon_relaxation": float(args.relaxation),
            "recon_num_subsets": int(args.num_subsets),
            "recon_batch_size": int(args.batch_size),
            "recon_enforce_nonnegativity": bool(args.enforce_nonnegativity),
            "recon_image_size": int(args.image_size),
            "recon_voxel_size": float(args.voxel_size),
            "recon_spacing": list(spacing),
            "recon_origin": list(origin),
            "recon_direction": list(direction),
            "recon_coordinate_system": "algorithmic volume with zero origin, not patient DICOM coordinates",
            "flip_u_for_recon_resolved": bool(flip_u_for_recon),
            "device": device,
            "fbp_filter": args.fbp_filter,
        }
    )
    return metadata_out


def run_reco(args):
    """重建脚本主入口。

    整体流程如下：
    1. 读取 fan-beam 投影与附带 metadata
    2. 解析重建方向、设备与输出标签
    3. 调用 fan-beam 重建核心算法
    4. 转换为 HU，并写出带几何信息的结果文件
    """
    path_out = Path(args.path_out)
    path_out.mkdir(parents=True, exist_ok=True)

    # 读取 helical_to_fanbeam.py 输出的 fan-beam TIFF 及其几何 metadata。
    projections, metadata = load_tiff_stack_with_metadata(Path(args.path_proj))
    require_projection_metadata(metadata)

    # detector-u 的方向约定与厂商有关，因此保留单独的解析步骤。
    flip_u_for_recon = resolve_flip_u_setting(args.flip_u_for_recon, metadata)
    device = resolve_device(args.device)
    recon_tag = build_recon_tag(
        method=args.method,
        fbp_filter=args.fbp_filter,
        num_iters=args.num_iters,
        relaxation=args.relaxation,
        num_subsets=args.num_subsets,
        enforce_nonnegativity=args.enforce_nonnegativity,
    )

    # 调用统一重建入口，具体方法由 `--method` 参数控制。
    recon = _reconstruct_volume(args, projections, metadata, flip_u_for_recon, device)
    recon_hu = _convert_to_hu(recon, metadata)

    # 为输出 NIfTI 附加一套自洽的算法坐标系几何信息。
    spacing, origin, direction = build_recon_geometry(args.voxel_size, metadata)

    _save_reconstruction_outputs(
        path_out=path_out,
        scan_id=args.scan_id,
        recon_tag=recon_tag,
        voxel_size=args.voxel_size,
        recon_hu=recon_hu,
        spacing=spacing,
        origin=origin,
        direction=direction,
    )

    metadata_out = _build_output_metadata(args, metadata, spacing, origin, direction, flip_u_for_recon, device)
    save_to_json(metadata_out, path_out / f"{args.scan_id}_metadata.json")

    # 同时生成一张适合快速浏览的预览图，用于展示中间投影与重建结果。
    save_preview_figure(path_out / "example_recon.png", projections, recon_hu, args.method)

    print(f"Reconstruction saved to {path_out}.")
    return projections, recon_hu, metadata_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_proj", type=str, required=True, help="Local path of fan beam projection data.")
    parser.add_argument("--path_out", type=str, default="out", help="Output path of rebinned data.")
    parser.add_argument("--scan_id", type=str, default="L067", help="Custom scan ID.")
    parser.add_argument("--image_size", type=int, default=512, help="Size of reconstructed image.")
    parser.add_argument("--voxel_size", type=float, default=1.0, help="In-slice voxel size [mm].")
    parser.add_argument(
        "--method",
        type=str,
        default="fbp",
        choices=["fbp", "cg", "art", "sart", "sirt"],
        help="Reconstruction algorithm.",
    )
    parser.add_argument(
        "--num_iters",
        type=int,
        default=20,
        help="Number of iterations for iterative methods.",
    )
    parser.add_argument(
        "--relaxation",
        type=float,
        default=1.0,
        help="Relaxation parameter used by ART/SART/SIRT.",
    )
    parser.add_argument(
        "--num_subsets",
        type=int,
        default=8,
        help="Number of ordered subsets used by SART.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of slices reconstructed together as one torch-radon batch.",
    )
    parser.add_argument(
        "--enforce_nonnegativity",
        action="store_true",
        help="Clamp iterative reconstruction updates to non-negative values.",
    )
    parser.add_argument(
        "--flip_u_for_recon",
        type=str,
        choices=["auto", "true", "false"],
        default="auto",
        help="Whether to flip detector-u before fan-beam reconstruction.",
    )
    parser.add_argument(
        "--fbp_filter",
        type=str,
        default="ram-lak",
        choices=["ram-lak", "ramp", "shepp-logan", "cosine", "hamming", "hann"],
        help="Filter used for FBP.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda"],
        help="Device used for torch-radon reconstruction.",
    )
    run_reco(parser.parse_args())
