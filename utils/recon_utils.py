from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def resolve_flip_u_setting(mode, metadata):
    """根据厂商或显式参数决定是否翻转 detector-u。"""
    manufacturer = str(metadata.get("manufacturer", "")).upper()
    if mode == "true":
        return True
    if mode == "false":
        return False
    return manufacturer.startswith("SIEMENS")


def resolve_device(requested_device):
    """解析重建设备。当前工程使用的是 CUDA 版 torch-radon。"""
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("`--device cuda` was requested, but CUDA is not available.")
        return "cuda"
    if torch.cuda.is_available():
        return "cuda"
    raise RuntimeError("CUDA is not available. `torch-radon` reconstruction in this script requires a CUDA device.")


def require_projection_metadata(metadata):
    """检查 fan-beam TIFF metadata 是否包含重建所需几何量。"""
    if metadata is None:
        raise ValueError("Projection TIFF metadata is missing.")

    required_keys = ("angles", "rotview", "dso", "ddo", "du")
    missing_keys = [key for key in required_keys if key not in metadata]
    if missing_keys:
        raise KeyError(f"Projection metadata is missing required keys: {missing_keys}")


def get_signed_z_step(metadata):
    """计算带方向符号的 z spacing。

    这里的符号仅用于保持重建体在 z 轴上的排列方向与重排输出一致，
    不等价于病人坐标系中的 superior/inferior 方向定义。
    """
    z_positions = np.asarray(metadata.get("z_positions", []), dtype=np.float32)
    if z_positions.size >= 2:
        z_direction = np.sign(z_positions[-1] - z_positions[0])
    else:
        z_direction = 1.0

    if z_direction == 0:
        z_direction = np.sign(float(metadata.get("pitch", 0.0)))
    if z_direction == 0:
        z_direction = 1.0

    return float(metadata.get("dv_rebinned", 1.0)) * float(z_direction)


def build_recon_geometry(voxel_size, metadata):
    """为输出 NIfTI 构造一套自洽的算法坐标系。

    由于 raw projection header 不能直接恢复 patient-space origin，
    这里显式采用“零原点 + 正确 spacing/direction”的工程化约定。
    这样做的目标是保证体素尺寸和轴向顺序正确，而不是伪造临床坐标。
    """
    signed_z_step = get_signed_z_step(metadata)
    spacing = (float(voxel_size), float(voxel_size), abs(signed_z_step))
    origin = (0.0, 0.0, 0.0)
    direction = (
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0 if signed_z_step >= 0 else -1.0,
    )
    return spacing, origin, direction


def set_image_geometry(image, spacing, origin, direction):
    """把重建体的几何信息写回 SimpleITK 图像对象。"""
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image.SetDirection(direction)


def format_float_for_tag(value):
    """把浮点参数转成适合文件名的稳定字符串。"""
    return str(value).replace(".", "p")


def build_recon_tag(method, fbp_filter, num_iters, relaxation, num_subsets, enforce_nonnegativity):
    """构造输出文件名使用的重建方法标签。"""
    method = str(method).lower()
    if method == "fbp":
        return fbp_filter

    parts = [method, f"iter{int(num_iters)}"]
    if method in {"art", "sart", "sirt"}:
        parts.append(f"rel{format_float_for_tag(relaxation)}")
    if method == "sart":
        parts.append(f"sub{int(num_subsets)}")
    if enforce_nonnegativity:
        parts.append("nn")
    return "_".join(parts)


def save_preview_figure(path, projections, recon_hu, method):
    """生成一张更适合展示的预览图。

    图中同时展示：
    1. 中间 z slice 对应的 fan-beam projection 预览
    2. 中间重建层面的 HU 图像
    3. 两者的 shape，便于快速核对输入输出张量布局
    """
    path = Path(path)
    proj_mid = np.transpose(projections[:, :, projections.shape[2] // 2])
    recon_mid = recon_hu[recon_hu.shape[0] // 2]

    bg_color = "#f4f1ea"
    card_color = "#fbfaf7"
    edge_color = "#d9d2c5"
    title_color = "#1f2a2e"
    text_color = "#5d6a6f"
    accent_color = "#9a6a4a"

    # 采用浅暖色背景和卡片式布局，便于在 README 或实验记录中直接展示。
    fig = plt.figure(figsize=(11.5, 5.8), facecolor=bg_color)
    gs = fig.add_gridspec(1, 2, left=0.04, right=0.96, top=0.84, bottom=0.12, wspace=0.08)
    axes = [fig.add_subplot(gs[0, i]) for i in range(2)]

    fig.text(
        0.05,
        0.93,
        "Fan-beam Reconstruction Preview",
        fontsize=20,
        fontweight="bold",
        color=title_color,
        family="DejaVu Sans",
    )
    fig.text(
        0.05,
        0.885,
        f"Method: {str(method).upper()}",
        fontsize=10.5,
        color=accent_color,
        family="DejaVu Sans",
    )

    for ax in axes:
        ax.set_facecolor(card_color)
        for spine in ax.spines.values():
            spine.set_edgecolor(edge_color)
            spine.set_linewidth(1.2)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].imshow(proj_mid, cmap="bone")
    axes[0].set_title("Rebinned Projections", fontsize=14, color=title_color, pad=14, fontweight="semibold")
    axes[0].text(
        0.03,
        0.04,
        f"shape {tuple(projections.shape)}",
        transform=axes[0].transAxes,
        fontsize=10,
        color=text_color,
        family="DejaVu Sans Mono",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f1ece3", edgecolor=edge_color, linewidth=0.8),
    )

    axes[1].imshow(recon_mid, cmap="gray", vmin=-800, vmax=800)
    axes[1].set_title("Reconstruction", fontsize=14, color=title_color, pad=14, fontweight="semibold")
    axes[1].text(
        0.03,
        0.04,
        f"shape {tuple(recon_hu.shape)}",
        transform=axes[1].transAxes,
        fontsize=10,
        color=text_color,
        family="DejaVu Sans Mono",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f1ece3", edgecolor=edge_color, linewidth=0.8),
    )

    fig.text(
        0.05,
        0.06,
        "Left: middle fan-beam sinogram slice preview    Right: middle reconstructed slice in HU",
        fontsize=9.5,
        color=text_color,
        family="DejaVu Sans",
    )

    fig.savefig(path, dpi=320, bbox_inches="tight", pad_inches=0.2, facecolor=fig.get_facecolor())
    plt.close(fig)
