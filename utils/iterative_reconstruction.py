import numpy as np
import torch
from torch_radon import FanBeam
from torch_radon.volumes import Volume2D


def resolve_fbp_filter_name(filter_name):
    """统一项目内部与 torch-radon 的滤波器命名。"""
    filter_name = str(filter_name).lower()
    return "ramp" if filter_name == "ram-lak" else filter_name


def build_fanbeam_operator(metadata, det_count, image_size, voxel_size, device):
    """根据 fan-beam metadata 构造 torch-radon 算子。

    输入 metadata 中的距离和探测器采样间隔都以 mm 为单位；
    torch-radon 内部使用的是“以像素为单位”的离散体坐标，因此这里会按 voxel_size 做归一化。
    """
    angles = torch.as_tensor(
        np.asarray(metadata["angles"], dtype=np.float32)[: int(metadata["rotview"])] + (np.pi / 2),
        dtype=torch.float32,
        device=device,
    )

    volume = Volume2D(center=(0.0, 0.0), voxel_size=(float(voxel_size), float(voxel_size)))
    volume.set_size(int(image_size), int(image_size))

    return FanBeam(
        det_count=int(det_count),
        angles=angles,
        src_dist=(1.0 / float(voxel_size)) * float(metadata["dso"]),
        det_dist=(1.0 / float(voxel_size)) * float(metadata["ddo"]),
        det_spacing=(1.0 / float(voxel_size)) * float(metadata["du"]),
        volume=volume,
    )


def prepare_sinogram_batch(projections, start_idx, stop_idx, flip_u_for_recon, voxel_size, device):
    """把 [angle, detector, slice] 的 numpy 数据转成 [batch, angle, detector] 的 torch tensor。

    这是当前项目里重建算子要求的标准输入布局。
    每个 batch 对应若干个独立的 2D fan-beam slice。
    """
    batch = projections[:, :, start_idx:stop_idx]
    if flip_u_for_recon:
        batch = np.flip(batch, axis=1)
    batch = np.transpose(batch, (2, 0, 1)).astype(np.float32, copy=False)
    batch = batch * (1.0 / float(voxel_size))
    return torch.as_tensor(batch, dtype=torch.float32, device=device).contiguous()


def _safe_reciprocal(x, eps=1e-6):
    """对接近 0 的元素做保护性倒数，避免归一化权重数值爆炸。"""
    out = torch.zeros_like(x)
    mask = torch.abs(x) > eps
    out[mask] = 1.0 / x[mask]
    return out


def _sum_of_squares(x):
    """按样本维之外的所有维度求平方和。"""
    dims = tuple(range(1, x.ndim))
    return torch.sum(x * x, dim=dims, keepdim=True)


def _build_ordered_subsets(num_angles, num_subsets):
    """构造 ordered subsets，尽量让每个 subset 的角度在全角度范围内均匀分布。"""
    num_subsets = max(1, min(int(num_subsets), int(num_angles)))
    return [torch.arange(i, num_angles, num_subsets, dtype=torch.long) for i in range(num_subsets)]


def _compute_subset_weights(radon, subset_indices, image_size, det_count, device):
    """计算 SIRT/SART/ART 中常用的行归一化和列归一化权重。

    - row_weights: 用于补偿不同射线/角度采样路径长度差异
    - col_weights: 用于补偿不同像素被 backproject 的累积权重差异
    """
    subset_indices = subset_indices.to(device=device)
    subset_angles = radon.angles[subset_indices]
    ones_image = torch.ones((1, image_size, image_size), dtype=torch.float32, device=device)
    ones_sinogram = torch.ones((1, len(subset_indices), det_count), dtype=torch.float32, device=device)

    projection_norm = radon.forward(ones_image, angles=subset_angles)
    backprojection_norm = radon.backward(ones_sinogram, angles=subset_angles)

    return {
        "angles": subset_angles,
        "row_weights": _safe_reciprocal(projection_norm),
        "col_weights": _safe_reciprocal(backprojection_norm),
        "backprojection_norm": backprojection_norm,
    }


def reconstruct_fbp(radon, sinograms, fbp_filter):
    """经典滤波反投影重建。"""
    filtered_sinogram = radon.filter_sinogram(sinograms, filter_name=resolve_fbp_filter_name(fbp_filter))
    return radon.backward(filtered_sinogram)


def reconstruct_cg(radon, sinograms, image_size, num_iters, enforce_nonnegativity=False):
    """最小二乘意义下的共轭梯度实现，具体采用 CGLS/CGNR 形式。"""
    x = torch.zeros((sinograms.shape[0], image_size, image_size), dtype=sinograms.dtype, device=sinograms.device)
    residual = sinograms.clone()
    gradient = radon.backward(residual)
    direction = gradient.clone()
    gamma = _sum_of_squares(gradient)

    for _ in range(int(num_iters)):
        forward_direction = radon.forward(direction)
        denom = _sum_of_squares(forward_direction).clamp_min(1e-6)
        alpha = gamma / denom
        x = x + alpha * direction
        if enforce_nonnegativity:
            x.clamp_min_(0.0)

        residual = residual - alpha * forward_direction
        gradient = radon.backward(residual)
        gamma_new = _sum_of_squares(gradient)
        beta = gamma_new / gamma.clamp_min(1e-6)
        direction = gradient + beta * direction
        gamma = gamma_new

    return x


def reconstruct_sirt(radon, sinograms, image_size, det_count, num_iters, relaxation, enforce_nonnegativity=False):
    """Simultaneous Iterative Reconstruction Technique。

    每一轮都使用全部角度同时更新，因此通常更稳定，但单次迭代的收敛速度偏慢。
    """
    x = torch.zeros((sinograms.shape[0], image_size, image_size), dtype=sinograms.dtype, device=sinograms.device)
    weights = _compute_subset_weights(
        radon,
        torch.arange(sinograms.shape[1], dtype=torch.long, device=sinograms.device),
        image_size,
        det_count,
        sinograms.device,
    )

    for _ in range(int(num_iters)):
        residual = sinograms - radon.forward(x)
        correction = radon.backward(weights["row_weights"] * residual)
        x = x + float(relaxation) * weights["col_weights"] * correction
        if enforce_nonnegativity:
            x.clamp_min_(0.0)

    return x


def reconstruct_sart(
    radon,
    sinograms,
    image_size,
    det_count,
    num_iters,
    relaxation,
    num_subsets,
    enforce_nonnegativity=False,
):
    """Simultaneous Algebraic Reconstruction Technique with ordered subsets。

    SART 可以看作在 ART 和 SIRT 之间做折中：
    每次使用一个 subset 的角度进行更新，通常具有更快的视觉收敛速度。
    """
    x = torch.zeros((sinograms.shape[0], image_size, image_size), dtype=sinograms.dtype, device=sinograms.device)
    subset_indices_list = _build_ordered_subsets(sinograms.shape[1], num_subsets)
    subset_weights = [
        _compute_subset_weights(radon, subset_indices, image_size, det_count, sinograms.device)
        for subset_indices in subset_indices_list
    ]

    for _ in range(int(num_iters)):
        for subset_indices, weights in zip(subset_indices_list, subset_weights):
            subset_indices = subset_indices.to(device=sinograms.device)
            subset_sinogram = sinograms[:, subset_indices, :]
            residual = subset_sinogram - radon.forward(x, angles=weights["angles"])
            correction = radon.backward(weights["row_weights"] * residual, angles=weights["angles"])
            x = x + float(relaxation) * weights["col_weights"] * correction
            if enforce_nonnegativity:
                x.clamp_min_(0.0)

    return x


def reconstruct_art(radon, sinograms, image_size, det_count, num_iters, relaxation, enforce_nonnegativity=False):
    """顺序 ART。

    这里按“单个投影视角”为单位做 Kaczmarz 风格更新，而不是细到单条射线。
    对当前 fan-beam 算子接口来说，这是更实际也更稳定的实现方式。
    """
    x = torch.zeros((sinograms.shape[0], image_size, image_size), dtype=sinograms.dtype, device=sinograms.device)
    # 这里把单个投影视角视为一个最小更新单元，便于和当前 fan-beam 算子接口对齐。
    subset_indices_list = [torch.tensor([i], dtype=torch.long) for i in range(sinograms.shape[1])]
    subset_weights = [
        _compute_subset_weights(radon, subset_indices, image_size, det_count, sinograms.device)
        for subset_indices in subset_indices_list
    ]

    for _ in range(int(num_iters)):
        for subset_indices, weights in zip(subset_indices_list, subset_weights):
            subset_indices = subset_indices.to(device=sinograms.device)
            subset_sinogram = sinograms[:, subset_indices, :]
            residual = subset_sinogram - radon.forward(x, angles=weights["angles"])
            correction = radon.backward(weights["row_weights"] * residual, angles=weights["angles"])

            # ART 不做逐像素列归一化，而是使用单视角 backprojection norm 的最大值做整体缩放。
            scale = 1.0 / torch.amax(weights["backprojection_norm"]).clamp_min(1e-6)
            x = x + float(relaxation) * scale * correction
            if enforce_nonnegativity:
                x.clamp_min_(0.0)

    return x


def reconstruct_batch(
    radon,
    sinograms,
    method,
    image_size,
    det_count,
    num_iters,
    relaxation,
    num_subsets,
    fbp_filter,
    enforce_nonnegativity=False,
):
    """按指定方法调度单个 batch 的 2D fan-beam 重建。"""
    method = str(method).lower()

    if method == "fbp":
        return reconstruct_fbp(radon, sinograms, fbp_filter)
    if method == "cg":
        return reconstruct_cg(radon, sinograms, image_size, num_iters, enforce_nonnegativity=enforce_nonnegativity)
    if method == "sirt":
        return reconstruct_sirt(
            radon,
            sinograms,
            image_size,
            det_count,
            num_iters,
            relaxation,
            enforce_nonnegativity=enforce_nonnegativity,
        )
    if method == "sart":
        return reconstruct_sart(
            radon,
            sinograms,
            image_size,
            det_count,
            num_iters,
            relaxation,
            num_subsets,
            enforce_nonnegativity=enforce_nonnegativity,
        )
    if method == "art":
        return reconstruct_art(
            radon,
            sinograms,
            image_size,
            det_count,
            num_iters,
            relaxation,
            enforce_nonnegativity=enforce_nonnegativity,
        )

    raise ValueError(f"Unsupported reconstruction method: {method}")
