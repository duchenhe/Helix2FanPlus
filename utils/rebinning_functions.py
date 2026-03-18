import numpy as np
import tqdm


def _detector_u_positions(args):
    # 采用“像素中心”坐标，而不是像素边界坐标，因此带有 0.5 像素偏移。
    return (np.arange(args.nu, dtype=np.float32) - args.nu / 2 + 0.5) * args.du


def _detector_v_positions(args):
    # v 方向与 u 方向同理，统一在物理长度单位 mm 下处理。
    return (np.arange(args.nv, dtype=np.float32) - args.nv / 2 + 0.5) * args.dv


def _build_curved_to_flat_cache(args):
    """预计算曲面探测器到平面探测器所需的双线性插值索引与权重。

    几何含义是：
    1. 先在虚拟 flat detector 上定义像素中心
    2. 从 source 向这些像素发射射线
    3. 找到对应射线与 curved detector 的交点
    4. 再把交点映射回离散曲面探测器索引上做双线性插值
    """
    x_det = _detector_u_positions(args)
    z_det = _detector_v_positions(args)
    x_grid, z_grid = np.meshgrid(x_det, z_det)

    radius = np.sqrt(x_grid * x_grid + args.dsd * args.dsd + z_grid * z_grid)
    phi_on_curved_det = np.arcsin(x_grid / radius)
    z_on_curved_det = (z_grid / radius) * args.dsd
    dphi_curved = 2.0 * np.arctan(args.du / (2.0 * args.dsd))

    col_map = phi_on_curved_det / dphi_curved + (args.nu - args.det_central_element[0])
    row_map = z_on_curved_det / args.dv + (args.nv - args.det_central_element[1])

    row0 = np.floor(row_map).astype(np.int32)
    col0 = np.floor(col_map).astype(np.int32)
    row1 = row0 + 1
    col1 = col0 + 1

    row_alpha = (row_map - row0).astype(np.float32)
    col_alpha = (col_map - col0).astype(np.float32)

    valid_a = (row0 >= 0) & (row0 < args.nv) & (col0 >= 0) & (col0 < args.nu)
    valid_b = (row0 >= 0) & (row0 < args.nv) & (col1 >= 0) & (col1 < args.nu)
    valid_c = (row1 >= 0) & (row1 < args.nv) & (col0 >= 0) & (col0 < args.nu)
    valid_d = (row1 >= 0) & (row1 < args.nv) & (col1 >= 0) & (col1 < args.nu)

    # 这里直接缓存双线性插值的四邻点和权重，后续每个角度只做数组索引和加权求和。
    return {
        "row0": np.clip(row0, 0, args.nv - 1),
        "row1": np.clip(row1, 0, args.nv - 1),
        "col0": np.clip(col0, 0, args.nu - 1),
        "col1": np.clip(col1, 0, args.nu - 1),
        "wa": ((1.0 - row_alpha) * (1.0 - col_alpha) * valid_a).astype(np.float32),
        "wb": ((1.0 - row_alpha) * col_alpha * valid_b).astype(np.float32),
        "wc": (row_alpha * (1.0 - col_alpha) * valid_c).astype(np.float32),
        "wd": (row_alpha * col_alpha * valid_d).astype(np.float32),
    }


def build_curved_to_flat_cache(args):
    """对外暴露的 cache 构造接口。

    主脚本在串行和并行两条路径里都可能复用同一份 cache，
    单独保留这个函数可以让调用侧不必依赖内部实现名。
    """
    return _build_curved_to_flat_cache(args)


def rebin_curved_to_flat_detector_single_angle(args, proj_curved_helic, i_angle, cache=None):
    """重排单个视角下的曲面探测器投影。"""
    if cache is None:
        cache = _build_curved_to_flat_cache(args)

    grid = proj_curved_helic[i_angle]
    return (
        cache["wa"] * grid[cache["row0"], cache["col0"]]
        + cache["wb"] * grid[cache["row0"], cache["col1"]]
        + cache["wc"] * grid[cache["row1"], cache["col0"]]
        + cache["wd"] * grid[cache["row1"], cache["col1"]]
    ).astype(np.float32)


def rebin_curved_to_flat_detector(args, proj_curved_helic):
    """曲面探测器到平面探测器重排。

    这一阶段只改变探测器参数化方式，不改变投影对应的 source 位置。
    """
    cache = _build_curved_to_flat_cache(args)
    proj_flat_helic = np.empty_like(proj_curved_helic, dtype=np.float32)

    for i_angle in tqdm.tqdm(range(proj_curved_helic.shape[0]), desc="Rebin curved to flat detector"):
        proj_flat_helic[i_angle] = rebin_curved_to_flat_detector_single_angle(
            args,
            proj_curved_helic,
            i_angle,
            cache=cache,
        )

    return proj_flat_helic


def _get_rebinned_z_step_from_positions(z_positions, pitch, dv_rebinned):
    # rebinned z 方向需要保留扫描方向符号，否则负向扫描会发生 slice 映射错误。
    z_direction = np.sign(z_positions[-1] - z_positions[0])
    if z_direction == 0:
        z_direction = np.sign(pitch)
    if z_direction == 0:
        z_direction = 1.0
    return float(dv_rebinned * z_direction)


def _fan_rebin_cache(args):
    """缓存 helical -> fan 重排中与视角无关的几何量。"""
    z_positions = np.asarray(args.z_positions, dtype=np.float32)
    z_step = _get_rebinned_z_step_from_positions(z_positions, args.pitch, args.dv_rebinned)
    z_resampled = z_positions[0] + np.arange(args.nz_rebinned, dtype=np.float32) * z_step

    u_positions = _detector_u_positions(args)
    scale_base = u_positions * u_positions + args.dsd * args.dsd
    return {
        "z_positions": z_positions,
        "z_step": z_step,
        "z_resampled": z_resampled,
        # full-scan SSRB 情况下，每个 source 对应的有效 z 覆盖范围取半个 pitch 的上下邻域。
        "half_pitch": float(0.5 * abs(args.pitch)),
        "scale_base": scale_base.astype(np.float32),
        "scale_num": np.sqrt(scale_base).astype(np.float32),
        "geom_factor": (scale_base / (args.dso * args.dsd)).astype(np.float32),
        "col_idx": np.arange(args.nu, dtype=np.int32)[None, :],
        "v_min": float((-args.nv / 2 + 0.5) * args.dv),
    }


def build_fan_rebin_cache(args):
    """对外暴露的 fan-beam 重排 cache 接口。"""
    return _fan_rebin_cache(args)


def _interp_detector_rows_clamped(projection, v_precise, cache, args):
    """在探测器 v 方向上做逐列线性插值。

    它与逐列 `np.interp(..., left=fp[0], right=fp[-1])` 等价，
    但这里写成向量化形式，避免 Python 层的 detector-column 循环。
    """
    # 先把物理坐标 v_precise 映射回离散 detector-row 索引。
    row_position = (v_precise - cache["v_min"]) / args.dv
    row_position = np.clip(row_position, 0.0, args.nv - 1.0)

    row0 = np.floor(row_position).astype(np.int32)
    row1 = np.minimum(row0 + 1, args.nv - 1)
    alpha = row_position - row0

    sample0 = projection[row0, cache["col_idx"]]
    sample1 = projection[row1, cache["col_idx"]]
    return ((1.0 - alpha) * sample0 + alpha * sample1).astype(np.float32)


def rebin_helical_to_fan_beam_trajectory_single_angle(args, proj_helic, s_angle, cache=None):
    """对单个扇束角做 SSRB 重排。

    固定一个输出 fan angle 后，会在原始螺旋投影里取同一方位角的所有 source 位置，
    再根据输出 z slice 落在哪个 source 的有效覆盖范围内，沿 detector v 方向做插值。
    """
    if cache is None:
        cache = _fan_rebin_cache(args)

    z_positions = cache["z_positions"]
    z_step = cache["z_step"]
    z_poses_resampled = cache["z_resampled"]
    # 输出布局采用 [detector_u, rebinned_z]，与主流程里最终 stack 的内存布局保持一致。
    proj_rebinned_s_angle = np.zeros((args.nu, args.nz_rebinned), dtype=np.float32)
    # 同一个 rebinned fan angle，对应原始螺旋轨迹中每转一次出现一次的同方位角投影。
    proj_indices = np.arange(s_angle, proj_helic.shape[0], args.rotview, dtype=np.int32)
    z_poses_valid = z_positions[proj_indices]

    for local_idx, z_source in enumerate(z_poses_valid):
        # 对给定 source，只在其 axial coverage 内写入数据。
        lower_lim = z_source - cache["half_pitch"]
        upper_lim = z_source + cache["half_pitch"]

        # 把物理 z 区间映射成 rebinned z 网格上的离散索引范围。
        idx_a = (lower_lim - z_positions[0]) / z_step
        idx_b = (upper_lim - z_positions[0]) / z_step
        i_lower_lim = int(np.clip(np.floor(min(idx_a, idx_b)), a_min=0, a_max=args.nz_rebinned))
        i_upper_lim = int(np.clip(np.ceil(max(idx_a, idx_b)), a_min=0, a_max=args.nz_rebinned))
        if i_lower_lim >= i_upper_lim:
            continue

        z_block = z_poses_resampled[i_lower_lim:i_upper_lim]
        delta_z = (z_source - z_block).astype(np.float32)[:, None]

        # 参考文献 `@article{Frédéric Noo_1999}` 中的 v_precise：
        # 给定输出 slice 后，对应到真实 helical projection 上的 detector v 位置。
        v_precise = delta_z * cache["geom_factor"][None, :]
        v_interp = _interp_detector_rows_clamped(
            proj_helic[proj_indices[local_idx]],
            v_precise,
            cache,
            args,
        )
        scale_den = np.sqrt(cache["scale_base"][None, :] + v_precise * v_precise)

        # 这里的比例因子对应论文中的几何校正项，用于补偿扇束与锥束参数化差异。
        block = (cache["scale_num"][None, :] / scale_den) * v_interp
        # 同一输出角度下，不同 source 会负责不同 z 段；这里直接覆盖各自负责的轴向区间。
        proj_rebinned_s_angle[:, i_lower_lim:i_upper_lim] = block.T

    return proj_rebinned_s_angle


def rebin_helical_to_fan_beam_trajectory(args, proj_helic):
    """螺旋锥束到 2D fan-beam 轨迹重排。"""
    cache = _fan_rebin_cache(args)
    proj_rebinned = np.empty((args.rotview, args.nu, args.nz_rebinned), dtype=np.float32)

    for s_angle in tqdm.tqdm(range(args.rotview), desc="Rebin helical to fan-beam geometry"):
        proj_rebinned[s_angle] = rebin_helical_to_fan_beam_trajectory_single_angle(
            args,
            proj_helic,
            s_angle,
            cache=cache,
        )

    return proj_rebinned
