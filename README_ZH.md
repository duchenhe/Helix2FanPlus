# 🌪️ Helix2Fan+

> **Fast & Enhanced** helical cone-beam to flat-detector fan-beam rebinning for `DICOM-CT-PD` raw projections

[English Version](README.md)

`Helix2Fan+` 是一个面向符合 [DICOM-CT-PD](https://doi.org/10.1118/1.4935406) 格式的原始投影数据的 CT 几何处理项目，用于将 **Helical Cone-Beam Projection （螺旋锥束投影）** 重排为 **Flat-detector 2D Fan-Beam Projection （平面探测器 2D 扇形束投影）**，并完成逐层 fan-beam FBP 重建。

`Helix2Fan+` 本质上是对原始仓库 [faebstn96/helix2fan](https://github.com/faebstn96/helix2fan) 的继承、清理与工程化增强版本。在核心方法和功能目标上，它与原始仓库保持一致，仍然围绕 **single-slice rebinning** 这条主线展开；本仓库的主要工作聚焦于：

- **提升对真实数据的兼容性**（特别是 GE 负向 z 扫描数据）
- **大幅优化重排速度（≈100倍）**，以更好地适应实际研究和开发中的需求。

它适合这样的工作流：

- 🔄 从原始螺旋投影出发，做几何重排
- 🩻 生成适合 fan-beam 重建器的 sinogram
- 🧠 快速验证几何、方向、spacing 与重建质量

_如果你在开发 CT 重建算法，希望在真实采集的 sinogram 数据上进行实验验证，但是又苦于缺少资源，那么希望这个项目能够帮助到你。_

![Helix2Fan+ reconstruction preview](Figures/example_recon.png)

---

## ✨ Overview

项目提供两项主功能：

### 1. `helical_to_fanbeam.py`

将螺旋锥束投影重排为平面探测器 fan-beam 投影。

### 2. `recon_from_rebined_fanbeam_sino.py`

对重排后的 fan-beam sinogram 做逐层 FBP 重建，并输出 HU 结果。

同时支持经典迭代重建方法：`CG`、`ART`、`SART`、`SIRT`.

---

## 🚀 Quick Start

如果你已经具备运行环境，可以先直接使用下面两条命令；依赖、项目背景和更多说明见后文。

### Step 1. Rebin helical projections

```bash
python helical_to_fanbeam.py \
  --path_dicom /path/to/dicom_ct_pd_folder \
  --path_out out \
  --scan_id scan_001 \
  --n_jobs 4
```

### Step 2. Reconstruct fan-beam slices

```bash
python recon_from_rebined_fanbeam_sino.py \
  --path_proj out/scan_001_flat_fan_projections.tif \
  --path_out out \
  --scan_id scan_001 \
  --image_size 512 \
  --voxel_size 1.0 \
  --method fbp \
  --fbp_filter hann
```

---

## 🧬 Project Lineage

这不是一个从零开始、另起炉灶的项目，而是一个**继承自原始 `helix2fan` 思路**的实用化 fork。

它继承了原始仓库最核心的内容：

- 从 `DICOM-CT-PD` 原始投影出发
- 将 helical cone-beam 数据重排为 flat-detector 2D fan-beam 数据
- 再将重排结果交给 2D fan-beam 重建器处理

如果你已经了解原始 `helix2fan`，那么可以把本仓库理解为：

> `Helix2Fan+` 是一个保持原始方法学主线不变、但在实际数据兼容性、运行效率和工程可读性上显著强化的版本。

---

## ⚡ What Is Improved In This Fork?

相较于原始仓库，这个版本最重要的变化有两点。

### 1. 速度大幅提升

针对 `DICOM-CT-PD` 数据“**文件数量大、单文件很小**”的特点，项目对**读取阶段**和 **rebin 阶段**都做了系统优化：

- 只读取最小必要 DICOM tags，减少 header 解析开销
- 预分配投影数组，避免边读边扩容
- 对几何量、插值索引和权重做 cache 化
- 使用向量化替代 Python 层循环
- 并行时采用共享内存线程，避免大数组跨进程序列化

在当前目标数据与实测环境下，端到端重排时间已经从**几十分钟量级缩短到几十秒量级**。  
实际速度仍取决于数据规模、磁盘性能和 CPU 线程数，但这确实是本 fork 最核心、最直接的工程收益之一。

### 2. 增加了对 GE 负向 z 扫描数据的支持

本仓库重点适配了公开数据集 [`LDCT-and-Projection-data`](https://doi.org/10.1002/mp.14594) 中由 GE 设备采集的一部分原始投影数据。

这类数据的关键特征是：

- source 在扫描过程中沿 **负 z 方向推进**
- `StartPosition` 与 `EndPosition` 可能同时为负
- 但真正重要的是 **z 方向符号不能在重排时被抹掉**

为此，当前实现补上了对以下问题的处理：

- 保留有符号的 rebinned z spacing
- 修正负向扫描时的 axial 区间映射
- 调整 `LDCT-and-Projection-data` 中 GE 数据的 detector-row 方向解释

因此，这个版本现在可以更稳定地支持 `LDCT-and-Projection-data` 中 GE 设备采集、并沿负 z 方向推进的扫描数据。

---

## 🌟 Features

- 直接读取 `DICOM-CT-PD` 投影和私有几何 tag
- 支持 `helical cone-beam → flat fan-beam` 的重排流程
- 兼容 `Siemens / GE` 等不同探测器行方向约定
- 输出 `TIFF + NIfTI + JSON metadata`
- 重建结果自动写入 `spacing / direction / origin`
- 针对大量小 DICOM 文件做了读取与重排性能优化

---

## 🗂️ Project Layout

```text
.
├── helical_to_fanbeam.py
├── recon_from_rebined_fanbeam_sino.py
├── utils
│   ├── helper.py
│   ├── read_data.py
│   └── rebinning_functions.py
└── Noo_1999_Phys._Med._Biol._44_561.pdf
```

---

## 📦 Dependencies

核心依赖如下：

- `numpy`
- `pydicom`
- `tifffile`
- `SimpleITK`
- `joblib`
- `tqdm`
- `matplotlib`
- `torch`
- `carterbox/torch-radon`

说明：

- 这里的 `torch-radon` 特指 [`carterbox/torch-radon`](https://github.com/carterbox/torch-radon)
- 该依赖仅在重建阶段需要
- 当前使用的是 CUDA 版 `carterbox/torch-radon`，因此重建默认要求可用 CUDA 环境
- 当然您可以更换为其他您更熟悉的 fan-beam 重建器，包括但不限于 `astra toolbox`、`TIGRE`、`CIL` 等等。

---

## 🧭 Rebinning

脚本：`helical_to_fanbeam.py`

作用：

- 读取原始 `DICOM-CT-PD` 螺旋投影
- 解析采集几何
- 执行 `curved detector → flat detector`
- 执行 `helical trajectory → fan-beam trajectory`

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--path_dicom` | 原始投影目录 |
| `--path_out` | 输出目录 |
| `--scan_id` | 输出文件前缀 |
| `--idx_proj_start` | 起始投影索引 |
| `--idx_proj_stop` | 结束投影索引 |
| `--save_all` | 保存中间结果 |
| `--no_multiprocessing` | 关闭重排阶段线程并行 |
| `--n_jobs` | 重排阶段线程数 |

主要输出：

- `*_flat_fan_projections.tif`
- `*_flat_fan_projections.nii.gz`
- `*_metadata.json`

---

## 🧩 Reconstruction

脚本：`recon_from_rebined_fanbeam_sino.py`

作用：

- 读取 fan-beam sinogram
- 逐层执行 2D FBP
- 支持 `FBP / CG / ART / SART / SIRT`
- 输出浮点重建和 HU 重建
- 为 NIfTI 写入几何信息

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--path_proj` | 输入 fan-beam TIFF |
| `--path_out` | 输出目录 |
| `--scan_id` | 输出文件前缀 |
| `--image_size` | 重建图像边长 |
| `--voxel_size` | 重建体素尺寸，单位 mm |
| `--method` | 重建算法，可选 `fbp / cg / art / sart / sirt` |
| `--num_iters` | 迭代法的迭代次数 |
| `--relaxation` | `ART / SART / SIRT` 的松弛参数 |
| `--num_subsets` | `SART` 的 ordered subsets 数量 |
| `--batch_size` | 每次一起重建的 slice 数量 |
| `--enforce_nonnegativity` | 是否对迭代更新做非负约束 |
| `--flip_u_for_recon` | 是否翻转 detector-u，默认自动判断 |
| `--fbp_filter` | FBP 滤波器 |
| `--device` | `auto` 或 `cuda` |

主要输出：

- `*_recon_HU_<filter>_<voxel_size>.nii.gz`
- `*_recon_<filter>_<voxel_size>.nii.gz`
- `*_recon_<filter>.tif`
- `*_metadata.json`
- `example_recon.png`

---

## 🎯 Recommended Use Cases

- 原始 CT 投影几何研究
- fan-beam 重建实验
- 不同厂商 raw projection 方向/采样差异排查
- helical → fan-beam 数据预处理管线搭建

---

## 🙏 Acknowledgements

本项目的核心方法、问题设定与原始实现路径，直接来源于：

- 原始仓库：[`faebstn96/helix2fan`](https://github.com/faebstn96/helix2fan)
- 相关论文：Wagner et al., *On the Benefit of Dual-domain Denoising in a Self-Supervised Low-dose CT Setting*, ISBI 2023  
  DOI: <https://doi.org/10.1109/ISBI53787.2023.10230511>

同时，本仓库的重要测试和兼容性改进建立在公开数据集之上：

- Moen et al., *Low-dose CT image and projection dataset*, *Medical Physics*  
  DOI: <https://doi.org/10.1002/mp.14594>

在此对原始 `helix2fan` 项目的作者、相关论文作者，以及 `LDCT-and-Projection-data` 数据集的发布者表示感谢。  
本仓库更适合被理解为：**在原始方法之上的一轮面向真实数据和工程交付的增强实现**。

---

## 📚 Reference

项目中的 single-slice rebinning 实现主要参考以下文献：

```bibtex
@article{Frédéric Noo_1999,
  author  = {Frédéric Noo and Michel Defrise and Rolf Clackdoyle},
  title   = {Single-slice rebinning method for helical cone-beam CT},
  journal = {Physics in Medicine & Biology},
  year    = {1999},
  month   = feb,
  volume  = {44},
  number  = {2},
  pages   = {561},
  doi     = {10.1088/0031-9155/44/2/019},
  url     = {https://doi.org/10.1088/0031-9155/44/2/019},
}
```
