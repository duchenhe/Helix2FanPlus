# 🔬 Helical CT Rebin 原理详解

> 从 `DICOM-CT-PD` 原始螺旋锥束投影，到可用于逐层重建的 **2D fan-beam sinogram**

这篇文档面向已经接触过 CT 几何、但希望进一步理解 `Helix2Fan+` 重排流程的人。它不是一份“代码注释汇总”，而是从**扫描几何、公式来源、工程化实现、数值细节、常见问题**五个角度，对当前 `helical_to_fanbeam.py` 的 rebin 逻辑进行一次完整说明。

---

## 1. 为什么要做 Rebin？

原始 `DICOM-CT-PD` 数据对应的是 **helical cone-beam CT**：

- x-ray source 沿螺旋轨迹绕物体旋转
- 探测器在横向是扇束采样，在轴向则具有一定高度
- 单幅投影本质上是一个二维 area detector 上的锥束投影

而很多成熟、快速、稳定的重建工具更适合处理的是 **2D fan-beam sinogram**：

- 每个切片单独重建
- 每个切片只需要一个二维 sinogram
- 算法、实现、调参和验证都更直接

因此，rebin 的目标不是“凭空改变数据”，而是把原始螺旋锥束采样，重排为一组更适合逐层重建的 fan-beam 投影。

在本项目中，这个过程分成两个连续阶段：

1. **Curved detector → flat detector**
2. **Helical cone-beam trajectory → 2D fan-beam trajectory**

---

## 2. 本项目实际处理的数据是什么？

当前项目读取的是 `DICOM-CT-PD` 格式的原始投影。读取后，内部数据布局约定为：

```text
raw_projections.shape = [n_proj, nv, nu]
```

其中：

- `n_proj`：总投影视角数
- `nv`：探测器轴向方向采样数，对应 detector `v`
- `nu`：探测器横向方向采样数，对应 detector `u`

这和图像重建中的常见 `H × W` 语义不同。这里：

- `u` 对应旋转平面内的扇束展开方向
- `v` 对应平行于机架旋转轴的方向

同时，header 中会恢复出一组关键几何量：

- `R = dso`：source 到旋转轴距离
- `D = dsd`：source 到 detector 距离
- `ddo = dsd - dso`
- `du, dv`：探测器像素间距
- `angles`：每幅投影对应的 source 方位角
- `z_positions`：每幅投影对应的 source 轴向位置
- `det_central_element`：探测器中心元索引
- `pitch`：每转一圈 source 的轴向位移

这组量构成了后续 rebin 的全部几何基础。

---

## 3. 几何图像化理解

可以把整个问题想象成下面这件事：

- 真实扫描时，source 沿一条螺旋线运动
- 每个 source 位置对应一张二维 cone-beam 投影
- 我们希望针对某个固定轴向切片 `z = const`
  - 构造出一整圈围绕该切片的 2D fan-beam sinogram

也就是说，输出不是“重新生成一份 3D cone-beam 数据”，而是：

```text
proj_fan_geometry.shape = [rotview, nu, nz_rebinned]
```

其中：

- 第一维 `rotview`：一圈等价 fan-angle 的采样数
- 第二维 `nu`：fan-beam detector 横向采样
- 第三维 `nz_rebinned`：重排后的轴向 slice 数

如果把它换个视角理解：

- 固定 `z`，得到一张 2D fan-beam sinogram
- 固定 `fan angle`，则得到该角度下沿 `z` 堆叠的一张“multislice projection”

---

## 4. 第一步：为什么要先做 Curved Detector → Flat Detector？

### 4.1 物理动机

虽然原始数据头信息可以标记 detector 为 cylindrical，但后续很多 fan-beam 重建实现默认处理的是 **线性 flat detector**。  
因此本项目首先把曲面探测器上的采样，映射到一个虚拟的平面探测器上。

这一步的本质是：

- **source 位置不变**
- **射线路径尽量保持一致**
- 只改变 detector 的参数化方式

所以它不是重建，而是一次几何重采样。

### 4.2 当前实现的坐标定义

对平面探测器，横纵坐标使用像素中心定义：

\[
u_i = \left(i - \frac{n_u}{2} + 0.5\right)\, d_u
\]

\[
v_j = \left(j - \frac{n_v}{2} + 0.5\right)\, d_v
\]

这意味着代码里的探测器坐标不是像素边界，而是**像素中心**。

### 4.3 从 flat detector 点反推到 curved detector

对虚拟 flat detector 上一点 \((u, v)\)，当前实现构造一条从 source 指向该点的射线，然后求这条射线与 curved detector 的等效交点。

代码中对应的关键量是：

\[
\rho = \sqrt{u^2 + D^2 + v^2}
\]

\[
\phi_{\text{curved}} = \arcsin\left(\frac{u}{\rho}\right)
\]

\[
v_{\text{curved}} = \frac{v}{\rho} D
\]

其中：

- \(\phi_{\text{curved}}\) 表示在曲面探测器上的弧角位置
- \(v_{\text{curved}}\) 表示轴向位置

然后将它们映射到离散 detector 索引：

\[
\text{col} = \frac{\phi_{\text{curved}}}{\Delta \phi} + \left(n_u - c_u\right)
\]

\[
\text{row} = \frac{v_{\text{curved}}}{d_v} + \left(n_v - c_v\right)
\]

其中：

- \((c_u, c_v)\) 来自 `det_central_element`
- \(\Delta \phi = 2 \arctan\left(\frac{d_u}{2D}\right)\)

最后，使用双线性插值从原始曲面投影取样，得到 flat detector 上对应像素值。

### 4.4 这一步的工程含义

这一阶段的结果仍然是螺旋轨迹投影：

```text
proj_flat_detector.shape = [n_proj, nv, nu]
```

只是 detector 的几何从 curved 变成了 flat，**source 的角度和 z 位置都没有改变**。

---

## 5. 第二步：Helical Cone-beam → 2D Fan-beam

这一步才是整个 rebin 的核心。

### 5.1 目标

对每个输出切片 \(z\)，构造一组 fan-beam 投影：

\[
p_z(\phi, u)
\]

其中：

- \(z\) 表示目标轴向位置
- \(\phi\) 表示虚拟 fan-source 的方位角
- \(u\) 表示 fan-beam detector 的横向位置

本项目采用的是 **single-slice rebinning** 的思路，核心参考文献在本文末尾以 `BibTeX` 形式给出，引用 key 为：

```bibtex
@article{Frédéric Noo_1999}
```

### 5.2 直观理解

对于某个固定输出 fan angle \(\phi\)：

- 在原始螺旋轨迹中，会有一串 source 位置具有相同的方位角模 \(2\pi\)
- 这些 source 分布在不同的轴向高度
- 对某个目标 slice \(z\)，需要从这些 source 中找到“轴向上最合适”的那一个
- 再从该 source 对应的 cone-beam 投影上，取出代表该 slice 的 detector 行位置

于是，一个 3D 问题被近似为：

- 固定一个方位角
- 在轴向上做选择与插值
- 得到该 slice 的一条 2D fan-beam 投影

---

## 6. `Frédéric Noo_1999` 的核心公式

论文给出的 single-slice rebin 近似可写为：

\[
p_z(\phi, u) \approx \frac{\sqrt{u^2 + D^2}}{\sqrt{u^2 + v^2 + D^2}} \, g_{\lambda}(u, v)
\]

其中：

- \(g_{\lambda}(u, v)\) 是原始 cone-beam 投影
- \(p_z(\phi, u)\) 是目标 fan-beam 投影
- \(D\) 是 source 到 detector 距离

而 detector 上应取样的位置 \(v\) 为：

\[
v = \Delta z \cdot \frac{u^2 + D^2}{R D}
\]

其中：

- \(R\) 是 source 到旋转轴距离
- \(\Delta z\) 是当前 cone-beam source 与目标 slice 之间的轴向距离

这两个式子正对应到本项目中的两行关键实现：

```python
v_precise = delta_z * ((u**2 + D**2) / (R * D))
scale = sqrt(u**2 + D**2) / sqrt(u**2 + D**2 + v_precise**2)
```

这就是为什么当前代码里会看到：

- `geom_factor = (u^2 + D^2) / (R * D)`
- `v_precise = delta_z * geom_factor`
- `scale_num / scale_den`

---

## 7. 当前代码如何把公式落地？

### 7.1 固定 fan angle，收集同方位 source

在实现中，对某个输出角度 `s_angle`，会取：

```python
proj_indices = np.arange(s_angle, proj_helic.shape[0], args.rotview)
```

这意味着：

- `rotview` 定义了一圈内等价方位角的数量
- 每隔 `rotview` 张投影，source 会回到同一 azimuth
- 因而这一串投影正好构成“同一 fan angle，不同轴向位置”的样本集合

### 7.2 只在 source 的有效轴向范围内写入

当前实现采用 full-scan SSRB 的简单版本，每个 source 对应一个轴向覆盖范围：

\[
[z_s - \tfrac{P}{2},\, z_s + \tfrac{P}{2}]
\]

其中：

- \(z_s\) 是该 source 的轴向位置
- \(P\) 是 pitch

在代码中：

```python
lower_lim = z_source - 0.5 * abs(pitch)
upper_lim = z_source + 0.5 * abs(pitch)
```

然后把这个物理 z 区间映射到 rebinned z 网格的离散索引范围。

### 7.3 负向扫描为什么容易出错？

这个项目里一个曾经非常关键的问题是：

- `StartPosition` 和 `EndPosition` 可能都为负数
- 但真正重要的不是“是否为负”，而是**扫描方向是否为负**

例如：

- `StartPosition = -10`
- `EndPosition = -200`

这表示扫描沿负 z 方向推进。  
如果在 z 映射时简单使用 `abs(...)`，会把方向信息抹掉，导致：

- source 的有效 coverage 仍在
- 但 slice 被映射到错误的 source 区间
- 结果表现为几何错位、轻微重影或模糊

因此当前实现专门保留了有符号 z 步长：

\[
\Delta z_{\text{rebinned}} = \operatorname{sign}(z_{N-1} - z_0)\, d_{v,\text{rebinned}}
\]

---

## 8. Detector v 方向插值是怎么做的？

公式给出的 \(v\) 通常不是整数 detector 行索引，因此必须插值。  
本项目采用的是**逐列线性插值**，也就是：

- 对每个 \(u\)
- 根据 \(v_{\text{precise}}\) 找到相邻两行
- 做线性加权

这一步在数值上等价于逐列 `np.interp`，但代码采用了向量化实现：

- 先把物理坐标映射到 row index
- 再统一计算 `row0 / row1 / alpha`
- 最后批量完成线性插值

这样做的优势是：

- 保持和论文近似一致
- 避免 Python 层 detector-column 循环
- 并且更容易保证不同厂商数据在插值边界上的行为一致

---

## 9. 当前实现与原始论文的关系

这里有一个很重要但常被忽略的事实：

### 当前项目并不是论文的“逐字复现”

它更准确的定位是：

> 以 `@article{Frédéric Noo_1999}` 的 single-slice rebin 思想为核心，结合当前 `DICOM-CT-PD` 数据格式、现有 fan-beam 重建器接口和工程性能需求，做出的一个实用化版本。

主要差异包括：

### 9.1 论文中讨论了 short-scan + Parker weighting

`@article{Frédéric Noo_1999}` 对 short-scan 的归一化权重给出了系统描述。  
当前项目为了和下游 fan-beam 重建器更直接对接，采用的是更直观的 full-scan stack 输出，并没有在 rebin 阶段显式实现 Parker 权重。

### 9.2 当前实现先做了 curved → flat

论文正文中为了简化 exposition，假设 area detector 是 flat。  
而实际数据可能是 cylindrical，因此本项目在 SSRB 之前增加了一层 detector 参数化转换。

### 9.3 当前实现强调工程可运行性

为了在大量小 DICOM 文件和较大投影堆栈上稳定运行，代码中额外做了：

- geometry cache 预计算
- detector 插值向量化
- 角度维静态分块并行
- metadata 回写，供后续重建复用

这些都不是论文主旨的一部分，但对工程交付是必要的。

---

## 10. 从代码角度看，整个数据流是什么？

可以把当前 `helical_to_fanbeam.py` 的 rebin 主流程压缩为下面这四步：

### Step 1. 读取原始投影与几何

输入：

- `DICOM-CT-PD` projection folder

输出：

- `raw_projections [n_proj, nv, nu]`
- `angles, z_positions, dso, dsd, du, dv, det_central_element, pitch, ...`

### Step 2. 曲面探测器重排到平面探测器

输入：

- `raw_projections`

输出：

- `proj_flat_detector [n_proj, nv, nu]`

### Step 3. 螺旋轨迹重排到 2D fan-beam stack

输入：

- `proj_flat_detector`
- `angles`
- `z_positions`

输出：

- `proj_fan_geometry [rotview, nu, nz_rebinned]`

### Step 4. 保存结果与 metadata

输出文件通常包括：

- `*_flat_fan_projections.tif`
- `*_flat_fan_projections.nii.gz`
- `*_metadata.json`

其中 metadata 非常重要，因为后续重建脚本需要从中恢复：

- `angles`
- `rotview`
- `dso / ddo / du`
- `dv_rebinned`
- `manufacturer`
- `hu_factor`

---

## 11. 为什么这种近似在实践中通常“比想象中更好”？

从理论上说，helical cone-beam 是三维采样问题，而逐层 fan-beam 重建本质上是二维模型。  
因此 SSRB 一定是近似，而不是精确反演。

但它在很多数据上的效果仍然很好，原因通常有三点：

### 11.1 每个 fan-beam ray 都选取了“同一垂直平面内”的 cone-beam ray

这让误差主要来自轴向近似，而不是横向几何错配。

### 11.2 对医学 CT 的常见 pitch 和体部尺度，轴向误差往往可接受

尤其当探测器轴向采样较细、重建 slice 不追求过高 z 分辨率时，这种近似的视觉质量通常较好。

### 11.3 2D fan-beam 重建器成熟且稳定

即使 rebin 本身是近似，只要 sinogram 的几何一致性较好，最终重建往往仍然能得到清晰、稳定、低噪的结果。

---

## 12. 当前实现里最值得注意的工程细节

### 12.1 几何方向比数值正负更重要

一个常见误区是只盯着 `StartPosition` 和 `EndPosition` 的符号。  
真正关键的是：

\[
\text{scan direction} = \operatorname{sign}(z_{\text{end}} - z_{\text{start}})
\]

只要这个方向处理错，即使所有绝对值都“看起来合理”，重建也会出现错位模糊。

### 12.2 厂商差异通常先体现在 detector 索引约定上

不同厂商虽然都可能标记为 cylindrical detector，但在这些细节上并不一定完全一致：

- detector row 是否需要翻转
- `det_central_element` 的索引参考
- angle 增长方向
- header 中物理量的符号约定

因此，rebin 成功与否常常首先取决于**索引体系是否解释正确**，而不是公式本身是否复杂。

### 12.3 性能瓶颈并不只在公式计算

在实际工程里，时间往往消耗在：

- 大量小 DICOM 文件读取
- Python 层逐像素循环
- 大数组跨进程序列化

因此本项目在实现上做了三类优化：

- 只读取必要 DICOM tags
- 用 cache 和向量化替代重复计算
- 并行时使用共享内存线程，避免大数组拷贝

---

## 13. 这一套 Rebin 的边界在哪里？

需要明确的是，当前方法虽然实用，但并不是“对所有 helical cone-beam 数据都严格最优”。

它的边界包括：

### 13.1 它是近似 SSRB，不是精确 3D 反演

当 cone angle 很大、pitch 很大、或轴向细节要求很高时，误差会更明显。

### 13.2 它依赖 header 中几何量足够可信

如果原始 DICOM 私有 tag 与实际 scanner 坐标约定存在偏差，rebin 会受到直接影响。

### 13.3 它面向的是“重排后再用 2D fan-beam 重建”

因此它天然偏向：

- 快速实验
- 工程验证
- geometry debugging
- fan-beam reconstruction pipeline

而不是把所有数据都强行压到某个统一的“临床标准重建”结果上。

---

## 14. 一句话概括当前项目的 Rebin 思路

如果要把整套实现压缩成一句话，可以这样说：

> 本项目先把原始 helical cone-beam 投影从曲面探测器重参数化到虚拟平面探测器，再沿每个方位角收集同 azimuth 的 source，并依据 single-slice rebinning 公式，把每个目标 slice 映射到真实 cone-beam 投影上的 detector `v` 位置，从而构造出可供逐层 2D fan-beam 重建的 sinogram stack。

---

## 15. 对应代码位置

如果你想把本文的几何说明和代码逐段对照，建议重点看这几个文件：

- [helical_to_fanbeam.py](./helical_to_fanbeam.py)
- [utils/read_data.py](./utils/read_data.py)
- [utils/rebinning_functions.py](./utils/rebinning_functions.py)

其中：

- `read_data.py` 负责恢复原始几何
- `rebinning_functions.py` 负责两阶段重排的核心公式
- `helical_to_fanbeam.py` 负责整体流程调度与并行执行

---

## 16. 参考资料

1. 核心参考文献的标准 `BibTeX` 条目如下：

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

2. 本仓库附带论文 PDF：`Noo_1999_Phys._Med._Biol._44_561.pdf`

---

## 17. 结语

好的 rebin 实现，关键不在于把公式写得多复杂，而在于三件事是否同时成立：

- **几何解释正确**
- **方向与索引一致**
- **工程实现稳定可复现**

当前项目的价值，正是在于把一条论文中的 single-slice rebin 思路，落成了一条可以直接处理 `DICOM-CT-PD` 数据、输出可重建 fan-beam sinogram 的工程流程。

如果把它看作一个专业工程模块，那么它最核心的职责并不是“追求理论上最完美”，而是：

> 在可接受的近似误差下，把复杂的 helical cone-beam 数据，可靠地整理成一个可验证、可重建、可调试的 fan-beam 表达。
