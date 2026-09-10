# FY-CSI-DRCT：风云4A至风云3C晴空指数超分辨率

本目录是 `Super-Resolution` 工作区中的 DRCT 模型实验模块。完整流程将“风云4A（FY-4A）低空间分辨率地表太阳辐照度 → 风云3C（FY-3C）高空间分辨率地表太阳辐照度”构造成晴空指数（Clear-sky Index, CSI）×4 超分辨率任务，并采用单通道 DRCT-L 进行重建。

> 原始卫星产品、MERRA-2 下载文件、REST2 中间结果、训练 patch 和 checkpoint 体积较大或受数据分发条件限制，均不上传。仓库保留全部处理、训练、评价和绘图代码，以及固定的数据划分清单与最终小体积结果。

## 完整流程

```text
FY-4A SSI + FY-3C Gc/Sz + MERRA-2 大气参数
            │
            ├─ FY-3C 像元观测时刻反演（Sz + 经纬度）
            ├─ FY-4A 半开 15 min 时间窗逐像元匹配
            ├─ FY-4A 最近邻映射到 FY-3C 0.01°网格
            ├─ REST2 晴空辐照度计算与重采样
            └─ CSI 构建、质量控制和 32×32 / 128×128 patch 提取
                         │
                         ▼
        FY-4A LR CSI (1×32×32) → DRCT-L ×4 → FY-3C HR CSI (1×128×128)
                         │
                         ▼
           独立测试集评价、GHI 物理量恢复和论文图生成
```

## 目录说明

```text
Super-Resolution/
├─ FY_SSI_processing/             # FY-3C/FY-4A 时空匹配、CSI 与 patch 构建（共享）
│  ├─ fy_io.py                    # 产品读取、网格、掩膜与公共路径
│  ├─ step_02_*.py                # FY-3C 像元观测时刻反演
│  ├─ step_03_*.py                # FY-4A 时间匹配、空间映射及 REST2 重采样
│  ├─ step_04_*.py                # CSI 质量控制与 ×4 样本提取
│  └─ plot_fig5_*.py              # 第5章数据与方法图
├─ MERRA2/                        # MERRA-2 下载、校验、修复及 REST2 驱动脚本（共享）
├─ clear_sky.py, merra2.py        # REST2 所需公共模块（共享）
├─ environment.yml                 # FY 数据处理环境（共享）
└─ DRCT/
   ├─ data/                        # DRCT 数据集读取器
   ├─ datasets/                    # 50k / 10k / 10k 固定划分清单
   ├─ exp/train_x4.json            # 当前单通道 DRCT-L ×4 配置
   ├─ models/                      # DRCT 网络定义
   ├─ main.py                      # 训练、续训、验证与独立测试入口
   ├─ experiments/                 # 论文结果图绘制脚本
   ├─ results/                     # 运行配置、最终指标和论文图（不含权重）
   └─ requirements.txt             # DRCT 训练与绘图依赖
```

## 环境

数据预处理命令均在工作区根目录 `Super-Resolution/` 执行，建议使用：

```powershell
conda env create -f environment.yml
conda activate super-resolution
```

DRCT 训练与结果绘制需在含 PyTorch/CUDA 的环境中安装：

```powershell
pip install -r requirements.txt
```

## 数据与路径

处理后的训练 patch 目录约定为：

```text
outputs/fy/samples/<YYYYMMDD>_utc/
├─ sample_index.csv
└─ patches/batch_*.npz
```

仓库优先从本仓库内的 `outputs/fy/samples/` 读取样本；也可显式指定外部路径：

```powershell
$env:FY_CSI_SAMPLES_ROOT = 'D:\path\to\outputs\fy\samples'
```

原始数据需要自行准备：FY-4A AGRI L2 SSI、FY-3C Gc/Sz 及对应 MERRA-2 数据。详细产品读取、有效值筛选和目录约定见 [../FY_SSI_processing/README.md](../FY_SSI_processing/README.md)。

## 数据集构建

以下命令按处理依赖关系执行；日期、区域和源目录由各脚本命令行参数控制。

```powershell
# 1. 由 FY-3C Sz 与经纬度反演逐像元观测时刻
python FY_SSI_processing/step_02_estimate_fy3c_timestamp_from_sz.py

# 2. 将每个实际 FY-4A 时次映射至 FY-3C 公共网格
python FY_SSI_processing/step_03_resample_fy4_times_to_fy3_grid.py

# 3. 使用 [t0, t0 + 15 min) 半开时间窗进行逐像元时次匹配
python FY_SSI_processing/step_03b_fy3_fy4_slot_map.py

# 4. 计算/缓存 REST2 晴空辐照度并映射至公共网格
python FY_SSI_processing/build_rest2_ghi_slot_cache.py
python FY_SSI_processing/step_03c_resample_rest2_ghi_to_fy3_grid.py

# 5. 构建 LR 32×32 与 HR 128×128 的 CSI 配对样本
python FY_SSI_processing/step_04_build_fy3_fy4_samples_20180501.py
```

`step_04` 的低分辨率输入直接来自真实 FY-4A SSI；FY-3C 仅提供高分辨率监督参考，不通过人工降采样 FY-3C 构造 LR 输入。

## DRCT 实验

当前冻结实验为单通道 DRCT-L ×4：

| 项目 | 设置 |
|---|---|
| 输入 / 输出 | FY-4A CSI `1×32×32` → FY-3C CSI `1×128×128` |
| 初始化 | 从头训练 |
| 窗口大小 | 8 |
| 损失函数 | L1 / 归一化 CSI MAE |
| 训练集 | 50,000 对固定样本 |
| 验证集 | 10,000 对固定样本 |
| 独立测试集 | 10,000 对固定样本 |
| 有效 batch size | 16 |
| 初始学习率 | `2e-4` |
| 最终模型 | 第 271 epoch 收敛模型 |

训练入口仅为 `main.py`。例如，将训练上限设置为 300 epoch：

```powershell
$env:DRCT_EPOCHS = 300
python main.py
```

训练过程中的监控、可视化、固定划分与输出目录均由 `main.py` 管理。请使用新的结果目录进行后续续训或对照实验，不覆盖现有冻结结果。

## 固定划分与最终结果

`datasets/fy4a_fy3c_csi_x4/` 中保存实际使用的 train / validation / test 清单：50,000 / 10,000 / 10,000 对。三者样本互斥；清单仅记录日期、批次文件和批内位置，不复制原始 patch。

最终模型在独立 10,000 对测试样本上的结果如下：

| 方法 | R² | RMSE / (W·m⁻²) | MAE / (W·m⁻²) | MBE / (W·m⁻²) |
|---|---:|---:|---:|---:|
| Bicubic | 0.647 | 180.22 | 132.57 | 83.12 |
| DRCT | 0.843 | 120.27 | 78.21 | 4.66 |

对应的 CSI 图像域 DRCT 指标为 PSNR = 23.00 dB、SSIM = 0.596。完整数值见：

```text
results/fy4a_fy3c_drctl_w8_scratch_50k_b16_e50/final/
```

## 论文图

`experiments/` 保存论文结果图脚本；图5-8散点图同时提供 PNG 和混合矢量 SVG（坐标、文字和边框为矢量，密集点云为嵌入式高分辨率图层）。

```text
results/.../final/paper_figures/
├─ scatter_ghi_bicubic_epoch271_independent_test.svg
├─ scatter_ghi_bicubic_epoch271_independent_test.png
├─ ten_sample_epoch271_independent_test.png
└─ figure5_10_fy3c_drct_bicubic_samples.png
```

## 说明

- 不上传原始卫星产品、MERRA-2 下载文件、`outputs/` patch、训练缓存和 `.pth` 权重。
- 运行本仓库前，请依据数据使用许可自行获取相应 FY-4A、FY-3C 与 MERRA-2 数据。
- 保留原始 DRCT 的 MIT License，网络基础实现来源见其文件头和 LICENSE。
