# FY-4A / FY-3C 太阳辐照度超分辨率

本项目用 DRCT-L 将 FY-4A 地表太阳辐照度产品的晴空指数（CSI）从 **4 km 重建到 1 km**，并用时空配对的 FY-3C 观测评价。模型读取单通道 `32 × 32` 图块，输出 `128 × 128` 图块。训练和预测均使用原始 CSI。评价在 CSI 空间计算图像重建指标，并通过对应的晴空全球水平辐照度（GHI）将结果转换回 W/m²。

- [正式结果](#正式结果)
- [数据来源与预处理](#数据来源与预处理)
- [固定划分与数据格式](#固定划分与数据格式)
- [训练和独立测试](#训练和独立测试)
- [GHI 复算与校验](#ghi-复算与校验)
- [论文绘图](#论文绘图)
- [地基验证](#地基验证)

## 正式结果

以下结果来自 2018 年 5 月至 2019 年 12 月样本库的固定清洗版独立测试集，共 **35,476 个图块**。粗体标出每项最佳值。MBE 按绝对值比较。

| 方法 | CSI PSNR（dB）↑ | CSI SSIM ↑ | GHI R² ↑ | GHI RMSE（W/m²）↓ | GHI MAE（W/m²）↓ | GHI MBE（W/m²） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bicubic | 20.014796 | 0.568643 | 0.670413 | 154.304341 | 117.707811 | 24.069970 |
| EDSR | 25.420816 | 0.674701 | 0.890815 | 88.812736 | **57.075084** | 1.491762 |
| SwinIR | 25.424788 | 0.680952 | 0.890279 | 89.030317 | 57.645776 | 1.580331 |
| DRCT-L | **25.454710** | **0.682477** | **0.892074** | **88.299011** | 57.286309 | **1.095085** |

仓库提供 DRCT-L 的训练与测试代码、Bicubic 的评价实现，以及 EDSR 和 SwinIR 的正式对照指标。完整精度与各方法的来源见 [指标 CSV](evaluation/formal_metrics.csv) 和 [评价说明](evaluation/README.md)。

## 目录结构

| 路径 | 用途 |
| --- | --- |
| [DRCT/](DRCT/) | 模型、训练和测试入口、正式配置、依赖和上游许可 |
| [FY_SSI_processing/](FY_SSI_processing/) | FY-3C 时间估计、FY-4A 重采样、配对和图块生成 |
| [MERRA2/](MERRA2/) | 气象数据获取与核验、REST2 晴空 GHI 计算 |
| [splits/](splits/) | 训练、验证和测试的固定样本索引 |
| [evaluation/](evaluation/) | DRCT、Bicubic、EDSR 和 SwinIR 的 GHI 评价与指标汇总 |
| [figures/](figures/) | 论文图 1–3 的定稿绘图脚本 |
| [FY4A-FY3C-Ground-Validation/](FY4A-FY3C-Ground-Validation/) | 已配对卫星样本与 CMA 分钟 GHI 的地基验证 |
| [environment.yml](environment.yml) | FY 和 MERRA-2 预处理所用的 Python 环境 |

## 数据来源与预处理

| 数据 | 在流程中的用途 |
| --- | --- |
| FY-4A AGRI SSI | 4 km 的低分辨率辐照度输入 |
| FY-3C Gc 和太阳天顶角 Sz | 1 km 的参考辐照度与观测时间估计 |
| MERRA-2 气象输入与 REST2 | 计算晴空 GHI，构造 CSI 并恢复辐照度 |

1. 根据 FY-3C 的太阳天顶角估算观测时间。
2. 将 FY-4A 产品映射到 FY-3C 网格，并以 FY-4A 扫描起点为准，使用左闭右开的 15 分钟窗口进行时空配对。
3. 将 REST2 晴空 GHI 映射到相应格点，以观测辐照度除以对应晴空 GHI 构造 CSI。
4. 生成配对的 `32 × 32` 低分辨率与 `128 × 128` 高分辨率图块。

处理入口、时区约定和各阶段输入输出见 [FY 预处理说明](FY_SSI_processing/README.md)。正式实验使用已归档的原始配对，其中复用了历史 `slot_map`。重新运行预处理时应独立核验配对，再与正式实验对照。

### MERRA-2 数据目录

三个 tavg1_2d_* 产品目录可放在仓库外。将它们的共同父目录写入 MERRA2/data_root.local，文件只写一行路径。也可用环境变量 MERRA2_DATA_ROOT 临时覆盖。下载、校验、REST2 计算和 FY 样本重建共用这个输入路径。未配置时，脚本读取仓库下的 MERRA2 目录。

## 固定划分与数据格式

| 划分 | 样本数 | 索引 |
| --- | ---: | --- |
| 训练 | 177,380 | [train_samples.csv](splits/train_samples.csv) |
| 验证 | 35,482 | [validation_samples.csv](splits/validation_samples.csv) |
| 测试 | 35,476 | [test_samples.csv](splits/test_samples.csv) |

准备训练数据时，将上述三个 CSV 放在数据根目录的 `splits/`，将归档的 NPZ 放在 `samples/YYYYMMDD/`。数据加载器按 CSV 的日期、`npz_file` 和 `array_index` 读取指定样本。同一 NPZ 可以包含来自不同划分的样本，复跑时须保留 CSV 行顺序与数组内索引。

| NPZ 字段 | 单个样本的形状 | 内容 |
| --- | --- | --- |
| `fy4` | `32 × 32` | FY-4A 辐照度 |
| `fy3` | `128 × 128` | FY-3C 参考辐照度 |
| `lr_csi` | `32 × 32` | 模型输入 |
| `hr_csi` | `128 × 128` | 模型监督目标 |

四个数组的第一维为样本维，数据类型为 `float32`。正式样本、原始产品、完整预测、晴空 GHI 和权重独立归档。索引、正式权重和源代码的 SHA-256 记录在 [formal_run_reference.json](evaluation/formal_run_reference.json)。

## 训练和独立测试

正式实验在阿里云 PAI-DSW 的 **4 张 PPU-ZW810E** 上训练。全局 batch 为 64，每设备 batch 为 16。模型从随机初始化开始，使用 L1 损失、Adam、初始学习率 `1e-4` 和 EMA。训练按验证集 L1 的收敛规则停止，正式评价使用归档的第 65 轮 EMA 权重。详细设置见 [train_x4.json](DRCT/exp/train_x4.json)。

PyTorch、驱动和通信后端由运行环境提供。普通依赖见 [requirements.txt](DRCT/requirements.txt)。以下命令在兼容的四设备环境运行，先将路径换成实际位置。

~~~bash
cd DRCT
python -m pip install -r requirements.txt
python prepare.py env

torchrun --standalone --nproc_per_node=4 train_DRCT.py \
  --opt exp/train_x4.json --device cuda --precision fp16 --backend nccl \
  --micro-batch 16 --plain-adam \
  --data-root /path/to/prepared-data --output-dir /path/to/run \
  --metric-reference ../evaluation/metric_reference.json
~~~

训练完成后，使用选定的冻结权重运行独立测试。

~~~bash
python test_DRCT.py --opt exp/train_x4.json \
  --device cuda --precision fp16 \
  --data-root /path/to/prepared-data --output-dir /path/to/run \
  --checkpoint /path/to/best_epoch_065.pth
~~~

`train_DRCT.py --help` 和 `test_DRCT.py --help` 列出运行参数。使用 `--dry-run` 可检查传给训练或测试后端的配置。

## GHI 复算与校验

- **CSI 指标**按图块计算 PSNR 和 SSIM，再对测试图块取均值。参考范围由训练集 HR CSI 冻结，具体数值记录在 [metric_reference.json](evaluation/metric_reference.json)。
- **GHI 重建**采用 `GHI = CSI × 对齐的晴空 GHI`。正式复算沿用已归档并对齐的派生晴空数组。输入 CSI 和输出 CSI 保留原始尺度，计算中没有额外乘以 1.5 或裁剪预测。
- **GHI 指标**以 NPZ 的 FY-3C 辐照度为参考，在共同有效像元上累计 R²、RMSE、MAE 和 MBE。MBE 定义为预测减参考。表中 GHI 误差以 W/m² 为单位。

从仓库根目录运行 `evaluation/export_test_predictions.py`，`--mode smoke` 检查前两个测试图块，`--mode full` 运行完整独立测试。该脚本需要支持 CUDA 的运行环境，以及正式权重、样本、晴空数组和行号映射。复算命令与各输入路径见 [evaluation/README.md](evaluation/README.md)。

结果导出后，可用 `evaluation/verify_test_export.py --run-dir <运行目录>` 核对预测、数据顺序和来源校验值。

## 论文绘图

[figures/](figures/) 收录架构图排版、横向 Viridis 密度散点图和共用分位数色标的 CSI 对比图。各图所需的模板和正式实验输入，以及可直接运行的命令见 [绘图说明](figures/README.md)。

## 地基验证

地基验证沿用正式测试集的卫星配对。先按站点经纬度查找测试图块，再按已有的 FY-4A 扫描窗口取 CMA 分钟 GHI。四种卫星产品在共同配对上使用同一份地基参考。站点和分钟数据模板、匹配参数以及运行步骤见 [地基验证说明](FY4A-FY3C-Ground-Validation/README.md)。

## 许可证

DRCT、BasicSR 和 timm 的上游许可见 [DRCT/LICENSE](DRCT/LICENSE)。
