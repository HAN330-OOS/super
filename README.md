# FY-4A / FY-3C 太阳辐照度超分辨率

DRCT-L 将 FY-4A 的 32×32 原始晴空指数（CSI）图块重建为 128×128，并用配对的 FY-3C 数据评价。正式数据覆盖 2018 年 5 月至 2019 年 12 月。

| 目录 | 内容 |
| --- | --- |
| [DRCT](DRCT/) | 正式云端模型、训练与测试入口、实验配置和上游许可证 |
| [FY_SSI_processing](FY_SSI_processing/) | FY 数据对齐、晴空辐照度映射与 CSI 样本生成 |
| [MERRA2](MERRA2/) | MERRA-2 气象输入与 REST2 晴空辐照度计算 |
| [splits](splits/) | 正式清洗后划分，共 177,380 个训练、35,482 个验证和 35,476 个测试样本 |
| [evaluation](evaluation/) | 正式 GHI 评价、四方法指标表和实验来源校验信息 |

## 数据准备

从 FY-3C 时间估计、FY-4A 重采样和时间窗匹配开始，计算并映射 REST2 晴空辐照度，再生成 CSI 样本。处理顺序、输入文件和时区约定见 [FY 预处理说明](FY_SSI_processing/README.md)。本地处理环境见 [environment.yml](environment.yml)。

训练和评价沿用归档的原始配对与仓库中的清洗版固定划分。重跑预处理生成的匹配结果需单独核验。

## 训练和测试

在 DRCT 目录运行以下命令。PyTorch、驱动和通信后端由运行环境提供。训练快照保留正式配置，可通过命令行指定数据与输出路径。数据根目录应有 samples/YYYYMMDD 下的 NPZ 文件，以及与本仓库一致的 splits。

    cd DRCT
    python -m pip install -r requirements.txt
    python prepare.py env
    torchrun --standalone --nproc_per_node=4 train_DRCT.py --opt exp/train_x4.json --device cuda --precision fp16 --backend nccl --micro-batch 16 --data-root /path/to/prepared-data --output-dir /path/to/run
    python test_DRCT.py --opt exp/train_x4.json --data-root /path/to/prepared-data --output-dir /path/to/run --checkpoint /path/to/checkpoint.pth

正式训练的全局 batch 为 64。复跑沿用固定 CSV 行顺序。需要复现已报告的权重结果时，可用[正式实验记录](evaluation/formal_run_reference.json)里的 SHA‑256 核对权重。

## 正式结果

[四方法指标表](evaluation/formal_metrics.csv)汇总 Bicubic、EDSR、SwinIR 和 DRCT-L。PSNR 和 SSIM 按 CSI 计算，R²、RMSE、MAE 和 MBE 按 GHI 计算。MBE 定义为预测减 FY-3C 参考。复算所需文件和命令见[评价说明](evaluation/README.md)。

DRCT、BasicSR 和 timm 的上游许可见 [DRCT/LICENSE](DRCT/LICENSE)。

## 数据与权重

卫星原始文件、NPZ 样本、晴空 GHI 数组、完整预测和权重单独归档。此目录保留划分索引、数据结构说明和正式权重校验值。
