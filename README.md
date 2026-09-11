# Super-Resolution 工作区

本仓库含两套相对独立的遥感数据处理流程，原始数据主要在 **F 盘**，代码在 **D:\Super-Resolution**。

## FY-3C / FY-4A SSI

目录：[`FY_SSI_processing/`](FY_SSI_processing/)

生产工作流（共用 `fy_io.py`）：

1. `step_01_estimate_fy3c_timestamp_from_sz.py` — FY-3C 像元时间戳
2. `step_02_resample_fy4_times_to_fy3_grid.py` — FY-4A 批量重采样到 FY-3C 网格
3. `step_03_build_fy3_fy4_slot_map.py` — 半开 15 min 窗口像元匹配
4. `step_04_resample_rest2_ghi_to_fy3_grid.py` — REST2 晴空辐照度重采样到公共网格
5. `step_05_build_fy3_fy4_samples.py` — 时空配对切训练 patch

论文及检查图脚本集中在 [`FY_SSI_processing/plots/`](FY_SSI_processing/plots/)。

**FY-4A 时间**：NC 文件名与 step_02 输出标签均为 **UTC**；FY-3 `best_minute` 为 **CST**。详见 [`FY-4A_AGRI_L2_SSI_NOM_4000M_V1.1.pdf`](FY-4A_AGRI_L2_SSI_NOM_4000M_V1.1.pdf) 与 [FY_SSI_processing/README.md](FY_SSI_processing/README.md)。

**FY-4A SSI 清洗**（`fy_io`）：剔除 `-999`/`65532`/`65535` 与 `0–1500` W/m² 越界；`65532` 在 20180501 V1.0 数据中表示 SZA>70°。DQF==3 默认掩膜（V1.0 若 DQF 全为 3 则自动跳过）。

详细说明见 [FY_SSI_processing/README.md](FY_SSI_processing/README.md)。

FY 相关输出在 `outputs/fy/`：

| 子目录 | 内容 |
|--------|------|
| `timestamp/` | FY-3C 像元时间戳（step_01，CST） |
| `resampled_on_fy3/<date>_utc/` | 各时次 `FY4_HHMM_on_FY3.npz`（step_02，**HHMM=UTC**） |
| `matching/<date>_utc/` | slot 匹配 npz/图（step_03） |
| `rest2_on_fy3/<date>_utc/` | REST2 晴空辐照度场（step_04） |
| `samples/<date>_utc/` | step_05 训练 patch（`--output-dir` 指定，如 `20180501_utc`） |

## MERRA-2 + REST2

目录：[`MERRA2/`](MERRA2/)

仅在需要获取或修复本地 MERRA-2 原始数据时执行：

1. `step_01_download_merra2.py`
2. `step_02_validate_merra2_downloads.py`
3. `step_03_repair_merra2_downloads.py`

FY 主流程的 REST2 缓存由 `build_rest2_ghi_slot_cache.py` 生成，并由 `FY_SSI_processing/step_04_resample_rest2_ghi_to_fy3_grid.py` 调用。根目录 `merra2.py`、`clear_sky.py` 是这条 REST2 链路的依赖，请勿删除。

## FY-CSI-DRCT 模型实验

模型训练、评估和论文绘图代码在 [`DRCT/`](DRCT/)；该目录复用本仓库根目录的 `FY_SSI_processing/`、`MERRA2/`、`clear_sky.py`、`merra2.py` 与 `environment.yml`，不再维护副本。

DRCT 的固定模型数据集单独存放在 `E:\Super-Resolution-Processed\`，并按 `train/`、`validation/`、`test/` 三个真实目录划分。该目录由 `DRCT/data/export_fixed_splits.py` 从 `outputs/fy/samples/` 导出，体积较大，不纳入 Git；DRCT 检测到该目录后会优先直接读取它。

## 其他

- `outputs/hu_references_extracted.txt`：文献摘录，与 FY 流程无关。
- 运行 Python 脚本建议使用：`py -3 <脚本路径>`
