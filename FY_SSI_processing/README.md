# FY-3C / FY-4A SSI 处理

## 课题

对比 FY-3C Gc（按自然日归档的过境逐像元瞬时 SSI，观测时刻由 Sz 反演 best_minute 给出，沿经度条带变化）与 FY-4A 高时空分辨率 SSI：先做全圆盘经纬度可视化与空间对齐（重采样到 FY-3C 0.01° 网格），为后续超分辨率 / 时空融合与 REST2 验证打基础。

## FY-4A 时间约定（必读）

**官方规格**：[`FY-4A_AGRI_L2_SSI_NOM_4000M_V1.1.pdf`](../FY-4A_AGRI_L2_SSI_NOM_4000M_V1.1.pdf)（项目根目录）

| 来源 | 时区 |
|------|------|
| FY-4A NC 文件名 `YYYYMMDDHHMMSS`（起始）/ `yyyymmddhhmmss`（结束） | **UTC** |
| NC 全局属性 `time_coverage_start` / `time_coverage_end` | **UTC**（格式 `...Z`） |
| step_02 FY-3C `best_minute` | **CST / 北京时间（UTC+8）** |
| step_03 输出 `FY4_HHMM_on_FY3.npz` 文件名与 `time_utc` | **UTC**（`time_cst` 为对照） |

匹配逻辑：FY-4 半开窗 `[UTC start, start+15min)` 由文件名定义；FY-3 `best_minute`（CST）在匹配前转为 UTC 再比较（见 `fy_io.match_pixels_to_half_open_windows`）。

## FY-4A SSI 无效像元与 DQF（`fy_io.clean_ssi`）

规格：[`FY-4A_AGRI_L2_SSI_NOM_4000M_V1.1.pdf`](../FY-4A_AGRI_L2_SSI_NOM_4000M_V1.1.pdf)

| 规则 | 说明 |
|------|------|
| 填充值 | `-999`（缺测）、`65532`（SZA 超阈值）、`65535`（圆盘外）→ NaN |
| 有效范围 | `0–1500` W/m²（NC 变量 `valid_range`） |
| DQF | 默认掩膜 `DQF==3`；可选 `--mask-dqf2` 掩膜 `DQF==2` |
| V1.0 注意 | **20180501** 为 V1.0（`Version Of Software`≈V1.0），`65532` 表示 **SZA>70°**（V1.1 文档为 >90°）；该日 DQF 常全为 3（未分级），此时跳过 DQF 掩膜，依赖 65532/65535 清洗 |

step_03 / `diagnose_fy4_daily_timeseries.py` 均通过 `read_fy4_ssi_with_dqf()` 共用上述规则。经纬度 raw 缺失时 step_03 自动用 NSMC/CGMS 公式（104.7°E）生成。

> **迁移说明**：2026-06 之前曾将 UTC 文件名误标为 CST 本地时间；旧产物已清理。当前生产输出统一写入 `*_utc` 子目录（如 `resampled_on_fy3/20180501_utc/`、`matching/20180501_utc/`、`samples/20180501_utc/`）。

## 生产工作流（按顺序）

| 步骤 | 脚本 | 作用 |
|------|------|------|
| 共用 | `fy_io.py` | 常量、F 盘默认路径、FY-3/FY-4 数据读取、UTC/CST 转换 |
| step_01（可选 QC） | `step_01_plot_compare_lonlat.py` | 全圆盘经纬度并排对比图（不重采样） |
| step_02 | `step_02_estimate_fy3c_timestamp_from_sz.py` | 用 FY-3C Sz 太阳天顶角逐像元反推观测时间（CST） |
| step_03 | `step_03_resample_fy4_times_to_fy3_grid.py` | 批量将 FY-4A 各时次（默认 **UTC 00:00–06:00** ≈ CST 08:00–14:00）重采样到 FY-3C 全幅网格，输出 `FY4_HHMM_on_FY3.npz` |
| step_03b | `step_03b_fy3_fy4_slot_map.py` | **师兄半开窗口匹配**：FY-3 CST `best_minute` 转 UTC 后落入 FY-4 `[start, start+15min)` 才归属该时次；输出 `fy4_slot_map` 与统计 |
| step_04 | `step_04_build_fy3_fy4_samples_20180501.py` | 从 step_03 重采样 npz 取 SSI，滑动切 **FY-3 64×64 / FY-4 16×16** patch（slot≥0 覆盖率≥0.99 且仅单一 FY-4 时次）；FY-4 为 **stride 取样** `patch[::4,::4]`，非 4×4 块平均 |
| step_04 后处理（可选） | `step_04_sample_build_summary.py` | 汇总 step_04 输出、写 `build_summary.*` 与 patch 空间分布图 |

> **时间匹配（师兄逻辑）**：FY-3C Gc 为**按自然日归档的过境逐像元瞬时 SSI**，观测时刻由 Sz 反演 `best_minute` 给出（**CST**，沿经度条带变化）；FY-4A 为各时次 NC，文件名时次为 **UTC**。每个 FY-4 文件对应 UTC 半开窗 `[HH:MM, HH:MM+15min)`；FY-3 像元仅在 `best_minute`（转 UTC 后）落入该窗时匹配该时次（**非最近邻**）。step_03b 生成逐像元 `fy4_slot_map`；step_04 要求 patch 内 `slot>=0` 像元仅含单一 FY-4 时次后，从 `FY4_*_on_FY3.npz` 取对应 FY-4 SSI。

## 默认数据路径（F 盘）

| 数据 | 路径 |
|------|------|
| FY-3C Gc | `F:\FY-3C\SSI\Gc_<YYYYMMDD>.dat` |
| FY-3C Sz | `F:\FY-3C\SSI\Sz_<YYYYMMDD>.dat` |
| FY-4A SSI NC | `F:\FY-4A\<YYYYMMDD>\` 下 `*.NC` |
| FY-4A 经纬度 raw | `F:\经纬度文件\FY4A_FullMask_Grid_4000.raw` |

FY-3C 网格：3600×6200，73–135°E、54–18°N，步长 0.01°。  
FY-4A：2748×2748，raw 文件通道 0=纬度、1=经度，`1000000` 为无效填充。

## 运行

在项目根目录 `D:\Super-Resolution` 执行。

**日期范围**：各 step 脚本支持可选 `--end-date YYYYMMDD`（含），与 `--date` 一起使用时在脚本内逐日循环，并打印 `[3/31] 20180503` 进度；省略 `--end-date` 时行为与原先单 `--date` 相同。

```powershell
# 整月示例（step_02 → step_03b）
py -3 FY_SSI_processing\step_02_estimate_fy3c_timestamp_from_sz.py --date 20180501 --end-date 20180531 --stride 1 --use-local-time-prior --prior-window-minutes 120
py -3 FY_SSI_processing\step_03_resample_fy4_times_to_fy3_grid.py --date 20180501 --end-date 20180531 --skip-existing
py -3 MERRA2\build_rest2_ghi_slot_cache.py --date 20180501 --end-date 20180531 --skip-existing
py -3 FY_SSI_processing\step_03c_resample_rest2_ghi_to_fy3_grid.py --date 20180501 --end-date 20180531 --skip-existing --build-rest2-cache
```

### step_01 — 经纬度全圆盘对比图（可选）

```powershell
py -3 FY_SSI_processing\step_01_plot_compare_lonlat.py
```

默认输出：`outputs/fy/compare/fy3_fy4_lonlat_full_disk.png`

### step_02 — 用 FY-3C Sz 估计像元观测时间戳

```powershell
py -3 FY_SSI_processing\step_02_estimate_fy3c_timestamp_from_sz.py --date 20180501 --stride 4
```

默认读取 `F:\FY-3C\SSI\Sz_20180501.dat`，在北京时间/CST(UTC+8) `08:00-12:00` 窗口内逐分钟计算太阳天顶角，并与 FY-3C Sz 找最小绝对差值。若要跑全分辨率，使用 `--stride 1`（默认），但计算量约为 3600×6200×241 次像元分钟，耗时会明显更长。

推荐启用地方时先验（仓库内保留版本为 `prior1015w120`）：

```powershell
py -3 FY_SSI_processing\step_02_estimate_fy3c_timestamp_from_sz.py `
  --date 20180501 --stride 1 --start 08:00 --end 14:00 `
  --use-local-time-prior --local-overpass-time 10:15 --prior-window-minutes 120
```

推荐输出（`outputs/fy/timestamp/`）：

- `fy3c_timestamp_20180501_prior1015w120_stride1.npz` — `best_minute`、`best_diff`、`lons`、`lats` 等
- 同名 `.png` 与 `fy3c_timestamp_error_*.png`

### step_03 — 批量 FY-4A 重采样到 FY-3C 网格

```powershell
py -3 FY_SSI_processing\step_03_resample_fy4_times_to_fy3_grid.py --list-only
py -3 FY_SSI_processing\step_03_resample_fy4_times_to_fy3_grid.py --limit 2
py -3 FY_SSI_processing\step_03_resample_fy4_times_to_fy3_grid.py --skip-existing
```

默认 `--start 00:00 --end 06:00`（**UTC**；约等于 CST 08:00–14:00）。

默认输出目录 `outputs/fy/resampled_on_fy3/20180501_utc/`（`--output-dir` 可改）：

- `FY4_0000_on_FY3.npz` — `ssi`、`lons`/`lats`、`source_nc`、`time_utc`、`time_cst` 等

20180501 在 UTC 00:00–06:00 窗口内 9 个时次（文件名实测）：**0000、0100、0200、0245、0300、0400、0500、0545、0600**（约 CST 08:00–14:00）。

空间重采样方法：`--method nearest|gauss`（pyresample 的**空间**插值，与时间匹配无关）。

### step_03b — FY-3 / FY-4 半开窗口匹配统计

```powershell
py -3 FY_SSI_processing\step_03b_fy3_fy4_slot_map.py
```

默认输入：step_02 时间戳 npz、step_03 目录 `outputs/fy/resampled_on_fy3/20180501_utc/`。

默认输出 `outputs/fy/matching/20180501_utc/`：

- `fy4_slot_map_20180501_prior1015w120_stride1.npz`
- `fy4_slot_map_20180501.png`

### step_04 — FY-3C / FY-4A 时空配对切样本

```powershell
py -3 FY_SSI_processing\step_04_build_fy3_fy4_samples_20180501.py --max-samples 5
py -3 FY_SSI_processing\step_04_build_fy3_fy4_samples_20180501.py
```

默认输入：step_02 时间戳、step_03 `FY4_*_on_FY3.npz`、step_03b `fy4_slot_map`（存在则加载，否则在线半开窗口计算）。

默认输出 `outputs/fy/samples/20180501_utc/`（可用 `--output-dir` 指定）：

- `sample_index.csv`、`patches/batch_*.npz`、`build_meta.npz`

**FY-4 patch 下采样（step_04）**：从与 FY-3 同范围的 64×64 重采样 SSI 上取 `patch[::4, ::4]` 得到 16×16（`pool=4`），与 FY-4 4 km 等效分辨率对应；**不是**对每 4×4 像元做平均。64×64 重采样场在图上可能呈阶梯状（step_03 `nearest` 引起），属预期。

### step_04 后处理 — 样本构建摘要图

```powershell
py -3 FY_SSI_processing\step_04_sample_build_summary.py --samples-dir outputs\fy\samples\20180501_utc
```

写入 `build_summary.txt/json`，并在 `outputs/fy/matching/<date>/` 生成 patch 中心分布图与时次直方图。

### 原始 NC 诊断与检视

```powershell
py -3 FY_SSI_processing\diagnose_fy4_daily_timeseries.py --date 20180501
py -3 FY_SSI_processing\analyze_fy4_daily_cycle_cst.py --date 20180501
```

`diagnose_fy4_daily_timeseries.py` 遍历某日全部 FY-4A SSI NC（优先 `data/fy4_raw/<date>/`，否则 `F:\FY-4A\<date>\`），按 **UTC 文件名时次** 输出统计 CSV、时序图与各时次预览帧（默认写入 `outputs/fy/diagnostics/<date>_utc/`（**可选调试产物，阅后可删**））。`analyze_fy4_daily_cycle_cst.py` 可将 diagnose 输出转为 CST 日变化 mosaic/时序图（按需重跑，输出目录可 `--output-dir` 指定）。

## 输出目录说明

| 子目录 | 内容 |
|--------|------|
| `timestamp/` | step_02 时间戳 npz 与诊断图 |
| `resampled_on_fy3/<date>_utc/` | step_03 各时次 `FY4_HHMM_on_FY3.npz`（**HHMM = UTC**） |
| `matching/<date>_utc/` | step_03b slot 图与 step_04 摘要图 |
| `samples/<date>_utc/` | step_04 训练 patch |
| `diagnostics/<date>_utc/` | （可选）`diagnose_*` / `analyze_fy4_daily_cycle_cst.py` 临时输出，**不纳入版本库长期保留** |

已清理的过时中间结果（按需重新生成）：`outputs/fy/resample/`（原 step_03a）、`samples/_archive/`、`diagnostics/<date>_utc/`（diagnose 预览帧等）、一次性重采样对比脚本与 `diagnostics/*_compare_*` 验证图等；无 `_utc` 后缀的 20180501 流水线输出（CST 误标时代产物）应全部删除；`compare/` 为 step_01 对比图，需要时可由 step_01 重新生成到 `outputs/fy/compare/`（step_01 可选 QC 默认仅写入 `outputs/fy/compare/fy3_fy4_lonlat_full_disk.png`）。

## 依赖

```powershell
pip install numpy matplotlib netCDF4 pyresample
```

读取 FY-4A NC 时 `netCDF4` 与 `xarray` 二选一即可。

## 帮助

```powershell
py -3 FY_SSI_processing\step_01_plot_compare_lonlat.py --help
py -3 FY_SSI_processing\step_02_estimate_fy3c_timestamp_from_sz.py --help
py -3 FY_SSI_processing\step_03_resample_fy4_times_to_fy3_grid.py --help
py -3 FY_SSI_processing\step_03b_fy3_fy4_slot_map.py --help
py -3 FY_SSI_processing\step_04_build_fy3_fy4_samples_20180501.py --help
py -3 FY_SSI_processing\step_04_sample_build_summary.py --help
```
