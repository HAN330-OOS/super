# FY-3C / FY-4A SSI 处理

## 课题

对比 FY-3C Gc（按自然日归档的过境逐像元瞬时 SSI，观测时刻由 Sz 反演 best_minute 给出，沿经度条带变化）与 FY-4A 高时空分辨率 SSI：先做全圆盘经纬度可视化与空间对齐（重采样到 FY-3C 0.01° 网格），为后续超分辨率 / 时空融合与 REST2 验证打基础。

## FY-4A 时间约定（必读）

**参考规格**为 FY-4A AGRI L2 SSI 4000 m 产品说明（V1.1）。

| 来源 | 时区 |
|------|------|
| FY-4A NC 文件名 `YYYYMMDDHHMMSS`（起始）/ `yyyymmddhhmmss`（结束） | **UTC** |
| NC 全局属性 `time_coverage_start` / `time_coverage_end` | **UTC**（格式 `...Z`） |
| step_01 FY-3C `best_minute` | **CST / 北京时间（UTC+8）** |
| step_02 输出 `FY4_HHMM_on_FY3.npz` 文件名与 `time_utc` | **UTC**（`time_cst` 为对照） |

匹配逻辑：FY-4 半开窗 `[UTC start, start+15min)` 由文件名定义；FY-3 `best_minute`（CST）在匹配前转为 UTC 再比较（见 `fy_io.match_pixels_to_half_open_windows`）。

## FY-4A SSI 无效像元与 DQF（`fy_io.clean_ssi`）

| 规则 | 说明 |
|------|------|
| 填充值 | `-999`（缺测）、`65532`（SZA 超阈值）、`65535`（圆盘外）→ NaN |
| 有效范围 | `0–1500` W/m²（NC 变量 `valid_range`） |
| DQF | 默认掩膜 `DQF==3`；可选 `--mask-dqf2` 掩膜 `DQF==2` |
| V1.0 注意 | **20180501** 为 V1.0（`Version Of Software`≈V1.0），`65532` 表示 **SZA>70°**（V1.1 文档为 >90°）；该日 DQF 常全为 3（未分级），此时跳过 DQF 掩膜，依赖 65532/65535 清洗 |

step_02 通过 `read_fy4_ssi_with_dqf()` 共用上述规则。经纬度 raw 缺失时 step_02 自动用 NSMC/CGMS 公式（104.7°E）生成。

> **迁移说明**：2026-06 之前曾将 UTC 文件名误标为 CST 本地时间；旧产物已清理。当前生产输出统一写入 `*_utc` 子目录（如 `resampled_on_fy3/20180501_utc/`、`matching/20180501_utc/`、`samples/20180501_utc/`）。

## 生产工作流（按顺序）

| 步骤 | 脚本 | 作用 |
|------|------|------|
| 共用 | `fy_io.py` | 常量、F 盘默认路径、FY-3/FY-4 数据读取、UTC/CST 转换 |
| step_01 | `step_01_estimate_fy3c_timestamp_from_sz.py` | 用 FY-3C Sz 太阳天顶角逐像元反推观测时间（CST） |
| step_02 | `step_02_resample_fy4_times_to_fy3_grid.py` | 批量将 FY-4A 各时次（默认 **UTC 00:00–06:00** ≈ CST 08:00–14:00）重采样到 FY-3C 全幅网格，输出 `FY4_HHMM_on_FY3.npz` |
| step_03 | `step_03_build_fy3_fy4_slot_map.py` | **师兄半开窗口匹配**：FY-3 CST `best_minute` 转 UTC 后落入 FY-4 `[start, start+15min)` 才归属该时次；输出 `fy4_slot_map` 与统计 |
| step_04 | `step_04_resample_rest2_ghi_to_fy3_grid.py` | 将 REST2 晴空辐照度重采样到 FY-3C 公共网格 |
| step_05 | `step_05_build_fy3_fy4_samples.py` | 从第 2–4 步产物构建 **FY-3 64×64 / FY-4 16×16** CSI 训练 patch |

> **时间匹配（师兄逻辑）**：FY-3C Gc 为**按自然日归档的过境逐像元瞬时 SSI**，观测时刻由 Sz 反演 `best_minute` 给出（**CST**，沿经度条带变化）；FY-4A 为各时次 NC，文件名时次为 **UTC**。每个 FY-4 文件对应 UTC 半开窗 `[HH:MM, HH:MM+15min)`；FY-3 像元仅在 `best_minute`（转 UTC 后）落入该窗时匹配该时次（**非最近邻**）。step_03 生成逐像元 `fy4_slot_map`；step_05 要求 patch 内 `slot>=0` 像元仅含单一 FY-4 时次后，从 `FY4_*_on_FY3.npz` 取对应 FY-4 SSI。

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

推荐使用项目独立 Conda 环境：

```powershell
& D:\ana\Scripts\conda.exe env create -f environment.yml
& D:\ana\Scripts\conda.exe run -n super-resolution python --version
```

环境名固定为 `super-resolution`；依赖版本记录在项目根目录的 `environment.yml`。
在 Anaconda Prompt 中可执行 `conda activate super-resolution`；若 PowerShell 执行策略禁止
加载 Conda hook，则直接使用 `conda.exe run -n super-resolution ...`，无需放宽系统策略。

**日期范围**：各 step 脚本支持可选 `--end-date YYYYMMDD`（含），与 `--date` 一起使用时在脚本内逐日循环，并打印 `[3/31] 20180503` 进度；省略 `--end-date` 时行为与原先单 `--date` 相同。

step_05 默认在 FY-3 全国覆盖网格 `73–135°E, 18–54°N` 上切样本；FY-3 patch 为 `128×128`，滑窗步长同为 `128`，因此空间样本互不重叠。沿用原质控容差：时次覆盖率至少 99%、联合有效率与 CSI 有效率至少 95%、时间反演误差超限比例不超过 5%；写盘前另行要求四个最终数组全部为有限值。CSI 不按 `[0, 1.5]` 等数值范围筛选或截断。每个未压缩 NPZ batch 同时保存 `fy3`、`fy4`、`hr_csi` 与 `lr_csi` 四个配对数组。

```powershell
# 整月预处理示例（step_01、step_02 与 step_04）
py -3 FY_SSI_processing\step_01_estimate_fy3c_timestamp_from_sz.py --date 20180501 --end-date 20180531 --stride 1 --start 07:00 --end 15:30 --use-local-time-prior --prior-window-minutes 120 --max-error-deg 0.5
py -3 FY_SSI_processing\step_02_resample_fy4_times_to_fy3_grid.py --date 20180501 --end-date 20180531 --skip-existing
py -3 MERRA2\build_rest2_ghi_slot_cache.py --date 20180501 --end-date 20180531 --skip-existing
py -3 FY_SSI_processing\step_04_resample_rest2_ghi_to_fy3_grid.py --date 20180501 --end-date 20180531 --skip-existing --build-rest2-cache
```

### step_01 — 用 FY-3C Sz 估计像元观测时间戳

```powershell
py -3 FY_SSI_processing\step_01_estimate_fy3c_timestamp_from_sz.py --date 20180501 --stride 4
```

默认读取 `F:\FY-3C\SSI\Sz_20180501.dat`，在北京时间/CST(UTC+8) `08:00-12:00` 窗口内逐分钟计算太阳天顶角，并与 FY-3C Sz 找最小绝对差值。若要跑全分辨率，使用 `--stride 1`（默认），但计算量约为 3600×6200×241 次像元分钟，耗时会明显更长。

推荐启用地方时先验（仓库内保留版本为 `prior1015w120`）：

```powershell
py -3 FY_SSI_processing\step_01_estimate_fy3c_timestamp_from_sz.py `
  --date 20180501 --stride 1 --start 07:00 --end 15:30 `
  --use-local-time-prior --local-overpass-time 10:15 --prior-window-minutes 120 `
  --max-error-deg 0.5
```

全国 73–135°E 网格上的 `10:15 ±120 min` 地方太阳时先验约对应 CST
`07:15–15:23`，因此全局搜索窗需覆盖该范围；`08:00–14:00` 会截断东西部像元。
原始 `Sz == 0` 按产品填充值处理，不参与反演；`--max-error-deg` 会将不能在阈值内
重现观测 Sz 的时间戳置为无效，同时在 npz 中保留 `best_diff` 供诊断。

推荐输出（`outputs/fy/timestamp/`）：

- `fy3c_timestamp_20180501_prior1015w120_stride1.npz` — `best_minute`、`best_diff`、`lons`、`lats` 等
- 同名 `.png` 与 `fy3c_timestamp_error_*.png`

### step_02 — 批量 FY-4A 重采样到 FY-3C 网格

```powershell
py -3 FY_SSI_processing\step_02_resample_fy4_times_to_fy3_grid.py --list-only
py -3 FY_SSI_processing\step_02_resample_fy4_times_to_fy3_grid.py --limit 2
py -3 FY_SSI_processing\step_02_resample_fy4_times_to_fy3_grid.py --skip-existing
```

默认 `--start 00:00 --end 06:00`（**UTC**；约等于 CST 08:00–14:00）。

默认输出目录 `outputs/fy/resampled_on_fy3/20180501_utc/`（`--output-dir` 可改）：

- `FY4_0000_on_FY3.npz` — `ssi`、`lons`/`lats`、`source_nc`、`time_utc`、`time_cst` 等

20180501 在 UTC 00:00–06:00 窗口内 9 个时次（文件名实测）：**0000、0100、0200、0245、0300、0400、0500、0545、0600**（约 CST 08:00–14:00）。

空间重采样方法：`--method nearest|gauss`（pyresample 的**空间**插值，与时间匹配无关）。

### step_03 — FY-3 / FY-4 半开窗口匹配统计

```powershell
py -3 FY_SSI_processing\step_03_build_fy3_fy4_slot_map.py
```

默认输入：step_01 时间戳 npz、step_02 目录 `outputs/fy/resampled_on_fy3/20180501_utc/`。

默认输出 `outputs/fy/matching/20180501_utc/`：

- `fy4_slot_map_20180501_prior1015w120_stride1.npz`
- `fy4_slot_map_20180501.png`

### step_04 — REST2 晴空辐照度重采样到 FY-3C 网格

```powershell
py -3 FY_SSI_processing\step_04_resample_rest2_ghi_to_fy3_grid.py --date 20180501 --skip-existing --build-rest2-cache
```

默认读取 MERRA2 REST2 时次缓存，并写入 `outputs/fy/rest2_on_fy3/20180501_utc/`。传入 `--build-rest2-cache` 时，若缓存缺失会调用 `MERRA2\build_rest2_ghi_slot_cache.py` 创建。

### step_05 — FY-3C / FY-4A 时空配对切样本

```powershell
py -3 FY_SSI_processing\step_05_build_fy3_fy4_samples.py --max-samples 5
py -3 FY_SSI_processing\step_05_build_fy3_fy4_samples.py
```

默认输入：step_01 时间戳、step_02 `FY4_*_on_FY3.npz`、step_03 `fy4_slot_map` 与 step_04 REST2 场。

两年数据集使用总控脚本逐日重建，所有最终样本与中间文件均写入 E 盘：

```powershell
python FY_SSI_processing\rebuild_2018_2019_dataset.py run --date 20180501
python FY_SSI_processing\rebuild_2018_2019_dataset.py run
python FY_SSI_processing\rebuild_2018_2019_dataset.py validate --date 20180501
python FY_SSI_processing\rebuild_2018_2019_dataset.py finalize
```

根目录默认为 `E:\Super-Resolution-Samples-2018-2019`。每日最终目录直接包含：

批量模式默认同时处理 2 天（`--workers 2`），适配当前 16 GB 内存；如需降低资源占用可用 `--workers 1`。样本步骤会直接在线构建 slot map，不再额外写匹配图和匹配 NPZ；时间反演也不生成诊断图。

- `manifest.csv`
- `YYYYMMDD_0001.npz` 等未压缩 batch
- `build_meta.npz`

**FY-4 patch 下采样（step_05）**：从与 FY-3 同范围的 128×128 重采样 SSI 上取 `patch[::4, ::4]` 得到 32×32（`pool=4`），与 FY-4 4 km 等效分辨率对应；**不是**对每 4×4 像元做平均。128×128 重采样场在图上可能呈阶梯状（step_02 `nearest` 引起），属预期。

## 输出目录说明

| 子目录 | 内容 |
|--------|------|
| `timestamp/` | step_01 时间戳 npz 与诊断图 |
| `resampled_on_fy3/<date>_utc/` | step_02 各时次 `FY4_HHMM_on_FY3.npz`（**HHMM = UTC**） |
| `matching/<date>_utc/` | step_03 slot 图 |
| `rest2_on_fy3/<date>_utc/` | step_04 晴空辐照度场 |
| `samples/<date>_utc/` | step_05 训练 patch |

已清理的过时中间结果（按需重新生成）：`outputs/fy/resample/`（旧版中间产物）、`samples/_archive/`、一次性重采样对比脚本与验证图等；无 `_utc` 后缀的 20180501 流水线输出（CST 误标时代产物）应全部删除。

## 依赖

```powershell
pip install numpy matplotlib netCDF4 pyresample
```

读取 FY-4A NC 时 `netCDF4` 与 `xarray` 二选一即可。

## 帮助

```powershell
py -3 FY_SSI_processing\step_01_estimate_fy3c_timestamp_from_sz.py --help
py -3 FY_SSI_processing\step_02_resample_fy4_times_to_fy3_grid.py --help
py -3 FY_SSI_processing\step_03_build_fy3_fy4_slot_map.py --help
py -3 FY_SSI_processing\step_04_resample_rest2_ghi_to_fy3_grid.py --help
py -3 FY_SSI_processing\step_05_build_fy3_fy4_samples.py --help
```
