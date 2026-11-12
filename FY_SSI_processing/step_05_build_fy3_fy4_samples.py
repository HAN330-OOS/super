"""
FY-3C / FY-4A 时空配对切训练样本（默认 20180501 单日）。

核心思想：
  1) FY-3C 与 FY-4A 的 SSI 以及 REST2 晴空 GHI 都已经重采样到同一个
     0.01° 网格（3600×6200）上，可以逐像元对齐。
  2) 在公共网格上滑动窗口切配对 patch：
       - HR 真值 = FY-3C Gc 128×128（约 128 km 见方，每格约 1 km）；
       - LR 输入 = FY-4A SSI 32×32（同一区域内的真实 4 km 像元，128/4=32）。
     两者地理范围严格相同，尺寸比 4:1，构成标准 ×4 超分任务。
  3) 同时计算晴空指数（CSI = SSI / REST2 晴空 GHI）：
       - hr_csi = FY3_Gc / REST2_GHI      （128×128）
       - lr_csi = FY4_SSI_LR / REST2_LR   （32×32，先抽粗网格再相除）
  4) 时间配对沿用“师兄半开窗口”逻辑：FY-4 时次对应 UTC 半开窗
     [start, start+15min)，FY-3C best_minute（CST）转 UTC 后落入该窗才匹配。
  5) QC 要求：patch 内 slot 匹配覆盖率、SSI 有效比例、best_diff 比例、
     GHI>0 且 CSI 为有限值的比例都达标，才保留该样本；CSI 不设数值范围。

输出：sample_index.csv（每个样本的索引行）+ patches/batch_*.npz
（含 fy3/fy4 SSI 与 hr_csi/lr_csi）+ build_meta.npz（构建参数与统计）。
"""

# ============================================================
# 模块级说明：下面这段注释帮你快速理解本脚本的骨架
# ------------------------------------------------------------
# 1) 解析命令行参数（日期、区域、patch 尺寸、QC 阈值、输出目录）。
# 2) 从 step_02 输出目录发现当天 FY-4A 时次（UTC 半开窗）。
# 3) 加载 FY-3C 时间戳（step_01）与 slot 匹配图（step_03，缺失则在线算）。
# 4) 裁出研究区（经纬度 → 行列号），预加载 Gc / FY-4 SSI / REST2 GHI。
# 5) 滑动窗口逐个 patch 做 QC，通过后生成 SSI 与 CSI 配对并攒批写盘。
# 6) 写样本索引 CSV 与构建元数据，打印接受/拒绝统计。
# ============================================================

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fy_io import (
    DEFAULT_FY3_ROOT,
    DEFAULT_LAT_MAX,
    DEFAULT_LAT_MIN,
    DEFAULT_LON_MAX,
    DEFAULT_LON_MIN,
    FY3_STEP,
    OUTPUT_MATCHING_DIR,
    OUTPUT_RESAMPLED_ON_FY3_DIR,
    OUTPUT_REST2_ON_FY3_DIR,
    OUTPUT_SAMPLES_DIR,
    PROJECT_ROOT,
    build_target_grid,
    default_timestamp_npz,
    discover_fy4_slots_from_resampled,
    dual_time_label,
    match_pixels_to_half_open_windows,
    minute_to_hhmm,
    read_fy3_binary,
    region_slice,
    region_tag,
    resampled_on_fy3_path,
    rest2_on_fy3_path,
    utc_minutes_to_cst_minutes,
)

# ================= 默认 patch 尺寸 =================
# FY-3C HR patch 128×128：0.01° 格点，约 128 km 见方，每格约 1 km。
# FY-4A LR patch 32×32：同一地理区域内真实 4 km 像元数（128 / 4 = 32）。
# 两个 patch 地理范围严格相同，尺寸比固定 4:1 → 标准 ×4 超分任务。
# DRCT 的 window_size=16，LR 32×32 正好切成 2×2 共 4 个窗口，
# 让 W-MSA / SW-MSA 能正常发挥跨窗口信息交换。
FY3_PATCH = 128
FY4_PATCH = 32
FY4_POOL = FY3_PATCH // FY4_PATCH  # 4 km effective from 1 km grid

# step_01 输出的默认时间戳文件（20180501 单日）。
DEFAULT_TIMESTAMP_NPZ = default_timestamp_npz("20180501")


@dataclass(frozen=True)
class Fy4Slot:
    """FY-4A 单个时次（slot）的元信息。

    utc_label: 文件名时次，如 "0400"（UTC，半开窗 [04:00, 04:15)）。
    cst_label: 对应的北京时间标签（UTC+8）。
    start_minute_utc: 窗口起始分钟（UTC，取值 0-1439）。
    path: step_02 重采样 npz（FY4_HHMM_on_FY3.npz）的路径。
    """
    utc_label: str
    cst_label: str
    start_minute_utc: int
    path: Path

    @property
    def label(self) -> str:
        return self.utc_label


def parse_args() -> argparse.Namespace:
    # 解析命令行参数：日期、输入目录、patch 尺寸、QC 阈值与输出路径。
    parser = argparse.ArgumentParser(
        description="FY-3C / FY-4A 时空配对切训练样本（师兄半开窗口匹配，默认 20180501）。",
        epilog=(
            "小规模测试示例:\n"
            "  py -3 FY_SSI_processing\\step_05_build_fy3_fy4_samples.py "
            "--max-samples 5\n\n"
            "默认使用 FY-3 全国覆盖网格 73-135E, 18-54N。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--date", default="20180501", help="日期 YYYYMMDD。")
    parser.add_argument("--fy3-gc", type=Path, default=None, help="FY-3C Gc .dat 路径。")
    parser.add_argument(
        "--timestamp-npz",
        type=Path,
        default=DEFAULT_TIMESTAMP_NPZ,
    )
    parser.add_argument(
        "--resampled-dir",
        type=Path,
        default=None,
        help="step_02 输出的 FY4_*_on_FY3.npz 目录。",
    )
    parser.add_argument(
        "--slot-map-npz",
        type=Path,
        default=None,
        help="step_03 输出的 fy4_slot_map npz；缺省则按同逻辑在线计算。",
    )
    parser.add_argument("--window-minutes", type=int, default=15, help="半开窗宽度（分钟）。")
    parser.add_argument("--lon-min", type=float, default=DEFAULT_LON_MIN)
    parser.add_argument("--lon-max", type=float, default=DEFAULT_LON_MAX)
    parser.add_argument("--lat-min", type=float, default=DEFAULT_LAT_MIN)
    parser.add_argument("--lat-max", type=float, default=DEFAULT_LAT_MAX)
    parser.add_argument(
        "--patch-stride",
        type=int,
        default=FY3_PATCH,
        help="滑动窗口步长（像元）；默认等于 FY-3 patch 尺寸，样本互不重叠。",
    )
    parser.add_argument("--fy3-patch", type=int, default=FY3_PATCH)
    parser.add_argument("--fy4-patch", type=int, default=FY4_PATCH)
    # ---- REST2 / 晴空指数（CSI）相关参数 ----
    # CSI = SSI / REST2 晴空 GHI；只要求 GHI>0 且计算结果有限，
    # 不按 CSI 数值范围筛选样本。
    parser.add_argument("--rest2-dir", type=Path, default=None,
                        help="REST2 GHI resampled dir (step_04 output).")
    parser.add_argument("--min-csi-valid-frac", type=float, default=0.95,
                        help="CSI 为有限值的最小像元比例；不限制 CSI 数值范围。")
    parser.add_argument("--max-samples", type=int, default=None, help="最多保留样本数；测试时可设小值。")
    parser.add_argument("--batch-size", type=int, default=64, help="每个 patch npz 打包的样本数。")
    parser.add_argument("--max-diff-deg", type=float, default=0.5, help="像元 best_diff 上限（度）。")
    parser.add_argument(
        "--max-bad-diff-frac",
        type=float,
        default=0.05,
        help="patch 内 best_diff 超阈值像元比例上限。",
    )
    parser.add_argument(
        "--min-valid-frac",
        type=float,
        default=0.95,
        help="patch 内 FY-3 Gc 与 FY-4 SSI 同时有效的最小比例。",
    )
    parser.add_argument(
        "--min-slot-valid-frac",
        type=float,
        default=0.99,
        help=(
            "patch 内有时戳匹配的像元比例下限："
            "slot_valid_frac = N(slot>=0) / (fy3_patch^2)，默认 0.99。"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--estimate-only", action="store_true", help="仅统计候选 patch 数，不写样本。")
    return parser.parse_args()


def default_fy3_gc_path(date: str) -> Path:
    """返回 FY-3C Gc 二进制文件的默认路径（E:\\FY-3C\\SSI\\Gc_<date>.dat）。"""
    return DEFAULT_FY3_ROOT / f"Gc_{date}.dat"


def default_slot_map_npz(date: str, timestamp_npz: Path) -> Path:
    """按时间戳文件名推断 step_03 slot map 的默认输出路径。

    timestamp npz 名称含 "prior1015w120_stride1" 时，slot map 也带相同标签，
    保证两者出自同一套时间反演参数。
    """
    tag = "prior1015w120_stride1" if "prior1015w120" in timestamp_npz.name else "timestamp"
    return OUTPUT_MATCHING_DIR / f"{date}_utc" / f"fy4_slot_map_{date}_{tag}.npz"


def load_timestamp(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """读取 step_01 时间戳 npz。

    返回:
      best_minute: 每个 FY-3C 像元反演的观测时刻（CST 分钟数，无效为负值）。
      best_diff:   反演时刻与观测太阳天顶角的偏差（度），用于 QC。
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"时间戳 npz 不存在: {npz_path}")
    data = np.load(npz_path)
    if "best_minute" not in data or "best_diff" not in data:
        raise KeyError(f"{npz_path} 缺少 best_minute / best_diff。")
    return data["best_minute"], data["best_diff"].astype(np.float32)


def build_fy4_slots(resampled_dir: Path, window_minutes: int) -> list[Fy4Slot]:
    """从 step_02 输出目录发现当天所有 FY-4A 时次并封装成 Fy4Slot 列表。"""
    discovered = discover_fy4_slots_from_resampled(resampled_dir, window_minutes)
    return [
        Fy4Slot(
            utc_label=utc_label,
            cst_label=cst_label,
            start_minute_utc=start,
            path=resampled_on_fy3_path(resampled_dir, utc_label),
        )
        for utc_label, start, _end, cst_label in discovered
    ]


def load_or_build_slot_map(
    best_minute: np.ndarray,
    best_diff: np.ndarray,
    slot_starts: np.ndarray,
    window_minutes: int,
    slot_map_npz: Path | None,
) -> np.ndarray:
    """优先加载 step_03 的 slot map；不存在时按半开窗口逻辑在线计算。

    返回的数组与 FY-3C 网格同尺寸，每个像元的值为 FY-4A 时次下标，
    -1 表示该像元没有匹配到任何 FY-4A 时次。
    """
    if slot_map_npz is not None and slot_map_npz.exists():
        data = np.load(slot_map_npz)
        if "fy4_slot_map" not in data:
            raise KeyError(f"{slot_map_npz} 缺少 fy4_slot_map。")
        slot_map = data["fy4_slot_map"]
        if slot_map.shape != best_minute.shape:
            raise ValueError(
                f"slot_map shape {slot_map.shape} 与 best_minute {best_minute.shape} 不一致。"
            )
        print(f"  已加载 slot_map: {slot_map_npz}")
        return slot_map.astype(np.int16, copy=False)

    valid = (best_minute >= 0) & np.isfinite(best_diff)
    slot_map = match_pixels_to_half_open_windows(
        best_minute,
        valid,
        slot_starts,
        window_minutes=window_minutes,
        best_minute_tz="CST",
    )
    print("  在线计算 slot_map（半开窗口，非最近邻）。")
    return slot_map


def load_resampled_ssi(path: Path) -> np.ndarray:
    """读取 step_02 输出的 FY-4A SSI 重采样 npz（键名 ssi，3600×6200）。"""
    if not path.exists():
        raise FileNotFoundError(f"重采样 npz 不存在: {path}")
    data = np.load(path)
    if "ssi" not in data:
        raise KeyError(f"{path} 缺少 ssi 数组。")
    return data["ssi"].astype(np.float32, copy=False)


def load_rest2_ghi(path: Path) -> np.ndarray:
    """读取 step_04 输出的 REST2 晴空 GHI 重采样 npz（键名 ghi，3600×6200）。"""
    if not path.exists():
        raise FileNotFoundError(f"REST2 npz 不存在: {path}")
    data = np.load(path)
    if "ghi" not in data:
        raise KeyError(f"{path} 缺少 ghi 数组。")
    return data["ghi"].astype(np.float32, copy=False)


def subsample_fy4_patch(patch: np.ndarray, pool: int, fy4_patch: int) -> np.ndarray:
    """把 0.01° 网格上的 FY-4 展开图抽回真实粗网格。

    重采样后的 FY-4 SSI 每个 4 km 像元占 4×4 个细格点，
    这里每隔 pool 个像元取一个（patch[::pool, ::pool]），
    得到 (fy4_patch, fy4_patch) 的无冗余 LR patch。
    """
    sampled = patch[::pool, ::pool]
    if sampled.shape != (fy4_patch, fy4_patch):
        raise ValueError(f"FY-4 patch shape {sampled.shape} != ({fy4_patch}, {fy4_patch})。")
    return sampled.astype(np.float32, copy=False)


def patch_single_slot(slot_patch: np.ndarray) -> tuple[int | None, int]:
    """判断 patch 内 slot>=0 像元是否只属于一个 FY-4 时次。

    返回 (dominant_slot, n_slots)：
      - 只含一个时次时返回该时次下标；
      - 没有匹配或含多个时次时返回 (None, n_slots)，调用方应丢弃该 patch。
    """
    valid = slot_patch >= 0
    if not np.any(valid):
        return None, 0

    slots = slot_patch[valid]
    values = np.unique(slots)
    n_slots = int(values.size)
    if n_slots != 1:
        return None, n_slots
    return int(values[0]), n_slots


def patch_passes_qc(
    gc_patch: np.ndarray,
    fy4_patch_full: np.ndarray,
    rest2_patch: np.ndarray,
    diff_patch: np.ndarray,
    slot_patch: np.ndarray,
    dominant_slot: int,
    args: argparse.Namespace,
) -> tuple[bool, dict[str, float]]:
    """对单个 128×128 patch 做质量控制，并统计各项覆盖率。

    有效条件（逐像元同时满足）：
      - FY-3C Gc 有限；
      - FY-4A SSI 有限；
      - 像元有匹配时次（slot >= 0）；
      - best_diff 小于 max_diff_deg（时间反演误差可控）；
      - REST2 GHI 有限且 > 0（晴空分母有效）。
    另外要求 HR 晴空指数为有限值的像元比例不低于
    min_csi_valid_frac，但不限制 CSI 的数值范围。
    """
    # ---- 各数据源的有效性掩膜 ----
    valid_gc = np.isfinite(gc_patch)
    valid_fy4 = np.isfinite(fy4_patch_full)
    valid_time = slot_patch >= 0
    valid_diff = np.isfinite(diff_patch) & (diff_patch < args.max_diff_deg)
    valid_rest2 = np.isfinite(rest2_patch) & (rest2_patch > 0)
    joint = valid_gc & valid_fy4 & valid_time & valid_diff & valid_rest2

    # ---- 计算 HR 晴空指数并统计有效比例（只用于 QC） ----
    hr_csi = np.full(gc_patch.shape, np.nan, dtype=np.float32)
    np.divide(gc_patch, rest2_patch, out=hr_csi, where=valid_rest2 & valid_gc)
    valid_csi = np.isfinite(hr_csi)

    # ---- 各项统计量：有效覆盖率 / 坏时间误差比例 / 主时次占比 / CSI 占比 ----
    valid_frac = float(joint.mean())
    bad_diff_frac = float((valid_time & np.isfinite(diff_patch) & (diff_patch >= args.max_diff_deg)).mean())
    dominant_slot_frac = float((slot_patch == dominant_slot).mean())
    csi_valid_frac = float((valid_rest2 & valid_gc & valid_csi).mean())

    stats = {
        "valid_frac": valid_frac,
        "bad_diff_frac": bad_diff_frac,
        "dominant_slot_frac": dominant_slot_frac,
        "csi_valid_frac": csi_valid_frac,
        "n_slots": float(np.unique(slot_patch[valid_time]).size),
    }

    # ---- 硬性门槛：任一不达标即拒绝该 patch ----
    if valid_frac < args.min_valid_frac:
        return False, stats
    if bad_diff_frac > args.max_bad_diff_frac:
        return False, stats
    if csi_valid_frac < args.min_csi_valid_frac:
        return False, stats
    if not np.any(joint):
        return False, stats
    return True, stats


def count_candidate_patches(nrows: int, ncols: int, patch: int, stride: int) -> int:
    """统计滑动窗口在 (nrows, ncols) 区域内的候选 patch 总数（QC 前）。"""
    if nrows < patch or ncols < patch:
        return 0
    row_starts = range(0, nrows - patch + 1, stride)
    col_starts = range(0, ncols - patch + 1, stride)
    return len(row_starts) * len(col_starts)


def write_batch_npz(
    batch_path: Path,
    fy3_list: list[np.ndarray],
    fy4_list: list[np.ndarray],
    hr_csi_list: list[np.ndarray],
    lr_csi_list: list[np.ndarray],
    meta: dict[str, list],
) -> None:
    """把一个 batch 的样本写入压缩 npz。

    存储内容：
      fy3 / fy4 —— FY-3C HR 与 FY-4A LR 的 SSI；
      hr_csi / lr_csi —— HR 128×128 / LR 32×32 的晴空指数；
      meta —— 每个样本的行列起点、中心经纬度、时次、QC 统计等。
    """
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        batch_path,
        fy3=np.stack(fy3_list, axis=0).astype(np.float32),
        fy4=np.stack(fy4_list, axis=0).astype(np.float32),
        hr_csi=np.stack(hr_csi_list, axis=0).astype(np.float32),
        lr_csi=np.stack(lr_csi_list, axis=0).astype(np.float32),
        **{key: np.asarray(values) for key, values in meta.items()},
    )


def main() -> None:
    # ---------- 1. 参数解析与基本校验 ----------
    args = parse_args()
    # FY-3 与 FY-4 patch 必须满足 fy3_patch = pool * fy4_patch（默认 pool=4），
    # 这样 HR 与 LR 才能覆盖同一地理范围。
    pool = args.fy3_patch // args.fy4_patch
    if args.fy3_patch != pool * args.fy4_patch:
        raise ValueError(
            f"fy3_patch={args.fy3_patch} 必须能被 fy4_patch={args.fy4_patch} 整除。"
        )

    # ---------- 2. 确定输入 / 输出目录 ----------
    # resampled_dir: step_02 的 FY-4A SSI 重采样结果；
    # rest2_dir:     step_04 的 REST2 晴空 GHI 重采样结果；
    # 两者都按 <date>_utc 子目录组织（UTC 时次）。
    resampled_dir = args.resampled_dir or (OUTPUT_RESAMPLED_ON_FY3_DIR / f"{args.date}_utc")
    rest2_dir = args.rest2_dir or (OUTPUT_REST2_ON_FY3_DIR / f"{args.date}_utc")
    slot_map_npz = args.slot_map_npz or default_slot_map_npz(args.date, args.timestamp_npz)

    # 输出目录：outputs/fy/samples/<date>/，NPZ 与 manifest.csv 直接放在日期目录。
    tag = region_tag(args.lon_min, args.lon_max, args.lat_min, args.lat_max)
    out_dir = args.output_dir or (OUTPUT_SAMPLES_DIR / f"{args.date}_utc")
    patch_dir = out_dir

    # FY-3C Gc 二进制（3600×6200 float32，0.01° 网格，逐像元瞬时 SSI）。
    gc_path = args.fy3_gc or default_fy3_gc_path(args.date)

    # ---------- 3. 打印运行配置，便于核对每次构建的参数 ----------
    print("FY-3C / FY-4A 配对切样本（师兄半开窗口）:")
    print(f"  日期: {args.date}")
    print(f"  FY-3C Gc: {gc_path}")
    print(f"  时间戳: {args.timestamp_npz}")
    print(f"  FY-4 重采样目录: {resampled_dir}")
    print(f"  slot_map: {slot_map_npz if slot_map_npz.exists() else '(在线计算)'}")
    print(f"  区域: {args.lon_min}-{args.lon_max}E, {args.lat_min}-{args.lat_max}N")
    print(
        f"  Patch: FY-3 {args.fy3_patch}x{args.fy3_patch}, "
        f"FY-4 {args.fy4_patch}x{args.fy4_patch}, stride={args.patch_stride}, "
        f"slot_valid>={args.min_slot_valid_frac:.0%}, single FY-4 slot among slot>=0"
    )

    # ---------- 4. 发现当天所有 FY-4A 时次（slot） ----------
    # 每个 slot 对应一个 UTC 半开窗 [start, start+15min)。
    slots = build_fy4_slots(resampled_dir, args.window_minutes)
    slot_starts = np.array([slot.start_minute_utc for slot in slots], dtype=np.int32)
    print(f"  FY-4A 时次: {len(slots)} 个")
    for slot in slots:
        end_utc = slot.start_minute_utc + args.window_minutes
        end_cst = int(utc_minutes_to_cst_minutes(end_utc))
        print(
            f"    FY4_{slot.utc_label} → "
            f"[{minute_to_hhmm(slot.start_minute_utc)}, {minute_to_hhmm(end_utc)}) UTC / "
            f"[{slot.cst_label}, {minute_to_hhmm(end_cst)}) CST"
        )

    # ---------- 5. 加载 FY-3C 时间戳与 slot 匹配图 ----------
    # best_minute 是 FY-3C 每像元的观测时刻（CST），slot map 给出每个像元
    # 匹配到的 FY-4A 时次下标（-1 表示无匹配）。
    best_minute, best_diff = load_timestamp(args.timestamp_npz)
    slot_index_full = load_or_build_slot_map(
        best_minute,
        best_diff,
        slot_starts,
        args.window_minutes,
        slot_map_npz if slot_map_npz.exists() else None,
    )

    # ---------- 6. 裁研究区（经纬度 → 行列号） ----------
    # region_slice 按 FY-3C 网格（73-135°E, 54-18°N, 0.01°）把
    # 经纬度范围换算成行列切片，之后所有数组都只保留这个子区域。
    row_start, row_end, col_start, col_end = region_slice(
        args.lon_min, args.lon_max, args.lat_min, args.lat_max
    )
    # 读取全幅 FY-3C Gc 并裁出研究区（float32，节省后续计算内存）。
    gc_full = read_fy3_binary(gc_path)
    gc_region = gc_full[row_start:row_end, col_start:col_end].astype(np.float32, copy=False)
    diff_region = best_diff[row_start:row_end, col_start:col_end]
    slot_region = slot_index_full[row_start:row_end, col_start:col_end]

    # 统计 QC 前的候选 patch 数量（滑动窗口总数），用于估算构建规模。
    nrows, ncols = gc_region.shape
    candidate_count = count_candidate_patches(nrows, ncols, args.fy3_patch, args.patch_stride)
    print(f"  区域 shape: {gc_region.shape}, 候选 patch 数（QC 前）: {candidate_count}")

    # 仅估算模式：只打印候选数量，不写任何样本。
    if args.estimate_only:
        print("estimate-only: 跳过写样本。")
        return

    # 构建目标网格经纬度（lons_1d/lats_1d），用于记录每个 patch 的中心坐标。
    lons_1d, lats_1d, _target_lons, _target_lats = build_target_grid(
        args.lon_min, args.lon_max, args.lat_min, args.lat_max, FY3_STEP
    )
    if lats_1d.shape[0] != gc_region.shape[0]:
        raise ValueError(f"目标网格行数 {lats_1d.shape[0]} 与 Gc 子区域 {gc_region.shape[0]} 不一致。")

    # 找出研究区内实际出现的 FY-4A 时次，只加载这些时次的数据，避免浪费内存。
    used_slots = sorted(int(v) for v in np.unique(slot_region[slot_region >= 0]))
    print(f"  区域用到的 FY-4 时次: {len(used_slots)} / {len(slots)}")

    # 按时次预加载 FY-4 SSI 与 REST2 GHI（都只裁研究区）。
    resampled_by_slot: dict[int, np.ndarray] = {}
    rest2_by_slot: dict[int, np.ndarray] = {}
    for slot_idx in used_slots:
        slot = slots[slot_idx]
        print(f"  读取 {dual_time_label(slot.utc_label)} ← {slot.path.name}")
        resampled_by_slot[slot_idx] = load_resampled_ssi(slot.path)[
            row_start:row_end, col_start:col_end
        ]
        rest2_by_slot[slot_idx] = load_rest2_ghi(
            rest2_on_fy3_path(rest2_dir, slot.utc_label)
        )[row_start:row_end, col_start:col_end]

    # ---------- 7. 初始化样本缓冲与统计 ----------
    # 每个样本同时保存 SSI 对与 CSI 对，使用未压缩 NPZ 加快后续读取。
    index_rows: list[dict[str, object]] = []
    fy3_batch: list[np.ndarray] = []
    fy4_batch: list[np.ndarray] = []
    hr_csi_batch: list[np.ndarray] = []
    lr_csi_batch: list[np.ndarray] = []
    batch_meta: dict[str, list] = {
        "row0": [],
        "col0": [],
        "center_lon": [],
        "center_lat": [],
        "dominant_slot": [],
        "dominant_slot_minute_utc": [],
        "dominant_slot_minute_cst": [],
        "valid_frac": [],
        "bad_diff_frac": [],
        "dominant_slot_frac": [],
        "csi_valid_frac": [],
    }
    batch_id = 0
    sample_id = 0
    rejected_slot_valid_frac = 0
    rejected_multi_slot = 0
    rejected_qc = 0
    patch_pixels = args.fy3_patch * args.fy3_patch
    accepted_per_slot: dict[int, int] = {idx: 0 for idx in range(len(slots))}

    # ---------- 8. 滑动窗口切 patch + QC + 生成 CSI ----------
    for row0 in range(0, nrows - args.fy3_patch + 1, args.patch_stride):
        for col0 in range(0, ncols - args.fy3_patch + 1, args.patch_stride):
            # 先检查时间匹配覆盖率：patch 内 slot>=0 像元占比必须达标。
            slot_patch = slot_region[row0 : row0 + args.fy3_patch, col0 : col0 + args.fy3_patch]
            slot_valid_frac = float((slot_patch >= 0).sum()) / patch_pixels
            if slot_valid_frac < args.min_slot_valid_frac:
                rejected_slot_valid_frac += 1
                continue

            # 再检查 patch 内匹配像元是否只属于一个 FY-4A 时次（避免混时次）。
            dominant_slot, n_slots_in_patch = patch_single_slot(slot_patch)
            if dominant_slot is None:
                rejected_multi_slot += 1
                continue

            # 从研究区数组里切出 128×128 的各字段 patch：
            # gc（HR SSI）、fy4 展开场（每 4×4 格一个粗像元）、rest2、diff、slot。
            gc_patch = gc_region[row0 : row0 + args.fy3_patch, col0 : col0 + args.fy3_patch]
            fy4_layer = resampled_by_slot[dominant_slot]
            fy4_patch_full = fy4_layer[row0 : row0 + args.fy3_patch, col0 : col0 + args.fy3_patch]
            rest2_layer = rest2_by_slot[dominant_slot]
            rest2_patch = rest2_layer[row0 : row0 + args.fy3_patch, col0 : col0 + args.fy3_patch]
            diff_patch = diff_region[row0 : row0 + args.fy3_patch, col0 : col0 + args.fy3_patch]

            # 统一 QC：SSI/时间/REST2 有效覆盖率、误差比例、CSI 范围占比。
            ok, stats = patch_passes_qc(
                gc_patch, fy4_patch_full, rest2_patch,
                diff_patch, slot_patch, dominant_slot, args
            )
            if not ok:
                rejected_qc += 1
                continue

            # 记录主时次的 UTC/CST 分钟数以及 patch 中心经纬度，便于追溯。
            dominant_minute_utc = int(slots[dominant_slot].start_minute_utc)
            dominant_minute_cst = int(utc_minutes_to_cst_minutes(dominant_minute_utc))
            center_row = row0 + args.fy3_patch // 2
            center_col = col0 + args.fy3_patch // 2
            center_lon = float(lons_1d[center_col])
            center_lat = float(lats_1d[center_row])
            fy3_center_minute_cst = int(best_minute[row_start + center_row, col_start + center_col])

            # ---- 生成晴空指数（CSI）----
            # HR: kc_FY3 = FY3_Gc / REST2_GHI（128×128）
            # LR: 先把 FY-4 SSI 与 REST2 都抽回 32×32 粗网格，再相除
            #      kc_FY4 = FY4_SSI_LR / REST2_LR（32×32）
            # 分母要求 GHI>0；其余位置保持 NaN，训练时作为无效像元掩膜。
            valid_hr_csi = np.isfinite(rest2_patch) & (rest2_patch > 0) & np.isfinite(gc_patch)
            hr_csi = np.full(gc_patch.shape, np.nan, dtype=np.float32)
            np.divide(gc_patch, rest2_patch, out=hr_csi, where=valid_hr_csi)
            lr_fy4 = subsample_fy4_patch(fy4_patch_full, pool, args.fy4_patch)
            lr_rest2 = subsample_fy4_patch(rest2_patch, pool, args.fy4_patch)
            lr_csi = np.full(lr_fy4.shape, np.nan, dtype=np.float32)
            np.divide(
                lr_fy4, lr_rest2, out=lr_csi,
                where=np.isfinite(lr_fy4) & np.isfinite(lr_rest2) & (lr_rest2 > 0),
            )
            if not all(
                np.isfinite(array).all()
                for array in (gc_patch, lr_fy4, hr_csi, lr_csi)
            ):
                rejected_qc += 1
                continue
            # 把 SSI 与 CSI 四个数组收进 batch 缓冲。
            fy3_batch.append(gc_patch.copy())
            fy4_batch.append(lr_fy4.copy())
            hr_csi_batch.append(hr_csi)
            lr_csi_batch.append(lr_csi)

            # 记录该样本的 QC 统计到 batch 元数据（每个样本一行）。
            batch_meta["row0"].append(row0)
            batch_meta["col0"].append(col0)
            batch_meta["center_lon"].append(center_lon)
            batch_meta["center_lat"].append(center_lat)
            batch_meta["dominant_slot"].append(dominant_slot)
            batch_meta["dominant_slot_minute_utc"].append(dominant_minute_utc)
            batch_meta["dominant_slot_minute_cst"].append(dominant_minute_cst)
            batch_meta["valid_frac"].append(stats["valid_frac"])
            batch_meta["bad_diff_frac"].append(stats["bad_diff_frac"])
            batch_meta["dominant_slot_frac"].append(stats["dominant_slot_frac"])
            batch_meta["csi_valid_frac"].append(stats["csi_valid_frac"])

            # 写索引行：batch 文件名 + batch 内下标 + 全局行列 + 时次 + QC 值。
            patch_name = f"{args.date}_{batch_id + 1:04d}.npz"
            index_rows.append(
                {
                    "sample_id": sample_id,
                    "date": args.date,
                    "fy3_time": minute_to_hhmm(fy3_center_minute_cst),
                    "fy4_time": minute_to_hhmm(dominant_minute_utc),
                    "row": row_start + row0,
                    "col": col_start + col0,
                    "center_lon": center_lon,
                    "center_lat": center_lat,
                    "npz_file": patch_name,
                    "array_index": len(hr_csi_batch) - 1,
                }
            )
            sample_id += 1
            accepted_per_slot[dominant_slot] = accepted_per_slot.get(dominant_slot, 0) + 1

            # --max-samples 是调试开关：攒够即停，不写完整批次。
            if args.max_samples is not None and sample_id >= args.max_samples:
                break

            # 攒满一个 batch（默认 64 个样本）就写盘，控制内存占用。
            if len(hr_csi_batch) >= args.batch_size:
                write_batch_npz(
                    patch_dir / patch_name,
                    fy3_batch,
                    fy4_batch,
                    hr_csi_batch,
                    lr_csi_batch,
                    batch_meta,
                )
                fy3_batch, fy4_batch = [], []
                hr_csi_batch, lr_csi_batch = [], []
                batch_meta = {key: [] for key in batch_meta}
                batch_id += 1

        if args.max_samples is not None and sample_id >= args.max_samples:
            break

    # 写最后不足一组的残余样本。
    if hr_csi_batch:
        write_batch_npz(
            patch_dir / f"{args.date}_{batch_id + 1:04d}.npz",
            fy3_batch,
            fy4_batch,
            hr_csi_batch,
            lr_csi_batch,
            batch_meta,
        )

    # ---------- 9. 写样本索引 CSV ----------
    # 每行一个样本，训练/验证时按 batch_file + batch_index 定位。
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "manifest.csv"
    fieldnames = list(index_rows[0].keys()) if index_rows else [
        "sample_id", "date", "fy3_time", "fy4_time", "row", "col",
        "center_lon", "center_lat", "npz_file", "array_index",
    ]
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(index_rows)

    # ---------- 10. 写构建元数据（便于追溯参数与统计） ----------
    rejected_total = rejected_slot_valid_frac + rejected_multi_slot + rejected_qc
    slot_count_labels = np.array([slot.utc_label for slot in slots])
    slot_count_values = np.array(
        [accepted_per_slot.get(idx, 0) for idx in range(len(slots))],
        dtype=np.int64,
    )
    meta_path = out_dir / "build_meta.npz"
    np.savez_compressed(
        meta_path,
        date=args.date,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        region_tag=tag,
        fy3_patch=args.fy3_patch,
        fy4_patch=args.fy4_patch,
        patch_stride=args.patch_stride,
        window_minutes=args.window_minutes,
        min_slot_valid_frac=args.min_slot_valid_frac,
        patch_slot_rule="single unique FY-4 slot among slot>=0 pixels",
        matching_rule="half_open [start, start+15) UTC from filename; FY-3 best_minute CST→UTC; not nearest-neighbor",
        max_diff_deg=args.max_diff_deg,
        max_bad_diff_frac=args.max_bad_diff_frac,
        min_valid_frac=args.min_valid_frac,
        rest2_dir=str(rest2_dir),
        min_csi_valid_frac=args.min_csi_valid_frac,
        candidate_patches=candidate_count,
        accepted_samples=sample_id,
        rejected_patches=rejected_total,
        rejected_slot_valid_frac=rejected_slot_valid_frac,
        rejected_multi_slot=rejected_multi_slot,
        rejected_qc=rejected_qc,
        accepted_per_slot_labels=slot_count_labels,
        accepted_per_slot_counts=slot_count_values,
        used_slot_minutes_utc=slot_starts[used_slots],
        slot_labels_utc=np.array([slots[i].utc_label for i in used_slots]),
        slot_labels_cst=np.array([slots[i].cst_label for i in used_slots]),
        resampled_dir=str(resampled_dir),
        slot_map_npz=str(slot_map_npz) if slot_map_npz.exists() else "",
        timestamp_npz=str(args.timestamp_npz),
    )

    # ---------- 11. 打印汇总 ----------
    print()
    print(f"完成: 接受 {sample_id} 个样本（候选 {candidate_count}）。")
    print(
        f"  拒绝: slot 覆盖率不足 {rejected_slot_valid_frac}, "
        f"多时次 {rejected_multi_slot}, QC 未过 {rejected_qc}, 合计 {rejected_total}"
    )
    print("  各时次接受 patch 数:")
    for idx, slot in enumerate(slots):
        print(f"    {slot.utc_label} ({slot.cst_label} CST): {accepted_per_slot.get(idx, 0)}")
    print(f"  索引: {index_path}")
    print(f"  patch: {patch_dir}")
    print(f"  元数据: {meta_path}")


if __name__ == "__main__":
    main()
