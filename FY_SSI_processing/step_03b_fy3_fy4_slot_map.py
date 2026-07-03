"""
FY-3C / FY-4A 师兄半开窗口匹配统计（非最近邻）。

每个 FY4_HHMM 对应 UTC 半开窗 [HH:MM, HH:MM+15min)（文件名时次，见产品规格 PDF）；
FY-3 像元 best_minute 为 CST，匹配时转为 UTC 与 FY-4 窗口比较。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from fy_io import (
    FY3_LAT_START,
    FY3_LON_START,
    FY3_STEP,
    OUTPUT_MATCHING_DIR,
    OUTPUT_RESAMPLED_ON_FY3_DIR,
    PROJECT_ROOT,
    cst_minutes_to_utc_minutes,
    default_timestamp_npz,
    discover_fy4_slots_from_resampled,
    dual_time_label,
    iter_dates,
    match_pixels_to_half_open_windows,
    minute_to_hhmm,
    utc_minutes_to_cst_minutes,
)

DEFAULT_TIMESTAMP_KWARGS = dict(
    use_prior=True,
    local_overpass_time="10:15",
    prior_window_minutes=120,
    stride=1,
)

# 20180501 白天 9 时次（UTC 文件名，00:00–06:00 窗内实测）
USER_WINDOW_SPECS: list[tuple[str, int, int]] = [
    ("00:00-00:15", 0, 15),
    ("01:00-01:15", 60, 75),
    ("02:00-02:15", 120, 135),
    ("02:45-03:00", 165, 180),
    ("03:00-03:15", 180, 195),
    ("04:00-04:15", 240, 255),
    ("05:00-05:15", 300, 315),
    ("05:45-06:00", 345, 360),
    ("06:00-06:15", 360, 375),
]

ORBIT_STRIP_CENTERS_E = (74.0, 89.0, 114.0, 132.0)
STRIP_LON_HALF_WIDTH = 1.0

# 条带参考时间范围（CST，用于核对）
STRIP_REFERENCE_RANGES: dict[float, tuple[str, str]] = {
    74.0: ("08:39", "08:48"),
    89.0: ("10:21", "10:29"),
    114.0: ("12:03", "12:11"),
    132.0: ("13:45", "13:52"),
}


@dataclass(frozen=True)
class Fy4Window:
    utc_label: str
    start_minute: int
    end_minute: int
    cst_label: str

    @property
    def label(self) -> str:
        """主键：UTC HHMM（与 npz 文件名一致）。"""
        return self.utc_label

    @property
    def range_label_utc(self) -> str:
        return f"{minute_to_hhmm(self.start_minute)}-{minute_to_hhmm(self.end_minute)}"

    @property
    def range_label_cst(self) -> str:
        cst_start = int(utc_minutes_to_cst_minutes(self.start_minute))
        cst_end = int(utc_minutes_to_cst_minutes(self.end_minute))
        return f"{minute_to_hhmm(cst_start)}-{minute_to_hhmm(cst_end)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FY-3C best_minute 与 FY-4 半开 15min 窗口匹配统计（师兄逻辑）。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--date", default="20180501", help="日期 YYYYMMDD。")
    parser.add_argument("--end-date", default=None, help="结束日期 YYYYMMDD（含）；给定则自 --date 起逐日处理。")
    parser.add_argument(
        "--timestamp-npz",
        type=Path,
        default=None,
        help="step_02 输出的 best_minute / best_diff npz；默认按 --date 推断。",
    )
    parser.add_argument(
        "--resampled-dir",
        type=Path,
        default=None,
        help="FY4_*_on_FY3.npz 目录，默认 outputs/fy/resampled_on_fy3/<date>/。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录，默认 outputs/fy/matching/<date>/。",
    )
    parser.add_argument("--window-minutes", type=int, default=15, help="半开窗宽度（分钟）。")
    parser.add_argument("--plot-max-pixels", type=int, default=1600)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--patch-stride", type=int, default=32)
    parser.add_argument(
        "--patch-subsample",
        type=int,
        default=4,
        help="patch 粗估时的网格下采样步长（仅用于快速估计）。",
    )
    parser.add_argument("--strip-lon-half-width", type=float, default=STRIP_LON_HALF_WIDTH)
    return parser.parse_args()


def parse_hhmm_range_part(text: str) -> int:
    text = text.strip()
    if ":" in text:
        hour, minute = text.split(":", maxsplit=1)
        return int(hour) * 60 + int(minute)
    return int(text[:2]) * 60 + int(text[2:])


def minute_label(minute_of_day: int | float) -> str:
    if not np.isfinite(minute_of_day):
        return "NA"
    minute = int(minute_of_day)
    return f"{minute // 60:02d}:{minute % 60:02d}"


def load_timestamp(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not npz_path.exists():
        raise FileNotFoundError(f"时间戳 npz 不存在: {npz_path}")
    data = np.load(npz_path)
    if "best_minute" not in data or "best_diff" not in data:
        raise KeyError(f"{npz_path} 缺少 best_minute / best_diff。")
    best_minute = data["best_minute"]
    best_diff = data["best_diff"].astype(np.float32)
    if "lons" in data and "lats" in data:
        lons = np.asarray(data["lons"], dtype=np.float64)
        lats = np.asarray(data["lats"], dtype=np.float64)
    else:
        rows, cols = best_minute.shape
        lons = FY3_LON_START + np.arange(cols, dtype=np.float64) * FY3_STEP
        lats = FY3_LAT_START - np.arange(rows, dtype=np.float64) * FY3_STEP
    return best_minute, best_diff, lons, lats


def build_fy4_windows(resampled_dir: Path, window_minutes: int) -> list[Fy4Window]:
    discovered = discover_fy4_slots_from_resampled(resampled_dir, window_minutes)
    return [
        Fy4Window(utc_label=utc_label, start_minute=start, end_minute=end, cst_label=cst_label)
        for utc_label, start, end, cst_label in discovered
    ]


def verify_user_windows(windows: list[Fy4Window]) -> None:
    expected_starts = {start for _, start, _ in USER_WINDOW_SPECS}
    actual_starts = {w.start_minute for w in windows}
    if expected_starts != actual_starts:
        missing = expected_starts - actual_starts
        extra = actual_starts - expected_starts
        print(
            "警告: 重采样时次与用户 9 窗不完全一致 — "
            f"缺少 {sorted(missing) if missing else '无'}, 多余 {sorted(extra) if extra else '无'}"
        )


def count_pixels_per_window(
    best_minute: np.ndarray,
    valid: np.ndarray,
    windows: list[Fy4Window],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    n_valid = int(valid.sum())
    best_minute_utc = cst_minutes_to_utc_minutes(best_minute.astype(np.int32))
    for slot_idx, window in enumerate(windows):
        in_window = (
            valid
            & (best_minute_utc >= window.start_minute)
            & (best_minute_utc < window.end_minute)
        )
        count = int(in_window.sum())
        rows.append(
            {
                "slot_idx": slot_idx,
                "label": window.utc_label,
                "range_utc": window.range_label_utc,
                "range_cst": window.range_label_cst,
                "start_minute_utc": window.start_minute,
                "end_minute_utc": window.end_minute,
                "matched_pixels": count,
                "pct_of_valid_fy3": 100.0 * count / n_valid if n_valid else 0.0,
            }
        )
    unmatched = int((valid & (best_minute >= 0)).sum()) - sum(r["matched_pixels"] for r in rows)
    rows.append(
        {
            "slot_idx": -1,
            "label": "unmatched",
            "range_utc": "—",
            "range_cst": "—",
            "start_minute_utc": -1,
            "end_minute_utc": -1,
            "matched_pixels": unmatched,
            "pct_of_valid_fy3": 100.0 * unmatched / n_valid if n_valid else 0.0,
        }
    )
    return rows


def classify_strip_slot(
    matched: int,
    strip_valid: int,
    strip_minute_min: int,
    strip_minute_max: int,
    window: Fy4Window,
) -> str:
    if strip_valid == 0:
        return "无数据"
    if matched == 0:
        return "废掉"
    share = matched / strip_valid
    span_overlaps = not (strip_minute_max < window.start_minute or strip_minute_min >= window.end_minute)
    if share >= 0.5 and span_overlaps:
        return "充分"
    if matched > 0:
        return "部分"
    return "废掉"


def analyze_orbit_strips(
    best_minute: np.ndarray,
    valid: np.ndarray,
    slot_map: np.ndarray,
    lons: np.ndarray,
    windows: list[Fy4Window],
    strip_centers: tuple[float, ...],
    lon_half_width: float,
) -> list[dict[str, object]]:
    lons_2d = lons[None, :]
    results: list[dict[str, object]] = []
    best_minute_utc = cst_minutes_to_utc_minutes(best_minute.astype(np.int32))

    for center in strip_centers:
        lon_mask = (lons_2d >= center - lon_half_width) & (lons_2d <= center + lon_half_width)
        strip_valid = valid & lon_mask
        n_valid = int(strip_valid.sum())
        ref = STRIP_REFERENCE_RANGES.get(center)

        if n_valid == 0:
            results.append(
                {
                    "center_lon": center,
                    "n_valid": 0,
                    "time_min_cst": "NA",
                    "time_max_cst": "NA",
                    "ref_range_cst": f"{ref[0]}-{ref[1]}" if ref else "",
                    "per_slot": [],
                }
            )
            continue

        minutes = best_minute[strip_valid].astype(np.int32)
        t_min = int(minutes.min())
        t_max = int(minutes.max())
        per_slot: list[dict[str, object]] = []

        for slot_idx, window in enumerate(windows):
            in_window = (
                strip_valid
                & (best_minute_utc >= window.start_minute)
                & (best_minute_utc < window.end_minute)
            )
            matched = int(in_window.sum())
            status = classify_strip_slot(matched, n_valid, t_min, t_max, window)
            per_slot.append(
                {
                    "slot_idx": slot_idx,
                    "label": window.utc_label,
                    "matched": matched,
                    "pct": 100.0 * matched / n_valid,
                    "status": status,
                }
            )

        dominant = max(per_slot, key=lambda row: row["matched"])
        results.append(
            {
                "center_lon": center,
                "n_valid": n_valid,
                "time_min_cst": minute_label(t_min),
                "time_max_cst": minute_label(t_max),
                "time_minute_range": (t_min, t_max),
                "ref_range_cst": f"{ref[0]}-{ref[1]}" if ref else "",
                "dominant_slot": dominant["label"],
                "dominant_pct": dominant["pct"],
                "per_slot": per_slot,
            }
        )
    return results


def estimate_majority_patches(
    slot_map: np.ndarray,
    patch_size: int,
    patch_stride: int,
    subsample: int,
    majority_frac: float = 0.5,
) -> dict[str, float]:
    coarse = slot_map[::subsample, ::subsample]
    nrows, ncols = coarse.shape
    if nrows < patch_size or ncols < patch_size:
        return {"n_patch_grid": 0, "n_majority_same_slot": 0, "frac_majority": 0.0}

    coarse_patch = max(1, patch_size // subsample)
    coarse_stride = max(1, patch_stride // subsample)
    n_total = 0
    n_majority = 0

    for row0 in range(0, nrows - coarse_patch + 1, coarse_stride):
        for col0 in range(0, ncols - coarse_patch + 1, coarse_stride):
            patch = coarse[row0 : row0 + coarse_patch, col0 : col0 + coarse_patch]
            n_total += 1
            valid = patch >= 0
            n_valid = int(valid.sum())
            if n_valid == 0:
                continue
            slots = patch[valid]
            _, counts = np.unique(slots, return_counts=True)
            if counts.max() > n_valid * majority_frac:
                n_majority += 1

    full = _count_majority_patches_full(slot_map, patch_size, patch_stride)
    return {
        "subsample": float(subsample),
        "coarse_patch": float(coarse_patch),
        "coarse_stride": float(coarse_stride),
        "n_patch_grid_coarse": float(n_total),
        "n_majority_same_slot_coarse": float(n_majority),
        "frac_majority_coarse": n_majority / n_total if n_total else 0.0,
        **full,
    }


def _count_majority_patches_full(
    slot_map: np.ndarray,
    patch_size: int,
    patch_stride: int,
    majority_frac: float = 0.5,
) -> dict[str, float]:
    nrows, ncols = slot_map.shape
    n_total = 0
    n_majority = 0
    for row0 in range(0, nrows - patch_size + 1, patch_stride):
        for col0 in range(0, ncols - patch_size + 1, patch_stride):
            patch = slot_map[row0 : row0 + patch_size, col0 : col0 + patch_size]
            n_total += 1
            valid = patch >= 0
            n_valid = int(valid.sum())
            if n_valid == 0:
                continue
            _, counts = np.unique(patch[valid], return_counts=True)
            if counts.max() > n_valid * majority_frac:
                n_majority += 1
    return {
        "n_patch_grid_full": float(n_total),
        "n_majority_same_slot_full": float(n_majority),
        "frac_majority_full": n_majority / n_total if n_total else 0.0,
    }


def downsample_for_plot(data: np.ndarray, max_pixels: int) -> tuple[np.ndarray, int]:
    rows, cols = data.shape
    step = max(1, int(np.ceil(max(rows, cols) / max_pixels)))
    return data[::step, ::step], step


def plot_slot_map(
    slot_map: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    windows: list[Fy4Window],
    date: str,
    output: Path,
    plot_max_pixels: int,
) -> None:
    plot_data, plot_step = downsample_for_plot(slot_map.astype(np.float32), plot_max_pixels)
    plot_data = np.where(plot_data >= 0, plot_data, np.nan)
    extent = (float(lons[0]), float(lons[-1]), float(lats[-1]), float(lats[0]))

    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    im = ax.imshow(
        plot_data,
        origin="upper",
        extent=extent,
        cmap="tab10",
        vmin=-0.5,
        vmax=len(windows) - 0.5,
        aspect="auto",
    )
    ax.set_title(f"FY-3C → FY-4 slot map (half-open 15min UTC), {date}")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.grid(True, linewidth=0.3, alpha=0.35)
    cbar = fig.colorbar(im, ax=ax, ticks=np.arange(len(windows)), shrink=0.86)
    cbar.ax.set_yticklabels([dual_time_label(w.utc_label) for w in windows])
    cbar.set_label("FY-4 slot index")
    ax.text(
        0.01,
        0.02,
        f"plot stride={plot_step}; -1=unmatched",
        transform=ax.transAxes,
        fontsize=9,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.45, "pad": 3},
    )
    for center in ORBIT_STRIP_CENTERS_E:
        ax.axvline(center, color="white", linewidth=0.6, linestyle="--", alpha=0.7)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def print_chinese_summary(
    date: str,
    n_valid: int,
    window_counts: list[dict[str, object]],
    windows: list[Fy4Window],
    strip_results: list[dict[str, object]],
    patch_est: dict[str, float],
    patch_stride: int,
    out_npz: Path,
    out_png: Path,
) -> None:
    matched_total = sum(r["matched_pixels"] for r in window_counts if r["slot_idx"] >= 0)
    unmatched_row = next(r for r in window_counts if r["slot_idx"] < 0)

    print()
    print(f"======== FY-3 / FY-4 师兄半开窗匹配统计 ({date}) ========")
    print(f"有效 FY-3 像元 (best_minute>=0 且 best_diff 有限): {n_valid:,}")
    print(f"落入 9 个 FY-4 半开窗的像元合计: {matched_total:,} ({100*matched_total/n_valid:.2f}%)")
    print(f"未匹配任何窗: {unmatched_row['matched_pixels']:,} ({unmatched_row['pct_of_valid_fy3']:.2f}%)")
    print()
    print("各 FY-4 半开窗 [start, start+15) 像元数:")
    print(f"{'UTC':<8} {'半开窗(UTC)':<14} {'半开窗(CST)':<14} {'匹配像元':>12} {'占有效%':>10}")
    print("-" * 62)
    for row in window_counts:
        if row["slot_idx"] < 0:
            continue
        print(
            f"{row['label']:<8} {row['range_utc']:<14} {row['range_cst']:<14} "
            f"{row['matched_pixels']:>12,} {row['pct_of_valid_fy3']:>9.2f}%"
        )
    print(
        f"{'(未匹配)':<8} {'—':<14} {'—':<14} "
        f"{unmatched_row['matched_pixels']:>12,} {unmatched_row['pct_of_valid_fy3']:>9.2f}%"
    )

    print()
    print("参考时间 vs 实测 best_minute 范围 (核对师兄条带举例):")
    for strip in strip_results:
        if strip["n_valid"] == 0 or not strip["ref_range_cst"]:
            continue
        ref_parts = strip["ref_range_cst"].split("-")
        ref_lo = parse_hhmm_range_part(ref_parts[0])
        ref_hi = parse_hhmm_range_part(ref_parts[1])
        t_lo, t_hi = strip["time_minute_range"]
        overlaps = not (t_hi < ref_lo or t_lo > ref_hi)
        obs = f"{strip['time_min_cst']}-{strip['time_max_cst']}"
        note = "有重叠" if overlaps else "不重叠"
        print(f"  {strip['center_lon']:.0f}E: 参考 {strip['ref_range_cst']}, 实测 {obs} ({note})")

    print()
    print("轨道条带 (经度中心 ±1°) — 观测时间范围与各窗匹配:")
    print(f"{'条带':<6} {'有效像元':>10} {'观测范围':<14} {'参考范围':<14} {'主导时次':<8} {'主导%':>7}")
    print("-" * 72)
    for strip in strip_results:
        if strip["n_valid"] == 0:
            print(f"{strip['center_lon']:.0f}E   {'0':>10} {'—':<14} {strip['ref_range_cst']:<14} {'—':<8} {'—':>7}")
            continue
        obs = f"{strip['time_min_cst']}-{strip['time_max_cst']}"
        print(
            f"{strip['center_lon']:.0f}E   {strip['n_valid']:>10,} {obs:<14} "
            f"{strip['ref_range_cst']:<14} {strip['dominant_slot']:<8} {strip['dominant_pct']:>6.1f}%"
        )

    print()
    print("条带 × 时次 状态 (废掉/部分/充分):")
    header = "条带 " + " ".join(f"{w.utc_label:>6}" for w in windows)
    print(header)
    for strip in strip_results:
        if strip["n_valid"] == 0:
            continue
        cells = []
        for slot in strip["per_slot"]:
            status = slot["status"]
            if status == "废掉":
                cells.append("  废 ")
            elif status == "部分":
                cells.append("  部 ")
            else:
                cells.append("  充 ")
        print(f"{strip['center_lon']:.0f}E  " + "".join(cells))

    print()
    print(
        f"Patch 统计 (64×64, stride={patch_stride}, >50% 同 slot): "
        f"全幅 {int(patch_est['n_patch_grid_full']):,} 个候选, "
        f"通过 {int(patch_est['n_majority_same_slot_full']):,} "
        f"({100*patch_est['frac_majority_full']:.1f}%)"
    )
    print(
        f"  (子采样×{int(patch_est['subsample'])} 粗估: "
        f"{int(patch_est['n_majority_same_slot_coarse']):,}/"
        f"{int(patch_est['n_patch_grid_coarse']):,} = "
        f"{100*patch_est['frac_majority_coarse']:.1f}%)"
    )
    print()
    print(f"输出 npz: {out_npz}")
    print(f"输出 png: {out_png}")


def process_date(args: argparse.Namespace, date: str) -> None:
    if args.end_date and (args.timestamp_npz or args.resampled_dir or args.output_dir):
        raise ValueError("日期范围模式下不可指定 --timestamp-npz / --resampled-dir / --output-dir。")

    timestamp_npz = args.timestamp_npz or default_timestamp_npz(date, **DEFAULT_TIMESTAMP_KWARGS)
    resampled_dir = args.resampled_dir or (OUTPUT_RESAMPLED_ON_FY3_DIR / f"{date}_utc")
    out_dir = args.output_dir or (OUTPUT_MATCHING_DIR / f"{date}_utc")
    if not timestamp_npz.exists():
        if args.end_date:
            print(f"  跳过 {date}: 时间戳 npz 不存在 {timestamp_npz}")
            return
        raise FileNotFoundError(f"时间戳 npz 不存在: {timestamp_npz}")
    if not resampled_dir.is_dir():
        if args.end_date:
            print(f"  跳过 {date}: 重采样目录不存在 {resampled_dir}")
            return
        raise FileNotFoundError(f"重采样目录不存在: {resampled_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("FY-3C / FY-4A 师兄半开窗匹配:")
    print(f"  日期: {date}")
    print(f"  时间戳: {timestamp_npz}")
    print(f"  FY-4 重采样目录: {resampled_dir}")

    best_minute, best_diff, lons, lats = load_timestamp(timestamp_npz)
    valid = (best_minute >= 0) & np.isfinite(best_diff)
    n_valid = int(valid.sum())

    windows = build_fy4_windows(resampled_dir, args.window_minutes)
    verify_user_windows(windows)
    slot_starts = np.array([w.start_minute for w in windows], dtype=np.int32)

    print(f"  FY-4 时次: {len(windows)} 个")
    for w in windows:
        print(f"    FY4_{w.utc_label} → [{w.range_label_utc}) UTC / [{w.range_label_cst}) CST")

    slot_map = match_pixels_to_half_open_windows(
        best_minute,
        valid,
        slot_starts,
        window_minutes=args.window_minutes,
        best_minute_tz="CST",
    )
    window_counts = count_pixels_per_window(best_minute, valid, windows)
    strip_results = analyze_orbit_strips(
        best_minute,
        valid,
        slot_map,
        lons,
        windows,
        ORBIT_STRIP_CENTERS_E,
        args.strip_lon_half_width,
    )
    patch_est = estimate_majority_patches(
        slot_map,
        args.patch_size,
        args.patch_stride,
        args.patch_subsample,
    )

    tag = "prior1015w120_stride1" if "prior1015w120" in timestamp_npz.name else "timestamp"
    out_npz = out_dir / f"fy4_slot_map_{date}_{tag}.npz"

    out_png = out_dir / f"fy4_slot_map_{date}.png"

    np.savez_compressed(
        out_npz,
        fy4_slot_map=slot_map,
        best_minute=best_minute,
        best_diff=best_diff,
        valid_mask=valid,
        slot_labels=np.array([w.utc_label for w in windows]),
        slot_labels_cst=np.array([w.cst_label for w in windows]),
        slot_start_minutes_utc=slot_starts,
        slot_end_minutes_utc=np.array([w.end_minute for w in windows], dtype=np.int32),
        window_minutes=args.window_minutes,
        matched_per_slot=np.array([r["matched_pixels"] for r in window_counts if r["slot_idx"] >= 0]),
        date=date,
        timestamp_npz=str(timestamp_npz),
        resampled_dir=str(resampled_dir),
        matching_rule="half_open [start, start+15) UTC from filename; FY-3 best_minute CST→UTC; not nearest-neighbor",
        orbit_strip_centers_e=np.array(ORBIT_STRIP_CENTERS_E, dtype=np.float32),
        patch_majority_estimate=patch_est,
    )

    plot_slot_map(slot_map, lons, lats, windows, date, out_png, args.plot_max_pixels)

    print_chinese_summary(
        date,
        n_valid,
        window_counts,
        windows,
        strip_results,
        patch_est,
        args.patch_stride,
        out_npz,
        out_png,
    )


def main() -> None:
    args = parse_args()
    dates = list(iter_dates(args.date, args.end_date))
    for idx, date in enumerate(dates, start=1):
        if len(dates) > 1:
            print(f"\n[{idx}/{len(dates)}] {date}")
        process_date(args, date)


if __name__ == "__main__":
    main()
