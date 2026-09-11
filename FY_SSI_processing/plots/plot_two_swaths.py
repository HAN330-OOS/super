"""
Compare FY-3C Gc vs FY-4A SSI vs REST2 clear-sky GHI for each main orbital
swath strip on a given day.

Identifies swaths from fy4_slot_map (all slots with matched pixel count >=
min_swath_pixels within the region). Each swath is plotted on the full map
extent with only that swath's slot-matched pixels visible (NaN elsewhere) to
preserve stripe structure.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

FY_PROCESSING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FY_PROCESSING_ROOT))

from fy_io import (
    OUTPUT_MATCHING_DIR,
    OUTPUT_PLOTS_TWO_SWATHS_DIR,
    OUTPUT_RESAMPLED_ON_FY3_DIR,
    OUTPUT_REST2_ON_FY3_DIR,
    SSI_VALID_MAX,
    SSI_VALID_MIN,
    build_target_grid,
    clean_ssi,
    default_timestamp_npz,
    discover_fy4_slots_from_resampled,
    dual_time_label,
    iter_dates,
    read_fy3_binary,
    region_slice,
    resampled_on_fy3_path,
    resolve_fy3_gc_path,
    rest2_on_fy3_path,
)

DEFAULT_DATE = "20180501"
FULL_LON_MIN, FULL_LON_MAX = 73.0, 135.0
FULL_LAT_MIN, FULL_LAT_MAX = 18.0, 54.0
IRRAD_CMAP = "inferno"
IRRAD_VMIN = 0.0
IRRAD_VMAX = 1200.0
BRIGHT_GC_MIN = 200.0


@dataclass(frozen=True)
class SwathInfo:
    swath_id: int
    slot_idx: int
    slot_utc: str
    mask: np.ndarray
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    n_matched: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FY-3 / FY-4 comparison for each main orbital swath strip.",
        epilog=(
            "示例:\n"
            "  py -3 FY_SSI_processing\\plots\\plot_two_swaths.py --date 20180501"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--date", default=DEFAULT_DATE)
    parser.add_argument("--end-date", default=None, help="结束日期 YYYYMMDD（含）；给定则自 --date 起逐日处理。")
    parser.add_argument("--fy3-gc", type=Path, default=None)
    parser.add_argument("--timestamp-npz", type=Path, default=None)
    parser.add_argument("--resampled-dir", type=Path, default=None)
    parser.add_argument("--rest2-dir", type=Path, default=None)
    parser.add_argument("--slot-map-npz", type=Path, default=None)
    parser.add_argument("--window-minutes", type=int, default=15)
    parser.add_argument("--lon-min", type=float, default=FULL_LON_MIN)
    parser.add_argument("--lon-max", type=float, default=FULL_LON_MAX)
    parser.add_argument("--lat-min", type=float, default=FULL_LAT_MIN)
    parser.add_argument("--lat-max", type=float, default=FULL_LAT_MAX)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--plot-max-pixels", type=int, default=1800)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--min-swath-pixels", type=int, default=10_000)
    parser.add_argument("--zoom-if-lon-span-deg", type=float, default=35.0)
    return parser.parse_args()


def default_slot_map_npz(date: str, timestamp_npz: Path) -> Path:
    tag = "prior1015w120_stride1" if "prior1015w120" in timestamp_npz.name else "timestamp"
    return OUTPUT_MATCHING_DIR / f"{date}_utc" / f"fy4_slot_map_{date}_{tag}.npz"


def clean_gc(data: np.ndarray) -> np.ndarray:
    return clean_ghi(data)


def clean_ghi(data: np.ndarray) -> np.ndarray:
    clean = data.astype(np.float32, copy=True)
    clean[~np.isfinite(clean)] = np.nan
    clean[clean < SSI_VALID_MIN] = np.nan
    clean[clean > SSI_VALID_MAX] = np.nan
    return clean


def downsample_for_plot(data: np.ndarray, max_pixels: int) -> np.ndarray:
    rows, cols = data.shape
    step = max(1, int(math.ceil(max(rows, cols) / max_pixels)))
    return data[::step, ::step]


def plot_extent(lons: np.ndarray, lats: np.ndarray) -> tuple[float, float, float, float]:
    return (float(lons[0]), float(lons[-1]), float(lats[-1]), float(lats[0]))


def mask_to_swath(data: np.ndarray, swath_mask: np.ndarray) -> np.ndarray:
    out = np.full(data.shape, np.nan, dtype=np.float32)
    out[swath_mask] = data[swath_mask]
    return out


def swath_bounds(
    swath_mask: np.ndarray,
    lons_1d: np.ndarray,
    lats_1d: np.ndarray,
) -> tuple[float, float, float, float]:
    rows, cols = np.where(swath_mask)
    if rows.size == 0:
        nan = float("nan")
        return nan, nan, nan, nan
    return (
        float(lons_1d[cols.min()]),
        float(lons_1d[cols.max()]),
        float(lats_1d[rows.max()]),
        float(lats_1d[rows.min()]),
    )


def identify_swaths(
    slot_region: np.ndarray,
    slot_labels: list[str],
    lons_1d: np.ndarray,
    lats_1d: np.ndarray,
    min_pixels: int,
) -> list[SwathInfo]:
    matched = slot_region >= 0
    if not np.any(matched):
        raise ValueError("slot_map 在选定区域内无 matched 像元。")

    slot_counts: list[tuple[int, int]] = []
    for slot_idx in range(len(slot_labels)):
        count = int((slot_region == slot_idx).sum())
        if count >= min_pixels:
            slot_counts.append((count, slot_idx))
    slot_counts.sort(reverse=True)

    if not slot_counts:
        raise ValueError(
            f"未找到足够大的 slot 条带 (min={min_pixels})。"
        )

    swaths: list[SwathInfo] = []
    for swath_id, (_count, slot_idx) in enumerate(slot_counts, start=1):
        mask = slot_region == slot_idx
        lon_min, lon_max, lat_min, lat_max = swath_bounds(mask, lons_1d, lats_1d)
        swaths.append(
            SwathInfo(
                swath_id=swath_id,
                slot_idx=slot_idx,
                slot_utc=slot_labels[slot_idx],
                mask=mask,
                lon_min=lon_min,
                lon_max=lon_max,
                lat_min=lat_min,
                lat_max=lat_max,
                n_matched=int(mask.sum()),
            )
        )
    return swaths


def _nan_stats() -> dict[str, float | int]:
    nan = float("nan")
    return {
        "n_valid": 0,
        "mean_fy3": nan,
        "mean_fy4": nan,
        "bias": nan,
        "relative_bias_pct": nan,
        "rmse": nan,
        "mae": nan,
        "r": nan,
        "r2": nan,
        "ratio_mean": nan,
        "ratio_p50": nan,
        "ratio_p5": nan,
        "ratio_p95": nan,
    }


def compute_fy3_fy4_stats(
    fy4: np.ndarray,
    fy3: np.ndarray,
    swath_mask: np.ndarray,
) -> dict[str, float | int]:
    """Stats on pixels where both FY-3 Gc and FY-4 SSI are valid."""
    valid = (
        swath_mask
        & np.isfinite(fy4)
        & np.isfinite(fy3)
        & (fy3 > 0.0)
        & (fy4 > 0.0)
    )
    n_valid = int(valid.sum())
    if n_valid == 0:
        return _nan_stats()

    fy4_vals = fy4[valid].astype(np.float64)
    fy3_vals = fy3[valid].astype(np.float64)
    diff = fy4_vals - fy3_vals
    rel = diff / fy3_vals
    ratios = fy4_vals / fy3_vals

    if np.std(fy4_vals) > 0 and np.std(fy3_vals) > 0:
        r = float(np.corrcoef(fy4_vals, fy3_vals)[0, 1])
    else:
        r = float("nan")
    r2 = float(r * r) if np.isfinite(r) else float("nan")

    return {
        "n_valid": n_valid,
        "mean_fy3": float(np.mean(fy3_vals)),
        "mean_fy4": float(np.mean(fy4_vals)),
        "bias": float(np.mean(diff)),
        "relative_bias_pct": float(np.mean(rel) * 100.0),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "mae": float(np.mean(np.abs(diff))),
        "r": r,
        "r2": r2,
        "ratio_mean": float(np.mean(ratios)),
        "ratio_p50": float(np.median(ratios)),
        "ratio_p5": float(np.percentile(ratios, 5)),
        "ratio_p95": float(np.percentile(ratios, 95)),
    }


def compare_pair_stats(
    x: np.ndarray,
    y: np.ndarray,
    swath_mask: np.ndarray,
    bright_ref: np.ndarray | None = None,
    bright_min: float = BRIGHT_GC_MIN,
) -> dict[str, float | int]:
    valid = swath_mask & np.isfinite(x) & np.isfinite(y) & (y > 0.0) & (x > 0.0)
    ref = y if bright_ref is None else bright_ref
    bright = valid & np.isfinite(ref) & (ref >= bright_min)
    n_valid = int(valid.sum())
    n_bright = int(bright.sum())

    def _stats(mask: np.ndarray) -> tuple[float, float, float, float]:
        n = int(mask.sum())
        if n == 0:
            return float("nan"), float("nan"), float("nan"), float("nan")
        x_vals = x[mask].astype(np.float64)
        y_vals = y[mask].astype(np.float64)
        bias = float(np.mean(x_vals - y_vals))
        ratios = x_vals / y_vals
        median_ratio = float(np.median(ratios))
        mean_ratio = float(np.mean(ratios))
        if np.std(x_vals) > 0 and np.std(y_vals) > 0:
            r = float(np.corrcoef(x_vals, y_vals)[0, 1])
        else:
            r = float("nan")
        return r, median_ratio, mean_ratio, bias

    r, median_ratio, mean_ratio, bias = _stats(valid)
    r_bright, median_ratio_bright, mean_ratio_bright, bias_bright = _stats(bright)

    return {
        "n_valid": n_valid,
        "n_bright": n_bright,
        "r": r,
        "median_ratio": median_ratio,
        "mean_ratio": mean_ratio,
        "bias": bias,
        "r_bright": r_bright,
        "median_ratio_bright": median_ratio_bright,
        "mean_ratio_bright": mean_ratio_bright,
        "bias_bright": bias_bright,
    }


def compare_stats(
    fy4: np.ndarray,
    fy3: np.ndarray,
    swath_mask: np.ndarray,
    bright_min: float = BRIGHT_GC_MIN,
) -> dict[str, float | int | str]:
    full = compute_fy3_fy4_stats(fy4, fy3, swath_mask)
    pair = compare_pair_stats(fy4, fy3, swath_mask, bright_ref=fy3, bright_min=bright_min)
    return {
        "n_matched": int(swath_mask.sum()),
        **full,
        "median_ratio": full["ratio_p50"],
        "mean_ratio": full["ratio_mean"],
        "n_bright": pair["n_bright"],
        "r_bright": pair["r_bright"],
        "median_ratio_bright": pair["median_ratio_bright"],
        "mean_ratio_bright": pair["mean_ratio_bright"],
        "bias_bright": pair["bias_bright"],
    }


def plot_triple_panel(
    panels: list[tuple[np.ndarray, str, str]],
    lons: np.ndarray,
    lats: np.ndarray,
    title: str,
    output: Path,
    extent: tuple[float, float, float, float],
    plot_max_pixels: int,
    dpi: int,
    vmin: float = IRRAD_VMIN,
    vmax: float = IRRAD_VMAX,
    cmap: str = IRRAD_CMAP,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0), constrained_layout=True)
    for axis, (data, panel_title, cbar_label) in zip(axes, panels):
        plot_data = downsample_for_plot(data, plot_max_pixels)
        im = axis.imshow(
            plot_data,
            origin="upper",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )
        finite = data[np.isfinite(data)]
        valid_frac = float(finite.size / data.size) if data.size else 0.0
        mean_val = float(np.mean(finite)) if finite.size else float("nan")
        axis.set_title(
            f"{panel_title}\nvalid={100 * valid_frac:.2f}%  mean={mean_val:.0f}",
            fontsize=9,
        )
        axis.set_xlabel("Lon (°E)")
        axis.set_ylabel("Lat (°N)")
        axis.grid(True, linewidth=0.25, alpha=0.3)
        fig.colorbar(im, ax=axis, shrink=0.9, pad=0.02, label=cbar_label)
    fig.suptitle(title, fontsize=11)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def zoom_extent(
    swath: SwathInfo,
    pad_deg: float = 2.0,
    full_extent: tuple[float, float, float, float] | None = None,
) -> tuple[float, float, float, float]:
    lon_min = swath.lon_min - pad_deg
    lon_max = swath.lon_max + pad_deg
    lat_min = swath.lat_min - pad_deg
    lat_max = swath.lat_max + pad_deg
    if full_extent is not None:
        flon_min, flon_max, flat_min, flat_max = full_extent
        lon_min = max(lon_min, flon_min)
        lon_max = min(lon_max, flon_max)
        lat_min = max(lat_min, flat_min)
        lat_max = min(lat_max, flat_max)
    return lon_min, lon_max, lat_min, lat_max


def plot_swath_triple(
    swath: SwathInfo,
    fy3: np.ndarray,
    fy4: np.ndarray,
    rest2: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    date: str,
    output_dir: Path,
    plot_max_pixels: int,
    dpi: int,
    zoom_lon_span_threshold: float,
) -> None:
    prefix = f"swath{swath.swath_id}"
    slot_label = dual_time_label(swath.slot_utc)
    subtitle = (
        f"{slot_label}  slot={swath.slot_idx}  "
        f"lon {swath.lon_min:.1f}–{swath.lon_max:.1f}°E"
    )

    full_extent = plot_extent(lons, lats)
    lon_span = swath.lon_max - swath.lon_min
    use_zoom = lon_span < zoom_lon_span_threshold
    extent = zoom_extent(swath, full_extent=full_extent) if use_zoom else full_extent
    zoom_tag = "_zoom" if use_zoom else ""
    date_tag = f"_{date}"

    plot_triple_panel(
        [
            (fy3, "FY-3C Gc", "Gc (W/m²)"),
            (fy4, "FY-4A SSI", "SSI (W/m²)"),
            (rest2, "REST2 clear-sky GHI", "GHI (W/m²)"),
        ],
        lons,
        lats,
        f"FY-3 / FY-4 / REST2 — swath {swath.swath_id}\n{subtitle} — {date}",
        output_dir / f"{prefix}_triple{zoom_tag}{date_tag}.png",
        extent,
        plot_max_pixels,
        dpi,
    )


def process_date(args: argparse.Namespace, date: str) -> None:
    if args.end_date and (
        args.fy3_gc or args.timestamp_npz or args.resampled_dir or args.rest2_dir or args.slot_map_npz or args.output_dir
    ):
        raise ValueError(
            "日期范围模式下不可指定 --fy3-gc / --timestamp-npz / --resampled-dir / "
            "--rest2-dir / --slot-map-npz / --output-dir。"
        )

    timestamp_npz = args.timestamp_npz or default_timestamp_npz(date)
    resampled_dir = args.resampled_dir or (OUTPUT_RESAMPLED_ON_FY3_DIR / f"{date}_utc")
    rest2_dir = args.rest2_dir or (OUTPUT_REST2_ON_FY3_DIR / f"{date}_utc")
    slot_map_npz = args.slot_map_npz or default_slot_map_npz(date, timestamp_npz)
    output_dir = args.output_dir or (OUTPUT_PLOTS_TWO_SWATHS_DIR / f"{date}_utc")
    gc_path = args.fy3_gc or resolve_fy3_gc_path(date)

    region_label = (
        f"region {args.lon_min:.0f}–{args.lon_max:.0f}°E, "
        f"{args.lat_min:.0f}–{args.lat_max:.0f}°N"
    )

    print(f"Swath FY-3 / FY-4 / REST2 comparison — {date}")
    print(f"  Gc: {gc_path}")
    print(f"  slot_map: {slot_map_npz}")
    print(f"  resampled: {resampled_dir}")
    print(f"  REST2: {rest2_dir}")
    print(f"  {region_label}")
    print(f"  output: {output_dir}")

    if not slot_map_npz.exists():
        if args.end_date:
            print(f"  跳过 {date}: slot_map npz 不存在 {slot_map_npz}")
            return
        raise FileNotFoundError(f"slot_map npz 不存在: {slot_map_npz}")
    if not gc_path.exists():
        if args.end_date:
            print(f"  跳过 {date}: FY-3 Gc 不存在 {gc_path}")
            return
        raise FileNotFoundError(f"FY-3 Gc 不存在: {gc_path}")
    if not rest2_dir.exists():
        if args.end_date:
            print(f"  跳过 {date}: REST2 目录不存在 {rest2_dir}")
            return
        raise FileNotFoundError(f"REST2 目录不存在: {rest2_dir}")

    slot_entries = discover_fy4_slots_from_resampled(resampled_dir, args.window_minutes)
    slot_labels = [entry[0] for entry in slot_entries]

    with np.load(slot_map_npz) as slot_data:
        slot_map_full = slot_data["fy4_slot_map"].astype(np.int16, copy=False)

    row_start, row_end, col_start, col_end = region_slice(
        args.lon_min, args.lon_max, args.lat_min, args.lat_max
    )
    lons_1d, lats_1d, _, _ = build_target_grid(
        args.lon_min, args.lon_max, args.lat_min, args.lat_max
    )

    gc_full = clean_gc(read_fy3_binary(gc_path))
    gc_region = gc_full[row_start:row_end, col_start:col_end]
    slot_region = slot_map_full[row_start:row_end, col_start:col_end]

    swaths = identify_swaths(
        slot_region, slot_labels, lons_1d, lats_1d, args.min_swath_pixels
    )
    print(f"  found {len(swaths)} swath(s)")

    resampled_by_slot: dict[int, np.ndarray] = {}
    rest2_by_slot: dict[int, np.ndarray] = {}
    for slot_idx, (utc_label, _start, _end, _cst) in enumerate(slot_entries):
        path = resampled_on_fy3_path(resampled_dir, utc_label)
        with np.load(path) as data:
            layer = clean_ssi(data["ssi"]).astype(np.float32, copy=False)
        resampled_by_slot[slot_idx] = layer[row_start:row_end, col_start:col_end]

        rest2_path = rest2_on_fy3_path(rest2_dir, utc_label)
        if not rest2_path.exists():
            raise FileNotFoundError(f"REST2 npz 不存在: {rest2_path}")
        with np.load(rest2_path) as data:
            rest2_layer = clean_ghi(data["ghi"]).astype(np.float32, copy=False)
        rest2_by_slot[slot_idx] = rest2_layer[row_start:row_end, col_start:col_end]

    stats_rows: list[dict[str, object]] = []
    for swath in swaths:
        fy3 = mask_to_swath(gc_region, swath.mask)
        fy4 = mask_to_swath(resampled_by_slot[swath.slot_idx], swath.mask)
        rest2 = mask_to_swath(rest2_by_slot[swath.slot_idx], swath.mask)
        stats = compare_stats(fy4, fy3, swath.mask)
        stats_fy4_rest2 = compare_pair_stats(
            fy4, rest2, swath.mask, bright_ref=rest2
        )
        stats_fy3_rest2 = compare_pair_stats(
            fy3, rest2, swath.mask, bright_ref=rest2
        )
        stats_rows.append(
            {
                "swath_id": swath.swath_id,
                "slot_idx": swath.slot_idx,
                "slot_utc": swath.slot_utc,
                "slot_label_cst": dual_time_label(swath.slot_utc),
                "lon_min": round(swath.lon_min, 2),
                "lon_max": round(swath.lon_max, 2),
                "lat_min": round(swath.lat_min, 2),
                "lat_max": round(swath.lat_max, 2),
                "n_matched": stats["n_matched"],
                "n_valid": stats["n_valid"],
                "mean_fy3": stats["mean_fy3"],
                "mean_fy4": stats["mean_fy4"],
                "bias": stats["bias"],
                "relative_bias_pct": stats["relative_bias_pct"],
                "rmse": stats["rmse"],
                "mae": stats["mae"],
                "r": stats["r"],
                "r2": stats["r2"],
                "ratio_mean": stats["ratio_mean"],
                "ratio_p50": stats["ratio_p50"],
                "ratio_p5": stats["ratio_p5"],
                "ratio_p95": stats["ratio_p95"],
                "n_bright": stats["n_bright"],
                "median_ratio": stats["median_ratio"],
                "r_bright": stats["r_bright"],
                "median_ratio_bright": stats["median_ratio_bright"],
                "bias_bright": stats["bias_bright"],
                "r_fy4_rest2": stats_fy4_rest2["r"],
                "median_csi": stats_fy4_rest2["median_ratio"],
                "mean_csi": stats_fy4_rest2["mean_ratio"],
                "bias_fy4_rest2": stats_fy4_rest2["bias"],
                "csi_mean_bright": stats_fy4_rest2["mean_ratio_bright"],
                "csi_median_bright": stats_fy4_rest2["median_ratio_bright"],
                "bias_fy4_rest2_bright": stats_fy4_rest2["bias_bright"],
                "r_fy3_rest2": stats_fy3_rest2["r"],
                "median_ratio_fy3_rest2": stats_fy3_rest2["median_ratio"],
                "bias_fy3_rest2": stats_fy3_rest2["bias"],
                "median_ratio_fy3_rest2_bright": stats_fy3_rest2["median_ratio_bright"],
                "bias_fy3_rest2_bright": stats_fy3_rest2["bias_bright"],
            }
        )
        print(
            f"  swath {swath.swath_id}: slot {swath.slot_idx} {dual_time_label(swath.slot_utc)} "
            f"lon {swath.lon_min:.1f}–{swath.lon_max:.1f}°E  n={stats['n_valid']:,}  "
            f"Bias={stats['bias']:.1f}  Rel.Bias={stats['relative_bias_pct']:.1f}%  "
            f"RMSE={stats['rmse']:.1f}  R2={stats['r2']:.3f}  Ratio P50={stats['ratio_p50']:.3f}  "
            f"FY4/REST2 r={stats_fy4_rest2['r']:.3f}  CSI_med={stats_fy4_rest2['median_ratio']:.3f}  "
            f"bias={stats_fy4_rest2['bias']:.1f}  CSI_bright={stats_fy4_rest2['mean_ratio_bright']:.3f}  "
            f"FY3/REST2 r={stats_fy3_rest2['r']:.3f}  med.ratio={stats_fy3_rest2['median_ratio']:.3f}  "
            f"bias={stats_fy3_rest2['bias']:.1f}"
        )
        plot_swath_triple(
            swath,
            fy3,
            fy4,
            rest2,
            lons_1d,
            lats_1d,
            date,
            output_dir,
            args.plot_max_pixels,
            args.dpi,
            args.zoom_if_lon_span_deg,
        )

    stats_path = output_dir / f"swath_stats_{date}.csv"
    fieldnames = [
        "swath_id",
        "slot_idx",
        "slot_utc",
        "slot_label_cst",
        "lon_min",
        "lon_max",
        "lat_min",
        "lat_max",
        "n_matched",
        "n_valid",
        "mean_fy3",
        "mean_fy4",
        "bias",
        "relative_bias_pct",
        "rmse",
        "mae",
        "r",
        "r2",
        "ratio_mean",
        "ratio_p50",
        "ratio_p5",
        "ratio_p95",
        "n_bright",
        "median_ratio",
        "r_bright",
        "median_ratio_bright",
        "bias_bright",
        "r_fy4_rest2",
        "median_csi",
        "mean_csi",
        "bias_fy4_rest2",
        "csi_mean_bright",
        "csi_median_bright",
        "bias_fy4_rest2_bright",
        "r_fy3_rest2",
        "median_ratio_fy3_rest2",
        "bias_fy3_rest2",
        "median_ratio_fy3_rest2_bright",
        "bias_fy3_rest2_bright",
    ]
    with stats_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats_rows)

    # A rerun can find fewer swaths than an earlier run. Remove only obsolete
    # generated plots for this same date after all replacement plots and the
    # statistics CSV have been written successfully.
    expected_plot_names: set[str] = set()
    for swath in swaths:
        lon_span = swath.lon_max - swath.lon_min
        zoom_tag = "_zoom" if lon_span < args.zoom_if_lon_span_deg else ""
        expected_plot_names.add(f"swath{swath.swath_id}_triple{zoom_tag}_{date}.png")
    for old_plot in output_dir.glob(f"swath*_triple*_{date}.png"):
        if old_plot.name not in expected_plot_names:
            old_plot.unlink()
            print(f"  removed stale plot: {old_plot}")

    print()
    print("完成:")
    for swath in swaths:
        prefix = f"swath{swath.swath_id}"
        lon_span = swath.lon_max - swath.lon_min
        zoom_tag = "_zoom" if lon_span < args.zoom_if_lon_span_deg else ""
        date_tag = f"_{date}"
        print(f"  {output_dir / f'{prefix}_triple{zoom_tag}{date_tag}.png'}")
    print(f"  stats: {stats_path}")


def main() -> None:
    args = parse_args()
    dates = list(iter_dates(args.date, args.end_date))
    for idx, date in enumerate(dates, start=1):
        if len(dates) > 1:
            print(f"\n[{idx}/{len(dates)}] {date}")
        process_date(args, date)


if __name__ == "__main__":
    main()
