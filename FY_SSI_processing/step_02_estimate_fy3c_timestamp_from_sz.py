"""
Estimate FY-3C observation timestamps from the Sz solar zenith angle product.

Assumption:
The search window is defined in Beijing clock time (CST, UTC+8), e.g. 08:00-12:00.
For each candidate minute, the solar zenith angle is still computed with each
pixel's longitude, so the solar geometry uses local true solar time internally.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from fy_io import DEFAULT_FY3_ROOT, FY3_LAT_START, FY3_LON_START, FY3_SHAPE, FY3_STEP, PROJECT_ROOT, iter_dates

OUTPUT_TIMESTAMP_DIR = PROJECT_ROOT / "outputs" / "fy" / "timestamp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用 FY-3C Sz 太阳天顶角产品逐像元估计观测时间戳。",
        epilog=(
            "示例:\n"
            "  py -3 FY_SSI_processing\\step_02_estimate_fy3c_timestamp_from_sz.py --date 20180501 --stride 4\n"
            "  py -3 FY_SSI_processing\\step_02_estimate_fy3c_timestamp_from_sz.py --date 20180501 --stride 1\n\n"
            "时间假设: 候选分钟为北京时间/CST(UTC+8)，太阳高度计算包含经度修正的真太阳时。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--date", default="20180501", help="日期 YYYYMMDD。")
    parser.add_argument("--end-date", default=None, help="结束日期 YYYYMMDD（含）；给定则自 --date 起逐日处理。")
    parser.add_argument("--sz", type=Path, default=None, help="FY-3C Sz .dat 路径。")
    parser.add_argument("--start", default="08:00", help="搜索开始时间，CST HH:MM。")
    parser.add_argument("--end", default="12:00", help="搜索结束时间，CST HH:MM，包含该分钟。")
    parser.add_argument("--timezone-hours", type=float, default=8.0, help="候选时间的时区，默认 CST=UTC+8。")
    parser.add_argument(
        "--use-local-time-prior",
        action="store_true",
        help="按经度约束候选时间到 FY-3C 上午过境地方太阳时附近；默认关闭以保留全局搜索。",
    )
    parser.add_argument("--local-overpass-time", default="10:15", help="地方太阳时过境先验中心，HH:MM。")
    parser.add_argument("--prior-window-minutes", type=int, default=60, help="地方时先验半窗宽，单位分钟。")
    parser.add_argument("--stride", type=int, default=1, help="空间下采样步长；1 为全分辨率。")
    parser.add_argument("--block-rows", type=int, default=256, help="逐块处理的行数。")
    parser.add_argument(
        "--smooth-method",
        choices=["none", "median", "gaussian"],
        default="none",
        help="时间场后处理平滑方式：none 不平滑(默认，保持旧行为)、median 中值、gaussian 高斯。",
    )
    parser.add_argument(
        "--smooth-size",
        type=float,
        default=5.0,
        help="平滑尺度(像元)：median 为窗口边长(取最近奇数)，gaussian 为 sigma。",
    )
    parser.add_argument(
        "--max-error-deg",
        type=float,
        default=None,
        help="质控阈值(度)：拟合残差>该值的像元判为低质量；不给则不按残差剔除。",
    )
    parser.add_argument(
        "--fill-from-neighbors",
        action="store_true",
        help="将低质量/未拟合但 Sz 有效的像元用邻域平滑值填补；否则置 NaN。",
    )
    parser.add_argument("--output-npz", type=Path, default=None)
    parser.add_argument("--output-fig", type=Path, default=None)
    parser.add_argument("--output-error-fig", type=Path, default=None)
    parser.add_argument("--plot-max-pixels", type=int, default=1600, help="绘图最大边长，避免 PNG 过大。")
    parser.add_argument("--sample-count", type=int, default=10, help="控制台打印的随机有效像元数。")
    parser.add_argument("--seed", type=int, default=20260501)
    return parser.parse_args()


def default_sz_path(date: str) -> Path:
    return DEFAULT_FY3_ROOT / f"Sz_{date}.dat"


def resolve_sz_path(date: str, path: Path | None) -> Path:
    if path is not None:
        return path

    candidates = [
        DEFAULT_FY3_ROOT / f"Sz_{date}.dat",
        DEFAULT_FY3_ROOT / f"SZ_{date}.dat",
        DEFAULT_FY3_ROOT / f"sz_{date}.dat",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def parse_hhmm(value: str) -> int:
    try:
        hour_text, minute_text = value.split(":", maxsplit=1)
        hour = int(hour_text)
        minute = int(minute_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"时间应为 HH:MM，实际 {value!r}。") from exc
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise argparse.ArgumentTypeError(f"时间超出范围: {value!r}。")
    return hour * 60 + minute


def minute_label(minute_of_day: int | float) -> str:
    if not np.isfinite(minute_of_day):
        return "NA"
    minute = int(round(float(minute_of_day)))
    return f"{minute // 60:02d}:{minute % 60:02d}"


def minute_filename_label(minute_of_day: int | float) -> str:
    if not np.isfinite(minute_of_day):
        return "NA"
    minute = int(round(float(minute_of_day)))
    return f"{minute // 60:02d}{minute % 60:02d}"


def date_to_doy(date: str) -> int:
    return datetime.strptime(date, "%Y%m%d").timetuple().tm_yday


def build_local_time_prior(
    lons: np.ndarray,
    start_minute: int,
    end_minute: int,
    local_overpass_minute: int,
    prior_window_minutes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center_minutes = local_overpass_minute + (120.0 - lons.astype(np.float64)) * 4.0
    lower = np.ceil(np.maximum(start_minute, center_minutes - prior_window_minutes)).astype(np.int16)
    upper = np.floor(np.minimum(end_minute, center_minutes + prior_window_minutes)).astype(np.int16)
    return center_minutes.astype(np.float32), lower, upper

def read_sz_product(path: Path, stride: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"FY-3C Sz 文件不存在: {path}")
    expected_bytes = int(np.prod(FY3_SHAPE)) * np.dtype("<f4").itemsize
    actual_bytes = path.stat().st_size
    if actual_bytes < expected_bytes:
        raise ValueError(f"{path} 文件过小，期望至少 {expected_bytes} bytes，实际 {actual_bytes}。")

    data = np.memmap(path, dtype="<f4", mode="r", shape=FY3_SHAPE)
    if stride > 1:
        return np.asarray(data[::stride, ::stride], dtype=np.float32)
    return np.asarray(data, dtype=np.float32)

def build_sampled_grid(stride: int, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = shape
    row_indices = np.arange(rows, dtype=np.float64) * stride
    col_indices = np.arange(cols, dtype=np.float64) * stride
    lats = FY3_LAT_START - row_indices * FY3_STEP
    lons = FY3_LON_START + col_indices * FY3_STEP
    return lons, lats

def clean_sz(sz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    clean = sz.astype(np.float32, copy=True)
    clean[np.abs(clean) > 1.0e20] = np.nan
    # FY-3C products use exact zero in missing/fill blocks. Treating those
    # pixels as a physical 0-degree zenith angle creates plausible-looking but
    # spurious timestamps, especially near the sub-solar latitude.
    valid = np.isfinite(clean) & (clean > 0.0) & (clean <= 180.0)
    clean[~valid] = np.nan
    return clean, valid


def apply_fit_quality_mask(
    best_minute: np.ndarray,
    best_diff: np.ndarray,
    max_error_deg: float | None,
) -> np.ndarray:
    """Invalidate timestamps that do not reproduce the observed zenith angle."""
    fitted = (best_minute >= 0) & np.isfinite(best_diff)
    if max_error_deg is None:
        return fitted

    quality_mask = fitted & (best_diff <= max_error_deg)
    best_minute[~quality_mask] = -1
    return quality_mask

def solar_zenith_deg(
    doy: int,
    minute_of_day: int,
    lons: np.ndarray,
    sin_lats: np.ndarray,
    cos_lats: np.ndarray,
    timezone_hours: float,
) -> np.ndarray:
    """NOAA-style solar zenith angle in degrees for a latitude block and all longitudes."""
    fractional_hour = minute_of_day / 60.0
    gamma = 2.0 * np.pi / 365.0 * (doy - 1 + (fractional_hour - 12.0) / 24.0)

    equation_of_time = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2.0 * gamma)
        - 0.040849 * np.sin(2.0 * gamma)
    )
    declination = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2.0 * gamma)
        + 0.000907 * np.sin(2.0 * gamma)
        - 0.002697 * np.cos(3.0 * gamma)
        + 0.00148 * np.sin(3.0 * gamma)
    )

    true_solar_time = minute_of_day + equation_of_time + 4.0 * lons - 60.0 * timezone_hours
    hour_angle = np.deg2rad(true_solar_time / 4.0 - 180.0)

    cos_zenith = (
        sin_lats[:, None] * np.sin(declination)
        + cos_lats[:, None] * np.cos(declination) * np.cos(hour_angle)[None, :]
    )
    return np.rad2deg(np.arccos(np.clip(cos_zenith, -1.0, 1.0))).astype(np.float32)

def estimate_timestamp(
    sz: np.ndarray,
    valid: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    date: str,
    candidate_minutes: np.ndarray,
    timezone_hours: float,
    block_rows: int,
    prior_min_by_lon: np.ndarray | None = None,
    prior_max_by_lon: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    best_diff = np.full(sz.shape, np.inf, dtype=np.float32)
    best_minute = np.full(sz.shape, -1, dtype=np.int16)
    doy = date_to_doy(date)
    lat_radians = np.deg2rad(lats)
    sin_lats_all = np.sin(lat_radians)
    cos_lats_all = np.cos(lat_radians)

    total_blocks = int(np.ceil(sz.shape[0] / block_rows))
    for block_index, row_start in enumerate(range(0, sz.shape[0], block_rows), start=1):
        row_end = min(row_start + block_rows, sz.shape[0])
        sz_block = sz[row_start:row_end]
        valid_block = valid[row_start:row_end]
        if not np.any(valid_block):
            continue

        best_diff_block = best_diff[row_start:row_end]
        best_minute_block = best_minute[row_start:row_end]
        sin_lats = sin_lats_all[row_start:row_end]
        cos_lats = cos_lats_all[row_start:row_end]

        for minute in candidate_minutes:
            if prior_min_by_lon is None:
                allowed_cols = None
                active_cols = slice(None)
            else:
                allowed_cols = (minute >= prior_min_by_lon) & (minute <= prior_max_by_lon)
                if not np.any(allowed_cols):
                    continue
                allowed_indices = np.flatnonzero(allowed_cols)
                active_cols = slice(int(allowed_indices[0]), int(allowed_indices[-1]) + 1)

            sza = solar_zenith_deg(
                doy,
                int(minute),
                lons[active_cols],
                sin_lats,
                cos_lats,
                timezone_hours,
            )
            sz_active = sz_block[:, active_cols]
            valid_active = valid_block[:, active_cols]
            best_diff_active = best_diff_block[:, active_cols]
            best_minute_active = best_minute_block[:, active_cols]
            diff = np.abs(sza - sz_active)
            update = valid_active & (diff < best_diff_active)
            if allowed_cols is not None:
                update &= allowed_cols[active_cols][None, :]
            best_diff_active[update] = diff[update]
            best_minute_active[update] = int(minute)

        print(
            f"  block {block_index}/{total_blocks}: rows {row_start}-{row_end - 1}, "
            f"valid={int(valid_block.sum())}"
        )

    best_diff[~valid] = np.nan
    return best_minute, best_diff

def downsample_for_plot(data: np.ndarray, max_pixels: int) -> tuple[np.ndarray, int]:
    if max_pixels <= 0:
        return data, 1
    step = max(1, int(np.ceil(max(data.shape) / max_pixels)))
    return data[::step, ::step], step

def plot_timestamp(
    best_minute: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    start_minute: int,
    end_minute: int,
    date: str,
    output: Path,
    plot_max_pixels: int,
    stride: int,
    run_label: str | None = None,
) -> None:
    plot_data, plot_step = downsample_for_plot(best_minute.astype(np.float32), plot_max_pixels)
    plot_data = np.where(plot_data >= 0, plot_data, np.nan)
    extent = (float(lons[0]), float(lons[-1]), float(lats[-1]), float(lats[0]))
    ticks = np.arange(start_minute, end_minute + 1, 30)

    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#bdbdbd")
    im = ax.imshow(
        plot_data,
        origin="upper",
        extent=extent,
        cmap=cmap,
        vmin=start_minute,
        vmax=end_minute,
        aspect="auto",
    )
    title_suffix = f"\n{run_label}" if run_label else ""
    ax.set_title(f"FY-3C estimated timestamp from Sz, {date} CST{title_suffix}")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.grid(True, linewidth=0.3, alpha=0.35)
    cbar = fig.colorbar(im, ax=ax, ticks=ticks, shrink=0.86)
    cbar.ax.set_yticklabels([minute_label(tick) for tick in ticks])
    cbar.set_label("Estimated time (CST)")
    ax.text(
        0.01,
        0.02,
        f"data stride={stride}, plot stride={plot_step}",
        transform=ax.transAxes,
        fontsize=9,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.45, "pad": 3},
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Description": run_label} if run_label else None
    fig.savefig(output, dpi=170, metadata=metadata)
    plt.close(fig)

def plot_error(
    best_diff: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    date: str,
    output: Path,
    plot_max_pixels: int,
    stride: int,
    run_label: str | None = None,
) -> None:
    plot_data, plot_step = downsample_for_plot(best_diff, plot_max_pixels)
    finite = plot_data[np.isfinite(plot_data)]
    vmax = float(np.nanpercentile(finite, 98.0)) if finite.size else 1.0
    vmax = max(vmax, 1.0)
    extent = (float(lons[0]), float(lons[-1]), float(lats[-1]), float(lats[0]))

    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color="#bdbdbd")
    im = ax.imshow(
        plot_data,
        origin="upper",
        extent=extent,
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        aspect="auto",
    )
    title_suffix = f"\n{run_label}" if run_label else ""
    ax.set_title(f"FY-3C Sz timestamp fit error, {date}{title_suffix}")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.grid(True, linewidth=0.3, alpha=0.35)
    cbar = fig.colorbar(im, ax=ax, shrink=0.86)
    cbar.set_label("Minimum |calculated SZA - FY-3C Sz| (deg)")
    ax.text(
        0.99,
        0.02,
        "gray = invalid/fill",
        transform=ax.transAxes,
        ha="right",
        fontsize=9,
        color="black",
        bbox={"facecolor": "#bdbdbd", "alpha": 0.8, "pad": 3},
    )
    ax.text(
        0.01,
        0.02,
        f"data stride={stride}, plot stride={plot_step}",
        transform=ax.transAxes,
        fontsize=9,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.45, "pad": 3},
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Description": run_label} if run_label else None
    fig.savefig(output, dpi=170, metadata=metadata)
    plt.close(fig)

def describe_results(
    sz: np.ndarray,
    valid: np.ndarray,
    best_minute: np.ndarray,
    best_diff: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    start_minute: int,
    end_minute: int,
    sample_count: int,
    seed: int,
    prior_min_by_lon: np.ndarray | None = None,
    prior_max_by_lon: np.ndarray | None = None,
) -> None:
    fitted = valid & (best_minute >= 0) & np.isfinite(best_diff)
    print(f"有效 Sz 像元: {int(valid.sum())}/{valid.size} ({100.0 * valid.sum() / valid.size:.2f}%)")
    print(f"完成匹配像元: {int(fitted.sum())}/{int(valid.sum())}")
    if not np.any(fitted):
        return

    minutes = best_minute[fitted].astype(np.int32)
    diffs = best_diff[fitted]
    percentiles = np.nanpercentile(diffs, [0.0, 5.0, 50.0, 95.0, 99.0, 100.0])
    print(
        "估计时间范围: "
        f"{minute_label(np.nanmin(minutes))}-{minute_label(np.nanmax(minutes))} CST"
    )
    print(
        "最小 SZA 差值(deg): "
        f"min={percentiles[0]:.3f}, p5={percentiles[1]:.3f}, "
        f"median={percentiles[2]:.3f}, p95={percentiles[3]:.3f}, "
        f"p99={percentiles[4]:.3f}, max={percentiles[5]:.3f}"
    )
    start_edge = np.count_nonzero(fitted & (best_minute == start_minute))
    end_edge = np.count_nonzero(fitted & (best_minute == end_minute))
    near_start = np.count_nonzero(fitted & (best_minute <= start_minute + 5))
    near_end = np.count_nonzero(fitted & (best_minute >= end_minute - 5))
    fitted_count = int(fitted.sum())
    print(
        "贴边统计: "
        f"exact {minute_label(start_minute)}={start_edge} ({100 * start_edge / fitted_count:.2f}%), "
        f"exact {minute_label(end_minute)}={end_edge} ({100 * end_edge / fitted_count:.2f}%), "
        f"within 5 min start={near_start} ({100 * near_start / fitted_count:.2f}%), "
        f"within 5 min end={near_end} ({100 * near_end / fitted_count:.2f}%)"
    )
    if prior_min_by_lon is not None and prior_max_by_lon is not None:
        prior_min_grid = prior_min_by_lon[None, :]
        prior_max_grid = prior_max_by_lon[None, :]
        prior_lower_edge = np.count_nonzero(fitted & (best_minute == prior_min_grid))
        prior_upper_edge = np.count_nonzero(fitted & (best_minute == prior_max_grid))
        prior_near_lower = np.count_nonzero(fitted & (best_minute <= prior_min_grid + 5))
        prior_near_upper = np.count_nonzero(fitted & (best_minute >= prior_max_grid - 5))
        print(
            "地方时先验贴边统计: "
            f"lower={prior_lower_edge} ({100 * prior_lower_edge / fitted_count:.2f}%), "
            f"upper={prior_upper_edge} ({100 * prior_upper_edge / fitted_count:.2f}%), "
            f"within 5 min lower={prior_near_lower} ({100 * prior_near_lower / fitted_count:.2f}%), "
            f"within 5 min upper={prior_near_upper} ({100 * prior_near_upper / fitted_count:.2f}%)"
        )

    rng = np.random.default_rng(seed)
    rows, cols = np.where(fitted)
    take = min(sample_count, rows.size)
    chosen = rng.choice(rows.size, size=take, replace=False)
    print(f"随机 {take} 个有效像元:")
    for idx in chosen:
        row = int(rows[idx])
        col = int(cols[idx])
        print(
            f"  lon={lons[col]:7.2f}, lat={lats[row]:6.2f}, "
            f"Sz={sz[row, col]:7.3f}, time={minute_label(best_minute[row, col])}, "
            f"err={best_diff[row, col]:6.3f} deg"
        )


def process_date(args: argparse.Namespace, date: str) -> None:
    if args.max_error_deg is not None and args.max_error_deg <= 0:
        raise ValueError("--max-error-deg must be > 0.")
    if args.stride < 1:
        raise ValueError("--stride 必须 >= 1。")
    if args.block_rows < 1:
        raise ValueError("--block-rows 必须 >= 1。")
    if args.prior_window_minutes < 0:
        raise ValueError("--prior-window-minutes 必须 >= 0。")
    if args.end_date and (args.output_npz or args.output_fig or args.output_error_fig):
        raise ValueError("日期范围模式下不可指定 --output-npz / --output-fig / --output-error-fig。")

    start_minute = parse_hhmm(args.start)
    end_minute = parse_hhmm(args.end)
    if end_minute < start_minute:
        raise ValueError("--end 必须不早于 --start。")
    local_overpass_minute = parse_hhmm(args.local_overpass_time)

    sz_path = resolve_sz_path(date, args.sz if args.end_date is None else None)
    if not sz_path.exists():
        if args.end_date:
            print(f"  跳过 {date}: Sz 文件不存在 {sz_path}")
            return
        raise FileNotFoundError(f"FY-3C Sz 文件不存在: {sz_path}")
    output_suffix = ""
    if args.use_local_time_prior:
        output_suffix = f"_prior{minute_filename_label(local_overpass_minute)}w{args.prior_window_minutes}"
    out_npz = args.output_npz or OUTPUT_TIMESTAMP_DIR / f"fy3c_timestamp_{date}{output_suffix}_stride{args.stride}.npz"
    out_fig = args.output_fig or OUTPUT_TIMESTAMP_DIR / f"fy3c_timestamp_{date}{output_suffix}.png"
    out_error_fig = args.output_error_fig or OUTPUT_TIMESTAMP_DIR / f"fy3c_timestamp_error_{date}{output_suffix}.png"
    candidate_minutes = np.arange(start_minute, end_minute + 1, dtype=np.int16)

    print("FY-3C Sz 时间戳估计:")
    print(f"  Sz: {sz_path}")
    print(f"  日期: {date}")
    print(
        f"  搜索窗: {minute_label(start_minute)}-{minute_label(end_minute)} CST "
        f"(UTC{args.timezone_hours:+g})"
    )
    print("  假设: 候选分钟为北京时间；太阳天顶角计算使用经度修正的真太阳时。")
    print(f"  stride={args.stride}, block_rows={args.block_rows}, candidate_minutes={len(candidate_minutes)}")

    sz_raw = read_sz_product(sz_path, args.stride)
    sz, valid = clean_sz(sz_raw)
    zero_fill_count = int(np.count_nonzero(np.isfinite(sz_raw) & (sz_raw == 0.0)))
    print(f"  Sz exact-zero fill pixels excluded: {zero_fill_count}")
    lons, lats = build_sampled_grid(args.stride, sz.shape)
    prior_center_by_lon = None
    prior_min_by_lon = None
    prior_max_by_lon = None
    run_label = None
    if args.use_local_time_prior:
        prior_center_by_lon, prior_min_by_lon, prior_max_by_lon = build_local_time_prior(
            lons=lons,
            start_minute=start_minute,
            end_minute=end_minute,
            local_overpass_minute=local_overpass_minute,
            prior_window_minutes=args.prior_window_minutes,
        )
        usable_cols = prior_min_by_lon <= prior_max_by_lon
        run_label = f"local prior {minute_label(local_overpass_minute)} +/-{args.prior_window_minutes}min"
        print(f"  地方时先验: {run_label}")
        print(
            "  先验中心(CST): "
            f"{minute_label(np.nanmin(prior_center_by_lon))}-{minute_label(np.nanmax(prior_center_by_lon))}; "
            f"可用经度列={int(usable_cols.sum())}/{usable_cols.size}"
        )
        required_start = int(np.ceil(np.nanmin(prior_center_by_lon) - args.prior_window_minutes))
        required_end = int(np.floor(np.nanmax(prior_center_by_lon) + args.prior_window_minutes))
        if start_minute > required_start or end_minute < required_end:
            print(
                "  WARNING: global search window clips the longitude-dependent prior; "
                f"full coverage needs {minute_label(required_start)}-"
                f"{minute_label(required_end)} CST"
            )
    else:
        print("  地方时先验: off")

    finite_raw = sz_raw[np.isfinite(sz_raw)]
    if finite_raw.size:
        raw_p = np.nanpercentile(finite_raw, [0.0, 2.0, 50.0, 98.0, 100.0])
        print(
            "Sz 原始统计: "
            f"min={raw_p[0]:.3f}, p2={raw_p[1]:.3f}, median={raw_p[2]:.3f}, "
            f"p98={raw_p[3]:.3f}, max={raw_p[4]:.3f}"
        )

    best_minute, best_diff = estimate_timestamp(
        sz=sz,
        valid=valid,
        lons=lons,
        lats=lats,
        date=date,
        candidate_minutes=candidate_minutes,
        timezone_hours=args.timezone_hours,
        block_rows=args.block_rows,
        prior_min_by_lon=prior_min_by_lon,
        prior_max_by_lon=prior_max_by_lon,
    )

    fitted_before_qc = (best_minute >= 0) & np.isfinite(best_diff)
    quality_mask = apply_fit_quality_mask(best_minute, best_diff, args.max_error_deg)
    rejected_by_error = int(np.count_nonzero(fitted_before_qc & ~quality_mask))
    if args.max_error_deg is not None:
        print(
            f"  Residual QC: <= {args.max_error_deg:g} deg; "
            f"rejected={rejected_by_error}/{int(fitted_before_qc.sum())}"
        )

    describe_results(
        sz=sz,
        valid=valid,
        best_minute=best_minute,
        best_diff=best_diff,
        lons=lons,
        lats=lats,
        start_minute=start_minute,
        end_minute=end_minute,
        sample_count=args.sample_count,
        seed=args.seed,
        prior_min_by_lon=prior_min_by_lon,
        prior_max_by_lon=prior_max_by_lon,
    )

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        best_minute=best_minute,
        best_diff=best_diff,
        lons=lons,
        lats=lats,
        lon_min=float(lons[0]),
        lon_max=float(lons[-1]),
        lat_min=float(lats[-1]),
        lat_max=float(lats[0]),
        fy3_lon_start=FY3_LON_START,
        fy3_lat_start=FY3_LAT_START,
        fy3_step=FY3_STEP,
        stride=args.stride,
        date=date,
        start_minute=start_minute,
        end_minute=end_minute,
        timezone_hours=args.timezone_hours,
        use_local_time_prior=bool(args.use_local_time_prior),
        local_overpass_minute=local_overpass_minute,
        prior_window_minutes=args.prior_window_minutes,
        local_time_prior_label=run_label or "",
        prior_center_by_lon=prior_center_by_lon if prior_center_by_lon is not None else np.array([], dtype=np.float32),
        prior_min_by_lon=prior_min_by_lon if prior_min_by_lon is not None else np.array([], dtype=np.int16),
        prior_max_by_lon=prior_max_by_lon if prior_max_by_lon is not None else np.array([], dtype=np.int16),
        quality_mask=quality_mask,
        max_error_deg=np.nan if args.max_error_deg is None else float(args.max_error_deg),
        zero_fill_count=zero_fill_count,
        rejected_by_error=rejected_by_error,
        time_assumption="candidate minutes are CST clock time; SZA uses longitude-corrected true solar time",
        sz_path=str(sz_path),
    )
    print(f"已保存: {out_npz}")

    plot_timestamp(
        best_minute,
        lons,
        lats,
        start_minute,
        end_minute,
        date,
        out_fig,
        args.plot_max_pixels,
        args.stride,
        run_label=run_label,
    )
    print(f"已保存: {out_fig}")
    plot_error(best_diff, lons, lats, date, out_error_fig, args.plot_max_pixels, args.stride, run_label=run_label)
    print(f"已保存: {out_error_fig}")


def main() -> None:
    args = parse_args()
    dates = list(iter_dates(args.date, args.end_date))
    for idx, date in enumerate(dates, start=1):
        if len(dates) > 1:
            print(f"\n[{idx}/{len(dates)}] {date}")
        process_date(args, date)


if __name__ == "__main__":
    main()
