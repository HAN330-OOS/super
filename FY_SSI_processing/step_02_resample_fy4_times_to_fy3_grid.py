"""
工作流第一步：将 FY-4A 各时次 4 km SSI 重采样到 FY-3C 全幅 0.01° 网格（3600×6200）。

默认处理 20180501、UTC 00:00–06:00 窗口内全部 NC（≈ CST 08:00–14:00），输出 FY4_HHMM_on_FY3.npz。
文件名 HHMM 为 UTC，见 FY-4A_AGRI_L2_SSI_NOM_4000M_V1.1.pdf。
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

try:
    from pyresample.geometry import GridDefinition, SwathDefinition
    from pyresample.kd_tree import resample_gauss, resample_nearest
except ImportError as exc:
    raise SystemExit("pyresample 未安装，请运行: pip install pyresample") from exc

from fy_io import (
    DEFAULT_FY4_GEO_RAW,
    DEFAULT_FY4_ROOT,
    FY3_STEP,
    OUTPUT_RESAMPLED_ON_FY3_DIR,
    build_target_grid,
    dual_time_label,
    fy3_extent,
    iter_dates,
    list_fy4_nc_files_in_utc_window,
    minute_to_hhmm,
    read_fy4_lon_lat,
    read_fy4_ssi_with_dqf,
    resampled_on_fy3_path,
    utc_minutes_to_cst_minutes,
)

def parse_args() -> argparse.Namespace:
    lon_min, lon_max, lat_min, lat_max = fy3_extent()
    parser = argparse.ArgumentParser(
        description="批量将 FY-4A 各时次 SSI 重采样到 FY-3C 规则网格（工作流 step 03 主流程）。",
        epilog=(
            "示例:\n"
            "  # 仅列出 00:00-06:00 UTC 时次（≈ CST 08:00-14:00）\n"
            "  py -3 FY_SSI_processing\\step_02_resample_fy4_times_to_fy3_grid.py --list-only\n\n"
            "  # 冒烟：前 2 个时次\n"
            "  py -3 FY_SSI_processing\\step_02_resample_fy4_times_to_fy3_grid.py --limit 2\n\n"
            "  # 全部时次（默认全幅 73-135E, 18-54N）\n"
            "  py -3 FY_SSI_processing\\step_02_resample_fy4_times_to_fy3_grid.py --skip-existing"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--date", default="20180501", help="日期 YYYYMMDD。")
    parser.add_argument("--end-date", default=None, help="结束日期 YYYYMMDD（含）；给定则自 --date 起逐日处理。")
    parser.add_argument("--fy4-root", type=Path, default=DEFAULT_FY4_ROOT)
    parser.add_argument("--fy4-geo-raw", type=Path, default=DEFAULT_FY4_GEO_RAW)
    parser.add_argument("--fy4-variable", default="SSI")
    parser.add_argument("--start", default="00:00", help="UTC 窗口起始 HH:MM（含）。")
    parser.add_argument("--end", default="06:00", help="UTC 窗口结束 HH:MM（含）。")
    parser.add_argument("--lon-min", type=float, default=lon_min)
    parser.add_argument("--lon-max", type=float, default=lon_max)
    parser.add_argument("--lat-min", type=float, default=lat_min)
    parser.add_argument("--lat-max", type=float, default=lat_max)
    parser.add_argument("--step", type=float, default=FY3_STEP)
    parser.add_argument("--method", choices=("nearest", "gauss"), default="nearest")
    parser.add_argument("--radius-m", type=float, default=25000.0)
    parser.add_argument("--min-valid-ssi", type=float, default=0.0)
    parser.add_argument(
        "--mask-dqf2",
        action="store_true",
        help="同时掩膜 DQF==2（默认仅 DQF==3；V1.0 若 DQF 全为 3 则自动跳过）。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录，默认 outputs/fy/resampled_on_fy3/<date>/。",
    )
    parser.add_argument("--skip-existing", action="store_true", help="跳过已存在的 npz。")
    parser.add_argument("--write-geotiff", action="store_true", help="同时写出 GeoTIFF（需 rasterio）。")
    parser.add_argument("--list-only", action="store_true", help="仅列出时次，不重采样。")
    parser.add_argument("--limit", type=int, default=None, help="最多处理前 N 个时次（测试用）。")
    return parser.parse_args()


def resample_fy4(
    ssi: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    target_lons: np.ndarray,
    target_lats: np.ndarray,
    method: str,
    radius_m: float,
) -> np.ndarray:
    valid = np.isfinite(ssi) & np.isfinite(lon) & np.isfinite(lat)
    data = np.where(valid, ssi, np.nan)

    source_def = SwathDefinition(lons=lon, lats=lat)
    target_def = GridDefinition(lons=target_lons, lats=target_lats)

    if method == "nearest":
        return resample_nearest(
            source_def,
            data,
            target_def,
            radius_of_influence=radius_m,
            fill_value=np.nan,
        ).astype(np.float32)

    return resample_gauss(
        source_def,
        data,
        target_def,
        radius_of_influence=radius_m * 2,
        sigmas=radius_m,
        neighbours=32,
        fill_value=np.nan,
    ).astype(np.float32)


def valid_fraction(data: np.ndarray) -> float:
    return float(np.isfinite(data).sum()) / data.size if data.size else 0.0


def describe_resampled(label: str, data: np.ndarray) -> None:
    frac = 100.0 * valid_fraction(data)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        print(f"  {label}: shape={data.shape}, valid=0.0%")
        return
    print(
        f"  {label}: shape={data.shape}, valid={frac:.1f}%, "
        f"min={np.nanmin(finite):.2f}, median={np.nanmedian(finite):.2f}, "
        f"max={np.nanmax(finite):.2f} W/m2"
    )


def write_geotiff(path: Path, data: np.ndarray, lons_1d: np.ndarray, lats_1d: np.ndarray) -> None:
    try:
        import rasterio
        from rasterio.transform import from_bounds
    except ImportError as exc:
        raise RuntimeError("写出 GeoTIFF 需要 rasterio: pip install rasterio") from exc

    lon_min = float(lons_1d[0])
    lon_max = float(lons_1d[-1])
    lat_max = float(lats_1d[0])
    lat_min = float(lats_1d[-1])
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, data.shape[1], data.shape[0])

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(np.where(np.isfinite(data), data, np.nan).astype(np.float32), 1)


def process_slot(
    nc_path: Path,
    start_minute: int,
    fy4_lon: np.ndarray,
    fy4_lat: np.ndarray,
    target_lons: np.ndarray,
    target_lats: np.ndarray,
    lons_1d: np.ndarray,
    lats_1d: np.ndarray,
    args: argparse.Namespace,
    output_dir: Path,
    date: str,
) -> Path | None:
    time_label = minute_to_hhmm(start_minute)
    cst_label = minute_to_hhmm(int(utc_minutes_to_cst_minutes(start_minute)))
    out_npz = resampled_on_fy3_path(output_dir, time_label)
    out_tif = out_npz.with_suffix(".tif")

    if args.skip_existing and out_npz.exists():
        print(f"[跳过] {dual_time_label(time_label)} — 已存在 {out_npz.name}")
        return None

    print(f"[重采样] {dual_time_label(time_label)} ← {nc_path.name}")
    t0 = time.perf_counter()

    ssi, ssi_meta = read_fy4_ssi_with_dqf(
        nc_path,
        variable=args.fy4_variable,
        mask_dqf2=args.mask_dqf2,
        min_valid_ssi=args.min_valid_ssi,
    )
    src_valid = valid_fraction(ssi)
    print(f"  源 SSI 有效像元: {100 * src_valid:.1f}%")

    resampled = resample_fy4(
        ssi,
        fy4_lon,
        fy4_lat,
        target_lons,
        target_lats,
        args.method,
        args.radius_m,
    )
    describe_resampled("FY-4 on FY-3 grid", resampled)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        ssi=resampled,
        lons=lons_1d,
        lats=lats_1d,
        source_nc=str(nc_path),
        time_utc=time_label,
        time_cst=cst_label,
        start_minute_utc=int(start_minute),
        start_minute_cst=int(utc_minutes_to_cst_minutes(start_minute)),
        shape=np.array(resampled.shape, dtype=np.int32),
        date=date,
        method=args.method,
        radius_m=args.radius_m,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        step=args.step,
        version_of_software=ssi_meta.get("version_of_software") or "",
    )
    elapsed = time.perf_counter() - t0
    size_mb = out_npz.stat().st_size / (1024 * 1024)
    print(f"  已保存 {out_npz.name} ({size_mb:.1f} MB, {elapsed:.1f} s)")

    if args.write_geotiff:
        write_geotiff(out_tif, resampled, lons_1d, lats_1d)
        print(f"  已保存 {out_tif.name}")

    return out_npz


def process_date(args: argparse.Namespace, date: str) -> None:
    output_dir = args.output_dir or (OUTPUT_RESAMPLED_ON_FY3_DIR / f"{date}_utc")
    if args.end_date and args.output_dir is not None:
        raise ValueError("日期范围模式下不可指定 --output-dir。")

    try:
        entries = list_fy4_nc_files_in_utc_window(
            date, args.start, args.end, args.fy4_root
        )
    except FileNotFoundError as exc:
        if args.end_date:
            print(f"  跳过 {date}: {exc}")
            return
        raise
    if args.limit is not None:
        entries = entries[: args.limit]

    print("FY-4A → FY-3C 批量重采样（工作流 step 03）")
    print(f"  日期: {date}")
    print(f"  FY-4 目录: {args.fy4_root / date}")
    print(f"  时刻窗: {args.start}–{args.end} UTC（文件名解析，含端点；≈ CST +8h）")
    print(f"  目标区域: {args.lon_min}-{args.lon_max}E, {args.lat_min}-{args.lat_max}N, step={args.step}°")
    print(f"  方法: {args.method}, radius={args.radius_m} m")
    print(f"  输出: {output_dir}")
    print(f"  匹配时次: {len(entries)} 个")
    for path, minute in entries:
        label = minute_to_hhmm(minute)
        print(f"    {dual_time_label(label)}  {path.name}")
    if args.list_only:
        return

    if not entries:
        print(f"  窗口内无 FY-4A NC 文件，跳过 {date}。")
        return

    lons_1d, lats_1d, target_lons, target_lats = build_target_grid(
        args.lon_min, args.lon_max, args.lat_min, args.lat_max, args.step
    )
    print(f"  目标网格 shape: ({len(lats_1d)}, {len(lons_1d)})")

    if args.fy4_geo_raw.exists():
        print(f"读取 FY-4A 经纬度 raw（一次）: {args.fy4_geo_raw}")
    else:
        print(f"FY-4A 经纬度 raw 不存在 ({args.fy4_geo_raw})，使用 NSMC/CGMS 公式生成...")
    fy4_lon, fy4_lat = read_fy4_lon_lat(args.fy4_geo_raw)
    geo_valid = np.isfinite(fy4_lon) & np.isfinite(fy4_lat)
    print(f"  经纬度有效像元: {100 * geo_valid.sum() / geo_valid.size:.1f}%")

    written = 0
    skipped = 0
    t_all = time.perf_counter()
    for idx, (nc_path, start_minute) in enumerate(entries, start=1):
        print(f"\n--- [{idx}/{len(entries)}] ---")
        result = process_slot(
            nc_path,
            start_minute,
            fy4_lon,
            fy4_lat,
            target_lons,
            target_lats,
            lons_1d,
            lats_1d,
            args,
            output_dir,
            date,
        )
        if result is None:
            skipped += 1
        else:
            written += 1

    total_s = time.perf_counter() - t_all
    print(
        f"\n完成: 写入 {written} 个, 跳过 {skipped} 个, "
        f"共 {len(entries)} 时次, 耗时 {total_s:.1f} s"
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
