"""
工作流 step 03c：将 REST2 晴空 GHI（0.5° 计算网格）双线性重采样到 FY-3C 0.01° 网格。

目标网格由 fy_io 常量定义（FY3_LON_START, FY3_LAT_START, FY3_SHAPE, FY3_STEP），
经 build_target_grid() + fy3_extent() 生成，与 step_02 / step_05 一致（3600×6200，73–135°E，18–54°N）。
FY4_*_on_FY3.npz 仅用于读取 SSI 计算 CSI；lons/lats 不取自 FY4。

默认 9 个 UTC 时次（20180501）：0000, 0100, 0200, 0245, 0300, 0400, 0500, 0545, 0600。

示例:
    py -3 FY_SSI_processing\\step_04_resample_rest2_ghi_to_fy3_grid.py
    py -3 FY_SSI_processing\\step_04_resample_rest2_ghi_to_fy3_grid.py --date 20180501 --write-csi
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from fy_io import (
    OUTPUT_RESAMPLED_ON_FY3_DIR,
    OUTPUT_REST2_ON_FY3_DIR,
    PROJECT_ROOT,
    build_target_grid,
    clean_ssi,
    csi_on_fy3_path,
    dual_time_label,
    fy3_extent,
    iter_dates,
    resampled_on_fy3_path,
    rest2_cache_path,
    rest2_on_fy3_path,
    utc_minutes_to_cst_minutes,
    utc_label_to_cst_label,
)

FY3_FY4_SLOT_SPECS = [
    "0000", "0100", "0200", "0245", "0300", "0400", "0500", "0545", "0600",
]

DEFAULT_REST2_CACHE = PROJECT_ROOT / "MERRA2" / "rest2_outputs"
DEFAULT_GRID_STEP = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 REST2 晴空 GHI 重采样到 FY-3C 网格（与 FY4_*_on_FY3 对齐）。",
        epilog=(
            "示例:\n"
            "  py -3 FY_SSI_processing\\step_04_resample_rest2_ghi_to_fy3_grid.py\n"
            "  py -3 FY_SSI_processing\\step_04_resample_rest2_ghi_to_fy3_grid.py --write-csi --skip-existing"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--date", default="20180501", help="日期 YYYYMMDD。")
    parser.add_argument("--end-date", default=None, help="结束日期 YYYYMMDD（含）；给定则自 --date 起逐日处理。")
    parser.add_argument(
        "--rest2-cache-dir",
        type=Path,
        default=DEFAULT_REST2_CACHE,
        help="REST2 0.5° 缓存目录（rest2_ghi_<date>_<slot>_step*.npz）。",
    )
    parser.add_argument(
        "--rest2-grid-step",
        type=float,
        default=DEFAULT_GRID_STEP,
        help="REST2 计算网格步长（度），默认 0.5。",
    )
    parser.add_argument(
        "--resampled-dir",
        type=Path,
        default=None,
        help="FY4_*_on_FY3.npz 目录（默认 outputs/fy/resampled_on_fy3/<date>_utc/）。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录（默认 outputs/fy/rest2_on_fy3/<date>_utc/）。",
    )
    parser.add_argument("--skip-existing", action="store_true", help="跳过已存在的 npz。")
    parser.add_argument(
        "--write-csi",
        action="store_true",
        help="同时写出 CSI_<slot>_on_FY3.npz（csi = FY4 SSI / REST2 GHI）。",
    )
    parser.add_argument("--list-only", action="store_true", help="仅列出时次，不重采样。")
    parser.add_argument(
        "--build-rest2-cache",
        action="store_true",
        help="若某日 REST2 缓存缺失，调用 MERRA2/build_rest2_ghi_slot_cache.py 生成。",
    )
    return parser.parse_args()


def default_resampled_dir(date: str) -> Path:
    return OUTPUT_RESAMPLED_ON_FY3_DIR / f"{date}_utc"


def default_output_dir(date: str) -> Path:
    return OUTPUT_REST2_ON_FY3_DIR / f"{date}_utc"


def load_rest2_coarse(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as archive:
        ghi = archive["ghi"].astype(np.float32)
        lons = archive["lons"].astype(np.float64)
        lats = archive["lats"].astype(np.float64)
    ghi = np.where(np.isfinite(ghi) & (ghi > 0.0), ghi, np.nan)
    return ghi, lons, lats


def load_fy4_ssi(path: Path) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Load FY4 SSI and optional lons/lats for grid verification only."""
    with np.load(path, allow_pickle=True) as archive:
        ssi = clean_ssi(archive["ssi"]) if "ssi" in archive else None
        lons = archive["lons"].astype(np.float64) if "lons" in archive else None
        lats = archive["lats"].astype(np.float64) if "lats" in archive else None
    return ssi, lons, lats


def resample_rest2_to_fy3(
    ghi: np.ndarray,
    lons_src: np.ndarray,
    lats_src_desc: np.ndarray,
    lons_tgt_2d: np.ndarray,
    lats_tgt_2d: np.ndarray,
) -> np.ndarray:
    """Bilinear REST2 GHI (descending lat storage) onto FY-3 target mesh."""
    lats_asc = np.ascontiguousarray(lats_src_desc[::-1])
    lons_src = np.ascontiguousarray(lons_src)
    ghi_asc = np.ascontiguousarray(ghi[::-1, :], dtype=np.float64)
    interp = RegularGridInterpolator(
        (lats_asc, lons_src),
        ghi_asc,
        bounds_error=False,
        fill_value=np.nan,
    )
    target_points = np.ascontiguousarray(
        np.column_stack([lats_tgt_2d.ravel(), lons_tgt_2d.ravel()])
    )
    return interp(target_points).reshape(lats_tgt_2d.shape).astype(np.float32)


def valid_fraction(data: np.ndarray) -> float:
    return float(np.isfinite(data).sum()) / data.size if data.size else 0.0


def describe_field(label: str, data: np.ndarray) -> None:
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


def grids_match(
    lons_a: np.ndarray,
    lats_a: np.ndarray,
    lons_b: np.ndarray,
    lats_b: np.ndarray,
    atol: float = 1e-6,
) -> bool:
    return bool(
        lons_a.shape == lons_b.shape
        and lats_a.shape == lats_b.shape
        and np.allclose(lons_a, lons_b, atol=atol, rtol=0.0)
        and np.allclose(lats_a, lats_b, atol=atol, rtol=0.0)
    )


def compute_csi(fy4_ssi: np.ndarray, rest2_ghi: np.ndarray) -> np.ndarray:
    valid = np.isfinite(fy4_ssi) & np.isfinite(rest2_ghi) & (rest2_ghi > 0.0)
    csi = np.full(fy4_ssi.shape, np.nan, dtype=np.float32)
    csi[valid] = (fy4_ssi[valid] / rest2_ghi[valid]).astype(np.float32)
    return csi


def process_slot(
    utc_label: str,
    date: str,
    resampled_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    grid_step: float,
    write_csi: bool,
    skip_existing: bool,
) -> dict[str, object] | None:
    fy4_path = resampled_on_fy3_path(resampled_dir, utc_label)
    cache_path = rest2_cache_path(cache_dir, date, utc_label, grid_step)
    out_rest2 = rest2_on_fy3_path(output_dir, utc_label)
    out_csi = csi_on_fy3_path(output_dir, utc_label)

    if skip_existing and out_rest2.exists() and (not write_csi or out_csi.exists()):
        print(f"[跳过] {dual_time_label(utc_label)} — 已存在")
        return None

    if not fy4_path.exists():
        print(f"[缺失] {dual_time_label(utc_label)} — 无 {fy4_path.name}")
        return None
    if not cache_path.exists():
        print(f"[缺失] {dual_time_label(utc_label)} — 无 {cache_path.name}")
        return None

    print(f"[重采样] {dual_time_label(utc_label)}")
    t0 = time.perf_counter()

    lon_min, lon_max, lat_min, lat_max = fy3_extent()
    lons_1d, lats_1d, lons_2d, lats_2d = build_target_grid(lon_min, lon_max, lat_min, lat_max)
    fy4_ssi, fy4_lons, fy4_lats = load_fy4_ssi(fy4_path)
    if fy4_lons is not None and fy4_lats is not None:
        if not grids_match(fy4_lons, fy4_lats, lons_1d, lats_1d):
            print(
                f"  [警告] {fy4_path.name} 网格与 FY-3C 标准网格不一致 "
                f"(FY4 lon [{fy4_lons[0]:.2f},{fy4_lons[-1]:.2f}] "
                f"lat [{fy4_lats[0]:.2f},{fy4_lats[-1]:.2f}])"
            )

    ghi_coarse, lons_coarse, lats_coarse = load_rest2_coarse(cache_path)
    ghi_on_fy3 = resample_rest2_to_fy3(ghi_coarse, lons_coarse, lats_coarse, lons_2d, lats_2d)
    describe_field("REST2 on FY-3 grid", ghi_on_fy3)
    cst_label = utc_label_to_cst_label(utc_label)
    start_minute_utc = int(utc_label[:2]) * 60 + int(utc_label[2:])

    out_rest2.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_rest2,
        ghi=ghi_on_fy3,
        lons=lons_1d,
        lats=lats_1d,
        slot_utc=utc_label,
        time_utc=utc_label,
        time_cst=cst_label,
        start_minute_utc=start_minute_utc,
        start_minute_cst=int(utc_minutes_to_cst_minutes(start_minute_utc)),
        source="REST2",
        grid_step_computation=grid_step,
        grid_step_target=0.01,
        date=date,
        shape=np.array(ghi_on_fy3.shape, dtype=np.int32),
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        rest2_cache=str(cache_path),
        fy4_reference=str(fy4_path),
    )
    elapsed = time.perf_counter() - t0
    size_mb = out_rest2.stat().st_size / (1024 * 1024)
    print(f"  已保存 {out_rest2.name} ({size_mb:.1f} MB, {elapsed:.1f} s)")

    result: dict[str, object] = {
        "slot": utc_label,
        "rest2_path": out_rest2,
        "shape": ghi_on_fy3.shape,
        "valid_frac": valid_fraction(ghi_on_fy3),
    }

    if write_csi and fy4_ssi is not None:
        csi = compute_csi(fy4_ssi, ghi_on_fy3)
        describe_field("CSI (FY4/REST2)", csi)
        np.savez_compressed(
            out_csi,
            csi=csi,
            lons=lons_1d,
            lats=lats_1d,
            slot_utc=utc_label,
            time_utc=utc_label,
            time_cst=cst_label,
            source_fy4=str(fy4_path),
            source_rest2=str(out_rest2),
            date=date,
            shape=np.array(csi.shape, dtype=np.int32),
        )
        csi_mb = out_csi.stat().st_size / (1024 * 1024)
        print(f"  已保存 {out_csi.name} ({csi_mb:.1f} MB)")
        result["csi_path"] = out_csi

    return result


def verify_outputs(results: list[dict[str, object]]) -> None:
    print("\n=== 网格验证 ===")
    lon_min, lon_max, lat_min, lat_max = fy3_extent()
    ref_lons, ref_lats, _, _ = build_target_grid(lon_min, lon_max, lat_min, lat_max)
    expected_shape = (len(ref_lats), len(ref_lons))
    all_ok = True

    for row in results:
        slot = str(row["slot"])
        rest2_path = row["rest2_path"]
        assert isinstance(rest2_path, Path)

        with np.load(rest2_path) as rest2:
            rest2_lons = rest2["lons"]
            rest2_lats = rest2["lats"]
            rest2_shape = rest2["ghi"].shape

        shape_ok = rest2_shape == expected_shape
        grid_ok = grids_match(ref_lons, ref_lats, rest2_lons, rest2_lats)
        ok = shape_ok and grid_ok
        all_ok = all_ok and ok
        status = "OK" if ok else "MISMATCH"
        print(
            f"  {slot}: shape={rest2_shape} (expected {expected_shape}) "
            f"grid={status} valid={100 * float(row['valid_frac']):.1f}%"
        )

    print(
        f"  FY-3C 标准网格: {ref_lons.shape[0]} lons × {ref_lats.shape[0]} lats, "
        f"lon [{ref_lons[0]:.2f}, {ref_lons[-1]:.2f}], "
        f"lat [{ref_lats[0]:.2f}, {ref_lats[-1]:.2f}]"
    )
    print(f"  全部匹配: {'是' if all_ok else '否'}")


def ensure_rest2_cache(date: str, cache_dir: Path, grid_step: float) -> None:
    """Build REST2 slot cache for one date when any expected file is missing."""
    missing = [
        slot
        for slot in FY3_FY4_SLOT_SPECS
        if not rest2_cache_path(cache_dir, date, slot, grid_step).exists()
    ]
    if not missing:
        return
    script = PROJECT_ROOT / "MERRA2" / "build_rest2_ghi_slot_cache.py"
    if not script.exists():
        raise FileNotFoundError(f"REST2 缓存缺失且未找到构建脚本: {script}")
    print(f"  构建 REST2 缓存 ({len(missing)} 个时次缺失): {date}")
    cmd = [
        sys.executable,
        str(script),
        "--date",
        date,
        "--output-dir",
        str(cache_dir),
        "--grid-step",
        str(grid_step),
        "--skip-existing",
    ]
    subprocess.run(cmd, check=True)


def process_date(args: argparse.Namespace, date: str) -> None:
    if args.end_date and (args.resampled_dir or args.output_dir):
        raise ValueError("日期范围模式下不可指定 --resampled-dir / --output-dir。")

    resampled_dir = (args.resampled_dir or default_resampled_dir(date)).resolve()
    output_dir = (args.output_dir or default_output_dir(date)).resolve()
    cache_dir = args.rest2_cache_dir.resolve()
    slots = sorted(
        match.group(1)
        for path in resampled_dir.glob("FY4_*_on_FY3.npz")
        if (match := re.fullmatch(r"FY4_(\d{4})_on_FY3", path.stem))
    )
    if not slots:
        slots = FY3_FY4_SLOT_SPECS

    print("REST2 GHI → FY-3C 网格重采样（工作流 step 03c）")
    print(f"  日期: {date}")
    print(f"  REST2 缓存: {cache_dir} (step={args.rest2_grid_step}°)")
    print(f"  FY4 参考: {resampled_dir}")
    print(f"  输出: {output_dir}")
    print(f"  时次: {len(slots)} 个")
    for slot in slots:
        print(f"    {dual_time_label(slot)}")

    if args.list_only:
        return

    if not resampled_dir.is_dir():
        print(f"FY4 重采样目录不存在，跳过 {date}: {resampled_dir}", file=sys.stderr)
        return
    if not cache_dir.is_dir():
        print(f"REST2 缓存目录不存在: {cache_dir}", file=sys.stderr)
        sys.exit(1)

    if args.build_rest2_cache:
        ensure_rest2_cache(date, cache_dir, args.rest2_grid_step)

    results: list[dict[str, object]] = []
    skipped = 0
    t_all = time.perf_counter()

    for idx, slot in enumerate(slots, start=1):
        print(f"\n--- [{idx}/{len(slots)}] ---")
        row = process_slot(
            slot,
            date,
            resampled_dir,
            cache_dir,
            output_dir,
            args.rest2_grid_step,
            args.write_csi,
            args.skip_existing,
        )
        if row is None:
            skipped += 1
        else:
            results.append(row)

    total_s = time.perf_counter() - t_all
    print(
        f"\n完成: 写入 {len(results)} 个, 跳过 {skipped} 个, "
        f"共 {len(slots)} 时次, 耗时 {total_s:.1f} s"
    )

    if results:
        verify_outputs(results)
        print("\n字段名（step_04 集成）:")
        print("  REST2_*_on_FY3.npz: ghi, lons, lats, slot_utc, source, grid_step_computation")
        if args.write_csi:
            print("  CSI_*_on_FY3.npz: csi, lons, lats, slot_utc")


def main() -> None:
    args = parse_args()
    dates = list(iter_dates(args.date, args.end_date))
    for idx, date in enumerate(dates, start=1):
        if len(dates) > 1:
            print(f"\n[{idx}/{len(dates)}] {date}")
        process_date(args, date)


if __name__ == "__main__":
    main()
