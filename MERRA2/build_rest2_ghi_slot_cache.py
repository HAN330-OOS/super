"""
Build REST2 0.5° GHI cache files for FY-3/FY-4 pipeline step_03c.

Outputs one npz per UTC slot:
  rest2_ghi_<YYYYMMDD>_<HHMM>_step0.50.npz
with fields ghi, lons, lats, time_utc, grid_step.

Uses local daily MERRA-2 SUB.nc4 files from the configured data directory.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from data_paths import merra2_data_root

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "FY_SSI_processing"))

from fy_io import DEFAULT_REST2_CACHE_DIR, iter_dates, rest2_cache_path  # noqa: E402

FY3_FY4_SLOT_SPECS = [
    "0000", "0100", "0200", "0245", "0300", "0400", "0500", "0545", "0600",
]

LON_MIN, LON_MAX = 73.0, 135.0
LAT_MIN, LAT_MAX = 18.0, 54.0
DEFAULT_GRID_STEP = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build REST2 0.5° GHI cache npz for FY pipeline slots.",
    )
    parser.add_argument("--date", required=True, help="YYYYMMDD（起始日期）")
    parser.add_argument("--end-date", default=None, help="结束日期 YYYYMMDD（含）；给定则自 --date 起逐日处理。")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_REST2_CACHE_DIR,
        help="Cache directory (default MERRA2/rest2_outputs).",
    )
    parser.add_argument("--grid-step", type=float, default=DEFAULT_GRID_STEP)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--slots", nargs="*", default=None, help="Optional HHMM list.")
    return parser.parse_args()


def build_target_axes(step: float) -> tuple[np.ndarray, np.ndarray]:
    half = step / 2.0
    lons = np.arange(LON_MIN + half, LON_MAX + step / 2, step, dtype=np.float64)
    lats = np.arange(LAT_MAX, LAT_MIN - step / 2, -step, dtype=np.float64)
    return lons, lats


def slot_to_timestamp(date: str, slot: str) -> pd.Timestamp:
    return pd.Timestamp(
        year=int(date[:4]),
        month=int(date[4:6]),
        day=int(date[6:8]),
        hour=int(slot[:2]),
        minute=int(slot[2:4]),
        tz="UTC",
    )


def _file_for(product: str, date: str) -> Path:
    return merra2_data_root() / product / f"MERRA2_400.{product}.{date}.SUB.nc4"


def _install_bsrn_compat_modules() -> None:
    geometry = types.ModuleType("bsrn.physics.geometry")
    geometry.get_bni_extra = lambda index, solar_constant=1366.1: pd.Series(
        solar_constant, index=pd.DatetimeIndex(index)
    )
    physics = types.ModuleType("bsrn.physics")
    physics.geometry = geometry
    constants = types.ModuleType("bsrn.constants")
    constants.BSRN_STATIONS = {}
    constants.LINKE_TURBIDITY = {}
    constants.MERRA2_HF_REPO_ID = ""
    constants.HF_MAINTAINER_EMAIL = ""
    io_module = types.ModuleType("bsrn.io")
    mcclear = types.ModuleType("bsrn.io.mcclear")
    mcclear.fetch_mcclear = None
    merra2_placeholder = types.ModuleType("bsrn.io.merra2")
    merra2_placeholder.fetch_rest2 = None
    bsrn = types.ModuleType("bsrn")
    bsrn.physics = physics
    bsrn.constants = constants
    bsrn.io = io_module
    sys.modules.setdefault("bsrn", bsrn)
    sys.modules.setdefault("bsrn.physics", physics)
    sys.modules.setdefault("bsrn.physics.geometry", geometry)
    sys.modules.setdefault("bsrn.constants", constants)
    sys.modules.setdefault("bsrn.io", io_module)
    sys.modules.setdefault("bsrn.io.mcclear", mcclear)
    sys.modules.setdefault("bsrn.io.merra2", merra2_placeholder)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _solar_zenith_grid(timestamp: pd.Timestamp, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    idx = pd.DatetimeIndex([timestamp], tz="UTC")
    fractional_hour = (
        idx.hour.to_numpy(dtype=float)
        + idx.minute.to_numpy(dtype=float) / 60.0
        + idx.second.to_numpy(dtype=float) / 3600.0
    )
    day_of_year = idx.dayofyear.to_numpy(dtype=float)
    gamma = 2.0 * np.pi / 365.0 * (day_of_year - 1.0 + (fractional_hour - 12.0) / 24.0)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2.0 * gamma)
        - 0.040849 * np.sin(2.0 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2.0 * gamma)
        + 0.000907 * np.sin(2.0 * gamma)
        - 0.002697 * np.cos(3.0 * gamma)
        + 0.00148 * np.sin(3.0 * gamma)
    )
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    true_solar_time = (fractional_hour * 60.0 + eqtime + 4.0 * lon_grid) % 1440.0
    hour_angle = np.radians(true_solar_time / 4.0 - 180.0)
    lat_rad = np.radians(lat_grid)
    cos_zenith = (
        np.sin(lat_rad) * np.sin(decl)
        + np.cos(lat_rad) * np.cos(decl) * np.cos(hour_angle)
    )
    return np.degrees(np.arccos(np.clip(cos_zenith, -1.0, 1.0))).astype(np.float64)


def _open_daily_field(path: Path, var_map: dict[str, str]) -> xr.Dataset:
    if not path.exists():
        raise FileNotFoundError(f"Missing MERRA-2 file: {path}")
    ds = xr.open_dataset(path, engine="netcdf4")
    renamed = {}
    for source_name, target_name in var_map.items():
        if source_name not in ds:
            ds.close()
            raise KeyError(f"{source_name} not found in {path.name}")
        renamed[target_name] = ds[source_name]
    out = xr.Dataset(renamed)
    out = out.sel(lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX))
    ds.close()
    return out


def _load_merra2_cube(date: str) -> xr.Dataset:
    aer = _open_daily_field(
        _file_for("tavg1_2d_aer_Nx", date),
        {"TOTEXTTAU": "AOD55", "TOTANGSTR": "ALPHA"},
    )
    rad = _open_daily_field(
        _file_for("tavg1_2d_rad_Nx", date),
        {"ALBEDO": "ALBEDO"},
    )
    slv = _open_daily_field(
        _file_for("tavg1_2d_slv_Nx", date),
        {"PS": "PS", "TO3": "TO3", "TQV": "TQV"},
    )
    merged = xr.merge([aer, rad, slv], join="inner")
    return merged


def _interp_field_to_grid(
    field: np.ndarray,
    src_lats: np.ndarray,
    src_lons: np.ndarray,
    tgt_lons: np.ndarray,
    tgt_lats: np.ndarray,
) -> np.ndarray:
    lats_asc = src_lats if src_lats[0] < src_lats[-1] else src_lats[::-1]
    data = field if src_lats[0] < src_lats[-1] else field[::-1, :]
    interp = RegularGridInterpolator(
        (lats_asc, src_lons),
        data,
        bounds_error=False,
        fill_value=np.nan,
    )
    lon_grid, lat_grid = np.meshgrid(tgt_lons, tgt_lats)
    points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    return interp(points).reshape(len(tgt_lats), len(tgt_lons))


def _rest2_inputs_at_time(cube: xr.Dataset, timestamp: pd.Timestamp) -> dict[str, np.ndarray]:
    snap = cube.interp(time=np.datetime64(timestamp.tz_convert("UTC").to_pydatetime()), method="linear")
    src_lats = snap["lat"].values.astype(np.float64)
    src_lons = snap["lon"].values.astype(np.float64)
    tgt_lons, tgt_lats = build_target_axes(DEFAULT_GRID_STEP)
    fields: dict[str, np.ndarray] = {}
    for name in ("AOD55", "ALPHA", "ALBEDO", "PS", "TO3", "TQV"):
        fields[name] = _interp_field_to_grid(
            snap[name].values.astype(np.float64),
            src_lats,
            src_lons,
            tgt_lons,
            tgt_lats,
        )
    ps = fields["PS"] / 100.0
    tqv = fields["TQV"] / 10.0
    to3 = fields["TO3"] / 1000.0
    aod55 = np.where(fields["AOD55"] == 0.0, 0.001, fields["AOD55"])
    beta = aod55 * (0.55 ** fields["ALPHA"])
    return {
        "PS": ps,
        "ALBEDO": fields["ALBEDO"],
        "ALPHA": fields["ALPHA"],
        "BETA": beta,
        "TO3": to3,
        "TQV": tqv,
    }


def build_slot_cache(
    date: str,
    slot: str,
    cube: xr.Dataset,
    tgt_lons: np.ndarray,
    tgt_lats: np.ndarray,
    clear_sky,
    output_dir: Path,
    grid_step: float,
) -> Path:
    timestamp = slot_to_timestamp(date, slot)
    out_path = rest2_cache_path(output_dir, date, slot, grid_step)
    inputs = _rest2_inputs_at_time(cube, timestamp)
    zenith = _solar_zenith_grid(timestamp, tgt_lons, tgt_lats)
    flat_shape = zenith.size
    target_index = pd.DatetimeIndex([timestamp] * flat_shape, tz="UTC")
    rest2_inputs = pd.DataFrame(
        {name: values.ravel() for name, values in inputs.items()}
    )
    ghi, _bni, _dhi = clear_sky.rest2_model(
        target_index,
        zenith.ravel(),
        rest2_inputs,
    )
    ghi_grid = np.asarray(ghi, dtype=np.float32).reshape(zenith.shape)
    ghi_grid = np.where(np.isfinite(ghi_grid) & (ghi_grid > 0.0), ghi_grid, np.nan)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        ghi=ghi_grid,
        lons=tgt_lons.astype(np.float64),
        lats=tgt_lats.astype(np.float64),
        time_utc=f"{date}{slot}00",
        grid_step=np.float64(grid_step),
    )
    return out_path


def process_date(args: argparse.Namespace, date: str) -> None:
    slots = args.slots or FY3_FY4_SLOT_SPECS
    output_dir = args.output_dir.resolve()
    tgt_lons, tgt_lats = build_target_axes(args.grid_step)

    print(f"REST2 slot cache builder — {date}")
    print(f"  output: {output_dir}")
    print(f"  grid: {len(tgt_lons)}×{len(tgt_lats)} @ {args.grid_step}°")

    _install_bsrn_compat_modules()
    clear_sky = _load_module(PROJECT_ROOT / "clear_sky.py", "local_clear_sky")
    cube = _load_merra2_cube(date)

    written = 0
    skipped = 0
    for slot in slots:
        out_path = rest2_cache_path(output_dir, date, slot, args.grid_step)
        if args.skip_existing and out_path.exists():
            print(f"  [skip] {slot} — {out_path.name}")
            skipped += 1
            continue
        print(f"  [build] {slot}")
        build_slot_cache(
            date,
            slot,
            cube,
            tgt_lons,
            tgt_lats,
            clear_sky,
            output_dir,
            args.grid_step,
        )
        written += 1
    cube.close()
    print(f"Done: wrote {written}, skipped {skipped}, slots={len(slots)}")


def main() -> None:
    args = parse_args()
    dates = list(iter_dates(args.date, args.end_date))
    for idx, date in enumerate(dates, start=1):
        if len(dates) > 1:
            print(f"\n[{idx}/{len(dates)}] {date}")
        process_date(args, date)


if __name__ == "__main__":
    main()
