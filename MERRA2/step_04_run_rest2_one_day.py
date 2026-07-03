"""
Run REST2 clear-sky radiation for one day and one point using local MERRA-2 nc4 files.

Default test case:
    date = 2018-01-01
    lat  = 47.80
    lon  = 124.49
    elev = 0 m

Outputs:
    rest2_outputs/rest2_20180101_lat47.80_lon124.49.csv
    rest2_outputs/rest2_20180101_lat47.80_lon124.49.png
"""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

DATE = "2018-01-01"
LAT = 47.80
LON = 124.49
ELEV = 0.0

OUT_DIR = ROOT / "rest2_outputs"


def _get_bni_extra(index: pd.DatetimeIndex, solar_constant: float = 1366.1) -> pd.Series:
    """Approximate extraterrestrial normal irradiance, matching common solar geometry usage."""
    idx = pd.DatetimeIndex(index)
    doy = idx.dayofyear.to_numpy(dtype=float)
    day_angle = 2.0 * np.pi * (doy - 1.0) / 365.0
    eccentricity = (
        1.00011
        + 0.034221 * np.cos(day_angle)
        + 0.00128 * np.sin(day_angle)
        + 0.000719 * np.cos(2.0 * day_angle)
        + 0.000077 * np.sin(2.0 * day_angle)
    )
    return pd.Series(solar_constant * eccentricity, index=idx)


def _solar_zenith_noaa(index: pd.DatetimeIndex, lat: float, lon: float) -> pd.Series:
    """NOAA-style solar zenith approximation for UTC timestamps."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")

    day_of_year = idx.dayofyear.to_numpy(dtype=float)
    fractional_hour = (
        idx.hour.to_numpy(dtype=float)
        + idx.minute.to_numpy(dtype=float) / 60.0
        + idx.second.to_numpy(dtype=float) / 3600.0
    )
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

    true_solar_time = (fractional_hour * 60.0 + eqtime + 4.0 * lon) % 1440.0
    hour_angle = np.radians(true_solar_time / 4.0 - 180.0)
    lat_rad = math.radians(lat)

    cos_zenith = (
        math.sin(lat_rad) * np.sin(decl)
        + math.cos(lat_rad) * np.cos(decl) * np.cos(hour_angle)
    )
    zenith = np.degrees(np.arccos(np.clip(cos_zenith, -1.0, 1.0)))
    return pd.Series(zenith, index=idx)


def _install_bsrn_compat_modules() -> None:
    """
    The reference files in this workspace are standalone copies that import the original
    package namespace. Install tiny compatibility modules so we can reuse their functions.
    """
    geometry = types.ModuleType("bsrn.physics.geometry")
    geometry.get_bni_extra = _get_bni_extra

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


def _file_for(product: str, date: str) -> Path:
    day = date.replace("-", "")
    return ROOT / product / f"MERRA2_400.{product}.{day}.SUB.nc4"


def _read_point_series(path: Path, variables: dict[str, str], lat: float, lon: float) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    ds = xr.open_dataset(path, engine="netcdf4")
    try:
        point = ds.sel(lat=lat, lon=lon, method="nearest")
        nearest_lat = float(point["lat"].values)
        nearest_lon = float(point["lon"].values)
        data = {}
        for source_name, target_name in variables.items():
            if source_name not in point:
                raise KeyError(f"{source_name} not found in {path.name}")
            data[target_name] = np.asarray(point[source_name].to_numpy(), dtype=float)

        index = pd.to_datetime(point["time"].values)
        index = pd.DatetimeIndex(index).tz_localize("UTC")
        df = pd.DataFrame(data, index=index)
        df.attrs["nearest_lat"] = nearest_lat
        df.attrs["nearest_lon"] = nearest_lon
        return df
    finally:
        ds.close()


def _load_local_merra2_raw(date: str, lat: float, lon: float) -> pd.DataFrame:
    aer = _read_point_series(
        _file_for("tavg1_2d_aer_Nx", date),
        {"TOTEXTTAU": "AOD55", "TOTANGSTR": "ALPHA"},
        lat,
        lon,
    )
    rad = _read_point_series(
        _file_for("tavg1_2d_rad_Nx", date),
        {"ALBEDO": "ALBEDO"},
        lat,
        lon,
    )
    slv = _read_point_series(
        _file_for("tavg1_2d_slv_Nx", date),
        {"PS": "PS", "TO3": "TO3", "TQV": "TQV"},
        lat,
        lon,
    )

    raw = pd.concat([aer, rad, slv], axis=1).sort_index()
    raw = raw[~raw.index.duplicated(keep="first")]
    raw.attrs["nearest_lat"] = aer.attrs["nearest_lat"]
    raw.attrs["nearest_lon"] = aer.attrs["nearest_lon"]
    return raw


def main() -> None:
    _install_bsrn_compat_modules()
    clear_sky = _load_module(PROJECT_ROOT / "clear_sky.py", "local_clear_sky")
    merra2 = _load_module(PROJECT_ROOT / "merra2.py", "local_merra2")

    target_index = pd.date_range(
        DATE,
        periods=24 * 60,
        freq="min",
        tz="UTC",
    )

    raw = _load_local_merra2_raw(DATE, LAT, LON)
    rest2_inputs = merra2._parse_merra2_for_rest2(raw, target_index)
    zenith = _solar_zenith_noaa(target_index, LAT, LON)
    ghi_clear, bni_clear, dhi_clear = clear_sky.rest2_model(
        target_index,
        zenith,
        rest2_inputs,
    )

    result = rest2_inputs.copy()
    result.insert(0, "zenith", zenith.to_numpy(dtype=float))
    result["ghi_clear"] = ghi_clear
    result["bni_clear"] = bni_clear
    result["dhi_clear"] = dhi_clear

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"rest2_{DATE.replace('-', '')}_lat{LAT:.2f}_lon{LON:.2f}"
    csv_path = OUT_DIR / f"{stem}.csv"
    png_path = OUT_DIR / f"{stem}.png"

    result.to_csv(csv_path, index_label="time_utc")

    fig, ax = plt.subplots(figsize=(11, 5))
    result[["ghi_clear", "bni_clear", "dhi_clear"]].plot(ax=ax)
    ax.set_title(f"REST2 clear-sky irradiance {DATE} ({LAT:.2f}, {LON:.2f})")
    ax.set_xlabel("UTC time")
    ax.set_ylabel("Irradiance (W/m^2)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    daytime = result["zenith"] < 90.0
    print(f"Raw MERRA-2 nearest grid: lat={raw.attrs['nearest_lat']:.3f}, lon={raw.attrs['nearest_lon']:.3f}")
    print(f"Rows: {len(result)}")
    print(f"Daytime rows: {int(daytime.sum())}")
    print(f"CSV: {csv_path}")
    print(f"PNG: {png_path}")
    print("Peak irradiance:")
    print(f"  ghi_clear: {result['ghi_clear'].max():.2f} W/m^2")
    print(f"  bni_clear: {result['bni_clear'].max():.2f} W/m^2")
    print(f"  dhi_clear: {result['dhi_clear'].max():.2f} W/m^2")


if __name__ == "__main__":
    main()
