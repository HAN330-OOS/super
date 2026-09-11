"""Plot the raw-product illustration used as Figure 5-1.

This script deliberately stays before the matching and super-resolution stages:
it reads a native FY-4A AGRI SSI product and a native FY-3C Gc grid only.  It
does not estimate FY-3C pixel time, construct CSI, call REST2, resample either
field, or read any training sample/checkpoint.

The figure always covers the complete study region (73–135°E, 18–54°N), so it
can show the native FY-3C coverage/strip features as well as the spatial-scale
difference between the two products.  Use ``--date`` only to change the
same-day raw-product example.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FY_PROCESSING_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = FY_PROCESSING_ROOT.parent
sys.path.insert(0, str(FY_PROCESSING_ROOT))

from fy_io import (  # noqa: E402
    DEFAULT_FY3_ROOT,
    DEFAULT_FY4_ROOT,
    SSI_VALID_MAX,
    SSI_VALID_MIN,
    list_fy4_nc_files,
    read_fy3_binary,
    read_fy4_lon_lat,
    read_fy4_ssi_with_dqf,
    region_slice,
)


TARGET_FY4_UTC_MINUTE = 2 * 60
DEFAULT_DATE = "20180510"
STUDY_EXTENT = (73.0, 135.0, 18.0, 54.0)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "fy" / "figures"


def clean_irradiance(field: np.ndarray) -> np.ndarray:
    """Keep only finite, physically valid irradiance values in W m-2."""
    clean = np.asarray(field, dtype=np.float32).copy()
    invalid = (~np.isfinite(clean)) | (clean < SSI_VALID_MIN) | (clean > SSI_VALID_MAX)
    clean[invalid] = np.nan
    return clean


def crop_fy3(
    fy3: np.ndarray, extent: tuple[float, float, float, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop the native FY-3C 0.01° grid without resampling it."""
    lon_min, lon_max, lat_min, lat_max = extent
    row_start, row_end, col_start, col_end = region_slice(lon_min, lon_max, lat_min, lat_max)
    data = fy3[row_start:row_end, col_start:col_end]
    lons = 73.0 + np.arange(col_start, col_end, dtype=np.float32) * 0.01
    lats = 54.0 - np.arange(row_start, row_end, dtype=np.float32) * 0.01
    return data, lons, lats


def crop_fy4(
    fy4: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    extent: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop the native FY-4A geostationary grid; no interpolation is performed."""
    lon_min, lon_max, lat_min, lat_max = extent
    in_extent = (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)
    rows = np.flatnonzero(np.any(in_extent, axis=1))
    cols = np.flatnonzero(np.any(in_extent, axis=0))
    if rows.size == 0 or cols.size == 0:
        raise ValueError(f"FY-4A geolocation has no native pixels in extent {extent}.")
    row_slice = slice(rows[0], rows[-1] + 1)
    col_slice = slice(cols[0], cols[-1] + 1)
    data = fy4[row_slice, col_slice].copy()
    lon_crop = lon[row_slice, col_slice]
    lat_crop = lat[row_slice, col_slice]
    data[~in_extent[row_slice, col_slice]] = np.nan
    return data, lon_crop, lat_crop


def nearest_daytime_fy4_file(date: str) -> tuple[Path, int]:
    """Select the available FY-4A file closest to 02:00 UTC on this date."""
    entries = list_fy4_nc_files(date, DEFAULT_FY4_ROOT)
    return min(entries, key=lambda item: abs(item[1] - TARGET_FY4_UTC_MINUTE))


def load_scene(
    date: str, extent: tuple[float, float, float, float]
) -> tuple[Path, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and crop one raw FY-4A/FY-3C pair without any matching operation."""
    fy3_path = DEFAULT_FY3_ROOT / f"Gc_{date}.dat"
    if not fy3_path.exists():
        raise FileNotFoundError(f"FY-3C Gc file does not exist: {fy3_path}")
    fy4_path, fy4_utc_minute = nearest_daytime_fy4_file(date)

    fy4_lon, fy4_lat = read_fy4_lon_lat()
    fy4 = clean_irradiance(read_fy4_ssi_with_dqf(fy4_path)[0])
    fy3 = clean_irradiance(read_fy3_binary(fy3_path))
    fy4_crop, fy4_lon_crop, fy4_lat_crop = crop_fy4(fy4, fy4_lon, fy4_lat, extent)
    fy3_crop, fy3_lons, fy3_lats = crop_fy3(fy3, extent)
    return (
        fy4_path,
        fy4_utc_minute,
        fy4_crop,
        fy4_lon_crop,
        fy4_lat_crop,
        fy3_crop,
        fy3_lons,
        fy3_lats,
    )


def rounded_common_limits(fy4: np.ndarray, fy3: np.ndarray) -> tuple[float, float]:
    """Derive one common physical colour range from all valid raw pixels."""
    values = np.concatenate((fy4[np.isfinite(fy4)], fy3[np.isfinite(fy3)]))
    if values.size == 0:
        raise RuntimeError("No valid values remain after basic product quality control.")
    vmin = float(math.floor(float(np.min(values)) / 50.0) * 50.0)
    vmax = float(math.ceil(float(np.max(values)) / 50.0) * 50.0)
    if vmin == vmax:
        vmax = vmin + 50.0
    return vmin, vmax


def centre_grid_to_cell_edges(centre_grid: np.ndarray) -> np.ndarray:
    """Approximate geolocation cell edges from native FY-4A cell centres.

    This only provides the quadrilateral boundaries that ``pcolormesh`` needs
    for its native FY-4A pixels.  The SSI data array itself is not interpolated
    or changed.
    """
    padded = np.pad(np.asarray(centre_grid, dtype=np.float64), 1, mode="edge")
    return 0.25 * (
        padded[:-1, :-1]
        + padded[1:, :-1]
        + padded[:-1, 1:]
        + padded[1:, 1:]
    )


def fill_nonfinite_plot_coordinates(coordinates: np.ndarray) -> np.ndarray:
    """Fill coordinates only for already-masked, off-disk FY-4A cells.

    The full study-region bounding rectangle includes a few FY-4A disk-exterior
    cells.  Their SSI values are masked in :func:`crop_fy4`, but Matplotlib
    requires finite coordinates even for invisible cells.  This function fills
    those coordinate placeholders by one-dimensional nearest extrapolation;
    it never changes any SSI value or makes an off-disk pixel visible.
    """
    filled = np.asarray(coordinates, dtype=np.float64).copy()
    column_index = np.arange(filled.shape[1])
    for row in range(filled.shape[0]):
        valid = np.isfinite(filled[row])
        if np.any(valid):
            filled[row] = np.interp(column_index, column_index[valid], filled[row, valid])

    row_index = np.arange(filled.shape[0])
    for column in range(filled.shape[1]):
        valid = np.isfinite(filled[:, column])
        if np.any(valid):
            filled[:, column] = np.interp(row_index, row_index[valid], filled[valid, column])
    if not np.all(np.isfinite(filled)):
        raise ValueError("FY-4A plot-coordinate completion failed for the selected extent.")
    return filled


def format_time(minute: int) -> str:
    """Format a UTC minute-of-day as HH:MM UTC."""
    return f"{minute // 60:02d}:{minute % 60:02d} UTC"


def draw_figure(
    fy4: np.ndarray,
    fy4_lon: np.ndarray,
    fy4_lat: np.ndarray,
    fy3: np.ndarray,
    fy3_lons: np.ndarray,
    fy3_lats: np.ndarray,
    extent: tuple[float, float, float, float],
    png_path: Path,
    pdf_path: Path,
) -> tuple[float, float]:
    """Draw aligned native-resolution map panels with one shared colour scale."""
    plt.rcParams.update(
        {
            # 9 pt is Word small-five.  English uses Times New Roman; Chinese
            # glyphs fall back to the locally installed SimSun (宋体).
            "font.family": ["Times New Roman", "SimSun"],
            "font.serif": ["Times New Roman", "SimSun"],
            "font.size": 9.0,
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    vmin, vmax = rounded_common_limits(fy4, fy3)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")

    # A right-side shared colour bar keeps the two raw-product maps compact.
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.30))
    fig.subplots_adjust(left=0.085, right=0.895, bottom=0.190, top=0.880, wspace=0.18)
    lon_min, lon_max, lat_min, lat_max = extent
    image = None
    # FY-4A uses its native two-dimensional geolocation grid.  FY-3C is a
    # regular 0.01° grid; ``nearest`` displays the raw cells without creating a
    # resampled data array, which keeps a full-domain 6200 × 3600 plot tractable.
    image = axes[0].pcolormesh(
        centre_grid_to_cell_edges(fill_nonfinite_plot_coordinates(fy4_lon)),
        centre_grid_to_cell_edges(fill_nonfinite_plot_coordinates(fy4_lat)),
        np.ma.masked_invalid(fy4),
        shading="flat",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    fy3_cell_extent = (
        float(fy3_lons[0] - 0.005),
        float(fy3_lons[-1] + 0.005),
        float(fy3_lats[-1] - 0.005),
        float(fy3_lats[0] + 0.005),
    )
    axes[1].imshow(
        np.ma.masked_invalid(fy3),
        extent=fy3_cell_extent,
        origin="upper",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    for axis, title in zip(axes, ("(a) FY-4A SSI", "(b) FY-3C Gc")):
        axis.set_xlim(lon_min, lon_max)
        axis.set_ylim(lat_min, lat_max)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(title, fontsize=9.0, fontweight="normal", pad=4)
        axis.set_xlabel("Longitude (°E)", fontsize=9.0)
        axis.set_ylabel("Latitude (°N)", fontsize=9.0)
        axis.tick_params(labelsize=9.0, width=0.75, length=3.0)
        axis.set_xticks(np.arange(math.ceil(lon_min / 10.0) * 10.0, lon_max + 0.1, 10.0))
        axis.set_yticks(np.arange(math.ceil(lat_min / 5.0) * 5.0, lat_max + 0.1, 5.0))
        for spine in axis.spines.values():
            spine.set_linewidth(0.8)

    # The two maps use identical latitude limits, so show the ordinate once,
    # following the compact multi-panel layout used in the reference figure.
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="y", left=False, labelleft=False)

    # Obtain the final equal-aspect map bounds first, then match the colour
    # bar height exactly to the map height rather than to its enclosing axes.
    fig.canvas.draw()
    map_box = axes[1].get_position()
    cax = fig.add_axes((map_box.x1 + 0.014, map_box.y0, 0.014, map_box.height))
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label("Solar irradiance (W m$^{-2}$)", fontsize=9.0, labelpad=4)
    colorbar.ax.tick_params(labelsize=9.0, width=0.75, length=2.8, pad=1.5)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return vmin, vmax


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date",
        default=DEFAULT_DATE,
        help=f"FY-3C daily Gc date (YYYYMMDD; default: {DEFAULT_DATE}).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    date = args.date
    extent = STUDY_EXTENT
    print("\nFull-study-region raw-product example:")
    print(f"  date: {date}")
    print(f"  geographic extent: {extent[0]:.1f}–{extent[1]:.1f}°E, {extent[2]:.1f}–{extent[3]:.1f}°N")

    (
        fy4_path,
        fy4_utc_minute,
        fy4,
        fy4_lon,
        fy4_lat,
        fy3,
        fy3_lons,
        fy3_lats,
    ) = load_scene(date, extent)

    png_path = OUTPUT_DIR / "Fig5-1_FY4A_FY3C_products.png"
    pdf_path = OUTPUT_DIR / "Fig5-1_FY4A_FY3C_products.pdf"
    vmin, vmax = draw_figure(
        fy4,
        fy4_lon,
        fy4_lat,
        fy3,
        fy3_lons,
        fy3_lats,
        extent,
        png_path,
        pdf_path,
    )

    fy3_path = DEFAULT_FY3_ROOT / f"Gc_{date}.dat"
    print("\n========== FIGURE 5-1 COMPLETE ==========")
    print(f"FY-4A source file: {fy4_path}")
    print(f"FY-4A nominal time: {format_time(fy4_utc_minute)}")
    print(f"FY-3C source file: {fy3_path}")
    print("FY-3C time: daily Gc grid; no per-pixel overpass-time estimate was applied.")
    print(f"Shared geographic extent: {extent[0]:.1f}–{extent[1]:.1f}°E, {extent[2]:.1f}–{extent[3]:.1f}°N")
    print(f"Shared colour range: {vmin:.0f}–{vmax:.0f} W m-2")
    print("Data treatment: native grids, basic invalid-value masking only; no matching/resampling/CSI/REST2/SR.")
    print(f"PNG (400 dpi): {png_path}")
    print(f"PDF: {pdf_path}")


if __name__ == "__main__":
    main()
