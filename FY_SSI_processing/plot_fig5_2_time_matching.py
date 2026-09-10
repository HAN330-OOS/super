"""Plot thesis Figure 5-2: FY-3C observation time and FY-4A slot matching.

The figure intentionally preserves white pixels with no actual FY-4A 15-minute
half-open-window match.  They document source-data time coverage, not a
plotting defect.  The black frame indicates the effective matched strip shown
in Figure 5-3.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))
from matching_figure_data import (  # noqa: E402
    DATE,
    OUTPUT_DIR,
    build_matching_result,
    cell_extent,
    minute_label,
)


ZOOM_EXTENT = (73.0, 85.0, 26.0, 54.0)


def decorate_map_axis(axis: plt.Axes, title: str) -> None:
    axis.set_title(title, fontsize=9.0, fontweight="normal", pad=4)
    axis.set_xlabel("Longitude (°E)", fontsize=9.0)
    axis.set_ylabel("Latitude (°N)", fontsize=9.0)
    axis.set_xlim(73.0, 135.0)
    axis.set_ylim(18.0, 54.0)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks(np.arange(80.0, 136.0, 10.0))
    axis.set_yticks(np.arange(20.0, 55.0, 10.0))
    axis.tick_params(labelsize=9.0, width=0.7, length=2.8)
    for spine in axis.spines.values():
        spine.set_linewidth(0.75)


def hour_formatter(value: float, _position: int) -> str:
    return minute_label(value * 60.0)


def plot_figure(date: str) -> None:
    result = build_matching_result(date)
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
    extent = cell_extent(result.lons, result.lats)
    time_values = result.time_utc_hours[np.isfinite(result.time_utc_hours)]
    time_min = math.floor(float(time_values.min()) * 4.0) / 4.0
    time_max = math.ceil(float(time_values.max()) * 4.0) / 4.0
    # Represent the matched 15 min slot by its actual UTC start time.  Panel
    # (b) remains discrete in value, while it can use the same physical time
    # scale as panel (a), so the figure needs only one shared colour bar.
    matched_slot_time_hours = np.full(result.slot_map.shape, np.nan, dtype=np.float32)
    for slot_index, slot in enumerate(result.slots):
        matched_slot_time_hours[result.slot_map == slot_index] = slot.start_minute_utc / 60.0

    time_cmap = plt.get_cmap("viridis").copy()
    time_cmap.set_bad("white")

    # The two panels use one common UTC colour scale and a single right-side
    # colour bar, matching the compact shared-scale reference layout.
    fig = plt.figure(figsize=(7.25, 3.30))
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=(1.0, 1.0, 0.052),
        left=0.075,
        right=0.940,
        bottom=0.190,
        top=0.880,
        wspace=0.08,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    cax = fig.add_subplot(grid[0, 2])

    im_a = ax_a.imshow(
        np.ma.masked_invalid(result.time_utc_hours),
        extent=extent,
        origin="upper",
        interpolation="nearest",
        cmap=time_cmap,
        vmin=time_min,
        vmax=time_max,
        rasterized=True,
    )
    im_b = ax_b.imshow(
        np.ma.masked_invalid(matched_slot_time_hours),
        extent=extent,
        origin="upper",
        interpolation="nearest",
        cmap=time_cmap,
        vmin=time_min,
        vmax=time_max,
        rasterized=True,
    )
    for axis, title in (
        (ax_a, "(a) FY-3C pixel observation time"),
        (ax_b, "(b) Matched FY-4A time slot"),
    ):
        decorate_map_axis(axis, title)
        lon_min, lon_max, lat_min, lat_max = ZOOM_EXTENT
        axis.add_patch(
            Rectangle(
                (lon_min, lat_min),
                lon_max - lon_min,
                lat_max - lat_min,
                fill=False,
                edgecolor="black",
                linewidth=0.9,
                linestyle="-",
                zorder=5,
            )
        )

    fig.canvas.draw()
    map_box_b = ax_b.get_position()
    bar_box = cax.get_position()
    cax.set_position((bar_box.x0, map_box_b.y0, bar_box.width, map_box_b.height))

    colorbar = fig.colorbar(
        im_a,
        cax=cax,
        ticks=np.array([0.0, 2.0, 4.0, 6.0]),
    )
    colorbar.set_label("UTC time", fontsize=9.0, labelpad=4)
    colorbar.ax.yaxis.set_major_formatter(FuncFormatter(hour_formatter))
    colorbar.ax.tick_params(labelsize=9.0, width=0.65, length=2.3, pad=1.0)

    # The two maps have identical latitude limits.  Retaining latitude labels
    # only once prevents them competing with the central right-side colour bar.
    ax_b.set_ylabel("")
    ax_b.tick_params(axis="y", left=False, labelleft=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "Fig5-2_FY3C_observation_time_matching.png"
    pdf_path = OUTPUT_DIR / "Fig5-2_FY3C_observation_time_matching.pdf"
    fig.savefig(png_path, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("========== FIGURE 5-2 COMPLETE ==========")
    print(f"Date: {result.date}")
    print(f"Timestamp product: {result.timestamp_path}")
    print(f"Morning prior: {result.prior_label}")
    print(f"FY-3C original valid Sz pixels: {result.valid_sz_count:,}")
    print(
        f"Sz residual < 0.5 deg: {result.qc_valid_count:,} "
        f"({100.0 * result.qc_valid_count / result.valid_sz_count:.3f}%)"
    )
    print(f"Estimated UTC range: {minute_label(time_values.min() * 60.0)}–{minute_label(time_values.max() * 60.0)}")
    print(f"Matched FY-4A-window pixels: {int(result.slot_counts.sum()):,}")
    print(f"No actual FY-4A-window match: {result.unmatched_count:,}")
    print(f"Figure 5-3 frame: {ZOOM_EXTENT[0]:.0f}–{ZOOM_EXTENT[1]:.0f}°E, {ZOOM_EXTENT[2]:.0f}–{ZOOM_EXTENT[3]:.0f}°N")
    print("Matched FY-4A slots:")
    for index, slot in enumerate(result.slots):
        print(f"  {minute_label(slot.start_minute_utc)} UTC: {int(result.slot_counts[index]):,} pixels")
    print(f"PNG (400 dpi): {png_path}")
    print(f"PDF: {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=DATE)
    args = parser.parse_args()
    plot_figure(args.date)


if __name__ == "__main__":
    main()
