"""Plot thesis Figure 5-3: irradiance fields in the effective matched strip.

The fields are prepared through the same per-pixel half-open-window matching
and nearest-neighbour FY-4A resampling used in formal sample construction.  In
the selected west-side effective strip, the real 2018-05-10 outcome happens to
be one dominant 05:00 UTC slot; this is reported rather than replaced by an
artificial multi-time composite.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))
from matching_figure_data import (  # noqa: E402
    DATE,
    OUTPUT_DIR,
    build_matching_result,
    extent_slices,
    minute_label,
    rounded_irradiance_limits,
)


# This is also the Figure 5-4 CSI frame.  The 73.00-73.24°E edge is excluded
# because the corresponding REST2 clear-sky field is absent there; cropping it
# is preferable to displaying a non-physical white CSI seam.
ZOOM_EXTENT = (73.25, 85.0, 26.0, 54.0)


def decorate_map_axis(axis: plt.Axes, title: str) -> None:
    axis.set_title(title, fontsize=9.0, fontweight="normal", pad=4)
    axis.set_xlabel("Longitude (°E)", fontsize=9.0)
    axis.set_ylabel("Latitude (°N)", fontsize=9.0)
    axis.tick_params(labelsize=9.0, width=0.7, length=2.6)
    for spine in axis.spines.values():
        spine.set_linewidth(0.75)


def plot_figure(date: str) -> None:
    result = build_matching_result(date)
    row_slice, col_slice = extent_slices(result.lons, result.lats, ZOOM_EXTENT)
    ssi = result.matched_ssi[row_slice, col_slice]
    gc = result.gc[row_slice, col_slice]
    final_mask = result.final_mask[row_slice, col_slice]
    if not np.array_equal(np.isfinite(ssi), np.isfinite(gc)):
        raise AssertionError("Figure 5-3 panels do not have the same final valid mask.")

    coverage = float(final_mask.mean())
    if coverage < 0.99:
        raise AssertionError(f"Selected Figure 5-3 frame coverage is unexpectedly low: {coverage:.2%}")
    vmin, vmax = rounded_irradiance_limits(ssi, gc)
    zoom_extent = (
        float(result.lons[col_slice.start] - 0.005),
        float(result.lons[col_slice.stop - 1] + 0.005),
        float(result.lats[row_slice.stop - 1] - 0.005),
        float(result.lats[row_slice.start] + 0.005),
    )
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
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")

    # The valid matched swath is tall and narrow (73–85°E, 26–54°N).  A
    # compact portrait figure gives both maps their natural equal aspect ratio
    # without the large empty gap created by a wide landscape canvas.
    # This compact single-column source size is suitable for direct Word
    # insertion while preserving the requested 9 pt (small-five) labels.
    fig = plt.figure(figsize=(2.70, 3.05))
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=(1.0, 1.0, 0.055),
        left=0.195,
        right=0.865,
        bottom=0.155,
        top=0.875,
        wspace=0.08,
    )
    axes = (fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]))
    cax = fig.add_subplot(grid[0, 2])
    im = axes[0].imshow(
        np.ma.masked_invalid(ssi),
        extent=zoom_extent,
        origin="upper",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    axes[1].imshow(
        np.ma.masked_invalid(gc),
        extent=zoom_extent,
        origin="upper",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    for axis, title in (
        (axes[0], "(a) FY-4A SSI"),
        (axes[1], "(b) FY-3C Gc"),
    ):
        axis.set_xlim(ZOOM_EXTENT[0], ZOOM_EXTENT[1])
        axis.set_ylim(ZOOM_EXTENT[2], ZOOM_EXTENT[3])
        axis.set_aspect("equal", adjustable="box")
        # Both panels show exactly the same 73–85°E frame, so their longitude
        # ticks must be identical.  Two interior ticks keep the compact Word
        # version readable without creating duplicate labels at the centre gap.
        axis.set_xticks([75.0, 80.0])
        axis.set_yticks([30.0, 40.0, 50.0])
        decorate_map_axis(axis, title)

    # Both panels share the same latitude range.  Showing the latitude label
    # and tick labels once prevents the centre gap from becoming wider than the
    # maps themselves while preserving an unambiguous geographic reference.
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    axes[0].set_xlabel("")
    axes[1].set_xlabel("")
    fig.supxlabel("Longitude (°E)", fontsize=9.0, y=0.045)

    # Equal-aspect maps shrink inside their allocated axes.  Bind the colour
    # bar to the final map bounds so its length exactly matches the image.
    fig.canvas.draw()
    map_box = axes[1].get_position()
    bar_box = cax.get_position()
    cax.set_position((bar_box.x0, map_box.y0, bar_box.width, map_box.height))
    colorbar = fig.colorbar(im, cax=cax)
    colorbar.set_label("Solar irradiance (W m$^{-2}$)", fontsize=9.0, labelpad=4)
    colorbar.ax.tick_params(labelsize=9.0, width=0.65, length=2.5)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "Fig5-3_FY4A_FY3C_matched_irradiance.png"
    pdf_path = OUTPUT_DIR / "Fig5-3_FY4A_FY3C_matched_irradiance.pdf"
    fig.savefig(png_path, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    local_slot_counts = np.bincount(
        result.slot_map[row_slice, col_slice][final_mask], minlength=len(result.slots)
    )
    contributing = np.flatnonzero(local_slot_counts > 0)
    print("========== FIGURE 5-3 COMPLETE ==========")
    print(f"Date: {result.date}")
    print(f"Geographic frame: {ZOOM_EXTENT[0]:.0f}–{ZOOM_EXTENT[1]:.0f}°E, {ZOOM_EXTENT[2]:.0f}–{ZOOM_EXTENT[3]:.0f}°N")
    print(f"Strictly matched coverage in frame: {int(final_mask.sum()):,}/{final_mask.size:,} ({coverage:.2%})")
    print(f"Final common valid pixels: {int(final_mask.sum()):,}")
    print(f"Shared SSI/Gc colour range: {vmin:.0f}–{vmax:.0f} W m-2")
    print("FY-4A source files contributing inside the Figure 5-3 frame:")
    for index in contributing:
        slot = result.slots[index]
        print(
            f"  {minute_label(slot.start_minute_utc)} UTC: {int(local_slot_counts[index]):,} pixels | "
            f"{slot.source_nc}"
        )
    global_contributing = int((result.slot_counts > 0).sum())
    print(
        f"Per-pixel assembly check: passed; {global_contributing} FY-4A slots contribute globally, "
        f"and {contributing.size} contributes inside this local frame."
    )
    print("Common mask check for panels (a)/(b): passed.")
    print(f"PNG (400 dpi): {png_path}")
    print(f"PDF: {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=DATE)
    args = parser.parse_args()
    plot_figure(args.date)


if __name__ == "__main__":
    main()
