"""Plot thesis Figure 5-4: matched FY-4A and FY-3C clear-sky indices.

This is a read-only plotting script.  It reuses the formal per-pixel
time-matching result for 2018-05-10 and the corresponding REST2 fields.  Both
panels therefore use one identical CSI-valid mask; no interpolation, patching,
or super-resolution operation is performed here.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


FY_PROCESSING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FY_PROCESSING_ROOT))
from fy_io import OUTPUT_REST2_ON_FY3_DIR, rest2_on_fy3_path  # noqa: E402
from matching_figure_data import (  # noqa: E402
    DATE,
    OUTPUT_DIR,
    build_matching_result,
    extent_slices,
    minute_label,
)


# Deliberately identical to the geographic frame used for Figure 5-3.  The
# 73.00-73.24°E boundary has no valid REST2 GHI, so it is outside both display
# frames instead of appearing as a non-physical white CSI seam.
ZOOM_EXTENT = (73.25, 85.0, 26.0, 54.0)
CSI_MIN = 0.0
CSI_MAX = 1.5


def _configure_publication_style() -> None:
    """Use Word small-five (9 pt), Times New Roman and SimSun fallback."""
    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "SimSun"],
            "font.serif": ["Times New Roman", "SimSun"],
            "font.size": 9.0,
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _assemble_matched_rest2(
    result: object,
    row_slice: slice,
    col_slice: slice,
) -> tuple[np.ndarray, list[tuple[str, Path, int]]]:
    """Assemble REST2 GHI by the same pixel-to-slot map as FY-4A SSI."""
    slot_map = result.slot_map[row_slice, col_slice]
    clear_ghi = np.full(slot_map.shape, np.nan, dtype=np.float32)
    used_slots: list[tuple[str, Path, int]] = []
    rest2_dir = OUTPUT_REST2_ON_FY3_DIR / f"{result.date}_utc"

    for index, slot in enumerate(result.slots):
        selected = slot_map == index
        if not np.any(selected):
            continue
        path = rest2_on_fy3_path(rest2_dir, slot.label_utc)
        if not path.exists():
            raise FileNotFoundError(f"Missing REST2 field for matched slot: {path}")
        with np.load(path, allow_pickle=False) as archive:
            if "ghi" not in archive.files:
                raise KeyError(f"{path} is missing the GHI field.")
            ghi = np.asarray(archive["ghi"], dtype=np.float32)
        if ghi.shape != result.slot_map.shape:
            raise ValueError(f"{path.name}: incompatible FY-3C-grid shape {ghi.shape}.")
        local_ghi = ghi[row_slice, col_slice]
        usable = selected & np.isfinite(local_ghi) & (local_ghi > 0.0)
        clear_ghi[usable] = local_ghi[usable]
        used_slots.append((slot.label_utc, path, int(selected.sum())))
    return clear_ghi, used_slots


def _decorate_map_axis(axis: plt.Axes, title: str) -> None:
    axis.set_title(title, fontsize=9.0, fontweight="normal", pad=4)
    axis.tick_params(labelsize=9.0, width=0.7, length=2.6)
    for spine in axis.spines.values():
        spine.set_linewidth(0.75)


def plot_figure(date: str) -> None:
    """Generate Figure 5-4 from the existing formal matching products."""
    result = build_matching_result(date)
    row_slice, col_slice = extent_slices(result.lons, result.lats, ZOOM_EXTENT)
    fy4_ssi = result.matched_ssi[row_slice, col_slice]
    fy3_gc = result.gc[row_slice, col_slice]
    matching_mask = result.final_mask[row_slice, col_slice]
    clear_ghi, used_slots = _assemble_matched_rest2(result, row_slice, col_slice)

    fy4_csi = np.full(fy4_ssi.shape, np.nan, dtype=np.float32)
    fy3_csi = np.full(fy3_gc.shape, np.nan, dtype=np.float32)
    valid_denominator = np.isfinite(clear_ghi) & (clear_ghi > 0.0)
    np.divide(fy4_ssi, clear_ghi, out=fy4_csi, where=matching_mask & valid_denominator)
    np.divide(fy3_gc, clear_ghi, out=fy3_csi, where=matching_mask & valid_denominator)

    # Preserve the CSI quality range already used to build the real training
    # pairs.  This is a shared QC mask rather than a visual-only truncation.
    common_mask = (
        matching_mask
        & valid_denominator
        & np.isfinite(fy4_csi)
        & np.isfinite(fy3_csi)
        & (fy4_csi >= CSI_MIN)
        & (fy4_csi <= CSI_MAX)
        & (fy3_csi >= CSI_MIN)
        & (fy3_csi <= CSI_MAX)
    )
    fy4_csi[~common_mask] = np.nan
    fy3_csi[~common_mask] = np.nan
    if not np.array_equal(np.isfinite(fy4_csi), np.isfinite(fy3_csi)):
        raise AssertionError("Figure 5-4 CSI panels do not use an identical mask.")
    if not np.any(common_mask):
        raise RuntimeError("No CSI-valid pixels remain in the Figure 5-4 frame.")

    zoom_extent = (
        float(result.lons[col_slice.start] - 0.005),
        float(result.lons[col_slice.stop - 1] + 0.005),
        float(result.lats[row_slice.stop - 1] - 0.005),
        float(result.lats[row_slice.start] + 0.005),
    )
    _configure_publication_style()
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")

    # Compact single-column source size.  The only latitude labels and the
    # shared colour bar avoid a wide central gap while retaining equal map
    # aspect ratios.
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
    image = axes[0].imshow(
        np.ma.masked_invalid(fy4_csi),
        extent=zoom_extent,
        origin="upper",
        interpolation="nearest",
        cmap=cmap,
        vmin=CSI_MIN,
        vmax=CSI_MAX,
        rasterized=True,
    )
    axes[1].imshow(
        np.ma.masked_invalid(fy3_csi),
        extent=zoom_extent,
        origin="upper",
        interpolation="nearest",
        cmap=cmap,
        vmin=CSI_MIN,
        vmax=CSI_MAX,
        rasterized=True,
    )
    for axis, title in ((axes[0], "(a) FY-4A CSI"), (axes[1], "(b) FY-3C CSI")):
        axis.set_xlim(ZOOM_EXTENT[0], ZOOM_EXTENT[1])
        axis.set_ylim(ZOOM_EXTENT[2], ZOOM_EXTENT[3])
        axis.set_aspect("equal", adjustable="box")
        axis.set_xticks([75.0, 80.0])
        axis.set_yticks([30.0, 40.0, 50.0])
        _decorate_map_axis(axis, title)
    axes[0].set_ylabel("Latitude (°N)", fontsize=9.0)
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    axes[0].set_xlabel("")
    axes[1].set_xlabel("")
    fig.supxlabel("Longitude (°E)", fontsize=9.0, y=0.045)

    # Bind the colour bar to the final rendered map height, not the pre-aspect
    # GridSpec box, so it remains exactly as long as the two images.
    fig.canvas.draw()
    map_box = axes[1].get_position()
    bar_box = cax.get_position()
    cax.set_position((bar_box.x0, map_box.y0, bar_box.width, map_box.height))
    colour_bar = fig.colorbar(image, cax=cax, ticks=[0.0, 0.5, 1.0, 1.5])
    colour_bar.set_label("Clear-sky index (-)", fontsize=9.0, labelpad=4)
    colour_bar.ax.tick_params(labelsize=9.0, width=0.65, length=2.5)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "Fig5-4_FY4A_FY3C_csi.png"
    pdf_path = OUTPUT_DIR / "Fig5-4_FY4A_FY3C_csi.pdf"
    fig.savefig(png_path, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    matching_count = int(matching_mask.sum())
    common_count = int(common_mask.sum())
    print("========== FIGURE 5-4 COMPLETE ==========")
    print(f"Date: {result.date}")
    print(
        "Geographic frame: "
        f"{ZOOM_EXTENT[0]:.0f}-{ZOOM_EXTENT[1]:.0f}°E, "
        f"{ZOOM_EXTENT[2]:.0f}-{ZOOM_EXTENT[3]:.0f}°N"
    )
    print(f"Figure 5-3 matching mask in frame: {matching_count:,}/{matching_mask.size:,}")
    print(f"Final common CSI mask: {common_count:,}/{matching_mask.size:,} ({common_mask.mean():.2%})")
    print(f"CSI QC range: [{CSI_MIN:.1f}, {CSI_MAX:.1f}]")
    print("REST2 fields selected by the same per-pixel FY-4A slot map:")
    for label, path, selected_count in used_slots:
        print(f"  {minute_label(int(label[:2]) * 60 + int(label[2:]))} UTC: {selected_count:,} pixels | {path}")
    print("FY-4A/FY-3C CSI common-mask check: passed.")
    print(f"PNG (400 dpi): {png_path}")
    print(f"PDF: {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=DATE)
    args = parser.parse_args()
    plot_figure(args.date)


if __name__ == "__main__":
    main()
