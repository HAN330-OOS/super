"""Plot thesis Figure 5-5 from one real FY-4A/FY-3C CSI training pair.

The two arrays are read directly from the formal sample archive.  The left
panel is the saved 32 x 32 FY-4A CSI input and the right panel is the saved
128 x 128 FY-3C CSI target for precisely the same geographic footprint.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FY_PROCESSING_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = FY_PROCESSING_ROOT.parent
SAMPLES_ROOT = PROJECT_ROOT / "outputs" / "fy" / "samples"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "fy" / "figures"
SAMPLE_DATE = "20180510"
SAMPLE_ID = 2000
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


def _find_sample(date: str, sample_id: int) -> tuple[dict[str, str], Path]:
    """Resolve a stored sample from the real CSV index without modifying it."""
    sample_dir = SAMPLES_ROOT / f"{date}_utc"
    index_path = sample_dir / "sample_index.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing formal sample index: {index_path}")
    with index_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["sample_id"]) == sample_id:
                archive_path = sample_dir / "patches" / row["batch_file"]
                if not archive_path.exists():
                    raise FileNotFoundError(f"Missing sample archive: {archive_path}")
                return row, archive_path
    raise LookupError(f"Sample ID {sample_id} is not present in {index_path}")


def _load_real_pair(date: str, sample_id: int) -> tuple[np.ndarray, np.ndarray, dict[str, str], Path]:
    """Read the original, unnormalised CSI arrays stored by step_05."""
    row, archive_path = _find_sample(date, sample_id)
    batch_index = int(row["batch_index"])
    with np.load(archive_path, allow_pickle=False) as archive:
        required = {"lr_csi", "hr_csi"}
        missing = required - set(archive.files)
        if missing:
            raise KeyError(f"{archive_path} is missing {sorted(missing)}")
        lr_csi = np.asarray(archive["lr_csi"][batch_index], dtype=np.float32)
        hr_csi = np.asarray(archive["hr_csi"][batch_index], dtype=np.float32)
    if lr_csi.shape != (32, 32) or hr_csi.shape != (128, 128):
        raise ValueError(f"Unexpected pair shapes: LR {lr_csi.shape}, HR {hr_csi.shape}")
    valid_pair = (
        np.isfinite(lr_csi).all()
        and np.isfinite(hr_csi).all()
        and CSI_MIN <= float(lr_csi.min()) <= float(lr_csi.max()) <= CSI_MAX
        and CSI_MIN <= float(hr_csi.min()) <= float(hr_csi.max()) <= CSI_MAX
    )
    if not valid_pair:
        raise ValueError("Selected sample is not a valid CSI pair within the formal [0, 1.5] QC range.")
    return lr_csi, hr_csi, row, archive_path


def plot_figure(date: str, sample_id: int) -> None:
    """Generate the compact two-panel real training-sample figure."""
    lr_csi, hr_csi, metadata, archive_path = _load_real_pair(date, sample_id)
    _configure_publication_style()
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")

    # The compact canvas is deliberately similar in physical size to the map
    # figures.  There are no geographic axes because both panels represent the
    # same stored patch footprint; their original pixel dimensions are stated
    # in the panel titles.
    fig = plt.figure(figsize=(4.25, 2.25))
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=(1.0, 1.0, 0.060),
        left=0.075,
        right=0.865,
        bottom=0.095,
        top=0.760,
        wspace=0.12,
    )
    axes = (fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]))
    cax = fig.add_subplot(grid[0, 2])
    image = axes[0].imshow(
        lr_csi,
        cmap=cmap,
        vmin=CSI_MIN,
        vmax=CSI_MAX,
        interpolation="nearest",
        resample=False,
        rasterized=True,
    )
    axes[1].imshow(
        hr_csi,
        cmap=cmap,
        vmin=CSI_MIN,
        vmax=CSI_MAX,
        interpolation="nearest",
        resample=False,
        rasterized=True,
    )
    for axis, title in (
        (axes[0], "(a) FY-4A CSI\n(32 × 32)"),
        (axes[1], "(b) FY-3C CSI\n(128 × 128)"),
    ):
        axis.set_title(title, fontsize=9.0, fontweight="normal", pad=4, linespacing=0.9)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_aspect("equal", adjustable="box")
        for spine in axis.spines.values():
            spine.set_linewidth(0.75)

    # As in Figure 5-4, use the rendered image bounds to keep the colour bar
    # exactly the same height as the two square sample panels.
    fig.canvas.draw()
    map_box = axes[1].get_position()
    bar_box = cax.get_position()
    cax.set_position((bar_box.x0, map_box.y0, bar_box.width, map_box.height))
    colour_bar = fig.colorbar(image, cax=cax, ticks=[0.0, 0.5, 1.0, 1.5])
    colour_bar.set_label("Clear-sky index (-)", fontsize=9.0, labelpad=3)
    colour_bar.ax.tick_params(labelsize=9.0, width=0.65, length=2.5)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "Fig5-5_FY4A_FY3C_training_sample.png"
    pdf_path = OUTPUT_DIR / "Fig5-5_FY4A_FY3C_training_sample.pdf"
    fig.savefig(png_path, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("========== FIGURE 5-5 COMPLETE ==========")
    print("Stored real paired CSI sample (no FY-3C artificial downsampling):")
    print(f"  Date / sample ID: {date} / {sample_id}")
    print(f"  Source archive: {archive_path}")
    print(f"  Archive batch index: {metadata['batch_index']}")
    print(
        "  Centre: "
        f"{float(metadata['center_lon']):.2f}°E, {float(metadata['center_lat']):.2f}°N | "
        f"FY-4A slot {metadata['dominant_slot_label']} UTC"
    )
    print(f"  LR / HR shapes: {lr_csi.shape} / {hr_csi.shape}")
    print(
        "  LR CSI range: "
        f"{float(lr_csi.min()):.3f}-{float(lr_csi.max()):.3f}; "
        f"HR CSI range: {float(hr_csi.min()):.3f}-{float(hr_csi.max()):.3f}"
    )
    print("  Pair QC and real-source check: passed.")
    print(f"PNG (400 dpi): {png_path}")
    print(f"PDF: {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=SAMPLE_DATE)
    parser.add_argument("--sample-id", type=int, default=SAMPLE_ID)
    args = parser.parse_args()
    plot_figure(args.date, args.sample_id)


if __name__ == "__main__":
    main()
