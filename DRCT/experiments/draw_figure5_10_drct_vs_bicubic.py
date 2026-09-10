"""Draw Figure 5-10: FY-3C, DRCT and bicubic CSI for fixed test samples."""

from __future__ import annotations

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
SAMPLES_ROOT = (
    PROJECT_ROOT / "outputs" / "fy" / "samples"
    if (PROJECT_ROOT / "outputs" / "fy" / "samples").exists()
    else WORKSPACE_ROOT / "outputs" / "fy" / "samples"
)
RUN_DIR = PROJECT_ROOT / "results" / "fy4a_fy3c_drctl_w8_scratch_50k_b16_e50"
DATASET_ROOT = PROJECT_ROOT / "datasets" / "fy4a_fy3c_csi_x4"
CHECKPOINT = RUN_DIR / "checkpoints" / "epoch_271_converged_final.pth"
TEST_SUBSET = DATASET_ROOT / "test" / "samples.json"
OUTPUT = RUN_DIR / "final" / "paper_figures" / "figure5_10_fy3c_drct_bicubic_samples.png"
VAL_DATES = [f"201806{day:02d}" for day in range(20, 31)]
CSI_SCALE = 1.5

# Fixed samples used by Figure 5-9.  Their order must not be changed.
SAMPLE_INDICES = [
    1031,
    58439,
    21120,
    53083,
    62649,
    29293,
    32370,
    10470,
    12731,
    43022,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--save-data", type=Path, default=None)
    parser.add_argument("--inference-only", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def build_model(device: "torch.device") -> "torch.nn.Module":
    import json

    with (PROJECT_ROOT / "exp" / "train_x4.json").open(encoding="utf-8") as handle:
        network = dict(json.load(handle)["network"])
    network.update({"img_size": 32, "window_size": 8, "in_chans": 1, "upscale": 4})

    sys.path.insert(0, str(PROJECT_ROOT))
    from models.network_drct import DRCT

    return DRCT(**network).to(device)


def infer_fixed_samples() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate FY-3C, DRCT and formal-bicubic fields for the Figure 5-9 samples."""
    import torch
    from torch.utils.data import DataLoader, Subset

    sys.path.insert(0, str(PROJECT_ROOT))
    from data.dataset_SR_satellite import split_by_dates
    import main as experiment

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _, heldout_dataset = split_by_dates(
        samples_root=SAMPLES_ROOT,
        val_dates=VAL_DATES,
        augment=False,
    )
    validation_indices = experiment.load_or_create_validation_subset(heldout_dataset)
    test_indices = experiment.load_or_create_independent_test_subset(
        heldout_dataset,
        validation_indices,
        TEST_SUBSET,
        10_000,
    )
    if not set(SAMPLE_INDICES).issubset(test_indices):
        raise ValueError("Figure 5-9 sample indices are not all in the independent test set.")

    model = build_model(device)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["params_ema"], strict=True)
    model.eval()

    loader = DataLoader(
        Subset(heldout_dataset, SAMPLE_INDICES),
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )
    drct_parts: list[np.ndarray] = []
    bicubic_parts: list[np.ndarray] = []
    reference_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for low_resolution, reference in loader:
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if device.type == "cuda"
                else nullcontext()
            )
            with autocast:
                prediction = model(low_resolution.to(device)).float().cpu().numpy()[:, 0]
            drct_parts.append(prediction)
            bicubic_parts.append(experiment.bicubic_batch(low_resolution).numpy()[:, 0])
            reference_parts.append(reference.numpy()[:, 0])

    return (
        np.concatenate(reference_parts),
        np.concatenate(drct_parts),
        np.concatenate(bicubic_parts),
    )


def draw_figure(
    reference: np.ndarray,
    drct: np.ndarray,
    bicubic: np.ndarray,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "SimSun"],
            "font.serif": ["Times New Roman", "SimSun"],
            "font.size": 9,
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )

    images = [reference * CSI_SCALE, drct * CSI_SCALE, bicubic * CSI_SCALE]
    row_labels = ["FY-3C", "DRCT", "Bicubic"]
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=0.0, vmax=CSI_SCALE)

    # Match Figure 5-9: same tile size, black header strips, right-side labels,
    # zero white gap between image tiles, and a single shared CSI colourbar.
    fig = plt.figure(figsize=(6.25, 2.15), facecolor="white")
    left = 0.050
    tile_width = 0.085
    gap = 0.0
    tile_height = tile_width * 6.25 / 2.15
    grid_bottom = 0.185
    header_height = 0.070
    label_width = 0.020
    label_x = left + 10 * tile_width
    header_y = grid_bottom + 3 * tile_height

    for column in range(10):
        x = left + column * (tile_width + gap)
        header = fig.add_axes([x, header_y, tile_width, header_height])
        header.set_facecolor("black")
        header.text(
            0.5,
            0.5,
            f"Sample {column + 1}",
            color="white",
            ha="center",
            va="center",
            fontsize=9,
        )
        header.set_xticks([])
        header.set_yticks([])
        for spine in header.spines.values():
            spine.set_visible(False)
        if column < 9:
            header.spines["right"].set_visible(True)
            header.spines["right"].set_color("#EBEBEB")
            header.spines["right"].set_linewidth(0.45)

    for row, (row_images, row_label) in enumerate(zip(images, row_labels)):
        y = grid_bottom + (2 - row) * tile_height
        for column in range(10):
            x = left + column * (tile_width + gap)
            ax = fig.add_axes([x, y, tile_width, tile_height])
            ax.imshow(row_images[column], cmap=cmap, norm=norm, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("black")
                spine.set_linewidth(0.35)

        label_ax = fig.add_axes([label_x, y, label_width, tile_height])
        label_ax.set_facecolor("black")
        label_ax.text(
            0.5,
            0.5,
            row_label,
            color="white",
            rotation=-90,
            ha="center",
            va="center",
            fontsize=9,
        )
        label_ax.set_xticks([])
        label_ax.set_yticks([])
        for spine in label_ax.spines.values():
            spine.set_visible(False)

    colorbar_ax = fig.add_axes([0.550, 0.085, 0.220, 0.030])
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=colorbar_ax,
        orientation="horizontal",
    )
    colorbar.set_ticks(np.arange(0.0, CSI_SCALE + 0.001, 0.3))
    colorbar.ax.tick_params(labelsize=9, length=2.5, pad=2)
    colorbar.outline.set_visible(False)
    fig.text(0.540, 0.100, "Clear-sky index (CSI)", ha="right", va="center", fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.data is not None:
        with np.load(args.data) as data:
            reference = data["reference"]
            drct = data["drct"]
            bicubic = data["bicubic"]
    else:
        reference, drct, bicubic = infer_fixed_samples()

    if args.save_data is not None:
        args.save_data.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.save_data,
            reference=reference,
            drct=drct,
            bicubic=bicubic,
        )
        print(f"DATA: {args.save_data}")
    if args.inference_only:
        return

    draw_figure(reference, drct, bicubic, args.output.resolve())
    print(f"CHECKPOINT: {CHECKPOINT}")
    print(f"SAMPLE_INDICES: {SAMPLE_INDICES}")
    print(f"OUTPUT: {args.output.resolve()}")


if __name__ == "__main__":
    main()
