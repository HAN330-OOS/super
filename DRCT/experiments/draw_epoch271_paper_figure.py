"""Draw independent-test CSI samples from the frozen final model.

The figure follows the project's paper-figure layout:
ten columns, three rows (LR / observation / DRCT), a shared physical CSI
color scale, and a caption suitable for a paper manuscript.
"""

from __future__ import annotations

import argparse
import json
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
OPTION_PATH = PROJECT_ROOT / "exp" / "train_x4.json"
RUN_DIR = (
    PROJECT_ROOT
    / "results"
    / "fy4a_fy3c_drctl_w8_scratch_50k_b16_e50"
)
DATASET_ROOT = PROJECT_ROOT / "datasets" / "fy4a_fy3c_csi_x4"
FINAL_CHECKPOINT = RUN_DIR / "checkpoints" / "epoch_271_converged_final.pth"
DEFAULT_OUTPUT = RUN_DIR / "final" / "paper_figures" / "ten_sample_epoch271_independent_test.png"
TEST_SUBSET = DATASET_ROOT / "test" / "samples.json"
VAL_DATES = [f"201806{day:02d}" for day in range(20, 31)]
CSI_SCALE = 1.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=FINAL_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--inference-samples",
        type=int,
        default=120,
        help="Independent test patches used to choose representative montage samples.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Previously saved inference arrays; skips checkpoint inference.",
    )
    parser.add_argument(
        "--save-data",
        type=Path,
        default=None,
        help="Save inferred arrays for rendering in another environment.",
    )
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Run checkpoint inference and save arrays without drawing.",
    )
    return parser.parse_args()


def build_model(device: torch.device) -> torch.nn.Module:
    import torch

    # Keep the final experiment architecture: 32x32 LR, window 8, x4 output.
    with OPTION_PATH.open(encoding="utf-8") as handle:
        network = dict(json.load(handle)["network"])
    network.update({"img_size": 32, "window_size": 8, "in_chans": 1, "upscale": 4})

    sys.path.insert(0, str(PROJECT_ROOT))
    from models.network_drct import DRCT

    return DRCT(**network).to(device)


def load_final_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    import torch

    model = build_model(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("params_ema")
    if state is None:
        state = checkpoint.get("model")
    if state is None:
        raise KeyError(f"No params_ema/model state found in {checkpoint_path}")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def infer_test_samples(
    model: torch.nn.Module,
    heldout_dataset: torch.utils.data.Dataset,
    test_indices: list[int],
    sample_count: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import torch
    from torch.utils.data import DataLoader, Subset

    sample_count = min(sample_count, len(test_indices))
    positions = np.linspace(0, len(test_indices) - 1, sample_count, dtype=np.int64)
    indices = np.asarray(test_indices, dtype=np.int64)[positions]
    loader = DataLoader(
        Subset(heldout_dataset, indices.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    lr_parts: list[np.ndarray] = []
    hr_parts: list[np.ndarray] = []
    prediction_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for low_resolution, observation in loader:
            low_resolution = low_resolution.to(device, non_blocking=True)
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if device.type == "cuda"
                else nullcontext()
            )
            with autocast:
                prediction = model(low_resolution).float().cpu().numpy()[:, 0]
            lr_parts.append(low_resolution.cpu().numpy()[:, 0])
            hr_parts.append(observation.numpy()[:, 0])
            prediction_parts.append(prediction)

    lr = np.concatenate(lr_parts, axis=0)
    observation = np.concatenate(hr_parts, axis=0)
    prediction = np.concatenate(prediction_parts, axis=0)

    # Match the original paper-figure selection: span low to high spatial
    # variability rather than hand-picking visually favorable samples.
    order = np.argsort(observation.std(axis=(1, 2)))
    ranks = np.rint(np.linspace(0.15, 0.98, 10) * (sample_count - 1)).astype(np.int64)
    positions = order[ranks]
    return lr[positions], observation[positions], prediction[positions], indices[positions]


def draw_paper_figure(
    lr: np.ndarray,
    observation: np.ndarray,
    prediction: np.ndarray,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    # Match the typography and compact grid layout used by the reference paper.
    plt.rcParams.update(
        {
            # Word small-five is 9 pt.  Times New Roman is used for English;
            # SimSun provides the specified Chinese fallback when needed.
            "font.family": ["Times New Roman", "SimSun"],
            "font.serif": ["Times New Roman", "SimSun"],
            "font.size": 9,
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )

    images = [lr * CSI_SCALE, observation * CSI_SCALE, prediction * CSI_SCALE]
    row_labels = ["LR", "OB", "SR"]
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=0.0, vmax=CSI_SCALE)

    # The rendered width is 6.25 in, which fits a Word text line without
    # downscaling. Therefore 9-pt labels remain Word small-five (小五).
    # Adjacent axes use a zero gap so no white gutters appear between samples.
    fig = plt.figure(figsize=(6.25, 2.15), facecolor="white")
    left = 0.050
    tile_width = 0.085
    gap = 0.0
    tile_height = tile_width * 6.25 / 2.15
    grid_bottom = 0.185
    header_height = 0.070
    label_width = 0.020
    label_x = left + 10 * tile_width + 10 * gap

    header_y = grid_bottom + 3 * tile_height + 2 * gap
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
        # Separate sample titles without introducing white gutters in the
        # three image rows below.
        if column < 9:
            header.spines["right"].set_visible(True)
            header.spines["right"].set_color("#EBEBEB")
            header.spines["right"].set_linewidth(0.45)

    for row, (row_images, row_label) in enumerate(zip(images, row_labels)):
        y = grid_bottom + (2 - row) * (tile_height + gap)
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
    checkpoint_path = args.checkpoint.resolve()
    output_path = args.output.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    if args.data is not None:
        with np.load(args.data) as data:
            lr = data["lr"]
            observation = data["observation"]
            prediction = data["prediction"]
            indices = data["indices"]
    else:
        import torch

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"CHECKPOINT: {checkpoint_path}", flush=True)
        print(f"DEVICE: {device}", flush=True)

        sys.path.insert(0, str(PROJECT_ROOT))
        from data.dataset_SR_satellite import split_by_dates
        import main as experiment

        _, validation_dataset = split_by_dates(
            samples_root=SAMPLES_ROOT,
            val_dates=VAL_DATES,
            augment=False,
        )
        validation_indices = experiment.load_or_create_validation_subset(
            validation_dataset
        )
        test_indices = experiment.load_or_create_independent_test_subset(
            validation_dataset,
            validation_indices,
            TEST_SUBSET,
            10_000,
        )
        model = load_final_model(checkpoint_path, device)
        lr, observation, prediction, indices = infer_test_samples(
            model,
            validation_dataset,
            test_indices,
            args.inference_samples,
            args.batch_size,
            device,
        )

    if args.save_data is not None:
        args.save_data.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.save_data,
            lr=lr,
            observation=observation,
            prediction=prediction,
            indices=indices,
        )
        print(f"DATA: {args.save_data.resolve()}", flush=True)
    if args.inference_only:
        return

    draw_paper_figure(lr, observation, prediction, output_path)
    print(f"SELECTED_INDICES: {indices.tolist()}", flush=True)
    print(f"OUTPUT: {output_path}", flush=True)


if __name__ == "__main__":
    main()
