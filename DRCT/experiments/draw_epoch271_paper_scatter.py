"""Draw independent-test GHI scatter panels for the frozen epoch-271 model."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = PROJECT_ROOT / "results" / "fy4a_fy3c_drctl_w8_scratch_50k_b16_e50"
DATASET_ROOT = PROJECT_ROOT / "datasets" / "fy4a_fy3c_csi_x4"
CHECKPOINT = RUN_DIR / "checkpoints" / "epoch_271_converged_final.pth"
METRICS = RUN_DIR / "final" / "epoch_271_converged_final_independent_test_metrics.json"
DEFAULT_OUTPUT = RUN_DIR / "final" / "paper_figures" / "scatter_ghi_bicubic_epoch271_independent_test.png"
TEST_SUBSET = DATASET_ROOT / "test" / "samples.json"
SAMPLES_ROOT = (
    PROJECT_ROOT / "outputs" / "fy" / "samples"
    if (PROJECT_ROOT / "outputs" / "fy" / "samples").exists()
    else PROJECT_ROOT.parent / "outputs" / "fy" / "samples"
)
VAL_DATES = [f"201806{day:02d}" for day in range(20, 31)]
FINAL_TEST_SAMPLES = 10_000
CSI_SCALE = 1.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--metrics", type=Path, default=METRICS)
    parser.add_argument("--subset", type=Path, default=TEST_SUBSET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument(
        "--save-data",
        type=Path,
        default=None,
        help="Optional temporary inference-array file; not saved by default.",
    )
    parser.add_argument("--inference-only", action="store_true")
    return parser.parse_args()


def infer_scatter_data(
    checkpoint_path: Path,
    subset_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    import torch

    sys.path.insert(0, str(PROJECT_ROOT))
    import main as experiment

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _, validation_dataset = experiment.split_by_dates(
        samples_root=SAMPLES_ROOT,
        val_dates=VAL_DATES,
        augment=False,
    )
    validation_indices = experiment.load_or_create_validation_subset(
        validation_dataset
    )
    formal_indices = experiment.load_or_create_independent_test_subset(
        validation_dataset,
        validation_indices,
        subset_path,
        FINAL_TEST_SAMPLES,
    )
    model = experiment.build_model(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["params_ema"], strict=True)
    model.eval()
    clear_sky = experiment.ClearSkyProvider(validation_dataset)
    loader = experiment.validation_loader(validation_dataset, formal_indices, 8)

    observation_parts: list[np.ndarray] = []
    bicubic_parts: list[np.ndarray] = []
    drct_parts: list[np.ndarray] = []
    rng = np.random.default_rng(experiment.SEED)
    full_min = float("inf")
    full_max = float("-inf")
    offset = 0
    with torch.inference_mode():
        for batch_number, (low_resolution, target) in enumerate(loader, 1):
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if device.type == "cuda"
                else nullcontext()
            )
            with autocast:
                prediction = model(low_resolution.to(device)).float().cpu()
            bicubic = experiment.bicubic_batch(low_resolution)
            for item in range(target.shape[0]):
                dataset_index = formal_indices[offset]
                offset += 1
                clear = clear_sky.get(dataset_index)
                observation_ghi = target[item, 0].numpy() * CSI_SCALE * clear
                bicubic_ghi = bicubic[item, 0].numpy() * CSI_SCALE * clear
                drct_ghi = prediction[item, 0].numpy() * CSI_SCALE * clear
                valid = (
                    np.isfinite(observation_ghi)
                    & np.isfinite(bicubic_ghi)
                    & np.isfinite(drct_ghi)
                )
                observation_valid = observation_ghi[valid]
                bicubic_valid = bicubic_ghi[valid]
                drct_valid = drct_ghi[valid]
                if observation_valid.size == 0:
                    continue
                full_min = min(
                    full_min,
                    float(observation_valid.min()),
                    float(bicubic_valid.min()),
                    float(drct_valid.min()),
                )
                full_max = max(
                    full_max,
                    float(observation_valid.max()),
                    float(bicubic_valid.max()),
                    float(drct_valid.max()),
                )
                count = min(64, observation_valid.size)
                choice = rng.choice(observation_valid.size, size=count, replace=False)
                observation_parts.append(observation_valid[choice].astype(np.float32))
                bicubic_parts.append(bicubic_valid[choice].astype(np.float32))
                drct_parts.append(drct_valid[choice].astype(np.float32))
            if batch_number % 100 == 0 or batch_number == len(loader):
                print(f"[scatter inference] {offset}/{len(formal_indices)}", flush=True)

    observation = np.concatenate(observation_parts)
    bicubic = np.concatenate(bicubic_parts)
    drct = np.concatenate(drct_parts)
    span = max(full_max - full_min, 1.0)
    margin = 0.035 * span
    return observation, bicubic, drct, full_min - margin, full_max + margin


def draw_paper_scatter(
    observation: np.ndarray,
    bicubic: np.ndarray,
    drct: np.ndarray,
    axis_min: float,
    axis_max: float,
    metrics_path: Path,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import MaxNLocator

    # Keep every visible string in Times New Roman rather than Matplotlib's
    # default sans-serif family, including labels that contain superscripts.
    plt.rcParams.update(
        {
            # Word small-five is 9 pt.  Times New Roman is used for all English
            # labels; SimSun is retained as the fallback for any Chinese text.
            "font.family": ["Times New Roman", "SimSun"],
            "font.serif": ["Times New Roman", "SimSun"],
            "font.size": 9,
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )

    with metrics_path.open(encoding="utf-8") as handle:
        metrics = json.load(handle)["ghi"]

    # Use one shared random subset and one shared domain for both panels.
    rng = np.random.default_rng(60)
    point_count = min(150_000, observation.size)
    choice = rng.choice(observation.size, size=point_count, replace=False)
    x = observation[choice]
    y_values = [bicubic[choice], drct[choice]]
    titles = ["Bicubic: 4 km to 1 km", "DRCT: 4 km to 1 km"]
    colors = ["#F2A23A", "#4C9BD5"]
    methods = ["bicubic", "drct"]

    # Two square panels are arranged horizontally so the figure fits a single
    # Word text line while retaining identical x/y ranges and aspect ratios.
    fig = plt.figure(figsize=(6.4, 3.2), facecolor="white")
    axes = [
        fig.add_axes([0.100, 0.145, 0.390, 0.780]),
        fig.add_axes([0.600, 0.145, 0.390, 0.780]),
    ]
    for panel_index, (ax, y, title, color, method) in enumerate(
        zip(axes, y_values, titles, colors, methods)
    ):
        ax.scatter(
            x,
            y,
            s=0.45,
            alpha=0.075,
            color=color,
            edgecolors="none",
            rasterized=True,
        )
        ax.plot(
            [axis_min, axis_max],
            [axis_min, axis_max],
            color="black",
            linestyle=(0, (6, 5)),
            linewidth=1.1,
            zorder=3,
        )
        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)
        ax.set_aspect("equal", adjustable="box")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
        ax.tick_params(labelsize=9, width=0.7, length=3, pad=3)
        ax.grid(True, color="#9E9E9E", linewidth=0.45, alpha=0.75)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(0.75)
            spine.set_color("black")

        # Black title strip and panel label, matching the sample paper figure.
        ax.add_patch(
            Rectangle(
                (0.0, 1.005),
                1.0,
                0.060,
                transform=ax.transAxes,
                facecolor="black",
                edgecolor="none",
                clip_on=False,
                zorder=4,
            )
        )
        ax.text(
            0.5,
            1.036,
            title,
            transform=ax.transAxes,
            color="white",
            ha="center",
            va="center",
            fontsize=9,
            zorder=5,
        )
        ax.text(
            -0.20,
            1.035,
            "(a)" if panel_index == 0 else "(b)",
            transform=ax.transAxes,
            color="black",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
        panel_metrics = metrics[method]
        if False:  # Metrics are reported in the result table, not in this figure.
            ax.text(
            0.055,
            0.955,
            "\n".join(
                [
                    f"R² = {panel_metrics['r2']:.3f}",
                    f"RMSE = {panel_metrics['rmse']:.2f} W/m²",
                    f"MAE = {panel_metrics['mae']:.2f} W/m²",
                ]
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="black",
            linespacing=1.35,
        )
        method_label = "Bicubic" if method == "bicubic" else "DRCT"
        ax.set_ylabel(f"{method_label} Result [W/m²]", fontsize=9, labelpad=10)
        ax.set_xlabel("FY-3C Observation [W/m²]", fontsize=9, labelpad=5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.data is not None:
        with np.load(args.data) as data:
            observation = data["observation"]
            bicubic = data["bicubic"]
            drct = data["drct"]
            axis_min = float(data["axis_min"])
            axis_max = float(data["axis_max"])
    else:
        observation, bicubic, drct, axis_min, axis_max = infer_scatter_data(
            args.checkpoint.resolve(), args.subset.resolve()
        )
    if args.save_data is not None:
        args.save_data.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.save_data,
            observation=observation,
            bicubic=bicubic,
            drct=drct,
            axis_min=np.float64(axis_min),
            axis_max=np.float64(axis_max),
        )
        print(f"DATA: {args.save_data.resolve()}", flush=True)
    if args.inference_only:
        return
    draw_paper_scatter(
        observation,
        bicubic,
        drct,
        axis_min,
        axis_max,
        args.metrics.resolve(),
        args.output.resolve(),
    )
    print(f"OUTPUT: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
