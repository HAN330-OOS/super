"""Plot random final samples for visual spatiotemporal alignment QA."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ROOT = Path(r"E:\Super-Resolution-Samples-2018-2019")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261024)
    return parser.parse_args()


def read_candidates(root: Path) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for manifest in sorted((root / "samples").glob("????????/manifest.csv")):
        with manifest.open(encoding="utf-8", newline="") as handle:
            candidates.extend(csv.DictReader(handle))
    return candidates


def nearest_4x(array: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(array, 4, axis=0), 4, axis=1)


def plot_sample(root: Path, row: dict[str, str], output: Path) -> None:
    date_dir = root / "samples" / row["date"]
    with np.load(date_dir / row["npz_file"], allow_pickle=False) as archive:
        index = int(row["array_index"])
        fy3 = archive["fy3"][index]
        fy4 = archive["fy4"][index]
        hr_csi = archive["hr_csi"][index]

    fy4_up = nearest_4x(fy4)
    rest2 = np.full_like(fy3, np.nan)
    np.divide(fy3, hr_csi, out=rest2, where=np.abs(hr_csi) > 1e-6)

    row0, col0 = int(row["row"]), int(row["col"])
    extent = (
        73.0 + col0 * 0.01,
        73.0 + (col0 + 128) * 0.01,
        54.0 - (row0 + 128) * 0.01,
        54.0 - row0 * 0.01,
    )
    fields = (fy3, fy4_up, rest2)
    titles = ("FY-3C SSI (128x128)", "FY-4A SSI (32x32, nearest x4)", "REST2 clear-sky GHI")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for axis, field, title in zip(axes, fields, titles):
        image = axis.imshow(field, extent=extent, origin="upper", cmap="inferno", vmin=0, vmax=1200)
        axis.set_title(f"{title}\nmean={np.nanmean(field):.1f} W/m2")
        axis.set_xlabel("Longitude (deg E)")
        axis.set_ylabel("Latitude (deg N)")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fy4_cst_minutes = (int(row["fy4_time"][:2]) * 60 + int(row["fy4_time"][2:]) + 480) % 1440
    fy4_cst = f"{fy4_cst_minutes // 60:02d}{fy4_cst_minutes % 60:02d}"
    fig.suptitle(
        f"Spatiotemporal QA | {row['date']} | FY-4 slot {row['fy4_time']} UTC / {fy4_cst} CST | "
        f"sample {row['sample_id']} | center ({float(row['center_lon']):.2f}E, {float(row['center_lat']):.2f}N)"
    )
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    candidates = read_candidates(root)
    if args.count < 1 or args.count > len(candidates):
        raise ValueError("invalid sample count")
    selected = random.Random(args.seed).sample(candidates, args.count)
    output_dir = root / "qc_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "random_selection.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=selected[0].keys())
        writer.writeheader()
        writer.writerows(selected)
    for number, row in enumerate(selected, start=1):
        output = output_dir / f"qc_{number:02d}_{row['date']}_sample_{int(row['sample_id']):04d}.png"
        plot_sample(root, row, output)
        print(output)


if __name__ == "__main__":
    main()
