"""Export REST2 clear-sky GHI aligned row-for-row with test_samples.csv."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"E:\Super-Resolution-Samples-2018-2019"),
    )
    parser.add_argument("--split", default="test_samples.csv")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def fill_undefined_ghi(ghi: np.ndarray, valid: np.ndarray) -> None:
    missing = np.argwhere(~valid)
    for row, col in missing:
        for radius in range(1, max(ghi.shape)):
            row0 = max(0, row - radius)
            row1 = min(ghi.shape[0], row + radius + 1)
            col0 = max(0, col - radius)
            col1 = min(ghi.shape[1], col + radius + 1)
            nearby = ghi[row0:row1, col0:col1]
            nearby_valid = valid[row0:row1, col0:col1]
            if nearby_valid.any():
                ghi[row, col] = float(np.median(nearby[nearby_valid]))
                valid[row, col] = True
                break
        else:
            raise ValueError("Cannot recover GHI for a patch without valid pixels")


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    split_path = root / "splits" / args.split
    output = args.output or (root / "splits" / "test_clear_ghi.npy")
    metadata_path = output.with_suffix(".json")

    with split_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    groups: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for csv_index, row in enumerate(rows):
        groups[(row["date"], row["npz_file"])].append(
            (csv_index, int(row["array_index"]))
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    clear_ghi = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.float32, shape=(len(rows), 128, 128)
    )

    recovered_pixels = 0
    max_reconstruction_error = 0.0
    ordered_groups = sorted(groups.items())
    for group_number, ((date, npz_name), entries) in enumerate(ordered_groups, 1):
        npz_path = root / "samples" / date / npz_name
        with np.load(npz_path) as archive:
            indices = [array_index for _, array_index in entries]
            fy3 = archive["fy3"][indices].astype(np.float32, copy=False)
            hr_csi = archive["hr_csi"][indices].astype(np.float32, copy=False)

        for local_index, (csv_index, _) in enumerate(entries):
            numerator = fy3[local_index]
            denominator = hr_csi[local_index]
            valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0)
            ghi = np.empty((128, 128), dtype=np.float32)
            np.divide(numerator, denominator, out=ghi, where=valid)
            missing_count = int((~valid).sum())
            if missing_count:
                fill_undefined_ghi(ghi, valid)
                recovered_pixels += missing_count
            clear_ghi[csv_index] = ghi

            reconstructed = denominator * ghi
            if valid.any():
                error = float(np.max(np.abs(reconstructed[valid] - numerator[valid])))
                max_reconstruction_error = max(max_reconstruction_error, error)

        if group_number % 250 == 0 or group_number == len(groups):
            clear_ghi.flush()
            print(f"{group_number}/{len(groups)} NPZ files")

    if not np.isfinite(clear_ghi).all() or not (clear_ghi > 0).all():
        raise ValueError("Exported clear-sky GHI contains invalid values")

    metadata = {
        "split_csv": str(split_path),
        "output": str(output),
        "shape": list(clear_ghi.shape),
        "dtype": str(clear_ghi.dtype),
        "alignment": "axis 0 exactly follows test_samples.csv row order",
        "derivation": "fy3 / hr_csi",
        "recovered_zero_csi_pixels": recovered_pixels,
        "zero_csi_recovery": "local median of valid clear-sky GHI pixels",
        "max_abs_reconstruction_error_w_m2": max_reconstruction_error,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
