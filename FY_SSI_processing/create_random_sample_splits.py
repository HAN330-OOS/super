"""Create deterministic sample-level random train/validation/test splits."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


DEFAULT_ROOT = Path(r"E:\Super-Resolution-Samples-2018-2019")
RATIO = (5, 1, 1)


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    all_samples = root / "all_samples.csv"
    with all_samples.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ())
        rows = list(reader)
    if not rows:
        raise ValueError("all_samples.csv is empty")

    permutation = np.random.default_rng(args.seed).permutation(len(rows))
    exact_counts = np.asarray(RATIO, dtype=np.float64) * len(rows) / sum(RATIO)
    counts = np.floor(exact_counts).astype(int).tolist()
    remainder = len(rows) - sum(counts)
    fractional_order = np.argsort(-(exact_counts - np.floor(exact_counts)))
    for index in fractional_order[:remainder]:
        counts[int(index)] += 1
    boundaries = np.cumsum([0, *counts])
    split_names = ("train", "validation", "test")
    split_dir = root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    selected_ids: set[int] = set()
    for split_index, split_name in enumerate(split_names):
        indices = permutation[boundaries[split_index] : boundaries[split_index + 1]]
        output = split_dir / f"{split_name}_samples.csv"
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row_index in indices:
                row = rows[int(row_index)]
                sample_id = int(row["sample_id"])
                if sample_id in selected_ids:
                    raise RuntimeError(f"duplicate sample_id across splits: {sample_id}")
                selected_ids.add(sample_id)
                writer.writerow(row)

    if len(selected_ids) != len(rows):
        raise RuntimeError("random splits do not cover all samples")
    for obsolete in ("train_dates.txt", "val_dates.txt", "test_dates.txt"):
        (split_dir / obsolete).unlink(missing_ok=True)

    summary_path = root / "status" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["split"] = {
        "mode": "sample_level_random",
        "ratio": "5:1:1",
        "seed": args.seed,
        "train_samples": counts[0],
        "validation_samples": counts[1],
        "test_samples": counts[2],
    }
    summary.pop("train_dates", None)
    summary.pop("val_dates", None)
    summary.pop("test_dates", None)
    atomic_json(summary_path, summary)
    print(json.dumps(summary["split"], ensure_ascii=False))


if __name__ == "__main__":
    main()
