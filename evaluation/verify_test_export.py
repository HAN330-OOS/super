"""Verify a completed formal test export without running the model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def close_enough(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-11):
        raise ValueError(f"Metric mismatch {label}: {actual} != {expected}")


def verify(root: Path) -> dict:
    started = time.perf_counter()
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    config = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    if status["state"] != "complete" or not metrics["is_full_test"] or metrics["samples"] != 35476:
        raise ValueError("Expected a completed full 35476-patch export")
    if list(root.glob("*.partial.npy")):
        raise ValueError("Unexpected incomplete prediction file")
    arguments = config["arguments"]
    source_csv = Path(arguments["splits_root"]) / "test_samples.csv"
    source_rows = read_csv(source_csv)
    output_rows = read_csv(root / "test_order.csv")
    patch_rows = read_csv(root / "per_patch_metrics.csv")
    if len(source_rows) != 35476 or len(output_rows) != 35476 or len(patch_rows) != 35476:
        raise ValueError("Sample/metric CSV row count differs from 35476")
    if config["numerics"]["normalization"] != "none" or config["numerics"]["csi_multiplier"] != 1.0 or config["numerics"]["clipping"]:
        raise ValueError("Unexpected CSI processing configuration")
    if config["strict_load"] != {"missing_keys": [], "unexpected_keys": []}:
        raise ValueError("Weights were not strictly loaded")
    for position, record in enumerate(source_rows):
        output_record = output_rows[position]
        if int(output_record["prediction_index"]) != position or int(output_record["csv_row_index"]) != position:
            raise ValueError(f"Prediction ordering mismatch at {position}")
        if any(output_record[field] != value for field, value in record.items()):
            raise ValueError(f"Source CSV differs at {position}")
        if int(patch_rows[position]["csv_row_index"]) != position or patch_rows[position]["sample_id"] != record["sample_id"]:
            raise ValueError(f"Per-patch metric ordering mismatch at {position}")
    for split, expected_hash in config["data_identity"]["split_csv_sha256"].items():
        if sha256_file(Path(arguments["splits_root"]) / f"{split}_samples.csv") != expected_hash:
            raise ValueError(f"Source CSV changed: {split}")
    if sha256_file(Path(arguments["checkpoint"])) != config["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed")
    if sha256_file(root / "export_test_predictions.py") != config["script_sha256"]:
        raise ValueError("Inference script snapshot hash mismatch")
    for name, expected_hash in config["code_sha256"].items():
        if sha256_file(Path(arguments["code_root"]) / name) != expected_hash:
            raise ValueError(f"Original code changed: {name}")
    prediction = np.load(root / "drct_prediction_csi.npy", mmap_mode="r", allow_pickle=False)
    clear_ghi = np.load(Path(arguments["clear_ghi"]), mmap_mode="r", allow_pickle=False)
    if prediction.shape != (35476, 128, 128) or prediction.dtype != np.float32 or clear_ghi.shape != prediction.shape:
        raise ValueError("Prediction or clear-GHI shape/dtype mismatch")
    digest = hashlib.sha256()
    prediction_min, prediction_max = math.inf, -math.inf
    negative_values = 0
    for start in range(0, len(prediction), 128):
        values = np.asarray(prediction[start:start + 128])
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite prediction at {start}")
        prediction_min = min(prediction_min, float(values.min()))
        prediction_max = max(prediction_max, float(values.max()))
        negative_values += int((values < 0).sum())
        digest.update(values.tobytes(order="C"))
    if digest.hexdigest() != metrics["prediction"]["payload_sha256"]:
        raise ValueError("Prediction payload SHA256 mismatch")
    for method in ("bicubic", "drct"):
        for metric in ("psnr", "ssim"):
            average = float(np.mean([float(record[f"{method}_{metric}"]) for record in patch_rows]))
            close_enough(average, float(metrics["csi"][method][metric]), f"{method}_{metric}")
    if sum(int(record["ghi_valid_pixels"]) for record in patch_rows) != metrics["valid_pixels"]:
        raise ValueError("Per-patch GHI valid-pixel counts do not add up")
    spec = importlib.util.spec_from_file_location("formal_inference_snapshot", root / "export_test_predictions.py")
    inference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inference)
    sys.path.insert(0, arguments["code_root"])
    from utils.training import bicubic_batch
    import torch

    sample_reader = inference.PatchReader(Path(arguments["samples_root"]))
    try:
        with np.load(root / "ghi_scatter_data.npz", allow_pickle=False) as scatter:
            keys = scatter.files
            scatter_count = len(scatter["global_pixel_index"])
            global_indices = scatter["global_pixel_index"]
            csv_indices = scatter["csv_row_index"].astype(np.int64)
            flat_indices = scatter["pixel_flat_index"].astype(np.int64)
            if not scatter_count or scatter_count != metrics["scatter"]["saved_pixels"]:
                raise ValueError("Scatter point count mismatch")
            if any(len(scatter[name]) != scatter_count for name in keys):
                raise ValueError("Scatter arrays have different lengths")
            if not np.all(np.diff(global_indices) > 0) or not np.array_equal(global_indices, csv_indices * 16384 + flat_indices):
                raise ValueError("Invalid or repeated scatter indices")
            if np.any(csv_indices < 0) or np.any(csv_indices >= 35476) or np.any(flat_indices < 0) or np.any(flat_indices >= 16384):
                raise ValueError("Scatter index out of range")
            if not np.array_equal(flat_indices, scatter["pixel_row"].astype(np.int64) * 128 + scatter["pixel_col"]):
                raise ValueError("Scatter pixel row/column mismatch")
            expected_ids = np.array([int(record["sample_id"]) for record in source_rows], dtype=np.int64)
            if not np.array_equal(scatter["sample_id"], expected_ids[csv_indices]):
                raise ValueError("Scatter sample IDs mismatch")
            if not np.array_equal(scatter["drct_csi"], prediction.reshape(-1)[global_indices]):
                raise ValueError("Scatter differs from complete prediction")
            if not np.array_equal(scatter["clear_ghi"], clear_ghi.reshape(-1)[global_indices]):
                raise ValueError("Scatter clear GHI is misaligned")
            for method in ("bicubic", "drct"):
                if not np.array_equal(scatter[f"{method}_ghi"], scatter[f"{method}_csi"] * scatter["clear_ghi"]):
                    raise ValueError(f"Scatter GHI restoration mismatch: {method}")
            rng = np.random.default_rng(config["scatter"]["seed"])
            candidates = np.sort(rng.choice(35476 * 16384, size=min(config["scatter"]["max_points"], 35476 * 16384), replace=False))
            if not np.isin(global_indices, candidates).all():
                raise ValueError("Scatter random seed/index mismatch")
            if metrics["invalid_observation_pixels"] == 0 and not np.array_equal(global_indices, candidates):
                raise ValueError("Unexpected missing scatter candidates")
            unique_rows = np.unique(csv_indices)
            check_rows = unique_rows[np.unique(np.linspace(0, len(unique_rows) - 1, min(64, len(unique_rows)), dtype=int))]
            for row_index in check_rows:
                low, target, observed = sample_reader.batch([source_rows[int(row_index)]])
                selected = csv_indices == row_index
                pixel_indices = flat_indices[selected]
                baseline = bicubic_batch(torch.from_numpy(low))[:, 0].numpy()
                for name, values in (("target_csi", target), ("fy3_reference", observed), ("bicubic_csi", baseline)):
                    if not np.array_equal(scatter[name][selected], values.reshape(-1)[pixel_indices]):
                        raise ValueError(f"Scatter source values differ: {name}, row {row_index}")
            if any(not np.isfinite(scatter[name]).all() for name in keys):
                raise ValueError("Nonfinite scatter values")
    finally:
        sample_reader.close()
    return {
        "passed": True, "samples": 35476, "prediction_shape": list(prediction.shape),
        "prediction_dtype": str(prediction.dtype), "prediction_nonfinite": 0,
        "prediction_min": prediction_min, "prediction_max": prediction_max,
        "negative_prediction_values_retained": negative_values,
        "prediction_payload_sha256": digest.hexdigest(),
        "csv_order_exact": True, "per_patch_metric_means_match": True,
        "scatter_points": scatter_count, "scatter_seed_and_indices_match": True,
        "scatter_restoration_exact": True, "source_spotcheck_patches": len(check_rows),
        "checkpoint_and_source_hashes_match": True, "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    for name in ("verification_report.json", "SHA256SUMS.txt", "verify_test_export.py"):
        if (root / name).exists():
            raise FileExistsError(f"Refusing to overwrite {root / name}")
    print("VERIFY_START", root, flush=True)
    report = verify(root)
    shutil.copyfile(Path(__file__), root / "verify_test_export.py")
    (root / "verification_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    with (root / "SHA256SUMS.txt").open("x", encoding="utf-8", newline="\n") as handle:
        for path in sorted(root.iterdir()):
            if path.is_file() and path.name != "SHA256SUMS.txt":
                checksum = sha256_file(path)
                handle.write(f"{checksum}  {path.name}\n")
                print("SHA256", checksum, path.name, flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
