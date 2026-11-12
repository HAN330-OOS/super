"""Evaluate the saved epoch-37 SwinIR predictions; all source files are read-only.

Requires Python 3.10+ and NumPy. No inference, resplitting, normalization or
clipping is performed. The GHI protocol is the verified formal DRCT-L protocol.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import platform
import struct
import time
import zipfile
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np


DEFAULT_OUTPUT = Path(r"E:\DRCT_Backup_20261102\SwinIR")
DEFAULT_DATASET = Path(r"E:\Super-Resolution-Samples-2018-2019")
DEFAULT_FORMAL = Path(r"E:\FY4A-FY3C-Formal-Analysis-2018-2019\full_20261029_141002_912bff84")
EXPECTED_SHAPE = (35476, 128, 128)
EXPECTED_CSV_SHA256 = "3f3a72f49919a93228f6673932397316dbd6ed5989769f044253e4a21882d710"
EXPECTED_PREDICTION_SHA256 = "87bb6c29277aefae0dd13f583ea8b9804ad687c47c9bea5a84c3d5f31b2185f6"
FORMAL_SCRIPT_SHA256 = "7c4d3eb84eec48cffef3f7292ac9abdd2dc939b28f9d77a1f972a9a5995e23dc"
FILL_VALUES = (-999.0, 65532.0, 65535.0)
SUMMARY_COLUMNS = ["方法", "CSI PSNR", "CSI SSIM", "GHI R²", "GHI RMSE", "GHI MAE", "GHI MBE"]
SHANGHAI = timezone(timedelta(hours=8))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require({"npz_file", "array_index", "date", "sample_id"}.issubset(reader.fieldnames or []), f"Missing CSV columns: {path}")
        rows = list(reader)
    require(bool(rows), f"Empty CSV: {path}")
    require(all(None not in row and all(value is not None for value in row.values()) for row in rows), f"Malformed CSV: {path}")
    return rows


def stat_identity(path: Path) -> dict:
    value = path.stat()
    return {"bytes": value.st_size, "mtime_ns": value.st_mtime_ns}


def close_mmap(values: np.ndarray) -> None:
    if isinstance(values, np.memmap):
        values._mmap.close()


def npz_member(path: Path, archive: zipfile.ZipFile, name: str) -> np.ndarray:
    """Map an uncompressed NPY member, or load one member of a compressed NPZ.

    At most one NPZ group is open. No temporary extraction files are written.
    """
    entry = archive.getinfo(name + ".npy")
    require(not entry.flag_bits & 1, f"Encrypted NPZ member: {path}/{name}")
    if entry.compress_type != zipfile.ZIP_STORED:
        with archive.open(entry) as handle:
            return np.load(handle, allow_pickle=False)
    with path.open("rb") as handle:
        handle.seek(entry.header_offset)
        header = handle.read(30)
        require(len(header) == 30 and header[:4] == b"PK\x03\x04", f"Invalid ZIP header: {path}")
        name_length, extra_length = struct.unpack_from("<HH", header, 26)
        handle.seek(name_length + extra_length, 1)
        data_start = handle.tell()
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise ValueError(f"Unsupported NPY header version: {path}/{name}: {version}")
        offset = handle.tell()
    require(dtype == np.dtype("float32") and not dtype.hasobject, f"Expected float32 source: {path}/{name}")
    expected_end = offset + math.prod(shape) * dtype.itemsize
    require(expected_end == data_start + entry.file_size and expected_end <= path.stat().st_size, f"Truncated or inconsistent NPY member: {path}/{name}")
    return np.memmap(path, mode="r", dtype=dtype, shape=shape, offset=offset, order="F" if fortran else "C")


def load_original_accumulator(source: Path, expected_sha256: str):
    """Reuse ONLY the original numerical class without importing Torch/training.

    The complete source hash must equal that recorded in the formal DRCT run.
    AST extraction executes the already verified RegressionAccumulator only.
    """
    require(sha256_file(source) == expected_sha256, f"Formal accumulator source changed: {source}")
    tree = ast.parse(source.read_text(encoding="utf-8-sig"))
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RegressionAccumulator"]
    require(len(nodes) == 1, "Cannot identify original RegressionAccumulator")
    module = ast.Module(body=nodes, type_ignores=[])
    environment = {"np": np, "math": math}
    exec(compile(module, str(source), "exec"), environment)
    return environment["RegressionAccumulator"]


def scan_array(values: np.ndarray, label: str, batch: int, positive: bool = False) -> dict:
    digest = hashlib.sha256()
    result = {"shape": list(values.shape), "dtype": str(values.dtype), "nonfinite": 0, "negative": 0, "zero": 0, "min": math.inf, "max": -math.inf}
    for start in range(0, len(values), batch):
        block = np.asarray(values[start:start + batch])
        bad = ~np.isfinite(block)
        if bad.any():
            position = np.argwhere(bad)[0]
            raise FloatingPointError(f"{label}: nonfinite value at CSV row {start + int(position[0])}, pixel {position[1:].tolist()}")
        if positive and (block <= 0).any():
            position = np.argwhere(block <= 0)[0]
            raise ValueError(f"{label}: clear GHI must be > 0, CSV row {start + int(position[0])}, pixel {position[1:].tolist()}")
        result["negative"] += int(np.count_nonzero(block < 0))
        result["zero"] += int(np.count_nonzero(block == 0))
        result["min"] = min(result["min"], float(block.min()))
        result["max"] = max(result["max"], float(block.max()))
        digest.update(block.tobytes(order="C"))
    result["payload_sha256"] = digest.hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--formal-drct-root", type=Path, default=DEFAULT_FORMAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    require(1 <= args.batch_size <= 128, "batch-size must be between 1 and 128")
    started = time.perf_counter()
    created = datetime.now(SHANGHAI).isoformat(timespec="seconds")
    output = args.output_dir.resolve()
    require(output.is_dir(), f"Output directory must already exist: {output}")
    json_path = output / "SwinIR_GHI_Metrics.json"
    csv_path = output / "SwinIR_正式指标汇总.csv"
    for path in (json_path, csv_path):
        require(not path.exists() and not path.with_suffix(path.suffix + ".tmp").exists(), f"Refusing to overwrite existing result: {path}")

    extracted = args.backup_root / "Extracted"
    located = {}
    for name in ("swinir_prediction_csi.npy", "test_samples.csv", "independent_test_metrics.json"):
        candidates = sorted(extracted.rglob(name))
        require(len(candidates) == 1, f"Expected one {name}, found {len(candidates)} under {extracted}")
        located[name] = candidates[0]
    require(len({path.parent for path in located.values()}) == 1, "Prediction, metrics and CSV must come from the same formal evaluation")
    prediction_path = located["swinir_prediction_csi.npy"]
    test_csv = located["test_samples.csv"]
    csi_json_path = located["independent_test_metrics.json"]
    saved = read_json(csi_json_path)
    require(saved["samples"] == EXPECTED_SHAPE[0], "SwinIR JSON sample count mismatch")
    require(Path(saved["checkpoint"]).name == "best_epoch_037.pth", "Expected epoch-37 checkpoint")
    require(saved["full_prediction"]["checkpoint"] == saved["checkpoint"], "Prediction/checkpoint metadata mismatch")
    require(tuple(saved["full_prediction"]["shape"]) == EXPECTED_SHAPE and saved["full_prediction"]["dtype"] == "float32", "Saved prediction shape/dtype metadata mismatch")
    require(saved["metric_reference"]["normalization"] == "none" and saved["metric_reference"]["representation"] == "raw_csi", "SwinIR metadata must specify raw, unnormalized CSI")
    csi = {key: float(saved["csi"][key]) for key in ("psnr", "ssim")}
    require(all(math.isfinite(value) for value in csi.values()), "Nonfinite saved CSI metrics")

    formal = args.formal_drct_root
    formal_config = read_json(formal / "run_config.json")
    formal_metrics = read_json(formal / "metrics.json")
    require(read_json(formal / "status.json")["state"] == "complete" and read_json(formal / "verification_report.json")["passed"] is True, "Original formal DRCT run is not complete/verified")
    require(formal_metrics["is_full_test"] is True and formal_metrics["samples"] == EXPECTED_SHAPE[0], "Wrong formal DRCT test population")
    require(sha256_file(formal / "export_test_predictions.py") == FORMAL_SCRIPT_SHA256 == formal_config["script_sha256"], "Formal GHI protocol script hash mismatch")
    require(formal_config["numerics"]["csi_multiplier"] == 1.0 and formal_config["numerics"]["normalization"] == "none" and formal_config["numerics"]["clipping"] is False, "Formal numerical protocol differs")
    require(saved["metric_reference"] == formal_config["metric_reference"], "CSI metric-reference metadata differs from formal DRCT")
    csv_hash = sha256_file(test_csv)
    require(csv_hash == EXPECTED_CSV_SHA256 == saved["full_prediction"]["test_csv_sha256"] == formal_config["data_identity"]["split_csv_sha256"]["test"], "Formal cleaned test CSV SHA256 mismatch")
    rows = read_csv(test_csv)
    require(len(rows) == EXPECTED_SHAPE[0], "Formal CSV must contain exactly 35476 samples")
    original_csv = args.dataset_root / "splits/test_samples.csv"
    original_rows = read_csv(original_csv)
    require(sha256_file(original_csv) == formal_config["data_identity"]["original_test_csv_sha256"], "Original local CSV changed from formal DRCT run")
    indices_path = Path(formal_config["arguments"]["clear_indices"])
    require(sha256_file(indices_path) == formal_config["data_identity"]["clear_indices_sha256"], "Existing cleaned-test index array changed")
    indices = np.load(indices_path, allow_pickle=False)
    require(indices.shape == (len(rows),) and np.issubdtype(indices.dtype, np.integer), "Invalid existing cleaned-test indices")
    require(np.all(np.diff(indices.astype(np.int64)) > 0) and int(indices[0]) >= 0 and int(indices[-1]) < len(original_rows), "Invalid cleaned-index range/order")
    for position, original_index in enumerate(indices):
        require(rows[position] == original_rows[int(original_index)], f"Existing CSV/index alignment mismatch at row {position}")
    # This only validates the existing mapping; no dataset/CSV/indices are built.
    print(f"IDENTITY PASS: epoch 37; {len(rows):,} cleaned test rows; original CSV has {len(original_rows):,} rows", flush=True)

    clear_path = args.dataset_root / "splits/test_clear_ghi_clean.npy"
    clear_metadata_path = args.dataset_root / "splits/test_clear_ghi.json"
    require(clear_path.resolve() == Path(formal_config["arguments"]["clear_ghi"]).resolve(), "Clear-GHI path differs from formal DRCT run")
    clear_metadata = read_json(clear_metadata_path)
    require(clear_metadata == formal_config["ghi"]["clear_provenance"], "Clear-GHI provenance changed from formal DRCT run")
    training_source = Path(formal_config["arguments"]["code_root"]) / "utils/training.py"
    accumulator_class = load_original_accumulator(training_source, formal_config["code_sha256"]["utils/training.py"])
    tracked = {str(path): stat_identity(path) for path in [*located.values(), original_csv, indices_path, clear_path, clear_metadata_path, training_source, formal / "metrics.json", formal / "run_config.json", formal / "export_test_predictions.py"]}
    prediction = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
    clear = np.load(clear_path, mmap_mode="r", allow_pickle=False)
    try:
        require(prediction.shape == clear.shape == EXPECTED_SHAPE, f"Wrong array shapes: prediction={prediction.shape}, clear={clear.shape}")
        require(prediction.dtype == clear.dtype == np.dtype("float32"), "Prediction/clear arrays must both be float32")
        print("PRECHECK: scanning every prediction and clear-GHI value with read-only memory maps ...", flush=True)
        prediction_scan = scan_array(prediction, "prediction CSI", 64)
        clear_scan = scan_array(clear, "clear GHI", 64, positive=True)
        prediction_hash = sha256_file(prediction_path)
        require(prediction_hash == EXPECTED_PREDICTION_SHA256 == saved["full_prediction"]["sha256"], "Prediction file SHA256 mismatch")
        require(clear_scan["payload_sha256"] == formal_metrics["selected_clear_ghi_payload_sha256"], "Clear-GHI payload differs from original formal DRCT evaluation")
        print(f"PRECHECK PASS: shape={EXPECTED_SHAPE}; all finite; negative CSI retained={prediction_scan['negative']:,}; SHA256 verified", flush=True)

        groups = OrderedDict()
        identities, sample_ids = set(), set()
        for position, row in enumerate(rows):
            name, date = row["npz_file"], row["date"]
            require(len(date) == 8 and date.isdigit() and date[:4] in ("2018", "2019"), f"Invalid date at CSV row {position}")
            require(Path(name).name == name and "\\" not in name and name.startswith(date + "_") and name.endswith(".npz"), f"Unsafe/unexpected NPZ path at CSV row {position}")
            member = int(row["array_index"])
            require(member >= 0, f"Negative array index at CSV row {position}")
            identity = (date, name, member)
            require(identity not in identities and row["sample_id"] not in sample_ids, f"Duplicate sample identity at CSV row {position}")
            identities.add(identity)
            sample_ids.add(row["sample_id"])
            path = args.dataset_root / "samples" / date / name
            groups.setdefault(path, []).append((position, member))

        accumulator = accumulator_class()
        processed = np.zeros(len(rows), dtype=bool)
        per_patch_valid = np.zeros(len(rows), dtype=np.int32)
        diagnostics = {"reference_nonfinite": 0, "reference_negative": 0, "reference_fill_values": 0, "reference_above_1500_retained": 0, "invalid_reference_pixels": 0, "prediction_ghi_negative_retained": 0, "target_zero_csi_pixels": 0, "max_abs_hr_csi_times_clear_minus_fy3": 0.0}
        source_records = {}
        reference_min, reference_max = math.inf, -math.inf
        ghi_min, ghi_max = math.inf, -math.inf
        loop_started = time.perf_counter()
        for group_number, (path, members) in enumerate(groups.items(), 1):
            require(path.is_file(), f"Missing source NPZ: {path}")
            before = stat_identity(path)
            source_records[str(path)] = before
            arrays = {}
            try:
                with zipfile.ZipFile(path) as archive:
                    for name in ("fy3", "hr_csi"):
                        arrays[name] = npz_member(path, archive, name)
                    source_shape = arrays["fy3"].shape
                    require(len(source_shape) == 3 and source_shape[1:] == (128, 128) and 1 <= source_shape[0] <= 64, f"Unexpected FY3 shape: {path}: {source_shape}")
                    require(arrays["hr_csi"].shape == source_shape and all(value.dtype == np.float32 for value in arrays.values()), f"Source shape/dtype mismatch: {path}")
                    for start in range(0, len(members), args.batch_size):
                        batch = members[start:start + args.batch_size]
                        positions = np.array([item[0] for item in batch], dtype=np.int64)
                        array_indices = np.array([item[1] for item in batch], dtype=np.int64)
                        require(int(array_indices.max()) < source_shape[0], f"NPZ array_index out of bounds: {path}, CSV rows {positions.tolist()}")
                        require(not processed[positions].any(), "A sample was processed twice")
                        observed = np.asarray(arrays["fy3"][array_indices])
                        target_csi = np.asarray(arrays["hr_csi"][array_indices])
                        require(np.isfinite(target_csi).all(), f"Nonfinite HR CSI: {path}, CSV rows {positions.tolist()}")
                        predicted_csi = np.asarray(prediction[positions])
                        clear_batch = np.asarray(clear[positions])
                        with np.errstate(over="raise", invalid="raise"):
                            predicted_ghi = predicted_csi * clear_batch  # float32, exactly as formal DRCT.
                            reconstructed_reference = target_csi * clear_batch
                        require(predicted_ghi.dtype == np.float32 and np.isfinite(predicted_ghi).all(), f"Nonfinite GHI prediction at CSV rows {positions.tolist()}")
                        observed_valid = np.isfinite(observed) & (observed >= 0) & ~np.isin(observed, FILL_VALUES)
                        mask = observed_valid & np.isfinite(clear_batch) & (clear_batch > 0) & np.isfinite(predicted_ghi)
                        valid_per_patch = mask.reshape(len(batch), -1).sum(axis=1)
                        require(np.all(valid_per_patch > 0), f"A sample has no valid GHI pixels: CSV rows {positions.tolist()}")
                        require(np.allclose(reconstructed_reference[observed_valid], observed[observed_valid], rtol=2e-6, atol=0.001), f"Clear/FY3/HR-CSI alignment check failed: CSV rows {positions.tolist()}")
                        diagnostics["reference_nonfinite"] += int(np.count_nonzero(~np.isfinite(observed)))
                        diagnostics["reference_negative"] += int(np.count_nonzero(observed < 0))
                        diagnostics["reference_fill_values"] += int(np.count_nonzero(np.isin(observed, FILL_VALUES)))
                        diagnostics["reference_above_1500_retained"] += int(np.count_nonzero(mask & (observed > 1500)))
                        diagnostics["invalid_reference_pixels"] += int(np.count_nonzero(~observed_valid))
                        diagnostics["prediction_ghi_negative_retained"] += int(np.count_nonzero(mask & (predicted_ghi < 0)))
                        diagnostics["target_zero_csi_pixels"] += int(np.count_nonzero(target_csi == 0))
                        diagnostics["max_abs_hr_csi_times_clear_minus_fy3"] = max(diagnostics["max_abs_hr_csi_times_clear_minus_fy3"], float(np.abs(reconstructed_reference[observed_valid] - observed[observed_valid]).max()))
                        accumulator.update(observed[mask], predicted_ghi[mask])
                        reference_min = min(reference_min, float(observed[mask].min()))
                        reference_max = max(reference_max, float(observed[mask].max()))
                        ghi_min = min(ghi_min, float(predicted_ghi[mask].min()))
                        ghi_max = max(ghi_max, float(predicted_ghi[mask].max()))
                        per_patch_valid[positions] = valid_per_patch
                        processed[positions] = True
                require(stat_identity(path) == before, f"Source NPZ changed while reading: {path}")
            finally:
                for values in arrays.values():
                    close_mmap(values)
            if group_number == 1 or group_number % 400 == 0 or group_number == len(groups):
                done = int(processed.sum())
                elapsed = time.perf_counter() - loop_started
                print(f"PROGRESS: {done:,}/{len(rows):,} samples; {group_number:,}/{len(groups):,} NPZ groups; {done/max(elapsed, 1e-9):.1f} samples/s; elapsed {elapsed:.1f}s", flush=True)

        require(processed.all(), f"Samples not evaluated: {np.flatnonzero(~processed).tolist()[:20]}")
        require(accumulator.n == int(per_patch_valid.sum()) == formal_metrics["valid_pixels"] == math.prod(EXPECTED_SHAPE), "Valid population differs from original formal DRCT evaluation")
        require(diagnostics["invalid_reference_pixels"] == formal_metrics["invalid_observation_pixels"] == 0, "Reference-mask population differs from original formal DRCT")
        require(diagnostics["target_zero_csi_pixels"] == formal_metrics["target_zero_csi_pixels"], "Reference CSI identity differs from formal DRCT")
        # Compare each sample's mask size with the original formal DRCT record.
        with (formal / "per_patch_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
            old_rows = list(csv.DictReader(handle))
        require(len(old_rows) == len(rows), "Original per-patch record count differs")
        require(all(int(old["csv_row_index"]) == index and old["sample_id"] == rows[index]["sample_id"] and int(old["ghi_valid_pixels"]) == int(per_patch_valid[index]) for index, old in enumerate(old_rows)), "Per-patch valid populations differ from formal DRCT")
        for name, before in {**tracked, **source_records}.items():
            require(stat_identity(Path(name)) == before, f"Source changed during evaluation: {name}")
        ghi = accumulator.metrics()
        ghi["mbe"] = ghi.pop("bias")
        require(all(math.isfinite(value) for value in ghi.values()), "Nonfinite final GHI metric")
        result = {
            "method": "SwinIR", "checkpoint_epoch": 37, "checkpoint": saved["checkpoint"],
            "state": "complete", "full_test": True, "samples": len(rows), "valid_pixels": accumulator.n,
            "created": created, "elapsed_seconds": time.perf_counter() - started,
            "csi": csi, "ghi": ghi,
            "units": {"csi_psnr": "dB", "csi_ssim": "dimensionless", "ghi_r2": "dimensionless", "ghi_rmse_mae_mbe": "W/m²"},
            "evaluation_protocol": {
                "reconstruction": "GHI_pred = raw_CSI_pred * GHI_clear; float32 multiplication",
                "normalization": "none", "csi_multiplier": 1.0, "clipping": False,
                "mask": "FY3 finite AND >=0 AND not -999/65532/65535; clear GHI finite AND >0; GHI prediction finite",
                "anomaly_handling": "Nonfinite prediction/clear or nonpositive clear raises; no silently skipped samples. Invalid FY3 pixels are counted explicitly; changed formal population raises.",
                "negative_predictions": "retained", "zero_reference": "retained", "upper_irradiance_cap": None,
                "aggregation": "Global valid pixels with original float64 RegressionAccumulator; not a mean of patch-level metrics",
                "r2": "1 - SSE / max(sum(y^2) - sum(y)^2/N, 1e-12)",
                "rmse": "sqrt(SSE/N)", "mae": "sum(abs(prediction-reference))/N", "mbe": "sum(prediction-reference)/N",
                "csi_metrics": "Read unchanged from independent_test_metrics.json at csi.psnr and csi.ssim; not recomputed",
                "csi_data_range": saved["metric_reference"]["data_range"],
                "clear_ghi_provenance": clear_metadata,
                "clear_ghi_provenance_caveat": "Existing metadata says fy3/hr_csi with median recovery of 15707 zero-CSI pixels. This is NOT established as an independently recomputed native REST2 output. The clean array is bit-identical to the one used by the formal DRCT evaluation.",
            },
            "comparability_to_formal_drct": {
                "confirmed": True, "formal_result_directory": str(formal),
                "same_test_csv_sha256": csv_hash, "same_clean_clear_payload_sha256": clear_scan["payload_sha256"],
                "same_reference_mask_every_patch": True, "same_original_accumulator": True,
                "formal_protocol_script_sha256": FORMAL_SCRIPT_SHA256,
                "accumulator_source": str(training_source), "accumulator_source_sha256": sha256_file(training_source),
                "numerical_difference": "NPZ-grouped batches change float64 addition order relative to original CSV-ordered two-patch batches; last-bit reduction rounding may differ. Definitions, mask and statistical population are unchanged.",
            },
            "validation": {"prediction": prediction_scan, "clear_ghi": clear_scan, "diagnostics": diagnostics, "all_samples_evaluated_once": True, "no_samples_skipped": True, "original_csv_samples": len(original_rows), "existing_clean_index_mapping_verified": True, "npz_groups": len(groups), "reference_ghi_min": reference_min, "reference_ghi_max": reference_max, "predicted_ghi_min": ghi_min, "predicted_ghi_max": ghi_max, "all_source_size_mtime_unchanged": True},
            "global_sufficient_statistics": {"n": accumulator.n, "sum_reference": accumulator.sum_observation, "sum_reference_sq": accumulator.sum_observation_sq, "sum_error_sq": accumulator.sum_error_sq, "sum_abs_error": accumulator.sum_abs_error, "sum_error": accumulator.sum_error},
            "inputs": {"prediction": str(prediction_path), "prediction_file_sha256": prediction_hash, "test_csv": str(test_csv), "test_csv_sha256": csv_hash, "original_test_csv": str(original_csv), "existing_clean_indices": str(indices_path), "clear_ghi": str(clear_path), "csi_metrics_json": str(csi_json_path), "csi_metrics_json_sha256": sha256_file(csi_json_path), "npz_member_mapping": "samples/<date>/<npz_file>, member fy3[array_index]", "source_npz_stat_manifest_sha256": hashlib.sha256(json.dumps(source_records, sort_keys=True).encode("utf-8")).hexdigest()},
            "execution": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(), "batch_size": args.batch_size, "read_only_memmaps": True, "script_sha256": sha256_file(Path(__file__))},
            "outputs": {"json": str(json_path), "csv": str(csv_path), "script": str(Path(__file__).resolve())},
        }
        row_values = ["SwinIR", csi["psnr"], csi["ssim"], ghi["r2"], ghi["rmse"], ghi["mae"], ghi["mbe"]]
        json_tmp, csv_tmp = json_path.with_suffix(".json.tmp"), csv_path.with_suffix(".csv.tmp")
        with json_tmp.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        with csv_tmp.open("x", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(SUMMARY_COLUMNS)
            writer.writerow(row_values)
        with csv_tmp.open(encoding="utf-8-sig", newline="") as handle:
            readback = list(csv.reader(handle))
        require(len(readback) == 2 and readback[0] == SUMMARY_COLUMNS and readback[1][0] == "SwinIR" and [float(value) for value in readback[1][1:]] == row_values[1:], "Summary CSV readback mismatch")
        require(read_json(json_tmp)["ghi"] == ghi, "JSON readback mismatch")
        os.replace(csv_tmp, csv_path)
        os.replace(json_tmp, json_path)
        print(json.dumps({"state": "complete", "samples": len(rows), "valid_pixels": accumulator.n, "csi": csi, "ghi": ghi, "elapsed_seconds": result["elapsed_seconds"], "outputs": result["outputs"]}, ensure_ascii=False, indent=2), flush=True)
    finally:
        close_mmap(prediction)
        close_mmap(clear)


if __name__ == "__main__":
    main()
