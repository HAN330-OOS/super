"""Export raw-CSI predictions and GHI statistics for the formal 600-day test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import platform
import random
import shutil
import struct
import sys
import time
import traceback
import uuid
import zipfile
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKUP_ROOT = PROJECT_ROOT / "external_data"
DATASET_ROOT = BACKUP_ROOT
EXPECTED_COUNTS = {"train": 177380, "validation": 35482, "test": 35476}
PATCH_PIXELS = 128 * 128
SHANGHAI = timezone(timedelta(hours=8))
FILL_VALUES = (-999.0, 65532.0, 65535.0)


def timestamp() -> str:
    return datetime.now(SHANGHAI).isoformat(timespec="seconds")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def json_number(value: float) -> float | str:
    if math.isfinite(value):
        return value
    return "+Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--code-root", type=Path, default=PROJECT_ROOT / "DRCT")
    parser.add_argument("--checkpoint", type=Path, default=BACKUP_ROOT / "checkpoints/best_epoch_065.pth")
    parser.add_argument("--splits-root", type=Path, default=PROJECT_ROOT / "splits")
    parser.add_argument("--samples-root", type=Path, default=DATASET_ROOT / "samples")
    parser.add_argument("--clear-ghi", type=Path, default=DATASET_ROOT / "splits/test_clear_ghi_clean.npy")
    parser.add_argument("--original-test-csv", type=Path, default=DATASET_ROOT / "splits/test_samples.csv")
    parser.add_argument("--clear-indices", type=Path, default=BACKUP_ROOT / "test_clear_ghi_indices.npy")
    parser.add_argument("--clear-metadata", type=Path, default=DATASET_ROOT / "splits/test_clear_ghi.json")
    parser.add_argument("--reference-evaluation", type=Path, default=BACKUP_ROOT / "evaluation/best_epoch_065_20261028_160928_dfcfc555")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "evaluation")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--scatter-seed", type=int, default=60)
    parser.add_argument("--scatter-max-points", type=int, default=150000)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    for name in ("batch_size", "scatter_max_points", "cpu_threads", "log_every"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.seed < 0 or args.scatter_seed < 0:
        parser.error("Seeds must be nonnegative")
    return args


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        required = {"sample_id", "date", "npz_file", "array_index"}
        if not required.issubset(fields):
            raise ValueError(f"Missing CSV fields: {path}")
        return fields, list(reader)


def validate_splits(args: argparse.Namespace) -> tuple[list[str], list[dict], dict]:
    hashes = {}
    all_ids, all_members = set(), set()
    selected_fields, test_rows = [], []
    for split, expected in EXPECTED_COUNTS.items():
        path = args.splits_root / f"{split}_samples.csv"
        fields, rows = read_csv(path)
        if len(rows) != expected:
            raise ValueError(f"Formal {split} requires {expected} rows; found {len(rows)}")
        for record in rows:
            date, filename = record["date"], record["npz_file"]
            array_index = int(record["array_index"])
            sample_id = int(record["sample_id"])
            if len(date) != 8 or not date.isdigit() or date[:4] not in ("2018", "2019"):
                raise ValueError(f"Unexpected formal date: {date}")
            if Path(filename).name != filename or "\\" in filename or not filename.startswith(date + "_") or not filename.endswith(".npz"):
                raise ValueError(f"Unexpected NPZ name: {filename}")
            member = (date, filename, array_index)
            if array_index < 0 or sample_id < 0 or sample_id in all_ids or member in all_members:
                raise ValueError(f"Duplicate or invalid sample: {member}")
            all_ids.add(sample_id)
            all_members.add(member)
        hashes[split] = file_sha256(path)
        if split == "test":
            selected_fields, test_rows = fields, rows
    _, original_rows = read_csv(args.original_test_csv)
    indices = np.load(args.clear_indices, allow_pickle=False)
    if indices.shape != (len(test_rows),) or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("Invalid cleaned-test index array")
    if int(indices[0]) < 0 or int(indices[-1]) >= len(original_rows) or not np.all(np.diff(indices.astype(np.int64)) > 0):
        raise ValueError("Cleaned-test indices must be increasing, unique and in range")
    for position, original_index in enumerate(indices):
        if test_rows[position] != original_rows[int(original_index)]:
            raise ValueError(f"CSV/clear-GHI index mismatch at row {position}")
    return selected_fields, test_rows, {
        "counts": EXPECTED_COUNTS,
        "split_csv_sha256": hashes,
        "original_test_csv_sha256": file_sha256(args.original_test_csv),
        "clear_indices_sha256": file_sha256(args.clear_indices),
        "clean_csv_matches_original_indices": True,
    }


class PatchReader:
    def __init__(self, samples_root: Path, capacity: int = 8):
        self.samples_root = samples_root
        self.capacity = capacity
        self.cache = OrderedDict()
        self.source_files = {}

    def close(self) -> None:
        for arrays in self.cache.values():
            self.close_arrays(arrays)
        self.cache.clear()

    @staticmethod
    def close_arrays(arrays: dict) -> None:
        for values in arrays.values():
            if isinstance(values, np.memmap):
                values._mmap.close()

    @staticmethod
    def stored_array(path: Path, entry: zipfile.ZipInfo) -> np.ndarray:
        with path.open("rb") as handle:
            handle.seek(entry.header_offset)
            header = handle.read(30)
            if len(header) != 30 or header[:4] != b"PK\x03\x04":
                raise ValueError(f"Invalid ZIP local header: {path}")
            filename_length, extra_length = struct.unpack_from("<HH", header, 26)
            handle.seek(filename_length + extra_length, 1)
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
            elif version == (2, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
            else:
                raise ValueError(f"Unsupported NPY version in {path}: {version}")
            offset = handle.tell()
        if dtype.hasobject or dtype != np.dtype("float32"):
            raise TypeError(f"Expected float32 source array: {path}/{entry.filename}")
        return np.memmap(path, mode="r", dtype=dtype, offset=offset, shape=shape, order="F" if fortran_order else "C")

    def arrays(self, record: dict) -> dict:
        key = (record["date"], record["npz_file"])
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        path = self.samples_root / key[0] / key[1]
        arrays = {}
        try:
            with zipfile.ZipFile(path) as archive:
                for name in ("lr_csi", "hr_csi", "fy3"):
                    entry = archive.getinfo(name + ".npy")
                    if entry.compress_type == zipfile.ZIP_STORED:
                        arrays[name] = self.stored_array(path, entry)
                    else:
                        with archive.open(entry) as handle:
                            arrays[name] = np.load(handle, allow_pickle=False)
            count = arrays["lr_csi"].shape[0]
            for name, expected in (("lr_csi", (count, 32, 32)), ("hr_csi", (count, 128, 128)), ("fy3", (count, 128, 128))):
                if arrays[name].shape != expected or arrays[name].dtype != np.float32:
                    raise ValueError(f"Source shape/dtype mismatch: {path}/{name}")
        except BaseException:
            self.close_arrays(arrays)
            raise
        if len(self.cache) >= self.capacity:
            _, previous = self.cache.popitem(last=False)
            self.close_arrays(previous)
        self.cache[key] = arrays
        stat = path.stat()
        self.source_files[str(path)] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        return arrays

    def batch(self, records: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lows, targets, observations = [], [], []
        for record in records:
            arrays = self.arrays(record)
            array_index = int(record["array_index"])
            if array_index >= len(arrays["lr_csi"]):
                raise IndexError(f"Source array_index out of bounds: {record}")
            lows.append(np.array(arrays["lr_csi"][array_index], copy=True))
            targets.append(np.array(arrays["hr_csi"][array_index], copy=True))
            observations.append(np.array(arrays["fy3"][array_index], copy=True))
        return np.stack(lows)[:, None], np.stack(targets), np.stack(observations)


def summary(values: np.ndarray) -> dict:
    finite = np.isfinite(values)
    available = values[finite]
    return {
        "shape": list(values.shape), "dtype": str(values.dtype),
        "nonfinite": int((~finite).sum()), "negative": int((values < 0).sum()),
        "zero": int((values == 0).sum()),
        "min": float(available.min()) if available.size else None,
        "max": float(available.max()) if available.size else None,
    }


def metrics_with_mbe(accumulator) -> dict:
    result = accumulator.metrics()
    result["mbe"] = result.pop("bias")
    result["valid_pixels"] = accumulator.n
    return result


def run(args: argparse.Namespace, output: Path) -> None:
    started = time.perf_counter()
    log_handle = (output / "run.log").open("x", encoding="utf-8", buffering=1)

    def log(message: str) -> None:
        line = f"{timestamp()} {message}"
        print(line, flush=True)
        log_handle.write(line + "\n")

    reader = None
    prediction_store = None
    status = {"state": "running", "mode": args.mode, "processed_samples": 0, "started": timestamp()}
    write_json(output / "status.json", status)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU is required; CPU inference is not used by this runner")
        fields, test_rows, identity = validate_splits(args)
        count = 2 if args.mode == "smoke" else len(test_rows)
        records = test_rows[:count]
        clear_ghi = np.load(args.clear_ghi, mmap_mode="r", allow_pickle=False)
        if clear_ghi.shape != (EXPECTED_COUNTS["test"], 128, 128) or clear_ghi.dtype != np.float32:
            raise ValueError("Clear GHI must be float32 (35476,128,128)")
        sys.path.insert(0, str(args.code_root.resolve()))
        helpers = importlib.import_module("utils.training")
        metrics_module = importlib.import_module("utils.metrics")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        network = helpers.load_network_options()
        if checkpoint.get("network") != network or checkpoint.get("config", {}).get("network") != network:
            raise ValueError("Checkpoint network/config differs from the original architecture")
        if checkpoint.get("epoch") != 65 or checkpoint["config"].get("data_counts") != EXPECTED_COUNTS | {"fixed_visual": 0}:
            raise ValueError("Expected the formal epoch-65 checkpoint with the cleaned split counts")
        config_source = checkpoint["config"]
        if config_source.get("normalization") != "none" or config_source.get("representation") != "raw_csi" or config_source.get("value_clipping") is not False:
            raise ValueError("Checkpoint must declare raw CSI without normalization or clipping")
        reference = config_source["metric_reference"]
        if reference.get("training_csv_sha256") != identity["split_csv_sha256"]["train"]:
            raise ValueError("Frozen CSI metric reference was fitted on a different training CSV")
        data_range = float(reference["data_range"])
        random.seed(args.seed)
        np.random.seed(args.seed % (2**32))
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.set_num_threads(args.cpu_threads)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.use_deterministic_algorithms(False)
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        free_start, total_memory = torch.cuda.mem_get_info(device)
        torch.cuda.reset_peak_memory_stats(device)
        model = helpers.build_model(device)
        load_result = model.load_state_dict(checkpoint["params_ema"], strict=True)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count != 27382021:
            raise ValueError(f"Unexpected formal DRCT-L parameter count: {parameter_count}")
        model.eval()
        del checkpoint
        code_files = ["models/network_drct.py", "models/initialization.py", "utils/training.py", "utils/metrics.py", "exp/train_x4.json"]
        code_hashes = {name: file_sha256(args.code_root / name) for name in code_files}
        import cv2
        import skimage

        config = {
            "created": timestamp(), "mode": args.mode, "scope": "first_two_test_rows" if args.mode == "smoke" else "full_formal_test",
            "sample_count": count, "formal_test_count": len(test_rows),
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "data_identity": identity, "checkpoint_sha256": file_sha256(args.checkpoint),
            "code_sha256": code_hashes, "script_sha256": file_sha256(Path(__file__)),
            "checkpoint_epoch": 65, "weights_key": "params_ema", "strict_load": {"missing_keys": list(load_result.missing_keys), "unexpected_keys": list(load_result.unexpected_keys)},
            "parameter_count": parameter_count, "network": network, "metric_reference": reference,
            "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda_build": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "numpy": np.__version__, "opencv": cv2.__version__, "skimage": skimage.__version__, "gpu": torch.cuda.get_device_name(device), "gpu_capability": list(torch.cuda.get_device_capability(device)), "gpu_total_bytes": total_memory, "gpu_free_start_bytes": free_start},
            "numerics": {"precision": args.precision, "matmul_tf32": False, "cudnn_tf32": True, "cudnn_benchmark": False, "deterministic_algorithms": False, "normalization": "none", "csi_multiplier": 1.0, "clipping": False, "bicubic": "original utils.training.bicubic_batch: OpenCV INTER_CUBIC on CPU", "csi_metrics": "original utils.metrics.batch_psnr_ssim: float64, per-patch arithmetic mean, no border crop", "ghi_arithmetic": "float32 CSI multiplied by float32 clear GHI; float64 regression accumulation", "r2": "1 - sum((prediction-reference)^2) / sum((reference-mean(reference))^2)", "mbe": "mean(prediction-reference)"},
            "ghi": {"units": "W m-2", "observation": "stored fy3, unscaled float32 copied from FY-3C Gc DAT by the source pipeline", "observation_valid": "finite, >=0 and not equal to raw fill sentinels -999/65532/65535; zero is valid; no upper irradiance cap", "common_mask": "valid observed fy3 AND finite positive clear GHI AND finite Bicubic/DRCT GHI", "prediction_range_filter": "none; finite negative predictions remain included", "alignment": "clean test CSV == original CSV[clear_indices], and per-patch hr_csi * clear GHI agrees with stored fy3", "reconstruction_tolerance": {"rtol": 2e-6, "atol_w_m2": 0.001}, "clear_provenance": json.loads(args.clear_metadata.read_text(encoding="utf-8-sig"))},
            "scatter": {"seed": args.scatter_seed, "max_points": args.scatter_max_points, "selection": "uniform global pixel indices without replacement before inference; sorted into CSV/pixel order; common invalid pixels removed", "index_convention": "zero-based CSV row and pixel row/column; global_pixel_index=csv_row_index*16384+pixel_flat_index"},
            "future_figures": {"font_family": "Times New Roman", "font_size_pt": "set explicitly to the paper requirement at plotting time", "format": "PDF with embedded fonts and vector text/axes; raw arrays remain unchanged"},
        }
        write_json(output / "run_config.json", config)
        shutil.copyfile(Path(__file__), output / Path(__file__).name)
        readme = Path(__file__).with_name("README.md")
        if readme.is_file():
            shutil.copyfile(readme, output / "README.md")
        with (output / "test_order.csv").open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["prediction_index", "csv_row_index", *fields])
            writer.writeheader()
            for position, record in enumerate(records):
                writer.writerow({"prediction_index": position, "csv_row_index": position, **record})
        prediction_path = output / "drct_prediction_csi.partial.npy"
        prediction_store = np.lib.format.open_memmap(prediction_path, mode="w+", dtype=np.float32, shape=(count, 128, 128))
        rng = np.random.default_rng(args.scatter_seed)
        candidates = np.sort(rng.choice(count * PATCH_PIXELS, size=min(args.scatter_max_points, count * PATCH_PIXELS), replace=False))
        scatter = {name: [] for name in ("global_pixel_index", "csv_row_index", "sample_id", "pixel_flat_index", "pixel_row", "pixel_col", "fy3_reference", "bicubic_ghi", "drct_ghi", "clear_ghi", "target_csi", "bicubic_csi", "drct_csi")}
        accumulators = {name: helpers.RegressionAccumulator() for name in ("bicubic", "drct")}
        csi_values = {name: {"psnr": [], "ssim": []} for name in accumulators}
        max_alignment_error, invalid_observation_pixels, observed_above_1500 = 0.0, 0, 0
        target_zero_pixels = 0
        prediction_digest, input_digest, clear_digest = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
        reader = PatchReader(args.samples_root)
        per_patch_fields = ["csv_row_index", "sample_id", "bicubic_psnr", "bicubic_ssim", "drct_psnr", "drct_ssim", "ghi_valid_pixels", "ghi_reconstruction_max_abs_error_w_m2"]
        diagnostics = []
        subset_comparison = None
        log(f"START mode={args.mode}; patches={count}; batch={args.batch_size}; GPU={config['environment']['gpu']}; strict EMA load passed")
        gpu_seconds = 0.0
        with (output / "per_patch_metrics.csv").open("x", encoding="utf-8", newline="") as patch_handle:
            patch_writer = csv.DictWriter(patch_handle, fieldnames=per_patch_fields)
            patch_writer.writeheader()
            with torch.inference_mode():
                for start in range(0, count, args.batch_size):
                    end = min(start + args.batch_size, count)
                    low, target, observed = reader.batch(records[start:end])
                    clear = np.asarray(clear_ghi[start:end])
                    if not np.isfinite(low).all() or not np.isfinite(target).all():
                        raise FloatingPointError(f"Nonfinite input/target CSI in rows {start}:{end}")
                    if not np.isfinite(clear).all() or not (clear > 0).all():
                        raise FloatingPointError(f"Invalid clear GHI in rows {start}:{end}")
                    observed_valid = np.isfinite(observed) & (observed >= 0) & ~np.isin(observed, FILL_VALUES)
                    invalid_observation_pixels += int((~observed_valid).sum())
                    observed_above_1500 += int((observed_valid & (observed > 1500)).sum())
                    target_zero_pixels += int((target == 0).sum())
                    reconstruction = target * clear
                    if not np.allclose(reconstruction[observed_valid], observed[observed_valid], rtol=2e-6, atol=0.001):
                        raise ValueError(f"GHI/sample alignment check failed in CSV rows {start}:{end}")
                    error = np.abs(reconstruction - observed)
                    alignment_errors = [float(error[position][observed_valid[position]].max()) if observed_valid[position].any() else 0.0 for position in range(end - start)]
                    max_alignment_error = max(max_alignment_error, *alignment_errors)
                    low_tensor = torch.from_numpy(low)
                    torch.cuda.synchronize(device)
                    gpu_started = time.perf_counter()
                    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=args.precision == "fp16"):
                        predicted_tensor = model(low_tensor.to(device)).float().cpu()
                    torch.cuda.synchronize(device)
                    gpu_seconds += time.perf_counter() - gpu_started
                    if predicted_tensor.shape != (end - start, 1, 128, 128):
                        raise ValueError(f"Unexpected model output shape: {predicted_tensor.shape}")
                    prediction = predicted_tensor[:, 0].numpy()
                    bicubic = helpers.bicubic_batch(low_tensor)[:, 0].numpy()
                    if not np.isfinite(prediction).all() or not np.isfinite(bicubic).all():
                        raise FloatingPointError(f"Nonfinite prediction/Bicubic CSI in rows {start}:{end}")
                    predicted_ghi, bicubic_ghi = prediction * clear, bicubic * clear
                    if not np.isfinite(predicted_ghi).all() or not np.isfinite(bicubic_ghi).all():
                        raise FloatingPointError(f"GHI multiplication overflow in rows {start}:{end}")
                    common = observed_valid & np.isfinite(bicubic_ghi) & np.isfinite(predicted_ghi)
                    for method, values, ghi_values in (("bicubic", bicubic, bicubic_ghi), ("drct", prediction, predicted_ghi)):
                        psnr, ssim = metrics_module.batch_psnr_ssim(values, target, data_range=data_range)
                        csi_values[method]["psnr"].extend(psnr)
                        csi_values[method]["ssim"].extend(ssim)
                        accumulators[method].update(observed[common], ghi_values[common])
                    prediction_store[start:end] = prediction
                    prediction_store.flush()
                    prediction_digest.update(prediction.tobytes(order="C"))
                    for values in (low, target, observed):
                        input_digest.update(np.ascontiguousarray(values).tobytes())
                    clear_digest.update(clear.tobytes(order="C"))
                    for position in range(end - start):
                        csv_index = start + position
                        patch_writer.writerow({"csv_row_index": csv_index, "sample_id": records[csv_index]["sample_id"], "bicubic_psnr": csi_values["bicubic"]["psnr"][csv_index], "bicubic_ssim": csi_values["bicubic"]["ssim"][csv_index], "drct_psnr": csi_values["drct"]["psnr"][csv_index], "drct_ssim": csi_values["drct"]["ssim"][csv_index], "ghi_valid_pixels": int(common[position].sum()), "ghi_reconstruction_max_abs_error_w_m2": alignment_errors[position]})
                    selected = candidates[np.searchsorted(candidates, start * PATCH_PIXELS):np.searchsorted(candidates, end * PATCH_PIXELS)]
                    local_flat = selected - start * PATCH_PIXELS
                    selected = selected[common.reshape(-1)[local_flat]]
                    local_flat = selected - start * PATCH_PIXELS
                    csv_indices = selected // PATCH_PIXELS
                    pixel_indices = selected % PATCH_PIXELS
                    scatter["global_pixel_index"].append(selected.astype(np.int64))
                    scatter["csv_row_index"].append(csv_indices.astype(np.int32))
                    scatter["sample_id"].append(np.array([int(records[int(index)]["sample_id"]) for index in csv_indices], dtype=np.int64))
                    scatter["pixel_flat_index"].append(pixel_indices.astype(np.int32))
                    scatter["pixel_row"].append((pixel_indices // 128).astype(np.int16))
                    scatter["pixel_col"].append((pixel_indices % 128).astype(np.int16))
                    for name, values in (("fy3_reference", observed), ("bicubic_ghi", bicubic_ghi), ("drct_ghi", predicted_ghi), ("clear_ghi", clear), ("target_csi", target), ("bicubic_csi", bicubic), ("drct_csi", prediction)):
                        scatter[name].append(values.reshape(-1)[local_flat].astype(np.float32, copy=True))
                    if args.mode == "smoke":
                        diagnostics.append({"csv_rows": [start, end], "input": summary(low), "target": summary(target), "prediction": summary(prediction), "bicubic": summary(bicubic), "fy3": summary(observed), "clear_ghi": summary(clear), "drct_ghi": summary(predicted_ghi), "bicubic_ghi": summary(bicubic_ghi)})
                        saved_path = args.reference_evaluation / "csi_comparison_data.npz"
                        with np.load(saved_path, allow_pickle=False) as saved:
                            if not np.array_equal(low[:, 0], saved["low"][start:end]) or not np.array_equal(target, saved["target"][start:end]):
                                raise ValueError("Smoke inputs differ from the DSW comparison samples")
                            saved_prediction = saved["prediction"][start:end]
                            old_psnr, old_ssim = metrics_module.batch_psnr_ssim(saved_prediction, target, data_range=data_range)
                            diagnostics[-1]["dsw_comparison"] = {"same_input_and_target": True, "bicubic_exact_equal": bool(np.array_equal(bicubic, saved["bicubic"][start:end])), "prediction_max_abs_difference": float(np.max(np.abs(prediction - saved_prediction))), "prediction_mae_difference": float(np.mean(np.abs(prediction.astype(np.float64) - saved_prediction))), "saved_psnr": old_psnr, "saved_ssim": old_ssim}
                    status.update(processed_samples=end, total_samples=count, updated=timestamp())
                    write_json(output / "status.json", status)
                    if end == count or end % args.log_every < args.batch_size:
                        elapsed = time.perf_counter() - started
                        log(f"progress {end}/{count}; elapsed={elapsed:.2f}s; peak_allocated={torch.cuda.max_memory_allocated(device)/1024**2:.1f} MiB")
        prediction_store.flush()
        del prediction_store
        prediction_store = None
        final_path = output / "drct_prediction_csi.npy"
        prediction_path.replace(final_path)
        disk_prediction = np.load(final_path, mmap_mode="r", allow_pickle=False)
        readback_digest = hashlib.sha256()
        for start in range(0, count, 128):
            block = np.asarray(disk_prediction[start:start + 128])
            if not np.isfinite(block).all():
                raise FloatingPointError("Nonfinite saved prediction during readback")
            readback_digest.update(block.tobytes(order="C"))
        if readback_digest.hexdigest() != prediction_digest.hexdigest():
            raise ValueError("Saved prediction payload differs from model output")
        scatter_values = {name: np.concatenate(parts) for name, parts in scatter.items()}
        np.savez(output / "ghi_scatter_data.npz", **scatter_values)
        ghi_metrics = {method: metrics_with_mbe(accumulator) for method, accumulator in accumulators.items()}
        if not accumulators["drct"].n or accumulators["drct"].n != accumulators["bicubic"].n:
            raise ValueError("Common GHI pixel counts disagree or are empty")
        csi_metrics = {method: {metric: json_number(float(np.mean(values))) for metric, values in methods.items()} for method, methods in csi_values.items()}
        previous = json.loads((args.reference_evaluation / "independent_test_metrics.json").read_text(encoding="utf-8"))
        if args.mode == "smoke":
            old_psnrs = [number for diagnostic in diagnostics for number in diagnostic["dsw_comparison"]["saved_psnr"]]
            old_ssims = [number for diagnostic in diagnostics for number in diagnostic["dsw_comparison"]["saved_ssim"]]
            subset_comparison = {"scope": "same first two patches only", "saved_dsw_psnr": float(np.mean(old_psnrs)), "saved_dsw_ssim": float(np.mean(old_ssims)), "local_minus_saved_psnr_db": float(np.mean(csi_values["drct"]["psnr"])) - float(np.mean(old_psnrs)), "local_minus_saved_ssim": float(np.mean(csi_values["drct"]["ssim"])) - float(np.mean(old_ssims))}
        result = {
            "mode": args.mode, "samples": count, "is_full_test": args.mode == "full", "csi": csi_metrics,
            "ghi": ghi_metrics, "ghi_units": "W m-2", "mbe_sign": "prediction minus FY-3C reference",
            "valid_pixels": accumulators["drct"].n, "invalid_observation_pixels": invalid_observation_pixels,
            "observed_pixels_above_1500_not_removed": observed_above_1500,
            "target_zero_csi_pixels": target_zero_pixels,
            "max_abs_ghi_alignment_error_w_m2": max_alignment_error,
            "scatter": {"candidate_pixels": len(candidates), "saved_pixels": len(scatter_values["csv_row_index"]), "seed": args.scatter_seed},
            "prediction": {"file": final_path.name, "shape": list(disk_prediction.shape), "dtype": str(disk_prediction.dtype), "payload_sha256": readback_digest.hexdigest(), "readback_exact": True},
            "loaded_input_payload_sha256": input_digest.hexdigest(), "selected_clear_ghi_payload_sha256": clear_digest.hexdigest(),
            "gpu": {"name": torch.cuda.get_device_name(device), "total_bytes": total_memory, "peak_allocated_bytes": torch.cuda.max_memory_allocated(device), "peak_reserved_bytes": torch.cuda.max_memory_reserved(device), "free_end_bytes": torch.cuda.mem_get_info(device)[0], "inference_seconds": gpu_seconds},
            "elapsed_seconds": time.perf_counter() - started,
            "full_dsw_reference": {"samples": previous["samples"], "psnr": previous["csi"]["psnr"], "ssim": previous["csi"]["ssim"], "comparable_to_this_run": args.mode == "full"},
            "same_patch_dsw_comparison": subset_comparison,
            "difference_sources": ["DSW used a PPU CUDA-compatible backend; local execution uses NVIDIA CUDA/cuDNN", "Original test batch was 8; this run records its actual batch size", "FP16 autocast and device kernels can change rounding even with the same raw CSI and EMA weights", "PSNR/SSIM reference range and original metric functions are unchanged"],
            "smoke_diagnostics": diagnostics,
        }
        if args.mode == "full":
            result["full_dsw_reference"]["local_minus_saved_psnr_db"] = float(csi_metrics["drct"]["psnr"]) - previous["csi"]["psnr"]
            result["full_dsw_reference"]["local_minus_saved_ssim"] = float(csi_metrics["drct"]["ssim"]) - previous["csi"]["ssim"]
        write_json(output / "metrics.json", result)
        write_json(output / "source_files.json", reader.source_files)
        status.update(state="complete", completed=timestamp(), output_prediction=final_path.name, processed_samples=count)
        write_json(output / "status.json", status)
        log(f"COMPLETE CSI PSNR={csi_metrics['drct']['psnr']}; SSIM={csi_metrics['drct']['ssim']}; common GHI pixels={accumulators['drct'].n}; output={output}")
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    except BaseException as error:
        status.update(state="failed", updated=timestamp(), error=f"{type(error).__name__}: {error}")
        write_json(output / "status.json", status)
        (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        log(f"FAILED {type(error).__name__}: {error}")
        raise
    finally:
        if prediction_store is not None:
            prediction_store.flush()
        if reader is not None:
            reader.close()
        log_handle.close()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    for source in (args.samples_root, args.splits_root, args.code_root, args.checkpoint.parent):
        resolved = source.resolve()
        if output_root == resolved or resolved in output_root.parents:
            raise ValueError(f"Output root must be outside source directories: {resolved}")
    run_name = f"{args.mode}_{datetime.now(SHANGHAI):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    output = output_root / run_name
    output.mkdir(parents=True, exist_ok=False)
    print(f"OUTPUT_DIRECTORY={output}", flush=True)
    run(args, output)


if __name__ == "__main__":
    main()
