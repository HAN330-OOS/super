"""CSI metrics with an explicitly frozen empirical training-target range.

CSI has no assumed [0, 1] bound. Values are never clipped here. This module
imports NumPy/skimage only; torch tensors are accepted through their public
detach/cpu/numpy methods without importing torch or BasicSR.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity

POLICY = "training_target_range"
KIND = "empirical_fixed_reference_not_physical_bound"
CACHE_FORMAT = "npz_csi_cache_v2_raw"


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    with path.open(encoding="utf-8-sig") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return result


def _selection(indices, count: int):
    if indices is None:
        return None
    if not isinstance(indices, list) or not indices:
        raise ValueError("train_indices must be None or a nonempty list of TRAIN row positions")
    if any(isinstance(index, bool) or not isinstance(index, (int, np.integer)) for index in indices):
        raise ValueError("TRAIN row positions must be integers")
    selected = [int(index) for index in indices]
    if len(set(selected)) != len(selected) or min(selected) < 0 or max(selected) >= count:
        raise ValueError("TRAIN row selection has duplicates or out-of-range positions")
    return selected


def _training_source(root: Path, train_indices=None) -> dict:
    root = Path(root).expanduser().resolve()
    csv_path = root / "splits" / "train_samples.csv"
    cache_path = root / "cloud_cache.json"
    cache = _json(cache_path) if cache_path.exists() else None
    summary_path = root / "status" / "summary.json"
    summary = _json(summary_path) if summary_path.exists() else {}
    if csv_path.is_file():
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"date", "npz_file", "array_index"}.issubset(reader.fieldnames or ()):
                raise ValueError("TRAIN CSV requires date, npz_file, array_index")
            rows = list(reader)
        if not rows:
            raise ValueError("TRAIN CSV is empty")
        keys = set()
        for row in rows:
            date, filename = row["date"], row["npz_file"]
            if len(date) != 8 or not date.isdigit() or Path(filename).name != filename or "\\" in filename or not filename.endswith(".npz"):
                raise ValueError("Unsafe date/archive name in TRAIN CSV")
            index = int(row["array_index"])
            key = (date, filename, index)
            if index < 0 or key in keys:
                raise ValueError("Negative or duplicate archive member in TRAIN CSV")
            keys.add(key)
        manifest, manifest_hash = csv_path, _hash(csv_path)
        if cache is not None:
            if cache.get("format") != CACHE_FORMAT or cache.get("state") != "complete":
                raise ValueError("CSI cache is incomplete, old normalized format, or unrecognized; raw CSI cache v2 is required")
            if cache.get("representation") != "raw_csi" or cache.get("normalization") != "none":
                raise ValueError("CSI cache must explicitly store raw_csi with normalization='none'")
            if cache.get("split_csv_sha256", {}).get("train") != manifest_hash:
                raise ValueError("Cached TRAIN CSV SHA256 does not match its source metadata")
            if int(cache.get("sample_counts", {}).get("train", -1)) != len(rows):
                raise ValueError("Cached TRAIN sample count does not match its CSV")
        source_format = "npz_csv" if cache is None else "npy_cache"
        csv_hash = manifest_hash
    else:
        raise ValueError("Raw CSI TRAIN CSV is required; legacy normalized prepared data is incompatible")
    selected = _selection(train_indices, len(rows))
    return {"root": root, "rows": rows, "selected": selected,
            "manifest": manifest, "manifest_hash": manifest_hash, "csv_hash": csv_hash,
            "format": source_format, "cache": cache,
            "preview_only": bool(summary.get("preview_only") or (cache or {}).get("preview_only"))}


def _positive_range(value, label="data_range") -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be a positive finite number")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be a positive finite number") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be positive and finite; supply an explicit raw CSI metric range")
    return value


def _validate_range(reference: dict) -> None:
    if reference.get("policy") not in (POLICY, "user_declared_fixed_reference") or reference.get("source_split") != "train":
        raise ValueError("Metric reference must describe TRAIN targets or an explicitly declared fixed range")
    if reference.get("representation") != "raw_csi" or reference.get("normalization") != "none":
        raise ValueError("Metric reference must use raw_csi with normalization='none'; old scaled references are incompatible")
    raw_range = _positive_range(reference.get("raw_data_range"), "raw_data_range")
    data_range = _positive_range(reference.get("data_range"))
    if not math.isclose(data_range, raw_range, rel_tol=1e-12, abs_tol=0.0):
        raise ValueError("Metric data_range must equal raw CSI range; CSI is not rescaled")
    if reference.get("policy") == "user_declared_fixed_reference":
        return
    try:
        low, high = float(reference["raw_min"]), float(reference["raw_max"])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("Metric reference requires finite raw training extrema") from error
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        raise ValueError("Training targets do not define a positive finite range; supply an explicit raw CSI metric range")
    if not math.isclose(raw_range, high - low, rel_tol=1e-12, abs_tol=0.0):
        raise ValueError("Raw range and training extrema disagree")


def _reference(source: dict, low: float, high: float, method: str) -> dict:
    reference = {"format_version": 2, "policy": POLICY, "kind": KIND, "source_split": "train",
                 "raw_min": low, "raw_max": high, "raw_data_range": high - low,
                 "data_range": high - low, "representation": "raw_csi", "normalization": "none",
                 "training_csv_sha256": source["csv_hash"],
                 "training_manifest_sha256": source["manifest_hash"],
                 "training_manifest": source["manifest"].relative_to(source["root"]).as_posix(),
                 "training_csv_samples": len(source["rows"]),
                 "training_samples": len(source["selected"]) if source["selected"] is not None else len(source["rows"]),
                 "selected_train_indices": source["selected"], "preview_only": bool(source["preview_only"] or source["selected"] is not None),
                 "dataset_format": source["format"], "fit_method": method,
                 "created_utc": datetime.now(timezone.utc).isoformat(),
                 "reference_scope": "Frozen empirical range of selected TRAIN HR CSI targets; not a physical bound; no clipping"}
    _validate_range(reference)
    return reference


def fit_training_reference(root: Path, train_indices: list[int] | None = None) -> dict:
    """Fit TRAIN HR extrema; open each selected raw NPZ archive only once.

    ``train_indices`` indexes TRAIN CSV rows, not archive members or validation
    rows. None means the entire existing TRAIN split. Old scaled NPY caches are
    rejected; only explicitly identified raw CSI caches may be scanned/reused.
    """
    source = _training_source(root, train_indices)
    selected = source["selected"] if source["selected"] is not None else list(range(len(source["rows"])))
    if source["cache"] is not None and source["selected"] is None:
        cached = source["cache"].get("metric_reference")
        if isinstance(cached, dict):
            if cached.get("policy") != POLICY:
                raise ValueError("Cached empirical fitting requires a TRAIN-target range, not a user-declared range")
            _validate_range(cached)
            if cached.get("selected_train_indices") is not None:
                raise ValueError("A subset metric reference cannot represent the complete cached TRAIN split")
            if cached.get("training_csv_sha256", source["csv_hash"]) != source["csv_hash"]:
                raise ValueError("Cached metric reference TRAIN CSV SHA256 mismatch")
            if int(cached.get("training_samples", len(source["rows"]))) != len(source["rows"]):
                raise ValueError("Cached metric reference TRAIN count mismatch")
            return _reference(source, float(cached["raw_min"]), float(cached["raw_max"]), "cache_metadata_training_raw_extrema")
    low, high = math.inf, -math.inf
    if source["format"] == "npz_csv":
        grouped = defaultdict(list)
        for position in selected:
            row = source["rows"][position]
            grouped[(row["date"], row["npz_file"])].append(int(row["array_index"]))
        for (date, filename), members in grouped.items():
            path = source["root"] / "samples" / date / filename
            with np.load(path, allow_pickle=False) as archive:
                targets = archive["hr_csi"]
                if targets.dtype != np.float32 or targets.ndim != 3 or targets.shape[1:] != (128, 128):
                    raise ValueError(f"Expected raw float32 HR CSI (N,128,128): {path}")
                if max(members) >= len(targets):
                    raise ValueError(f"TRAIN CSV array_index exceeds archive size: {path}")
                values = targets[np.asarray(members, dtype=np.int64)]
                if not np.isfinite(values).all():
                    raise ValueError(f"Non-finite TRAIN HR CSI: {path}; supply/review an explicit raw CSI metric range")
                low, high = min(low, float(values.min())), max(high, float(values.max()))
        method = "raw_npz_train_csv_grouped_selected_members_only"
    else:
        path = source["root"] / "train" / "hr_csi.npy"
        targets = np.load(path, mmap_mode="r", allow_pickle=False)
        try:
            if targets.dtype != np.float32 or targets.shape != (len(source["rows"]), 1, 128, 128):
                raise ValueError(f"Expected raw float32 TRAIN HR NPY (N,1,128,128): {path}")
            for start in range(0, len(selected), 256):
                values = targets[np.asarray(selected[start : start + 256], dtype=np.int64)]
                if not np.isfinite(values).all():
                    raise ValueError("Non-finite raw TRAIN HR CSI; an automatic reference is unavailable")
                low, high = min(low, float(values.min())), max(high, float(values.max()))
        finally:
            if isinstance(targets, np.memmap):
                targets._mmap.close()
        method = "raw_train_npy_memmap_extrema"
    if _hash(source["manifest"]) != source["manifest_hash"]:
        raise ValueError("TRAIN manifest changed during metric-reference fitting; rerun on a fixed dataset")
    return _reference(source, low, high, method)


def load_metric_reference(path: Path, root: Path, train_indices: list[int] | None = None) -> dict:
    """Validate a frozen reference against this exact TRAIN manifest/selection."""
    reference = _json(Path(path))
    source = _training_source(root, train_indices)
    _validate_range(reference)
    explicit = reference["policy"] == "user_declared_fixed_reference"
    expected = {"kind": "explicit_reference_not_a_physical_bound" if explicit else KIND,
                "training_csv_sha256": source["csv_hash"],
                "training_samples": len(source["selected"]) if source["selected"] is not None else len(source["rows"]),
                "selected_train_indices": source["selected"]}
    extra = {"format_version": 2, "training_manifest_sha256": source["manifest_hash"],
             "training_manifest": source["manifest"].relative_to(source["root"]).as_posix(),
             "training_csv_samples": len(source["rows"])}
    for key, value in extra.items():
        if not explicit or key in reference:
            expected[key] = value
    preview = bool(source["preview_only"] or source["selected"] is not None)
    if not explicit or reference.get("preview_only") or source["selected"] is not None:
        expected["preview_only"] = preview
    for key, value in expected.items():
        if key not in reference or reference[key] != value:
            raise ValueError(f"Metric reference mismatch: {key}; a preview/subset or different TRAIN CSV cannot be reused")
    return reference


def _images(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 4 and result.shape[1] == 1:
        result = result[:, 0]
    if result.ndim != 3 or result.shape[0] == 0:
        raise ValueError("CSI metrics require a nonempty (N,1,H,W) or (N,H,W) batch")
    if not np.isfinite(result).all():
        raise ValueError("CSI metric inputs must be finite; no clipping or finite-value masking is applied")
    return result


def batch_psnr_ssim(prediction, target, *, data_range: float) -> tuple[list[float], list[float]]:
    """Per-image PSNR/SSIM; the CSI range is required, never inferred.

    Float64 MSE is used; zero error gives standard positive-infinity PSNR.
    Predictions and targets may lie outside the frozen training reference.
    """
    data_range = _positive_range(data_range)
    prediction, target = _images(prediction), _images(target)
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target CSI batch shapes differ")
    mse = np.square(prediction - target).mean(axis=(1, 2), dtype=np.float64)
    if not np.isfinite(mse).all():
        raise ValueError("CSI MSE overflowed despite finite inputs")
    psnr = [math.inf if error == 0 else 20.0 * math.log10(data_range) - 10.0 * math.log10(float(error)) for error in mse]
    ssim = [float(structural_similarity(pred, obs, data_range=data_range)) for pred, obs in zip(prediction, target)]
    return psnr, ssim
