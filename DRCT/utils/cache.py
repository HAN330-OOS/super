"""Build a local, memory-mapped CSI cache from the existing 2018--2019 splits.

Example:
    python cache_npz.py --source /data/Super-Resolution-Samples-2018-2019 \
        --destination /data/drct_csi_cache

The source is read only. The destination must not exist. CSV membership,
order and raw CSI values are preserved exactly, without scaling or clipping.
The cache contains CSI pairs only, not physical SSI or matched clear-sky GHI.
Its metric reference is the fixed empirical range of raw training HR targets;
validation and test values never contribute to that reference.
Interrupted destinations retain a .building marker and must be inspected;
this command never overwrites or resumes an existing destination.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from data.dataset_csi import (
    SPLITS,
    file_hash,
    load_npz_splits,
    npz_data_identity,
)


FORMAT = "npz_csi_cache_v2_raw"
LR_SHAPE = (1, 32, 32)
HR_SHAPE = (1, 128, 128)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_new_json(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _existing_ancestor(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise FileNotFoundError(f"No existing destination volume: {path}")
        candidate = parent
    if not candidate.is_dir():
        raise NotADirectoryError(candidate)
    return candidate


def _copy_exact(source: Path, destination: Path, expected_sha256: str) -> None:
    with source.open("rb") as input_handle, destination.open("xb") as output_handle:
        shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
        output_handle.flush()
        os.fsync(output_handle.fileno())
    if file_hash(destination) != expected_sha256:
        raise RuntimeError(f"Source metadata changed during conversion: {source}")


def _training_metric_reference(raw_min: float, raw_max: float, *, training_csv_sha256: str,
                               training_samples: int, preview_only: bool) -> dict:
    finite = bool(np.isfinite(raw_min) and np.isfinite(raw_max))
    raw_range = raw_max - raw_min if finite else None
    reference = {
        "format_version": 2,
        "policy": "training_target_range",
        "kind": "empirical_fixed_reference_not_physical_bound",
        "raw_min": raw_min if finite else None,
        "raw_max": raw_max if finite else None,
        "raw_data_range": raw_range,
        "data_range": raw_range,
        "representation": "raw_csi",
        "normalization": "none",
        "source_split": "train",
        "reference_kind": "empirical fixed training reference; not physical bound",
        "training_csv_sha256": training_csv_sha256,
        "training_manifest_sha256": training_csv_sha256,
        "training_manifest": "splits/train_samples.csv",
        "training_samples": training_samples,
        "training_csv_samples": training_samples,
        "selected_train_indices": None,
        "preview_only": preview_only,
    }
    if raw_range is None or not np.isfinite(raw_range) or raw_range <= 0:
        reference["status"] = "requires_explicit_raw_range"
        reference["reason"] = (
            "A positive finite range cannot be derived from training HR targets; "
            "formal training/evaluation requires an explicit raw data range."
        )
    return reference


def convert_cache(source: Path, destination: Path) -> dict:
    """Validate metadata, then create an entirely new CSI cache directory."""
    source = Path(source).expanduser().resolve()
    requested_destination = Path(destination).expanduser()
    if requested_destination.exists() or requested_destination.is_symlink():
        raise FileExistsError(f"Destination already exists; nothing was overwritten: {requested_destination}")
    destination = requested_destination.resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"Source dataset does not exist: {source}")
    if destination.exists():
        raise FileExistsError(f"Destination already exists; nothing was overwritten: {destination}")
    if destination == source or source in destination.parents:
        raise ValueError("Destination must be outside the read-only source dataset")
    if (source / "cloud_cache.json").exists():
        raise ValueError("Source must be the raw CSV/NPZ dataset, not an existing cache")

    # This validates every row, split disjointness, and all referenced paths
    # without reading any array payload or creating destination files.
    datasets = load_npz_splits(source)
    counts = {split: len(dataset) for split, dataset in zip(SPLITS, datasets)}
    total_samples = sum(counts.values())
    identity = npz_data_identity(source)
    split_hashes = {
        split: identity["files"][f"splits/{split}_samples.csv"]["sha256"]
        for split in SPLITS
    }
    summary_path = source / "status" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    if "sample_count" in summary and int(summary["sample_count"]) != total_samples:
        raise ValueError(f"Summary sample_count differs from CSV rows: {summary['sample_count']} != {total_samples}")
    split_protocol = summary.get("split", {})
    for split, count in counts.items():
        declared = split_protocol.get(f"{split}_samples")
        if declared is not None and int(declared) != count:
            raise ValueError(f"Summary {split} count differs from CSV rows: {declared} != {count}")

    # A source archive can contain samples from all three splits. Read each
    # archive once and scatter its selected members into the original CSV rows.
    grouped: dict[tuple[str, str], list[tuple[str, int, int]]] = defaultdict(list)
    for split, dataset in zip(SPLITS, datasets):
        for position, row in enumerate(dataset.rows):
            grouped[(row["date"], row["npz_file"])].append((split, position, int(row["array_index"])))

    array_bytes = total_samples * (np.prod(LR_SHAPE).item() + np.prod(HR_SHAPE).item()) * 4 + len(SPLITS) * 2 * 128
    metadata_bytes = sum(
        int(value["bytes"])
        for name, value in identity["files"].items()
        if name.startswith("splits/") or name == "status/summary.json"
    )
    # Leave room for JSON provenance and filesystem allocation overhead.
    reserve_bytes = max(64 * 1024**2, array_bytes // 100)
    required_bytes = int(array_bytes + metadata_bytes + reserve_bytes)
    volume = _existing_ancestor(destination.parent)
    free_bytes = shutil.disk_usage(volume).free
    if free_bytes < required_bytes:
        raise OSError(
            f"Insufficient free space on {volume}: need {required_bytes:,} bytes "
            f"including reserve; available {free_bytes:,} bytes"
        )

    print(
        f"Preflight passed: {total_samples:,} samples, {len(grouped):,} source archives; "
        f"cache arrays {array_bytes / 1024**3:.3f} GiB, free {free_bytes / 1024**3:.3f} GiB",
        flush=True,
    )
    destination.mkdir(parents=True, exist_ok=False)
    marker_path = destination / ".building"
    _write_new_json(marker_path, {
        "format": FORMAT, "state": "building", "started_utc": _utc_now(),
        "source": str(source), "sample_counts": counts,
    })
    started = time.monotonic()
    maps: dict[str, tuple[np.memmap, np.memmap]] = {}
    training_raw_min = float("inf")
    training_raw_max = -float("inf")
    source_summary_hash = identity["files"].get("status/summary.json", {}).get("sha256")
    try:
        (destination / "splits").mkdir()
        for split in SPLITS:
            _copy_exact(
                source / "splits" / f"{split}_samples.csv",
                destination / "splits" / f"{split}_samples.csv",
                split_hashes[split],
            )
            split_dir = destination / split
            split_dir.mkdir()
            maps[split] = (
                np.lib.format.open_memmap(split_dir / "lr_csi.npy", mode="w+", dtype=np.float32, shape=(counts[split], *LR_SHAPE)),
                np.lib.format.open_memmap(split_dir / "hr_csi.npy", mode="w+", dtype=np.float32, shape=(counts[split], *HR_SHAPE)),
            )
        if summary_path.exists():
            (destination / "status").mkdir()
            _copy_exact(summary_path, destination / "status" / "summary.json", source_summary_hash)

        for number, ((date, filename), members) in enumerate(sorted(grouped.items()), start=1):
            relative = f"samples/{date}/{filename}"
            archive_path = source / "samples" / date / filename
            expected_size = int(identity["files"][relative]["bytes"])
            if archive_path.stat().st_size != expected_size:
                raise RuntimeError(f"Source archive size changed after preflight: {archive_path}")
            with np.load(archive_path, allow_pickle=False) as archive:
                if not {"lr_csi", "hr_csi"}.issubset(archive.files):
                    raise KeyError(f"Source archive is missing CSI arrays: {archive_path}")
                low = np.asarray(archive["lr_csi"])
                high = np.asarray(archive["hr_csi"])
            if low.dtype != np.float32 or high.dtype != np.float32:
                raise TypeError(f"Source CSI arrays must be raw float32: {archive_path}")
            if low.ndim != 3 or low.shape[1:] != (32, 32) or high.shape != (low.shape[0], 128, 128) or not 1 <= low.shape[0] <= 64:
                raise ValueError(f"Invalid source CSI shapes: {archive_path}: {low.shape}, {high.shape}")
            if any(index >= low.shape[0] for _, _, index in members):
                raise IndexError(f"CSV array_index is outside source batch: {archive_path}")
            # Preserve the raw float32 payload exactly. No scaling, clipping,
            # imputation, channel mixing, or physical-unit reconstruction.
            for split in SPLITS:
                selected = [(position, index) for role, position, index in members if role == split]
                if not selected:
                    continue
                positions = np.fromiter((position for position, _ in selected), dtype=np.int64)
                indices = np.fromiter((index for _, index in selected), dtype=np.int64)
                low_selected = low[indices]
                high_selected = high[indices]
                if not np.isfinite(low_selected).all() or not np.isfinite(high_selected).all():
                    raise FloatingPointError(f"Non-finite selected CSI values: {archive_path}")
                if split == "train":
                    # Select raw TRAIN HR members explicitly. An archive may
                    # also contain validation/test outliers, which must never
                    # influence the fixed metric reference. Do not infer a raw
                    # range from transformed values or from LR inputs.
                    training_targets = high[indices]
                    training_raw_min = min(training_raw_min, float(training_targets.min()))
                    training_raw_max = max(training_raw_max, float(training_targets.max()))
                low_output, high_output = maps[split]
                low_output[positions, 0] = low_selected
                high_output[positions, 0] = high_selected
            if archive_path.stat().st_size != expected_size:
                raise RuntimeError(f"Source archive changed during read: {archive_path}")
            if number % 100 == 0 or number == len(grouped):
                print(f"Converted {number:,}/{len(grouped):,} archives in {time.monotonic() - started:.1f}s", flush=True)

        for low_output, high_output in maps.values():
            low_output.flush()
            high_output.flush()
        for arrays in maps.values():
            for array in arrays:
                array._mmap.close()
        maps.clear()
        # Finish file synchronization and verify published NPY headers before
        # writing the only metadata that identifies this as a complete cache.
        output_arrays = {}
        for split in SPLITS:
            output_arrays[split] = {}
            for name, shape in (("lr_csi", LR_SHAPE), ("hr_csi", HR_SHAPE)):
                path = destination / split / f"{name}.npy"
                with path.open("r+b") as handle:
                    os.fsync(handle.fileno())
                checked = np.load(path, mmap_mode="r", allow_pickle=False)
                if checked.shape != (counts[split], *shape) or checked.dtype != np.float32:
                    raise RuntimeError(f"Invalid output NPY header: {path}")
                output_arrays[split][name] = {"shape": list(checked.shape), "dtype": "float32", "bytes": path.stat().st_size}
                checked._mmap.close()
        for split in SPLITS:
            if file_hash(source / "splits" / f"{split}_samples.csv") != split_hashes[split]:
                raise RuntimeError(f"Source split CSV changed during conversion: {split}")
        if source_summary_hash and file_hash(summary_path) != source_summary_hash:
            raise RuntimeError("Source summary changed during conversion")
        metric_reference = _training_metric_reference(
            training_raw_min, training_raw_max, training_csv_sha256=split_hashes["train"],
            training_samples=counts["train"], preview_only=bool(identity["preview_only"]),
        )
        if metric_reference.get("status") == "requires_explicit_raw_range":
            print(f"Metric reference unavailable: {metric_reference['reason']}", flush=True)
        metadata = {
            "format": FORMAT, "state": "complete", "completed_utc": _utc_now(),
            "source_root": str(source), "sample_count": total_samples, "sample_counts": counts,
            "split_csv_sha256": split_hashes, "preview_only": bool(identity["preview_only"]),
            "split": split_protocol, "normalization": "none", "representation": "raw_csi",
            "metric_reference": metric_reference,
            "dtype": "float32", "arrays": output_arrays,
            "source_identity": identity,
            "provenance_scope": "Exact split CSV SHA256 and source archive file sizes; no full NPZ payload hashes.",
            "physical_ghi_available": False,
        }
        _write_new_json(destination / "cloud_cache.json", metadata)
        marker_path.unlink()
        print(f"Complete CSI cache: {destination} ({time.monotonic() - started:.1f}s)", flush=True)
        return metadata
    except BaseException:
        # Keep incomplete files for inspection. A failed conversion never
        # receives complete metadata and never removes source data.
        for arrays in maps.values():
            for array in arrays:
                try:
                    array._mmap.close()
                except Exception:
                    pass
        print(f"Conversion stopped; incomplete destination retained: {destination}", flush=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="Raw samples/splits dataset root")
    parser.add_argument("--destination", type=Path, required=True, help="New local cache directory; must not exist")
    args = parser.parse_args()
    convert_cache(args.source, args.destination)


if __name__ == "__main__":
    main()
