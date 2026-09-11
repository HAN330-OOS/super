"""Export the fixed DRCT train/validation/test splits into self-contained NPY datasets.

The production FY pipeline stores samples in date/batch NPZ archives.  A batch
can contain members of several fixed DRCT splits, so moving those archives into
three directories would be incorrect.  This exporter writes the exact selected
members into three independent, memory-mappable directories instead.

Default destination (outside the Git repository):

    E:\\Super-Resolution-Processed\\
        train/       50,000 model-input pairs
        validation/  10,000 official pairs + 5 fixed visual pairs
        test/         10,000 independent pairs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "DRCT") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "DRCT"))

from data.dataset_SR_satellite import CSI_SCALE
from utils.utils_paths import FY_SAMPLES_ROOT


DATASET_ROOT = PROJECT_ROOT / "DRCT" / "datasets" / "fy4a_fy3c_csi_x4"
TRAIN_MANIFEST = DATASET_ROOT / "train" / "samples.json"
VALIDATION_MANIFEST = DATASET_ROOT / "validation" / "samples.json"
TEST_MANIFEST = DATASET_ROOT / "test" / "samples.json"
FIXED_VISUAL_MANIFEST = PROJECT_ROOT / "DRCT" / "exp" / "fixed_visual_samples.json"

LR_SHAPE = (1, 32, 32)
HR_SHAPE = (1, 128, 128)
PATCH_SIZE = 128
FORMAT_VERSION = 1


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_manifest(path: Path, expected_count: int) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    samples = list(payload.get("samples", []))
    if len(samples) != expected_count:
        raise ValueError(f"{path} contains {len(samples)} samples, expected {expected_count}")
    return payload, sha256_bytes(raw)


def direct_record(source: dict[str, Any], position: int, role: str) -> dict[str, Any]:
    return {
        "index": int(position),
        "source_index": int(source["index"]),
        "date": str(source["date"]),
        "batch_file": str(source["batch_file"]),
        "batch_index": int(source["batch_index"]),
        "role": role,
    }


def prepared_records() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, str],
]:
    train, train_hash = load_manifest(TRAIN_MANIFEST, 50_000)
    validation, validation_hash = load_manifest(VALIDATION_MANIFEST, 10_000)
    test, test_hash = load_manifest(TEST_MANIFEST, 10_000)
    visual_raw = FIXED_VISUAL_MANIFEST.read_bytes()
    visual = json.loads(visual_raw.decode("utf-8"))

    train_rows = [direct_record(row, position, "train") for position, row in enumerate(train["samples"])]
    validation_rows = [
        direct_record(row, position, "validation")
        for position, row in enumerate(validation["samples"])
    ]
    test_rows = [direct_record(row, position, "test") for position, row in enumerate(test["samples"])]

    occupied = {
        (row["date"], row["batch_file"], row["batch_index"])
        for row in validation_rows + test_rows
    }
    for visual_position, row in enumerate(visual.get("samples", [])):
        source = {
            "index": int(row["val_index"]),
            "date": str(row["date"]),
            "batch_file": str(row["batch_file"]),
            "batch_index": int(row["batch_index"]),
        }
        key = (source["date"], source["batch_file"], source["batch_index"])
        if key in occupied:
            raise ValueError(f"Fixed visual sample unexpectedly overlaps validation/test: {key}")
        validation_rows.append(
            direct_record(source, 10_000 + visual_position, "fixed_visual")
        )
        occupied.add(key)
    if len(validation_rows) != 10_005:
        raise ValueError("Expected exactly five additional fixed visual samples")

    return train_rows, validation_rows, test_rows, {
        "train": train_hash,
        "validation": validation_hash,
        "test": test_hash,
        "fixed_visual": sha256_bytes(visual_raw),
    }


def source_archive(samples_root: Path, record: dict[str, Any]) -> Path:
    return (
        samples_root
        / f"{record['date']}_utc"
        / "patches"
        / str(record["batch_file"])
    )


def rest2_directory(date_dir: Path, cache: dict[Path, Path]) -> Path:
    cached = cache.get(date_dir)
    if cached is not None:
        return cached
    metadata_path = date_dir / "build_meta.npz"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing sample build metadata: {metadata_path}")
    with np.load(metadata_path, allow_pickle=False) as metadata:
        if "rest2_dir" not in metadata.files:
            raise KeyError(f"{metadata_path} is missing rest2_dir")
        directory = Path(str(metadata["rest2_dir"].item()))
    if not directory.exists():
        raise FileNotFoundError(f"REST2 directory recorded by {metadata_path} does not exist: {directory}")
    cache[date_dir] = directory
    return directory


def flush_and_publish(temporary: Path, published: Path) -> None:
    if not temporary.exists():
        raise FileNotFoundError(f"Expected temporary export file: {temporary}")
    os.replace(temporary, published)


def export_split(
    *,
    split_name: str,
    records: list[dict[str, Any]],
    source_root: Path,
    destination_root: Path,
    source_hashes: dict[str, str],
    include_clear_ghi: bool,
) -> None:
    split_dir = destination_root / split_name
    split_dir.mkdir(parents=True, exist_ok=False)
    sample_count = len(records)
    lr_temporary = split_dir / "lr_csi.tmp.npy"
    hr_temporary = split_dir / "hr_csi.tmp.npy"
    clear_temporary = split_dir / "clear_ghi.tmp.npy"
    lr_path = split_dir / "lr_csi.npy"
    hr_path = split_dir / "hr_csi.npy"
    clear_path = split_dir / "clear_ghi.npy"

    lr_output = np.lib.format.open_memmap(
        lr_temporary, mode="w+", dtype=np.float32, shape=(sample_count, *LR_SHAPE)
    )
    hr_output = np.lib.format.open_memmap(
        hr_temporary, mode="w+", dtype=np.float32, shape=(sample_count, *HR_SHAPE)
    )
    clear_output = (
        np.lib.format.open_memmap(
            clear_temporary, mode="w+", dtype=np.float32, shape=(sample_count, PATCH_SIZE, PATCH_SIZE)
        )
        if include_clear_ghi
        else None
    )

    grouped: dict[Path, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for position, record in enumerate(records):
        if int(record["index"]) != position:
            raise ValueError(f"{split_name} record {position} has a non-contiguous index")
        archive_path = source_archive(source_root, record)
        if not archive_path.exists():
            raise FileNotFoundError(f"Source archive does not exist: {archive_path}")
        grouped[archive_path].append((position, record))

    clear_specs: dict[Path, list[tuple[int, int, int]]] = defaultdict(list)
    rest2_dirs: dict[Path, Path] = {}
    print(
        f"[{split_name}] writing {sample_count:,} samples from {len(grouped):,} NPZ batches ...",
        flush=True,
    )
    started = time.time()
    for archive_number, (archive_path, members) in enumerate(grouped.items(), 1):
        output_positions = np.fromiter((position for position, _ in members), dtype=np.int64)
        batch_positions = np.fromiter(
            (int(record["batch_index"]) for _, record in members), dtype=np.int64
        )
        with np.load(archive_path, allow_pickle=False) as archive:
            required = {"lr_csi", "hr_csi"}
            if include_clear_ghi:
                required.update({"row0", "col0", "dominant_slot_minute_utc"})
            missing = required.difference(archive.files)
            if missing:
                raise KeyError(f"{archive_path} is missing {sorted(missing)}")
            lr_values = np.clip(
                archive["lr_csi"][batch_positions].astype(np.float32, copy=False), 0.0, CSI_SCALE
            ) / CSI_SCALE
            hr_values = np.clip(
                archive["hr_csi"][batch_positions].astype(np.float32, copy=False), 0.0, CSI_SCALE
            ) / CSI_SCALE
            if lr_values.shape != (len(members), 32, 32) or hr_values.shape != (len(members), 128, 128):
                raise ValueError(
                    f"Unexpected CSI shapes in {archive_path}: lr={lr_values.shape}, hr={hr_values.shape}"
                )
            if not np.isfinite(lr_values).all() or not np.isfinite(hr_values).all():
                raise FloatingPointError(f"NaN/Inf in CSI source archive: {archive_path}")
            lr_output[output_positions, 0] = lr_values
            hr_output[output_positions, 0] = hr_values

            if include_clear_ghi:
                row0 = archive["row0"][batch_positions].astype(np.int64, copy=False)
                col0 = archive["col0"][batch_positions].astype(np.int64, copy=False)
                minutes = archive["dominant_slot_minute_utc"][batch_positions].astype(np.int64, copy=False)
                date_dir = archive_path.parents[1]
                ghi_dir = rest2_directory(date_dir, rest2_dirs)
                for output_position, row, col, minute in zip(output_positions, row0, col0, minutes):
                    if minute < 0 or minute >= 24 * 60:
                        raise ValueError(f"Invalid REST2 UTC minute {minute} in {archive_path}")
                    label = f"{minute // 60:02d}{minute % 60:02d}"
                    clear_specs[ghi_dir / f"REST2_{label}_on_FY3.npz"].append(
                        (int(output_position), int(row), int(col))
                    )
        if archive_number % 250 == 0 or archive_number == len(grouped):
            print(f"  [{split_name}] CSI {archive_number:,}/{len(grouped):,} batches", flush=True)

    if clear_output is not None:
        print(f"  [{split_name}] extracting clear-GHI patches from {len(clear_specs):,} REST2 fields ...", flush=True)
        for field_number, (clear_file, specifications) in enumerate(clear_specs.items(), 1):
            if not clear_file.exists():
                raise FileNotFoundError(f"REST2 field does not exist: {clear_file}")
            with np.load(clear_file, allow_pickle=False) as field:
                if "ghi" not in field.files:
                    raise KeyError(f"{clear_file} is missing ghi")
                ghi = field["ghi"].astype(np.float32, copy=False)
                for output_position, row, col in specifications:
                    patch = ghi[row : row + PATCH_SIZE, col : col + PATCH_SIZE]
                    if patch.shape != (PATCH_SIZE, PATCH_SIZE):
                        raise ValueError(f"Invalid clear-GHI crop {patch.shape} from {clear_file}")
                    if not np.isfinite(patch).all():
                        raise FloatingPointError(f"NaN/Inf in clear-GHI crop from {clear_file}")
                    clear_output[output_position] = patch
            if field_number % 25 == 0 or field_number == len(clear_specs):
                print(f"  [{split_name}] clear GHI {field_number:,}/{len(clear_specs):,} fields", flush=True)

    lr_output.flush()
    hr_output.flush()
    del lr_output, hr_output
    if clear_output is not None:
        clear_output.flush()
        del clear_output
    flush_and_publish(lr_temporary, lr_path)
    flush_and_publish(hr_temporary, hr_path)
    if include_clear_ghi:
        flush_and_publish(clear_temporary, clear_path)

    arrays = {
        "lr_csi": np.load(lr_path, mmap_mode="r"),
        "hr_csi": np.load(hr_path, mmap_mode="r"),
    }
    if arrays["lr_csi"].shape != (sample_count, *LR_SHAPE):
        raise ValueError(f"Published LR shape is invalid: {arrays['lr_csi'].shape}")
    if arrays["hr_csi"].shape != (sample_count, *HR_SHAPE):
        raise ValueError(f"Published HR shape is invalid: {arrays['hr_csi'].shape}")
    if include_clear_ghi:
        clear_check = np.load(clear_path, mmap_mode="r")
        if clear_check.shape != (sample_count, PATCH_SIZE, PATCH_SIZE):
            raise ValueError(f"Published clear-GHI shape is invalid: {clear_check.shape}")
        del clear_check
    del arrays

    role_counts: dict[str, int] = defaultdict(int)
    for record in records:
        role_counts[str(record["role"])] += 1
    write_json(
        split_dir / "samples.json",
        {
            "format_version": FORMAT_VERSION,
            "split": split_name,
            "sample_count": sample_count,
            "samples": records,
        },
    )
    write_json(
        split_dir / "metadata.json",
        {
            "format_version": FORMAT_VERSION,
            "split": split_name,
            "sample_count": sample_count,
            "role_counts": dict(sorted(role_counts.items())),
            "source_samples_root": str(source_root.resolve()),
            "source_manifest_sha256": source_hashes[split_name],
            "fixed_visual_spec_sha256": source_hashes["fixed_visual"] if split_name == "validation" else None,
            "normalization": "clip(csi, 0.0, 1.5) / 1.5",
            "dtype": "float32",
            "lr_csi_shape": [sample_count, *LR_SHAPE],
            "hr_csi_shape": [sample_count, *HR_SHAPE],
            "clear_ghi_shape": [sample_count, PATCH_SIZE, PATCH_SIZE] if include_clear_ghi else None,
            "clear_ghi_units": "W m^-2" if include_clear_ghi else None,
            "created_unix": time.time(),
        },
    )
    print(f"[{split_name}] complete in {(time.time() - started) / 60:.1f} min", flush=True)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument(
        "--destination",
        type=Path,
        default=Path(r"E:\\Super-Resolution-Processed"),
        help="New root directory for the self-contained processed dataset.",
    )
    command.add_argument(
        "--source-samples-root",
        type=Path,
        default=FY_SAMPLES_ROOT,
        help="Existing outputs/fy/samples directory containing batched NPZ source samples.",
    )
    return command


def main() -> None:
    args = parser().parse_args()
    destination = args.destination.resolve()
    source_root = args.source_samples_root.resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"Source samples root does not exist: {source_root}")
    if destination.exists():
        raise FileExistsError(
            f"Destination already exists and will not be overwritten: {destination}"
        )
    building = destination.with_name(destination.name + ".building")
    if building.exists():
        raise FileExistsError(
            f"Incomplete export directory exists; inspect it before retrying: {building}"
        )
    if destination.drive != building.drive:
        raise RuntimeError("Temporary and final dataset directories must be on the same drive")

    train_records, validation_records, test_records, source_hashes = prepared_records()
    print(f"Source samples: {source_root}", flush=True)
    print(f"Destination: {destination}", flush=True)
    building.mkdir(parents=True)
    try:
        export_split(
            split_name="train",
            records=train_records,
            source_root=source_root,
            destination_root=building,
            source_hashes=source_hashes,
            include_clear_ghi=False,
        )
        export_split(
            split_name="validation",
            records=validation_records,
            source_root=source_root,
            destination_root=building,
            source_hashes=source_hashes,
            include_clear_ghi=True,
        )
        export_split(
            split_name="test",
            records=test_records,
            source_root=source_root,
            destination_root=building,
            source_hashes=source_hashes,
            include_clear_ghi=True,
        )
        (building / "README.md").write_text(
            "# DRCT processed data\\n\\n"
            "This directory is generated by `DRCT/data/export_fixed_splits.py`.\\n"
            "It contains the exact fixed DRCT train, validation, and test splits.\\n"
            "Use it by setting `DRCT_PROCESSED_DATA_ROOT` to this directory.\\n",
            encoding="utf-8",
        )
        os.replace(building, destination)
    except Exception:
        print(f"Export stopped; incomplete files were retained for inspection: {building}", flush=True)
        raise
    print(f"Prepared DRCT dataset published: {destination}", flush=True)


if __name__ == "__main__":
    main()
