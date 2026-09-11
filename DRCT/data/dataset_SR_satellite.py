"""Read FY-4A/FY-3C CSI sample archives for the DRCT experiment.

Each ``outputs/fy/samples/<date>_utc`` directory contains a ``sample_index.csv``
and batched NPZ archives.  The CSV order is the canonical dataset order used by
the fixed training and validation manifests in ``DRCT/datasets``.
"""

from __future__ import annotations

import csv
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


CSI_SCALE = 1.5
_LR_SHAPE = (32, 32)
_HR_SHAPE = (128, 128)
_ARCHIVE_CACHE_SIZE = 4

SampleIndex = tuple[Path, int]


def _date_from_directory(directory: Path) -> str:
    name = directory.name
    if not name.endswith("_utc") or len(name) != len("YYYYMMDD_utc"):
        raise ValueError(f"Expected <YYYYMMDD>_utc sample directory, got {directory}")
    date = name.removesuffix("_utc")
    if not date.isdigit():
        raise ValueError(f"Invalid sample date directory: {directory}")
    return date


def _read_index(date_directory: Path) -> list[SampleIndex]:
    index_path = date_directory / "sample_index.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing sample index: {index_path}")

    entries: list[SampleIndex] = []
    with index_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"batch_file", "batch_index"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{index_path} must contain {sorted(required)}")
        for row_number, row in enumerate(reader, start=2):
            try:
                batch_index = int(row["batch_index"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid batch_index at {index_path}:{row_number}") from exc
            batch_file = (date_directory / "patches" / row["batch_file"]).resolve()
            entries.append((batch_file, batch_index))
    return entries


class SatelliteCSIDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """A read-only indexed view of normalized FY-4A LR / FY-3C HR CSI pairs."""

    def __init__(self, index: Sequence[SampleIndex], *, augment: bool = False) -> None:
        self.index = [(Path(path), int(batch_index)) for path, batch_index in index]
        self.augment = bool(augment)
        self._archive_cache: OrderedDict[Path, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._metadata_cache: OrderedDict[Path, dict[str, np.ndarray]] = OrderedDict()
        self._owner_pid: int | None = None

    def __getstate__(self) -> dict[str, object]:
        state = dict(self.__dict__)
        state["_archive_cache"] = OrderedDict()
        state["_metadata_cache"] = OrderedDict()
        state["_owner_pid"] = None
        return state

    def __len__(self) -> int:
        return len(self.index)

    def _archive_arrays(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        process_id = os.getpid()
        if self._owner_pid != process_id:
            self._archive_cache.clear()
            self._metadata_cache.clear()
            self._owner_pid = process_id

        cached = self._archive_cache.pop(path, None)
        if cached is not None:
            self._archive_cache[path] = cached
            return cached

        if not path.exists():
            raise FileNotFoundError(f"Missing sample archive: {path}")
        with np.load(path, allow_pickle=False) as archive:
            required = {"lr_csi", "hr_csi"}
            missing = required.difference(archive.files)
            if missing:
                raise KeyError(f"{path} is missing {sorted(missing)}")
            lr = np.asarray(archive["lr_csi"], dtype=np.float32)
            hr = np.asarray(archive["hr_csi"], dtype=np.float32)
        if lr.ndim != 3 or hr.ndim != 3 or lr.shape[1:] != _LR_SHAPE or hr.shape[1:] != _HR_SHAPE:
            raise ValueError(f"Unexpected CSI shapes in {path}: lr={lr.shape}, hr={hr.shape}")

        self._archive_cache[path] = (lr, hr)
        while len(self._archive_cache) > _ARCHIVE_CACHE_SIZE:
            self._archive_cache.popitem(last=False)
        return lr, hr

    def _load_batch(self, path: Path) -> dict[str, np.ndarray]:
        """Load the source metadata needed to restore physical GHI values.

        This intentionally reads only the three metadata arrays used by
        ``ClearSkyProvider`` rather than retaining full NPZ archives in memory.
        """
        path = Path(path)
        process_id = os.getpid()
        if self._owner_pid != process_id:
            self._archive_cache.clear()
            self._metadata_cache.clear()
            self._owner_pid = process_id

        cached = self._metadata_cache.pop(path, None)
        if cached is not None:
            self._metadata_cache[path] = cached
            return cached

        required = ("row0", "col0", "dominant_slot_minute_utc")
        if not path.exists():
            raise FileNotFoundError(f"Missing sample archive: {path}")
        with np.load(path, allow_pickle=False) as archive:
            missing = set(required).difference(archive.files)
            if missing:
                raise KeyError(f"{path} is missing {sorted(missing)}")
            metadata = {key: np.asarray(archive[key]) for key in required}
        self._metadata_cache[path] = metadata
        while len(self._metadata_cache) > _ARCHIVE_CACHE_SIZE:
            self._metadata_cache.popitem(last=False)
        return metadata

    @staticmethod
    def _normalized(array: np.ndarray, *, path: Path, batch_index: int) -> np.ndarray:
        result = np.asarray(array, dtype=np.float32)
        if not np.isfinite(result).all():
            raise FloatingPointError(f"NaN/Inf in {path} at batch_index={batch_index}")
        return np.clip(result, 0.0, CSI_SCALE) / CSI_SCALE

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, batch_index = self.index[int(item)]
        lr_batch, hr_batch = self._archive_arrays(path)
        if batch_index < 0 or batch_index >= len(lr_batch) or batch_index >= len(hr_batch):
            raise IndexError(f"batch_index={batch_index} is outside {path}")

        lr = self._normalized(lr_batch[batch_index], path=path, batch_index=batch_index)
        hr = self._normalized(hr_batch[batch_index], path=path, batch_index=batch_index)
        if self.augment:
            if torch.rand(()) < 0.5:
                lr = np.flip(lr, axis=0).copy()
                hr = np.flip(hr, axis=0).copy()
            if torch.rand(()) < 0.5:
                lr = np.flip(lr, axis=1).copy()
                hr = np.flip(hr, axis=1).copy()
            if torch.rand(()) < 0.5:
                lr = lr.T.copy()
                hr = hr.T.copy()

        return torch.from_numpy(lr[None].copy()), torch.from_numpy(hr[None].copy())


def split_by_dates(
    *,
    samples_root: Path,
    val_dates: Sequence[str],
    augment: bool = False,
) -> tuple[SatelliteCSIDataset, SatelliteCSIDataset]:
    """Build chronological train/validation datasets from FY sample directories.

    ``val_dates`` selects full UTC-day directories for validation.  Remaining
    dates become the training set.  This preserves the exact canonical order
    encoded in the committed fixed-split manifests.
    """

    root = Path(samples_root)
    if not root.exists():
        raise FileNotFoundError(f"FY sample root does not exist: {root}")
    validation_dates = {str(date) for date in val_dates}
    train_index: list[SampleIndex] = []
    validation_index: list[SampleIndex] = []
    directories = sorted(path for path in root.glob("*_utc") if path.is_dir())
    if not directories:
        raise FileNotFoundError(f"No <YYYYMMDD>_utc sample directories under {root}")

    for directory in directories:
        date = _date_from_directory(directory)
        entries = _read_index(directory)
        if date in validation_dates:
            validation_index.extend(entries)
        else:
            train_index.extend(entries)

    if not train_index or not validation_index:
        raise RuntimeError(
            f"Empty split from {root}: train={len(train_index)}, validation={len(validation_index)}"
        )
    return (
        SatelliteCSIDataset(train_index, augment=augment),
        SatelliteCSIDataset(validation_index, augment=False),
    )


class PreparedCSIDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Memory-mapped, physically separated DRCT split stored outside the repo.

    A prepared split contains normalized ``lr_csi.npy`` / ``hr_csi.npy`` arrays
    and a compact ``samples.json`` provenance manifest.  Validation and test
    splits additionally contain ``clear_ghi.npy`` for physical-unit evaluation.
    """

    def __init__(self, split_dir: Path, *, augment: bool = False) -> None:
        self.split_dir = Path(split_dir).resolve()
        self.augment = bool(augment)
        manifest_path = self.split_dir / "samples.json"
        metadata_path = self.split_dir / "metadata.json"
        if not manifest_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"Prepared split requires samples.json and metadata.json: {self.split_dir}"
            )
        with manifest_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        with metadata_path.open(encoding="utf-8") as handle:
            self.metadata = dict(json.load(handle))
        self.records = list(payload.get("samples", []))
        if not self.records:
            raise ValueError(f"Prepared split has no records: {manifest_path}")
        if int(self.metadata.get("sample_count", -1)) != len(self.records):
            raise ValueError(f"Prepared split sample count mismatch: {self.split_dir}")
        self.index: list[SampleIndex] = []
        self._role_indices: dict[str, list[int]] = {}
        for position, record in enumerate(self.records):
            if int(record.get("index", -1)) != position:
                raise ValueError(f"Non-contiguous prepared index at {manifest_path}:{position}")
            date = str(record["date"])
            batch_file = str(record["batch_file"])
            batch_index = int(record["batch_index"])
            # Keep the legacy provenance interface for date labels and manifests.
            virtual_path = Path(f"{date}_utc") / "patches" / batch_file
            self.index.append((virtual_path, batch_index))
            role = str(record.get("role", "sample"))
            self._role_indices.setdefault(role, []).append(position)

        self._lr_path = self.split_dir / "lr_csi.npy"
        self._hr_path = self.split_dir / "hr_csi.npy"
        self._clear_path = self.split_dir / "clear_ghi.npy"
        if not self._lr_path.exists() or not self._hr_path.exists():
            raise FileNotFoundError(f"Prepared CSI arrays are missing: {self.split_dir}")
        self._owner_pid: int | None = None
        self._lr: np.ndarray | None = None
        self._hr: np.ndarray | None = None
        self._clear: np.ndarray | None = None

    def __getstate__(self) -> dict[str, object]:
        state = dict(self.__dict__)
        state["_owner_pid"] = None
        state["_lr"] = None
        state["_hr"] = None
        state["_clear"] = None
        return state

    def __len__(self) -> int:
        return len(self.records)

    def indices_for_role(self, role: str) -> list[int]:
        return list(self._role_indices.get(role, []))

    def _ensure_open(self, *, include_clear: bool = False) -> None:
        process_id = os.getpid()
        if self._owner_pid != process_id:
            self._lr = np.load(self._lr_path, mmap_mode="r")
            self._hr = np.load(self._hr_path, mmap_mode="r")
            self._clear = None
            self._owner_pid = process_id
            expected_lr = (len(self), 1, *_LR_SHAPE)
            expected_hr = (len(self), 1, *_HR_SHAPE)
            if self._lr.shape != expected_lr or self._hr.shape != expected_hr:
                raise ValueError(
                    f"Unexpected prepared CSI shapes in {self.split_dir}: "
                    f"lr={self._lr.shape}, hr={self._hr.shape}"
                )
            if self._lr.dtype != np.float32 or self._hr.dtype != np.float32:
                raise TypeError(f"Prepared CSI arrays must be float32: {self.split_dir}")
        if include_clear and self._clear is None:
            if not self._clear_path.exists():
                raise FileNotFoundError(
                    f"Prepared split has no clear_ghi.npy: {self.split_dir}"
                )
            self._clear = np.load(self._clear_path, mmap_mode="r")
            if self._clear.shape != (len(self), *_HR_SHAPE) or self._clear.dtype != np.float32:
                raise ValueError(
                    f"Unexpected prepared clear-GHI shape in {self.split_dir}: {self._clear.shape}"
                )

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_open()
        if self._lr is None or self._hr is None:
            raise RuntimeError("Prepared CSI arrays did not open")
        position = int(item)
        lr = np.asarray(self._lr[position, 0], dtype=np.float32).copy()
        hr = np.asarray(self._hr[position, 0], dtype=np.float32).copy()
        if self.augment:
            if torch.rand(()) < 0.5:
                lr = np.flip(lr, axis=0).copy()
                hr = np.flip(hr, axis=0).copy()
            if torch.rand(()) < 0.5:
                lr = np.flip(lr, axis=1).copy()
                hr = np.flip(hr, axis=1).copy()
            if torch.rand(()) < 0.5:
                lr = lr.T.copy()
                hr = hr.T.copy()
        return torch.from_numpy(lr[None]), torch.from_numpy(hr[None])

    def clear_sky_patch(self, item: int) -> np.ndarray:
        self._ensure_open(include_clear=True)
        if self._clear is None:
            raise RuntimeError("Prepared clear-GHI array did not open")
        patch = np.asarray(self._clear[int(item)], dtype=np.float32)
        if not np.isfinite(patch).all():
            raise FloatingPointError(f"NaN/Inf in prepared clear-GHI: {self.split_dir}")
        return patch.copy()


def load_prepared_splits(
    root: Path,
    *,
    augment_train: bool = False,
) -> tuple[PreparedCSIDataset, PreparedCSIDataset, PreparedCSIDataset]:
    """Open the fixed train / validation / test datasets under ``root``."""
    prepared_root = Path(root)
    return (
        PreparedCSIDataset(prepared_root / "train", augment=augment_train),
        PreparedCSIDataset(prepared_root / "validation", augment=False),
        PreparedCSIDataset(prepared_root / "test", augment=False),
    )
