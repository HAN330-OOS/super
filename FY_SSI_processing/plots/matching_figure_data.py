"""Shared, read-only data preparation for thesis Figures 5-2 and 5-3.

The functions here reuse the project's existing FY-3C timestamp inversion,
FY-4A half-open-window matching function, and FY-4A nearest-neighbour
resampling products.  They never write or alter any matching/training data.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys

import numpy as np


FY_PROCESSING_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = FY_PROCESSING_ROOT.parent
sys.path.insert(0, str(FY_PROCESSING_ROOT))

from fy_io import (  # noqa: E402
    DEFAULT_FY3_ROOT,
    OUTPUT_RESAMPLED_ON_FY3_DIR,
    cst_minutes_to_utc_minutes,
    match_pixels_to_half_open_windows,
    read_fy3_binary,
)
from step_01_estimate_fy3c_timestamp_from_sz import clean_sz, read_sz_product  # noqa: E402


DATE = "20180510"
MAX_SZA_ERROR_DEG = 0.5
TIMESTAMP_DIR = PROJECT_ROOT / "outputs" / "fy" / "timestamp"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "fy" / "figures"


@dataclass(frozen=True)
class Fy4Slot:
    """One actual FY-4A product already nearest-mapped onto the FY-3C grid."""

    label_utc: str
    start_minute_utc: int
    source_nc: Path
    resampled_path: Path
    radius_m: float


@dataclass
class MatchingResult:
    """Actual matched fields and audit statistics for one date."""

    date: str
    lons: np.ndarray
    lats: np.ndarray
    time_utc_hours: np.ndarray
    slot_map: np.ndarray
    matched_ssi: np.ndarray
    gc: np.ndarray
    final_mask: np.ndarray
    slots: list[Fy4Slot]
    slot_counts: np.ndarray
    timestamp_path: Path
    sz_path: Path
    gc_path: Path
    prior_label: str
    valid_sz_count: int
    inversion_valid_count: int
    qc_valid_count: int
    unmatched_count: int
    spatial_valid_count: int


def minute_label(minute: float | int) -> str:
    """Format minute-of-day as ``HH:MM``."""
    value = int(round(float(minute))) % 1440
    return f"{value // 60:02d}:{value % 60:02d}"


def timestamp_path(date: str) -> Path:
    return TIMESTAMP_DIR / f"fy3c_timestamp_{date}_prior1015w120_stride1.npz"


def _load_timestamp(date: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    path = timestamp_path(date)
    if not path.exists():
        raise FileNotFoundError(f"Missing formal timestamp inversion output: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {"best_minute", "best_diff", "lons", "lats"}
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"{path} is missing: {sorted(missing)}")
        best_minute = np.asarray(data["best_minute"], dtype=np.int16)
        best_diff = np.asarray(data["best_diff"], dtype=np.float32)
        lons = np.asarray(data["lons"], dtype=np.float64)
        lats = np.asarray(data["lats"], dtype=np.float64)
        metadata = {
            key: data[key].item() if data[key].ndim == 0 else data[key]
            for key in (
                "use_local_time_prior",
                "local_overpass_minute",
                "prior_window_minutes",
                "local_time_prior_label",
                "sz_path",
            )
            if key in data.files
        }
    return best_minute, best_diff, lons, lats, metadata


def _load_slots(date: str, expected_shape: tuple[int, int]) -> list[Fy4Slot]:
    directory = OUTPUT_RESAMPLED_ON_FY3_DIR / f"{date}_utc"
    paths = sorted(directory.glob("FY4_*_on_FY3.npz"))
    if not paths:
        raise FileNotFoundError(f"Missing FY-4A nearest-resampling products: {directory}")

    slots: list[Fy4Slot] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            if tuple(data["ssi"].shape) != expected_shape:
                raise ValueError(f"{path.name}: incompatible FY-3C-grid shape.")
            if str(data["method"].item()) != "nearest":
                raise ValueError(f"{path.name}: Figure 5 requires nearest mapping.")
            slots.append(
                Fy4Slot(
                    label_utc=str(data["time_utc"].item()),
                    start_minute_utc=int(data["start_minute_utc"].item()),
                    source_nc=Path(str(data["source_nc"].item())),
                    resampled_path=path,
                    radius_m=float(data["radius_m"].item()),
                )
            )
    slots.sort(key=lambda slot: slot.start_minute_utc)
    return slots


def _original_sz_valid_mask(date: str, expected_shape: tuple[int, int]) -> tuple[Path, np.ndarray]:
    path = DEFAULT_FY3_ROOT / f"Sz_{date}.dat"
    _clean_sz, valid_sz = clean_sz(read_sz_product(path, stride=1))
    if valid_sz.shape != expected_shape:
        raise ValueError("Raw Sz and timestamp-inversion grids have different shapes.")
    return path, valid_sz


def _load_gc(date: str, expected_shape: tuple[int, int]) -> tuple[Path, np.ndarray]:
    path = DEFAULT_FY3_ROOT / f"Gc_{date}.dat"
    gc = read_fy3_binary(path).astype(np.float32, copy=False)
    if gc.shape != expected_shape:
        raise ValueError("Raw Gc and timestamp-inversion grids have different shapes.")
    gc = gc.copy()
    gc[~np.isfinite(gc)] = np.nan
    return path, gc


def _assemble_matched_ssi(slot_map: np.ndarray, slots: list[Fy4Slot]) -> tuple[np.ndarray, np.ndarray]:
    """Combine actual FY-4A slot fields according to every matched pixel."""
    matched = np.full(slot_map.shape, np.nan, dtype=np.float32)
    spatial_valid = np.zeros(slot_map.shape, dtype=bool)
    for index, slot in enumerate(slots):
        select = slot_map == index
        if not np.any(select):
            continue
        with np.load(slot.resampled_path, allow_pickle=False) as data:
            ssi = np.asarray(data["ssi"], dtype=np.float32)
        good = select & np.isfinite(ssi)
        matched[good] = ssi[good]
        spatial_valid[good] = True
        del ssi
    return matched, spatial_valid


def build_matching_result(date: str = DATE) -> MatchingResult:
    """Build the formal matching result using only existing real products."""
    best_minute, best_diff, lons, lats, metadata = _load_timestamp(date)
    shape = best_minute.shape
    if best_diff.shape != shape or shape != (lats.size, lons.size):
        raise ValueError("Timestamp arrays and FY-3C coordinate grid are inconsistent.")
    if not bool(metadata.get("use_local_time_prior", False)):
        raise ValueError("The required morning local-time prior is absent from the timestamp product.")
    if metadata.get("local_overpass_minute") != 615 or metadata.get("prior_window_minutes") != 120:
        raise ValueError("Expected the formal 10:15 local-time prior with a +/-120 min window.")

    sz_path, valid_sz = _original_sz_valid_mask(date, shape)
    inversion_valid = (best_minute >= 0) & np.isfinite(best_diff)
    # Exclude any historical neighbour-filled timestamp from the thesis figures:
    # all shown/matched pixels must be valid in original Sz and pass <0.5 degree.
    valid_qc = valid_sz & inversion_valid & (best_diff < MAX_SZA_ERROR_DEG)
    slots = _load_slots(date, shape)
    starts = np.asarray([slot.start_minute_utc for slot in slots], dtype=np.int32)
    slot_map = match_pixels_to_half_open_windows(
        best_minute,
        valid_qc,
        starts,
        window_minutes=15,
        best_minute_tz="CST",
    )

    utc_minutes = cst_minutes_to_utc_minutes(best_minute).astype(np.float32)
    time_utc_hours = utc_minutes / 60.0
    time_utc_hours[~valid_qc] = np.nan
    gc_path, gc = _load_gc(date, shape)
    matched_ssi, spatial_valid = _assemble_matched_ssi(slot_map, slots)
    final_mask = valid_qc & (slot_map >= 0) & spatial_valid & np.isfinite(gc)
    matched_ssi[~final_mask] = np.nan
    gc[~final_mask] = np.nan
    if not np.array_equal(np.isfinite(matched_ssi), np.isfinite(gc)):
        raise AssertionError("Matched FY-4A SSI and FY-3C Gc masks differ.")

    converted_utc = cst_minutes_to_utc_minutes(best_minute).astype(np.int32)
    for index, slot in enumerate(slots):
        selected = slot_map == index
        if np.any(selected):
            values = converted_utc[selected]
            if not np.all((values >= slot.start_minute_utc) & (values < slot.start_minute_utc + 15)):
                raise AssertionError(f"Half-open window check failed for FY-4A {slot.label_utc}.")

    slot_counts = np.bincount(slot_map[slot_map >= 0], minlength=len(slots))
    return MatchingResult(
        date=date,
        lons=lons,
        lats=lats,
        time_utc_hours=time_utc_hours,
        slot_map=slot_map,
        matched_ssi=matched_ssi,
        gc=gc,
        final_mask=final_mask,
        slots=slots,
        slot_counts=slot_counts,
        timestamp_path=timestamp_path(date),
        sz_path=sz_path,
        gc_path=gc_path,
        prior_label=str(metadata["local_time_prior_label"]),
        valid_sz_count=int(valid_sz.sum()),
        inversion_valid_count=int((valid_sz & inversion_valid).sum()),
        qc_valid_count=int(valid_qc.sum()),
        unmatched_count=int((valid_qc & (slot_map < 0)).sum()),
        spatial_valid_count=int(spatial_valid.sum()),
    )


def cell_extent(lons: np.ndarray, lats: np.ndarray) -> tuple[float, float, float, float]:
    """Return map edges for FY-3C cell centres."""
    lon_step = float(np.median(np.diff(lons)))
    lat_step = float(np.median(np.abs(np.diff(lats))))
    return (
        float(lons[0] - lon_step / 2),
        float(lons[-1] + lon_step / 2),
        float(lats[-1] - lat_step / 2),
        float(lats[0] + lat_step / 2),
    )


def rounded_irradiance_limits(*fields: np.ndarray) -> tuple[float, float]:
    """Derive a common physical SSI/Gc colour range from valid paired pixels."""
    values = np.concatenate([field[np.isfinite(field)] for field in fields])
    if values.size == 0:
        raise RuntimeError("No valid paired irradiance pixels remain.")
    lower = float(math.floor(float(values.min()) / 50.0) * 50.0)
    upper = float(math.ceil(float(values.max()) / 50.0) * 50.0)
    return lower, upper if upper > lower else lower + 50.0


def extent_slices(
    lons: np.ndarray,
    lats: np.ndarray,
    extent: tuple[float, float, float, float],
) -> tuple[slice, slice]:
    """Get raw-grid row/column slices for lon_min, lon_max, lat_min, lat_max."""
    lon_min, lon_max, lat_min, lat_max = extent
    col_start = int(np.searchsorted(lons, lon_min, side="left"))
    col_end = int(np.searchsorted(lons, lon_max, side="right"))
    row_start = int(np.searchsorted(-lats, -lat_max, side="left"))
    row_end = int(np.searchsorted(-lats, -lat_min, side="right"))
    return slice(row_start, row_end), slice(col_start, col_end)
