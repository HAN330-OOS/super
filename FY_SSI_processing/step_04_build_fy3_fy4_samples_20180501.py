"""
FY-3C / FY-4A 时空配对切样本（20180501 单日）。

流程：读取 FY-3C Gc 与时间戳 → 半开窗口 [start, start+15min) 匹配 FY-4 时次
（非最近邻）→ 从 step_03 重采样 FY4_*_on_FY3.npz 取 SSI → 滑动窗口切 FY-3 64×64 /
FY-4 16×16 配对 patch（slot>=0 覆盖率 >= min_slot_valid_frac，且 slot>=0 像元仅含单一 FY-4 时次），输出索引 CSV 与 npz。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fy_io import (
    DEFAULT_FY3_ROOT,
    DEFAULT_LAT_MAX,
    DEFAULT_LAT_MIN,
    DEFAULT_LON_MAX,
    DEFAULT_LON_MIN,
    FY3_STEP,
    OUTPUT_MATCHING_DIR,
    OUTPUT_RESAMPLED_ON_FY3_DIR,
    OUTPUT_SAMPLES_DIR,
    PROJECT_ROOT,
    build_target_grid,
    default_timestamp_npz,
    discover_fy4_slots_from_resampled,
    dual_time_label,
    match_pixels_to_half_open_windows,
    minute_to_hhmm,
    read_fy3_binary,
    region_slice,
    region_tag,
    resampled_on_fy3_path,
    utc_minutes_to_cst_minutes,
)

FY3_PATCH = 64
FY4_PATCH = 16
FY4_POOL = FY3_PATCH // FY4_PATCH  # 4 km effective from 1 km grid

DEFAULT_TIMESTAMP_NPZ = default_timestamp_npz("20180501")


@dataclass(frozen=True)
class Fy4Slot:
    utc_label: str
    cst_label: str
    start_minute_utc: int
    path: Path

    @property
    def label(self) -> str:
        return self.utc_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FY-3C / FY-4A 时空配对切训练样本（师兄半开窗口匹配，默认 20180501）。",
        epilog=(
            "小规模测试示例:\n"
            "  py -3 FY_SSI_processing\\step_04_build_fy3_fy4_samples_20180501.py "
            "--max-samples 5\n\n"
            "默认研究区 100-120E, 30-45N；全幅 FY-3 覆盖区可加 "
            "--lon-min 73 --lon-max 135 --lat-min 18 --lat-max 54"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--date", default="20180501", help="日期 YYYYMMDD。")
    parser.add_argument("--fy3-gc", type=Path, default=None, help="FY-3C Gc .dat 路径。")
    parser.add_argument(
        "--timestamp-npz",
        type=Path,
        default=DEFAULT_TIMESTAMP_NPZ,
    )
    parser.add_argument(
        "--resampled-dir",
        type=Path,
        default=None,
        help="step_03 输出的 FY4_*_on_FY3.npz 目录。",
    )
    parser.add_argument(
        "--slot-map-npz",
        type=Path,
        default=None,
        help="step_03b 输出的 fy4_slot_map npz；缺省则按同逻辑在线计算。",
    )
    parser.add_argument("--window-minutes", type=int, default=15, help="半开窗宽度（分钟）。")
    parser.add_argument("--lon-min", type=float, default=DEFAULT_LON_MIN)
    parser.add_argument("--lon-max", type=float, default=DEFAULT_LON_MAX)
    parser.add_argument("--lat-min", type=float, default=DEFAULT_LAT_MIN)
    parser.add_argument("--lat-max", type=float, default=DEFAULT_LAT_MAX)
    parser.add_argument("--patch-stride", type=int, default=32, help="滑动窗口步长（像元）。")
    parser.add_argument("--fy3-patch", type=int, default=FY3_PATCH)
    parser.add_argument("--fy4-patch", type=int, default=FY4_PATCH)
    parser.add_argument("--max-samples", type=int, default=None, help="最多保留样本数；测试时可设小值。")
    parser.add_argument("--batch-size", type=int, default=64, help="每个 patch npz 打包的样本数。")
    parser.add_argument("--max-diff-deg", type=float, default=0.5, help="像元 best_diff 上限（度）。")
    parser.add_argument(
        "--max-bad-diff-frac",
        type=float,
        default=0.05,
        help="patch 内 best_diff 超阈值像元比例上限。",
    )
    parser.add_argument(
        "--min-valid-frac",
        type=float,
        default=0.95,
        help="patch 内 FY-3 Gc 与 FY-4 SSI 同时有效的最小比例。",
    )
    parser.add_argument(
        "--min-slot-valid-frac",
        type=float,
        default=0.99,
        help=(
            "patch 内有时戳匹配的像元比例下限："
            "slot_valid_frac = N(slot>=0) / (fy3_patch^2)，默认 0.99（64×64 即至少 4055 像元）。"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--estimate-only", action="store_true", help="仅统计候选 patch 数，不写样本。")
    return parser.parse_args()


def default_fy3_gc_path(date: str) -> Path:
    return DEFAULT_FY3_ROOT / f"Gc_{date}.dat"


def default_slot_map_npz(date: str, timestamp_npz: Path) -> Path:
    tag = "prior1015w120_stride1" if "prior1015w120" in timestamp_npz.name else "timestamp"
    return OUTPUT_MATCHING_DIR / f"{date}_utc" / f"fy4_slot_map_{date}_{tag}.npz"


def load_timestamp(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not npz_path.exists():
        raise FileNotFoundError(f"时间戳 npz 不存在: {npz_path}")
    data = np.load(npz_path)
    if "best_minute" not in data or "best_diff" not in data:
        raise KeyError(f"{npz_path} 缺少 best_minute / best_diff。")
    return data["best_minute"], data["best_diff"].astype(np.float32)


def build_fy4_slots(resampled_dir: Path, window_minutes: int) -> list[Fy4Slot]:
    discovered = discover_fy4_slots_from_resampled(resampled_dir, window_minutes)
    return [
        Fy4Slot(
            utc_label=utc_label,
            cst_label=cst_label,
            start_minute_utc=start,
            path=resampled_on_fy3_path(resampled_dir, utc_label),
        )
        for utc_label, start, _end, cst_label in discovered
    ]


def load_or_build_slot_map(
    best_minute: np.ndarray,
    best_diff: np.ndarray,
    slot_starts: np.ndarray,
    window_minutes: int,
    slot_map_npz: Path | None,
) -> np.ndarray:
    if slot_map_npz is not None and slot_map_npz.exists():
        data = np.load(slot_map_npz)
        if "fy4_slot_map" not in data:
            raise KeyError(f"{slot_map_npz} 缺少 fy4_slot_map。")
        slot_map = data["fy4_slot_map"]
        if slot_map.shape != best_minute.shape:
            raise ValueError(
                f"slot_map shape {slot_map.shape} 与 best_minute {best_minute.shape} 不一致。"
            )
        print(f"  已加载 slot_map: {slot_map_npz}")
        return slot_map.astype(np.int16, copy=False)

    valid = (best_minute >= 0) & np.isfinite(best_diff)
    slot_map = match_pixels_to_half_open_windows(
        best_minute,
        valid,
        slot_starts,
        window_minutes=window_minutes,
        best_minute_tz="CST",
    )
    print("  在线计算 slot_map（半开窗口，非最近邻）。")
    return slot_map


def load_resampled_ssi(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"重采样 npz 不存在: {path}")
    data = np.load(path)
    if "ssi" not in data:
        raise KeyError(f"{path} 缺少 ssi 数组。")
    return data["ssi"].astype(np.float32, copy=False)


def subsample_fy4_patch(patch: np.ndarray, pool: int, fy4_patch: int) -> np.ndarray:
    sampled = patch[::pool, ::pool]
    if sampled.shape != (fy4_patch, fy4_patch):
        raise ValueError(f"FY-4 patch shape {sampled.shape} != ({fy4_patch}, {fy4_patch})。")
    return sampled.astype(np.float32, copy=False)


def patch_single_slot(slot_patch: np.ndarray) -> tuple[int | None, int]:
    """slot>=0 像元中须仅有一个 FY-4 时次；否则返回 (None, n_slots)。"""
    valid = slot_patch >= 0
    if not np.any(valid):
        return None, 0

    slots = slot_patch[valid]
    values = np.unique(slots)
    n_slots = int(values.size)
    if n_slots != 1:
        return None, n_slots
    return int(values[0]), n_slots


def patch_passes_qc(
    gc_patch: np.ndarray,
    fy4_patch_full: np.ndarray,
    diff_patch: np.ndarray,
    slot_patch: np.ndarray,
    dominant_slot: int,
    args: argparse.Namespace,
) -> tuple[bool, dict[str, float]]:
    valid_gc = np.isfinite(gc_patch)
    valid_fy4 = np.isfinite(fy4_patch_full)
    valid_time = slot_patch >= 0
    valid_diff = np.isfinite(diff_patch) & (diff_patch < args.max_diff_deg)
    joint = valid_gc & valid_fy4 & valid_time & valid_diff

    valid_frac = float(joint.mean())
    bad_diff_frac = float((valid_time & np.isfinite(diff_patch) & (diff_patch >= args.max_diff_deg)).mean())
    dominant_slot_frac = float((slot_patch == dominant_slot).mean())

    stats = {
        "valid_frac": valid_frac,
        "bad_diff_frac": bad_diff_frac,
        "dominant_slot_frac": dominant_slot_frac,
        "n_slots": float(np.unique(slot_patch[valid_time]).size),
    }

    if valid_frac < args.min_valid_frac:
        return False, stats
    if bad_diff_frac > args.max_bad_diff_frac:
        return False, stats
    if not np.any(joint):
        return False, stats
    return True, stats


def count_candidate_patches(nrows: int, ncols: int, patch: int, stride: int) -> int:
    if nrows < patch or ncols < patch:
        return 0
    row_starts = range(0, nrows - patch + 1, stride)
    col_starts = range(0, ncols - patch + 1, stride)
    return len(row_starts) * len(col_starts)


def write_batch_npz(
    batch_path: Path,
    fy3_list: list[np.ndarray],
    fy4_list: list[np.ndarray],
    meta: dict[str, list],
) -> None:
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        batch_path,
        fy3=np.stack(fy3_list, axis=0).astype(np.float32),
        fy4=np.stack(fy4_list, axis=0).astype(np.float32),
        **{key: np.asarray(values) for key, values in meta.items()},
    )


def main() -> None:
    args = parse_args()
    pool = args.fy3_patch // args.fy4_patch
    if args.fy3_patch != pool * args.fy4_patch:
        raise ValueError(
            f"fy3_patch={args.fy3_patch} 必须能被 fy4_patch={args.fy4_patch} 整除。"
        )

    resampled_dir = args.resampled_dir or (OUTPUT_RESAMPLED_ON_FY3_DIR / f"{args.date}_utc")
    slot_map_npz = args.slot_map_npz or default_slot_map_npz(args.date, args.timestamp_npz)

    tag = region_tag(args.lon_min, args.lon_max, args.lat_min, args.lat_max)
    out_dir = args.output_dir or (OUTPUT_SAMPLES_DIR / f"{args.date}_utc")
    patch_dir = out_dir / "patches"

    gc_path = args.fy3_gc or default_fy3_gc_path(args.date)

    print("FY-3C / FY-4A 配对切样本（师兄半开窗口）:")
    print(f"  日期: {args.date}")
    print(f"  FY-3C Gc: {gc_path}")
    print(f"  时间戳: {args.timestamp_npz}")
    print(f"  FY-4 重采样目录: {resampled_dir}")
    print(f"  slot_map: {slot_map_npz if slot_map_npz.exists() else '(在线计算)'}")
    print(f"  区域: {args.lon_min}-{args.lon_max}E, {args.lat_min}-{args.lat_max}N")
    print(
        f"  Patch: FY-3 {args.fy3_patch}x{args.fy3_patch}, "
        f"FY-4 {args.fy4_patch}x{args.fy4_patch}, stride={args.patch_stride}, "
        f"slot_valid>={args.min_slot_valid_frac:.0%}, single FY-4 slot among slot>=0"
    )

    slots = build_fy4_slots(resampled_dir, args.window_minutes)
    slot_starts = np.array([slot.start_minute_utc for slot in slots], dtype=np.int32)
    print(f"  FY-4A 时次: {len(slots)} 个")
    for slot in slots:
        end_utc = slot.start_minute_utc + args.window_minutes
        end_cst = int(utc_minutes_to_cst_minutes(end_utc))
        print(
            f"    FY4_{slot.utc_label} → "
            f"[{minute_to_hhmm(slot.start_minute_utc)}, {minute_to_hhmm(end_utc)}) UTC / "
            f"[{slot.cst_label}, {minute_to_hhmm(end_cst)}) CST"
        )

    best_minute, best_diff = load_timestamp(args.timestamp_npz)
    slot_index_full = load_or_build_slot_map(
        best_minute,
        best_diff,
        slot_starts,
        args.window_minutes,
        slot_map_npz if slot_map_npz.exists() else None,
    )

    row_start, row_end, col_start, col_end = region_slice(
        args.lon_min, args.lon_max, args.lat_min, args.lat_max
    )
    gc_full = read_fy3_binary(gc_path)
    gc_region = gc_full[row_start:row_end, col_start:col_end].astype(np.float32, copy=False)
    diff_region = best_diff[row_start:row_end, col_start:col_end]
    slot_region = slot_index_full[row_start:row_end, col_start:col_end]

    nrows, ncols = gc_region.shape
    candidate_count = count_candidate_patches(nrows, ncols, args.fy3_patch, args.patch_stride)
    print(f"  区域 shape: {gc_region.shape}, 候选 patch 数（QC 前）: {candidate_count}")

    if args.estimate_only:
        print("estimate-only: 跳过写样本。")
        return

    lons_1d, lats_1d, _target_lons, _target_lats = build_target_grid(
        args.lon_min, args.lon_max, args.lat_min, args.lat_max, FY3_STEP
    )
    if lats_1d.shape[0] != gc_region.shape[0]:
        raise ValueError(f"目标网格行数 {lats_1d.shape[0]} 与 Gc 子区域 {gc_region.shape[0]} 不一致。")

    used_slots = sorted(int(v) for v in np.unique(slot_region[slot_region >= 0]))
    print(f"  区域用到的 FY-4 时次: {len(used_slots)} / {len(slots)}")

    resampled_by_slot: dict[int, np.ndarray] = {}
    for slot_idx in used_slots:
        slot = slots[slot_idx]
        print(f"  读取 {dual_time_label(slot.utc_label)} ← {slot.path.name}")
        resampled_by_slot[slot_idx] = load_resampled_ssi(slot.path)[
            row_start:row_end, col_start:col_end
        ]

    index_rows: list[dict[str, object]] = []
    fy3_batch: list[np.ndarray] = []
    fy4_batch: list[np.ndarray] = []
    batch_meta: dict[str, list] = {
        "row0": [],
        "col0": [],
        "center_lon": [],
        "center_lat": [],
        "dominant_slot": [],
        "dominant_slot_minute_utc": [],
        "dominant_slot_minute_cst": [],
        "valid_frac": [],
        "bad_diff_frac": [],
        "dominant_slot_frac": [],
    }
    batch_id = 0
    sample_id = 0
    rejected_slot_valid_frac = 0
    rejected_multi_slot = 0
    rejected_qc = 0
    patch_pixels = args.fy3_patch * args.fy3_patch
    accepted_per_slot: dict[int, int] = {idx: 0 for idx in range(len(slots))}

    for row0 in range(0, nrows - args.fy3_patch + 1, args.patch_stride):
        for col0 in range(0, ncols - args.fy3_patch + 1, args.patch_stride):
            slot_patch = slot_region[row0 : row0 + args.fy3_patch, col0 : col0 + args.fy3_patch]
            slot_valid_frac = float((slot_patch >= 0).sum()) / patch_pixels
            if slot_valid_frac < args.min_slot_valid_frac:
                rejected_slot_valid_frac += 1
                continue

            dominant_slot, n_slots_in_patch = patch_single_slot(slot_patch)
            if dominant_slot is None:
                rejected_multi_slot += 1
                continue

            gc_patch = gc_region[row0 : row0 + args.fy3_patch, col0 : col0 + args.fy3_patch]
            fy4_layer = resampled_by_slot[dominant_slot]
            fy4_patch_full = fy4_layer[row0 : row0 + args.fy3_patch, col0 : col0 + args.fy3_patch]
            diff_patch = diff_region[row0 : row0 + args.fy3_patch, col0 : col0 + args.fy3_patch]

            ok, stats = patch_passes_qc(
                gc_patch, fy4_patch_full, diff_patch, slot_patch, dominant_slot, args
            )
            if not ok:
                rejected_qc += 1
                continue

            dominant_minute_utc = int(slots[dominant_slot].start_minute_utc)
            dominant_minute_cst = int(utc_minutes_to_cst_minutes(dominant_minute_utc))
            center_row = row0 + args.fy3_patch // 2
            center_col = col0 + args.fy3_patch // 2
            center_lon = float(lons_1d[center_col])
            center_lat = float(lats_1d[center_row])

            fy3_batch.append(gc_patch.copy())
            fy4_batch.append(subsample_fy4_patch(fy4_patch_full, pool, args.fy4_patch))

            batch_meta["row0"].append(row0)
            batch_meta["col0"].append(col0)
            batch_meta["center_lon"].append(center_lon)
            batch_meta["center_lat"].append(center_lat)
            batch_meta["dominant_slot"].append(dominant_slot)
            batch_meta["dominant_slot_minute_utc"].append(dominant_minute_utc)
            batch_meta["dominant_slot_minute_cst"].append(dominant_minute_cst)
            batch_meta["valid_frac"].append(stats["valid_frac"])
            batch_meta["bad_diff_frac"].append(stats["bad_diff_frac"])
            batch_meta["dominant_slot_frac"].append(stats["dominant_slot_frac"])

            patch_name = f"batch_{batch_id:04d}.npz"
            index_rows.append(
                {
                    "sample_id": sample_id,
                    "batch_file": patch_name,
                    "batch_index": len(fy3_batch) - 1,
                    "row0": row0,
                    "col0": col0,
                    "global_row0": row_start + row0,
                    "global_col0": col_start + col0,
                    "center_lon": center_lon,
                    "center_lat": center_lat,
                    "dominant_slot": dominant_slot,
                    "dominant_slot_label": slots[dominant_slot].utc_label,
                    "dominant_slot_label_cst": slots[dominant_slot].cst_label,
                    "dominant_slot_minute_utc": dominant_minute_utc,
                    "dominant_slot_minute_cst": dominant_minute_cst,
                    "dominant_slot_frac": stats["dominant_slot_frac"],
                    "valid_frac": stats["valid_frac"],
                    "bad_diff_frac": stats["bad_diff_frac"],
                    "n_slots_in_patch": n_slots_in_patch,
                }
            )
            sample_id += 1
            accepted_per_slot[dominant_slot] = accepted_per_slot.get(dominant_slot, 0) + 1

            if args.max_samples is not None and sample_id >= args.max_samples:
                break

            if len(fy3_batch) >= args.batch_size:
                write_batch_npz(patch_dir / patch_name, fy3_batch, fy4_batch, batch_meta)
                fy3_batch, fy4_batch = [], []
                batch_meta = {key: [] for key in batch_meta}
                batch_id += 1

        if args.max_samples is not None and sample_id >= args.max_samples:
            break

    if fy3_batch:
        write_batch_npz(patch_dir / f"batch_{batch_id:04d}.npz", fy3_batch, fy4_batch, batch_meta)

    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "sample_index.csv"
    fieldnames = list(index_rows[0].keys()) if index_rows else [
        "sample_id", "batch_file", "batch_index", "row0", "col0",
        "global_row0", "global_col0", "center_lon", "center_lat",
        "dominant_slot", "dominant_slot_label", "dominant_slot_label_cst",
        "dominant_slot_minute_utc", "dominant_slot_minute_cst",
        "dominant_slot_frac", "valid_frac", "bad_diff_frac", "n_slots_in_patch",
    ]
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(index_rows)

    rejected_total = rejected_slot_valid_frac + rejected_multi_slot + rejected_qc
    slot_count_labels = np.array([slot.utc_label for slot in slots])
    slot_count_values = np.array(
        [accepted_per_slot.get(idx, 0) for idx in range(len(slots))],
        dtype=np.int64,
    )
    meta_path = out_dir / "build_meta.npz"
    np.savez_compressed(
        meta_path,
        date=args.date,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        region_tag=tag,
        fy3_patch=args.fy3_patch,
        fy4_patch=args.fy4_patch,
        patch_stride=args.patch_stride,
        window_minutes=args.window_minutes,
        min_slot_valid_frac=args.min_slot_valid_frac,
        patch_slot_rule="single unique FY-4 slot among slot>=0 pixels",
        matching_rule="half_open [start, start+15) UTC from filename; FY-3 best_minute CST→UTC; not nearest-neighbor",
        max_diff_deg=args.max_diff_deg,
        max_bad_diff_frac=args.max_bad_diff_frac,
        min_valid_frac=args.min_valid_frac,
        candidate_patches=candidate_count,
        accepted_samples=sample_id,
        rejected_patches=rejected_total,
        rejected_slot_valid_frac=rejected_slot_valid_frac,
        rejected_multi_slot=rejected_multi_slot,
        rejected_qc=rejected_qc,
        accepted_per_slot_labels=slot_count_labels,
        accepted_per_slot_counts=slot_count_values,
        used_slot_minutes_utc=slot_starts[used_slots],
        slot_labels_utc=np.array([slots[i].utc_label for i in used_slots]),
        slot_labels_cst=np.array([slots[i].cst_label for i in used_slots]),
        resampled_dir=str(resampled_dir),
        slot_map_npz=str(slot_map_npz) if slot_map_npz.exists() else "",
        timestamp_npz=str(args.timestamp_npz),
    )

    print()
    print(f"完成: 接受 {sample_id} 个样本（候选 {candidate_count}）。")
    print(
        f"  拒绝: slot 覆盖率不足 {rejected_slot_valid_frac}, "
        f"多时次 {rejected_multi_slot}, QC 未过 {rejected_qc}, 合计 {rejected_total}"
    )
    print("  各时次接受 patch 数:")
    for idx, slot in enumerate(slots):
        print(f"    {slot.utc_label} ({slot.cst_label} CST): {accepted_per_slot.get(idx, 0)}")
    print(f"  索引: {index_path}")
    print(f"  patch: {patch_dir}")
    print(f"  元数据: {meta_path}")


if __name__ == "__main__":
    main()
