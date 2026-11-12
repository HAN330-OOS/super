"""Rebuild the fixed FY-3C/FY-4A 2018-2019 sample dataset on drive E.

Each date is processed independently into a staging directory, validated, and
then atomically published.  Existing D-drive products are read-only and are
never reused as final samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from fy_io import (
    DEFAULT_FY3_ROOT,
    DEFAULT_FY4_GEO_RAW,
    DEFAULT_FY4_ROOT,
    iter_dates,
    list_fy4_nc_files_in_utc_window,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path(r"E:\Super-Resolution-Samples-2018-2019")
FIRST_DATE = "20180501"
LAST_DATE = "20191231"
EXCLUDED_DATES = frozenset(iter_dates("20180522", "20180531"))
MANIFEST_FIELDS = (
    "sample_id",
    "date",
    "fy3_time",
    "fy4_time",
    "row",
    "col",
    "center_lon",
    "center_lat",
    "npz_file",
    "array_index",
)
ARRAY_SHAPES = {
    "fy3": (128, 128),
    "fy4": (32, 32),
    "hr_csi": (128, 128),
    "lr_csi": (32, 32),
}
MERRA_PRODUCTS = ("tavg1_2d_aer_Nx", "tavg1_2d_rad_Nx", "tavg1_2d_slv_Nx")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="重建 2018-2019 FY 超分辨率样本数据集。")
    parser.add_argument("command", choices=("run", "validate", "finalize"))
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--date", default=None, help="只处理或验证一个日期 YYYYMMDD。")
    parser.add_argument("--start-date", default=FIRST_DATE)
    parser.add_argument("--end-date", default=LAST_DATE)
    parser.add_argument("--keep-work", action="store_true", help="成功后保留可重建中间文件。")
    parser.add_argument("--force", action="store_true", help="重新处理已成功日期。")
    parser.add_argument("--workers", type=int, default=2, help="并行处理日期数；16 GB 内存默认 2。")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def ensure_layout(root: Path) -> None:
    for name in ("samples", "logs", "status", "splits", ".work", ".staging", ".quarantine"):
        (root / name).mkdir(parents=True, exist_ok=True)


def selected_dates(args: argparse.Namespace) -> list[str]:
    dates = [args.date] if args.date else list(iter_dates(args.start_date, args.end_date))
    return [date for date in dates if date not in EXCLUDED_DATES]


def merra_file(product: str, date: str) -> Path:
    return PROJECT_ROOT / "MERRA2" / product / f"MERRA2_400.{product}.{date}.SUB.nc4"


def verify_raw_inputs(date: str) -> None:
    required = [
        DEFAULT_FY3_ROOT / f"Gc_{date}.dat",
        DEFAULT_FY3_ROOT / f"Sz_{date}.dat",
        DEFAULT_FY4_ROOT / date,
        DEFAULT_FY4_GEO_RAW,
        *(merra_file(product, date) for product in MERRA_PRODUCTS),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少原始输入:\n  " + "\n  ".join(missing))


def safe_remove_tree(path: Path, root: Path) -> None:
    resolved = path.resolve()
    allowed = root.resolve()
    if resolved == allowed or allowed not in resolved.parents:
        raise ValueError(f"拒绝清理范围外路径: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def run_command(command: list[str], log_handle) -> None:
    log_handle.write("\n$ " + subprocess.list2cmdline(command) + "\n")
    log_handle.flush()
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    for line in process.stdout:
        try:
            print(line, end="")
        except UnicodeEncodeError:
            console_encoding = sys.stdout.encoding or "utf-8"
            safe_line = line.encode(console_encoding, errors="replace").decode(console_encoding)
            print(safe_line, end="")
        log_handle.write(line)
    return_code = process.wait()
    log_handle.flush()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def date_commands(date: str, work_dir: Path, staging_dir: Path) -> list[list[str]]:
    timestamp_dir = work_dir / "timestamp"
    resampled_dir = work_dir / "resampled"
    matching_dir = work_dir / "matching"
    rest2_cache_dir = work_dir / "rest2_cache"
    rest2_dir = work_dir / "rest2_on_fy3"
    timestamp_npz = timestamp_dir / f"fy3c_timestamp_{date}_prior1015w120_stride1.npz"
    fy4_slots = [
        f"{minute // 60:02d}{minute % 60:02d}"
        for _, minute in list_fy4_nc_files_in_utc_window(date, "00:00", "06:00", DEFAULT_FY4_ROOT)
    ]
    python = sys.executable
    return [
        [
            python,
            str(PROJECT_ROOT / "FY_SSI_processing" / "step_01_estimate_fy3c_timestamp_from_sz.py"),
            "--date", date,
            "--stride", "1",
            "--start", "07:00",
            "--end", "15:30",
            "--use-local-time-prior",
            "--local-overpass-time", "10:15",
            "--prior-window-minutes", "120",
            "--max-error-deg", "0.5",
            "--no-plots",
            "--output-npz", str(timestamp_npz),
            "--output-fig", str(timestamp_dir / f"fy3c_timestamp_{date}.png"),
            "--output-error-fig", str(timestamp_dir / f"fy3c_timestamp_error_{date}.png"),
        ],
        [
            python,
            str(PROJECT_ROOT / "FY_SSI_processing" / "step_02_resample_fy4_times_to_fy3_grid.py"),
            "--date", date,
            "--fy4-root", str(DEFAULT_FY4_ROOT),
            "--fy4-geo-raw", str(DEFAULT_FY4_GEO_RAW),
            "--output-dir", str(resampled_dir),
            "--skip-existing",
        ],
        [
            python,
            str(PROJECT_ROOT / "MERRA2" / "build_rest2_ghi_slot_cache.py"),
            "--date", date,
            "--output-dir", str(rest2_cache_dir),
            "--skip-existing",
            "--slots", *fy4_slots,
        ],
        [
            python,
            str(PROJECT_ROOT / "FY_SSI_processing" / "step_04_resample_rest2_ghi_to_fy3_grid.py"),
            "--date", date,
            "--rest2-cache-dir", str(rest2_cache_dir),
            "--resampled-dir", str(resampled_dir),
            "--output-dir", str(rest2_dir),
            "--skip-existing",
        ],
        [
            python,
            str(PROJECT_ROOT / "FY_SSI_processing" / "step_05_build_fy3_fy4_samples.py"),
            "--date", date,
            "--fy3-gc", str(DEFAULT_FY3_ROOT / f"Gc_{date}.dat"),
            "--timestamp-npz", str(timestamp_npz),
            "--resampled-dir", str(resampled_dir),
            "--rest2-dir", str(rest2_dir),
            "--lon-min", "73",
            "--lon-max", "135",
            "--lat-min", "18",
            "--lat-max", "54",
            "--fy3-patch", "128",
            "--fy4-patch", "32",
            "--patch-stride", "128",
            "--batch-size", "64",
            "--min-slot-valid-frac", "0.99",
            "--min-valid-frac", "0.95",
            "--min-csi-valid-frac", "0.95",
            "--max-bad-diff-frac", "0.05",
            "--output-dir", str(staging_dir),
        ],
    ]


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
            raise ValueError(f"manifest 字段不符: {path}")
        return list(reader)


def validate_date_directory(date_dir: Path, expected_date: str | None = None) -> dict[str, object]:
    date = expected_date or date_dir.name
    manifest_path = date_dir / "manifest.csv"
    meta_path = date_dir / "build_meta.npz"
    if not manifest_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"日期目录缺少 manifest.csv/build_meta.npz: {date_dir}")
    rows = read_manifest(manifest_path)
    if False and not rows:
        raise ValueError(f"日期 {date} 未生成任何样本")

    def hhmm_to_minutes(value: str) -> int:
        text = str(value).zfill(4)
        hours, minutes = int(text[:2]), int(text[2:])
        if not 0 <= hours < 24 or not 0 <= minutes < 60:
            raise ValueError(f"invalid HHMM value: {value}")
        return hours * 60 + minutes

    grouped: dict[str, list[dict[str, str]]] = {}
    spatial_keys: set[tuple[str, int, int]] = set()
    for expected_id, row in enumerate(rows):
        if row["date"] != date or int(row["sample_id"]) != expected_id:
            raise ValueError(f"manifest 日期或 sample_id 不连续: {date_dir}")
        grid_row, grid_col = int(row["row"]), int(row["col"])
        if grid_row % 128 or grid_col % 128:
            raise ValueError(f"样本窗口未对齐 128 网格: {date} row={grid_row} col={grid_col}")
        key = (row["fy4_time"], grid_row, grid_col)
        if key in spatial_keys:
            raise ValueError(f"同一时次存在重复空间窗口: {key}")
        spatial_keys.add(key)
        expected_lon = 73.0 + (grid_col + 64) * 0.01
        expected_lat = 54.0 - (grid_row + 64) * 0.01
        if not np.isclose(float(row["center_lon"]), expected_lon) or not np.isclose(
            float(row["center_lat"]), expected_lat
        ):
            raise ValueError(f"中心经纬度不匹配: {date} sample={expected_id}")
        grouped.setdefault(row["npz_file"], []).append(row)

    files = sorted(date_dir.glob(f"{date}_*.npz"))
    if {path.name for path in files} != set(grouped):
        raise ValueError(f"NPZ 文件集合与 manifest 不一致: {date_dir}")
    validated = 0
    for path in files:
        members = grouped[path.name]
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(ARRAY_SHAPES) | {
                "row0", "col0", "center_lon", "center_lat", "dominant_slot",
                "dominant_slot_minute_utc", "dominant_slot_minute_cst", "valid_frac",
                "bad_diff_frac", "dominant_slot_frac", "csi_valid_frac",
            }:
                raise ValueError(f"NPZ 字段不符合约定: {path}")
            batch_count = len(archive["fy3"])
            if not 1 <= batch_count <= 64 or batch_count != len(members):
                raise ValueError(f"NPZ 样本数量不符: {path}")
            expected_indices = list(range(batch_count))
            if [int(row["array_index"]) for row in members] != expected_indices:
                raise ValueError(f"array_index 不连续: {path}")
            expected_slot_minutes = np.asarray(
                [hhmm_to_minutes(row["fy4_time"]) for row in members], dtype=np.int32
            )
            if not np.array_equal(
                np.asarray(archive["dominant_slot_minute_utc"], dtype=np.int32),
                expected_slot_minutes,
            ):
                raise ValueError(f"NPZ/manifest time mismatch: {path}")
            if np.any(np.asarray(archive["dominant_slot_frac"]) < 0.99):
                raise ValueError(f"FY-4 slot coverage below 99%: {path}")
            if np.any(np.asarray(archive["bad_diff_frac"]) > 0.05):
                raise ValueError(f"time inversion error fraction above 5%: {path}")
            if np.any(np.asarray(archive["valid_frac"]) < 0.95):
                raise ValueError(f"joint valid fraction below 95%: {path}")
            if np.any(np.asarray(archive["csi_valid_frac"]) < 0.95):
                raise ValueError(f"CSI valid fraction below 95%: {path}")
            for name, sample_shape in ARRAY_SHAPES.items():
                array = archive[name]
                if array.shape != (batch_count, *sample_shape):
                    raise ValueError(f"数组形状错误: {path} {name}={array.shape}")
                if array.dtype != np.float32 or not np.isfinite(array).all():
                    raise ValueError(f"数组类型或有限值检查失败: {path} {name}")
            validated += batch_count
    if validated != len(rows):
        raise ValueError(f"样本总数与 manifest 不一致: {date_dir}")
    with np.load(meta_path, allow_pickle=False) as meta:
        statistics = {
            name: int(meta[name])
            for name in (
                "candidate_patches",
                "accepted_samples",
                "rejected_patches",
                "rejected_slot_valid_frac",
                "rejected_multi_slot",
                "rejected_qc",
            )
        }
    if statistics["accepted_samples"] != len(rows):
        raise ValueError(f"build_meta 样本数与 manifest 不一致: {date_dir}")
    return {
        "date": date,
        "sample_count": len(rows),
        "npz_count": len(files),
        **statistics,
    }


def status_is_valid(root: Path, date: str) -> bool:
    status_path = root / "status" / f"{date}.json"
    if not status_path.exists():
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("state") != "success":
            return False
        result = validate_date_directory(root / "samples" / date, date)
        return result["sample_count"] == status.get("sample_count")
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def process_date(root: Path, date: str, *, force: bool, keep_work: bool) -> None:
    status_path = root / "status" / f"{date}.json"
    if not force and status_is_valid(root, date):
        print(f"[skip] {date} 已完成并通过验证")
        return

    verify_raw_inputs(date)
    work_dir = root / ".work" / date
    staging_dir = root / ".staging" / date
    final_dir = root / "samples" / date
    if force:
        safe_remove_tree(work_dir, root)
        safe_remove_tree(staging_dir, root)
    work_dir.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    atomic_json(status_path, {"date": date, "state": "running", "started_at": utc_now()})

    log_path = root / "logs" / f"{date}.log"
    try:
        with log_path.open("w", encoding="utf-8", newline="") as log_handle:
            for command in date_commands(date, work_dir, staging_dir):
                script_name = Path(command[1]).name
                if script_name == "step_01_estimate_fy3c_timestamp_from_sz.py":
                    output_npz = Path(command[command.index("--output-npz") + 1])
                    if output_npz.exists():
                        log_handle.write(f"[resume] skip completed timestamp: {output_npz}\n")
                        continue
                run_command(command, log_handle)
        result = validate_date_directory(staging_dir, date)
        if final_dir.exists():
            quarantine = root / ".quarantine" / f"{date}_{int(time.time())}"
            os.replace(final_dir, quarantine)
        os.replace(staging_dir, final_dir)
        elapsed = time.time() - started
        atomic_json(
            status_path,
            {
                **result,
                "state": "success",
                "started_at": datetime.fromtimestamp(started, timezone.utc).isoformat(timespec="seconds"),
                "completed_at": utc_now(),
                "elapsed_seconds": round(elapsed, 3),
                "log": str(log_path),
            },
        )
        if not keep_work:
            safe_remove_tree(work_dir, root)
        print(f"[success] {date}: {result['sample_count']} samples")
    except Exception as exc:
        atomic_json(
            status_path,
            {
                "date": date,
                "state": "failed",
                "failed_at": utc_now(),
                "elapsed_seconds": round(time.time() - started, 3),
                "error": f"{type(exc).__name__}: {exc}",
                "log": str(log_path),
            },
        )
        raise


def write_date_list(path: Path, dates: Iterable[str]) -> None:
    values = list(dates)
    path.write_text("".join(f"{date}\n" for date in values), encoding="utf-8")


def finalize(root: Path, train_ratio: float, val_ratio: float) -> None:
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("拆分比例必须满足 train>0, val>0, train+val<1")
    date_dirs = sorted(path for path in (root / "samples").iterdir() if path.is_dir())
    summaries = [validate_date_directory(path, path.name) for path in date_dirs]
    dates = [str(item["date"]) for item in summaries]
    if not dates:
        raise ValueError("没有可汇总的成功日期")

    all_path = root / "all_samples.csv"
    with all_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        global_id = 0
        for date in dates:
            for row in read_manifest(root / "samples" / date / "manifest.csv"):
                row["sample_id"] = str(global_id)
                writer.writerow(row)
                global_id += 1

    train_end = max(1, int(len(dates) * train_ratio))
    val_end = max(train_end + 1, int(len(dates) * (train_ratio + val_ratio)))
    val_end = min(val_end, len(dates) - 1)
    write_date_list(root / "splits" / "train_dates.txt", dates[:train_end])
    write_date_list(root / "splits" / "val_dates.txt", dates[train_end:val_end])
    write_date_list(root / "splits" / "test_dates.txt", dates[val_end:])
    atomic_json(
        root / "status" / "summary.json",
        {
            "state": "complete",
            "completed_at": utc_now(),
            "date_count": len(dates),
            "sample_count": global_id,
            "npz_count": sum(int(item["npz_count"]) for item in summaries),
            "first_date": dates[0],
            "last_date": dates[-1],
            "train_dates": train_end,
            "val_dates": val_end - train_end,
            "test_dates": len(dates) - val_end,
        },
    )
    print(f"汇总完成: dates={len(dates)}, samples={global_id}, all_samples={all_path}")


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    ensure_layout(root)
    dates = selected_dates(args)
    if args.command == "run":
        if args.workers < 1:
            raise ValueError("--workers 必须 >= 1")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_date,
                    root,
                    date,
                    force=args.force,
                    keep_work=args.keep_work,
                ): date
                for date in dates
            }
            completed = 0
            for future in as_completed(futures):
                date = futures[future]
                completed += 1
                try:
                    future.result()
                except Exception as exc:
                    print(f"[failed {completed}/{len(dates)}] {date}: {exc}", file=sys.stderr)
                else:
                    print(f"[done {completed}/{len(dates)}] {date}")
        return
    if args.command == "validate":
        for date in dates:
            result = validate_date_directory(root / "samples" / date, date)
            print(json.dumps(result, ensure_ascii=False))
        return
    finalize(root, args.train_ratio, args.val_ratio)


if __name__ == "__main__":
    main()
