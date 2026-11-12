"""Generate one random full-swath FY-3/FY-4/REST2 QA plot per missing month."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rebuild_2018_2019_dataset import DEFAULT_DATASET_ROOT, PROJECT_ROOT, date_commands


MONTHS = [f"{year}{month:02d}" for year, start in ((2018, 7), (2019, 1)) for month in range(start, 13)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261024)
    return parser.parse_args()


def choose_dates(root: Path, seed: int) -> list[str]:
    rng = random.Random(seed)
    choices: list[str] = []
    for month in MONTHS:
        dates = []
        for date_dir in sorted((root / "samples").glob(f"{month}??")):
            manifest = date_dir / "manifest.csv"
            if manifest.exists() and manifest.stat().st_size > 120:
                dates.append(date_dir.name)
        if not dates:
            raise RuntimeError(f"no non-empty date for {month}")
        choices.append(rng.choice(dates))
    return choices


def run_logged(command: list[str], handle) -> None:
    handle.write("$ " + subprocess.list2cmdline(command) + "\n")
    handle.flush()
    subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT, text=True)


def process_date(root: Path, date: str) -> tuple[str, Path]:
    work = root / ".qc_swath_work" / date
    staging = work / "unused_staging"
    output_dir = root / "qc_monthly_swaths"
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    commands = date_commands(date, work, staging)[:4]
    timestamp = work / "timestamp" / f"fy3c_timestamp_{date}_prior1015w120_stride1.npz"
    resampled = work / "resampled"
    rest2 = work / "rest2_on_fy3"
    matching = work / "matching"
    plots = work / "plots"
    commands.append(
        [
            sys.executable,
            str(PROJECT_ROOT / "FY_SSI_processing" / "step_03_build_fy3_fy4_slot_map.py"),
            "--date", date,
            "--timestamp-npz", str(timestamp),
            "--resampled-dir", str(resampled),
            "--output-dir", str(matching),
        ]
    )

    with (log_dir / f"{date}.log").open("w", encoding="utf-8") as handle:
        for command in commands:
            run_logged(command, handle)
        slot_maps = sorted(matching.glob("fy4_slot_map_*.npz"))
        if len(slot_maps) != 1:
            raise RuntimeError(f"expected one slot map for {date}, found {len(slot_maps)}")
        plot_command = [
            sys.executable,
            str(PROJECT_ROOT / "FY_SSI_processing" / "plots" / "plot_two_swaths.py"),
            "--date", date,
            "--fy3-gc", rf"E:\FY-3C\SSI\Gc_{date}.dat",
            "--timestamp-npz", str(timestamp),
            "--resampled-dir", str(resampled),
            "--rest2-dir", str(rest2),
            "--slot-map-npz", str(slot_maps[0]),
            "--output-dir", str(plots),
        ]
        run_logged(plot_command, handle)

    candidates = sorted(plots.glob("swath1_triple*.png"))
    if not candidates:
        candidates = sorted(plots.glob("swath*_triple*.png"))
    if not candidates:
        raise RuntimeError(f"no swath plot generated for {date}")
    destination = output_dir / f"{date[:6]}_{date}_swath.png"
    shutil.copy2(candidates[0], destination)
    shutil.rmtree(work)
    return date, destination


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    dates = choose_dates(root, args.seed)
    selection_path = root / "qc_monthly_swaths" / "monthly_selection.csv"
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    with selection_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("month", "date"))
        writer.writerows((date[:6], date) for date in dates)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_date, root, date): date for date in dates}
        for number, future in enumerate(as_completed(futures), start=1):
            date, output = future.result()
            print(f"[{number}/{len(dates)}] {date}: {output}", flush=True)


if __name__ == "__main__":
    main()
