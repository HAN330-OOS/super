"""Live console monitor for the 2018-2019 dataset rebuild."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path


DEFAULT_ROOT = Path(r"E:\Super-Resolution-Samples-2018-2019")
TOTAL_DATES = 600


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实时显示两年样本重建进度")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}


def read_statuses(root: Path) -> list[dict[str, object]]:
    active_run = read_json(root / "status" / "active_run.json")
    run_started_at = str(active_run.get("started_at", ""))
    statuses: list[dict[str, object]] = []
    for path in sorted((root / "status").glob("*.json")):
        status = read_json(path)
        if not status.get("date"):
            continue
        timestamp = str(
            status.get("started_at")
            or status.get("completed_at")
            or status.get("failed_at")
            or ""
        )
        if run_started_at and timestamp < run_started_at:
            continue
        statuses.append(status)
    return statuses


def duration_text(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "计算中"
    hours, remainder = divmod(int(seconds), 3600)
    minutes, _ = divmod(remainder, 60)
    return f"{hours}小时{minutes:02d}分"


def render(root: Path) -> None:
    statuses = read_statuses(root)
    success = sorted(
        (item for item in statuses if item.get("state") == "success"),
        key=lambda item: str(item["date"]),
    )
    running = sorted(str(item["date"]) for item in statuses if item.get("state") == "running")
    failed = sorted(str(item["date"]) for item in statuses if item.get("state") == "failed")
    elapsed = [float(item["elapsed_seconds"]) for item in success if item.get("elapsed_seconds")]
    average = sum(elapsed) / len(elapsed) if elapsed else None
    remaining = (TOTAL_DATES - len(success)) * average / 2 if average else None
    sample_count = sum(int(item.get("sample_count", 0)) for item in success)
    latest = success[-1]["date"] if success else "无"
    percent = 100.0 * len(success) / TOTAL_DATES
    validation_log = root / "logs" / "final_validation.log"
    validated_dates = 0
    if validation_log.exists():
        try:
            log_bytes = validation_log.read_bytes()
            log_encoding = "utf-16" if log_bytes.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
            validated_dates = sum(
                line.startswith('{"date"')
                for line in log_bytes.decode(log_encoding).splitlines()
            )
        except (OSError, UnicodeDecodeError):
            validated_dates = 0
    summary_exists = (root / "status" / "summary.json").exists()
    all_samples_exists = (root / "all_samples.csv").exists()

    os.system("cls" if os.name == "nt" else "clear")
    print("FY-3C / FY-4A 两年样本重建实时进度")
    print("=" * 52)
    print(f"完成日期 : {len(success):3d} / {TOTAL_DATES}  ({percent:6.2f}%)")
    print(f"最新完成 : {latest}")
    print(f"正在处理 : {', '.join(running) if running else '无'}")
    print(f"失败日期 : {len(failed)}" + (f"  {', '.join(failed[-5:])}" if failed else ""))
    print(f"累计样本 : {sample_count:,}")
    print(f"平均耗时 : {duration_text(average)} / 天")
    print(f"预计剩余 : {duration_text(remaining)}（按2天并行粗估）")
    if len(success) == TOTAL_DATES:
        print(f"全量验证 : {validated_dates:3d} / {TOTAL_DATES}")
        print(f"最终汇总 : {'已完成' if summary_exists and all_samples_exists else '等待验证完成'}")
    print(f"更新时间 : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n按 Ctrl+C 退出监控；后台处理不会停止。")


def main() -> None:
    args = parse_args()
    while True:
        render(args.dataset_root.resolve())
        if args.once:
            return
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    main()
