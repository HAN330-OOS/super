"""
Repair invalid or missing MERRA-2 downloads using the validation report.

Workflow:
  1. Run step_02_validate_merra2_downloads.py first.
  2. This script reads validation_reports/merra2_validation_report.csv.
  3. Deletes invalid files and .part files for bad rows.
  4. Re-downloads only rows whose status is not "ok".

Environment:
  EARTHDATA_USER
  EARTHDATA_PASS
"""

from __future__ import annotations

import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from step_01_download_merra2 import (
    MAX_WORKERS,
    PRODUCTS,
    build_session,
    download_one,
    load_urls,
    target_filename,
)


ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "validation_reports" / "merra2_validation_report.csv"
REPAIR_WORKERS = min(4, MAX_WORKERS)


def _url_maps() -> dict[str, dict[str, str]]:
    maps: dict[str, dict[str, str]] = {}
    for product, cfg in PRODUCTS.items():
        date_to_url: dict[str, str] = {}
        for url in load_urls(cfg["txt"]):
            name = target_filename(url, cfg["short"])
            if not name:
                continue
            parts = name.split(".")
            if len(parts) >= 4:
                date_to_url[parts[2]] = url
        maps[product] = date_to_url
    return maps


def _load_bad_rows() -> list[dict[str, str]]:
    if not REPORT_PATH.exists():
        raise FileNotFoundError(f"Validation report missing: {REPORT_PATH}")
    with REPORT_PATH.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [row for row in rows if row["status"] != "ok"]


def _delete_bad_artifacts(rows: list[dict[str, str]]) -> None:
    deleted = 0
    for row in rows:
        path = Path(row["path"])
        for candidate in (path, path.with_suffix(path.suffix + ".part")):
            if candidate.exists():
                candidate.unlink()
                deleted += 1
    print(f"Deleted invalid/partial artifacts: {deleted}")


def _repair_one(row: dict[str, str], url_maps: dict[str, dict[str, str]]) -> tuple[str, str, str]:
    user = os.environ["EARTHDATA_USER"]
    password = os.environ["EARTHDATA_PASS"]
    product = row["product"]
    date = row["date"]
    path = Path(row["path"])
    url = url_maps[product].get(date)
    if not url:
        return "fail", f"{product} {date}", "URL not found in subset txt"

    session = build_session(user, password)
    status, msg = download_one(session, url, path, path.name)
    return status, f"{product} {date}", msg


def main() -> int:
    if not os.environ.get("EARTHDATA_USER") or not os.environ.get("EARTHDATA_PASS"):
        print("ERROR: set EARTHDATA_USER and EARTHDATA_PASS first.")
        return 2

    rows = _load_bad_rows()
    print(f"Rows needing repair: {len(rows)}")
    if not rows:
        print("Nothing to repair.")
        return 0

    _delete_bad_artifacts(rows)
    url_maps = _url_maps()

    ok = skip = fail = 0
    failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=REPAIR_WORKERS) as pool:
        futures = [pool.submit(_repair_one, row, url_maps) for row in rows]
        for done, fut in enumerate(as_completed(futures), start=1):
            status, label, msg = fut.result()
            if status == "ok":
                ok += 1
                print(f"[{done}/{len(futures)}] OK   {label}: {msg}", flush=True)
            elif status == "skip":
                skip += 1
                print(f"[{done}/{len(futures)}] SKIP {label}: {msg}", flush=True)
            else:
                fail += 1
                failures.append((label, msg))
                print(f"[{done}/{len(futures)}] FAIL {label}: {msg}", flush=True)

    print(f"Repair summary: ok={ok} skip={skip} fail={fail}")
    if failures:
        print("First failures:")
        for label, msg in failures[:20]:
            print(f"  {label}: {msg}")
        if len(failures) > 20:
            print(f"  ... {len(failures) - 20} more")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
