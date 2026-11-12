"""
Validate downloaded MERRA-2 nc4 files.

Checks:
  - expected file exists
  - file opens as NetCDF
  - required variables are present
  - internal time units/date match the date in the filename

Writes a CSV report to validation_reports/.
"""

from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path

from netCDF4 import Dataset

from data_paths import merra2_data_root


ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "validation_reports"

DATA_ROOT = merra2_data_root()

PRODUCTS = {
    "aer": {
        "dir": DATA_ROOT / "tavg1_2d_aer_Nx",
        "short": "tavg1_2d_aer_Nx",
        "vars": {"TOTEXTTAU", "TOTANGSTR", "time", "lat", "lon"},
        "expected_count": 730,
    },
    "rad": {
        "dir": DATA_ROOT / "tavg1_2d_rad_Nx",
        "short": "tavg1_2d_rad_Nx",
        "vars": {"ALBEDO", "time", "lat", "lon"},
        "expected_count": 731,
    },
    "slv": {
        "dir": DATA_ROOT / "tavg1_2d_slv_Nx",
        "short": "tavg1_2d_slv_Nx",
        "vars": {"PS", "TO3", "TQV", "time", "lat", "lon"},
        "expected_count": 730,
    },
}

DATE_IN_NAME = re.compile(r"\.(\d{8})\.SUB\.nc4$")
DATE_IN_UNITS = re.compile(r"since\s+(\d{4}-\d{2}-\d{2})")


def _expected_dates(product_key: str) -> list[str]:
    if product_key == "rad":
        start = datetime(2017, 12, 31)
        end = datetime(2019, 12, 31)
    else:
        start = datetime(2018, 1, 1)
        end = datetime(2019, 12, 31)

    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y%m%d"))
        current = current.fromordinal(current.toordinal() + 1)
    return dates


def _validate_file(path: Path, expected_vars: set[str], expected_date: str) -> tuple[str, str]:
    if not path.exists():
        return "missing", "file does not exist"
    if path.suffix == ".part":
        return "partial", "partial download file"

    try:
        with Dataset(str(path)) as ds:
            present = set(ds.variables)
            missing_vars = sorted(expected_vars - present)
            if missing_vars:
                return "bad_vars", f"missing variables: {missing_vars}"

            time_var = ds.variables["time"]
            units = getattr(time_var, "units", "")
            match = DATE_IN_UNITS.search(units)
            if not match:
                return "bad_time", f"cannot parse time units: {units!r}"

            internal_date = match.group(1).replace("-", "")
            if internal_date != expected_date:
                return "date_mismatch", f"name={expected_date}, internal={internal_date}, units={units!r}"

            if len(ds.dimensions.get("time", [])) != 24:
                return "bad_time_len", f"time length={len(ds.dimensions.get('time', []))}"

            return "ok", ""
    except Exception as exc:  # noqa: BLE001
        return "not_netcdf", f"{type(exc).__name__}: {exc}"


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "merra2_validation_report.csv"

    rows = []
    summary: dict[str, dict[str, int]] = {}

    for product_key, cfg in PRODUCTS.items():
        summary[product_key] = {}
        for date in _expected_dates(product_key):
            name = f"MERRA2_400.{cfg['short']}.{date}.SUB.nc4"
            path = cfg["dir"] / name
            status, message = _validate_file(path, cfg["vars"], date)
            rows.append(
                {
                    "product": product_key,
                    "date": date,
                    "path": str(path),
                    "status": status,
                    "message": message,
                }
            )
            summary[product_key][status] = summary[product_key].get(status, 0) + 1

    with report_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["product", "date", "path", "status", "message"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Report: {report_path}")
    print("Summary:")
    for product_key in PRODUCTS:
        parts = ", ".join(f"{k}={v}" for k, v in sorted(summary[product_key].items()))
        print(f"  {product_key}: {parts}")

    bad = [row for row in rows if row["status"] != "ok"]
    if bad:
        print("\nFirst bad rows:")
        for row in bad[:20]:
            print(f"  {row['product']} {row['date']} {row['status']}: {row['message']}")
        if len(bad) > 20:
            print(f"  ... {len(bad) - 20} more")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
