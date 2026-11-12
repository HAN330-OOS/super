"""
MERRA-2 batch downloader (Earthdata Login).

Reads three subset link text files and downloads:
  - tavg1_2d_aer_Nx (TOTEXTTAU, TOTANGSTR)  -> <data root>/tavg1_2d_aer_Nx/
  - tavg1_2d_rad_Nx (ALBEDO)                -> <data root>/tavg1_2d_rad_Nx/
  - tavg1_2d_slv_Nx (PS, TO3, TQV)          -> <data root>/tavg1_2d_slv_Nx/

Uses Earthdata Login Basic Auth, concurrent downloads, atomic writes,
skips already-downloaded files (resume), retries on transient errors.

Usage (PowerShell):
    $env:EARTHDATA_USER = "your_username"
    $env:EARTHDATA_PASS = "your_password"
    py -3 D:/Super-Resolution/MERRA2/step_01_download_merra2.py
"""

from __future__ import annotations

import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse, unquote, parse_qs

import requests

from data_paths import merra2_data_root

ROOT = Path(__file__).resolve().parent

DATA_ROOT = merra2_data_root()

PRODUCTS = {
    "aer": {
        "txt": ROOT / "subset_txt" / "subset_M2T1NXAER_5.12.4_20260522_053634_.txt",
        "out_dir": DATA_ROOT / "tavg1_2d_aer_Nx",
        "short": "tavg1_2d_aer_Nx",
    },
    "rad": {
        "txt": ROOT / "subset_txt" / "subset_links_6e5f34dd-5911-4a75-b53a-5f507f863d54.txt",
        "out_dir": DATA_ROOT / "tavg1_2d_rad_Nx",
        "short": "tavg1_2d_rad_Nx",
    },
    "slv": {
        "txt": ROOT / "subset_txt" / "subset_M2T1NXSLV_5.12.4_20260522_063115_.txt",
        "out_dir": DATA_ROOT / "tavg1_2d_slv_Nx",
        "short": "tavg1_2d_slv_Nx",
    },
}

DATE_RE = re.compile(r"\.(\d{8})[._]")

# Tunables
MAX_WORKERS = 8
CHUNK_SIZE = 1 << 16  # 64 KiB
PER_REQUEST_TIMEOUT = (30, 300)  # (connect, read) seconds
MAX_RETRIES = 5
RETRY_BACKOFF_BASE = 3.0
MIN_VALID_BYTES = 4096  # treat smaller files as broken


_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def target_filename(url: str, short_name: str) -> str | None:
    """Derive a stable filename from any of the three subset URL styles."""
    # 1) slv style: ...HTTP_services.cgi?...&LABEL=MERRA2_400.tavg1_2d_slv_Nx.20180101.SUB.nc&...
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    if "LABEL" in qs and qs["LABEL"]:
        label = unquote(qs["LABEL"][0])
        if label.endswith(".nc"):
            label += "4"  # standardize to .nc4
        return label

    # 2) opendap aer style: .../M2T1NXAER.5.12.4%3AMERRA2_400.tavg1_2d_aer_Nx.20180101.nc4.dap.nc4?dap4.ce=...
    # 3) harmony rad style: .../M2T1NXRAD.5.12.4_MERRA2_400.tavg1_2d_rad_Nx.20180101_ALBEDO_subsetted.nc4
    tail = unquote(parsed.path.rsplit("/", 1)[-1])
    m = DATE_RE.search(tail)
    if m:
        date = m.group(1)
        return f"MERRA2_400.{short_name}.{date}.SUB.nc4"

    return None


def load_urls(txt_path: Path) -> list[str]:
    urls: list[str] = []
    with txt_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "README" in line.upper() or line.lower().endswith(".pdf"):
                continue
            urls.append(line)
    return urls


def download_one(
    session: requests.Session,
    url: str,
    out_path: Path,
    label: str,
) -> tuple[str, str]:
    """
    Download a single URL with retries. Returns (status, message).
    status in {"ok", "skip", "fail"}.
    """
    if out_path.exists() and out_path.stat().st_size >= MIN_VALID_BYTES:
        return "skip", f"{label} exists ({out_path.stat().st_size} B)"

    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    last_err = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with session.get(
                url, stream=True, timeout=PER_REQUEST_TIMEOUT, allow_redirects=True
            ) as r:
                if r.status_code in (401, 403):
                    return "fail", (
                        f"HTTP {r.status_code} (auth)."
                        " Check EARTHDATA_USER/PASS and that you've accepted the"
                        " GES DISC / NASA Earthdata EULA in your profile."
                    )
                r.raise_for_status()
                total = int(r.headers.get("Content-Length") or 0)
                written = 0
                with tmp_path.open("wb") as fh:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            fh.write(chunk)
                            written += len(chunk)
                if total and written != total:
                    raise IOError(
                        f"short read: got {written} of {total} bytes"
                    )
                if written < MIN_VALID_BYTES:
                    raise IOError(f"file too small ({written} bytes)")
            os.replace(tmp_path, out_path)
            return "ok", f"{label} {written/1024/1024:.2f} MB"
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            if attempt < MAX_RETRIES:
                sleep_for = RETRY_BACKOFF_BASE * attempt
                log(f"  [retry {attempt}/{MAX_RETRIES-1}] {label}: {last_err}; sleep {sleep_for:.0f}s")
                time.sleep(sleep_for)
            else:
                return "fail", last_err
    return "fail", last_err


class SessionWithHeaderRedirection(requests.Session):
    """
    Earthdata authentication helper.

    The default requests behavior strips the ``Authorization`` header on
    cross-host redirects. Earthdata data servers (goldsmr4.gesdisc, opendap,
    harmony) redirect to ``urs.earthdata.nasa.gov`` for login, then back to
    the data host. We must keep Basic Auth attached for redirects between
    those hosts so the login server sees our credentials.
    """

    AUTH_HOST = "urs.earthdata.nasa.gov"

    def __init__(self, username: str, password: str) -> None:
        super().__init__()
        self.auth = (username, password)

    def rebuild_auth(self, prepared_request, response):  # type: ignore[override]
        headers = prepared_request.headers
        url = prepared_request.url
        if "Authorization" in headers:
            original_parsed = requests.utils.urlparse(response.request.url)
            redirect_parsed = requests.utils.urlparse(url)
            if (
                (original_parsed.hostname != redirect_parsed.hostname)
                and redirect_parsed.hostname != self.AUTH_HOST
                and original_parsed.hostname != self.AUTH_HOST
            ):
                del headers["Authorization"]
        return


def build_session(user: str, password: str) -> requests.Session:
    s = SessionWithHeaderRedirection(user, password)
    s.headers["User-Agent"] = "MERRA2-batch-downloader/1.0"
    return s


def run() -> int:
    user = os.environ.get("EARTHDATA_USER")
    password = os.environ.get("EARTHDATA_PASS")
    if not user or not password:
        log("ERROR: set EARTHDATA_USER and EARTHDATA_PASS environment variables.")
        return 2

    only = set((sys.argv[1:] or list(PRODUCTS.keys())))
    bad = only - set(PRODUCTS)
    if bad:
        log(f"Unknown product keys: {bad}. Valid: {list(PRODUCTS)}")
        return 2

    session = build_session(user, password)

    grand_ok = grand_skip = grand_fail = 0
    failures: list[tuple[str, str, str]] = []  # (product, filename, error)

    for key in ["aer", "rad", "slv"]:
        if key not in only:
            continue
        cfg = PRODUCTS[key]
        cfg["out_dir"].mkdir(parents=True, exist_ok=True)
        if not cfg["txt"].exists():
            log(f"!! {key}: link file missing: {cfg['txt']}")
            continue

        urls = load_urls(cfg["txt"])
        log(f"\n=== {key}: {len(urls)} URLs -> {cfg['out_dir']}")

        tasks: list[tuple[str, Path, str]] = []
        seen_names: set[str] = set()
        for url in urls:
            name = target_filename(url, cfg["short"])
            if not name:
                log(f"  ?? cannot derive name from URL: {url}")
                continue
            if name in seen_names:
                continue
            seen_names.add(name)
            tasks.append((url, cfg["out_dir"] / name, name))

        ok = skip = fail = 0
        start = time.time()
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futs = {
                pool.submit(download_one, session, url, path, label): (label, path)
                for (url, path, label) in tasks
            }
            done = 0
            total = len(futs)
            for fut in as_completed(futs):
                label, _path = futs[fut]
                status, msg = fut.result()
                done += 1
                if status == "ok":
                    ok += 1
                    log(f"[{key} {done}/{total}] OK   {msg}")
                elif status == "skip":
                    skip += 1
                    if done % 50 == 0 or done == total:
                        log(f"[{key} {done}/{total}] (skipped existing batch)")
                else:
                    fail += 1
                    failures.append((key, label, msg))
                    log(f"[{key} {done}/{total}] FAIL {label}: {msg}")

        elapsed = time.time() - start
        log(
            f"--- {key} summary: ok={ok}  skip={skip}  fail={fail}  "
            f"elapsed={elapsed/60:.1f} min"
        )
        grand_ok += ok
        grand_skip += skip
        grand_fail += fail

    log("\n=== ALL DONE ===")
    log(f"ok={grand_ok}  skip={grand_skip}  fail={grand_fail}")
    if failures:
        log("Failed downloads (rerun the script to retry):")
        for prod, name, err in failures[:20]:
            log(f"  - {prod}/{name}: {err}")
        if len(failures) > 20:
            log(f"  ... and {len(failures) - 20} more")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run())
