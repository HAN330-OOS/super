"""Resolve the directory containing local MERRA-2 input products."""
from __future__ import annotations

import os
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent


def merra2_data_root() -> Path:
    """Prefer the environment override, then the local setting, then MERRA2/."""
    configured = os.environ.get("MERRA2_DATA_ROOT", "").strip()
    settings = MODULE_DIR / "data_root.local"
    if not configured and settings.is_file():
        configured = settings.read_text(encoding="utf-8-sig").strip()
    if not configured:
        return MODULE_DIR
    return (MODULE_DIR / Path(configured).expanduser()).resolve()
