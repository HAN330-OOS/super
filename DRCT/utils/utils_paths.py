"""Filesystem paths shared by the active FY-4A/FY-3C DRCT experiment."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def resolve_fy_samples_root() -> Path:
    """Return the sample root for either a standalone clone or this workspace.

    ``FY_CSI_SAMPLES_ROOT`` takes precedence.  A repository-local ``outputs``
    directory is preferred for a fresh clone; the parent-workspace fallback
    keeps the current development layout compatible.
    """

    configured = os.environ.get("FY_CSI_SAMPLES_ROOT")
    candidates = [
        Path(configured).expanduser() if configured else None,
        PROJECT_ROOT / "outputs" / "fy" / "samples",
        WORKSPACE_ROOT / "outputs" / "fy" / "samples",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate.resolve()
    return (PROJECT_ROOT / "outputs" / "fy" / "samples").resolve()


FY_SAMPLES_ROOT = resolve_fy_samples_root()
