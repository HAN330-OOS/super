"""Fit/reuse a frozen TRAIN-only CSI metric reference without exporting data."""
from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

from utils.metrics import fit_training_reference, load_metric_reference


def prepare(data_root: Path, output: Path) -> dict:
    data_root, output = data_root.expanduser().resolve(), output.expanduser().resolve()
    try:
        output.relative_to(data_root)
    except ValueError:
        pass
    else:
        raise ValueError("--output must be outside the source dataset folder; source data is read-only")
    if output.exists():
        reference = load_metric_reference(output, data_root)
        return {"status": "reused_existing_reference", "output": str(output), "reference": reference}
    reference = fit_training_reference(data_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(reference, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # Windows rename refuses an existing target. POSIX hard linking
            # supplies the same atomic, non-replacing creation semantics.
            if os.name == "nt":
                os.rename(temporary, output)
            else:
                os.link(temporary, output)
        except FileExistsError:
            reference = load_metric_reference(output, data_root)
            return {"status": "reused_concurrently_created_reference", "output": str(output), "reference": reference}
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"status": "created_frozen_training_reference", "output": str(output), "reference": reference}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Frozen reference JSON outside the source data folder")
    args = parser.parse_args()
    print(json.dumps(prepare(args.data_root, args.output), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
