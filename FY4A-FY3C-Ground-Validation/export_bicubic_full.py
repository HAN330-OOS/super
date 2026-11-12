"""Export the full Bicubic CSI baseline in the formal SQLite test order."""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from contextlib import closing
from pathlib import Path

import numpy as np

from step_2_satellite import PatchReader, load_config, require, sha256, signature


def export(config, directory):
    import cv2

    require(cv2.__version__ == "5.0.0", "Use OpenCV 5.0.0 for the formal Bicubic baseline")
    cfg = load_config(config)
    index_dir = Path(cfg["_output"])
    identity = json.loads((index_dir / "satellite_identity.json").read_text(encoding="utf-8"))
    require(identity["status"] == "ready", "Build the satellite index first")
    require(sha256(cfg["_path"]) == identity["config_sha256"], "Config/index fingerprint mismatch")
    index_path = index_dir / "test_patch_index.sqlite"
    require(sha256(index_path) == identity["index_db_sha256"], "Test index checksum mismatch")
    with closing(sqlite3.connect(index_path.as_uri() + "?mode=ro", uri=True)) as db:
        rows = [json.loads(value) for value, in db.execute(
            "SELECT record_json FROM samples ORDER BY test_index")]
    count = cfg["expected_test_samples"]
    require(len(rows) == count and all(row["test_index"] == i for i, row in enumerate(rows)),
            "Test order is incomplete")

    groups = defaultdict(list)
    for row in rows:
        groups[Path(row["npz_path"])].append(row)

    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    final_path = directory / "bicubic_prediction_csi.npy"
    partial_path = final_path.with_suffix(".npy.partial")
    shape = (count, 128, 128)
    prediction = np.lib.format.open_memmap(partial_path, mode="w+", dtype=np.float32, shape=shape)
    reader = PatchReader(capacity=2)
    try:
        for path, members in groups.items():
            require(signature(path) == identity["npz_signatures"].get(str(path)),
                    f"NPZ differs from the validated index: {path}")
            low = reader.arrays(path)["lr_csi"]
            for row in members:
                sample = low[row["array_index"]]
                result = cv2.resize(sample, (128, 128), interpolation=cv2.INTER_CUBIC)
                require(np.isfinite(result).all(), f"Nonfinite prediction: {row['sample_id']}")
                prediction[row["test_index"]] = result
        prediction.flush()
    finally:
        reader.close()
        del prediction
    partial_path.rename(final_path)
    digest = sha256(final_path)
    manifest = {
        "method": "Bicubic",
        "source": "existing official 2018-2019 test samples, lr_csi",
        "test_index_source": str(index_path),
        "sample_count": count,
        "shape": list(shape),
        "dtype": "float32",
        "order": "test_patch_index.sqlite ORDER BY test_index; same official test indices as DRCT",
        "implementation": "cv2.resize((128,128), interpolation=cv2.INTER_CUBIC)",
        "opencv_version": cv2.__version__,
        "normalization": "none",
        "clipping": False,
        "sha256": digest,
        "output": str(final_path),
    }
    (directory / "bicubic_prediction_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--output-dir", type=Path, required=True, help="A new directory for the export")
    args = parser.parse_args()
    print(json.dumps(export(args.config, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
