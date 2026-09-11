"""Export the exact native CSI arrays used by the compact Figure 5-6.

Run with the DRCT environment.  This script is inference-only: it reads one
sample 6 from the final ten-sample paper figure and the epoch-271 checkpoint, then writes a
temporary 32×32 LR, 128×128 SR, and 128×128 HR array archive for the SVG
renderer.  It never writes training data or alters a checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch


FY_PROCESSING_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = FY_PROCESSING_ROOT.parent
DRCT_ROOT = PROJECT_ROOT / "drct"
CHECKPOINT_PATH = (
    DRCT_ROOT
    / "results"
    / "fy4a_fy3c_drctl_w8_scratch_50k_b16_e50"
    / "checkpoints"
    / "epoch_271_converged_final.pth"
)
CSI_MAX = 1.5
VAL_DATES = [f"201806{day:02d}" for day in range(20, 31)]
PAPER_INFERENCE_SAMPLES = 120
PAPER_SAMPLE_NUMBER = 6


def select_paper_sample() -> tuple[np.ndarray, np.ndarray, int]:
    """Reproduce the final paper figure's representative-sample selection."""
    sys.path.insert(0, str(DRCT_ROOT))
    from data.dataset_SR_satellite import split_by_dates

    _, validation_dataset = split_by_dates(
        samples_root=PROJECT_ROOT / "outputs" / "fy" / "samples",
        val_dates=VAL_DATES,
        augment=False,
    )
    indices = np.linspace(0, len(validation_dataset) - 1, PAPER_INFERENCE_SAMPLES, dtype=np.int64)
    observations = np.stack([validation_dataset[int(index)][1].numpy()[0] for index in indices])
    order = np.argsort(observations.std(axis=(1, 2)))
    ranks = np.rint(np.linspace(0.15, 0.98, 10) * (PAPER_INFERENCE_SAMPLES - 1)).astype(np.int64)
    selected_position = int(order[ranks[PAPER_SAMPLE_NUMBER - 1]])
    validation_index = int(indices[selected_position])
    lr, hr = validation_dataset[validation_index]
    return lr.numpy()[0], hr.numpy()[0], validation_index


def export_native_sample(output: Path) -> None:
    """Infer native 128×128 SR for sample 6 from the ten-sample figure."""
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Missing final DRCT checkpoint: {CHECKPOINT_PATH}")
    lr_normalized, hr_normalized, validation_index = select_paper_sample()
    lr_normalized = np.asarray(lr_normalized, dtype=np.float32)
    hr_normalized = np.asarray(hr_normalized, dtype=np.float32)
    lr = lr_normalized * CSI_MAX
    hr = hr_normalized * CSI_MAX
    if lr.shape != (32, 32) or hr.shape != (128, 128):
        raise ValueError(f"Expected native LR/HR shapes (32, 32)/(128, 128), got {lr.shape}/{hr.shape}")

    sys.path.insert(0, str(DRCT_ROOT))
    from models.network_drct import DRCT

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model = DRCT(**dict(checkpoint["network"]))
    model.load_state_dict(checkpoint["params_ema"], strict=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    with torch.inference_mode():
        prediction = model(torch.from_numpy(lr_normalized).view(1, 1, 32, 32).to(device)).float().cpu().numpy()[0, 0]
    sr = np.clip(prediction * CSI_MAX, 0.0, CSI_MAX).astype(np.float32)
    if sr.shape != (128, 128):
        raise ValueError(f"Final DRCT inference did not produce a 128×128 SR field: {sr.shape}")

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        lr_csi=lr,
        sr_csi=sr,
        hr_csi=hr,
        paper_sample_number=PAPER_SAMPLE_NUMBER,
        validation_index=validation_index,
        checkpoint=str(CHECKPOINT_PATH),
    )
    print(f"Paper figure sample: {PAPER_SAMPLE_NUMBER} | validation index: {validation_index}")
    print("Native array check: LR=(32, 32), SR=(128, 128), HR=(128, 128)")
    print(f"Temporary export: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_native_sample(args.output)
