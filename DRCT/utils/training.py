"""Model construction and helpers retained from the original DRCT experiment."""
from __future__ import annotations
import csv
import json
import math
import os
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from models.network_drct import DRCT

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPTION_PATH = PROJECT_ROOT / "exp" / "train_x4.json"
EARLY_STOP_MIN_DELTA = 1e-4
EARLY_STOP_PATIENCE = 5


def ui_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = ("arialbd.ttf", "arial.ttf") if bold else ("arial.ttf", "arialbd.ttf")
    for name in names:
        path = Path(r"C:\Windows\Fonts") / name
        try:
            return ImageFont.truetype(str(path), size)
        except OSError:
            continue
    for name in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
                 "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def date_for_index(dataset: Any, dataset_index: int) -> str:
    batch_path, _ = dataset.index[int(dataset_index)]
    return batch_path.parents[1].name.removesuffix("_utc")


def load_network_options() -> dict[str, Any]:
    with OPTION_PATH.open(encoding="utf-8") as handle:
        options = json.load(handle)["network"]
    options = dict(options)
    options.update({"img_size": 32, "window_size": 8, "in_chans": 1, "upscale": 4})
    return options


def build_model(device: torch.device) -> DRCT:
    # 此处禁止读取权重：新的正式训练必须从模型原生初始化开始。
    model = DRCT(**load_network_options())
    return model.to(device)


def sobel_magnitude(image: torch.Tensor) -> torch.Tensor:
    kernel_x = image.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(-1, -2)
    padded = F.pad(image.float(), (1, 1, 1, 1), mode="reflect")
    gradient_x = F.conv2d(padded, kernel_x)
    gradient_y = F.conv2d(padded, kernel_y)
    return torch.sqrt(gradient_x.square() + gradient_y.square() + 1e-12)


def bicubic_batch(low_resolution: torch.Tensor) -> torch.Tensor:
    arrays = [
        cv2.resize(image[0].numpy(), (128, 128), interpolation=cv2.INTER_CUBIC)
        for image in low_resolution.detach().cpu()
    ]
    return torch.from_numpy(np.stack(arrays)[:, None]).float()


def read_training_log(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def draw_progress(log_path: Path, baseline: dict[str, float], output_path: Path) -> None:
    rows = read_training_log(log_path)
    if not rows:
        return
    epoch = np.array([int(row["epoch"]) for row in rows])
    loss = np.array([float(row["val_l1_mae"]) for row in rows])
    psnr = np.array([float(row["val_psnr"]) for row in rows])
    ssim = np.array([float(row["val_ssim"]) for row in rows])
    grad = np.array([float(row["val_grad"]) for row in rows])
    width, height = 2200, 610
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = ui_font(27, bold=True)
    panel_font = ui_font(20, bold=True)
    axis_font = ui_font(15)
    heading = "DRCT-L W8 x4 From-Scratch Training Progress"
    heading_box = draw.textbbox((0, 0), heading, font=title_font)
    draw.text(((width - (heading_box[2] - heading_box[0])) / 2, 8), heading,
              fill="black", font=title_font)
    panels = [
        (loss, "Validation L1 / MAE", "Raw CSI MAE", None, "min"),
        (psnr, "Validation PSNR", "PSNR (dB)", baseline["psnr"], "max"),
        (ssim, "Validation SSIM", "SSIM", baseline["ssim"], "max"),
        (grad, "Validation Grad.", "Gradient preservation ratio", baseline["grad"], "one"),
    ]
    colors = [(50, 136, 189), (50, 136, 189), (44, 162, 95), (217, 95, 14)]
    panel_width = 530
    plot_top, plot_bottom = 98, 535
    for panel_index, ((values, title, ylabel, reference, best_rule), color) in enumerate(zip(panels, colors)):
        panel_left = 20 + panel_index * 545
        plot_left = panel_left + 78
        plot_right = panel_left + panel_width - 18
        box = draw.textbbox((0, 0), title, font=panel_font)
        draw.text((panel_left + (panel_width - (box[2] - box[0])) / 2, 52), title,
                  fill="black", font=panel_font)
        combined = values if reference is None else np.append(values, reference)
        value_min, value_max = float(np.min(combined)), float(np.max(combined))
        padding = max((value_max - value_min) * 0.10, 1e-5)
        value_min -= padding
        value_max += padding
        draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline=(80, 80, 80), width=1)
        for grid in range(1, 5):
            y = int(plot_top + grid * (plot_bottom - plot_top) / 5)
            draw.line((plot_left, y, plot_right, y), fill=(220, 220, 220), width=1)
        def map_point(x_value: float, y_value: float) -> tuple[int, int]:
            x_span = max(float(epoch[-1] - epoch[0]), 1.0)
            x = plot_left + int((x_value - float(epoch[0])) / x_span * (plot_right - plot_left))
            y = plot_bottom - int((y_value - value_min) / (value_max - value_min) * (plot_bottom - plot_top))
            return x, y
        points = [map_point(float(x), float(y)) for x, y in zip(epoch, values)]
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        for x, y in points:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
        if reference is not None:
            reference_y = map_point(float(epoch[0]), float(reference))[1]
            for x in range(plot_left, plot_right, 14):
                draw.line((x, reference_y, min(x + 8, plot_right), reference_y), fill=(242, 142, 0), width=2)
            if best_rule == "max":
                best_position = int(np.argmax(values))
            elif best_rule == "min":
                best_position = int(np.argmin(values))
            else:
                best_position = int(np.argmin(np.abs(values - 1.0)))
            best_x, best_y = points[best_position]
            draw.ellipse((best_x - 7, best_y - 7, best_x + 7, best_y + 7), fill=(214, 39, 40))
            legend = f"Bicubic {reference:.4f} | selected {values[best_position]:.4f}"
            draw.text((plot_left + 8, plot_top + 8), legend, fill=(40, 40, 40), font=axis_font)
        draw.text((plot_left - 5, plot_bottom + 8), f"Epoch {int(epoch[0])}", fill="black", font=axis_font)
        right_text = f"{int(epoch[-1])}"
        right_box = draw.textbbox((0, 0), right_text, font=axis_font)
        draw.text((plot_right - (right_box[2] - right_box[0]), plot_bottom + 8), right_text,
                  fill="black", font=axis_font)
        draw.text((panel_left + 2, plot_top), f"{value_max:.4f}", fill="black", font=axis_font)
        draw.text((panel_left + 2, plot_bottom - 18), f"{value_min:.4f}", fill="black", font=axis_font)
        draw.text((plot_left + 4, plot_bottom + 36), ylabel, fill=(70, 70, 70), font=axis_font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.png")
    canvas.save(temporary, format="PNG")
    os.replace(temporary, output_path)


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def derive_convergence_state(rows: list[dict[str, str]]) -> dict[str, Any]:
    best_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    for row in rows:
        epoch = int(row["epoch"])
        loss = float(row["val_l1_mae"])
        if not math.isfinite(loss):
            raise FloatingPointError(f"Non-finite epoch-average L1 in epoch {epoch}")
        if loss <= best_loss - EARLY_STOP_MIN_DELTA:
            best_loss = loss
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
    last_epoch = int(rows[-1]["epoch"]) if rows else 0
    should_stop = epochs_without_improvement >= EARLY_STOP_PATIENCE
    return {
        "metric": "epoch-average validation L1 / raw-CSI MAE",
        "mode": "min",
        "min_delta": EARLY_STOP_MIN_DELTA,
        "patience": EARLY_STOP_PATIENCE,
        "best_l1": best_loss,
        "best_epoch": best_epoch,
        "epochs_without_improvement": epochs_without_improvement,
        "last_completed_epoch": last_epoch,
        "should_stop": should_stop,
    }


def row_validation_metrics(row: dict[str, str]) -> dict[str, float]:
    return {
        "l1": float(row["val_l1_mae"]),
        "psnr": float(row["val_psnr"]),
        "ssim": float(row["val_ssim"]),
        "grad": float(row["val_grad"]),
        "output_std": float(row["output_std"]),
    }


class RegressionAccumulator:
    def __init__(self) -> None:
        self.n = 0
        self.sum_observation = 0.0
        self.sum_observation_sq = 0.0
        self.sum_error_sq = 0.0
        self.sum_abs_error = 0.0
        self.sum_error = 0.0

    def update(self, observation: np.ndarray, prediction: np.ndarray) -> None:
        observation = observation.astype(np.float64, copy=False)
        prediction = prediction.astype(np.float64, copy=False)
        error = prediction - observation
        self.n += observation.size
        self.sum_observation += float(observation.sum())
        self.sum_observation_sq += float(np.square(observation).sum())
        self.sum_error_sq += float(np.square(error).sum())
        self.sum_abs_error += float(np.abs(error).sum())
        self.sum_error += float(error.sum())

    def metrics(self) -> dict[str, float]:
        if self.n == 0:
            return {"r2": math.nan, "rmse": math.nan, "mae": math.nan, "bias": math.nan}
        denominator = self.sum_observation_sq - self.sum_observation**2 / self.n
        return {
            "r2": 1.0 - self.sum_error_sq / max(denominator, 1e-12),
            "rmse": math.sqrt(self.sum_error_sq / self.n),
            "mae": self.sum_abs_error / self.n,
            "bias": self.sum_error / self.n,
        }
