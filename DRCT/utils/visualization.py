"""Plots requested by the original experiment, with raw CSI kept unchanged.

Training charts use completed validation epochs. Final scatter plots and the
test montage are written only by the explicit frozen-checkpoint test entry.
Pillow/OpenCV are already used by the original project; no plotting service or
additional plotting framework is required.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {
    "training_curves": True,
    "refresh_updates": 100,
    "training_comparison": True,
    "sample_count": 10,
    "validation_indices": None,
    "test_indices": None,
    "csi_min": None,
    "csi_max": None,
    "scatter_csi": True,
    "scatter_ghi": True,
    "scatter_points_per_sample": 64,
    "scatter_max_points": 150000,
    "clear_ghi": None,
    "clear_ghi_csv": None,
}


def plot_options(config_file=None, *, force_visuals=False):
    supplied = {} if config_file is None else json.loads(Path(config_file).read_text(encoding="utf-8-sig")).get("visualization", {})
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULTS):
        raise ValueError("Unknown visualization configuration option")
    result = {**DEFAULTS, **supplied}
    for key in ("training_curves", "training_comparison", "scatter_csi", "scatter_ghi"):
        if not isinstance(result[key], bool):
            raise ValueError(f"visualization.{key} must be true or false")
    for key in ("refresh_updates", "sample_count", "scatter_points_per_sample", "scatter_max_points"):
        if isinstance(result[key], bool) or not isinstance(result[key], int) or result[key] < 1:
            raise ValueError(f"visualization.{key} must be a positive integer")
    for key in ("csi_min", "csi_max"):
        if result[key] is not None and not math.isfinite(float(result[key])):
            raise ValueError(f"visualization.{key} must be finite")
    if (result["clear_ghi"] is None) != (result["clear_ghi_csv"] is None):
        raise ValueError("Provide both clear_ghi and the exact aligned clear_ghi_csv")
    for key in ("clear_ghi", "clear_ghi_csv"):
        if result[key] is not None:
            path = Path(result[key]).expanduser()
            result[key] = str((path if path.is_absolute() else ROOT / path).resolve())
    if force_visuals:
        result["training_curves"] = result["training_comparison"] = True
    return result


def font(size, bold=False):
    names = ["C:/Windows/Fonts/timesbd.ttf" if bold else "C:/Windows/Fonts/times.ttf",
             "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf" if bold else "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
             "/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
             "DejaVuSerif-Bold.ttf" if bold else "DejaVuSerif.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def atomic_image(image, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.png")
    image.save(temporary, format="PNG", dpi=(300, 300))
    os.replace(temporary, path)


def centered(draw, box, text, selected_font, fill="black"):
    bounds = draw.textbbox((0, 0), text, font=selected_font)
    x = box[0] + (box[2] - box[0] - bounds[2] + bounds[0]) / 2
    y = box[1] + (box[3] - box[1] - bounds[3] + bounds[1]) / 2 - bounds[1]
    draw.text((x, y), text, font=selected_font, fill=fill)


def training_progress(history, baseline, destination, *, status=None, live_points=None):
    """Four original panels; running L1 is clearly separated from epoch means.

    Validation points exist only after validation finishes. No last-epoch metric
    is repeated as if it had been measured in the current epoch.
    """
    status = status or {"phase": "epoch committed", "epoch": int(history[-1]["epoch"]) if history else 0}
    live_points = live_points or []
    canvas = Image.new("RGB", (2200, 650), "white")
    draw = ImageDraw.Draw(canvas)
    from utils.training import ui_font
    title_font, small_font = ui_font(27, True), ui_font(17)
    centered(draw, (0, 0, 2200, 38), "DRCT-L W8 x4 From-Scratch Training Progress", title_font)
    message = f"Epoch {status.get('epoch', 0)} | {status.get('phase', '')}"
    if "update" in status:
        message += f" | update {status['update']}/{status['steps']} | global running L1 {status['running_l1']:.6f}"
    centered(draw, (0, 38, 2200, 66), message, small_font)
    panels = [("val_l1_mae", "Validation L1 / MAE", "Raw CSI MAE", None, "min"),
              ("val_psnr", "Validation PSNR", "PSNR (dB)", baseline["psnr"], "max"),
              ("val_ssim", "Validation SSIM", "SSIM", baseline["ssim"], "max"),
              ("val_grad", "Validation Grad.", "Gradient preservation ratio", baseline["grad"], "one")]
    current_epoch = max(1, int(status.get("epoch", 1)), int(history[-1]["epoch"]) if history else 1)
    x_start, x_end = 0 if not history else max(0, int(history[0]["epoch"]) - 1), current_epoch
    for panel, (key, title, label, reference, best_rule) in enumerate(panels):
        left, right, top, bottom = 60 + 550 * panel, 520 + 550 * panel, 125, 545
        values = [(float(row["epoch"]), float(row[key])) for row in history if math.isfinite(float(row[key]))]
        running = []  # Epoch-only validation curves
        all_y = [y for _, y in values] + [y for _, y in running]
        if reference is not None and math.isfinite(reference):
            all_y.append(reference)
        lower, upper = (min(all_y), max(all_y)) if all_y else (0, 1)
        padding = max((upper - lower) * 0.1, abs(upper) * 0.005, 1e-5)
        lower, upper = lower - padding, upper + padding
        def point(x, y):
            return (left + (x - x_start) / max(x_end - x_start, 1) * (right - left),
                    bottom - (y - lower) / (upper - lower) * (bottom - top))
        centered(draw, (left - 30, 76, right + 10, 112), title, ui_font(21, True))
        draw.rectangle((left, top, right, bottom), outline=(100, 100, 100))
        for fraction in (0.2, 0.4, 0.6, 0.8):
            y = top + fraction * (bottom - top)
            draw.line((left, y, right, y), fill=(220, 220, 220))
        if reference is not None and math.isfinite(reference):
            y = point(0, reference)[1]
            for x in range(left, right, 15):
                draw.line((x, y, min(x + 9, right), y), fill=(240, 140, 0), width=2)
        for series, colour, width in ((running, (140, 185, 220), 2), (values, ((50, 136, 189) if panel < 2 else (44, 162, 95) if panel == 2 else (217, 95, 14)), 3)):
            points = [point(x, y) for x, y in series]
            if len(points) > 1:
                draw.line(points, fill=colour, width=width)
            for x, y in points:
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=colour)
        if values and panel:
            selected = max(values, key=lambda pair: pair[1]) if best_rule == "max" else min(values, key=lambda pair: abs(pair[1] - 1))
            x, y = point(*selected)
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=(214, 39, 40))
            legend = f"Bicubic {reference:.4f} | selected {selected[1]:.4f}"
        elif panel == 0:
            legend = "Completed validation epochs; lower is better"
        else:
            legend = "Awaiting completed validation" if not history else "Nonfinite metric: see training_log.csv"
        draw.text((left + 5, top + 8), legend, fill="black", font=ui_font(15))
        draw.text((left - 42, top - 20), f"{upper:.4g}", fill="black", font=small_font)
        draw.text((left - 42, bottom - 18), f"{lower:.4g}", fill="black", font=small_font)
        draw.text((left, bottom + 10), f"Epoch {x_start}", fill="black", font=small_font)
        draw.text((right - 35, bottom + 10), str(x_end), fill="black", font=small_font)
        draw.text((left + 3, bottom + 43), label, fill=(70, 70, 70), font=small_font)
    atomic_image(canvas, destination)


def sample_indices(dataset, split, settings):
    selected = settings.get("validation_indices" if split == "validation" else "test_indices")
    if selected is None:
        return list(range(min(settings["sample_count"], len(dataset))))
    if not isinstance(selected, list) or not selected or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(dataset) for i in selected):
        raise ValueError(f"Invalid fixed {split} visualization CSV row indices")
    if len(set(selected)) != len(selected):
        raise ValueError("Visualization indices must be unique")
    return selected


def csi_limits(reference, settings):
    minimum = settings.get("csi_min")
    maximum = settings.get("csi_max")
    if minimum is None:
        minimum = reference.get("raw_min")
        minimum = 0.0 if minimum is None else minimum
    if maximum is None:
        maximum = reference.get("raw_max")
        maximum = minimum + reference["data_range"] if maximum is None else maximum
    if not math.isfinite(float(minimum)) or not math.isfinite(float(maximum)) or maximum <= minimum:
        raise ValueError("CSI plot limits must be finite and increasing")
    return float(minimum), float(maximum)


def comparison_grid(samples, reference, settings, destination, *, heading=None):
    """FY-4A / FY-3C / DRCT / Bicubic rows; fixed common display scale only."""
    if not samples:
        raise ValueError("No samples for the comparison figure")
    minimum, maximum = csi_limits(reference, settings)
    tile, left, top, stripe = 180, 50, 52, 40
    count = len(samples)
    header = 36 if heading else 0
    top += header
    image = Image.new("RGB", (left + count * tile + stripe + 35, top + 4 * tile + 115), "white")
    draw = ImageDraw.Draw(image)
    if heading:
        centered(draw, (0, 0, image.width, 35), heading, font(23, True))
    outside = 0
    for column, sample in enumerate(samples):
        x = left + column * tile
        draw.rectangle((x, top - 34, x + tile, top), fill="black")
        centered(draw, (x, top - 34, x + tile, top), f"Sample {column + 1}", font(25), "white")
        for row, key in enumerate(("low", "target", "prediction", "bicubic")):
            values = np.asarray(sample[key], dtype=np.float32)
            outside += int(((values < minimum) | (values > maximum)).sum())
            # Only the displayed colour pixels saturate. Raw arrays/metrics never do.
            display = np.rint(np.clip((values - minimum) / (maximum - minimum), 0, 1) * 255).astype(np.uint8)
            coloured = cv2.cvtColor(cv2.applyColorMap(display, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
            panel = Image.fromarray(coloured).resize((tile, tile), Image.Resampling.NEAREST)
            y = top + row * tile
            image.paste(panel, (x, y))
            draw.rectangle((x, y, x + tile, y + tile), outline="black", width=1)
    for row, label in enumerate(("FY-4A", "FY-3C", "DRCT", "Bicubic")):
        y = top + row * tile
        rotated = Image.new("RGB", (tile, stripe), "black")
        centered(ImageDraw.Draw(rotated), (0, 0, tile, stripe), label, font(25), "white")
        image.paste(rotated.rotate(-90, expand=True), (left + count * tile, y))
    bar_width = min(540, count * tile // 2)
    bx = left + (count * tile - bar_width) // 2
    by = top + 4 * tile + 34
    gradient = np.tile(np.arange(256, dtype=np.uint8), (22, 1))
    coloured = cv2.cvtColor(cv2.applyColorMap(gradient, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
    image.paste(Image.fromarray(coloured).resize((bar_width, 22)), (bx, by))
    draw.rectangle((bx, by, bx + bar_width, by + 22), outline="black")
    for step in range(6):
        x = bx + step * bar_width / 5
        draw.line((x, by + 22, x, by + 30), fill="black")
        centered(draw, (x - 45, by + 30, x + 45, by + 58), f"{minimum + step * (maximum - minimum) / 5:.2g}", font(22))
    centered(draw, (0, by - 8, max(40, bx - 12), by + 33), "Clear-sky index (CSI)", font(24))
    atomic_image(image, destination)
    return {"rows": ["FY-4A", "FY-3C", "DRCT", "Bicubic"], "columns": count,
            "csi_min": minimum, "csi_max": maximum, "outside_display_range_values": outside,
            "display_only_limits": True, "raw_values_clipped": False}


class ScatterSample:
    """Bounded paired reservoir; one common set of pixels for both methods."""
    def __init__(self, settings):
        self.rng = np.random.default_rng(60)
        self.per_sample = settings["scatter_points_per_sample"]
        self.capacity = settings["scatter_max_points"]
        self.values = np.empty((0, 3), dtype=np.float32)
        self.priority = np.empty(0)
        self.minimum, self.maximum = math.inf, -math.inf
        self.total_pixels = 0

    def add(self, observation, bicubic, drct):
        observation, bicubic, drct = (np.asarray(v).ravel() for v in (observation, bicubic, drct))
        valid = np.isfinite(observation) & np.isfinite(bicubic) & np.isfinite(drct)
        values = np.column_stack((observation[valid], bicubic[valid], drct[valid]))
        self.total_pixels += len(values)
        if not len(values):
            return
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))
        chosen = self.rng.choice(len(values), size=min(self.per_sample, len(values)), replace=False)
        self.values = np.concatenate((self.values, values[chosen].astype(np.float32)), axis=0)
        self.priority = np.concatenate((self.priority, self.rng.random(len(chosen))))
        if len(self.values) > 2 * self.capacity:
            self.trim()

    def trim(self):
        if len(self.values) > self.capacity:
            choice = np.argpartition(self.priority, self.capacity - 1)[:self.capacity]
            self.values, self.priority = self.values[choice], self.priority[choice]

    def summary(self):
        self.trim()
        return {"full_valid_pixels": self.total_pixels, "plot_pixels": len(self.values),
                "raw_min": self.minimum, "raw_max": self.maximum, "seed": 60,
                "points_per_sample": self.per_sample, "max_plot_pixels": self.capacity,
                "sampled_for_drawing_only": True, "metrics_use_all_test_pixels": True}


def scatter_plot(sample, metrics, destination, *, units="CSI"):
    """Two vertically stacked panels, common limits, orange/blue and y=x."""
    info = sample.summary()
    if not len(sample.values):
        raise ValueError("No finite paired values for the scatter figure")
    span = max(sample.maximum - sample.minimum, 1e-6)
    lower, upper = sample.minimum - 0.04 * span, sample.maximum + 0.04 * span
    left, tile, margin = 160, 620, 150
    image = Image.new("RGB", (840, 2 * (tile + margin) + 30), "white")
    draw = ImageDraw.Draw(image)
    for panel, (method, colour) in enumerate((("Bicubic", (240, 145, 32)), ("DRCT", (31, 119, 180)))):
        top = 60 + panel * (tile + margin)
        x, y = sample.values[:, 0], sample.values[:, panel + 1]
        px = np.minimum(tile - 1, np.maximum(0, np.rint((x - lower) / (upper - lower) * (tile - 1)).astype(np.int32)))
        py = np.minimum(tile - 1, np.maximum(0, np.rint((upper - y) / (upper - lower) * (tile - 1)).astype(np.int32)))
        density = np.zeros((tile, tile), dtype=np.uint32)
        np.add.at(density, (py, px), 1)
        alpha = 1 - np.exp(-density.astype(np.float32) * 0.18)
        rgb = 255 * (1 - alpha[:, :, None]) + np.array(colour)[None, None] * alpha[:, :, None]
        image.paste(Image.fromarray(np.rint(rgb).astype(np.uint8)), (left, top))
        for fraction in np.linspace(0, 1, 6):
            x_pixel, y_pixel = left + fraction * tile, top + (1 - fraction) * tile
            value = lower + fraction * (upper - lower)
            draw.line((x_pixel, top, x_pixel, top + tile), fill=(180, 180, 180))
            draw.line((left, y_pixel, left + tile, y_pixel), fill=(180, 180, 180))
            label = f"{value:.2g}" if units == "CSI" else f"{value:.0f}"
            centered(draw, (x_pixel - 48, top + tile + 8, x_pixel + 48, top + tile + 38), label, font(24))
            centered(draw, (left - 90, y_pixel - 18, left - 12, y_pixel + 18), label, font(24))
        draw.rectangle((left, top, left + tile, top + tile), outline="black", width=2)
        for offset in range(0, tile, 24):
            end = min(offset + 14, tile)
            draw.line((left + offset, top + tile - offset, left + end, top + tile - end), fill="black", width=2)
        draw.rectangle((left - 2, top - 32, left + tile + 2, top), fill="black")
        centered(draw, (left, top - 32, left + tile, top), f"{method}: 4 km to 1 km", font(25), "white")
        draw.text((left - 125, top - 35), f"({'ab'[panel]})", font=font(25), fill="black")
        x_label = "FY-3C CSI" if units == "CSI" else "FY-3C Observation [W/m²]"
        y_label = f"{method} CSI" if units == "CSI" else f"{method} Result [W/m²]"
        centered(draw, (left, top + tile + 46, left + tile, top + tile + 86), x_label, font(26))
        vertical = Image.new("RGB", (tile, 44), "white")
        centered(ImageDraw.Draw(vertical), (0, 0, tile, 44), y_label, font(26))
        image.paste(vertical.rotate(90, expand=True), (12, top))
        numbers = metrics["bicubic" if panel == 0 else "drct"]
        text = f"R²={numbers['r2']:.3f}  RMSE={numbers['rmse']:.3g}\nMAE={numbers['mae']:.3g}  MBE={numbers['bias']:.3g}"
        draw.rectangle((left + 10, top + 10, left + 440, top + 78), fill="white")
        draw.multiline_text((left + 18, top + 15), text, font=font(22), fill="black", spacing=4)
    atomic_image(image, destination)
    return {**info, "units": units, "axis_min": lower, "axis_max": upper}


class ClearGHI:
    """An independently supplied test array, bound to the exact test CSV."""
    def __init__(self, dataset, settings):
        self.array = None
        if settings["clear_ghi"] is None:
            return
        source_csv = dataset.root / "splits/test_samples.csv"
        matched_csv = Path(settings["clear_ghi_csv"])
        if hashlib.sha256(source_csv.read_bytes()).digest() != hashlib.sha256(matched_csv.read_bytes()).digest():
            raise ValueError("clear_ghi_csv must be an exact copy of the test CSV used to order clear_ghi.npy")
        if getattr(dataset, "storage_format", "") != "2018_2019_npz":
            raise ValueError("GHI evaluation requires the source NPZ dataset with actual fy3 observations")
        self.array = np.load(settings["clear_ghi"], mmap_mode="r", allow_pickle=False)
        if self.array.shape not in ((len(dataset), 128, 128), (len(dataset), 1, 128, 128)) or self.array.dtype != np.float32:
            raise ValueError("clear_ghi must be float32 N×128×128 or N×1×128×128, exactly ordered by the test CSV")

    @property
    def available(self):
        return self.array is not None

    def get(self, index):
        result = np.asarray(self.array[index], dtype=np.float32).reshape(128, 128)
        if not np.isfinite(result).all() or (result < 0).any():
            raise ValueError(f"Invalid supplied clear GHI at test CSV row {index}")
        return result


def training_comparison(model, dataset, device, settings, reference, destination, epoch):
    """Fixed validation rows only; never consult test samples during training."""
    import torch
    from utils.training import bicubic_batch
    indices = sample_indices(dataset, "validation", settings)
    samples = []
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for index in indices:
                low, target = dataset[index]
                predicted = model(low[None].to(device)).float().cpu()[0, 0].numpy()
                samples.append({"low": low[0].numpy(), "target": target[0].numpy(),
                                "prediction": predicted, "bicubic": bicubic_batch(low[None])[0, 0].numpy()})
        metadata = comparison_grid(samples, reference, settings, destination, heading=f"Validation samples | Epoch {epoch}")
        metadata.update({"split": "validation", "csv_row_indices": indices, "selection": "fixed CSV rows, not performance-selected",
                         "sample_ids": [dataset.records[i]["sample_id"] for i in indices]})
        from utils.training import atomic_json
        atomic_json(Path(destination).with_suffix(".json"), metadata)
    finally:
        model.train(was_training)
