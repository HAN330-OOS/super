"""风云4A/FY-3C 匹配 CSI 样本的 DRCT-L ×4 训练主程序。

该文件是正式实验、断点续训和冻结模型测试的唯一入口。正式训练从头开始，
仅在显式续训或测试时读取本实验目录中的 checkpoint。
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as ssim_metric
from torch.utils.data import DataLoader, Dataset, Subset

from data.dataset_SR_satellite import CSI_SCALE, split_by_dates
from models.network_drct import DRCT
from utils.utils_paths import FY_SAMPLES_ROOT


PROJECT_ROOT = Path(__file__).resolve().parent
OPTION_PATH = PROJECT_ROOT / "exp" / "train_x4.json"
VISUAL_SPEC_PATH = PROJECT_ROOT / "exp" / "fixed_visual_samples.json"
with OPTION_PATH.open(encoding="utf-8") as _options_handle:
    DATASET_OPTIONS = dict(json.load(_options_handle)["datasets"])

# --------------------------------------------------------
# 步骤 0（数据集划分与实验路径）
# --------------------------------------------------------
# 原始 FY-4A/FY-3C 批量 NPZ 样本始终保存在 SAMPLES_ROOT。下面三份清单
# 只记录训练、验证和测试成员，不重复复制卫星样本数据。
SAMPLES_ROOT = FY_SAMPLES_ROOT
DATASET_ROOT = (PROJECT_ROOT / DATASET_OPTIONS["dataset_root"]).resolve()
TRAIN_SPLIT_PATH = (PROJECT_ROOT / DATASET_OPTIONS["train_manifest"]).resolve()
VALIDATION_SPLIT_PATH = (PROJECT_ROOT / DATASET_OPTIONS["validation_manifest"]).resolve()
TEST_SPLIT_PATH = (PROJECT_ROOT / DATASET_OPTIONS["test_manifest"]).resolve()

EXPERIMENT_NAME = "fy4a_fy3c_drctl_w8_scratch_50k_b16_e50"
OUTPUT_DIR = PROJECT_ROOT / "results" / EXPERIMENT_NAME
HISTORICAL_MONITOR_PATH = OUTPUT_DIR / "training_monitor_subset_1000.json"

VAL_DATES = tuple(f"201806{day:02d}" for day in range(20, 31))
TRAIN_SAMPLES = 50_000
HISTORICAL_MONITOR_SAMPLES = 1_000
VALIDATION_SAMPLES = 10_000
FINAL_TEST_SAMPLES = 10_000
EFFECTIVE_BATCH = 16
# 硬上限 epoch 数；可用环境变量 DRCT_EPOCHS 覆盖（例如续跑到收敛时提高上限），
# 默认保持 100，不改变收敛/早停规则。
EPOCHS = int(os.environ.get("DRCT_EPOCHS", "100"))
STEPS_PER_EPOCH = TRAIN_SAMPLES // EFFECTIVE_BATCH
TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH
SEED = 60
LEARNING_RATE = 2e-4
CONTINUATION_START_EPOCH = 51
EARLY_STOP_MIN_DELTA = 1e-4
EARLY_STOP_PATIENCE = 5
EARLY_STOP_MIN_EPOCH = 56
BETAS = (0.9, 0.99)
EMA_DECAY = 0.999
SMOKE_STEPS = 30
VALIDATION_BATCH = 8
TRAIN_MICRO_BATCH_CAP = 8
NUM_WORKERS = 2
CACHE_WORKERS = int(os.environ.get("DRCT_TRAIN_WORKERS", "2"))
PIN_MEMORY = False
PREFETCH_FACTOR = 1
CACHE_VERSION = 1
CACHE_DIRNAME = "training_cache_float32"


# --------------------------------------------------------
# 步骤 1（公共绘图函数与实验配置）
# --------------------------------------------------------
def ui_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = ("arialbd.ttf", "arial.ttf") if bold else ("arial.ttf", "arialbd.ttf")
    for name in names:
        path = Path(r"C:\Windows\Fonts") / name
        try:
            return ImageFont.truetype(str(path), size)
        except OSError:
            continue
    return ImageFont.load_default()


def colorize_csi(array: np.ndarray) -> Image.Image:
    normalized = np.clip(np.nan_to_num(array, nan=0.5 * CSI_SCALE), 0.0, CSI_SCALE)
    uint8 = np.rint(normalized / CSI_SCALE * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(uint8, cv2.COLORMAP_VIRIDIS)
    return Image.fromarray(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))


@dataclass(frozen=True)
class ExperimentConfig:
    experiment: str = EXPERIMENT_NAME
    source: str = "matched FY-4A LR CSI -> FY-3C HR CSI"
    initialization: str = "from scratch"
    train_samples: int = TRAIN_SAMPLES
    historical_monitor_samples: int = HISTORICAL_MONITOR_SAMPLES
    validation_samples: int = VALIDATION_SAMPLES
    final_test_samples: int = FINAL_TEST_SAMPLES
    effective_batch: int = EFFECTIVE_BATCH
    epochs: int = EPOCHS
    steps_per_epoch: int = STEPS_PER_EPOCH
    total_steps: int = TOTAL_STEPS
    loss: str = "L1 / normalized-CSI MAE"
    optimizer: str = "Adam"
    learning_rate: float = LEARNING_RATE
    original_completed_epochs: int = 50
    original_learning_rate: float = 2e-4
    continuation_start_epoch: int = CONTINUATION_START_EPOCH
    stopping_target: str = "epoch-average training L1 convergence"
    hard_cap_epoch: int = EPOCHS
    early_stop_min_delta: float = EARLY_STOP_MIN_DELTA
    early_stop_patience: int = EARLY_STOP_PATIENCE
    early_stop_min_epoch: int = EARLY_STOP_MIN_EPOCH
    continuation_final_checkpoint: str = "checkpoints/epoch_NNN_{converged|hard_cap}_final.pth"
    preserved_best_checkpoint: str = "checkpoints/best_real.pth"
    beta1: float = BETAS[0]
    beta2: float = BETAS[1]
    weight_decay: float = 0.0
    scheduler: str = "none; fixed 2e-4 until convergence or epoch-100 hard cap"
    ema_decay: float = EMA_DECAY
    seed: int = SEED
    input_shape: str = "1x32x32"
    output_shape: str = "1x128x128"
    img_size: int = 32
    window_size: int = 8
    shift_size: int = 4
    augmentation: bool = False
    amp: str = "float16"
    training_cache: str = "read-only float32 NPY memmap; bitwise-verified against source NPZ"
    dataloader_workers: int = CACHE_WORKERS
    validation_workers: int = NUM_WORKERS
    persistent_workers: bool = False
    prefetch_factor: int = PREFETCH_FACTOR
    pin_memory: bool = PIN_MEMORY
    optimizer_implementation: str = "CUDA fused Adam"
    ema_implementation: str = "CUDA foreach"
    validation: str = "10000 fixed matched pairs; epoch-271 historically monitored 1000 of them"
    final_evaluation: str = "10000 disjoint held-out pairs after training is frozen"
    visuals: str = "same five validation pairs after every epoch; final paper figures use the held-out test subset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--smoke-test", action="store_true")
    modes.add_argument("--train", action="store_true")
    modes.add_argument("--resume", action="store_true")
    modes.add_argument("--validate", action="store_true")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Weights to validate; defaults to the completed run's final checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Independent experiment directory.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def default_validation_checkpoint(output_dir: Path) -> Path:
    """优先选择已完成训练的最终模型；若不存在则回退到当前最优模型。"""
    completion_path = output_dir / "training_completion.json"
    if completion_path.exists():
        with completion_path.open(encoding="utf-8") as handle:
            completion = json.load(handle)
        final_checkpoint = output_dir / completion.get("final_checkpoint", "")
        if final_checkpoint.is_file():
            return final_checkpoint
    return output_dir / "checkpoints" / "best_real.pth"


# --------------------------------------------------------
# 步骤 2（固定训练集、验证集与测试集清单）
# --------------------------------------------------------
def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def index_record(dataset: Any, dataset_index: int) -> dict[str, Any]:
    batch_path, batch_index = dataset.index[int(dataset_index)]
    return {
        "index": int(dataset_index),
        "date": batch_path.parents[1].name.removesuffix("_utc"),
        "batch_file": batch_path.name,
        "batch_index": int(batch_index),
    }


def load_or_create_train_subset(dataset: Any, path: Path) -> list[int]:
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        indices = [int(item["index"]) for item in payload["samples"]]
        if len(indices) != TRAIN_SAMPLES:
            raise RuntimeError(f"Train subset has {len(indices)} samples, expected {TRAIN_SAMPLES}")
        if payload.get("indices_sha256") != sha256_json(indices):
            raise RuntimeError("Train subset hash mismatch")
        return indices

    if len(dataset) < TRAIN_SAMPLES:
        raise RuntimeError(f"Only {len(dataset)} training pairs are available")
    rng = np.random.default_rng(SEED)
    indices = rng.choice(len(dataset), size=TRAIN_SAMPLES, replace=False).astype(int).tolist()
    payload = {
        "purpose": "fixed training subset",
        "seed": SEED,
        "source_count": len(dataset),
        "sample_count": len(indices),
        "indices_sha256": sha256_json(indices),
        "samples": [index_record(dataset, index) for index in indices],
    }
    atomic_json(path, payload)
    return indices


class Float32TrainingCache(Dataset):
    """从只读 NPY 内存映射中读取固定训练集。

    缓存保存的内容与原始数据集返回的归一化 float32 张量完全一致。读取时将
    单个样本复制为可写数组，再转换为张量，避免 Windows 为完整缓存预留内存。
    """

    def __init__(self, cache_dir: Path, sample_count: int) -> None:
        self.lr_path = Path(cache_dir) / "lr_float32.npy"
        self.hr_path = Path(cache_dir) / "hr_float32.npy"
        self.sample_count = int(sample_count)
        self._owner_pid: int | None = None
        self._lr: np.ndarray | None = None
        self._hr: np.ndarray | None = None

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_owner_pid"] = None
        state["_lr"] = None
        state["_hr"] = None
        return state

    def _ensure_open(self) -> None:
        process_id = os.getpid()
        if self._owner_pid == process_id and self._lr is not None and self._hr is not None:
            return
        self._lr = np.load(self.lr_path, mmap_mode="r")
        self._hr = np.load(self.hr_path, mmap_mode="r")
        self._owner_pid = process_id

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, position: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_open()
        if self._lr is None or self._hr is None:
            raise RuntimeError("Training cache did not open")
        return (
            torch.from_numpy(self._lr[position].copy()),
            torch.from_numpy(self._hr[position].copy()),
        )


# --------------------------------------------------------
# 步骤 3（模型、数据加载器与验证指标）
# --------------------------------------------------------
def cache_manifest_is_valid(
    cache_dir: Path,
    fixed_indices: list[int],
) -> bool:
    manifest_path = cache_dir / "manifest.json"
    lr_path = cache_dir / "lr_float32.npy"
    hr_path = cache_dir / "hr_float32.npy"
    if not manifest_path.exists() or not lr_path.exists() or not hr_path.exists():
        return False
    try:
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("cache_version") != CACHE_VERSION:
            return False
        if manifest.get("indices_sha256") != sha256_json(fixed_indices):
            return False
        if int(manifest.get("sample_count", -1)) != len(fixed_indices):
            return False
        lr = np.load(lr_path, mmap_mode="r")
        hr = np.load(hr_path, mmap_mode="r")
        valid = (
            lr.shape == (len(fixed_indices), 1, 32, 32)
            and hr.shape == (len(fixed_indices), 1, 128, 128)
            and lr.dtype == np.float32
            and hr.dtype == np.float32
        )
        del lr, hr
        return valid
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def build_training_cache(
    source_dataset: Any,
    fixed_indices: list[int],
    cache_dir: Path,
) -> None:
    """仅一次性生成 LR/HR CSI 缓存，不修改原始 NPZ 文件。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    lr_path = cache_dir / "lr_float32.npy"
    hr_path = cache_dir / "hr_float32.npy"
    lr_temporary = cache_dir / "lr_float32.tmp.npy"
    hr_temporary = cache_dir / "hr_float32.tmp.npy"
    for temporary in (lr_temporary, hr_temporary):
        if temporary.exists():
            temporary.unlink()

    sample_count = len(fixed_indices)
    lr_cache = np.lib.format.open_memmap(
        lr_temporary,
        mode="w+",
        dtype=np.float32,
        shape=(sample_count, 1, 32, 32),
    )
    hr_cache = np.lib.format.open_memmap(
        hr_temporary,
        mode="w+",
        dtype=np.float32,
        shape=(sample_count, 1, 128, 128),
    )

    grouped: dict[Path, list[tuple[int, int]]] = defaultdict(list)
    for cache_position, dataset_index in enumerate(fixed_indices):
        batch_path, batch_index = source_dataset.index[int(dataset_index)]
        grouped[batch_path].append((cache_position, int(batch_index)))

    print(
        f"Building lossless float32 training cache: {sample_count:,} samples "
        f"from {len(grouped):,} NPZ batches ...",
        flush=True,
    )
    started = time.time()
    for file_number, (batch_path, specifications) in enumerate(grouped.items(), 1):
        cache_positions = np.fromiter(
            (item[0] for item in specifications), dtype=np.int64
        )
        batch_positions = np.fromiter(
            (item[1] for item in specifications), dtype=np.int64
        )
        # NPZ 中每个成员单独压缩；只读取 LR/HR 两个数组，可避免解压重复的
        # FY-3C/FY-4A 数组和元数据。
        with np.load(batch_path) as batch:
            lr = np.clip(
                batch["lr_csi"][batch_positions].astype(np.float32, copy=False),
                0.0,
                CSI_SCALE,
            ) / CSI_SCALE
            hr = np.clip(
                batch["hr_csi"][batch_positions].astype(np.float32, copy=False),
                0.0,
                CSI_SCALE,
            ) / CSI_SCALE
        if not np.isfinite(lr).all() or not np.isfinite(hr).all():
            raise FloatingPointError(f"NaN/Inf found while caching {batch_path}")
        lr_cache[cache_positions, 0] = lr
        hr_cache[cache_positions, 0] = hr
        if file_number % 250 == 0 or file_number == len(grouped):
            print(
                f"  cache {file_number:,}/{len(grouped):,} NPZ batches",
                flush=True,
            )

    lr_cache.flush()
    hr_cache.flush()
    del lr_cache, hr_cache
    os.replace(lr_temporary, lr_path)
    os.replace(hr_temporary, hr_path)
    atomic_json(
        cache_dir / "manifest.json",
        {
            "cache_version": CACHE_VERSION,
            "source": "normalized lr_csi/hr_csi from original matched NPZ files",
            "dtype": "float32",
            "sample_count": sample_count,
            "indices_sha256": sha256_json(fixed_indices),
            "lr_shape": [sample_count, 1, 32, 32],
            "hr_shape": [sample_count, 1, 128, 128],
            "finite_checked": True,
            "original_data_modified": False,
            "build_seconds": time.time() - started,
        },
    )
    print(f"Training cache built in {(time.time() - started) / 60:.1f} min", flush=True)


def verify_training_cache(
    source_dataset: Any,
    fixed_indices: list[int],
    cache_dataset: Float32TrainingCache,
) -> None:
    """训练前在固定位置检查缓存与原始样本是否逐位一致。"""
    positions = np.linspace(0, len(fixed_indices) - 1, 64, dtype=np.int64)
    for position in positions:
        source_lr, source_hr = source_dataset[fixed_indices[int(position)]]
        cached_lr, cached_hr = cache_dataset[int(position)]
        if not torch.equal(source_lr, cached_lr) or not torch.equal(source_hr, cached_hr):
            raise RuntimeError(
                f"Training cache differs from source NPZ at cache position {position}"
            )
    print("Training cache verification: 64/64 samples bitwise identical", flush=True)


def prepare_training_cache(
    source_dataset: Any,
    fixed_indices: list[int],
    output_dir: Path,
) -> Float32TrainingCache:
    cache_dir = output_dir / CACHE_DIRNAME
    if not cache_manifest_is_valid(cache_dir, fixed_indices):
        build_training_cache(source_dataset, fixed_indices, cache_dir)
    else:
        print(f"Reusing verified training cache: {cache_dir.resolve()}", flush=True)
    cache_dataset = Float32TrainingCache(cache_dir, len(fixed_indices))
    verify_training_cache(source_dataset, fixed_indices, cache_dataset)
    return cache_dataset


def proportional_quotas(groups: dict[str, list[int]], sample_count: int) -> dict[str, int]:
    total = sum(len(values) for values in groups.values())
    if sample_count > total:
        raise ValueError(f"Requested {sample_count} validation samples from only {total}")
    exact = {date: sample_count * len(values) / total for date, values in groups.items()}
    quotas = {date: int(math.floor(value)) for date, value in exact.items()}
    remainder = sample_count - sum(quotas.values())
    ranked = sorted(groups, key=lambda date: (exact[date] - quotas[date], date), reverse=True)
    for date in ranked[:remainder]:
        quotas[date] += 1
    return quotas


def load_or_create_stratified_subset(
    dataset: Any,
    path: Path,
    sample_count: int,
    purpose: str,
) -> list[int]:
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        indices = [int(item["index"]) for item in payload["samples"]]
        if len(indices) != sample_count:
            raise RuntimeError(f"{purpose} has {len(indices)} samples, expected {sample_count}")
        if payload.get("indices_sha256") != sha256_json(indices):
            raise RuntimeError(f"{purpose} hash mismatch")
        return indices

    groups: dict[str, list[int]] = defaultdict(list)
    for dataset_index in range(len(dataset)):
        groups[date_for_index(dataset, dataset_index)].append(dataset_index)
    groups = {date: values for date, values in groups.items() if values}
    quotas = proportional_quotas(groups, sample_count)
    rng = np.random.default_rng(SEED)
    selected: list[int] = []
    for date in sorted(groups):
        chosen = rng.choice(groups[date], size=quotas[date], replace=False)
        selected.extend(int(index) for index in chosen)
    # 排序可提高 NPZ 缓存局部性，不影响整体验证指标。
    selected.sort()
    payload = {
        "purpose": purpose,
        "seed": SEED,
        "source_count": len(dataset),
        "sample_count": len(selected),
        "sampling": "proportional stratification by validation date",
        "date_source_counts": {date: len(groups[date]) for date in sorted(groups)},
        "date_sample_counts": {date: quotas[date] for date in sorted(quotas)},
        "indices_sha256": sha256_json(selected),
        "samples": [index_record(dataset, index) for index in selected],
    }
    atomic_json(path, payload)
    return selected


def load_manifest_indices(path: Path, sample_count: int, purpose: str) -> list[int]:
    """读取固定样本清单，并校验数量与索引哈希。"""
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    indices = [int(item["index"]) for item in payload["samples"]]
    if len(indices) != sample_count:
        raise RuntimeError(f"{purpose}包含 {len(indices)} 对样本，应为 {sample_count} 对")
    if payload.get("indices_sha256") != sha256_json(indices):
        raise RuntimeError(f"{purpose}索引哈希不一致")
    return indices


def load_or_create_validation_subset(dataset: Any) -> list[int]:
    """构建正式 1 万对验证集，并与最终测试集严格分离。

    已完成的 epoch-271 训练每轮只监控了其中保留的 1,000 对样本。这里
    保留那 1,000 对，并从未进入测试集的同日期池中按日期比例补充 9,000
    对，形成后续训练和论文报告统一使用的正式验证集。
    """
    test_indices = load_manifest_indices(
        TEST_SPLIT_PATH, FINAL_TEST_SAMPLES, "最终测试集"
    )
    test_set = set(test_indices)
    if VALIDATION_SPLIT_PATH.exists():
        selected = load_manifest_indices(
            VALIDATION_SPLIT_PATH, VALIDATION_SAMPLES, "正式验证集"
        )
        if set(selected) & test_set:
            raise RuntimeError("正式验证集与最终测试集存在重叠")
        return selected

    historical_indices = load_manifest_indices(
        HISTORICAL_MONITOR_PATH,
        HISTORICAL_MONITOR_SAMPLES,
        "历史训练监控集",
    )
    historical_set = set(historical_indices)
    if historical_set & test_set:
        raise RuntimeError("历史训练监控集与最终测试集存在重叠")

    groups: dict[str, list[int]] = defaultdict(list)
    excluded = historical_set | test_set
    for dataset_index in range(len(dataset)):
        if dataset_index not in excluded:
            groups[date_for_index(dataset, dataset_index)].append(dataset_index)
    groups = {date: values for date, values in groups.items() if values}
    additional_count = VALIDATION_SAMPLES - len(historical_indices)
    quotas = proportional_quotas(groups, additional_count)
    rng = np.random.default_rng(SEED + 1)
    additional: list[int] = []
    for date in sorted(groups):
        chosen = rng.choice(groups[date], size=quotas[date], replace=False)
        additional.extend(int(index) for index in chosen)
    selected = sorted(historical_indices + additional)
    if len(selected) != VALIDATION_SAMPLES or len(set(selected)) != VALIDATION_SAMPLES:
        raise RuntimeError("正式验证集构建后样本数量或唯一性错误")

    date_sample_counts: dict[str, int] = defaultdict(int)
    for index in selected:
        date_sample_counts[date_for_index(dataset, index)] += 1
    payload = {
        "purpose": "fixed official validation subset",
        "seed": SEED + 1,
        "source_count": len(dataset),
        "historical_monitor_count": len(historical_indices),
        "additional_validation_count": len(additional),
        "excluded_test_count": len(test_indices),
        "sampling": "1000 historical monitor pairs plus 9000 date-stratified pairs excluding the final test set",
        "date_source_counts": {date: len(groups[date]) for date in sorted(groups)},
        "date_sample_counts": {date: date_sample_counts[date] for date in sorted(date_sample_counts)},
        "indices_sha256": sha256_json(selected),
        "samples": [index_record(dataset, index) for index in selected],
    }
    atomic_json(VALIDATION_SPLIT_PATH, payload)
    return selected


def fixed_validation_visual_indices() -> list[int]:
    """返回已完成训练中每轮展示的 5 个固定验证样本。"""
    with VISUAL_SPEC_PATH.open(encoding="utf-8") as handle:
        specification = json.load(handle)
    indices = [int(item["val_index"]) for item in specification["samples"]]
    if len(indices) != 5 or len(set(indices)) != 5:
        raise RuntimeError("The fixed validation visual specification must contain five unique indices")
    return indices


def load_or_create_independent_test_subset(
    dataset: Any,
    validation_indices: list[int],
    path: Path,
    sample_count: int,
) -> list[int]:
    """读取最终 1 万对测试集，并检查其不与验证样本重叠。

    epoch-271 实验已将训练日期与保留日期分开。最终测试集还与正式验证集及
    每轮展示的 5 个样本分离，因此测试指标和论文结果图仅使用未参与验证的样本。
    """
    excluded = sorted(set(int(index) for index in validation_indices) | set(fixed_validation_visual_indices()))
    excluded_set = set(excluded)
    if any(index < 0 or index >= len(dataset) for index in excluded_set):
        raise RuntimeError("Validation exclusion index is outside the held-out dataset")

    if path.exists():
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        indices = [int(item["index"]) for item in payload["samples"]]
        if len(indices) != sample_count:
            raise RuntimeError(f"Independent test subset has {len(indices)} samples, expected {sample_count}")
        if payload.get("indices_sha256") != sha256_json(indices):
            raise RuntimeError("Independent test subset hash mismatch")
        if set(indices) & excluded_set:
            raise RuntimeError("Independent test subset overlaps training-time validation indices")
        return indices

    groups: dict[str, list[int]] = defaultdict(list)
    for dataset_index in range(len(dataset)):
        if dataset_index not in excluded_set:
            groups[date_for_index(dataset, dataset_index)].append(dataset_index)
    groups = {date: values for date, values in groups.items() if values}
    if sum(len(values) for values in groups.values()) < sample_count:
        raise ValueError("Not enough untouched held-out samples for the independent test subset")
    quotas = proportional_quotas(groups, sample_count)
    rng = np.random.default_rng(SEED + 2)
    selected: list[int] = []
    for date in sorted(groups):
        chosen = rng.choice(groups[date], size=quotas[date], replace=False)
        selected.extend(int(index) for index in chosen)
    selected.sort()
    payload = {
        "purpose": "fixed independent final test subset",
        "seed": SEED + 2,
        "source_count": len(dataset),
        "excluded_training_time_validation_count": len(excluded),
        "excluded_indices_sha256": sha256_json(excluded),
        "eligible_count": sum(len(values) for values in groups.values()),
        "sample_count": len(selected),
        "sampling": "proportional stratification by held-out date after excluding monitor and visual indices",
        "date_source_counts": {date: len(groups[date]) for date in sorted(groups)},
        "date_sample_counts": {date: quotas[date] for date in sorted(quotas)},
        "indices_sha256": sha256_json(selected),
        "samples": [index_record(dataset, index) for index in selected],
    }
    atomic_json(path, payload)
    return selected


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


def build_datasets() -> tuple[Any, Any, list[int], list[int]]:
    """建立原始样本索引，并读取固定训练集和正式验证集清单。"""
    train_dataset, validation_dataset = split_by_dates(
        samples_root=SAMPLES_ROOT,
        val_dates=list(VAL_DATES),
        augment=False,
    )
    if len(train_dataset) != 222_154:
        raise RuntimeError(f"Unexpected training pair count: {len(train_dataset)}")
    if len(validation_dataset) != 63_133:
        raise RuntimeError(f"Unexpected validation pair count: {len(validation_dataset)}")
    train_indices = load_or_create_train_subset(
        train_dataset, TRAIN_SPLIT_PATH
    )
    validation_indices = load_or_create_validation_subset(validation_dataset)
    return train_dataset, validation_dataset, train_indices, validation_indices


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


def batch_psnr_ssim(
    prediction: torch.Tensor, target: torch.Tensor
) -> tuple[list[float], list[float]]:
    mse = (prediction - target).square().mean(dim=(1, 2, 3))
    psnr = (10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))).cpu().tolist()
    prediction_np = prediction[:, 0].detach().cpu().numpy()
    target_np = target[:, 0].detach().cpu().numpy()
    ssim = [
        float(ssim_metric(pred, obs, data_range=1.0))
        for pred, obs in zip(prediction_np, target_np)
    ]
    return psnr, ssim


def validation_loader(dataset: Any, indices: list[int], batch_size: int) -> DataLoader:
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=False,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
    )


def validate_model(
    model: torch.nn.Module,
    dataset: Any,
    indices: list[int],
    device: torch.device,
    batch_size: int = VALIDATION_BATCH,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    psnrs: list[float] = []
    ssims: list[float] = []
    output_stds: list[float] = []
    prediction_gradient_sum = 0.0
    target_gradient_sum = 0.0
    with torch.inference_mode(), torch.amp.autocast(
        "cuda", dtype=torch.float16, enabled=device.type == "cuda"
    ):
        for low_resolution, target in validation_loader(dataset, indices, batch_size):
            low_resolution = low_resolution.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            prediction = model(low_resolution).float()
            batch_psnr, batch_ssim = batch_psnr_ssim(prediction, target)
            psnrs.extend(batch_psnr)
            ssims.extend(batch_ssim)
            output_stds.extend(
                prediction.std(dim=(1, 2, 3)).detach().cpu().tolist()
            )
            prediction_gradient_sum += float(sobel_magnitude(prediction).sum().item())
            target_gradient_sum += float(sobel_magnitude(target).sum().item())
    model.train(was_training)
    return {
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "grad": prediction_gradient_sum / max(target_gradient_sum, 1e-12),
        "output_std": float(np.mean(output_stds)),
    }


def bicubic_metrics(dataset: Any, indices: list[int]) -> dict[str, float]:
    psnrs: list[float] = []
    ssims: list[float] = []
    prediction_gradient_sum = 0.0
    target_gradient_sum = 0.0
    for low_resolution, target in validation_loader(dataset, indices, VALIDATION_BATCH):
        prediction = bicubic_batch(low_resolution)
        batch_psnr, batch_ssim = batch_psnr_ssim(prediction, target)
        psnrs.extend(batch_psnr)
        ssims.extend(batch_ssim)
        prediction_gradient_sum += float(sobel_magnitude(prediction).sum().item())
        target_gradient_sum += float(sobel_magnitude(target).sum().item())
    return {
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "grad": prediction_gradient_sum / max(target_gradient_sum, 1e-12),
    }


# --------------------------------------------------------
# 步骤 4（优化器、EMA、checkpoint 与可视化函数）
# --------------------------------------------------------
def ema_parameter_lists(
    ema: torch.nn.Module,
    model: torch.nn.Module,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    ema_named = list(ema.named_parameters())
    model_named = list(model.named_parameters())
    if [name for name, _ in ema_named] != [name for name, _ in model_named]:
        raise RuntimeError("EMA/model parameter order mismatch")
    return [value for _, value in ema_named], [value for _, value in model_named]


@torch.no_grad()
def update_ema(
    ema_parameters: list[torch.Tensor],
    model_parameters: list[torch.Tensor],
    decay: float,
) -> None:
    """使用两次批量 CUDA foreach 调用更新全部 EMA 参数。"""
    torch._foreach_mul_(ema_parameters, decay)
    torch._foreach_add_(ema_parameters, model_parameters, alpha=1.0 - decay)


def make_optimizer(model: torch.nn.Module) -> torch.optim.Adam:
    """使用 PyTorch fused CUDA Adam，不改变既定优化器超参数。"""
    return torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=BETAS,
        weight_decay=0.0,
        fused=True,
    )


def cached_training_loader(
    dataset: Dataset,
    order: list[int],
    batch_size: int,
) -> DataLoader:
    return DataLoader(
        Subset(dataset, order),
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=CACHE_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=False,
        prefetch_factor=PREFETCH_FACTOR if CACHE_WORKERS > 0 else None,
    )


def load_fixed_visuals(validation_dataset: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    indices = fixed_validation_visual_indices()
    return [validation_dataset[index] for index in indices]


def save_epoch_montage(
    ema: torch.nn.Module,
    samples: list[tuple[torch.Tensor, torch.Tensor]],
    output_path: Path,
    device: torch.device,
    epoch: int,
) -> None:
    was_training = ema.training
    ema.eval()
    low_resolution = torch.stack([sample[0] for sample in samples])
    target = torch.stack([sample[1] for sample in samples])
    with torch.inference_mode(), torch.amp.autocast(
        "cuda", dtype=torch.float16, enabled=device.type == "cuda"
    ):
        prediction = ema(low_resolution.to(device)).float().cpu()
    ema.train(was_training)
    nearest = F.interpolate(low_resolution, size=(128, 128), mode="nearest")
    bicubic = bicubic_batch(low_resolution)
    arrays = [nearest, bicubic, prediction, target]
    titles = ["FY-4A LR", "Bicubic x4", "DRCT EMA", "FY-3C Observation"]

    tile = 240
    gap = 10
    left = 100
    top = 90
    colorbar_width = 95
    width = left + 4 * (tile + gap) + colorbar_width
    height = top + 5 * (tile + gap) + 25
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = ui_font(25, bold=True)
    label_font = ui_font(19, bold=True)
    small_font = ui_font(17)
    heading = f"Matched FY-4A to FY-3C | DRCT-L W8 x4 | Epoch {epoch}"
    heading_box = draw.textbbox((0, 0), heading, font=title_font)
    draw.text(((width - (heading_box[2] - heading_box[0])) / 2, 8), heading,
              fill="black", font=title_font)
    for column, title in enumerate(titles):
        x = left + column * (tile + gap)
        box = draw.textbbox((0, 0), title, font=label_font)
        draw.text((x + (tile - (box[2] - box[0])) / 2, 52), title,
                  fill="black", font=label_font)
    for row in range(5):
        y = top + row * (tile + gap)
        draw.text((8, y + tile // 2 - 10), f"Sample {row + 1}", fill="black", font=small_font)
        for column in range(4):
            physical_csi = arrays[column][row, 0].numpy() * CSI_SCALE
            panel = colorize_csi(physical_csi).resize(
                (tile, tile), Image.Resampling.NEAREST if column == 0 else Image.Resampling.BILINEAR
            )
            canvas.paste(panel, (left + column * (tile + gap), y))
    bar_x = left + 4 * (tile + gap) + 15
    bar_y = top
    bar_height = 5 * tile + 4 * gap
    gradient = np.linspace(CSI_SCALE, 0.0, bar_height, dtype=np.float32)[:, None]
    gradient = np.repeat(gradient, 24, axis=1)
    canvas.paste(colorize_csi(gradient), (bar_x, bar_y))
    draw.text((bar_x + 30, bar_y - 8), "1.5", fill="black", font=small_font)
    draw.text((bar_x + 30, bar_y + bar_height // 2 - 10), "0.75", fill="black", font=small_font)
    draw.text((bar_x + 30, bar_y + bar_height - 18), "0.0", fill="black", font=small_font)
    draw.text((bar_x - 2, bar_y + bar_height + 4), "Physical CSI", fill="black", font=small_font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.png")
    canvas.save(temporary, format="PNG")
    os.replace(temporary, output_path)


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
    loss = np.array([float(row["train_l1_mae"]) for row in rows])
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
        (loss, "Epoch-average L1 / MAE", "Normalized CSI MAE", None, "min"),
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


def save_resume_checkpoint(
    path: Path,
    model: torch.nn.Module,
    ema: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    next_epoch: int,
    global_step: int,
    best: dict[str, float | int],
    micro_batch: int,
    convergence_state: dict[str, Any] | None = None,
) -> None:
    checkpoint = {
        "kind": "matched FY-4A/FY-3C from-scratch resume state",
        "experiment": EXPERIMENT_NAME,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "next_epoch": next_epoch,
        "global_step": global_step,
        "best": best,
        "micro_batch": micro_batch,
        "effective_batch": EFFECTIVE_BATCH,
        "convergence_state": convergence_state,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
        },
        "config": asdict(ExperimentConfig()),
    }
    atomic_torch_save(path, checkpoint)


def save_inference_checkpoint(
    path: Path,
    ema: torch.nn.Module,
    epoch: int,
    global_step: int,
    metrics: dict[str, float],
    label: str,
) -> None:
    atomic_torch_save(
        path,
        {
            "params_ema": cpu_state_dict(ema),
            "epoch": epoch,
            "global_step": global_step,
            "metrics": metrics,
            "label": label,
            "initialization": "from scratch",
            "network": load_network_options(),
            "config": asdict(ExperimentConfig()),
        },
    )


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def print_effective_configuration(
    output_dir: Path,
    micro_batch: int,
    train_count: int,
    validation_count: int,
) -> None:
    print("\n========== DRCT MATCHED FROM-SCRATCH TRAINING ==========")
    print("SOURCE:")
    print("Matched FY-4A LR CSI -> FY-3C HR CSI")
    print("INITIALIZATION:")
    print("FROM SCRATCH (NO PRETRAINED CHECKPOINT)")
    print("MODEL:")
    print("DRCT-L x4 | img_size=32 | window=8 | shift=4")
    print("DATA:")
    print(f"{train_count} fixed pairs | seed={SEED}")
    print("BATCH:")
    print(f"micro={micro_batch} | accumulation={EFFECTIVE_BATCH // micro_batch} | effective=16")
    print("LOSS:")
    print("L1 / normalized-CSI MAE")
    print("OPTIMIZER:")
    print(f"CUDA fused Adam(lr={LEARNING_RATE:.1e}, betas={BETAS}, weight_decay=0)")
    print("DATA PIPELINE:")
    print(
        "bitwise-verified float32 memmap | "
        f"train_workers={CACHE_WORKERS} val_workers={NUM_WORKERS} "
        f"persistent_workers=False prefetch_factor={PREFETCH_FACTOR} "
        f"pin_memory={PIN_MEMORY}"
    )
    print("TRAINING:")
    print(
        f"convergence-controlled | {STEPS_PER_EPOCH} updates/epoch | "
        f"epoch {EPOCHS} hard safety cap"
    )
    print("L1 EARLY STOP:")
    print(
        f"epoch-average training L1 | min_delta={EARLY_STOP_MIN_DELTA:.1e} "
        f"vs best | patience={EARLY_STOP_PATIENCE} completed epochs | "
        f"earliest_stop_epoch={EARLY_STOP_MIN_EPOCH}"
    )
    print("VALIDATION:")
    print(f"{validation_count} date-stratified matched pairs after every epoch")
    print("SAVE TO:")
    print(output_dir.resolve())
    print("=========================================================\n")


def train_epoch(
    model: torch.nn.Module,
    ema: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    dataset: Dataset,
    epoch: int,
    micro_batch: int,
    device: torch.device,
) -> tuple[float, int]:
    accumulation = EFFECTIVE_BATCH // micro_batch
    generator = torch.Generator().manual_seed(SEED + epoch)
    order = torch.randperm(len(dataset), generator=generator).tolist()
    loader = cached_training_loader(dataset, order, micro_batch)
    iterator = iter(loader)
    model.train()
    absolute_error_sum = torch.zeros((), dtype=torch.float64, device=device)
    pixel_count = 0
    updates = 0
    ema_parameters, model_parameters = ema_parameter_lists(ema, model)
    for update in range(1, STEPS_PER_EPOCH + 1):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accumulation):
            low_resolution, target = next(iterator)
            low_resolution = low_resolution.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                prediction = model(low_resolution)
                batch_l1 = F.l1_loss(prediction, target)
                loss = batch_l1 / accumulation
            scaler.scale(loss).backward()
            absolute_error_sum.add_(
                batch_l1.detach().to(torch.float64) * target.numel()
            )
            pixel_count += target.numel()
        scaler.step(optimizer)
        scaler.update()
        update_ema(ema_parameters, model_parameters, EMA_DECAY)
        updates += 1
        if update == 1 or update % 100 == 0 or update == STEPS_PER_EPOCH:
            running_mae = float(absolute_error_sum.item()) / max(pixel_count, 1)
            print(
                f"[epoch {epoch:03d}/{EPOCHS}] update {update:04d}/{STEPS_PER_EPOCH} "
                f"L1/MAE={running_mae:.7f} AMP={scaler.get_scale():.0f}",
                flush=True,
            )
    if updates != STEPS_PER_EPOCH or pixel_count != TRAIN_SAMPLES * 128 * 128:
        raise RuntimeError(
            f"Incomplete epoch: updates={updates}, pixels={pixel_count}, expected updates={STEPS_PER_EPOCH}"
        )
    return float(absolute_error_sum.item()) / pixel_count, updates


# --------------------------------------------------------
# 步骤 5（冒烟测试与正式训练）
# --------------------------------------------------------
def run_smoke_test(output_dir: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DRCT-L training")
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "run_config.json", asdict(ExperimentConfig()))
    train_dataset, validation_dataset, train_indices, validation_indices = build_datasets()
    training_cache = prepare_training_cache(train_dataset, train_indices, output_dir)
    device = torch.device("cuda:0")
    # 正式冒烟测试前已实测：物理 batch=16 在这张 8 GiB GPU 上占用约 7.86 GiB，
    # 首次更新超过两分钟未完成。因此视为内存抖动而不可用，而非“未 OOM 就可用”。
    attempts: list[dict[str, Any]] = [{
        "micro_batch": 16,
        "result": "rejected after measured preflight",
        "peak_gpu_mib": 7858,
        "reason": "GPU memory saturated and first update exceeded 120 seconds",
    }]
    for micro_batch in (8, 4):
        model = ema = optimizer = scaler = None
        try:
            seed_everything(SEED)
            torch.cuda.empty_cache()
            model = build_model(device)
            ema = copy.deepcopy(model).eval().requires_grad_(False)
            torch.cuda.reset_peak_memory_stats()
            optimizer = make_optimizer(model)
            scaler = torch.amp.GradScaler("cuda")
            accumulation = EFFECTIVE_BATCH // micro_batch
            # 复现正式第 1 个 epoch 使用的随机访问前缀。
            smoke_generator = torch.Generator().manual_seed(SEED + 1)
            smoke_positions = torch.randperm(
                len(training_cache), generator=smoke_generator
            )[: SMOKE_STEPS * EFFECTIVE_BATCH].tolist()
            loader = cached_training_loader(
                training_cache,
                smoke_positions,
                micro_batch,
            )
            iterator = iter(loader)
            losses: list[float] = []
            ema_parameters, model_parameters = ema_parameter_lists(ema, model)
            started = time.time()
            for step in range(1, SMOKE_STEPS + 1):
                optimizer.zero_grad(set_to_none=True)
                step_abs = torch.zeros((), dtype=torch.float64, device=device)
                step_pixels = 0
                for _ in range(accumulation):
                    low_resolution, target = next(iterator)
                    low_resolution = low_resolution.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
                    if low_resolution.shape != (micro_batch, 1, 32, 32):
                        raise RuntimeError(f"Unexpected FY-4A shape: {tuple(low_resolution.shape)}")
                    if target.shape != (micro_batch, 1, 128, 128):
                        raise RuntimeError(f"Unexpected FY-3C shape: {tuple(target.shape)}")
                    if step == 1 and (
                        not torch.isfinite(low_resolution).all()
                        or not torch.isfinite(target).all()
                    ):
                        raise FloatingPointError("Smoke cache batch contains NaN/Inf")
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        prediction = model(low_resolution)
                        batch_l1 = F.l1_loss(prediction, target)
                        loss = batch_l1 / accumulation
                    scaler.scale(loss).backward()
                    step_abs.add_(
                        batch_l1.detach().to(torch.float64) * target.numel()
                    )
                    step_pixels += target.numel()
                scaler.step(optimizer)
                scaler.update()
                update_ema(ema_parameters, model_parameters, EMA_DECAY)
                losses.append(float(step_abs.item()) / step_pixels)
                if not math.isfinite(losses[-1]):
                    raise FloatingPointError("Smoke test produced NaN/Inf loss")
                print(
                    f"[smoke micro={micro_batch}] step {step:02d}/{SMOKE_STEPS} "
                    f"L1={losses[-1]:.7f}",
                    flush=True,
                )

            training_seconds = time.time() - started

            metrics = validate_model(
                ema, validation_dataset, validation_indices[:32], device, batch_size=4
            )
            smoke_checkpoint = output_dir / "smoke_checkpoint.tmp.pth"
            save_resume_checkpoint(
                smoke_checkpoint,
                model,
                ema,
                optimizer,
                scaler,
                next_epoch=1,
                global_step=SMOKE_STEPS,
                best={"psnr": metrics["psnr"], "ssim": metrics["ssim"], "epoch": 0},
                micro_batch=micro_batch,
            )
            restored = torch.load(smoke_checkpoint, map_location="cpu", weights_only=False)
            required = {"model", "ema", "optimizer", "scaler", "next_epoch", "global_step", "rng"}
            if not required.issubset(restored):
                raise RuntimeError("Smoke checkpoint round-trip is incomplete")
            smoke_checkpoint.unlink()

            # 检查正式训练中的逐 epoch CSV 与 PIL/OpenCV 可视化输出路径。
            smoke_log = output_dir / "smoke_training_log.tmp.csv"
            smoke_progress = output_dir / "smoke_progress.tmp.png"
            smoke_visual = output_dir / "smoke_visual.tmp.png"
            smoke_baseline = bicubic_metrics(validation_dataset, validation_indices[:32])
            append_epoch_log(
                smoke_log,
                {
                    "epoch": 1,
                    "global_step": SMOKE_STEPS,
                    "train_l1_mae": f"{float(np.mean(losses)):.9f}",
                    "val_psnr": f"{metrics['psnr']:.6f}",
                    "val_ssim": f"{metrics['ssim']:.6f}",
                    "val_grad": f"{metrics['grad']:.6f}",
                    "output_std": f"{metrics['output_std']:.6f}",
                    "learning_rate": f"{LEARNING_RATE:.8g}",
                    "epoch_seconds": "0.0",
                    "elapsed_seconds": "0.0",
                    "best_real": 1,
                },
                new_file=True,
            )
            draw_progress(smoke_log, smoke_baseline, smoke_progress)
            save_epoch_montage(
                ema,
                load_fixed_visuals(validation_dataset),
                smoke_visual,
                device,
                epoch=0,
            )
            if len(read_training_log(smoke_log)) != 1:
                raise RuntimeError("Smoke epoch log did not round-trip as exactly one row")
            if not smoke_progress.exists() or not smoke_visual.exists():
                raise RuntimeError("Smoke visualization files were not created")
            smoke_log.unlink()
            smoke_progress.unlink()
            smoke_visual.unlink()
            elapsed = time.time() - started
            report = {
                "passed": True,
                "source": "matched FY-4A LR / FY-3C HR pairs",
                "initialization": "from scratch; no pretrained checkpoint loaded",
                "micro_batch": micro_batch,
                "accumulation": accumulation,
                "effective_batch": EFFECTIVE_BATCH,
                "steps": SMOKE_STEPS,
                "mean_l1": float(np.mean(losses)),
                "first_l1": losses[0],
                "last_l1": losses[-1],
                "validation_32": metrics,
                "amp_scale": scaler.get_scale(),
                "peak_gpu_mib": torch.cuda.max_memory_allocated() / 1024**2,
                "elapsed_seconds": elapsed,
                "training_seconds": training_seconds,
                "training_steps_per_second": SMOKE_STEPS / training_seconds,
                "cache": {
                    "dtype": "float32",
                    "sample_count": len(training_cache),
                    "indices_sha256": sha256_json(train_indices),
                    "source_parity": "64/64 bitwise identical",
                    "workers": CACHE_WORKERS,
                },
                "optimizer_implementation": "CUDA fused Adam",
                "ema_implementation": "CUDA foreach",
                "checkpoint_roundtrip": True,
                "epoch_log_roundtrip": True,
                "progress_plot_roundtrip": True,
                "fixed_visual_roundtrip": True,
                "attempts": attempts,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            atomic_json(output_dir / "smoke_report.json", report)
            print("\nSMOKE TEST PASSED")
            print(json.dumps(report, indent=2))
            return report
        except torch.OutOfMemoryError as exc:
            attempts.append({"micro_batch": micro_batch, "result": "CUDA OOM", "error": str(exc)})
            print(f"[smoke] physical/micro batch {micro_batch} OOM; trying a smaller micro batch")
        finally:
            del model, ema, optimizer, scaler
            gc.collect()
            torch.cuda.empty_cache()
    report = {"passed": False, "attempts": attempts, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    atomic_json(output_dir / "smoke_report.json", report)
    raise RuntimeError("No micro-batch setting could provide effective batch 16")


def append_epoch_log(path: Path, row: dict[str, Any], new_file: bool) -> None:
    fields = [
        "epoch",
        "global_step",
        "train_l1_mae",
        "val_psnr",
        "val_ssim",
        "val_grad",
        "output_std",
        "learning_rate",
        "epoch_seconds",
        "elapsed_seconds",
        "best_real",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if new_file else "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def derive_convergence_state(rows: list[dict[str, str]]) -> dict[str, Any]:
    best_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    for row in rows:
        epoch = int(row["epoch"])
        loss = float(row["train_l1_mae"])
        if not math.isfinite(loss):
            raise FloatingPointError(f"Non-finite epoch-average L1 in epoch {epoch}")
        if loss <= best_loss - EARLY_STOP_MIN_DELTA:
            best_loss = loss
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
    last_epoch = int(rows[-1]["epoch"]) if rows else 0
    should_stop = (
        last_epoch >= EARLY_STOP_MIN_EPOCH
        and epochs_without_improvement >= EARLY_STOP_PATIENCE
    )
    return {
        "metric": "epoch-average training L1 / normalized-CSI MAE",
        "mode": "min",
        "min_delta": EARLY_STOP_MIN_DELTA,
        "patience": EARLY_STOP_PATIENCE,
        "earliest_stop_epoch": EARLY_STOP_MIN_EPOCH,
        "best_l1": best_loss,
        "best_epoch": best_epoch,
        "epochs_without_improvement": epochs_without_improvement,
        "last_completed_epoch": last_epoch,
        "should_stop": should_stop,
    }


def row_validation_metrics(row: dict[str, str]) -> dict[str, float]:
    return {
        "psnr": float(row["val_psnr"]),
        "ssim": float(row["val_ssim"]),
        "grad": float(row["val_grad"]),
        "output_std": float(row["output_std"]),
    }


def run_training(output_dir: Path, resume: bool) -> tuple[Path, Path | None]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for formal training")
    smoke_path = output_dir / "smoke_report.json"
    if not smoke_path.exists():
        raise RuntimeError("Smoke test has not been run")
    with smoke_path.open(encoding="utf-8") as handle:
        smoke = json.load(handle)
    if not smoke.get("passed"):
        raise RuntimeError("Smoke test did not pass")
    micro_batch = min(int(smoke["micro_batch"]), TRAIN_MICRO_BATCH_CAP)
    if EFFECTIVE_BATCH % micro_batch:
        raise RuntimeError("Smoke-selected micro batch does not divide effective batch")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    visual_dir = output_dir / "visuals"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training_log.csv"
    latest_path = checkpoint_dir / "resume_latest.pth"
    best_path = checkpoint_dir / "best_real_continuation.pth"
    completion_path = output_dir / "training_completion.json"
    if resume and completion_path.exists():
        with completion_path.open(encoding="utf-8") as handle:
            completion = json.load(handle)
        completed_final = output_dir / completion["final_checkpoint"]
        if completion.get("training_status") == "complete" and completed_final.exists():
            print(
                f"TRAINING ALREADY STOPPED: reason={completion['stop_reason']} "
                f"epoch={completion['stop_epoch']}; continuing formal evaluation",
                flush=True,
            )
            return completed_final, best_path if best_path.exists() else None
    if not resume and (log_path.exists() or latest_path.exists()):
        raise FileExistsError(
            f"Formal outputs already exist in {output_dir}; use --resume rather than overwriting"
        )
    if resume and not latest_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {latest_path}")

    config_payload = asdict(ExperimentConfig())
    config_payload["continuation"] = {
        "start_source": "checkpoints/epoch50_final.pth params_ema",
        "resume_epoch": CONTINUATION_START_EPOCH,
        "global_step_at_start": 156250,
        "optimizer_state": "fresh at epoch 51; subsequently resume-safe",
        "learning_rate": LEARNING_RATE,
        "stopping_target": "epoch-average training L1 convergence",
        "early_stopping": {
            "min_delta": EARLY_STOP_MIN_DELTA,
            "patience": EARLY_STOP_PATIENCE,
            "earliest_stop_epoch": EARLY_STOP_MIN_EPOCH,
            "comparison": "current completed epoch versus best completed epoch-average L1",
        },
        "hard_cap_epoch": EPOCHS,
        "final_checkpoint_pattern": "checkpoints/epoch_NNN_{converged|hard_cap}_final.pth",
        "preserved_checkpoint": "checkpoints/best_real.pth",
        "audit_archive": "audit_lr1e3_epoch51_20260815",
    }
    atomic_json(output_dir / "run_config.json", config_payload)

    train_dataset, validation_dataset, train_indices, validation_indices = build_datasets()
    training_cache = prepare_training_cache(train_dataset, train_indices, output_dir)
    fixed_visuals = load_fixed_visuals(validation_dataset)
    baseline_path = output_dir / "bicubic_baseline.json"
    if baseline_path.exists():
        with baseline_path.open(encoding="utf-8") as handle:
            baseline = json.load(handle)
    else:
        print("Calculating the fixed date-stratified Bicubic baseline ...", flush=True)
        baseline = bicubic_metrics(validation_dataset, validation_indices)
        atomic_json(baseline_path, baseline)

    device = torch.device("cuda:0")
    seed_everything(SEED)
    model = build_model(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = make_optimizer(model)
    scaler = torch.amp.GradScaler("cuda")
    start_epoch = 1
    global_step = 0
    best: dict[str, float | int] = {"psnr": -math.inf, "ssim": -math.inf, "epoch": 0}
    elapsed_before = 0.0
    existing_rows: list[dict[str, str]] = []
    convergence_state = derive_convergence_state(existing_rows)

    if resume:
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        if checkpoint.get("experiment") != EXPERIMENT_NAME:
            raise RuntimeError("Resume checkpoint belongs to a different experiment")
        model.load_state_dict(checkpoint["model"], strict=True)
        ema.load_state_dict(checkpoint["ema"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = LEARNING_RATE
        optimizer_to_device(optimizer, device)
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["next_epoch"])
        global_step = int(checkpoint["global_step"])
        best = checkpoint["best"]
        rng = checkpoint["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state_all(rng["cuda"])
        existing_rows = read_training_log(log_path)
        if len(existing_rows) != start_epoch - 1:
            raise RuntimeError(
                f"Resume/log mismatch: checkpoint next_epoch={start_epoch}, log rows={len(existing_rows)}"
            )
        elapsed_before = sum(float(row["epoch_seconds"]) for row in existing_rows)
        if not (CONTINUATION_START_EPOCH <= start_epoch <= EPOCHS + 1):
            raise RuntimeError(
                f"Resume epoch {start_epoch} is outside continuation range "
                f"{CONTINUATION_START_EPOCH}-{EPOCHS + 1}"
            )
        convergence_state = derive_convergence_state(existing_rows)
        saved_convergence = checkpoint.get("convergence_state")
        if saved_convergence is not None and saved_convergence != convergence_state:
            raise RuntimeError("Resume checkpoint convergence state disagrees with training_log.csv")
        start_source = checkpoint.get("start_source", str(latest_path.resolve()))
        optimizer_origin = checkpoint.get("optimizer_state_origin", "restored from checkpoint")
        print(f"START SOURCE: {start_source}", flush=True)
        print(f"OPTIMIZER STATE: {optimizer_origin}", flush=True)
        print(
            f"RESUME VERIFIED: next_epoch={start_epoch}, global_step={global_step}, "
            f"optimizer_lr={optimizer.param_groups[0]['lr']:.8g}; scheduler=none",
            flush=True,
        )
        del checkpoint, rng
        gc.collect()

    print_effective_configuration(
        output_dir, micro_batch, len(train_indices), len(validation_indices)
    )
    print(f"Bicubic baseline: {json.dumps(baseline)}", flush=True)
    print(f"CONVERGENCE STATE AT RESUME: {json.dumps(convergence_state)}", flush=True)
    training_started = time.time()
    try:
        if convergence_state["should_stop"] or start_epoch > EPOCHS:
            stop_epoch = int(convergence_state["last_completed_epoch"])
            stop_reason = "l1_converged" if convergence_state["should_stop"] else "hard_cap"
            suffix = "converged" if stop_reason == "l1_converged" else "hard_cap"
            final_path = checkpoint_dir / f"epoch_{stop_epoch:03d}_{suffix}_final.pth"
            last_row = existing_rows[-1]
            save_inference_checkpoint(
                final_path,
                ema,
                stop_epoch,
                global_step,
                row_validation_metrics(last_row),
                f"epoch {stop_epoch} final ({stop_reason})",
            )
            atomic_json(
                completion_path,
                {
                    "training_status": "complete",
                    "stop_reason": stop_reason,
                    "stop_epoch": stop_epoch,
                    "final_checkpoint": str(final_path.relative_to(output_dir)).replace("\\", "/"),
                    "convergence_state": convergence_state,
                    "hard_cap_epoch": EPOCHS,
                },
            )
            print(f"TRAINING STOP RESTORED FROM LOG: {final_path.resolve()}", flush=True)
            return final_path, best_path if best_path.exists() else None

        final_path: Path | None = None
        for epoch in range(start_epoch, EPOCHS + 1):
            epoch_started = time.time()
            train_loss, updates = train_epoch(
                model,
                ema,
                optimizer,
                scaler,
                training_cache,
                epoch,
                micro_batch,
                device,
            )
            global_step += updates
            metrics = validate_model(ema, validation_dataset, validation_indices, device)
            epoch_seconds = time.time() - epoch_started
            elapsed = elapsed_before + time.time() - training_started
            is_best = (
                metrics["psnr"] > float(best["psnr"])
                or (
                    math.isclose(metrics["psnr"], float(best["psnr"]), abs_tol=1e-12)
                    and metrics["ssim"] > float(best["ssim"])
                )
            )
            if is_best:
                best = {"psnr": metrics["psnr"], "ssim": metrics["ssim"], "epoch": epoch}
                save_inference_checkpoint(
                    best_path, ema, epoch, global_step, metrics, "best real validation PSNR"
                )
            row = {
                "epoch": epoch,
                "global_step": global_step,
                "train_l1_mae": f"{train_loss:.9f}",
                "val_psnr": f"{metrics['psnr']:.6f}",
                "val_ssim": f"{metrics['ssim']:.6f}",
                "val_grad": f"{metrics['grad']:.6f}",
                "output_std": f"{metrics['output_std']:.6f}",
                "learning_rate": f"{optimizer.param_groups[0]['lr']:.8g}",
                "epoch_seconds": f"{epoch_seconds:.1f}",
                "elapsed_seconds": f"{elapsed:.1f}",
                "best_real": int(is_best),
            }
            append_epoch_log(log_path, row, new_file=(epoch == 1 and not resume))
            existing_rows.append({key: str(value) for key, value in row.items()})
            convergence_state = derive_convergence_state(existing_rows)
            save_epoch_montage(
                ema,
                fixed_visuals,
                visual_dir / f"epoch_{epoch:03d}.png",
                device,
                epoch,
            )
            draw_progress(log_path, baseline, output_dir / "training_progress.png")
            save_resume_checkpoint(
                latest_path,
                model,
                ema,
                optimizer,
                scaler,
                next_epoch=epoch + 1,
                global_step=global_step,
                best=best,
                micro_batch=micro_batch,
                convergence_state=convergence_state,
            )
            print(
                f"[epoch {epoch:03d} complete] L1={train_loss:.7f} "
                f"PSNR={metrics['psnr']:.4f} SSIM={metrics['ssim']:.5f} "
                f"Grad.={metrics['grad']:.5f} best={best} time={epoch_seconds / 60:.1f} min",
                flush=True,
            )
            print(
                f"[L1 convergence] best={convergence_state['best_l1']:.9f} "
                f"at epoch {convergence_state['best_epoch']}; "
                f"without_improvement={convergence_state['epochs_without_improvement']}/"
                f"{EARLY_STOP_PATIENCE}; earliest_stop={EARLY_STOP_MIN_EPOCH}",
                flush=True,
            )
            if convergence_state["should_stop"]:
                final_path = checkpoint_dir / f"epoch_{epoch:03d}_converged_final.pth"
                stop_reason = "l1_converged"
                break
        if final_path is None:
            epoch = EPOCHS
            final_path = checkpoint_dir / f"epoch_{epoch:03d}_hard_cap_final.pth"
            stop_reason = "hard_cap"
        save_inference_checkpoint(
            final_path,
            ema,
            epoch,
            global_step,
            metrics,
            f"epoch {epoch} final ({stop_reason})",
        )
        atomic_json(
            completion_path,
            {
                "training_status": "complete",
                "stop_reason": stop_reason,
                "stop_epoch": epoch,
                "final_checkpoint": str(final_path.relative_to(output_dir)).replace("\\", "/"),
                "convergence_state": convergence_state,
                "hard_cap_epoch": EPOCHS,
            },
        )
        print(
            f"FORMAL TRAINING COMPLETE: reason={stop_reason} epoch={epoch} "
            f"checkpoint={final_path.resolve()}",
            flush=True,
        )
        return final_path, best_path if best_path.exists() else None
    except BaseException:
        # 仅保留最后一个完整 epoch；续训时会重新执行被中断的部分 epoch，确保
        # 每条训练日志都对应完整的 50,000 对训练样本。
        print("Training interrupted; resume_latest.pth still represents the last completed epoch.", flush=True)
        raise


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


# --------------------------------------------------------
# 步骤 6（独立测试与论文散点图输出）
# --------------------------------------------------------
class ClearSkyProvider:
    def __init__(self, dataset: Any, cache_size: int = 2) -> None:
        self.dataset = dataset
        self.cache_size = cache_size
        self.cache: OrderedDict[Path, np.ndarray] = OrderedDict()
        self.rest2_dirs: dict[Path, Path] = {}

    def get(self, dataset_index: int) -> np.ndarray:
        batch_path, batch_index = self.dataset.index[int(dataset_index)]
        batch = self.dataset._load_batch(batch_path)
        row0 = int(batch["row0"][batch_index])
        col0 = int(batch["col0"][batch_index])
        minute = int(batch["dominant_slot_minute_utc"][batch_index])
        date_dir = batch_path.parents[1]
        if date_dir not in self.rest2_dirs:
            with np.load(date_dir / "build_meta.npz") as metadata:
                self.rest2_dirs[date_dir] = Path(str(metadata["rest2_dir"].item()))
        label = f"{minute // 60:02d}{minute % 60:02d}"
        clear_path = self.rest2_dirs[date_dir] / f"REST2_{label}_on_FY3.npz"
        if clear_path in self.cache:
            clear_field = self.cache.pop(clear_path)
            self.cache[clear_path] = clear_field
        else:
            with np.load(clear_path) as clear_file:
                clear_field = clear_file["ghi"].astype(np.float32)
            self.cache[clear_path] = clear_field
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        patch = clear_field[row0 : row0 + 128, col0 : col0 + 128]
        if patch.shape != (128, 128):
            raise RuntimeError(f"Invalid matched GHI_clear crop: {patch.shape}")
        return patch


def plot_ghi_scatter(
    observation: np.ndarray,
    bicubic: np.ndarray,
    drct: np.ndarray,
    bicubic_result: dict[str, float],
    drct_result: dict[str, float],
    axis_min: float,
    axis_max: float,
    output_path: Path,
) -> None:
    canvas_width, canvas_height = 1300, 2240
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = ui_font(24, bold=True)
    axis_font = ui_font(19)
    metric_font = ui_font(18)
    rng = np.random.default_rng(SEED)
    panels = [
        (bicubic, bicubic_result, "(a) Bicubic: 4 km to 1 km", "#64b5f6"),
        (drct, drct_result, "(b) DRCT: 4 km to 1 km", "#ffb74d"),
    ]
    plot_size = 880
    left = 220
    for panel_index, (prediction, metrics, title, color_hex) in enumerate(panels):
        top = 80 + panel_index * 1100
        title_box = draw.textbbox((0, 0), title, font=title_font)
        draw.text((left + (plot_size - (title_box[2] - title_box[0])) / 2, top - 48),
                  title, fill="black", font=title_font)
        plot = np.full((plot_size, plot_size, 3), 255, dtype=np.float32)
        sample_count = min(150_000, observation.size)
        if observation.size > sample_count:
            choice = rng.choice(observation.size, size=sample_count, replace=False)
            x_value = observation[choice]
            y_value = prediction[choice]
        else:
            x_value = observation
            y_value = prediction
        x_pixel = np.clip(
            ((x_value - axis_min) / (axis_max - axis_min) * (plot_size - 1)).astype(np.int32),
            0,
            plot_size - 1,
        )
        y_pixel = np.clip(
            ((axis_max - y_value) / (axis_max - axis_min) * (plot_size - 1)).astype(np.int32),
            0,
            plot_size - 1,
        )
        density = np.zeros((plot_size, plot_size), dtype=np.uint16)
        np.add.at(density, (y_pixel, x_pixel), 1)
        rgb = tuple(int(color_hex[index : index + 2], 16) for index in (1, 3, 5))
        alpha = 1.0 - np.exp(-density.astype(np.float32) * 0.18)
        for channel in range(3):
            plot[:, :, channel] = 255.0 * (1.0 - alpha) + rgb[channel] * alpha
        panel_image = Image.fromarray(np.rint(plot).astype(np.uint8))
        canvas.paste(panel_image, (left, top))
        draw.rectangle((left, top, left + plot_size, top + plot_size), outline=(70, 70, 70), width=2)
        for dash in range(0, plot_size, 24):
            draw.line((left + dash, top + plot_size - dash,
                       left + min(dash + 14, plot_size), top + plot_size - min(dash + 14, plot_size)),
                      fill="black", width=2)
        draw.text((left, top + plot_size + 18), f"{axis_min:.0f}", fill="black", font=axis_font)
        max_label = f"{axis_max:.0f}"
        max_box = draw.textbbox((0, 0), max_label, font=axis_font)
        draw.text((left + plot_size - (max_box[2] - max_box[0]), top + plot_size + 18),
                  max_label, fill="black", font=axis_font)
        x_label = "FY-3C Observation [W/m²]"
        x_box = draw.textbbox((0, 0), x_label, font=axis_font)
        draw.text((left + (plot_size - (x_box[2] - x_box[0])) / 2, top + plot_size + 50),
                  x_label, fill="black", font=axis_font)
        y_label = f"{'Bicubic' if panel_index == 0 else 'DRCT'} Result [W/m²]"
        vertical = Image.new("RGBA", (plot_size, 45), (255, 255, 255, 0))
        vertical_draw = ImageDraw.Draw(vertical)
        vertical_draw.text((0, 5), y_label, fill="black", font=axis_font)
        vertical = vertical.rotate(90, expand=True)
        canvas.paste(vertical, (120, top + (plot_size - vertical.height) // 2), vertical)
        text = (
            f"R² = {metrics['r2']:.4f}\n"
            f"RMSE = {metrics['rmse']:.2f} W/m²\n"
            f"MAE = {metrics['mae']:.2f} W/m²"
        )
        metric_box = (left + 22, top + 18, left + 350, top + 112)
        draw.rounded_rectangle(metric_box, radius=8, fill=(255, 255, 255), outline=(210, 210, 210))
        draw.multiline_text((left + 34, top + 27), text, fill="black", font=metric_font, spacing=6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.png")
    canvas.save(temporary, format="PNG")
    os.replace(temporary, output_path)


def formal_evaluation(
    checkpoint_path: Path,
    output_dir: Path,
    make_scatter: bool,
) -> dict[str, Any]:
    """使用冻结 checkpoint 在独立的 1 万对测试样本上计算最终指标。"""
    device = torch.device("cuda:0")
    _, validation_dataset = split_by_dates(
        samples_root=SAMPLES_ROOT, val_dates=list(VAL_DATES), augment=False
    )
    validation_indices = load_or_create_validation_subset(validation_dataset)
    test_indices = load_or_create_independent_test_subset(
        validation_dataset,
        validation_indices,
        TEST_SPLIT_PATH,
        FINAL_TEST_SAMPLES,
    )
    seed_everything(SEED)
    model = build_model(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "params_ema" not in checkpoint:
        raise KeyError(f"Inference checkpoint has no params_ema: {checkpoint_path}")
    model.load_state_dict(checkpoint["params_ema"], strict=True)
    model.eval()
    clear_sky = ClearSkyProvider(validation_dataset)
    bicubic_accumulator = RegressionAccumulator()
    drct_accumulator = RegressionAccumulator()
    per_date_bicubic: dict[str, RegressionAccumulator] = defaultdict(RegressionAccumulator)
    per_date_drct: dict[str, RegressionAccumulator] = defaultdict(RegressionAccumulator)
    psnrs: list[float] = []
    ssims: list[float] = []
    prediction_gradient_sum = 0.0
    target_gradient_sum = 0.0
    scatter_observation: list[np.ndarray] = []
    scatter_bicubic: list[np.ndarray] = []
    scatter_drct: list[np.ndarray] = []
    full_min = math.inf
    full_max = -math.inf
    rng = np.random.default_rng(SEED)
    loader = validation_loader(validation_dataset, test_indices, VALIDATION_BATCH)
    offset = 0
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
        for batch_number, (low_resolution, target) in enumerate(loader, 1):
            prediction = model(low_resolution.to(device, non_blocking=True)).float().cpu()
            bicubic = bicubic_batch(low_resolution)
            batch_psnr, batch_ssim = batch_psnr_ssim(prediction, target)
            psnrs.extend(batch_psnr)
            ssims.extend(batch_ssim)
            prediction_gradient_sum += float(sobel_magnitude(prediction).sum().item())
            target_gradient_sum += float(sobel_magnitude(target).sum().item())
            for item in range(target.shape[0]):
                dataset_index = test_indices[offset]
                offset += 1
                clear = clear_sky.get(dataset_index)
                observation_ghi = target[item, 0].numpy() * CSI_SCALE * clear
                bicubic_ghi = bicubic[item, 0].numpy() * CSI_SCALE * clear
                drct_ghi = prediction[item, 0].numpy() * CSI_SCALE * clear
                valid = (
                    np.isfinite(observation_ghi)
                    & np.isfinite(bicubic_ghi)
                    & np.isfinite(drct_ghi)
                )
                observation_valid = observation_ghi[valid]
                bicubic_valid = bicubic_ghi[valid]
                drct_valid = drct_ghi[valid]
                bicubic_accumulator.update(observation_valid, bicubic_valid)
                drct_accumulator.update(observation_valid, drct_valid)
                date = date_for_index(validation_dataset, dataset_index)
                per_date_bicubic[date].update(observation_valid, bicubic_valid)
                per_date_drct[date].update(observation_valid, drct_valid)
                if observation_valid.size:
                    full_min = min(full_min, float(observation_valid.min()), float(bicubic_valid.min()), float(drct_valid.min()))
                    full_max = max(full_max, float(observation_valid.max()), float(bicubic_valid.max()), float(drct_valid.max()))
                    count = min(64, observation_valid.size)
                    choice = rng.choice(observation_valid.size, size=count, replace=False)
                    scatter_observation.append(observation_valid[choice].astype(np.float32))
                    scatter_bicubic.append(bicubic_valid[choice].astype(np.float32))
                    scatter_drct.append(drct_valid[choice].astype(np.float32))
            if batch_number % 100 == 0 or batch_number == len(loader):
                print(f"[independent test evaluation] {offset}/{len(test_indices)}", flush=True)

    result = {
        "checkpoint": str(checkpoint_path.resolve()),
        "split": "independent_test_subset_of_heldout_dates",
        "exclusion": "disjoint from the 1000 per-epoch monitor indices and five training-time visual indices",
        "samples": len(test_indices),
        "csi": {
            "psnr": float(np.mean(psnrs)),
            "ssim": float(np.mean(ssims)),
            "grad": prediction_gradient_sum / max(target_gradient_sum, 1e-12),
        },
        "ghi": {
            "bicubic": bicubic_accumulator.metrics(),
            "drct": drct_accumulator.metrics(),
        },
        "by_date": {
            date: {
                "bicubic": per_date_bicubic[date].metrics(),
                "drct": per_date_drct[date].metrics(),
            }
            for date in sorted(per_date_drct)
        },
        "common_mask": "finite FY-3C/Bicubic/DRCT GHI; no value clipping",
        "ghi_restore": "normalized CSI * 1.5 * matched GHI_clear",
    }
    final_dir = output_dir / "final"
    label = checkpoint_path.stem
    atomic_json(final_dir / f"{label}_independent_test_metrics.json", result)
    with (final_dir / f"{label}_independent_test_metrics_by_date.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = ["date", "method", "r2", "rmse", "mae", "bias"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for date in sorted(result["by_date"]):
            for method in ("bicubic", "drct"):
                writer.writerow({"date": date, "method": method, **result["by_date"][date][method]})
    if make_scatter:
        observation = np.concatenate(scatter_observation)
        bicubic_values = np.concatenate(scatter_bicubic)
        drct_values = np.concatenate(scatter_drct)
        span = max(full_max - full_min, 1.0)
        margin = 0.035 * span
        plot_ghi_scatter(
            observation,
            bicubic_values,
            drct_values,
            result["ghi"]["bicubic"],
            result["ghi"]["drct"],
            full_min - margin,
            full_max + margin,
            final_dir / f"scatter_ghi_bicubic_{label}_independent_test.png",
        )
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    """执行一种明确模式：训练、续训、冒烟测试或最终测试。"""

    # ----------------------------------------
    # 步骤 1（读取运行配置）
    # ----------------------------------------
    args = parse_args()
    output_dir = args.output_dir.resolve()

    # ----------------------------------------
    # 步骤 2（执行指定实验模式）
    # ----------------------------------------
    if args.smoke_test:
        run_smoke_test(output_dir)
        return
    if args.train:
        final_path, best_path = run_training(output_dir, resume=False)
        formal_evaluation(final_path, output_dir, make_scatter=False)
        if best_path is not None:
            formal_evaluation(best_path, output_dir, make_scatter=True)
        return
    if args.resume:
        final_path, best_path = run_training(output_dir, resume=True)
        formal_evaluation(final_path, output_dir, make_scatter=False)
        if best_path is not None:
            formal_evaluation(best_path, output_dir, make_scatter=True)
        return

    # ----------------------------------------
    # 步骤 3（测试冻结的 checkpoint）
    # ----------------------------------------
    checkpoint = args.checkpoint or default_validation_checkpoint(output_dir)
    formal_evaluation(checkpoint.resolve(), output_dir, make_scatter=True)


if __name__ == "__main__":
    main()
