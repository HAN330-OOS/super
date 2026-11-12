"""DRCT training implementation for the fixed 2018-2019 raw CSI CSV splits.

The public prepare.py check entry uses the complete original DRCT-L network.
The internal parser retains the old explicitly labelled tiny-check option.
PPU execution is unverified: it requires a vendor-patched PyTorch runtime with
the CUDA-compatible APIs used below. This runner never installs that runtime.
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
import traceback
import uuid
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset

from utils import training as helpers
from utils.metrics import batch_psnr_ssim as csi_batch_metrics

FORMAT_VERSION = 2
GLOBAL_BATCH = 64
SEED = 60
EMA_DECAY = 0.999
LR = 1e-4
BETAS = (0.9, 0.999)


@dataclass
class Runtime:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    backend: str | None

    @property
    def primary(self) -> bool:
        return self.rank == 0


def batch_geometry(
    world_size: int, micro_batch: int | None = None, accumulation: int | None = None
) -> tuple[int, int]:
    """Preserve the original GLOBAL batch of sixteen, including accumulation."""
    if world_size < 1 or GLOBAL_BATCH % world_size:
        raise ValueError("world size must divide the fixed global batch of 16")
    if micro_batch is None:
        micro_batch = min(8, GLOBAL_BATCH // world_size)
    if micro_batch < 1 or GLOBAL_BATCH % (world_size * micro_batch):
        raise ValueError("world_size * micro_batch must divide 64")
    expected = GLOBAL_BATCH // (world_size * micro_batch)
    if accumulation is None:
        accumulation = expected
    if accumulation < 1 or world_size * micro_batch * accumulation != GLOBAL_BATCH:
        raise ValueError("require world_size * micro_batch * accumulation == 64")
    return micro_batch, accumulation


def local_epoch_order(
    sample_count: int,
    epoch: int,
    rank: int,
    world_size: int,
    micro_batch: int,
    accumulation: int,
    *,
    steps: int | None = None,
) -> list[int]:
    """Shard each consecutive global batch; never pad or repeat samples.

    Reconstruct a global update by visiting accumulation slots, then ranks,
    then each rank's micro-batch. Its members and order equal legacy's next
    sixteen entries of randperm(seed=60+epoch).
    """
    batch_geometry(world_size, micro_batch, accumulation)
    if not 0 <= rank < world_size:
        raise ValueError("rank is outside world size")
    complete_steps = sample_count // GLOBAL_BATCH
    max_steps = (sample_count + GLOBAL_BATCH - 1) // GLOBAL_BATCH
    steps = max_steps if steps is None else steps
    if steps < 1 or steps > max_steps:
        raise ValueError("not enough distinct samples for the requested global updates")
    generator = torch.Generator().manual_seed(SEED + epoch)
    global_order = torch.randperm(sample_count, generator=generator).tolist()
    order: list[int] = []
    for update in range(steps):
        start = update * GLOBAL_BATCH
        if start + GLOBAL_BATCH > sample_count:
            remaining = sample_count - start
            if accumulation != 1 or remaining % world_size:
                raise ValueError("Final batch requires equal DDP shards")
            per_rank = remaining // world_size
            offset = start + rank * per_rank
            order.extend(global_order[offset : offset + per_rank])
            continue
        for slot in range(accumulation):
            offset = start + (slot * world_size + rank) * micro_batch
            order.extend(global_order[offset : offset + micro_batch])
    return order


def setup_runtime(args: argparse.Namespace) -> Runtime:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    device_name = args.device
    if args.command == "check":
        device_name = "cpu"
    elif device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA-compatible PyTorch is unavailable; install the platform runtime separately")
        if local >= torch.cuda.device_count():
            raise RuntimeError(f"LOCAL_RANK={local} exceeds the visible device count")
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")
        torch.set_num_threads(args.cpu_threads)
    backend = args.backend or ("nccl" if device.type == "cuda" else "gloo")
    if backend == "nccl" and device.type != "cuda":
        raise ValueError("NCCL requires a CUDA-compatible device")
    if world > 1:
        dist.init_process_group(backend, timeout=timedelta(seconds=args.dist_timeout))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(args.deterministic)
    return Runtime(rank, world, local, device, backend if world > 1 else None)


def seed_all(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def rank0_call(runtime: Runtime, function: Callable[[], Any]) -> Any:
    """Broadcast rank-zero results or a Python error to waiting workers."""
    packet: list[Any] = [None]
    if runtime.primary:
        try:
            packet[0] = {"ok": True, "value": function()}
        except Exception:
            packet[0] = {"ok": False, "error": traceback.format_exc()}
    if runtime.world_size > 1:
        dist.broadcast_object_list(packet, src=0)
    if not packet[0]["ok"]:
        raise RuntimeError("Rank-zero operation failed:\n" + packet[0]["error"])
    return packet[0]["value"]


def collective_call(runtime: Runtime, function: Callable[[], Any]) -> Any:
    """Report setup failures on any rank before entering the next phase."""
    error, result = None, None
    try:
        result = function()
    except Exception:
        error = traceback.format_exc()
    errors: list[Any] = [None] * runtime.world_size
    if runtime.world_size > 1:
        dist.all_gather_object(errors, error)
    else:
        errors = [error]
    for rank, problem in enumerate(errors):
        if problem:
            raise RuntimeError(f"Rank {rank} failed:\n{problem}")
    return result


def assert_finite(value: torch.Tensor, label: str, runtime: Runtime) -> None:
    bad = (~torch.isfinite(value).all()).to(dtype=torch.int32, device=runtime.device)
    if runtime.world_size > 1:
        dist.all_reduce(bad, op=dist.ReduceOp.MAX)
    if bad.item():
        raise FloatingPointError(f"Non-finite {label}; last committed epoch remains resumable")


def unwrapped(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def autocast_context(runtime: Runtime, precision: str):
    return torch.amp.autocast("cuda", dtype=torch.float16) if precision == "fp16" else nullcontext()


def resolve_precision(args: argparse.Namespace, runtime: Runtime) -> str:
    precision = args.precision or ("fp16" if runtime.device.type == "cuda" else "fp32")
    if precision == "fp16" and runtime.device.type != "cuda":
        raise ValueError("FP16 execution requires a verified CUDA-compatible runtime")
    return precision




def dataset_format(args) -> str:
    if not (args.data_root / 'splits' / 'train_samples.csv').is_file():
        raise ValueError('This entry requires the 2018-2019 raw CSI CSV/NPZ dataset or its raw v2 cache')
    return 'npz'


def load_datasets(args, *, formal: bool):
    selection_path = getattr(args, 'download_selection', None)
    selected_indices = None
    if selection_path is not None:
        if formal or not getattr(args, 'allow_small_data', False):
            raise ValueError('Download selections are for check/smoke with --allow-small-data only')
        selection = json.loads(Path(selection_path).read_text(encoding='utf-8'))
        if selection.get('mode') != 'preview' or selection.get('state') != 'complete':
            raise ValueError('Expected an explicit preview download selection')
        from data.dataset_csi import file_hash
        selected_indices = selection['sample_indices']
        for split in ('train', 'validation', 'test'):
            path = args.data_root / 'splits' / (split + '_samples.csv')
            if file_hash(path) != selection['csv_sha256'][split]:
                raise ValueError(f'Download selection refers to a different {split} CSV')
    args._selected_train_indices = selected_indices['train'] if selected_indices else None
    if formal and (args.data_root / 'download_selection.json').is_file():
        downloaded = json.loads((args.data_root / 'download_selection.json').read_text(encoding='utf-8'))
        if downloaded.get('state') != 'complete' or downloaded.get('mode') != 'full':
            raise ValueError('Formal training/evaluation requires the completed full OSS download, not a preview')
    dataset_format(args)
    from data.dataset_csi import load_npz_splits
    datasets = load_npz_splits(args.data_root.expanduser().resolve(), selected_indices=selected_indices)
    train, validation, test = datasets
    roles = {"train": train.indices_for_role("train"),
             "validation": validation.indices_for_role("validation"),
             "test": test.indices_for_role("test"), "fixed_visual": []}
    for name, dataset in zip(("train", "validation", "test"), datasets):
        if roles[name] != list(range(len(dataset))):
            raise ValueError(f"NPZ {name} must consume the complete existing CSV, in its canonical order")
    counts = {name: len(indices) for name, indices in roles.items()}
    if min(counts[name] for name in ("train", "validation", "test")) < 1:
        raise ValueError("Existing NPZ CSV splits must all be nonempty")
    if formal or not getattr(args, "allow_small_data", False):
        expected = {"train": 177380, "validation": 35482, "test": 35476, "fixed_visual": 0}
        if counts != expected:
            raise ValueError(f"Noncanonical NPZ counts {counts}; fixture check/smoke requires --allow-small-data")
    return datasets, roles, counts


def representative_checks(datasets, roles) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name, dataset in zip(("train", "validation", "test"), datasets):
        indices = sorted(set((0, len(dataset) // 2, len(dataset) - 1)))
        for position in indices:
            low, high = dataset[position]
            if tuple(low.shape) != (1, 32, 32) or tuple(high.shape) != (1, 128, 128):
                raise ValueError(f"Unexpected shapes in {name} sample {position}")
            if low.dtype != torch.float32 or high.dtype != torch.float32:
                raise TypeError(f"Expected float32 in {name} sample {position}")
            if not torch.isfinite(low).all() or not torch.isfinite(high).all():
                raise FloatingPointError(f"Non-finite prepared sample: {name}/{position}")
            if name != "train" and hasattr(dataset, "clear_sky_patch"):
                clear = dataset.clear_sky_patch(position)
                if clear.shape != (128, 128) or not np.isfinite(clear).all():
                    raise ValueError(f"Invalid clear GHI in {name}/{position}")
            if hasattr(dataset, "physical_ghi_patch") and getattr(dataset, "storage_format", "") != "2018_2019_npy_cache":
                physical = dataset.physical_ghi_patch(position)
                if physical.shape != (128, 128) or not np.isfinite(physical).all():
                    raise ValueError(f"Invalid stored FY-3C GHI in {name}/{position}")
        report[name] = {"length": len(dataset), "checked_positions": indices, "finite_float32": True}
    return report


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_fingerprint(root: Path) -> dict[str, Any]:
    """Hash actual arrays as well as manifests; reject data edits on resume."""
    result: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        files = ["samples.json", "metadata.json", "lr_csi.npy", "hr_csi.npy"]
        if split != "train":
            files.append("clear_ghi.npy")
        for name in files:
            path = root / split / name
            result[f"{split}/{name}"] = {"bytes": path.stat().st_size, "sha256": hash_file(path)}
    return {"fingerprint_scope": "full SHA256 of prepared manifests, metadata and NPY contents", "files": result,
            "preview_only": False}


def cloud_data_identity(args):
    if dataset_format(args) == "npz":
        from data.dataset_csi import npz_data_identity
        return npz_data_identity(args.data_root.expanduser().resolve())
    return data_fingerprint(args.data_root.expanduser().resolve())


def loader(dataset, indices, batch_size: int, workers: int) -> DataLoader:
    return DataLoader(
        Subset(dataset, indices), batch_size=batch_size, shuffle=False,
        drop_last=False, num_workers=workers, pin_memory=False,
        persistent_workers=False, prefetch_factor=1 if workers else None,
        generator=torch.Generator().manual_seed(SEED),
    )


def validation_metrics(model, dataset, indices, runtime, precision, batch_size, workers, *, data_range, bicubic=False):
    """Fixed TRAIN reference range for PSNR/SSIM; no value clipping."""
    psnrs, ssims, stds = [], [], []
    l1_sum, l1_pixels = 0.0, 0
    predicted_gradient, observed_gradient = 0.0, 0.0
    was_training = model.training if model is not None else False
    if model is not None:
        model.eval()
    try:
        with torch.inference_mode():
            for low, target in loader(dataset, indices, batch_size, workers):
                if bicubic:
                    prediction = helpers.bicubic_batch(low)
                else:
                    target = target.to(runtime.device)
                    with autocast_context(runtime, precision):
                        prediction = model(low.to(runtime.device)).float()
                if not torch.isfinite(prediction).all():
                    raise FloatingPointError("Non-finite validation prediction")
                l1_sum += float((prediction - target).abs().sum().item())
                l1_pixels += target.numel()
                batch_psnr, batch_ssim = csi_batch_metrics(prediction, target, data_range=data_range)
                psnrs.extend(batch_psnr)
                ssims.extend(batch_ssim)
                stds.extend(prediction.std(dim=(1, 2, 3)).cpu().tolist())
                predicted_gradient += float(helpers.sobel_magnitude(prediction).sum().item())
                observed_gradient += float(helpers.sobel_magnitude(target).sum().item())
    finally:
        if model is not None:
            model.train(was_training)
    if l1_pixels == 0:
        raise ValueError("Empty validation data")
    result = {"l1": l1_sum / l1_pixels, "psnr": float(np.mean(psnrs)), "ssim": float(np.mean(ssims)),
              "grad": predicted_gradient / max(observed_gradient, 1e-12)}
    if not bicubic:
        result["output_std"] = float(np.mean(stds))
    if any(math.isnan(value) for value in result.values()) or any(not math.isfinite(value) for key, value in result.items() if key != 'psnr'):
        raise FloatingPointError("Non-finite validation metric")
    return result


def build_training_objects(args, runtime, precision):
    seed_all(SEED, runtime.device)
    model = helpers.build_model(runtime.device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=BETAS, weight_decay=0.0, fused=use_fused_adam(args, runtime))
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))
    return model, ema, optimizer, scaler


def use_fused_adam(args, runtime):
    if getattr(args, 'fused_adam', False):
        if runtime.device.type != 'cuda':
            raise ValueError('Fused Adam requires a verified accelerator implementation')
        return True
    if getattr(args, 'plain_adam', False) or runtime.device.type != 'cuda':
        return False
    # Preserve the original implementation on known NVIDIA devices. Do not
    # infer a vendor PPU's fused-kernel support merely from CUDA API names.
    return 'NVIDIA' in torch.cuda.get_device_name(runtime.device).upper()


def wrap_training_model(model, runtime):
    # Every rank has successfully allocated its model before DDP's constructor
    # starts collectives; a setup OOM can therefore be reported to all workers.
    if runtime.world_size > 1:
        kwargs = {"device_ids": [runtime.local_rank], "output_device": runtime.local_rank} if runtime.device.type == "cuda" else {}
        model = DistributedDataParallel(model, broadcast_buffers=True, **kwargs)
    seed_all(SEED + runtime.rank, runtime.device)
    return model


@torch.no_grad()
def update_ema(ema, model):
    ema_named = list(ema.named_parameters())
    model_named = list(unwrapped(model).named_parameters())
    if [name for name, _ in ema_named] != [name for name, _ in model_named]:
        raise RuntimeError("EMA/model parameter names differ")
    # Plain operations work on more vendor backends than private foreach APIs.
    for (_, averaged), (_, current) in zip(ema_named, model_named):
        averaged.mul_(EMA_DECAY).add_(current, alpha=1.0 - EMA_DECAY)


def train_updates(model, ema, optimizer, scaler, dataset, epoch, micro, accumulation, steps, runtime, precision, workers,
                  progress=None, progress_interval=100):
    order = local_epoch_order(len(dataset), epoch, runtime.rank, runtime.world_size, micro, accumulation, steps=steps)
    batches = iter(loader(dataset, order, micro, workers))
    model.train()
    absolute_error = torch.zeros((), dtype=torch.float64, device=runtime.device)
    pixel_count = 0
    parameters = list(unwrapped(model).parameters())
    skipped, successful, consecutive_skips = 0, 0, 0
    for update in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        for slot in range(accumulation):
            low, target = next(batches)
            low, target = low.to(runtime.device), target.to(runtime.device)
            assert_finite(low, "training input", runtime)
            assert_finite(target, "training target", runtime)
            syncing = model.no_sync() if isinstance(model, DistributedDataParallel) and slot + 1 < accumulation else nullcontext()
            with syncing:
                with autocast_context(runtime, precision):
                    prediction = model(low)
                    batch_l1 = F.l1_loss(prediction, target)
                    loss = batch_l1 / accumulation
                assert_finite(loss, "training loss", runtime)
                scaler.scale(loss).backward()
            absolute_error.add_(batch_l1.detach().to(torch.float64) * target.numel())
            pixel_count += target.numel()
        scaler.unscale_(optimizer)
        gradient_maxima = [parameter.grad.detach().abs().max() for parameter in parameters if parameter.grad is not None]
        if not gradient_maxima:
            raise RuntimeError("No training gradients were produced")
        finite_gradients = torch.isfinite(torch.stack(gradient_maxima)).all().to(dtype=torch.int32, device=runtime.device)
        if runtime.world_size > 1:
            dist.all_reduce(finite_gradients, op=dist.ReduceOp.MIN)
        if not finite_gradients.item():
            if precision != "fp16":
                raise FloatingPointError("Non-finite FP32 gradients; last committed epoch remains resumable")
            # All ranks make the same skip/backoff decision, including ranks
            # whose local unscale found no overflow. Never update EMA on a skip.
            scaler.update(new_scale=scaler.get_scale() * scaler.get_backoff_factor())
            skipped += 1
            consecutive_skips += 1
            if runtime.primary:
                print(f"AMP overflow: skipped optimizer and EMA update {update}; scale={scaler.get_scale()}", flush=True)
            if consecutive_skips >= 16:
                raise FloatingPointError("16 consecutive FP16 overflows; smoke/precision configuration needs review")
        else:
            scaler.step(optimizer)
            scaler.update()
            update_ema(ema, model)
            successful += 1
            consecutive_skips = 0
        if runtime.primary and (update == 1 or update % 100 == 0 or update == steps):
            print(f"epoch={epoch} update={update}/{steps} rank0_local_l1={absolute_error.item() / max(pixel_count, 1):.7f}", flush=True)
        if progress is not None and (update == 1 or update % progress_interval == 0 or update == steps):
            snapshot = torch.tensor([absolute_error.item(), float(pixel_count)], dtype=torch.float64, device=runtime.device)
            if runtime.world_size > 1:
                dist.all_reduce(snapshot, op=dist.ReduceOp.SUM)
            running_l1 = float((snapshot[0] / snapshot[1]).item())
            # Every rank reaches the same callback; rank0_call propagates any
            # drawing error before a later distributed collective can hang.
            progress(epoch, update, steps, running_l1)
    total = torch.tensor([absolute_error.item(), float(pixel_count)], dtype=torch.float64, device=runtime.device)
    if runtime.world_size > 1:
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    expected = min(steps * GLOBAL_BATCH, len(dataset)) * 128 * 128
    if int(total[1].item()) != expected:
        raise RuntimeError(f"Global epoch pixel count {total[1].item()} != {expected}")
    return float((total[0] / total[1]).item()), steps, {"optimizer_steps": successful, "amp_skipped_updates": skipped}


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def rng_state(runtime):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(runtime.device) if runtime.device.type == "cuda" else None}


def restore_rng(state, runtime):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if runtime.device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], runtime.device)


def save_state(path, model, ema, optimizer, scaler, history, best, config, identity, runtime):
    states: list[Any] = [None] * runtime.world_size
    if runtime.world_size > 1:
        dist.all_gather_object(states, rng_state(runtime))
    else:
        states = [rng_state(runtime)]

    def write():
        helpers.atomic_torch_save(path, {
            "format_version": FORMAT_VERSION, "kind": "drct_cloud_resume", "world_size": runtime.world_size,
            "model": cpu_tree(unwrapped(model).state_dict()), "params_ema": cpu_tree(ema.state_dict()),
            "optimizer": cpu_tree(optimizer.state_dict()), "scaler": scaler.state_dict(),
            "rng_by_rank": states, "config": config, "data_identity": identity,
            "history": history, "best": best, "next_epoch": len(history) + 1,
            "global_step": len(history) * config["steps_per_epoch"],
        })
    rank0_call(runtime, write)


def write_history(output: Path, history: list[dict[str, Any]]) -> None:
    """The checkpoint is authoritative; rebuild the CSV, never append ahead of it."""
    path = output / "training_log.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        fields = list(history[0]) if history else ["epoch", "global_step", "train_l1_mae", "val_l1_mae", "val_psnr", "val_ssim", "val_grad", "output_std", "learning_rate", "epoch_seconds", "elapsed_seconds", "best_real"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    os.replace(temporary, path)


def inference_checkpoint(path, ema, epoch, global_step, metrics, config, label):
    helpers.atomic_torch_save(path, {"params_ema": helpers.cpu_state_dict(ema), "epoch": epoch,
        "global_step": global_step, "metrics": metrics, "network": helpers.load_network_options(),
        "config": config, "label": label, "format_version": FORMAT_VERSION})


def run_config(args, runtime, precision, micro, accumulation, counts):
    return {
        "network": helpers.load_network_options(), "seed": SEED, "global_batch": GLOBAL_BATCH,
        "micro_batch": micro, "accumulation": accumulation, "world_size": runtime.world_size,
        "steps_per_epoch": (counts["train"] + GLOBAL_BATCH - 1) // GLOBAL_BATCH,
        "learning_rate": LR, "betas": list(BETAS), "weight_decay": 0.0, "scheduler": None,
        "augmentation": False, "loss": "L1 / raw-CSI MAE; no scaling or clipping", "ema_decay": EMA_DECAY,
        "early_stop": {"min_delta": 1e-4, "patience": 5, "metric": "validation epoch L1"},
        "precision": precision, "device_type": runtime.device.type, "backend": runtime.backend,
        "fused_adam": use_fused_adam(args, runtime), "ema_implementation": "plain mul_/add_", "workers": args.workers,
        "representation": "raw_csi", "normalization": "none", "value_clipping": False,
        "validation_batch": args.validation_batch, "visuals": args.visuals,
        "deterministic_algorithms": args.deterministic,
        "data_counts": counts, "dataset_format": dataset_format(args),
        "split_protocol": ("existing sample-level random 5:1:1 CSV splits, seed 20261024; no resplitting"
                           if dataset_format(args) == "npz" else "original fixed date-held-out prepared splits"),
        "visual_selection": ("first five canonical validation CSV rows; no test samples"
                             if dataset_format(args) == "npz" else "original five fixed_visual records"),
        "amp_overflow_policy": "synchronized optimizer/EMA skip and GradScaler backoff; FP32 nonfinite is fatal",
        "data_root": str(args.data_root.expanduser().resolve()),
        "initialization": "original model initialization; no pretrained weights",
        "vendor_ppu_verified": False,
        "metric_reference": args._metric_reference,
        "visualization": args._plot_options,
    }


def resolve_metric_reference(args, destination, *, saved_reference=None):
    """Freeze a TRAIN-only or explicitly declared range; never fit on test data."""
    from data.dataset_csi import file_hash, read_rows
    from utils.metrics import fit_training_reference, load_metric_reference
    selected = getattr(args, '_selected_train_indices', None)
    train_path = args.data_root / 'splits' / 'train_samples.csv'
    expected_hash = file_hash(train_path)
    supplied_range = getattr(args, 'metric_data_range', None)
    supplied_file = getattr(args, 'metric_reference', None)
    if supplied_file is not None and supplied_range is not None:
        raise ValueError('Choose --metric-reference or --metric-data-range, not both')
    existing = Path(destination) / 'metric_reference.json' if destination is not None else None
    if supplied_file is not None:
        reference = load_metric_reference(supplied_file, args.data_root, train_indices=selected)
    elif supplied_range is not None:
        if not math.isfinite(supplied_range) or supplied_range <= 0:
            raise ValueError('Explicit CSI metric data range must be finite and positive')
        reference = {'representation': 'raw_csi', 'normalization': 'none',
                     'policy': 'user_declared_fixed_reference', 'source_split': 'train',
                     'kind': 'explicit_reference_not_a_physical_bound',
                     'data_range': supplied_range, 'raw_data_range': supplied_range,
                     'raw_min': None, 'raw_max': None,
                     'training_csv_sha256': expected_hash,
                     'training_samples': len(selected) if selected else len(read_rows(args.data_root, 'train')),
                     'selected_train_indices': selected, 'preview_only': selected is not None}
    elif saved_reference is not None:
        reference = saved_reference
    elif existing is not None and existing.is_file():
        reference = load_metric_reference(existing, args.data_root, train_indices=selected)
    else:
        print('Determining the fixed metric reference from TRAIN HR CSI only ...', flush=True)
        reference = fit_training_reference(args.data_root, train_indices=selected)
    if reference.get('representation') != 'raw_csi' or reference.get('normalization') != 'none':
        raise ValueError('Metric reference must describe raw CSI without normalization')
    if reference.get('training_csv_sha256') != expected_hash or reference.get('selected_train_indices') != selected:
        raise ValueError('Metric reference belongs to a different training CSV/preview selection')
    if not math.isfinite(float(reference['data_range'])) or float(reference['data_range']) <= 0:
        raise ValueError('CSI metric reference requires a finite positive range')
    if saved_reference is not None and reference != saved_reference:
        raise ValueError('Evaluation must use the metric reference saved in the checkpoint')
    if existing is not None:
        if existing.exists():
            prior = json.loads(existing.read_text(encoding='utf-8'))
            if prior != reference:
                raise ValueError('An existing run has a different frozen metric reference')
        else:
            helpers.atomic_json(existing, reference)
    return reference


def convergence_state(history):
    result = helpers.derive_convergence_state(history)
    result['metric'] = 'epoch-average validation L1 / raw-CSI MAE'
    return result


def save_raw_montage(model, samples, destination, device, epoch, reference):
    """Display only: colour limits do not change the model or saved CSI values."""
    from PIL import Image, ImageDraw
    low = torch.stack([sample[0] for sample in samples])
    high = torch.stack([sample[1] for sample in samples])
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        prediction = model(low.to(device)).float().cpu()
    model.train(was_training)
    baseline = helpers.bicubic_batch(low)
    lower = reference.get('raw_min')
    lower = 0.0 if lower is None else float(lower)
    span = float(reference['data_range'])
    width, height, gap = 128, 128, 8
    canvas = Image.new('RGB', (3 * (width + gap) + gap, len(samples) * (height + gap) + 48), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 4), f'Epoch {epoch}; raw CSI; colour reference [{lower:.4g}, {lower+span:.4g}]', fill='black')
    for column, (label, arrays) in enumerate([('Bicubic', baseline), ('DRCT', prediction), ('FY-3C', high)]):
        draw.text((gap + column * (width + gap), 24), label, fill='black')
        for row, array in enumerate(arrays[:, 0].numpy()):
            colour = np.rint(np.clip((array - lower) / span, 0, 1) * 255).astype(np.uint8)
            panel = Image.fromarray(colour).convert('RGB')
            canvas.paste(panel, (gap + column * (width + gap), 44 + row * (height + gap)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def tiny_pipeline_check(train, destination: Path, *, full_model=False) -> dict[str, Any]:
    """CPU updates and exact next-update restore, optional full original model."""
    options = helpers.load_network_options()
    if not full_model:
        options.update(depths=[1], embed_dim=24, gc=8, num_heads=[2], drop_path_rate=0.0)
    seed_all(SEED, torch.device("cpu"))
    model = helpers.DRCT(**options)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=BETAS, weight_decay=0.0)
    low, target = train[0]
    low, target = low.unsqueeze(0), target.unsqueeze(0)

    def step(network, optim):
        network.train()
        optim.zero_grad(set_to_none=True)
        prediction = network(low)
        if prediction.shape != target.shape:
            raise ValueError("Tiny pipeline model violated the 32-to-128 shape contract")
        loss = F.l1_loss(prediction, target)
        loss.backward()
        if not torch.isfinite(loss) or any(not torch.isfinite(p.grad).all() for p in network.parameters() if p.grad is not None):
            raise FloatingPointError("Tiny CPU pipeline produced non-finite loss/gradients")
        optim.step()
        return float(loss.detach())

    first_loss = step(model, optimizer)
    temporary = destination / "pipeline_checkpoint.pth"
    helpers.atomic_torch_save(temporary, {"kind": "cpu_pipeline_check_only", "representation": "raw_csi", "network": options,
                                        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": torch.get_rng_state()})
    state = torch.load(temporary, map_location="cpu", weights_only=False)
    restored = helpers.DRCT(**options)
    restored.load_state_dict(state["model"], strict=True)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=LR, betas=BETAS, weight_decay=0.0)
    restored_optimizer.load_state_dict(state["optimizer"])
    model.eval()
    restored.eval()
    with torch.inference_mode():
        if not torch.equal(model(low), restored(low)):
            raise RuntimeError("Tiny checkpoint prediction roundtrip differs")
    torch.set_rng_state(state["rng"])
    reference_loss = step(model, optimizer)
    torch.set_rng_state(state["rng"])
    restored_loss = step(restored, restored_optimizer)
    if reference_loss != restored_loss or any(not torch.equal(tensor, restored.state_dict()[name]) for name, tensor in model.state_dict().items()):
        raise RuntimeError("Tiny restored next update differs from uninterrupted reference")
    log_path = destination / 'pipeline_log.csv'
    with log_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['update', 'raw_csi_l1'])
        writer.writeheader()
        writer.writerows([{'update': 1, 'raw_csi_l1': first_loss}, {'update': 2, 'raw_csi_l1': restored_loss}])
    return {"label": ("FULL ORIGINAL MODEL CPU PIPELINE; not a scientific experiment" if full_model else "TINY CPU PIPELINE ONLY; not full-model/scientific training"), "options": options,
            "parameters": sum(p.numel() for p in model.parameters()), "input_shape": list(low.shape),
            "output_shape": list(target.shape), "updates": 2, "first_l1": first_loss,
            "second_l1": restored_loss, "checkpoint_prediction_exact": True, "restored_next_update_exact": True,
            "saved_checkpoint": str(temporary), "saved_log": str(log_path), "checkpoint_after_update": 1}


def run_check(args, runtime):
    def check():
        datasets, roles, counts = load_datasets(args, formal=False)
        destination = args.output_dir / "checks" / (time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
        destination.mkdir(parents=True, exist_ok=False)
        samples = representative_checks(datasets, roles)
        preview_batch = next(iter(loader(datasets[0], list(range(min(2, len(datasets[0])))), 2, args.workers)))
        if preview_batch[0].shape[1:] != (1, 32, 32) or preview_batch[1].shape[1:] != (1, 128, 128):
            raise ValueError('DataLoader violated the LR/HR batch shape contract')
        reference = resolve_metric_reference(args, destination)
        seed_all(SEED, torch.device("cpu"))
        full = helpers.build_model(torch.device("cpu"))
        full_parameters = sum(parameter.numel() for parameter in full.parameters())
        del full
        gc.collect()
        report = {"passed": True, "mode": "CPU pipeline check", "data_counts": counts,
                  "dataset_format": dataset_format(args),
                  "storage_formats": [getattr(dataset, "storage_format", "prepared_npy") for dataset in datasets],
                  "representative_samples": samples, "full_original_model_parameters": full_parameters,
                  "dataloader_shapes": [list(value.shape) for value in preview_batch],
                  "metric_reference": reference,
                  "full_model_forward_run": args.full_model,
                  "model_pipeline": tiny_pipeline_check(datasets[0], destination, full_model=args.full_model),
                  "vendor_ppu_verified": False}
        helpers.atomic_json(destination / "check_report.json", report)
        print(json.dumps(report, indent=2), flush=True)
        return str(destination / "check_report.json")
    return rank0_call(runtime, check)


def run_smoke(args, runtime, precision):
    datasets, roles, counts = collective_call(runtime, lambda: load_datasets(args, formal=False))
    collective_call(runtime, lambda: representative_checks(datasets, roles))
    micro, accumulation = batch_geometry(runtime.world_size, args.micro_batch, args.accumulation)
    if args.steps * GLOBAL_BATCH > len(datasets[0]):
        raise ValueError("Smoke steps need steps*16 distinct training samples; lower --steps")
    destination = Path(rank0_call(runtime, lambda: str(args.output_dir / "smoke" / (time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]))))
    rank0_call(runtime, lambda: destination.mkdir(parents=True, exist_ok=False))
    reference = rank0_call(runtime, lambda: resolve_metric_reference(args, destination))
    args._metric_reference = reference
    model, ema, optimizer, scaler = collective_call(runtime, lambda: build_training_objects(args, runtime, precision))
    model = wrap_training_model(model, runtime)
    started = time.time()
    loss, updates, update_counts = train_updates(model, ema, optimizer, scaler, datasets[0], 1, micro, accumulation, args.steps, runtime, precision, args.workers)
    if update_counts["optimizer_steps"] == 0:
        raise RuntimeError("Smoke had no successful optimizer update; increase --steps or review precision")
    metrics = rank0_call(runtime, lambda: validation_metrics(ema, datasets[1], roles["validation"][:32], runtime, precision, args.validation_batch, args.workers, data_range=reference['data_range']))
    states: list[Any] = [None] * runtime.world_size
    if runtime.world_size > 1:
        dist.all_gather_object(states, rng_state(runtime))
    else:
        states = [rng_state(runtime)]

    def finish():
        path = destination / "smoke_checkpoint.pth"
        helpers.atomic_torch_save(path, {"kind": "smoke_only", "model": cpu_tree(unwrapped(model).state_dict()),
            "params_ema": cpu_tree(ema.state_dict()), "optimizer": cpu_tree(optimizer.state_dict()),
            "scaler": scaler.state_dict(), "rng_by_rank": states})
        restored = torch.load(path, map_location="cpu", weights_only=False)
        if any(not torch.equal(tensor.cpu(), restored["model"][name]) for name, tensor in unwrapped(model).state_dict().items()):
            raise RuntimeError("Full-model smoke checkpoint weights failed roundtrip")
        report = {"passed": True, "label": "FULL original model device smoke; separate from formal experiment",
            "parameters": sum(p.numel() for p in unwrapped(model).parameters()), "device": str(runtime.device),
            "world_size": runtime.world_size, "precision": precision, "backend": runtime.backend,
            "global_batch": GLOBAL_BATCH, "micro_batch": micro, "accumulation": accumulation,
            "updates": updates, "global_train_l1": loss, "validation_up_to_32": metrics,
            **update_counts, "dataset_format": dataset_format(args),
            "seconds": time.time() - started, "data_counts": counts, "small_data_allowed": args.allow_small_data,
            "checkpoint_weights_exact": True, "vendor_ppu_verified": False}
        report['metric_reference'] = reference
        helpers.atomic_json(destination / "smoke_report.json", report)
        print(json.dumps(report, indent=2), flush=True)
        return str(destination / "smoke_report.json")
    return rank0_call(runtime, finish)


def run_train(args, runtime, precision):
    datasets, roles, counts = collective_call(runtime, lambda: load_datasets(args, formal=True))
    collective_call(runtime, lambda: representative_checks(datasets, roles))
    micro, accumulation = batch_geometry(runtime.world_size, args.micro_batch, args.accumulation)
    reference = rank0_call(runtime, lambda: resolve_metric_reference(args, args.output_dir))
    args._metric_reference = reference
    config = run_config(args, runtime, precision, micro, accumulation, counts)
    steps_per_epoch = config["steps_per_epoch"]
    output = args.output_dir.expanduser().resolve()
    latest = output / "checkpoints" / "resume_latest.pth"

    def prepare():
        if args.resume:
            if not latest.is_file():
                raise FileNotFoundError(f"No cloud resume checkpoint: {latest}")
        elif latest.exists() or (output / "training_log.csv").exists() or (output / "training_completion.json").exists() or (output / "run_config.json").exists():
            raise FileExistsError("Formal experiment already exists; use --resume or another --output-dir")
        output.mkdir(parents=True, exist_ok=True)
        (output / "checkpoints").mkdir(exist_ok=True)
        print("Checking dataset identity (scope is recorded in the checkpoint) ...", flush=True)
        identity = cloud_data_identity(args)
        if identity.get("preview_only", False):
            raise ValueError("Preview-only data cannot be used for formal training")
        return identity
    identity = rank0_call(runtime, prepare)
    model, ema, optimizer, scaler = collective_call(runtime, lambda: build_training_objects(args, runtime, precision))
    model = wrap_training_model(model, runtime)
    history: list[dict[str, Any]] = []
    best: dict[str, Any] = {"psnr": -math.inf, "ssim": -math.inf, "epoch": 0, "checkpoint": None}
    if args.resume:
        def restore():
            checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
            if checkpoint.get("format_version") != FORMAT_VERSION or checkpoint.get("kind") != "drct_cloud_resume":
                raise ValueError("Only this runner's new cloud resume format is supported")
            if checkpoint.get("world_size") != runtime.world_size or checkpoint.get("config") != config:
                raise ValueError("Resume world size/config changed; use the original settings")
            if checkpoint.get("data_identity") != identity:
                raise ValueError("Prepared dataset changed since checkpoint creation")
            if checkpoint["next_epoch"] != len(checkpoint["history"]) + 1:
                raise ValueError("Checkpoint epoch/history mismatch")
            if checkpoint["global_step"] != len(checkpoint["history"]) * steps_per_epoch:
                raise ValueError("Checkpoint global step/history mismatch")
            unwrapped(model).load_state_dict(checkpoint["model"], strict=True)
            ema.load_state_dict(checkpoint["params_ema"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            helpers.optimizer_to_device(optimizer, runtime.device)
            scaler.load_state_dict(checkpoint["scaler"])
            restore_rng(checkpoint["rng_by_rank"][runtime.rank], runtime)
            return checkpoint["history"], checkpoint["best"]
        history, best = collective_call(runtime, restore)
    rank0_call(runtime, lambda: helpers.atomic_json(output / "run_config.json", {**config, "data_counts": counts}))
    rank0_call(runtime, lambda: write_history(output, history))
    if not args.resume:
        save_state(latest, model, ema, optimizer, scaler, history, best, config, identity, runtime)
    baseline_path = output / "bicubic_baseline.json"

    def baseline_value():
        if baseline_path.is_file():
            with baseline_path.open(encoding="utf-8") as handle:
                return json.load(handle)
        baseline = validation_metrics(None, datasets[1], roles["validation"], runtime, "fp32", args.validation_batch, args.workers, data_range=reference['data_range'], bicubic=True)
        helpers.atomic_json(baseline_path, baseline)
        return baseline
    baseline = rank0_call(runtime, baseline_value)
    def draw_status(status, live_points=None):
        from utils.visualization import training_progress
        helpers.atomic_json(output / "training_status.json", status)
        training_progress(history, baseline, output / "training_progress.png", status=status, live_points=live_points)
    if args._plot_options["training_curves"]:
        rank0_call(runtime, lambda: draw_status({"phase": "ready", "epoch": len(history) + 1}))
    elapsed = sum(float(row["epoch_seconds"]) for row in history)
    epoch = len(history) + 1
    while True:
        convergence = convergence_state(history)
        if convergence["should_stop"]:
            break
        started = time.time()
        live_points = []
        def record_progress(epoch_number, update, total_steps, running_l1):
            live_points.append((epoch_number - 1 + update / total_steps, running_l1))
            status = {"phase": "training; validation pending", "epoch": epoch_number, "update": update,
                      "steps": total_steps, "running_l1": running_l1, "partial_epoch": True}
            return rank0_call(runtime, lambda: draw_status(status, live_points))
        loss, updates, update_counts = train_updates(model, ema, optimizer, scaler, datasets[0], epoch, micro, accumulation, steps_per_epoch, runtime, precision, args.workers,
            progress=None,
            progress_interval=args._plot_options["refresh_updates"])
        if update_counts["optimizer_steps"] == 0:
            raise RuntimeError("No optimizer update succeeded in this epoch")
        metrics = rank0_call(runtime, lambda: validation_metrics(ema, datasets[1], roles["validation"], runtime, precision, args.validation_batch, args.workers, data_range=reference['data_range']))
        seconds = time.time() - started
        elapsed += seconds
        is_best = metrics["psnr"] > float(best["psnr"]) or (math.isclose(metrics["psnr"], float(best["psnr"]), abs_tol=1e-12) and metrics["ssim"] > float(best["ssim"]))
        if is_best:
            best = {"psnr": metrics["psnr"], "ssim": metrics["ssim"], "epoch": epoch,
                    "checkpoint": f"checkpoints/best_epoch_{epoch:03d}.pth"}
            rank0_call(runtime, lambda: inference_checkpoint(output / best["checkpoint"], ema, epoch, epoch * updates, metrics, config, "best validation PSNR"))
        history.append({"epoch": epoch, "global_step": epoch * updates, "train_l1_mae": f"{loss:.9f}", "val_l1_mae": f"{metrics['l1']:.9f}",
            "val_psnr": f"{metrics['psnr']:.6f}", "val_ssim": f"{metrics['ssim']:.6f}",
            "val_grad": f"{metrics['grad']:.6f}", "output_std": f"{metrics['output_std']:.6f}",
            "learning_rate": f"{LR:.8g}", "epoch_seconds": f"{seconds:.1f}", "elapsed_seconds": f"{elapsed:.1f}", "best_real": int(is_best)})
        history[-1].update(update_counts)
        save_state(latest, model, ema, optimizer, scaler, history, best, config, identity, runtime)

        def epoch_outputs():
            write_history(output, history)
            if args._plot_options["training_curves"]:
                draw_status({"phase": "epoch committed", "epoch": epoch}, live_points)
            if args._plot_options["training_comparison"]:
                from utils.visualization import training_comparison
                training_comparison(ema, datasets[1], runtime.device, args._plot_options, reference,
                                    output / "visuals" / f"epoch_{epoch:03d}.png", epoch)
            convergence = convergence_state(history)
            print(f"COMMITTED epoch={epoch} global_l1={loss:.9f} validation={metrics} early_stop={convergence}", flush=True)
            return convergence["should_stop"]
        if rank0_call(runtime, epoch_outputs):
            break
        epoch += 1
    if not history:
        raise RuntimeError("No epoch completed")

    def finish():
        convergence = convergence_state(history)
        epoch = int(history[-1]["epoch"])
        reason = "l1_converged"
        final = output / "checkpoints" / f"epoch_{epoch:03d}_{reason}_final.pth"
        metrics = helpers.row_validation_metrics(history[-1])
        inference_checkpoint(final, ema, epoch, epoch * steps_per_epoch, metrics, config, reason)
        completion = {"training_status": "complete", "stop_reason": reason, "stop_epoch": epoch,
            "final_checkpoint": final.relative_to(output).as_posix(), "best_checkpoint": best["checkpoint"],
            "convergence_state": convergence, "test_evaluation_run": False}
        helpers.atomic_json(output / "training_completion.json", completion)
        if args._plot_options["training_curves"]:
            draw_status({"phase": "training complete", "epoch": epoch})
        print(json.dumps(completion, indent=2), flush=True)
        return str(final)
    return rank0_call(runtime, finish)


def run_evaluate(args, runtime, precision):
    def evaluate():
        from utils.visualization import ClearGHI, ScatterSample, comparison_grid, sample_indices, scatter_plot
        datasets, roles, counts = load_datasets(args, formal=True)
        if dataset_format(args) == "npz" and cloud_data_identity(args).get("preview_only", False):
            raise ValueError("Preview-only data cannot be used for formal independent evaluation")
        test = datasets[2]
        clear_reference = ClearGHI(test, args._plot_options)
        has_clear_reference = clear_reference.available
        seed_all(SEED, runtime.device)
        model = helpers.build_model(runtime.device)
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        saved_reference = checkpoint.get('config', {}).get('metric_reference')
        if saved_reference is None:
            raise ValueError('Checkpoint lacks a frozen CSI metric reference; use this runner\'s new checkpoints')
        reference = resolve_metric_reference(args, None, saved_reference=saved_reference)
        checkpoint_network = checkpoint.get("network", checkpoint.get("config", {}).get("network"))
        if checkpoint_network is not None and checkpoint_network != helpers.load_network_options():
            raise ValueError("Checkpoint architecture differs from the original network")
        if "params_ema" not in checkpoint:
            raise ValueError("Evaluation requires a frozen params_ema checkpoint")
        model.load_state_dict(checkpoint["params_ema"], strict=True)
        model.eval()
        bicubic_acc, drct_acc = helpers.RegressionAccumulator(), helpers.RegressionAccumulator()
        csi_bicubic_acc, csi_drct_acc = helpers.RegressionAccumulator(), helpers.RegressionAccumulator()
        selected_visual_indices = sample_indices(test, "test", args._plot_options)
        selected_visual_set = set(selected_visual_indices)
        comparison_samples = {}
        csi_scatter = ScatterSample(args._plot_options)
        ghi_scatter = ScatterSample(args._plot_options) if has_clear_reference else None
        by_date_bicubic, by_date_drct = {}, {}
        psnrs, ssims = [], []
        prediction_gradient, target_gradient, offset = 0.0, 0.0, 0
        with torch.inference_mode():
            for low, target in loader(test, roles["test"], args.validation_batch, args.workers):
                if not torch.isfinite(low).all() or not torch.isfinite(target).all():
                    raise FloatingPointError("Non-finite independent test CSI sample")
                with autocast_context(runtime, precision):
                    prediction = model(low.to(runtime.device)).float().cpu()
                if not torch.isfinite(prediction).all():
                    raise FloatingPointError("Non-finite test prediction")
                bicubic = helpers.bicubic_batch(low)
                csi_bicubic_acc.update(target.numpy(), bicubic.numpy())
                csi_drct_acc.update(target.numpy(), prediction.numpy())
                p, s = csi_batch_metrics(prediction, target, data_range=reference['data_range'])
                psnrs.extend(p)
                ssims.extend(s)
                prediction_gradient += float(helpers.sobel_magnitude(prediction).sum().item())
                target_gradient += float(helpers.sobel_magnitude(target).sum().item())
                for item in range(len(target)):
                    index = roles["test"][offset]
                    offset += 1
                    raw_target = target[item, 0].numpy()
                    raw_baseline = bicubic[item, 0].numpy()
                    raw_result = prediction[item, 0].numpy()
                    csi_scatter.add(raw_target, raw_baseline, raw_result)
                    if index in selected_visual_set:
                        comparison_samples[index] = {"low": low[item, 0].numpy().copy(), "target": raw_target.copy(),
                            "prediction": raw_result.copy(), "bicubic": raw_baseline.copy()}
                    if not has_clear_reference:
                        continue
                    clear = clear_reference.get(index)
                    observation = test.physical_ghi_patch(index)
                    baseline = raw_baseline * clear
                    result = raw_result * clear
                    valid = np.isfinite(observation) & np.isfinite(baseline) & np.isfinite(result)
                    observation, baseline, result = observation[valid], baseline[valid], result[valid]
                    bicubic_acc.update(observation, baseline)
                    drct_acc.update(observation, result)
                    ghi_scatter.add(observation, baseline, result)
                    date = str(test.records[index]["date"])
                    by_date_bicubic.setdefault(date, helpers.RegressionAccumulator()).update(observation, baseline)
                    by_date_drct.setdefault(date, helpers.RegressionAccumulator()).update(observation, result)
                if offset % 1000 == 0 or offset == len(roles["test"]):
                    print(f"independent test: {offset}/{len(roles['test'])}", flush=True)
        result = {"checkpoint": str(args.checkpoint.resolve()), "split": "existing independent test split",
            "samples": offset, "precision": precision, "device": str(runtime.device), "data_counts": counts,
            "dataset_format": dataset_format(args),
            "metric_reference": reference,
            "csi": {"psnr": float(np.mean(psnrs)), "ssim": float(np.mean(ssims)),
                    "grad": prediction_gradient / max(target_gradient, 1e-12),
                    "regression": {"bicubic": csi_bicubic_acc.metrics(), "drct": csi_drct_acc.metrics()}},
            "ghi": {"bicubic": bicubic_acc.metrics(), "drct": drct_acc.metrics()} if has_clear_reference else None,
            "by_date": {date: {"bicubic": by_date_bicubic[date].metrics(), "drct": by_date_drct[date].metrics()} for date in sorted(by_date_drct)},
            "common_mask": "finite FY-3C/Bicubic/DRCT GHI; no value clipping" if has_clear_reference else None,
            "ghi_restore": "raw predicted/Bicubic CSI * supplied matched GHI_clear; observation is stored FY-3C SSI" if has_clear_reference else None,
            "ghi_note": ("Evaluated using the supplied matched clear-GHI reference" if has_clear_reference else
                         "CSI metrics only: the NPZ source stores observed FY-3C GHI but supplies no matched GHI_clear reference. Physical GHI predictions/metrics cannot be restored without that reference; it is not inferred from FY-3C GHI or HR CSI."),
            "vendor_ppu_verified": False}
        destination = args.output_dir / "evaluation" / (args.checkpoint.stem + "_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
        destination.mkdir(parents=True, exist_ok=False)
        helpers.atomic_json(destination / "independent_test_metrics.json", result)
        # Final figures are generated here only, after the frozen checkpoint has
        # finished the full test CSV. Training never calls this evaluation path.
        figures = {"checkpoint": str(args.checkpoint.resolve()), "split": "test",
                   "selection": "fixed CSV rows; not selected using model performance",
                   "csv_row_indices": selected_visual_indices,
                   "sample_records": [dict(test.records[i]) for i in selected_visual_indices]}
        ordered_samples = [comparison_samples[i] for i in selected_visual_indices]
        figures["comparison"] = comparison_grid(ordered_samples, reference, args._plot_options,
                                                 destination / "csi_comparison_10_samples.png")
        np.savez(destination / "csi_comparison_data.npz", **{
            key: np.stack([sample[key] for sample in ordered_samples])
            for key in ("low", "target", "prediction", "bicubic")})
        if args._plot_options["scatter_csi"]:
            figures["scatter_csi"] = scatter_plot(csi_scatter, result["csi"]["regression"],
                                                   destination / "scatter_csi.png", units="CSI")
            np.savez(destination / "scatter_csi_data.npz", paired_values=csi_scatter.values)
        if has_clear_reference and args._plot_options["scatter_ghi"]:
            figures["scatter_ghi"] = scatter_plot(ghi_scatter, result["ghi"], destination / "scatter_ghi.png", units="W/m²")
            np.savez(destination / "scatter_ghi_data.npz", paired_values=ghi_scatter.values)
        else:
            figures["scatter_ghi"] = {"status": "disabled" if has_clear_reference else "awaiting_aligned_clear_ghi"}
        helpers.atomic_json(destination / "figure_manifest.json", figures)
        result["figures"] = figures
        helpers.atomic_json(destination / "independent_test_metrics.json", result)
        with (destination / "metrics_by_date.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["date", "method", "r2", "rmse", "mae", "bias"])
            writer.writeheader()
            for date, values in result["by_date"].items():
                for method in ("bicubic", "drct"):
                    writer.writerow({"date": date, "method": method, **values[method]})
        print(json.dumps(result, indent=2), flush=True)
        return str(destination / "independent_test_metrics.json")
    return rank0_call(runtime, evaluate)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("check", "smoke", "train", "evaluate"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--data-root", required=True, type=Path, help="Existing prepared splits or 2018-2019 NPZ/CSI-cache root with fixed CSV splits")
        sub.add_argument("--dataset-format", choices=("auto", "npz"), default="auto",
                         help="auto detects existing splits/train_samples.csv; never creates or re-splits data")
        sub.add_argument('--download-selection', type=Path, help='Preview selection written by download_data.py; check/smoke only')
        sub.add_argument('--metric-reference', type=Path, help='Frozen TRAIN-only reference produced by prepare_metrics.py')
        sub.add_argument('--metric-data-range', type=float, help='Explicit fixed reference range in RAW CSI units; never auto inferred from validation/test')
        sub.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "drct_2018_2019")
        sub.add_argument("--visual-config", type=Path, default=None, help=argparse.SUPPRESS)
        sub.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        sub.add_argument("--backend", default=None, help="Process-group backend; default nccl for CUDA, gloo for CPU")
        sub.add_argument("--precision", choices=("fp16", "fp32"), default=None)
        sub.add_argument("--workers", type=int, default=2)
        sub.add_argument("--validation-batch", type=int, default=8)
        sub.add_argument("--cpu-threads", type=int, default=min(4, os.cpu_count() or 1))
        sub.add_argument("--dist-timeout", type=int, default=3600, help="Seconds; includes rank-zero full validation")
        sub.add_argument("--deterministic", action="store_true", help="Opt in to strict deterministic kernels; default is seeded execution")
        sub.add_argument("--local-rank", "--local_rank", type=int, default=None, help=argparse.SUPPRESS)
        if command in ("check", "smoke"):
            sub.add_argument("--allow-small-data", action="store_true", help="Explicit pipeline-only synthetic/test fixture use")
        if command == 'check':
            sub.add_argument('--full-model', action='store_true', help='Run CPU forward/backward/checkpoint restore with the complete original DRCT-L')
        if command in ("smoke", "train"):
            sub.add_argument("--micro-batch", type=int, default=None)
            sub.add_argument("--accumulation", type=int, default=None)
            adam = sub.add_mutually_exclusive_group()
            adam.add_argument("--fused-adam", action="store_true", help="Opt in on a vendor backend only after verification; automatic on NVIDIA")
            adam.add_argument("--plain-adam", action="store_true", help="Use regular Adam; automatic on CPU and unverified vendor accelerators")
            sub.add_argument("--allow-cpu-training", action="store_true", help="Explicitly allow slow full-model CPU updates")
        if command == "smoke":
            sub.add_argument("--steps", type=int, default=2)
        if command == "train":
            sub.add_argument("--resume", action="store_true", help="Restore authoritative cloud checkpoint, valid from epoch zero")
            sub.add_argument("--visuals", action="store_true", help="Write original five-pair montage/progress after each epoch")
        if command == "evaluate":
            sub.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.workers < 0 or args.validation_batch < 1 or args.cpu_threads < 1 or args.dist_timeout < 1:
        parser.error("workers must be nonnegative; batches/threads/timeout must be positive")
    if getattr(args, "steps", 1) < 1:
        parser.error("steps must be positive")
    data_root = args.data_root.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    if output_root == data_root or data_root in output_root.parents:
        parser.error('--output-dir must be outside the source dataset directory')
    from utils.visualization import plot_options
    args._plot_options = plot_options(args.visual_config, force_visuals=getattr(args, "visuals", False))
    return args


def main(argv=None):
    args = parse_args(argv)
    runtime = None
    try:
        runtime = setup_runtime(args)
        precision = resolve_precision(args, runtime)
        if args.command in ("smoke", "train") and runtime.device.type == "cpu" and not args.allow_cpu_training:
            raise RuntimeError("Full-model CPU updates require --allow-cpu-training. Use check for the tiny CPU pipeline test")
        if runtime.primary:
            print(f"mode={args.command} device={runtime.device} world={runtime.world_size} precision={precision} backend={runtime.backend}", flush=True)
            print("PPU execution has not been verified; use the vendor's compatible PyTorch environment and run full-model smoke first.", flush=True)
        functions = {"check": lambda: run_check(args, runtime), "smoke": lambda: run_smoke(args, runtime, precision),
                     "train": lambda: run_train(args, runtime, precision), "evaluate": lambda: run_evaluate(args, runtime, precision)}
        result = functions[args.command]()
        if runtime.primary:
            print(f"RESULT: {result}", flush=True)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
