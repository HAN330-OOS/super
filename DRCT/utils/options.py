"""Read the one experiment configuration and invoke the established trainer.

Only the entry/configuration interface is new. Updates, validation, checkpoints,
metrics and distributed execution remain in utils.trainer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPT = ROOT / "exp" / "train_x4.json"
NETWORK_SHA256 = "7fa84a3da198830f20391e20aad245dc11c3325ca41ff6f784a138387d85d639"
FIXED_TRAIN = {
    "global_batch": 64, "seed": 60, "learning_rate": 0.0001,
    "betas": [0.9, 0.999], "loss": "L1", "ema_decay": 0.999, "augmentation": False,
    "early_stop": {"min_delta": 0.0001, "patience": 5},
}


def project_path(value) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def load_options(path=DEFAULT_OPT):
    path = project_path(path)
    with path.open(encoding="utf-8-sig") as handle:
        config = json.load(handle)
    expected = {"path", "runtime", "network", "train", "metrics", "test", "visualization"}
    if not isinstance(config, dict) or set(config) != expected:
        raise ValueError("Configuration must contain path, runtime, network, train, metrics, test and visualization only")
    for key in expected:
        if not isinstance(config[key], dict):
            raise ValueError(f"Configuration section {key} must be an object")
    network_hash = hashlib.sha256(json.dumps(config["network"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if network_hash != NETWORK_SHA256:
        raise ValueError("This migration retains the original DRCT-L architecture. Restore the supplied network section")
    for key, expected_value in FIXED_TRAIN.items():
        if config["train"].get(key) != expected_value:
            raise ValueError(f"train.{key} must retain the original experiment value {expected_value!r}")
    if "epochs" in config["train"]:
        raise ValueError("train.epochs is no longer used")
    if "min_epoch" in config["train"].get("early_stop", {}):
        raise ValueError("train.early_stop.min_epoch is no longer used")
    for key in ("data_root", "output_dir"):
        value = config["path"].get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"path.{key} must be a nonempty path")
    device = config["runtime"].get("device", "auto")
    precision = config["runtime"].get("precision")
    if device not in ("auto", "cpu", "cuda") or precision not in (None, "fp16", "fp32"):
        raise ValueError("runtime.device: auto/cpu/cuda; runtime.precision: null/fp16/fp32")
    if config["metrics"].get("reference") is not None and config["metrics"].get("data_range") is not None:
        raise ValueError("Choose one metric reference file or one declared raw CSI range")
    return path, config


def add_arguments(parser, mode):
    parser.add_argument("--opt", default=DEFAULT_OPT, type=Path, help="Experiment JSON; default exp/train_x4.json")
    parser.add_argument("--data-root", type=Path, help="Override path.data_root")
    parser.add_argument("--output-dir", type=Path, help="Override path.output_dir")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--precision", choices=("fp16", "fp32"))
    parser.add_argument("--workers", type=int)
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--validation-batch", type=int)
    parser.add_argument("--backend", help="Use only a backend supported by the actual PyTorch image")
    parser.add_argument("--metric-reference", type=Path)
    parser.add_argument("--metric-data-range", type=float)
    parser.add_argument("--dry-run", action="store_true", help="Show effective arguments without importing torch or starting training")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None, help=argparse.SUPPRESS)
    if mode in ("train", "smoke"):
        parser.add_argument("--micro-batch", type=int, help="Per-device batch; accumulation keeps global batch at 64")
        parser.add_argument("--allow-cpu-training", action="store_true")
        adam = parser.add_mutually_exclusive_group()
        adam.add_argument("--fused-adam", action="store_true")
        adam.add_argument("--plain-adam", action="store_true")
    if mode == "train":
        parser.add_argument("--resume", action="store_true")
        parser.add_argument("--visuals", action="store_true")
    if mode == "evaluate":
        parser.add_argument("--checkpoint", type=Path, help="Default: best validation checkpoint in training_completion.json")
    if mode in ("check", "smoke"):
        parser.add_argument("--example", action="store_true", help="Use the included small preview only")
        parser.add_argument("--download-selection", type=Path, help="Selection JSON from an explicit OSS preview download")
    if mode == "smoke":
        parser.add_argument("--steps", type=int, default=2)


def backend_arguments(mode, args, config):
    settings = config["runtime"]
    example = bool(getattr(args, "example", False)) or (mode == "check" and args.data_root is None)
    if getattr(args, "example", False) and args.data_root is not None:
        raise ValueError("Choose --example or --data-root")
    data_root = ROOT / "data" / "example" if example else project_path(args.data_root or config["path"]["data_root"])
    default_output = config["path"]["output_dir"] if mode in ("train", "evaluate") else f"results/{mode}"
    output = project_path(args.output_dir or default_output)
    if output == data_root or data_root in output.parents:
        raise ValueError("Output directory must be outside the source dataset")
    result = [mode, "--data-root", str(data_root), "--output-dir", str(output),
              "--visual-config", str(project_path(args.opt))]
    values = {
        "device": "cpu" if mode == "check" else args.device or settings.get("device", "auto"),
        "precision": args.precision or settings.get("precision"),
        "workers": args.workers if args.workers is not None else (0 if mode == "check" else settings.get("workers", 2)),
        "cpu_threads": args.cpu_threads if args.cpu_threads is not None else settings.get("cpu_threads", 8),
        "validation_batch": args.validation_batch if args.validation_batch is not None else settings.get("validation_batch", 8),
        "backend": args.backend or settings.get("backend"),
        "dist_timeout": settings.get("dist_timeout", 3600),
    }
    if mode == "check":
        values["precision"] = "fp32"
    for name in ("workers", "cpu_threads", "validation_batch", "dist_timeout"):
        value = values[name]
        minimum = 0 if name == "workers" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    for name, value in values.items():
        if value is not None:
            result.extend(["--" + name.replace("_", "-"), str(value)])
    if settings.get("deterministic", False):
        result.append("--deterministic")
    reference = args.metric_reference
    declared_range = args.metric_data_range
    if reference is None and declared_range is None and not example:
        reference = config["metrics"].get("reference")
        declared_range = config["metrics"].get("data_range")
    if reference is not None and declared_range is not None:
        raise ValueError("Choose --metric-reference or --metric-data-range")
    if reference is not None:
        result.extend(["--metric-reference", str(project_path(reference))])
    if declared_range is not None:
        result.extend(["--metric-data-range", str(declared_range)])
    if mode in ("check", "smoke"):
        selection = getattr(args, "download_selection", None)
        if example or selection is not None:
            result.append("--allow-small-data")
        if selection is not None:
            result.extend(["--download-selection", str(project_path(selection))])
        if mode == "check":
            result.append("--full-model")
        else:
            result.extend(["--steps", str(args.steps)])
    if mode in ("train", "smoke"):
        micro = args.micro_batch if args.micro_batch is not None else settings.get("micro_batch")
        if micro is not None:
            world = int(os.environ.get("WORLD_SIZE", "1"))
            if isinstance(micro, bool) or not isinstance(micro, int) or micro < 1 or 64 % (world * micro):
                raise ValueError("world_size * micro_batch must divide the fixed global batch 64")
            result.extend(["--micro-batch", str(micro)])
        for flag in ("allow_cpu_training", "fused_adam", "plain_adam"):
            if getattr(args, flag, False):
                result.append("--" + flag.replace("_", "-"))
    if mode == "train":
        if args.resume:
            result.append("--resume")
        if args.visuals:
            result.append("--visuals")
    if mode == "evaluate":
        checkpoint = args.checkpoint or config["test"].get("checkpoint")
        if checkpoint is None:
            completion_file = output / "training_completion.json"
            if not completion_file.is_file():
                raise FileNotFoundError("Set --checkpoint, or finish training so training_completion.json exists")
            completion = json.loads(completion_file.read_text(encoding="utf-8"))
            relative = completion.get("best_checkpoint") or completion.get("final_checkpoint")
            if not relative:
                raise ValueError("Training completion does not identify a frozen checkpoint")
            checkpoint = output / relative
        result.extend(["--checkpoint", str(project_path(checkpoint))])
    return result


def execute(mode, args):
    option_path, config = load_options(args.opt)
    argv = backend_arguments(mode, args, config)
    if args.dry_run:
        print(json.dumps({"option_file": str(option_path), "mode": mode, "backend_arguments": argv,
              "representation": "raw_csi", "normalization": "none", "value_clipping": False,
              "global_batch": 64, "vendor_ppu_verified": False}, ensure_ascii=False, indent=2))
        return
    from utils import training, trainer
    training.OPTION_PATH = option_path
    # Parsing again retains the existing backend's checks and checkpoint protocol.
    trainer.main(argv)


def run(mode, argv=None):
    parser = argparse.ArgumentParser(description="DRCT-L x4: raw CSI with the existing fixed CSV splits")
    add_arguments(parser, mode)
    args = parser.parse_args(argv)
    try:
        execute(mode, args)
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as error:
        parser.error(str(error))
