"""One entry for environment checks, CPU tests and local dataset preparation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.options import DEFAULT_OPT, ROOT, add_arguments, execute, load_options, project_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("env", help="查看 Python、PyTorch、设备和真实磁盘空间")
    install = commands.add_parser("install", help="只补齐缺少的普通依赖，保留镜像的 PyTorch")
    for name in ("check", "smoke"):
        sub = commands.add_parser(name, help="完整 DRCT-L CPU 检查" if name == "check" else "完整模型单卡/多卡短训练")
        add_arguments(sub, name)
    for name in ("metrics", "cache"):
        sub = commands.add_parser(name, help={"metrics": "冻结训练集的 CSI 指标参考范围",
                                              "cache": "将 NPZ 转成本地原值缓存"}[name])
        sub.add_argument("--opt", type=Path, default=DEFAULT_OPT)
        sub.add_argument("--data-root", type=Path, help="Override the configured local dataset root")
        if name == "metrics":
            sub.add_argument("--output", type=Path)
        if name == "cache":
            sub.add_argument("--destination", type=Path, required=True)
    return parser, parser.parse_args(argv)


def main(argv=None):
    parser, args = parse_args(argv)
    if args.command == "env":
        from utils.environment import main as environment
        environment()
        return 0
    if args.command == "install":
        from utils.install import main as install
        install()
        return 0
    try:
        if args.command in ("check", "smoke"):
            execute(args.command, args)
            return 0
        _, config = load_options(args.opt)
        data_root = project_path(args.data_root or config["path"]["data_root"])
        output = project_path(config["path"]["output_dir"])
        if args.command == "metrics":
            from utils.prepare_metrics import prepare
            destination = project_path(args.output) if args.output else output / "metric_reference.json"
            print(json.dumps(prepare(data_root, destination), ensure_ascii=False, indent=2, allow_nan=False))
        if args.command == "cache":
            from utils.cache import convert_cache
            convert_cache(data_root, project_path(args.destination))
        return 0
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
