#!/usr/bin/env python3
"""Train controlled DSConv U-Net from a flat YAML/JSON config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eeg_denoise_benchmark.config import load_config, parse_scalar  # noqa: E402
from eeg_denoise_benchmark.training import train_experiment  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Flat YAML/JSON config.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override any flat config key. Can be repeated.",
    )
    return parser.parse_args()


def parse_key_value(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise ValueError(f"Expected KEY=VALUE override, got: {raw}")
    key, value = raw.split("=", 1)
    key = key.strip().replace("-", "_")
    if not key:
        raise ValueError(f"Empty override key in: {raw}")
    return key, parse_scalar(value)


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    overrides = {
        "output_dir": args.output_dir,
        "device": args.device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_train_batches": args.max_train_batches,
        "max_eval_batches": args.max_eval_batches,
    }
    merged = dict(config)
    for key, value in overrides.items():
        if value is not None:
            merged[key] = str(value) if isinstance(value, Path) else value
    for raw in args.set_values:
        key, value = parse_key_value(raw)
        merged[key] = value
    return merged


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    train_experiment(config)


if __name__ == "__main__":
    main()
