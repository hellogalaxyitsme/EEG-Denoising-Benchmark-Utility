#!/usr/bin/env python3
"""Evaluate a controlled DSConv U-Net checkpoint on a saved synthetic split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eeg_denoise_benchmark.checkpoints import load_model_from_checkpoint  # noqa: E402
from eeg_denoise_benchmark.data import SyntheticArrayDataset, load_synthetic_split  # noqa: E402
from eeg_denoise_benchmark.eval import compute_denoising_metrics  # noqa: E402
from eeg_denoise_benchmark.models import count_trainable_parameters  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Optional flat YAML/JSON config.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint path.")
    parser.add_argument("--data", type=Path, required=True, help="Split dir, .npy, .npz, or chunk dir.")
    parser.add_argument("--split", default="test", help="Split name when --data is a directory or .npz.")
    parser.add_argument("--fs", type=float, default=256.0, help="Sampling rate for PSD metrics.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda", help="cuda, cpu, or cuda:N.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional metrics output path.")
    parser.add_argument(
        "--non-strict",
        dest="strict",
        action="store_false",
        default=True,
        help="Allow non-strict checkpoint loading.",
    )
    if "--config" in sys.argv:
        parser.set_defaults(checkpoint=None, data=None)
        for action in parser._actions:
            if action.dest in {"checkpoint", "data"}:
                action.required = False
    args = parser.parse_args()
    if args.config is not None:
        args = _merge_config(args)
    if args.checkpoint is None or args.data is None:
        parser.error("--checkpoint and --data are required unless provided by --config")
    return args


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip("\"'")


def _load_config(path: Path) -> dict[str, Any]:
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    config: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"Unsupported config line in {path}: {raw_line}")
        key, value = line.split(":", 1)
        config[key.strip().replace("-", "_")] = _parse_scalar(value)
    return config


def _merge_config(args: argparse.Namespace) -> argparse.Namespace:
    config = _load_config(args.config)
    for key, value in config.items():
        if not hasattr(args, key):
            continue
        current = getattr(args, key)
        if current is None or key not in _explicit_cli_keys():
            if key in {"checkpoint", "data", "output_json"} and value is not None:
                value = Path(str(value))
            setattr(args, key, value)
    return args


def _explicit_cli_keys() -> set[str]:
    keys: set[str] = set()
    args = sys.argv[1:]
    for idx, token in enumerate(args):
        if token.startswith("--"):
            key = token[2:].split("=", 1)[0].replace("-", "_")
            keys.add(key)
            if "=" not in token and idx + 1 < len(args):
                continue
    return keys


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(v) for v in value]
    return value


@torch.no_grad()
def predict_clean(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for y, x, _a in loader:
        y = y.to(device, non_blocking=True)
        out = model(y)
        pred = out[:, 0:1, :].detach().cpu().numpy()[:, 0, :]
        target = x.detach().cpu().numpy()[:, 0, :]
        predictions.append(pred)
        targets.append(target)
    return np.concatenate(targets, axis=0), np.concatenate(predictions, axis=0)


def main() -> None:
    args = parse_args()
    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    model, checkpoint = load_model_from_checkpoint(
        args.checkpoint,
        map_location="cpu",
        strict=args.strict,
    )
    model.to(device)

    pack = load_synthetic_split(args.data, split=args.split)
    dataset = SyntheticArrayDataset(pack)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    target, prediction = predict_clean(model, loader, device)
    metrics = compute_denoising_metrics(target=target, prediction=prediction, fs=args.fs)

    result = {
        "checkpoint": str(args.checkpoint),
        "data": str(args.data),
        "split": args.split,
        "fs": args.fs,
        "device": str(device),
        "trainable_parameters": count_trainable_parameters(model),
        "metrics": metrics,
    }
    if isinstance(checkpoint, dict):
        result["checkpoint_cfg"] = _as_jsonable(checkpoint.get("cfg", {}))
        if "test" in checkpoint:
            result["embedded_test"] = _as_jsonable(checkpoint["test"])

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
