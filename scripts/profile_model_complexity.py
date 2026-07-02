#!/usr/bin/env python3
"""Profile denoiser variants for deployment-oriented complexity metrics."""

from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from eeg_denoise_benchmark.models import (  # noqa: E402
    TinyDenoiser,
    build_eegdn_baseline,
    count_trainable_parameters,
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    model_type: str
    base: int | str
    fixed_length: int | None = None


def int_list(raw: str) -> list[int]:
    return [int(part) for part in raw.replace(",", " ").split() if part.strip()]


def str_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.replace(",", " ").split() if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bases", default="2 4 6 8 16")
    parser.add_argument("--lengths", default="512")
    parser.add_argument("--batch-sizes", default="1 256")
    parser.add_argument("--devices", default="cpu cuda", help="Devices to profile, e.g. 'cpu cuda' or 'auto'.")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--extra-blocks", type=int, default=2)
    parser.add_argument("--conv-block", default="ds", choices=["ds", "standard"])
    parser.add_argument("--use-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-dwt", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-artifact-head", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-eegdn-baselines", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--eegdn-baselines",
        default="eegdn_cnn eegdn_rnn",
        help="EEGDenoiseNet baselines to include when --include-eegdn-baselines is true.",
    )
    return parser.parse_args()


def resolve_devices(raw: str) -> list[torch.device]:
    requested = str_list(raw)
    if requested == ["auto"]:
        requested = ["cuda"] if torch.cuda.is_available() else ["cpu"]
    devices: list[torch.device] = []
    for name in requested:
        if name.startswith("cuda") and not torch.cuda.is_available():
            print(f"[skip_device] {name} unavailable")
            continue
        device = torch.device(name)
        if str(device) not in {str(existing) for existing in devices}:
            devices.append(device)
    if not devices:
        devices.append(torch.device("cpu"))
    return devices


def make_specs(args: argparse.Namespace) -> list[ModelSpec]:
    specs = [
        ModelSpec(
            name=f"base{base}",
            family="controlled DSConv U-Net",
            model_type="controlled_backbone",
            base=base,
        )
        for base in int_list(args.bases)
    ]
    if args.include_eegdn_baselines:
        for model_type in str_list(args.eegdn_baselines):
            label = "EEGDN CNN" if "cnn" in model_type else "EEGDN RNN"
            specs.append(
                ModelSpec(
                    name=label,
                    family="EEGDenoiseNet",
                    model_type=model_type,
                    base="",
                    fixed_length=512,
                )
            )
    return specs


def build_model(args: argparse.Namespace, spec: ModelSpec, length: int, device: torch.device) -> nn.Module:
    if spec.family == "controlled DSConv U-Net":
        if not isinstance(spec.base, int):
            raise TypeError(f"controlled DSConv U-Net base must be int, got {spec.base!r}")
        model = TinyDenoiser(
            base=spec.base,
            extra_bottleneck_blocks=args.extra_blocks,
            use_dwt=args.use_dwt,
            use_gate=args.use_gate,
            use_attn=args.use_attn,
            use_artifact_head=args.use_artifact_head,
            conv_block=args.conv_block,
        )
    else:
        model = build_eegdn_baseline(model_type=spec.model_type, datanum=length)
    model.to(device)
    model.eval()
    return model


def state_size_bytes(model: nn.Module) -> int:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return int(buffer.tell())


class MacCounter:
    """Forward-hook MAC counter for Conv/Linear/LSTM layers used by the profiled models."""

    def __init__(self) -> None:
        self.total_macs = 0
        self.by_module: dict[str, int] = {}
        self.handles: list[Any] = []

    def add_hooks(self, model: nn.Module) -> None:
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear, nn.LSTM)):
                self.handles.append(module.register_forward_hook(self._hook(name)))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _hook(self, name: str):
        def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: Any) -> None:
            tensor_output = output[0] if isinstance(output, tuple) else output
            macs = self._module_macs(module, inputs[0], tensor_output)
            self.total_macs += macs
            self.by_module[name] = self.by_module.get(name, 0) + macs

        return hook

    @staticmethod
    def _module_macs(module: nn.Module, x: torch.Tensor, y: torch.Tensor) -> int:
        if isinstance(module, nn.Conv1d):
            batch = int(y.shape[0])
            out_channels = int(y.shape[1])
            out_len = int(y.shape[2])
            kernel = int(module.kernel_size[0])
            in_per_group = int(module.in_channels // module.groups)
            return batch * out_channels * out_len * in_per_group * kernel
        if isinstance(module, nn.ConvTranspose1d):
            batch = int(y.shape[0])
            out_channels = int(y.shape[1])
            out_len = int(y.shape[2])
            kernel = int(module.kernel_size[0])
            in_per_group = int(module.in_channels // module.groups)
            return batch * out_channels * out_len * in_per_group * kernel
        if isinstance(module, nn.Linear):
            return int(y.numel()) * int(module.in_features)
        if isinstance(module, nn.LSTM):
            batch_first = bool(module.batch_first)
            batch = int(x.shape[0] if batch_first else x.shape[1])
            seq_len = int(x.shape[1] if batch_first else x.shape[0])
            directions = 2 if module.bidirectional else 1
            hidden = int(module.hidden_size)
            total = 0
            input_size = int(module.input_size)
            for layer_index in range(module.num_layers):
                layer_input = input_size if layer_index == 0 else hidden * directions
                total += batch * seq_len * directions * 4 * hidden * (layer_input + hidden)
            return total
        return 0


@torch.inference_mode()
def count_macs(model: nn.Module, length: int, device: torch.device) -> dict[str, Any]:
    counter = MacCounter()
    counter.add_hooks(model)
    x = torch.randn(1, 1, length, device=device)
    try:
        _ = model(x)
    finally:
        counter.close()
    return {
        "macs_per_segment": int(counter.total_macs),
        "flops_per_segment": int(counter.total_macs * 2),
        "macs_by_module": counter.by_module,
    }


@torch.inference_mode()
def profile_latency(
    model: nn.Module,
    *,
    length: int,
    batch_size: int,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, float | int | str]:
    x = torch.randn(batch_size, 1, length, device=device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    for _ in range(warmup):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - start) * 1000.0)

    mean_ms = float(statistics.mean(samples))
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return {
        "device": str(device),
        "batch_size": int(batch_size),
        "length": int(length),
        "latency_mean_ms": mean_ms,
        "latency_median_ms": float(statistics.median(samples)),
        "latency_std_ms": float(statistics.stdev(samples)) if len(samples) > 1 else 0.0,
        "latency_min_ms": float(min(samples)),
        "latency_max_ms": float(max(samples)),
        "latency_per_segment_ms": float(mean_ms / batch_size),
        "throughput_segments_per_s": float(batch_size / (mean_ms / 1000.0)),
        "peak_memory_bytes": peak_memory,
        "peak_memory_kb": float(peak_memory / 1024.0),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "model_name",
        "model_family",
        "model_type",
        "base",
        "length",
        "batch_size",
        "device",
        "trainable_parameters",
        "state_size_kb",
        "macs_per_segment",
        "flops_per_segment",
        "latency_mean_ms",
        "latency_median_ms",
        "latency_std_ms",
        "latency_min_ms",
        "latency_max_ms",
        "latency_per_segment_ms",
        "throughput_segments_per_s",
        "peak_memory_bytes",
        "peak_memory_kb",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# Model Complexity Profile", ""]
    lines.append(
        "| Model | Family | Len | Batch | Device | Params | Size KB | MACs | FLOPs | "
        "Latency ms | ms/segment | Throughput/s | Peak mem KB |"
    )
    lines.append("|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['model_name']} | {row['model_family']} | {row['length']} | {row['batch_size']} | {row['device']} | "
            f"{row['trainable_parameters']} | {row['state_size_kb']:.2f} | "
            f"{row['macs_per_segment']} | {row['flops_per_segment']} | "
            f"{row['latency_mean_ms']:.4f} | {row['latency_per_segment_ms']:.6f} | "
            f"{row['throughput_segments_per_s']:.2f} | {row['peak_memory_kb']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
    devices = resolve_devices(args.devices)
    lengths = int_list(args.lengths)
    batch_sizes = int_list(args.batch_sizes)
    specs = make_specs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    mac_details: dict[str, Any] = {}
    print(
        f"[start] devices={[str(device) for device in devices]} "
        f"models={[spec.name for spec in specs]} lengths={lengths} batches={batch_sizes}"
    )

    for device in devices:
        for spec in specs:
            profile_lengths = [spec.fixed_length] if spec.fixed_length is not None else lengths
            for length in profile_lengths:
                assert length is not None
                model = build_model(args, spec, length, device)
                params = count_trainable_parameters(model)
                size_kb = state_size_bytes(model) / 1024.0
                mac_info = count_macs(model, length=length, device=device)
                mac_details[f"{spec.name}_{device}_len{length}"] = mac_info
                print(
                    f"[model] name={spec.name} device={device} len={length} "
                    f"params={params} size_kb={size_kb:.2f} macs={mac_info['macs_per_segment']}"
                )

                for batch_size in batch_sizes:
                    latency = profile_latency(
                        model,
                        length=length,
                        batch_size=batch_size,
                        device=device,
                        warmup=args.warmup,
                        repeats=args.repeats,
                    )
                    row = {
                        "model_name": spec.name,
                        "model_family": spec.family,
                        "model_type": spec.model_type,
                        "base": spec.base,
                        "trainable_parameters": params,
                        "state_size_kb": size_kb,
                        "macs_per_segment": mac_info["macs_per_segment"],
                        "flops_per_segment": mac_info["flops_per_segment"],
                        **latency,
                    }
                    rows.append(row)
                    print(
                        f"[result] model={spec.name} device={device} len={length} batch={batch_size} "
                        f"lat_mean_ms={row['latency_mean_ms']:.4f} "
                        f"per_segment_ms={row['latency_per_segment_ms']:.6f} "
                        f"throughput={row['throughput_segments_per_s']:.2f}/s"
                    )

                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    summary = {
        "architecture": {
            "extra_blocks": args.extra_blocks,
            "conv_block": args.conv_block,
            "use_attn": args.use_attn,
            "use_dwt": args.use_dwt,
            "use_gate": args.use_gate,
            "use_artifact_head": args.use_artifact_head,
            "eegdn_baselines": str_list(args.eegdn_baselines) if args.include_eegdn_baselines else [],
        },
        "devices": [str(device) for device in devices],
        "num_threads": torch.get_num_threads(),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "rows": rows,
        "mac_details": mac_details,
        "notes": [
            "MACs count multiply-accumulate operations for Conv1d, ConvTranspose1d, Linear, and LSTM layers.",
            "FLOPs are reported as 2x MACs.",
            "EEGDenoiseNet CNN/RNN baselines have dense heads fixed to 512-sample windows.",
        ],
    }
    json_path = args.output_dir / "complexity_profile.json"
    csv_path = args.output_dir / "complexity_profile.csv"
    md_path = args.output_dir / "complexity_profile.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    print(f"[written] {json_path}")
    print(f"[written] {csv_path}")
    print(f"[written] {md_path}")
    print("[done]")


if __name__ == "__main__":
    main()
