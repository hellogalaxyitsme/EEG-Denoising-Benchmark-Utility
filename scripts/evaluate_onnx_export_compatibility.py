#!/usr/bin/env python3
"""ONNX export and numerical compatibility audit for compact denoisers."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.evaluate_bci2a_downstream_csp_lda import (  # noqa: E402
    checkpoint_seed,
    load_checkpoint_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-glob", required=True)
    parser.add_argument("--bases", default="4,6,16")
    parser.add_argument("--checkpoint-seeds", default="42,43,44")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample-length", type=int, default=512)
    parser.add_argument("--n-test-inputs", type=int, default=32)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--rng-seed", type=int, default=20260816)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--dynamic-batch", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def parse_int_set(raw: str) -> set[int]:
    values = {int(part) for part in raw.replace(",", " ").split() if part.strip()}
    if not values:
        raise ValueError(f"Expected at least one integer in {raw!r}")
    return values


def natural_key(path: Path) -> tuple[int, int, str]:
    base = checkpoint_base(path)
    seed = checkpoint_seed_from_path(path)
    return base if base is not None else 10**9, seed if seed is not None else 10**9, str(path)


def checkpoint_base(path: Path) -> int | None:
    match = re.search(r"base(\d+)", str(path))
    return int(match.group(1)) if match else None


def checkpoint_seed_from_path(path: Path) -> int | None:
    match = re.search(r"seed(\d+)", str(path))
    return int(match.group(1)) if match else None


def expand_selected_checkpoints(pattern: str, bases: set[int], seeds: set[int]) -> list[Path]:
    paths = sorted((Path(path) for path in glob.glob(pattern)), key=natural_key)
    selected = []
    for path in paths:
        base = checkpoint_base(path)
        seed = checkpoint_seed_from_path(path)
        if base in bases and seed in seeds:
            selected.append(path)
    if not selected:
        raise FileNotFoundError(f"No checkpoints matched bases={sorted(bases)} seeds={sorted(seeds)} pattern={pattern}")
    return selected


class FullOutputWrapper(torch.nn.Module):
    """Wrap a checkpoint model so ONNX export preserves the same full output."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def file_size_rows(paths: dict[str, Path]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for label, path in paths.items():
        row[f"{label}_bytes"] = int(path.stat().st_size) if path.exists() else ""
        row[f"{label}_mib"] = float(path.stat().st_size / (1024 * 1024)) if path.exists() else ""
    return row


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    av = a.reshape(-1).astype(np.float64)
    bv = b.reshape(-1).astype(np.float64)
    if av.size == 0 or float(np.std(av)) == 0.0 or float(np.std(bv)) == 0.0:
        return float("nan")
    return float(np.corrcoef(av, bv)[0, 1])


def compare_outputs(torch_out: np.ndarray, ort_out: np.ndarray) -> dict[str, Any]:
    diff = ort_out.astype(np.float64) - torch_out.astype(np.float64)
    clean_diff = diff[:, 0:1, :] if diff.ndim == 3 and diff.shape[1] >= 1 else diff
    clean_ref = torch_out[:, 0:1, :] if torch_out.ndim == 3 and torch_out.shape[1] >= 1 else torch_out
    denom = max(float(np.max(np.abs(torch_out))), 1e-12)
    return {
        "max_abs_error": float(np.max(np.abs(diff))),
        "mean_abs_error": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "relative_rmse_vs_max_abs_pytorch": float(np.sqrt(np.mean(diff * diff)) / denom),
        "max_abs_error_clean_channel": float(np.max(np.abs(clean_diff))),
        "rmse_clean_channel": float(np.sqrt(np.mean(clean_diff * clean_diff))),
        "pearson_full_output": pearson_corr(torch_out, ort_out),
        "pearson_clean_channel": pearson_corr(clean_ref, ort_out[:, 0:1, :] if ort_out.ndim == 3 and ort_out.shape[1] >= 1 else ort_out),
        "allclose_atol_1e_4_rtol_1e_4": bool(np.allclose(torch_out, ort_out, atol=1e-4, rtol=1e-4)),
        "allclose_atol_1e_5_rtol_1e_5": bool(np.allclose(torch_out, ort_out, atol=1e-5, rtol=1e-5)),
    }


def runtime_info(args: argparse.Namespace) -> dict[str, Any]:
    import onnx
    import onnxruntime as ort

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "cpu_count_os": os.cpu_count(),
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "onnxruntime_available_providers": ort.get_available_providers(),
        "onnxruntime_backend": "CPUExecutionProvider",
        "requested_cpu_threads": int(args.cpu_threads),
        "sample_length": int(args.sample_length),
        "n_test_inputs": int(args.n_test_inputs),
        "opset": int(args.opset),
        "dynamic_batch": bool(args.dynamic_batch),
        "claim_guardrail": "Successful ONNX export verifies graph compatibility and numerical agreement, not demonstrated edge-device performance.",
    }


def export_one(
    checkpoint_path: Path,
    args: argparse.Namespace,
    device: torch.device,
    test_input: torch.Tensor,
    export_dir: Path,
) -> dict[str, Any]:
    import onnx
    import onnxruntime as ort

    model, cfg, n_params = load_checkpoint_model(checkpoint_path, device)
    model.eval()
    wrapped = FullOutputWrapper(model).to(device).eval()
    base = int(cfg.get("base", checkpoint_base(checkpoint_path) or -1))
    seed = checkpoint_seed(checkpoint_path, cfg)
    variant = str(cfg.get("variant") or cfg.get("model_type") or f"base{base}")
    onnx_path = export_dir / f"base{base}_seed{seed}_len{args.sample_length}.onnx"
    dummy = test_input[:1].to(device)
    dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}} if args.dynamic_batch else None

    export_start = time.perf_counter()
    torch.onnx.export(
        wrapped,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )
    export_seconds = time.perf_counter() - export_start
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    with torch.no_grad():
        torch_out = wrapped(test_input.to(device)).detach().cpu().numpy()

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = int(args.cpu_threads)
    session_options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), sess_options=session_options, providers=["CPUExecutionProvider"])
    ort_out = session.run(["output"], {"input": test_input.cpu().numpy()})[0]
    compare = compare_outputs(torch_out, ort_out)
    row: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "onnx_path": str(onnx_path),
        "variant": variant,
        "base": base,
        "checkpoint_seed": seed,
        "trainable_parameters": int(n_params),
        "input_shape": json.dumps(list(test_input.shape)),
        "output_shape": json.dumps(list(torch_out.shape)),
        "onnx_export_success": True,
        "onnx_check_success": True,
        "onnxruntime_inference_success": True,
        "export_seconds": float(export_seconds),
        "opset": int(args.opset),
    }
    row.update(file_size_rows({"checkpoint": checkpoint_path, "onnx": onnx_path}))
    row.update(compare)
    return row


def write_markdown(path: Path, rows: list[dict[str, Any]], info: dict[str, Any], args: argparse.Namespace) -> None:
    lines = [
        "# ONNX Export/Compatibility Audit",
        "",
        "## Scope",
        "",
        "- Export compact checkpoints to ONNX.",
        "- Verify successful ONNX Runtime CPU inference on 512-sample inputs.",
        "- Compare PyTorch and ONNX Runtime outputs numerically.",
        "- Report serialized checkpoint and ONNX model sizes.",
        "- This is a deployment-compatibility check, not edge-device performance evidence.",
        "",
        "## Runtime",
        "",
        f"- Platform: `{info['platform']}`",
        f"- Python: `{info['python']}`",
        f"- PyTorch: `{info['torch_version']}`",
        f"- ONNX: `{info['onnx_version']}`",
        f"- ONNX Runtime: `{info['onnxruntime_version']}`",
        f"- ONNX Runtime backend: `{info['onnxruntime_backend']}`",
        f"- CPU threads requested: `{info['requested_cpu_threads']}`",
        f"- OS CPU count: `{info['cpu_count_os']}`",
        f"- Sample length: `{args.sample_length}`",
        f"- Test inputs per checkpoint: `{args.n_test_inputs}`",
        "",
        "## Results",
        "",
        "| base | seed | params | checkpoint MiB | ONNX MiB | max abs err | RMSE | clean RMSE | corr | allclose 1e-5 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (int(item["base"]), int(item["checkpoint_seed"]))):
        lines.append(
            f"| {row['base']} | {row['checkpoint_seed']} | {row['trainable_parameters']} | "
            f"{float(row['checkpoint_mib']):.6f} | {float(row['onnx_mib']):.6f} | "
            f"{float(row['max_abs_error']):.3e} | {float(row['rmse']):.3e} | "
            f"{float(row['rmse_clean_channel']):.3e} | {float(row['pearson_full_output']):.9f} | "
            f"{row['allclose_atol_1e_5_rtol_1e_5']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- ONNX export and ONNX Runtime inference success support deployment-format compatibility.",
            "- Numerical agreement should be interpreted as PyTorch-to-ONNX consistency for the tested 512-sample operation.",
            "- These results do not demonstrate real edge-device latency, RAM use, thermal behavior, or sustained embedded throughput.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.set_num_threads(int(args.cpu_threads))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_dir = args.output_dir / "onnx_models"
    export_dir.mkdir(parents=True, exist_ok=True)
    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    bases = parse_int_set(args.bases)
    seeds = parse_int_set(args.checkpoint_seeds)
    checkpoints = expand_selected_checkpoints(args.checkpoint_glob, bases, seeds)
    rng = np.random.default_rng(args.rng_seed)
    test_np = rng.standard_normal((args.n_test_inputs, 1, args.sample_length)).astype(np.float32)
    test_input = torch.from_numpy(test_np)
    print(
        f"[start] run_id={args.run_id} checkpoints={len(checkpoints)} bases={sorted(bases)} "
        f"seeds={sorted(seeds)} device={device} sample_length={args.sample_length}",
        flush=True,
    )

    info = runtime_info(args)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for checkpoint_path in checkpoints:
        print(f"[export_start] {checkpoint_path}", flush=True)
        try:
            row = export_one(checkpoint_path, args, device, test_input, export_dir)
            rows.append(row)
            print(
                f"[export_done] base={row['base']} seed={row['checkpoint_seed']} "
                f"onnx_mib={float(row['onnx_mib']):.4f} max_abs={float(row['max_abs_error']):.3e}",
                flush=True,
            )
        except Exception as exc:
            failure = {
                "checkpoint": str(checkpoint_path),
                "base": checkpoint_base(checkpoint_path),
                "checkpoint_seed": checkpoint_seed_from_path(checkpoint_path),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(failure)
            print(f"[export_error] {failure}", flush=True)

    write_csv(args.output_dir / "ONNX_onnx_export_rows.csv", rows)
    write_csv(args.output_dir / "ONNX_onnx_export_failures.csv", failures)
    summary = {
        "info": {
            "run_id": args.run_id,
            "checkpoint_glob": args.checkpoint_glob,
            "bases": sorted(bases),
            "checkpoint_seeds": sorted(seeds),
            "runtime": info,
            "tflite_status": "not_attempted; ONNX was selected as the deployment-compatible graph format for this audit",
            "guardrail": "Do not equate successful export with demonstrated edge performance.",
        },
        "rows": rows,
        "failures": failures,
    }
    (args.output_dir / "ONNX_onnx_export_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.output_dir / "ONNX_onnx_export_summary.md", rows, info, args)
    if failures:
        raise RuntimeError(f"{len(failures)} export(s) failed; see ONNX_onnx_export_failures.csv")
    print(f"[written] {args.output_dir / 'ONNX_onnx_export_summary.md'}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
