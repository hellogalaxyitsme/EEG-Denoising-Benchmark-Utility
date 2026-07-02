#!/usr/bin/env python3
"""SNR/recipe/artifact-stratified evaluation for mixed-1M denoising checkpoints."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eeg_denoise_benchmark.eval.metrics import (  # noqa: E402
    cc_np,
    mse_np,
    psd_kld_np,
    psd_np,
    psd_wd_np,
    rmse_np,
    s_rrmse_from_psd_np,
    sdr_db_np,
    t_rrmse_np,
)
from eeg_denoise_benchmark.models import TinyDenoiser, count_trainable_parameters  # noqa: E402


METRIC_KEYS = ["CC", "MSE", "RMSE", "T_RRMSE", "S_RRMSE", "SDR", "PSD_KLD", "PSD_WD"]
RECIPE_NAMES = {
    0: "EOG",
    1: "EMG",
    2: "EOG+EMG",
    3: "EMG+LINE",
    4: "EOG+LINE",
    5: "EOG+EMG+LINE",
    6: "EOG+EMG+LINE+ECG",
}
FLAG_KEYS = ["ecg_used", "line_used", "elec_used"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Mixed-1M split root.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint-glob", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--fs", type=float, default=256.0)
    parser.add_argument("--snr-bins", default="-12 -8 -4 0 2")
    parser.add_argument("--max-chunks", type=int, default=None)
    return parser.parse_args()


def float_list(raw: str) -> list[float]:
    values = [float(part) for part in raw.replace(",", " ").split() if part.strip()]
    if len(values) < 2:
        raise ValueError("--snr-bins needs at least two edges")
    if values != sorted(values):
        raise ValueError("--snr-bins must be sorted")
    return values


def natural_key(path: Path) -> tuple[int, int, str]:
    base_match = re.search(r"base(\d+)", str(path))
    seed_match = re.search(r"seed(\d+)", str(path))
    base = int(base_match.group(1)) if base_match else 10**9
    seed = int(seed_match.group(1)) if seed_match else 10**9
    return base, seed, str(path)


def expand_checkpoints(pattern: str) -> list[Path]:
    checkpoints = sorted((Path(path) for path in glob.glob(pattern)), key=natural_key)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints matched: {pattern}")
    return checkpoints


def chunk_paths(data: Path, split: str, max_chunks: int | None) -> list[Path]:
    split_dir = data / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Expected split chunk directory: {split_dir}")
    paths = sorted(split_dir.glob("chunk_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No chunk_*.npz files found in {split_dir}")
    return paths[:max_chunks] if max_chunks is not None else paths


def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> tuple[TinyDenoiser, dict[str, Any], int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError(f"Checkpoint does not contain model_state: {checkpoint_path}")
    cfg = dict(checkpoint.get("cfg", {}))
    model = TinyDenoiser(
        base=int(cfg.get("base", 16)),
        extra_bottleneck_blocks=int(cfg.get("extra_blocks", 2)),
        use_dwt=bool(cfg.get("use_dwt", True)),
        use_gate=bool(cfg.get("use_gate", True)),
        use_attn=bool(cfg.get("use_attn", True)),
        use_artifact_head=bool(cfg.get("use_artifact_head", True)),
        conv_block=str(cfg.get("conv_block", "ds")),
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    model.eval()
    cfg.setdefault("seed", checkpoint.get("seed", ""))
    return model, cfg, count_trainable_parameters(model)


def checkpoint_seed(checkpoint_path: Path, cfg: dict[str, Any]) -> int | str:
    if "seed" in cfg and cfg["seed"] != "":
        return int(cfg["seed"])
    match = re.search(r"seed(\d+)", str(checkpoint_path))
    return int(match.group(1)) if match else ""


@torch.inference_mode()
def infer_clean(model: torch.nn.Module, noisy: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    predictions = np.zeros_like(noisy, dtype=np.float32)
    for start in range(0, len(noisy), batch_size):
        batch = torch.from_numpy(noisy[start : start + batch_size]).float().unsqueeze(1).to(device)
        output = model(batch)
        predictions[start : start + batch_size] = output[:, 0, :].detach().cpu().numpy()
    return predictions


def per_sample_metrics(target: np.ndarray, prediction: np.ndarray, fs: float) -> dict[str, np.ndarray]:
    target = np.asarray(target, dtype=np.float32)
    prediction = np.asarray(prediction, dtype=np.float32)
    if target.shape != prediction.shape:
        raise ValueError(f"target/prediction shape mismatch: {target.shape} vs {prediction.shape}")
    _, p_target = psd_np(target, fs=fs)
    _, p_prediction = psd_np(prediction, fs=fs)
    return {
        "CC": cc_np(target, prediction).astype(np.float64),
        "MSE": mse_np(target, prediction).astype(np.float64),
        "RMSE": rmse_np(target, prediction).astype(np.float64),
        "T_RRMSE": t_rrmse_np(target, prediction).astype(np.float64),
        "S_RRMSE": s_rrmse_from_psd_np(p_target, p_prediction).astype(np.float64),
        "SDR": sdr_db_np(target, prediction).astype(np.float64),
        "PSD_KLD": psd_kld_np(p_target, p_prediction).astype(np.float64),
        "PSD_WD": psd_wd_np(p_target, p_prediction).astype(np.float64),
    }


class MetricAccumulator:
    def __init__(self) -> None:
        self.n = 0
        self.sums = {key: 0.0 for key in METRIC_KEYS}

    def update(self, metrics: dict[str, np.ndarray], mask: np.ndarray) -> None:
        count = int(np.sum(mask))
        if count <= 0:
            return
        self.n += count
        for key in METRIC_KEYS:
            self.sums[key] += float(np.sum(metrics[key][mask]))

    def row(self) -> dict[str, Any]:
        row: dict[str, Any] = {"n_samples": self.n}
        for key in METRIC_KEYS:
            row[key] = self.sums[key] / self.n if self.n else float("nan")
        return row


def snr_bin_labels(edges: list[float]) -> list[str]:
    labels: list[str] = []
    for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        close = "]" if index == len(edges) - 2 else ")"
        labels.append(f"[{lo:g},{hi:g}{close}")
    return labels


def stratum_masks(z: np.lib.npyio.NpzFile, snr_edges: list[float]) -> list[tuple[str, str, str, np.ndarray]]:
    snr = np.asarray(z["snr_db"], dtype=np.float32)
    masks: list[tuple[str, str, str, np.ndarray]] = [("all", "all", "all", np.ones(snr.shape[0], dtype=bool))]

    labels = snr_bin_labels(snr_edges)
    for index, label in enumerate(labels):
        lo, hi = snr_edges[index], snr_edges[index + 1]
        if index == len(labels) - 1:
            mask = (snr >= lo) & (snr <= hi)
        else:
            mask = (snr >= lo) & (snr < hi)
        masks.append(("snr_bin", label, label, mask))

    if "recipe_id" in z.files:
        recipe = np.asarray(z["recipe_id"])
        for recipe_id in sorted(int(value) for value in np.unique(recipe)):
            label = RECIPE_NAMES.get(recipe_id, f"recipe_{recipe_id}")
            masks.append(("recipe", str(recipe_id), label, recipe == recipe_id))

    for flag in FLAG_KEYS:
        if flag in z.files:
            values = np.asarray(z[flag]).astype(np.int64)
            for value in [0, 1]:
                masks.append(("artifact_flag", f"{flag}={value}", f"{flag}={value}", values == value))

    if all(flag in z.files for flag in FLAG_KEYS):
        count = sum(np.asarray(z[flag]).astype(np.int64) for flag in FLAG_KEYS)
        for value in sorted(int(v) for v in np.unique(count)):
            masks.append(("aux_artifact_count", str(value), f"{value} aux artifacts", count == value))

    return masks


def metric_row(
    *,
    kind: str,
    stratum_type: str,
    stratum_id: str,
    stratum_label: str,
    accumulator: MetricAccumulator,
    base: int | str = "",
    train_seed: int | str = "",
    trainable_parameters: int | str = "",
    checkpoint: str = "",
) -> dict[str, Any]:
    row = {
        "kind": kind,
        "stratum_type": stratum_type,
        "stratum_id": stratum_id,
        "stratum_label": stratum_label,
        "base": base,
        "train_seed": train_seed,
        "trainable_parameters": trainable_parameters,
        "checkpoint": checkpoint,
    }
    row.update(accumulator.row())
    return row


def aggregate_model_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["kind"] != "model":
            continue
        grouped[
            (
                str(row["stratum_type"]),
                str(row["stratum_id"]),
                str(row["stratum_label"]),
                int(row["base"]),
            )
        ].append(row)

    aggregates: list[dict[str, Any]] = []
    for (stratum_type, stratum_id, stratum_label, base), items in sorted(
        grouped.items(), key=lambda item: stratum_sort_key(item[0][0], item[0][1], item[0][3])
    ):
        seeds = sorted(int(item["train_seed"]) for item in items if item["train_seed"] != "")
        aggregate: dict[str, Any] = {
            "kind": "model_aggregate",
            "stratum_type": stratum_type,
            "stratum_id": stratum_id,
            "stratum_label": stratum_label,
            "base": base,
            "n_checkpoints": len(items),
            "train_seeds": " ".join(str(seed) for seed in seeds),
            "n_samples_per_checkpoint": int(items[0]["n_samples"]),
            "trainable_parameters": int(items[0]["trainable_parameters"]),
        }
        for key in METRIC_KEYS:
            values = [float(item[key]) for item in items]
            aggregate[f"{key}_mean"] = float(statistics.mean(values))
            aggregate[f"{key}_std"] = float(statistics.stdev(values)) if len(values) > 1 else 0.0
        aggregates.append(aggregate)
    return aggregates


def stratum_sort_key(stratum_type: str, stratum_id: str, base: int | str = 0) -> tuple[int, float, str, int]:
    type_order = {
        "all": 0,
        "snr_bin": 1,
        "recipe": 2,
        "artifact_flag": 3,
        "aux_artifact_count": 4,
    }.get(stratum_type, 99)
    numeric = 0.0
    if stratum_type == "snr_bin":
        match = re.match(r"^\[?(-?\d+(?:\.\d+)?),", stratum_id)
        numeric = float(match.group(1)) if match else 0.0
    elif stratum_type in {"recipe", "aux_artifact_count"}:
        try:
            numeric = float(stratum_id)
        except ValueError:
            numeric = 0.0
    else:
        numeric = 0.0
    base_i = int(base) if str(base).strip() else 0
    return type_order, numeric, str(stratum_id), base_i


def add_noisy_deltas(
    aggregates: list[dict[str, Any]],
    noisy_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    noisy_lookup = {
        (row["stratum_type"], row["stratum_id"]): row
        for row in noisy_rows
        if row["kind"] == "noisy_baseline"
    }
    for row in aggregates:
        noisy = noisy_lookup.get((row["stratum_type"], row["stratum_id"]))
        if noisy is None:
            continue
        for key in METRIC_KEYS:
            row[f"{key}_delta_vs_noisy"] = float(row[f"{key}_mean"]) - float(noisy[key])
    return aggregates


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path,
    *,
    info: dict[str, Any],
    noisy_rows: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
) -> None:
    lines = ["# Mixed-1M Stratified Evaluation", ""]
    lines.append(
        "Protocol: full mixed-1M test split, stratified by SNR bins, artifact recipe, "
        "artifact flags, and auxiliary artifact count."
    )
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append(f"- Data: `{info['data']}` split `{info['split']}`.")
    lines.append(f"- Run ID: `{info['run_id']}`.")
    lines.append(f"- Checkpoints: `{info['checkpoint_glob']}`.")
    lines.append(f"- Chunks evaluated: `{info['n_chunks']}`; samples: `{info['n_samples']}`.")
    lines.append(f"- SNR bins: `{info['snr_bins']}`.")
    lines.append(f"- Metrics are averaged within each stratum; model aggregates are mean +/- sample SD over training seeds.")
    lines.append("")

    lines.append("## Noisy Baseline By SNR")
    lines.append("")
    lines.append("| SNR bin | n | CC | RMSE | SDR | PSD KLD |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in noisy_rows:
        if row["stratum_type"] != "snr_bin":
            continue
        lines.append(
            f"| {row['stratum_label']} | {row['n_samples']} | {row['CC']:.6f} | "
            f"{row['RMSE']:.6f} | {row['SDR']:.6f} | {row['PSD_KLD']:.6f} |"
        )
    lines.append("")

    for section_type, title in [
        ("snr_bin", "Model Aggregate By SNR"),
        ("recipe", "Model Aggregate By Recipe"),
        ("artifact_flag", "Model Aggregate By Artifact Flag"),
        ("aux_artifact_count", "Model Aggregate By Auxiliary Artifact Count"),
    ]:
        rows = [row for row in aggregates if row["stratum_type"] == section_type]
        if not rows:
            continue
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Stratum | Base | n ckpt | Seeds | Params | CC | RMSE | SDR | PSD KLD | Delta CC vs noisy | Delta SDR vs noisy |")
        lines.append("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in rows:
            lines.append(
                f"| {row['stratum_label']} | {row['base']} | {row['n_checkpoints']} | {row['train_seeds']} | "
                f"{row['trainable_parameters']} | "
                f"{row['CC_mean']:.6f} +/- {row['CC_std']:.6f} | "
                f"{row['RMSE_mean']:.6f} +/- {row['RMSE_std']:.6f} | "
                f"{row['SDR_mean']:.6f} +/- {row['SDR_std']:.6f} | "
                f"{row['PSD_KLD_mean']:.6f} +/- {row['PSD_KLD_std']:.6f} | "
                f"{row.get('CC_delta_vs_noisy', float('nan')):+.6f} | "
                f"{row.get('SDR_delta_vs_noisy', float('nan')):+.6f} |"
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    snr_edges = float_list(args.snr_bins)
    paths = chunk_paths(args.data, args.split, args.max_chunks)
    checkpoints = expand_checkpoints(args.checkpoint_glob)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[start] run_id={args.run_id} device={device} chunks={len(paths)} checkpoints={len(checkpoints)}")
    for checkpoint in checkpoints:
        print(f"  - {checkpoint}")

    noisy_accumulators: dict[tuple[str, str, str], MetricAccumulator] = defaultdict(MetricAccumulator)
    model_accumulators: dict[tuple[str, str, str, str], MetricAccumulator] = defaultdict(MetricAccumulator)
    checkpoint_meta: dict[str, dict[str, Any]] = {}
    n_samples = 0

    for checkpoint_path in checkpoints:
        model, cfg, n_params = load_checkpoint_model(checkpoint_path, device)
        base = int(cfg.get("base", -1))
        train_seed = checkpoint_seed(checkpoint_path, cfg)
        checkpoint_key = str(checkpoint_path)
        checkpoint_meta[checkpoint_key] = {
            "base": base,
            "train_seed": train_seed,
            "trainable_parameters": n_params,
        }
        print(f"[eval_start] base={base} seed={train_seed} params={n_params}")
        for chunk_index, chunk_path in enumerate(paths, start=1):
            z = np.load(chunk_path, allow_pickle=True)
            noisy = np.asarray(z["Y"], dtype=np.float32)
            clean = np.asarray(z["X"], dtype=np.float32)
            if checkpoint_path == checkpoints[0]:
                n_samples += int(clean.shape[0])
                noisy_metrics = per_sample_metrics(clean, noisy, fs=args.fs)
                for stratum_type, stratum_id, stratum_label, mask in stratum_masks(z, snr_edges):
                    noisy_accumulators[(stratum_type, stratum_id, stratum_label)].update(noisy_metrics, mask)
            prediction = infer_clean(model, noisy, device=device, batch_size=args.batch_size)
            metrics = per_sample_metrics(clean, prediction, fs=args.fs)
            for stratum_type, stratum_id, stratum_label, mask in stratum_masks(z, snr_edges):
                model_accumulators[(checkpoint_key, stratum_type, stratum_id, stratum_label)].update(metrics, mask)
            print(f"[chunk] base={base} seed={train_seed} {chunk_index}/{len(paths)} {chunk_path.name}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[eval_done] base={base} seed={train_seed}")

    noisy_rows = [
        metric_row(
            kind="noisy_baseline",
            stratum_type=stratum_type,
            stratum_id=stratum_id,
            stratum_label=stratum_label,
            accumulator=accumulator,
        )
        for (stratum_type, stratum_id, stratum_label), accumulator in sorted(
            noisy_accumulators.items(),
            key=lambda item: stratum_sort_key(item[0][0], item[0][1]),
        )
    ]
    model_rows: list[dict[str, Any]] = []
    for (checkpoint_key, stratum_type, stratum_id, stratum_label), accumulator in sorted(
        model_accumulators.items(),
        key=lambda item: (
            stratum_sort_key(item[0][1], item[0][2]),
            natural_key(Path(item[0][0])),
        ),
    ):
        meta = checkpoint_meta[checkpoint_key]
        model_rows.append(
            metric_row(
                kind="model",
                stratum_type=stratum_type,
                stratum_id=stratum_id,
                stratum_label=stratum_label,
                accumulator=accumulator,
                base=meta["base"],
                train_seed=meta["train_seed"],
                trainable_parameters=meta["trainable_parameters"],
                checkpoint=checkpoint_key,
            )
        )

    aggregates = add_noisy_deltas(aggregate_model_rows(model_rows), noisy_rows)
    info = {
        "run_id": args.run_id,
        "data": str(args.data),
        "split": args.split,
        "checkpoint_glob": args.checkpoint_glob,
        "device": str(device),
        "batch_size": int(args.batch_size),
        "fs": float(args.fs),
        "snr_bins": snr_edges,
        "n_chunks": len(paths),
        "n_samples": n_samples,
        "recipe_names": RECIPE_NAMES,
        "flag_keys": FLAG_KEYS,
    }
    summary = {
        "info": info,
        "noisy_rows": noisy_rows,
        "model_rows": model_rows,
        "aggregates": aggregates,
    }

    json_path = args.output_dir / "mixed1m_stratified_summary.json"
    rows_csv_path = args.output_dir / "mixed1m_stratified_rows.csv"
    noisy_csv_path = args.output_dir / "mixed1m_stratified_noisy_baseline.csv"
    aggregate_csv_path = args.output_dir / "mixed1m_stratified_aggregate.csv"
    md_path = args.output_dir / "mixed1m_stratified_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(rows_csv_path, model_rows)
    write_csv(noisy_csv_path, noisy_rows)
    write_csv(aggregate_csv_path, aggregates)
    write_markdown(md_path, info=info, noisy_rows=noisy_rows, aggregates=aggregates)
    print(f"[written] {json_path}")
    print(f"[written] {rows_csv_path}")
    print(f"[written] {noisy_csv_path}")
    print(f"[written] {aggregate_csv_path}")
    print(f"[written] {md_path}")
    print("[done]")


if __name__ == "__main__":
    main()
