#!/usr/bin/env python3
"""BCI IV-2a multichannel denoising control for CSP+LDA.

This control addresses the single-channel-to-multichannel confound in the
downstream utility-gap analysis. It compares the existing channel-wise denoising
protocol with a small multichannel denoiser trained on all 22 EEG channels
jointly, then evaluates matched CSP+LDA on the denoised train/evaluation
sessions.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.evaluate_bci2a_downstream_csp_lda import (  # noqa: E402
    FS_MODEL,
    METRIC_KEYS,
    bandpass_epochs,
    checkpoint_seed,
    denoise_epochs,
    evaluate_classifier,
    expand_checkpoints,
    fit_classifier,
    load_bci2a_trials,
    load_checkpoint_model,
    make_noisy_epochs,
)
from eeg_denoise_benchmark.models import count_trainable_parameters  # noqa: E402


DEFAULT_SUBJECTS = [f"A{index:02d}" for index in range(1, 10)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bci-dir",
        type=Path,
        default=Path("data/bci_iv_2a"),
    )
    parser.add_argument("--subjects", default=",".join(DEFAULT_SUBJECTS))
    parser.add_argument("--single-channel-checkpoint-glob", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--bases", default="6,16")
    parser.add_argument("--train-seeds", default="42,43,44")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--denoise-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--snr-min-db", type=float, default=-6.0)
    parser.add_argument("--snr-max-db", type=float, default=2.0)
    parser.add_argument("--trial-start-sec", type=float, default=2.5)
    parser.add_argument("--trial-stop-sec", type=float, default=6.0)
    parser.add_argument("--bandpass-low-hz", type=float, default=8.0)
    parser.add_argument("--bandpass-high-hz", type=float, default=30.0)
    parser.add_argument("--csp-components", type=int, default=8)
    parser.add_argument("--eog-index", type=int, default=22)
    parser.add_argument("--include-artifact-trials", action="store_true")
    parser.add_argument(
        "--skip-single-channel",
        action="store_true",
        help="Only train/evaluate multichannel denoisers.",
    )
    return parser.parse_args()


class ECA1D(nn.Module):
    def __init__(self, channels: int, k: int = 3) -> None:
        super().__init__()
        del channels
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg(x).transpose(1, 2)
        y = self.conv(y).transpose(1, 2)
        return x * torch.sigmoid(y)


class DSConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, *, dilation: int = 1, k: int = 9) -> None:
        super().__init__()
        pad = (k // 2) * dilation
        self.dw = nn.Conv1d(cin, cin, k, padding=pad, dilation=dilation, groups=cin, bias=False)
        self.pw = nn.Conv1d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm1d(cout)
        self.act = nn.SiLU()
        self.attn = ECA1D(cout)
        self.res = cin == cout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.attn(self.act(self.bn(self.pw(self.dw(x)))))
        return y + x if self.res else y


class MultiChannelLDNControl(nn.Module):
    """Small covariance-aware DSConv U-Net for 22-channel EEG trials."""

    def __init__(self, n_channels: int = 22, base: int = 6) -> None:
        super().__init__()
        h1 = base * 4
        h2 = base * 8
        h3 = base * 16
        self.stem = nn.Conv1d(n_channels, h1, 1, bias=False)
        self.e1 = DSConvBlock(h1, h1, dilation=1)
        self.down1 = nn.Conv1d(h1, h2, 4, stride=2, padding=1, bias=False)
        self.e2 = DSConvBlock(h2, h2, dilation=2)
        self.down2 = nn.Conv1d(h2, h3, 4, stride=2, padding=1, bias=False)
        self.b = nn.Sequential(
            DSConvBlock(h3, h3, dilation=4),
            DSConvBlock(h3, h3, dilation=8),
            DSConvBlock(h3, h3, dilation=16),
        )
        self.up2 = nn.ConvTranspose1d(h3, h2, 4, stride=2, padding=1, bias=False)
        self.d2 = DSConvBlock(h2 + h2, h2, dilation=2)
        self.up1 = nn.ConvTranspose1d(h2, h1, 4, stride=2, padding=1, bias=False)
        self.d1 = DSConvBlock(h1 + h1, h1, dilation=1)
        self.out = nn.Conv1d(h1, n_channels, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(x)
        s1 = self.e1(x0)
        x1 = self.down1(s1)
        s2 = self.e2(x1)
        x2 = self.down2(s2)
        xb = self.b(x2)
        u2 = self.up2(xb)
        if u2.shape[-1] != s2.shape[-1]:
            u2 = F.interpolate(u2, size=s2.shape[-1], mode="linear", align_corners=False)
        d2 = self.d2(torch.cat([u2, s2], dim=1))
        u1 = self.up1(d2)
        if u1.shape[-1] != s1.shape[-1]:
            u1 = F.interpolate(u1, size=s1.shape[-1], mode="linear", align_corners=False)
        d1 = self.d1(torch.cat([u1, s1], dim=1))
        return self.out(d1)


def parse_int_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer.")
    return values


def parse_subjects(raw: str) -> list[str]:
    subjects = [item.strip().upper() for item in raw.split(",") if item.strip()]
    if not subjects:
        raise ValueError("Expected at least one subject.")
    return subjects


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.stdev(values))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _average_tied_ranks(abs_values: list[float]) -> list[float]:
    order = sorted(range(len(abs_values)), key=lambda i: abs_values[i])
    ranks = [0.0] * len(abs_values)
    pos = 0
    while pos < len(order):
        end = pos + 1
        while end < len(order) and abs(abs_values[order[end]] - abs_values[order[pos]]) < 1e-12:
            end += 1
        avg_rank = (pos + 1 + end) / 2.0
        for idx in range(pos, end):
            ranks[order[idx]] = avg_rank
        pos = end
    return ranks


def exact_wilcoxon_p(deltas: list[float], alternative: str = "less") -> float:
    nz = [float(d) for d in deltas if abs(float(d)) > 1e-12]
    if not nz:
        return 1.0
    ranks = _average_tied_ranks([abs(d) for d in nz])
    observed = sum(rank for rank, delta in zip(ranks, nz) if delta > 0)
    null_sums = []
    for signs in itertools.product([0, 1], repeat=len(nz)):
        null_sums.append(sum(rank for rank, sign in zip(ranks, signs) if sign))
    if alternative == "less":
        return sum(1 for value in null_sums if value <= observed + 1e-12) / len(null_sums)
    if alternative == "greater":
        return sum(1 for value in null_sums if value >= observed - 1e-12) / len(null_sums)
    lower = sum(1 for value in null_sums if value <= observed + 1e-12) / len(null_sums)
    upper = sum(1 for value in null_sums if value >= observed - 1e-12) / len(null_sums)
    return min(1.0, 2.0 * min(lower, upper))


def bh_fdr(rows: list[dict[str, Any]], p_key: str, out_key: str) -> None:
    indexed = [(float(row[p_key]), idx) for idx, row in enumerate(rows) if row.get(p_key, "") != ""]
    indexed.sort()
    m = len(indexed)
    adjusted: dict[int, float] = {}
    prev = 1.0
    for rank_from_end, (p_value, index) in enumerate(reversed(indexed), start=1):
        rank = m - rank_from_end + 1
        q_value = min(prev, p_value * m / rank)
        prev = q_value
        adjusted[index] = min(1.0, q_value)
    for idx, row in enumerate(rows):
        row[out_key] = adjusted.get(idx, "")


def load_subject_arrays(args: argparse.Namespace, subject: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    train_mat = args.bci_dir / f"{subject}T.mat"
    test_mat = args.bci_dir / f"{subject}E.mat"
    train_clean, train_eog, y_train, train_info = load_bci2a_trials(
        train_mat,
        trial_start_sec=args.trial_start_sec,
        trial_stop_sec=args.trial_stop_sec,
        include_artifact_trials=args.include_artifact_trials,
        eog_index=args.eog_index,
    )
    test_clean, test_eog, y_test, test_info = load_bci2a_trials(
        test_mat,
        trial_start_sec=args.trial_start_sec,
        trial_stop_sec=args.trial_stop_sec,
        include_artifact_trials=args.include_artifact_trials,
        eog_index=args.eog_index,
    )
    train_noisy, _train_artifact, train_noise = make_noisy_epochs(
        train_clean,
        train_eog,
        seed=args.seed + 1000,
        snr_min_db=args.snr_min_db,
        snr_max_db=args.snr_max_db,
    )
    test_noisy, _test_artifact, test_noise = make_noisy_epochs(
        test_clean,
        test_eog,
        seed=args.seed,
        snr_min_db=args.snr_min_db,
        snr_max_db=args.snr_max_db,
    )
    info = {
        "subject": subject,
        "train": train_info,
        "test": test_info,
        "train_noise": train_noise,
        "test_noise": test_noise,
    }
    return train_clean, test_clean, train_noisy, test_noisy, y_train, y_test, info


def global_trial_normalize(noisy: np.ndarray, clean: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    sigma = np.std(noisy, axis=(1, 2), keepdims=True).astype(np.float32) + 1e-8
    noisy_norm = (noisy / sigma).astype(np.float32)
    clean_norm = None if clean is None else (clean / sigma).astype(np.float32)
    return noisy_norm, clean_norm, sigma.astype(np.float32)


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 2e-3) -> torch.Tensor:
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps**2) - eps)


def train_multichannel_model(
    *,
    train_noisy: np.ndarray,
    train_clean: np.ndarray,
    base: int,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    x_norm, y_norm, _sigma = global_trial_normalize(train_noisy, train_clean)
    dataset = TensorDataset(torch.from_numpy(x_norm).float(), torch.from_numpy(y_norm).float())
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator, drop_last=False)

    model = MultiChannelLDNControl(n_channels=train_noisy.shape[1], base=base).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(1, args.epochs + 1):
        losses: list[float] = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = charbonnier_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch in {1, args.epochs} or epoch % max(1, args.epochs // 4) == 0:
            history.append({"epoch": float(epoch), "loss": float(statistics.mean(losses))})
    model.eval()
    meta = {
        "train_seed": int(seed),
        "base": int(base),
        "trainable_parameters": int(count_trainable_parameters(model)),
        "final_train_loss": float(history[-1]["loss"]),
        "history": history,
    }
    return model, meta


@torch.no_grad()
def denoise_multichannel(model: nn.Module, noisy_epochs: np.ndarray, *, device: torch.device, batch_size: int) -> np.ndarray:
    x_norm, _target, sigma = global_trial_normalize(noisy_epochs, None)
    outputs = np.zeros_like(x_norm, dtype=np.float32)
    for start in range(0, x_norm.shape[0], batch_size):
        batch = torch.from_numpy(x_norm[start : start + batch_size]).float().to(device)
        pred = model(batch).detach().cpu().numpy().astype(np.float32)
        outputs[start : start + batch_size] = pred
    return (outputs * sigma).astype(np.float32)


def filter_checkpoints(checkpoints: list[Path], bases: set[int]) -> list[Path]:
    selected: list[Path] = []
    for checkpoint in checkpoints:
        model, cfg, _n_params = load_checkpoint_model(checkpoint, torch.device("cpu"))
        del model
        try:
            base = int(cfg.get("base", -1))
        except Exception:
            base = -1
        if base in bases:
            selected.append(checkpoint)
    return selected


def evaluate_subject(
    args: argparse.Namespace,
    subject: str,
    bases: list[int],
    train_seeds: list[int],
    single_checkpoints: list[Path],
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_clean, test_clean, train_noisy, test_noisy, y_train, y_test, info = load_subject_arrays(args, subject)

    train_noisy_features = bandpass_epochs(train_noisy, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz)
    test_noisy_features = bandpass_epochs(test_noisy, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz)
    noisy_classifier = fit_classifier(args, train_noisy_features, y_train)
    baseline_row = evaluate_classifier(noisy_classifier, test_noisy_features, y_test, condition="noisy_noisy")
    baseline_row.update({"subject": subject, "denoiser_scope": "baseline", "base": "", "train_seed": ""})
    print(f"[baseline] subject={subject} acc={baseline_row['accuracy']:.6f}", flush=True)

    rows: list[dict[str, Any]] = [baseline_row]

    if not args.skip_single_channel:
        for checkpoint_path in single_checkpoints:
            model, cfg, n_params = load_checkpoint_model(checkpoint_path, device)
            base = int(cfg.get("base", -1))
            if base not in bases:
                continue
            train_seed = checkpoint_seed(checkpoint_path, cfg)
            print(f"[single_start] subject={subject} base={base} seed={train_seed}", flush=True)
            train_den = denoise_epochs(model, train_noisy, device=device, batch_size=args.denoise_batch_size)
            test_den = denoise_epochs(model, test_noisy, device=device, batch_size=args.denoise_batch_size)
            train_features = bandpass_epochs(train_den, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz)
            test_features = bandpass_epochs(test_den, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz)
            classifier = fit_classifier(args, train_features, y_train)
            row = evaluate_classifier(
                classifier,
                test_features,
                y_test,
                condition="denoised_denoised",
                variant="single_channel_ldn_sweep",
                base=base,
                train_seed=train_seed,
                trainable_parameters=n_params,
                checkpoint=str(checkpoint_path),
            )
            row["subject"] = subject
            row["denoiser_scope"] = "single_channel"
            rows.append(row)
            print(f"[single_result] subject={subject} base={base} seed={train_seed} acc={row['accuracy']:.6f}", flush=True)

    for base in bases:
        for train_seed in train_seeds:
            print(f"[multi_train_start] subject={subject} base={base} seed={train_seed}", flush=True)
            model, meta = train_multichannel_model(
                train_noisy=train_noisy,
                train_clean=train_clean,
                base=base,
                seed=train_seed,
                args=args,
                device=device,
            )
            train_den = denoise_multichannel(model, train_noisy, device=device, batch_size=args.denoise_batch_size)
            test_den = denoise_multichannel(model, test_noisy, device=device, batch_size=args.denoise_batch_size)
            train_features = bandpass_epochs(train_den, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz)
            test_features = bandpass_epochs(test_den, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz)
            classifier = fit_classifier(args, train_features, y_train)
            row = evaluate_classifier(
                classifier,
                test_features,
                y_test,
                condition="denoised_denoised",
                variant="multichannel_ldn_control",
                base=base,
                train_seed=train_seed,
                trainable_parameters=meta["trainable_parameters"],
                checkpoint="",
            )
            row["subject"] = subject
            row["denoiser_scope"] = "multichannel"
            row["final_train_loss"] = meta["final_train_loss"]
            rows.append(row)
            print(
                f"[multi_result] subject={subject} base={base} seed={train_seed} "
                f"params={meta['trainable_parameters']} acc={row['accuracy']:.6f} loss={meta['final_train_loss']:.6f}",
                flush=True,
            )

    subject_aggregate = aggregate_subject(rows)
    return rows, subject_aggregate, info


def aggregate_subject(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = next(row for row in rows if row["denoiser_scope"] == "baseline")
    baseline_acc = float(baseline["accuracy"])
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["denoiser_scope"] == "baseline":
            continue
        grouped[(str(row["denoiser_scope"]), str(row["variant"]), str(row["base"]))].append(row)

    out: list[dict[str, Any]] = []
    for (scope, variant, base), items in sorted(grouped.items(), key=lambda kv: (kv[0][0], int(kv[0][2]))):
        entry: dict[str, Any] = {
            "subject": str(items[0]["subject"]),
            "denoiser_scope": scope,
            "variant": variant,
            "base": base,
            "condition": "denoised_denoised",
            "n": len(items),
            "train_seeds": " ".join(str(item["train_seed"]) for item in sorted(items, key=lambda r: int(r["train_seed"]))),
            "trainable_parameters": int(items[0]["trainable_parameters"]),
            "baseline_accuracy": baseline_acc,
        }
        for key in METRIC_KEYS:
            vals = [float(item[key]) for item in items]
            m, sd = mean_std(vals)
            entry[f"{key}_mean"] = m
            entry[f"{key}_std"] = sd
        entry["delta_accuracy_vs_noisy_noisy"] = float(entry["accuracy_mean"]) - baseline_acc
        out.append(entry)
    return out


def aggregate_all(rows: list[dict[str, Any]], subject_aggregates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_rows = [row for row in rows if row["denoiser_scope"] == "baseline"]
    baseline_acc_mean, baseline_acc_std = mean_std([float(row["accuracy"]) for row in baseline_rows])
    baseline = [{
        "condition": "noisy_noisy",
        "n_subjects": len(baseline_rows),
        "accuracy_mean": baseline_acc_mean,
        "accuracy_std": baseline_acc_std,
    }]

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in subject_aggregates:
        grouped[(str(row["denoiser_scope"]), str(row["variant"]), str(row["base"]))].append(row)

    out: list[dict[str, Any]] = []
    for (scope, variant, base), items in sorted(grouped.items(), key=lambda kv: (kv[0][0], int(kv[0][2]))):
        acc = [float(item["accuracy_mean"]) for item in items]
        deltas = [float(item["delta_accuracy_vs_noisy_noisy"]) for item in items]
        acc_mean, acc_sd = mean_std(acc)
        delta_mean, delta_sd = mean_std(deltas)
        out.append({
            "denoiser_scope": scope,
            "variant": variant,
            "base": base,
            "condition": "denoised_denoised",
            "n_subjects": len(items),
            "subjects": " ".join(sorted(str(item["subject"]) for item in items)),
            "n_train_seeds_per_subject": int(items[0]["n"]),
            "trainable_parameters": int(items[0]["trainable_parameters"]),
            "accuracy_subject_mean": acc_mean,
            "accuracy_subject_std": acc_sd,
            "delta_accuracy_vs_noisy_noisy_subject_mean": delta_mean,
            "delta_accuracy_vs_noisy_noisy_subject_std": delta_sd,
            "subjects_below_noisy_noisy": sum(1 for d in deltas if d < 0),
            "subjects_above_noisy_noisy": sum(1 for d in deltas if d > 0),
            "wilcoxon_p_denoised_lt_noisy": exact_wilcoxon_p(deltas, "less"),
            "wilcoxon_p_denoised_gt_noisy": exact_wilcoxon_p(deltas, "greater"),
        })
    bh_fdr(out, "wilcoxon_p_denoised_lt_noisy", "bh_fdr_p_denoised_lt_noisy")
    return baseline, out


def write_markdown(path: Path, baseline_rows: list[dict[str, Any]], aggregate_rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = ["# BCI IV-2a Multichannel Denoising CSP+LDA Control", ""]
    lines.append("This run compares channel-wise controlled backbone denoising with a jointly trained 22-channel DSConv U-Net control.")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append(f"- Subjects: `{args.subjects}`.")
    lines.append(f"- Bases: `{args.bases}`.")
    lines.append(f"- Multichannel training seeds: `{args.train_seeds}`.")
    lines.append(f"- Multichannel epochs: `{args.epochs}`.")
    lines.append(f"- Trial window: `{args.trial_start_sec:.2f}`--`{args.trial_stop_sec:.2f}` s.")
    lines.append(f"- Synthetic EOG SNR: `{args.snr_min_db:.1f}` to `{args.snr_max_db:.1f}` dB.")
    lines.append(f"- CSP band/components: `{args.bandpass_low_hz:.1f}`--`{args.bandpass_high_hz:.1f}` Hz, `{args.csp_components}` components.")
    lines.append("")
    lines.append("## Baseline")
    lines.append("")
    lines.append("| Condition | n subjects | Accuracy |")
    lines.append("|---|---:|---:|")
    for row in baseline_rows:
        lines.append(f"| {row['condition']} | {row['n_subjects']} | {row['accuracy_mean']:.6f} +/- {row['accuracy_std']:.6f} |")
    lines.append("")
    lines.append("## Denoised/Denoised Results")
    lines.append("")
    lines.append("| Scope | Variant | Base | n subjects | Params | Accuracy | Delta vs noisy/noisy | Below noisy | Wilcoxon p lower | BH-FDR p |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in aggregate_rows:
        lines.append(
            f"| {row['denoiser_scope']} | {row['variant']} | {row['base']} | {row['n_subjects']} | "
            f"{row['trainable_parameters']} | "
            f"{row['accuracy_subject_mean']:.6f} +/- {row['accuracy_subject_std']:.6f} | "
            f"{row['delta_accuracy_vs_noisy_noisy_subject_mean']:+.6f} +/- {row['delta_accuracy_vs_noisy_noisy_subject_std']:.6f} | "
            f"{row['subjects_below_noisy_noisy']} | {row['wilcoxon_p_denoised_lt_noisy']:.6f} | "
            f"{row.get('bh_fdr_p_denoised_lt_noisy', '')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    subjects = parse_subjects(args.subjects)
    bases = parse_int_list(args.bases)
    train_seeds = parse_int_list(args.train_seeds)

    all_checkpoints = expand_checkpoints(args.single_channel_checkpoint_glob)
    single_checkpoints = [] if args.skip_single_channel else filter_checkpoints(all_checkpoints, set(bases))
    print(
        f"[start] run_id={args.run_id} subjects={subjects} bases={bases} seeds={train_seeds} "
        f"single_checkpoints={len(single_checkpoints)} device={device}",
        flush=True,
    )

    all_rows: list[dict[str, Any]] = []
    subject_aggregates: list[dict[str, Any]] = []
    subject_info: list[dict[str, Any]] = []
    for subject in subjects:
        print(f"[subject_start] {subject}", flush=True)
        rows, aggregates, info = evaluate_subject(args, subject, bases, train_seeds, single_checkpoints, device)
        all_rows.extend(rows)
        subject_aggregates.extend(aggregates)
        subject_info.append(info)
        print(f"[subject_done] {subject}", flush=True)

    baseline_rows, aggregate_rows = aggregate_all(all_rows, subject_aggregates)
    summary = {
        "info": {
            "run_id": args.run_id,
            "subjects": subjects,
            "bases": bases,
            "train_seeds": train_seeds,
            "single_channel_checkpoint_glob": args.single_channel_checkpoint_glob,
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "batch_size": int(args.batch_size),
            "denoise_batch_size": int(args.denoise_batch_size),
            "include_artifact_trials": bool(args.include_artifact_trials),
        },
        "subject_info": subject_info,
        "baseline_aggregate": baseline_rows,
        "scope_aggregate": aggregate_rows,
    }
    (args.output_dir / "bci2a_multichannel_denoiser_csp_control_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "bci2a_multichannel_denoiser_csp_control_rows.csv", all_rows)
    write_csv(args.output_dir / "bci2a_multichannel_denoiser_csp_control_subject_aggregates.csv", subject_aggregates)
    write_csv(args.output_dir / "bci2a_multichannel_denoiser_csp_control_baseline_aggregate.csv", baseline_rows)
    write_csv(args.output_dir / "bci2a_multichannel_denoiser_csp_control_scope_aggregate.csv", aggregate_rows)
    write_markdown(args.output_dir / "bci2a_multichannel_denoiser_csp_control_summary.md", baseline_rows, aggregate_rows, args)
    print(f"[written] {args.output_dir / 'bci2a_multichannel_denoiser_csp_control_summary.md'}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
