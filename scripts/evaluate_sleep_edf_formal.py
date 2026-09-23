#!/usr/bin/env python3
"""Formal Sleep-EDF downstream/reconstruction denoising suite.

This script keeps the human subject as the inferential unit. Contamination
seeds and denoiser checkpoints are nuisance repetitions that are averaged
within subject before hypothesis testing.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from eeg_denoise_benchmark.checkpoints import load_model_from_checkpoint  # noqa: E402
from eeg_denoise_benchmark.eval.metrics import cc_np, psd_np, rmse_np, s_rrmse_from_psd_np, sdr_db_np, t_rrmse_np  # noqa: E402
from eeg_denoise_benchmark.models import count_trainable_parameters  # noqa: E402
from scripts.evaluate_bci2a_downstream_csp_lda import denoise_epochs  # noqa: E402
from scripts.sleep_edf_downstream_helpers import (  # noqa: E402
    checkpoint_base,
    checkpoint_seed_from_path,
    discover_recordings,
    evaluate,
    expand_model_specs,
    infer_model_label,
    load_recording,
    make_noisy_epochs,
    parse_int_list,
    select_subjects,
)


METRICS = ["CC", "RMSE", "T_RRMSE", "S_RRMSE", "SDR"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sleep-edf-dir", type=Path, required=True, help="Path to the licensed Sleep-EDF sleep-cassette directory.")
    parser.add_argument("--subjects", default="autoall", help="Comma list, autoN, or autoall.")
    parser.add_argument("--model-spec", action="append", default=[])
    parser.add_argument("--model-spec-file", type=Path, default=None)
    parser.add_argument("--train-seeds", default="1042,1043,1044,1045,1046")
    parser.add_argument("--test-seeds", default="42,43,44,45,46")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--snr-min-db", type=float, default=-6.0)
    parser.add_argument("--snr-max-db", type=float, default=2.0)
    parser.add_argument("--trim-wake-min", type=float, default=30.0)
    parser.add_argument("--max-train-epochs", type=int, default=1600)
    parser.add_argument("--max-test-epochs", type=int, default=1600)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260816)
    parser.add_argument("--include-real-denoised", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_model_specs(args: argparse.Namespace) -> list[str]:
    specs = list(args.model_spec)
    if args.model_spec_file is not None:
        for line in args.model_spec_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                specs.append(line)
    if not specs:
        raise ValueError("Provide at least one --model-spec or --model-spec-file entry.")
    return specs


def subject_selector(raw: str, grouped: dict[str, list[dict[str, Path | str]]]) -> list[str]:
    if raw.lower() in {"autoall", "all"}:
        return sorted(grouped)
    return select_subjects(raw, grouped)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    preferred = [
        "run_id",
        "dataset",
        "subject",
        "train_recording",
        "test_recording",
        "condition",
        "denoiser_label",
        "model_family",
        "base",
        "checkpoint_seed",
        "checkpoint",
        "trainable_parameters",
        "train_contamination_seed",
        "test_contamination_seed",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "cohen_kappa",
        "delta_accuracy_vs_noisy_noisy",
        "delta_balanced_accuracy_vs_noisy_noisy",
        "delta_accuracy_vs_raw_raw",
        "delta_balanced_accuracy_vs_raw_raw",
    ]
    for key in preferred:
        if any(key in row for row in rows):
            fieldnames.append(key)
            seen.add(key)
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def model_family(label: str) -> str:
    if label.startswith("base") and label[4:].isdigit():
        return "base_width"
    return label


def bh_fdr(p_values: list[float]) -> list[float]:
    n = len(p_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda idx: p_values[idx])
    adjusted = [1.0] * n
    running = 1.0
    for rank_from_end, idx in enumerate(reversed(order), start=1):
        rank = n - rank_from_end + 1
        running = min(running, p_values[idx] * n / rank)
        adjusted[idx] = min(1.0, running)
    return adjusted


def percentile(values: list[float], p: float) -> float:
    xs = sorted(values)
    if not xs:
        return float("nan")
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def bootstrap_ci(values: list[float], *, n_resamples: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    boot = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples)]
    return percentile(boot, 0.025), percentile(boot, 0.975)


def wilcoxon_less(values: list[float]) -> float:
    nz = [float(value) for value in values if abs(float(value)) > 1e-12]
    if not nz:
        return 1.0
    try:
        return float(wilcoxon(nz, alternative="less", zero_method="wilcox").pvalue)
    except Exception:
        return float("nan")


def wilcoxon_two_sided(values: list[float]) -> float:
    nz = [float(value) for value in values if abs(float(value)) > 1e-12]
    if not nz:
        return 1.0
    try:
        return float(wilcoxon(nz, alternative="two-sided", zero_method="wilcox").pvalue)
    except Exception:
        return float("nan")


def metric_row_values(clean: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    target = clean.reshape(-1, clean.shape[-1])
    pred = estimate.reshape(-1, estimate.shape[-1])
    _, p_target = psd_np(target, fs=256)
    _, p_pred = psd_np(pred, fs=256)
    return {
        "CC": float(np.mean(cc_np(target, pred))),
        "RMSE": float(np.mean(rmse_np(target, pred))),
        "T_RRMSE": float(np.mean(t_rrmse_np(target, pred))),
        "S_RRMSE": float(np.mean(s_rrmse_from_psd_np(p_target, p_pred))),
        "SDR": float(np.mean(sdr_db_np(target, pred))),
    }


def make_downstream_row(
    *,
    args: argparse.Namespace,
    subject: str,
    train_recording: str,
    test_recording: str,
    condition: str,
    metrics: dict[str, Any],
    raw_reference: dict[str, Any] | None = None,
    noisy_reference: dict[str, Any] | None = None,
    train_seed: int | str = "",
    test_seed: int | str = "",
    denoiser_label: str = "",
    base: int | str = "",
    checkpoint_seed: int | str = "",
    checkpoint: str = "",
    params: int | str = "",
) -> dict[str, Any]:
    row = dict(metrics)
    row.update(
        {
            "run_id": args.run_id,
            "dataset": "Sleep_EDF_SC",
            "subject": subject,
            "train_recording": train_recording,
            "test_recording": test_recording,
            "condition": condition,
            "denoiser_label": denoiser_label,
            "model_family": model_family(denoiser_label),
            "base": base,
            "checkpoint_seed": checkpoint_seed,
            "checkpoint": checkpoint,
            "trainable_parameters": params,
            "train_contamination_seed": train_seed,
            "test_contamination_seed": test_seed,
        }
    )
    if noisy_reference is not None:
        row["delta_accuracy_vs_noisy_noisy"] = float(row["accuracy"]) - float(noisy_reference["accuracy"])
        row["delta_balanced_accuracy_vs_noisy_noisy"] = float(row["balanced_accuracy"]) - float(noisy_reference["balanced_accuracy"])
    else:
        row["delta_accuracy_vs_noisy_noisy"] = ""
        row["delta_balanced_accuracy_vs_noisy_noisy"] = ""
    if raw_reference is not None:
        row["delta_accuracy_vs_raw_raw"] = float(row["accuracy"]) - float(raw_reference["accuracy"])
        row["delta_balanced_accuracy_vs_raw_raw"] = float(row["balanced_accuracy"]) - float(raw_reference["balanced_accuracy"])
    else:
        row["delta_accuracy_vs_raw_raw"] = ""
        row["delta_balanced_accuracy_vs_raw_raw"] = ""
    return row


def make_reconstruction_row(
    *,
    args: argparse.Namespace,
    subject: str,
    split: str,
    condition: str,
    values: dict[str, float],
    train_seed: int,
    test_seed: int,
    denoiser_label: str = "",
    base: int | str = "",
    checkpoint_seed: int | str = "",
    checkpoint: str = "",
    params: int | str = "",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_id": args.run_id,
        "dataset": "Sleep_EDF_SC",
        "subject": subject,
        "split": split,
        "condition": condition,
        "denoiser_label": denoiser_label,
        "model_family": model_family(denoiser_label),
        "base": base,
        "checkpoint_seed": checkpoint_seed,
        "checkpoint": checkpoint,
        "trainable_parameters": params,
        "train_contamination_seed": train_seed,
        "test_contamination_seed": test_seed,
    }
    row.update(values)
    return row


def completed_key(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(row.get("subject", "")),
        str(row.get("condition", "")),
        str(row.get("denoiser_label", "")),
        str(row.get("base", "")),
        str(row.get("checkpoint_seed", "")),
        str(row.get("train_contamination_seed", "")),
        str(row.get("test_contamination_seed", "")),
    )


def summarize_downstream(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    processed = [row for row in rows if row.get("condition") in {"denoised_denoised", "real_denoised_denoised"}]
    summaries: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in processed:
        delta_key = "delta_balanced_accuracy_vs_noisy_noisy" if row.get("condition") == "denoised_denoised" else "delta_balanced_accuracy_vs_raw_raw"
        if row.get(delta_key, "") == "":
            continue
        groups[(str(row["condition"]), str(row["denoiser_label"]))].append(row)

    p_indices: list[int] = []
    p_values: list[float] = []
    for (condition, label), items in sorted(groups.items()):
        by_subject: dict[str, list[float]] = defaultdict(list)
        by_checkpoint_subject: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        by_contam_subject: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        base_value = ""
        params_values: list[int] = []
        for row in items:
            delta_key = "delta_balanced_accuracy_vs_noisy_noisy" if condition == "denoised_denoised" else "delta_balanced_accuracy_vs_raw_raw"
            delta = float(row[delta_key])
            subject = str(row["subject"])
            ckpt = str(row.get("checkpoint_seed", ""))
            contam = f"{row.get('train_contamination_seed', '')}/{row.get('test_contamination_seed', '')}"
            by_subject[subject].append(delta)
            by_checkpoint_subject[subject][ckpt].append(delta)
            by_contam_subject[subject][contam].append(delta)
            if row.get("base", "") != "":
                base_value = str(row["base"])
            if str(row.get("trainable_parameters", "")).isdigit():
                params_values.append(int(row["trainable_parameters"]))
        subject_values = [float(statistics.mean(values)) for _subject, values in sorted(by_subject.items())]
        lo, hi = bootstrap_ci(subject_values, n_resamples=args.bootstrap_resamples, seed=args.bootstrap_seed + sum(ord(ch) for ch in condition + label))
        p = wilcoxon_less(subject_values)
        checkpoint_sds = []
        contamination_sds = []
        for subject in sorted(by_subject):
            ckpt_means = [float(statistics.mean(values)) for _key, values in sorted(by_checkpoint_subject[subject].items())]
            contam_means = [float(statistics.mean(values)) for _key, values in sorted(by_contam_subject[subject].items()) if _key != "/"]
            if len(ckpt_means) > 1:
                checkpoint_sds.append(float(statistics.stdev(ckpt_means)))
            if len(contam_means) > 1:
                contamination_sds.append(float(statistics.stdev(contam_means)))
        row = {
            "run_id": args.run_id,
            "condition": condition,
            "denoiser_label": label,
            "model_family": model_family(label),
            "base": base_value,
            "n_subjects": len(subject_values),
            "n_rows": len(items),
            "n_checkpoint_seeds": len({str(item.get("checkpoint_seed", "")) for item in items if str(item.get("checkpoint_seed", "")) != ""}),
            "n_contamination_pairs": len({f"{item.get('train_contamination_seed', '')}/{item.get('test_contamination_seed', '')}" for item in items if str(item.get("train_contamination_seed", "")) != ""}),
            "trainable_parameters_median": int(statistics.median(params_values)) if params_values else "",
            "mean_delta_balanced_accuracy": float(statistics.mean(subject_values)),
            "median_delta_balanced_accuracy": float(statistics.median(subject_values)),
            "subject_sd": float(statistics.stdev(subject_values)) if len(subject_values) > 1 else 0.0,
            "ci95_low": lo,
            "ci95_high": hi,
            "subjects_below_reference": int(sum(1 for value in subject_values if value < 0)),
            "wilcoxon_p_less": p,
            "mean_within_subject_checkpoint_sd": float(statistics.mean(checkpoint_sds)) if checkpoint_sds else "",
            "mean_within_subject_contamination_sd": float(statistics.mean(contamination_sds)) if contamination_sds else "",
        }
        summaries.append(row)
        p_indices.append(len(summaries) - 1)
        p_values.append(p)

    q_values = bh_fdr(p_values)
    for idx, q in zip(p_indices, q_values):
        summaries[idx]["bh_fdr_q"] = q
    return summaries


def summarize_reconstruction(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("split") != "test":
            continue
        groups[(str(row.get("condition", "")), str(row.get("denoiser_label", "")), str(row.get("base", "")))].append(row)
    out = []
    for (condition, label, base), items in sorted(groups.items()):
        by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in items:
            by_subject[str(row["subject"])].append(row)
        item: dict[str, Any] = {
            "run_id": args.run_id,
            "condition": condition,
            "denoiser_label": label,
            "model_family": model_family(label),
            "base": base,
            "n_subjects": len(by_subject),
            "n_rows": len(items),
        }
        for metric in METRICS:
            subject_values = [float(statistics.mean(float(row[metric]) for row in rows_for_subject)) for _subject, rows_for_subject in sorted(by_subject.items())]
            lo, hi = bootstrap_ci(subject_values, n_resamples=args.bootstrap_resamples, seed=args.bootstrap_seed + sum(ord(ch) for ch in condition + label + metric))
            item[f"{metric}_mean"] = float(statistics.mean(subject_values))
            item[f"{metric}_median"] = float(statistics.median(subject_values))
            item[f"{metric}_ci95_low"] = lo
            item[f"{metric}_ci95_high"] = hi
        out.append(item)
    return out


def metric_utility(rows: list[dict[str, Any]], recon_rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    downstream = [row for row in rows if row.get("condition") == "denoised_denoised" and row.get("delta_balanced_accuracy_vs_noisy_noisy", "") != ""]
    recon_index: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    for row in recon_rows:
        if row.get("split") != "test" or row.get("condition") != "denoised":
            continue
        key = (
            str(row["subject"]),
            str(row["denoiser_label"]),
            str(row.get("base", "")),
            str(row.get("checkpoint_seed", "")),
            str(row.get("train_contamination_seed", "")),
            str(row.get("test_contamination_seed", "")),
        )
        recon_index[key] = row
    joined = []
    for row in downstream:
        key = (
            str(row["subject"]),
            str(row["denoiser_label"]),
            str(row.get("base", "")),
            str(row.get("checkpoint_seed", "")),
            str(row.get("train_contamination_seed", "")),
            str(row.get("test_contamination_seed", "")),
        )
        metric_values = recon_index.get(key)
        if metric_values is None:
            continue
        joined.append((row, metric_values))

    out = []
    p_indices: list[int] = []
    p_values: list[float] = []
    for label in sorted({str(row["denoiser_label"]) for row, _metric_values in joined}):
        label_items = [(row, metric_values) for row, metric_values in joined if str(row["denoiser_label"]) == label]
        for metric in METRICS:
            slopes = []
            xs_pooled = []
            ys_pooled = []
            by_subject: dict[str, list[tuple[float, float]]] = defaultdict(list)
            for row, metric_values in label_items:
                by_subject[str(row["subject"])].append((float(metric_values[metric]), float(row["delta_balanced_accuracy_vs_noisy_noisy"])))
            for subject, pairs in sorted(by_subject.items()):
                if len(pairs) < 2:
                    continue
                xs = np.asarray([pair[0] for pair in pairs], dtype=float)
                ys = np.asarray([pair[1] for pair in pairs], dtype=float)
                if float(np.std(xs)) <= 1e-12:
                    continue
                slope = float(np.polyfit(xs, ys, deg=1)[0])
                slopes.append(slope)
                xs_pooled.extend((xs - float(np.mean(xs))).tolist())
                ys_pooled.extend((ys - float(np.mean(ys))).tolist())
            if not slopes:
                continue
            lo, hi = bootstrap_ci(slopes, n_resamples=args.bootstrap_resamples, seed=args.bootstrap_seed + sum(ord(ch) for ch in label + metric))
            p = wilcoxon_two_sided(slopes)
            pooled_pearson = pearsonr(xs_pooled, ys_pooled).statistic if len(xs_pooled) > 2 and np.std(xs_pooled) > 1e-12 and np.std(ys_pooled) > 1e-12 else float("nan")
            pooled_spearman = spearmanr(xs_pooled, ys_pooled).statistic if len(xs_pooled) > 2 and np.std(xs_pooled) > 1e-12 and np.std(ys_pooled) > 1e-12 else float("nan")
            row = {
                "run_id": args.run_id,
                "denoiser_label": label,
                "model_family": model_family(label),
                "metric": metric,
                "n_subjects_with_slope": len(slopes),
                "mean_subject_slope": float(statistics.mean(slopes)),
                "median_subject_slope": float(statistics.median(slopes)),
                "slope_ci95_low": lo,
                "slope_ci95_high": hi,
                "wilcoxon_p_two_sided": p,
                "pooled_subject_demeaned_pearson_r_descriptive": float(pooled_pearson),
                "pooled_subject_demeaned_spearman_r_descriptive": float(pooled_spearman),
            }
            out.append(row)
            p_indices.append(len(out) - 1)
            p_values.append(p)
    for idx, q in zip(p_indices, bh_fdr(p_values)):
        out[idx]["bh_fdr_q"] = q
    return out


def write_summary_md(args: argparse.Namespace, subjects: list[str], downstream_summary: list[dict[str, Any]]) -> None:
    lines = [
        "# Formal Sleep-EDF Downstream Denoising Suite",
        "",
        "## Protocol",
        "",
        f"- Subjects with paired recordings: `{len(subjects)}`.",
        "- Split: first available Sleep-EDF SC recording for training; second for testing.",
        "- Dataset: Sleep-EDF Expanded Sleep Cassette PSG files with paired hypnograms.",
        "- Task: 5-class sleep staging with labels W, N1, N2, N3, and REM; hypnogram stages 3 and 4 are merged into N3.",
        "- Epoching: non-overlapping 30 s annotated epochs, trimmed to 30 min before first non-wake through 30.5 min after last non-wake.",
        "- Epoch caps: up to 1600 training epochs and 1600 test epochs per subject recording; excess epochs are subsampled without replacement with fixed RNG seeds.",
        "- Signal preprocessing: PSG data are converted to microvolts, selected EEG/EOG channels are resampled to 256 Hz, and no extra classifier bandpass filter is applied.",
        "- Channel rule: use up to two EEG channels by normalized-name priority EEG Fpz-Cz and EEG Pz-Oz or matching aliases; use the first channel containing EOG as the artifact template.",
        "- Feature extraction: per EEG channel, demean each epoch; compute Welch PSD with 4 s segments and 2 s overlap; integrate total power from 0.5-30 Hz and delta/theta/alpha/sigma/beta bandpower.",
        "- Features: log total power; log and relative bandpower for 0.5-4, 4-8, 8-12, 12-16, and 16-30 Hz; mean, standard deviation, 95th-5th percentile range, RMS first difference, and mean absolute amplitude.",
        "- Classifier: scikit-learn Pipeline(StandardScaler(), LogisticRegression(solver='lbfgs', class_weight='balanced', max_iter=2000, random_state=0)).",
        "- Validation scheme: no cross-validation is used for the formal estimate; each subject uses night/session 1 for training and night/session 2 for held-out testing.",
        f"- Synthetic EOG contamination: SNR uniform `{args.snr_min_db}` to `{args.snr_max_db}` dB.",
        "- Synthetic artifact construction: the same-recording EOG epoch is demeaned and scaled separately for each EEG channel to the sampled SNR.",
        f"- Contamination seed pairs: train `{args.train_seeds}`, test `{args.test_seeds}`.",
        "- Denoising: apply each denoiser channel-wise after per-signal standard-deviation normalization and rescale afterward; fixed 512-sample models are applied to 30 s epochs by 50% overlap-add.",
        "- Denoiser provenance: external denoiser checkpoints are applied to Sleep-EDF without Sleep-EDF fine-tuning unless explicitly noted in the model specification.",
        "- Denoiser checkpoints are nuisance repetitions; subject remains the inferential unit.",
        "- CSP/decoder classifier seeds are not applicable here because the sleep-stage classifier is deterministic once inputs are fixed.",
        "- Reconstruction metrics use the same contaminated Sleep-EDF epochs that feed the downstream classifier; original Sleep-EDF EEG is treated as assumed-clean/low-artifact reference, not artifact-free neural ground truth.",
        "",
        "## Downstream Summary",
        "",
        "| condition | denoiser | n | checkpoints | contam pairs | mean Δ bal acc | median Δ | 95% CI | below ref | p | q |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in downstream_summary:
        lines.append(
            f"| {row['condition']} | {row['denoiser_label']} | {row['n_subjects']} | {row['n_checkpoint_seeds']} | {row['n_contamination_pairs']} | "
            f"{float(row['mean_delta_balanced_accuracy']):+.6f} | {float(row['median_delta_balanced_accuracy']):+.6f} | "
            f"[{float(row['ci95_low']):+.6f}, {float(row['ci95_high']):+.6f}] | {row['subjects_below_reference']} | "
            f"{float(row['wilcoxon_p_less']):.6g} | {float(row.get('bh_fdr_q', float('nan'))):.6g} |"
        )
    (args.output_dir / "sleep_edf_formal_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_seeds = parse_int_list(args.train_seeds)
    test_seeds = parse_int_list(args.test_seeds)
    if len(train_seeds) != len(test_seeds):
        raise ValueError("--train-seeds and --test-seeds must have equal length")
    grouped = discover_recordings(args.sleep_edf_dir)
    subjects = subject_selector(args.subjects, grouped)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    specs = read_model_specs(args)
    checkpoint_entries = expand_model_specs(specs)
    models = []
    for entry in checkpoint_entries:
        path = Path(entry["path"])
        model, checkpoint = load_model_from_checkpoint(path, map_location="cpu", strict=True)
        cfg = dict(checkpoint.get("cfg", {})) if isinstance(checkpoint, dict) else {}
        model.to(device)
        model.eval()
        seed = cfg.get("seed", checkpoint_seed_from_path(path))
        base = checkpoint_base(path)
        if base is None and str(cfg.get("base", "")).isdigit():
            base = int(cfg["base"])
        label = str(entry["label"]) or infer_model_label(path)
        models.append(
            {
                "label": label,
                "path": path,
                "model": model,
                "base": int(base) if base is not None else "",
                "checkpoint_seed": int(seed) if str(seed).isdigit() else str(seed or ""),
                "params": int(count_trainable_parameters(model)),
            }
        )

    row_path = args.output_dir / "sleep_edf_formal_downstream_rows.csv"
    recon_path = args.output_dir / "sleep_edf_formal_reconstruction_rows.csv"
    rows: list[dict[str, Any]] = read_csv(row_path) if args.resume else []
    recon_rows: list[dict[str, Any]] = read_csv(recon_path) if args.resume else []
    completed = {completed_key(row) for row in rows}
    recon_completed = {
        (
            str(row.get("subject", "")),
            str(row.get("split", "")),
            str(row.get("condition", "")),
            str(row.get("denoiser_label", "")),
            str(row.get("base", "")),
            str(row.get("checkpoint_seed", "")),
            str(row.get("train_contamination_seed", "")),
            str(row.get("test_contamination_seed", "")),
        )
        for row in recon_rows
    }
    subject_info: list[dict[str, Any]] = []

    print(f"[setup] subjects={len(subjects)} models={len(models)} seed_pairs={len(train_seeds)} device={device}", flush=True)
    for subject in subjects:
        recordings = grouped[subject]
        train_rec, test_rec = recordings[0], recordings[1]
        print(f"[subject] {subject} train={train_rec['recording']} test={test_rec['recording']}", flush=True)
        train_clean, train_eog, y_train, train_info = load_recording(
            Path(train_rec["psg"]),
            Path(train_rec["hyp"]),
            trim_wake_min=args.trim_wake_min,
            max_epochs=args.max_train_epochs,
            rng_seed=args.bootstrap_seed,
        )
        test_clean, test_eog, y_test, test_info = load_recording(
            Path(test_rec["psg"]),
            Path(test_rec["hyp"]),
            trim_wake_min=args.trim_wake_min,
            max_epochs=args.max_test_epochs,
            rng_seed=args.bootstrap_seed + 1,
        )
        train_info["recording"] = str(train_rec["recording"])
        test_info["recording"] = str(test_rec["recording"])
        subject_info.append({"subject": subject, "train": train_info, "test": test_info})
        raw_metrics = None
        raw_key = (subject, "raw_raw", "", "", "", "", "")
        existing_raw = [row for row in rows if completed_key(row) == raw_key]
        if existing_raw:
            raw_metrics = existing_raw[0]
        else:
            raw_metrics = evaluate(train_clean, y_train, test_clean, y_test)
            rows.append(
                make_downstream_row(
                    args=args,
                    subject=subject,
                    train_recording=str(train_rec["recording"]),
                    test_recording=str(test_rec["recording"]),
                    condition="raw_raw",
                    metrics=raw_metrics,
                )
            )
            completed.add(raw_key)
            write_csv(row_path, rows)

        if args.include_real_denoised:
            for item in models:
                key = (subject, "real_denoised_denoised", str(item["label"]), str(item["base"]), str(item["checkpoint_seed"]), "", "")
                if key in completed:
                    continue
                print(f"[real_denoise] {subject} model={item['label']} seed={item['checkpoint_seed']}", flush=True)
                train_den_real = denoise_epochs(item["model"], train_clean, device=device, batch_size=args.batch_size)
                test_den_real = denoise_epochs(item["model"], test_clean, device=device, batch_size=args.batch_size)
                rows.append(
                    make_downstream_row(
                        args=args,
                        subject=subject,
                        train_recording=str(train_rec["recording"]),
                        test_recording=str(test_rec["recording"]),
                        condition="real_denoised_denoised",
                        metrics=evaluate(train_den_real, y_train, test_den_real, y_test),
                        raw_reference=raw_metrics,
                        denoiser_label=item["label"],
                        base=item["base"],
                        checkpoint_seed=item["checkpoint_seed"],
                        checkpoint=str(item["path"]),
                        params=item["params"],
                    )
                )
                completed.add(key)
                write_csv(row_path, rows)

        for pair_index, (train_seed, test_seed) in enumerate(zip(train_seeds, test_seeds), start=1):
            print(f"[seed_pair] {subject} {pair_index}/{len(train_seeds)}", flush=True)
            train_noisy, _train_noise_info = make_noisy_epochs(
                train_clean,
                train_eog,
                seed=train_seed,
                snr_min_db=args.snr_min_db,
                snr_max_db=args.snr_max_db,
            )
            test_noisy, _test_noise_info = make_noisy_epochs(
                test_clean,
                test_eog,
                seed=test_seed,
                snr_min_db=args.snr_min_db,
                snr_max_db=args.snr_max_db,
            )
            noisy_key = (subject, "noisy_noisy", "", "", "", str(train_seed), str(test_seed))
            existing_noisy = [row for row in rows if completed_key(row) == noisy_key]
            if existing_noisy:
                noisy_metrics = existing_noisy[0]
            else:
                noisy_metrics = evaluate(train_noisy, y_train, test_noisy, y_test)
                rows.append(
                    make_downstream_row(
                        args=args,
                        subject=subject,
                        train_recording=str(train_rec["recording"]),
                        test_recording=str(test_rec["recording"]),
                        condition="noisy_noisy",
                        metrics=noisy_metrics,
                        raw_reference=raw_metrics,
                        train_seed=train_seed,
                        test_seed=test_seed,
                    )
                )
                completed.add(noisy_key)
                write_csv(row_path, rows)
            noisy_recon_key = (subject, "test", "noisy", "", "", "", str(train_seed), str(test_seed))
            if noisy_recon_key not in recon_completed:
                recon_rows.append(
                    make_reconstruction_row(
                        args=args,
                        subject=subject,
                        split="test",
                        condition="noisy",
                        values=metric_row_values(test_clean, test_noisy),
                        train_seed=train_seed,
                        test_seed=test_seed,
                    )
                )
                recon_completed.add(noisy_recon_key)
                write_csv(recon_path, recon_rows)

            for item in models:
                key = (
                    subject,
                    "denoised_denoised",
                    str(item["label"]),
                    str(item["base"]),
                    str(item["checkpoint_seed"]),
                    str(train_seed),
                    str(test_seed),
                )
                if key in completed:
                    continue
                print(f"[denoise] {subject} model={item['label']} base={item['base']} seed={item['checkpoint_seed']} pair={pair_index}", flush=True)
                train_den = denoise_epochs(item["model"], train_noisy, device=device, batch_size=args.batch_size)
                test_den = denoise_epochs(item["model"], test_noisy, device=device, batch_size=args.batch_size)
                rows.append(
                    make_downstream_row(
                        args=args,
                        subject=subject,
                        train_recording=str(train_rec["recording"]),
                        test_recording=str(test_rec["recording"]),
                        condition="denoised_denoised",
                        metrics=evaluate(train_den, y_train, test_den, y_test),
                        raw_reference=raw_metrics,
                        noisy_reference=noisy_metrics,
                        train_seed=train_seed,
                        test_seed=test_seed,
                        denoiser_label=item["label"],
                        base=item["base"],
                        checkpoint_seed=item["checkpoint_seed"],
                        checkpoint=str(item["path"]),
                        params=item["params"],
                    )
                )
                completed.add(key)
                rec_key = (subject, "test", "denoised", str(item["label"]), str(item["base"]), str(item["checkpoint_seed"]), str(train_seed), str(test_seed))
                if rec_key not in recon_completed:
                    recon_rows.append(
                        make_reconstruction_row(
                            args=args,
                            subject=subject,
                            split="test",
                            condition="denoised",
                            values=metric_row_values(test_clean, test_den),
                            train_seed=train_seed,
                            test_seed=test_seed,
                            denoiser_label=item["label"],
                            base=item["base"],
                            checkpoint_seed=item["checkpoint_seed"],
                            checkpoint=str(item["path"]),
                            params=item["params"],
                        )
                    )
                    recon_completed.add(rec_key)
                write_csv(row_path, rows)
                write_csv(recon_path, recon_rows)

    downstream_summary = summarize_downstream(rows, args)
    reconstruction_summary = summarize_reconstruction(recon_rows, args)
    metric_utility_rows = metric_utility(rows, recon_rows, args)
    write_csv(args.output_dir / "sleep_edf_formal_downstream_summary.csv", downstream_summary)
    write_csv(args.output_dir / "sleep_edf_formal_reconstruction_summary.csv", reconstruction_summary)
    write_csv(args.output_dir / "sleep_edf_formal_metric_utility.csv", metric_utility_rows)
    (args.output_dir / "sleep_edf_formal_subject_info.json").write_text(json.dumps(subject_info, indent=2) + "\n", encoding="utf-8")
    write_summary_md(args, subjects, downstream_summary)
    print(f"[done] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
