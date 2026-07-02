#!/usr/bin/env python3
"""BCI IV-2a CSP+LDA sensitivity to post-denoising calibration.

This ablation tests whether the CSP+LDA utility gap is mainly caused by simple
amplitude scaling or by cross-channel covariance distortion after independent
single-channel denoising. It reuses the BCI IV-2a CSP+LDA protocol and
evaluates matched denoised/denoised classification after several deterministic
post-processing transforms.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

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


DEFAULT_SUBJECTS = [f"A{index:02d}" for index in range(1, 10)]
CALIBRATIONS = [
    "none",
    "trial_zscore",
    "train_channel_zscore",
    "raw_variance_rescale",
    "covariance_recolor",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bci-dir",
        type=Path,
        default=Path("data/bci_iv_2a"),
    )
    parser.add_argument("--subjects", default=",".join(DEFAULT_SUBJECTS))
    parser.add_argument("--checkpoint-glob", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--protocol", choices=["synthetic_eog", "real_raw"], default="synthetic_eog")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--snr-min-db", type=float, default=-6.0)
    parser.add_argument("--snr-max-db", type=float, default=2.0)
    parser.add_argument("--trial-start-sec", type=float, default=2.5)
    parser.add_argument("--trial-stop-sec", type=float, default=6.0)
    parser.add_argument("--bandpass-low-hz", type=float, default=8.0)
    parser.add_argument("--bandpass-high-hz", type=float, default=30.0)
    parser.add_argument("--csp-components", type=int, default=8)
    parser.add_argument("--eog-index", type=int, default=22)
    parser.add_argument(
        "--include-artifact-trials",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include BCI IV-2a artifact-flagged trials. Use true for real-data relevance; default false matches the synthetic downstream protocol.",
    )
    parser.add_argument(
        "--calibrations",
        default=",".join(CALIBRATIONS),
        help=f"Comma-separated calibration names. Options: {', '.join(CALIBRATIONS)}.",
    )
    return parser.parse_args()


def parse_subjects(raw: str) -> list[str]:
    subjects = [item.strip().upper() for item in raw.split(",") if item.strip()]
    if not subjects:
        raise ValueError("At least one subject is required.")
    return subjects


def parse_calibrations(raw: str) -> list[str]:
    items = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(items) - set(CALIBRATIONS))
    if unknown:
        raise ValueError(f"Unknown calibrations: {unknown}; valid={CALIBRATIONS}")
    return items


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


def _flatten_epochs(epochs: np.ndarray) -> np.ndarray:
    # (trials, channels, time) -> (samples, channels)
    return np.transpose(epochs, (0, 2, 1)).reshape(-1, epochs.shape[1])


def _symmetric_matrix_sqrt(matrix: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    vals, vecs = np.linalg.eigh(matrix.astype(np.float64))
    vals = np.maximum(vals, eps)
    return (vecs * np.sqrt(vals)) @ vecs.T


def _symmetric_matrix_inv_sqrt(matrix: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    vals, vecs = np.linalg.eigh(matrix.astype(np.float64))
    vals = np.maximum(vals, eps)
    return (vecs * (1.0 / np.sqrt(vals))) @ vecs.T


def _covariance_matrix(epochs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    flat = _flatten_epochs(epochs)
    flat = flat - np.mean(flat, axis=0, keepdims=True)
    cov = np.cov(flat, rowvar=False)
    return cov + np.eye(cov.shape[0]) * eps


def _apply_linear_channel_transform(epochs: np.ndarray, transform: np.ndarray, mean: np.ndarray | None = None) -> np.ndarray:
    out = epochs.astype(np.float64, copy=True)
    if mean is not None:
        out = out - mean[None, :, None]
    # For every time sample, channel vector row is multiplied by transform.T.
    out = np.einsum("nct,dc->ndt", out, transform)
    return out.astype(np.float32)


def calibrate_epochs(
    *,
    name: str,
    train_denoised: np.ndarray,
    test_denoised: np.ndarray,
    train_reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    eps = 1e-6
    meta: dict[str, Any] = {"calibration": name}
    if name == "none":
        return train_denoised.astype(np.float32), test_denoised.astype(np.float32), meta

    if name == "trial_zscore":
        def transform(x: np.ndarray) -> np.ndarray:
            mu = np.mean(x, axis=-1, keepdims=True)
            sd = np.std(x, axis=-1, keepdims=True) + eps
            return ((x - mu) / sd).astype(np.float32)
        return transform(train_denoised), transform(test_denoised), meta

    if name == "train_channel_zscore":
        mu = np.mean(train_denoised, axis=(0, 2), keepdims=True)
        sd = np.std(train_denoised, axis=(0, 2), keepdims=True) + eps
        meta["train_channel_std_mean"] = float(np.mean(sd))
        return ((train_denoised - mu) / sd).astype(np.float32), ((test_denoised - mu) / sd).astype(np.float32), meta

    if name == "raw_variance_rescale":
        den_sd = np.std(train_denoised, axis=(0, 2), keepdims=True) + eps
        ref_sd = np.std(train_reference, axis=(0, 2), keepdims=True) + eps
        scale = ref_sd / den_sd
        meta["scale_mean"] = float(np.mean(scale))
        meta["scale_min"] = float(np.min(scale))
        meta["scale_max"] = float(np.max(scale))
        return (train_denoised * scale).astype(np.float32), (test_denoised * scale).astype(np.float32), meta

    if name == "covariance_recolor":
        den_mean = np.mean(train_denoised, axis=(0, 2))
        ref_mean = np.mean(train_reference, axis=(0, 2))
        cov_den = _covariance_matrix(train_denoised)
        cov_ref = _covariance_matrix(train_reference)
        transform = _symmetric_matrix_sqrt(cov_ref) @ _symmetric_matrix_inv_sqrt(cov_den)
        train_centered = _apply_linear_channel_transform(train_denoised, transform, den_mean) + ref_mean[None, :, None]
        test_centered = _apply_linear_channel_transform(test_denoised, transform, den_mean) + ref_mean[None, :, None]
        meta["cov_den_trace"] = float(np.trace(cov_den))
        meta["cov_ref_trace"] = float(np.trace(cov_ref))
        return train_centered.astype(np.float32), test_centered.astype(np.float32), meta

    raise ValueError(name)


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

    if args.protocol == "synthetic_eog":
        train_input, _train_artifact, train_noise = make_noisy_epochs(
            train_clean,
            train_eog,
            seed=args.seed + 1000,
            snr_min_db=args.snr_min_db,
            snr_max_db=args.snr_max_db,
        )
        test_input, _test_artifact, test_noise = make_noisy_epochs(
            test_clean,
            test_eog,
            seed=args.seed,
            snr_min_db=args.snr_min_db,
            snr_max_db=args.snr_max_db,
        )
        baseline_train = train_input
        baseline_test = test_input
        baseline_condition = "noisy_noisy"
    else:
        train_input = train_clean
        test_input = test_clean
        baseline_train = train_clean
        baseline_test = test_clean
        baseline_condition = "raw_raw"
        train_noise = {}
        test_noise = {}

    info = {
        "subject": subject,
        "train": train_info,
        "test": test_info,
        "train_noise": train_noise,
        "test_noise": test_noise,
        "baseline_condition": baseline_condition,
    }
    return train_input, test_input, y_train, y_test, baseline_train, baseline_test, info


def evaluate_subject(
    args: argparse.Namespace,
    subject: str,
    checkpoints: list[Path],
    calibrations: list[str],
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_input, test_input, y_train, y_test, baseline_train, baseline_test, info = load_subject_arrays(args, subject)
    baseline_train_features = bandpass_epochs(baseline_train, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz)
    baseline_test_features = bandpass_epochs(baseline_test, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz)
    baseline_classifier = fit_classifier(args, baseline_train_features, y_train)
    baseline_row = evaluate_classifier(
        baseline_classifier,
        baseline_test_features,
        y_test,
        condition=info["baseline_condition"],
    )
    baseline_row["subject"] = subject
    baseline_row["calibration"] = "baseline"
    print(f"[baseline] subject={subject} condition={baseline_row['condition']} acc={baseline_row['accuracy']:.6f}", flush=True)

    rows: list[dict[str, Any]] = [baseline_row]
    for checkpoint_path in checkpoints:
        model, cfg, n_params = load_checkpoint_model(checkpoint_path, device)
        base = cfg.get("base", "")
        variant = str(cfg.get("variant") or cfg.get("model_type") or f"base{base}")
        train_seed = checkpoint_seed(checkpoint_path, cfg)
        print(f"[denoise] subject={subject} base={base} seed={train_seed}", flush=True)
        train_denoised = denoise_epochs(model, train_input, device=device, batch_size=args.batch_size)
        test_denoised = denoise_epochs(model, test_input, device=device, batch_size=args.batch_size)

        for calibration in calibrations:
            cal_train, cal_test, cal_meta = calibrate_epochs(
                name=calibration,
                train_denoised=train_denoised,
                test_denoised=test_denoised,
                train_reference=baseline_train,
            )
            train_features = bandpass_epochs(cal_train, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz)
            test_features = bandpass_epochs(cal_test, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz)
            classifier = fit_classifier(args, train_features, y_train)
            row = evaluate_classifier(
                classifier,
                test_features,
                y_test,
                condition="denoised_denoised",
                variant=variant,
                base=base,
                train_seed=train_seed,
                trainable_parameters=n_params,
                checkpoint=str(checkpoint_path),
            )
            row["subject"] = subject
            row["calibration"] = calibration
            row.update({f"cal_{key}": value for key, value in cal_meta.items() if key != "calibration"})
            rows.append(row)
            print(
                f"[result] subject={subject} base={base} seed={train_seed} calibration={calibration} "
                f"acc={row['accuracy']:.6f}",
                flush=True,
            )

    subject_aggregates = aggregate_subject(rows)
    return rows, subject_aggregates, info


def aggregate_subject(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = next(row for row in rows if row["calibration"] == "baseline")
    baseline_acc = float(baseline["accuracy"])
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["calibration"] == "baseline":
            continue
        grouped[(str(row["variant"]), str(row["base"]), str(row["calibration"]), str(row["condition"]))].append(row)

    out: list[dict[str, Any]] = []
    for (variant, base, calibration, condition), items in sorted(grouped.items(), key=lambda kv: (kv[0][2], kv[0][1], kv[0][0])):
        entry: dict[str, Any] = {
            "subject": str(items[0]["subject"]),
            "condition": condition,
            "variant": variant,
            "base": base,
            "calibration": calibration,
            "n": len(items),
            "train_seeds": " ".join(str(item["train_seed"]) for item in sorted(items, key=lambda r: str(r["train_seed"]))),
            "trainable_parameters": int(items[0]["trainable_parameters"]),
            "baseline_condition": str(baseline["condition"]),
            "baseline_accuracy": baseline_acc,
        }
        for key in METRIC_KEYS:
            vals = [float(item[key]) for item in items]
            m, sd = mean_std(vals)
            entry[f"{key}_mean"] = m
            entry[f"{key}_std"] = sd
        entry["delta_accuracy_vs_baseline"] = float(entry["accuracy_mean"]) - baseline_acc
        out.append(entry)
    return out


def aggregate_all_subjects(rows: list[dict[str, Any]], subject_aggregates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_rows = [row for row in rows if row["calibration"] == "baseline"]
    baseline_values = [float(row["accuracy"]) for row in baseline_rows]
    baseline_mean, baseline_sd = mean_std(baseline_values)
    baseline = [{
        "condition": str(baseline_rows[0]["condition"]) if baseline_rows else "",
        "n_subjects": len(baseline_rows),
        "accuracy_mean": baseline_mean,
        "accuracy_std": baseline_sd,
    }]

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in subject_aggregates:
        grouped[(str(row["variant"]), str(row["base"]), str(row["calibration"]), str(row["condition"]))].append(row)

    out: list[dict[str, Any]] = []
    for (variant, base, calibration, condition), items in sorted(grouped.items(), key=lambda kv: (kv[0][2], kv[0][1], kv[0][0])):
        acc = [float(item["accuracy_mean"]) for item in items]
        deltas = [float(item["delta_accuracy_vs_baseline"]) for item in items]
        acc_mean, acc_sd = mean_std(acc)
        delta_mean, delta_sd = mean_std(deltas)
        out.append({
            "condition": condition,
            "variant": variant,
            "base": base,
            "calibration": calibration,
            "n_subjects": len(items),
            "subjects": " ".join(sorted(str(item["subject"]) for item in items)),
            "n_checkpoints_per_subject": int(items[0]["n"]),
            "trainable_parameters": int(items[0]["trainable_parameters"]),
            "accuracy_subject_mean": acc_mean,
            "accuracy_subject_std": acc_sd,
            "delta_accuracy_vs_baseline_subject_mean": delta_mean,
            "delta_accuracy_vs_baseline_subject_std": delta_sd,
            "subjects_below_baseline": sum(1 for d in deltas if d < 0),
            "subjects_above_baseline": sum(1 for d in deltas if d > 0),
            "wilcoxon_p_denoised_lt_baseline": exact_wilcoxon_p(deltas, "less"),
            "wilcoxon_p_denoised_gt_baseline": exact_wilcoxon_p(deltas, "greater"),
        })

    bh_fdr(out, "wilcoxon_p_denoised_lt_baseline", "bh_fdr_p_denoised_lt_baseline")
    return baseline, out


def write_markdown(path: Path, baseline_rows: list[dict[str, Any]], width_rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = ["# BCI IV-2a CSP+LDA Calibration Sensitivity", ""]
    lines.append(f"Protocol: `{args.protocol}`.")
    lines.append("Calibration variants: " + ", ".join(f"`{item}`" for item in parse_calibrations(args.calibrations)) + ".")
    lines.append("")
    lines.append("## Baseline")
    lines.append("")
    lines.append("| Condition | n subjects | Accuracy |")
    lines.append("|---|---:|---:|")
    for row in baseline_rows:
        lines.append(f"| {row['condition']} | {row['n_subjects']} | {row['accuracy_mean']:.6f} +/- {row['accuracy_std']:.6f} |")
    lines.append("")
    lines.append("## Matched Denoised/Denoised Calibration Results")
    lines.append("")
    lines.append("| Calibration | Base | Variant | n subjects | Accuracy | Delta vs baseline | Below baseline | Wilcoxon p (lower) | BH-FDR p |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for row in width_rows:
        lines.append(
            f"| {row['calibration']} | {row['base']} | {row['variant']} | {row['n_subjects']} | "
            f"{row['accuracy_subject_mean']:.6f} +/- {row['accuracy_subject_std']:.6f} | "
            f"{row['delta_accuracy_vs_baseline_subject_mean']:+.6f} +/- {row['delta_accuracy_vs_baseline_subject_std']:.6f} | "
            f"{row['subjects_below_baseline']} | {row['wilcoxon_p_denoised_lt_baseline']:.6f} | "
            f"{row.get('bh_fdr_p_denoised_lt_baseline', '')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibrations = parse_calibrations(args.calibrations)
    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    subjects = parse_subjects(args.subjects)
    checkpoints = sorted(expand_checkpoints(args.checkpoint_glob))

    print(
        f"[start] run_id={args.run_id} protocol={args.protocol} subjects={subjects} "
        f"checkpoints={len(checkpoints)} calibrations={calibrations} device={device}",
        flush=True,
    )
    all_rows: list[dict[str, Any]] = []
    subject_aggregates: list[dict[str, Any]] = []
    infos: list[dict[str, Any]] = []
    for subject in subjects:
        rows, aggregates, info = evaluate_subject(args, subject, checkpoints, calibrations, device)
        all_rows.extend(rows)
        subject_aggregates.extend(aggregates)
        infos.append(info)

    baseline_rows, width_rows = aggregate_all_subjects(all_rows, subject_aggregates)
    summary = {
        "info": {
            "run_id": args.run_id,
            "protocol": args.protocol,
            "subjects": subjects,
            "checkpoint_glob": args.checkpoint_glob,
            "calibrations": calibrations,
            "include_artifact_trials": bool(args.include_artifact_trials),
            "trial_start_sec": float(args.trial_start_sec),
            "trial_stop_sec": float(args.trial_stop_sec),
            "bandpass_low_hz": float(args.bandpass_low_hz),
            "bandpass_high_hz": float(args.bandpass_high_hz),
            "csp_components": int(args.csp_components),
        },
        "subject_info": infos,
        "baseline_aggregate": baseline_rows,
        "width_aggregate": width_rows,
    }
    (args.output_dir / "bci2a_csp_lda_calibration_sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "bci2a_csp_lda_calibration_sensitivity_rows.csv", all_rows)
    write_csv(args.output_dir / "bci2a_csp_lda_calibration_sensitivity_subject_aggregates.csv", subject_aggregates)
    write_csv(args.output_dir / "bci2a_csp_lda_calibration_sensitivity_baseline_aggregate.csv", baseline_rows)
    write_csv(args.output_dir / "bci2a_csp_lda_calibration_sensitivity_width_aggregate.csv", width_rows)
    write_markdown(args.output_dir / "bci2a_csp_lda_calibration_sensitivity_summary.md", baseline_rows, width_rows, args)
    print(f"[written] {args.output_dir / 'bci2a_csp_lda_calibration_sensitivity_summary.md'}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
