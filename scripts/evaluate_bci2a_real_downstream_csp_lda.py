#!/usr/bin/env python3
"""BCI IV-2a real-trial CSP+LDA evaluation without synthetic contamination.

This protocol applies denoisers directly to the recorded BCI IV-2a EEG trials
and compares CSP+LDA trained/tested on raw trials against CSP+LDA trained/tested
on denoised trials. It is intended to complement the synthetic EOG downstream
protocol with a real-data utility check.
"""

from __future__ import annotations

import argparse
import csv
import glob
import itertools
import json
import math
import statistics
import sys
import warnings
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
)


DEFAULT_SUBJECTS = [f"A{index:02d}" for index in range(1, 10)]


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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--trial-start-sec", type=float, default=2.5)
    parser.add_argument("--trial-stop-sec", type=float, default=6.0)
    parser.add_argument("--bandpass-low-hz", type=float, default=8.0)
    parser.add_argument("--bandpass-high-hz", type=float, default=30.0)
    parser.add_argument("--csp-components", type=int, default=8)
    parser.add_argument("--eog-index", type=int, default=22)
    parser.add_argument(
        "--include-artifact-trials",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include BCI IV-2a trials flagged as artifacts. Default: true for real-contamination relevance.",
    )
    return parser.parse_args()


def parse_subjects(raw: str) -> list[str]:
    subjects = [item.strip().upper() for item in raw.split(",") if item.strip()]
    if not subjects:
        raise ValueError("At least one subject is required.")
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
    indexed = []
    for index, row in enumerate(rows):
        value = row.get(p_key, "")
        if value == "":
            continue
        indexed.append((float(value), index))
    indexed.sort()
    m = len(indexed)
    adjusted = {}
    prev = 1.0
    for rank_from_end, (p_value, index) in enumerate(reversed(indexed), start=1):
        rank = m - rank_from_end + 1
        q_value = min(prev, p_value * m / rank)
        prev = q_value
        adjusted[index] = min(1.0, q_value)
    for index, row in enumerate(rows):
        row[out_key] = adjusted.get(index, "")


def evaluate_subject(
    args: argparse.Namespace,
    subject: str,
    checkpoints: list[Path],
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_mat = args.bci_dir / f"{subject}T.mat"
    test_mat = args.bci_dir / f"{subject}E.mat"
    train_raw, _train_eog, y_train, train_info = load_bci2a_trials(
        train_mat,
        trial_start_sec=args.trial_start_sec,
        trial_stop_sec=args.trial_stop_sec,
        include_artifact_trials=args.include_artifact_trials,
        eog_index=args.eog_index,
    )
    test_raw, _test_eog, y_test, test_info = load_bci2a_trials(
        test_mat,
        trial_start_sec=args.trial_start_sec,
        trial_stop_sec=args.trial_stop_sec,
        include_artifact_trials=args.include_artifact_trials,
        eog_index=args.eog_index,
    )

    train_features = bandpass_epochs(train_raw, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz)
    test_features = bandpass_epochs(test_raw, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz)
    raw_classifier = fit_classifier(args, train_features, y_train)

    rows: list[dict[str, Any]] = []
    raw_row = evaluate_classifier(raw_classifier, test_features, y_test, condition="raw_raw")
    raw_row["subject"] = subject
    rows.append(raw_row)
    print(f"[baseline] subject={subject} raw_raw_acc={raw_row['accuracy']:.6f}", flush=True)

    for checkpoint_path in checkpoints:
        model, cfg, n_params = load_checkpoint_model(checkpoint_path, device)
        base = cfg.get("base", "")
        variant = str(cfg.get("variant") or cfg.get("model_type") or f"base{base}")
        train_seed = checkpoint_seed(checkpoint_path, cfg)
        print(f"[eval_start] subject={subject} variant={variant} base={base} seed={train_seed}", flush=True)

        train_denoised = denoise_epochs(model, train_raw, device=device, batch_size=args.batch_size)
        test_denoised = denoise_epochs(model, test_raw, device=device, batch_size=args.batch_size)
        train_denoised_features = bandpass_epochs(train_denoised, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz)
        test_denoised_features = bandpass_epochs(test_denoised, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz)

        strict_row = evaluate_classifier(
            raw_classifier,
            test_denoised_features,
            y_test,
            condition="raw_denoised",
            variant=variant,
            base=base,
            train_seed=train_seed,
            trainable_parameters=n_params,
            checkpoint=str(checkpoint_path),
        )
        denoised_classifier = fit_classifier(args, train_denoised_features, y_train)
        matched_row = evaluate_classifier(
            denoised_classifier,
            test_denoised_features,
            y_test,
            condition="denoised_denoised",
            variant=variant,
            base=base,
            train_seed=train_seed,
            trainable_parameters=n_params,
            checkpoint=str(checkpoint_path),
        )
        for row in [strict_row, matched_row]:
            row["subject"] = subject
            rows.append(row)
        print(
            f"[result] subject={subject} variant={variant} seed={train_seed} "
            f"raw_denoised_acc={strict_row['accuracy']:.6f} "
            f"denoised_denoised_acc={matched_row['accuracy']:.6f}",
            flush=True,
        )

    aggregates = aggregate_subject(rows)
    info = {"subject": subject, "train": train_info, "test": test_info}
    return rows, aggregates, info


def aggregate_subject(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["condition"] not in {"raw_denoised", "denoised_denoised"}:
            continue
        grouped[(str(row["condition"]), str(row["variant"]), str(row["base"]))].append(row)

    out: list[dict[str, Any]] = []
    raw_acc = float(next(row["accuracy"] for row in rows if row["condition"] == "raw_raw"))
    for (condition, variant, base), items in sorted(grouped.items()):
        entry: dict[str, Any] = {
            "subject": str(items[0]["subject"]),
            "condition": condition,
            "variant": variant,
            "base": base,
            "n": len(items),
            "train_seeds": " ".join(str(item["train_seed"]) for item in sorted(items, key=lambda r: str(r["train_seed"]))),
            "trainable_parameters": int(items[0]["trainable_parameters"]),
            "raw_raw_accuracy": raw_acc,
        }
        for key in METRIC_KEYS:
            vals = [float(item[key]) for item in items]
            m, sd = mean_std(vals)
            entry[f"{key}_mean"] = m
            entry[f"{key}_std"] = sd
        entry["delta_accuracy_vs_raw_raw"] = float(entry["accuracy_mean"]) - raw_acc
        out.append(entry)
    return out


def aggregate_all_subjects(rows: list[dict[str, Any]], subject_aggregates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_rows = [row for row in rows if row["condition"] == "raw_raw"]
    raw_values = [float(row["accuracy"]) for row in raw_rows]
    raw_mean, raw_sd = mean_std(raw_values)
    baseline = [{
        "condition": "raw_raw",
        "n_subjects": len(raw_rows),
        "accuracy_mean": raw_mean,
        "accuracy_std": raw_sd,
    }]

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in subject_aggregates:
        grouped[(str(row["condition"]), str(row["variant"]), str(row["base"]))].append(row)

    width_rows: list[dict[str, Any]] = []
    for (condition, variant, base), items in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][2], kv[0][1])):
        acc = [float(item["accuracy_mean"]) for item in items]
        deltas = [float(item["delta_accuracy_vs_raw_raw"]) for item in items]
        acc_mean, acc_sd = mean_std(acc)
        delta_mean, delta_sd = mean_std(deltas)
        width_rows.append({
            "condition": condition,
            "variant": variant,
            "base": base,
            "n_subjects": len(items),
            "subjects": " ".join(sorted(str(item["subject"]) for item in items)),
            "n_checkpoints_per_subject": int(items[0]["n"]),
            "trainable_parameters": int(items[0]["trainable_parameters"]),
            "accuracy_subject_mean": acc_mean,
            "accuracy_subject_std": acc_sd,
            "delta_accuracy_vs_raw_raw_subject_mean": delta_mean,
            "delta_accuracy_vs_raw_raw_subject_std": delta_sd,
            "subjects_below_raw_raw": sum(1 for d in deltas if d < 0),
            "subjects_above_raw_raw": sum(1 for d in deltas if d > 0),
            "wilcoxon_p_denoised_lt_raw": exact_wilcoxon_p(deltas, "less"),
        })

    bh_fdr(
        [row for row in width_rows if row["condition"] == "denoised_denoised"],
        "wilcoxon_p_denoised_lt_raw",
        "bh_fdr_p_denoised_lt_raw",
    )
    return baseline, width_rows


def write_markdown(path: Path, baseline_rows: list[dict[str, Any]], width_rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = ["# Real BCI IV-2a Downstream CSP+LDA Evaluation", ""]
    lines.append("Protocol: denoisers are applied directly to recorded BCI IV-2a trials, without synthetic EOG injection.")
    lines.append(f"Artifact-labeled trials included: `{args.include_artifact_trials}`.")
    lines.append("")
    lines.append("## Raw Baseline")
    lines.append("")
    lines.append("| Condition | n subjects | Accuracy |")
    lines.append("|---|---:|---:|")
    for row in baseline_rows:
        lines.append(f"| {row['condition']} | {row['n_subjects']} | {row['accuracy_mean']:.6f} +/- {row['accuracy_std']:.6f} |")
    lines.append("")
    lines.append("## Denoised Results")
    lines.append("")
    lines.append("| Condition | Variant | Base | n subjects | Params | Accuracy | Delta vs raw/raw | Subjects below raw | Wilcoxon p | BH-FDR p |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in width_rows:
        lines.append(
            f"| {row['condition']} | {row['variant']} | {row['base']} | {row['n_subjects']} | "
            f"{row['trainable_parameters']} | "
            f"{row['accuracy_subject_mean']:.6f} +/- {row['accuracy_subject_std']:.6f} | "
            f"{row['delta_accuracy_vs_raw_raw_subject_mean']:+.6f} +/- {row['delta_accuracy_vs_raw_raw_subject_std']:.6f} | "
            f"{row['subjects_below_raw_raw']} | "
            f"{row['wilcoxon_p_denoised_lt_raw']:.6f} | "
            f"{row.get('bh_fdr_p_denoised_lt_raw', '')} |"
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
    checkpoints = sorted(expand_checkpoints(args.checkpoint_glob))

    print(f"[start] run_id={args.run_id} subjects={subjects} checkpoints={len(checkpoints)} device={device}", flush=True)
    all_rows: list[dict[str, Any]] = []
    all_subject_aggregates: list[dict[str, Any]] = []
    infos: list[dict[str, Any]] = []
    for subject in subjects:
        rows, subject_aggregates, info = evaluate_subject(args, subject, checkpoints, device)
        all_rows.extend(rows)
        all_subject_aggregates.extend(subject_aggregates)
        infos.append(info)

    baseline_rows, width_rows = aggregate_all_subjects(all_rows, all_subject_aggregates)
    summary = {
        "info": {
            "run_id": args.run_id,
            "subjects": subjects,
            "checkpoint_glob": args.checkpoint_glob,
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
    (args.output_dir / "bci2a_real_downstream_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.output_dir / "bci2a_real_downstream_rows.csv", all_rows)
    write_csv(args.output_dir / "bci2a_real_downstream_subject_aggregates.csv", all_subject_aggregates)
    write_csv(args.output_dir / "bci2a_real_downstream_baseline_aggregate.csv", baseline_rows)
    write_csv(args.output_dir / "bci2a_real_downstream_width_aggregate.csv", width_rows)
    write_markdown(args.output_dir / "bci2a_real_downstream_summary.md", baseline_rows, width_rows, args)
    print(f"[written] {args.output_dir / 'bci2a_real_downstream_summary.md'}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
