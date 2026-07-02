#!/usr/bin/env python3
"""Run BCI IV-2a downstream CSP+LDA validation for all subjects."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


METRIC_KEYS = ["accuracy", "balanced_accuracy", "cohen_kappa", "macro_f1"]
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
    parser.add_argument("--resume", action="store_true", help="Reuse completed per-subject summaries.")
    return parser.parse_args()


def parse_subjects(subjects_arg: str) -> list[str]:
    subjects = [item.strip().upper() for item in subjects_arg.split(",") if item.strip()]
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
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_subject(args: argparse.Namespace, subject: str) -> dict[str, Any]:
    subject_dir = args.output_dir / subject
    summary_path = subject_dir / "bci2a_csp_lda_summary.json"
    if args.resume and summary_path.exists():
        print(f"[subject_resume] {subject} summary={summary_path}", flush=True)
        return json.loads(summary_path.read_text(encoding="utf-8"))

    train_mat = args.bci_dir / f"{subject}T.mat"
    test_mat = args.bci_dir / f"{subject}E.mat"
    if not train_mat.exists():
        raise FileNotFoundError(train_mat)
    if not test_mat.exists():
        raise FileNotFoundError(test_mat)

    command = [
        sys.executable,
        "scripts/evaluate_bci2a_downstream_csp_lda.py",
        "--train-mat",
        str(train_mat),
        "--test-mat",
        str(test_mat),
        "--checkpoint-glob",
        args.checkpoint_glob,
        "--output-dir",
        str(subject_dir),
        "--run-id",
        f"{args.run_id}_{subject}",
        "--subject",
        subject,
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--snr-min-db",
        str(args.snr_min_db),
        "--snr-max-db",
        str(args.snr_max_db),
        "--trial-start-sec",
        str(args.trial_start_sec),
        "--trial-stop-sec",
        str(args.trial_stop_sec),
        "--bandpass-low-hz",
        str(args.bandpass_low_hz),
        "--bandpass-high-hz",
        str(args.bandpass_high_hz),
        "--csp-components",
        str(args.csp_components),
        "--eog-index",
        str(args.eog_index),
    ]
    if args.include_artifact_trials:
        command.append("--include-artifact-trials")

    print(f"[subject_start] {subject}", flush=True)
    subprocess.run(command, check=True)
    print(f"[subject_done] {subject}", flush=True)
    return json.loads(summary_path.read_text(encoding="utf-8"))


def collect_subject_rows(subject: str, summary: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    for row in summary["rows"]:
        copied = dict(row)
        copied["subject"] = subject
        rows.append(copied)
    for row in summary["aggregates"]:
        copied = dict(row)
        copied["subject"] = subject
        aggregates.append(copied)
    return rows, aggregates


def aggregate_baselines(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for condition in ["clean_clean", "clean_noisy", "noisy_noisy"]:
        items = [row for row in rows if row["condition"] == condition]
        entry: dict[str, Any] = {"condition": condition, "n_subjects": len(items)}
        for key in METRIC_KEYS:
            mean, std = mean_std([float(row[key]) for row in items])
            entry[f"{key}_mean"] = mean
            entry[f"{key}_std"] = std
        out.append(entry)
    return out


def aggregate_subject_widths(subject_aggregates: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    noisy_by_subject = {
        str(row["subject"]): float(row["accuracy"])
        for row in rows
        if row["condition"] == "noisy_noisy"
    }
    clean_by_subject = {
        str(row["subject"]): float(row["accuracy"])
        for row in rows
        if row["condition"] == "clean_clean"
    }

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in subject_aggregates:
        grouped.setdefault((str(row["condition"]), str(row["variant"]), str(row["base"])), []).append(row)

    out: list[dict[str, Any]] = []
    for (condition, variant, base), items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][2])):
        entry: dict[str, Any] = {
            "condition": condition,
            "variant": variant,
            "base": base,
            "n_subjects": len(items),
            "subjects": " ".join(sorted(str(item["subject"]) for item in items)),
            "n_checkpoints_per_subject": int(items[0].get("n", 0)),
            "trainable_parameters": int(items[0]["trainable_parameters"]),
        }
        for key in METRIC_KEYS:
            mean_key = f"{key}_mean"
            values = [float(item[mean_key]) for item in items]
            mean, std = mean_std(values)
            entry[f"{key}_subject_mean"] = mean
            entry[f"{key}_subject_std"] = std
        delta_noisy = [
            float(item["accuracy_mean"]) - noisy_by_subject[str(item["subject"])]
            for item in items
        ]
        delta_clean = [
            float(item["accuracy_mean"]) - clean_by_subject[str(item["subject"])]
            for item in items
        ]
        entry["delta_accuracy_vs_noisy_noisy_subject_mean"], entry["delta_accuracy_vs_noisy_noisy_subject_std"] = mean_std(delta_noisy)
        entry["delta_accuracy_vs_clean_clean_subject_mean"], entry["delta_accuracy_vs_clean_clean_subject_std"] = mean_std(delta_clean)
        out.append(entry)
    return out


def write_markdown(
    path: Path,
    *,
    subjects: list[str],
    baseline_aggregate: list[dict[str, Any]],
    width_aggregate: list[dict[str, Any]],
) -> None:
    lines = ["# BCI IV-2a All-Subject Downstream CSP+LDA Validation", ""]
    lines.append(f"Subjects: `{', '.join(subjects)}`.")
    lines.append("")
    lines.append("## Baseline Conditions Across Subjects")
    lines.append("")
    lines.append("| Condition | n | Accuracy | Balanced accuracy | Kappa | Macro F1 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in baseline_aggregate:
        lines.append(
            f"| {row['condition']} | {row['n_subjects']} | "
            f"{row['accuracy_mean']:.6f} +/- {row['accuracy_std']:.6f} | "
            f"{row['balanced_accuracy_mean']:.6f} +/- {row['balanced_accuracy_std']:.6f} | "
            f"{row['cohen_kappa_mean']:.6f} +/- {row['cohen_kappa_std']:.6f} | "
            f"{row['macro_f1_mean']:.6f} +/- {row['macro_f1_std']:.6f} |"
        )
    lines.append("")
    lines.append("## Denoised Conditions Across Subjects")
    lines.append("")
    lines.append("| Condition | Variant | Base | Subjects | Params | Accuracy | Delta vs noisy/noisy | Delta vs clean/clean |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in width_aggregate:
        lines.append(
            f"| {row['condition']} | {row['variant']} | {row['base']} | {row['n_subjects']} | "
            f"{row['trainable_parameters']} | "
            f"{row['accuracy_subject_mean']:.6f} +/- {row['accuracy_subject_std']:.6f} | "
            f"{row['delta_accuracy_vs_noisy_noisy_subject_mean']:+.6f} +/- {row['delta_accuracy_vs_noisy_noisy_subject_std']:.6f} | "
            f"{row['delta_accuracy_vs_clean_clean_subject_mean']:+.6f} +/- {row['delta_accuracy_vs_clean_clean_subject_std']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    subjects = parse_subjects(args.subjects)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[start] run_id={args.run_id} subjects={' '.join(subjects)}", flush=True)
    all_rows: list[dict[str, Any]] = []
    subject_aggregates: list[dict[str, Any]] = []
    subject_summaries: dict[str, Any] = {}

    for subject in subjects:
        summary = run_subject(args, subject)
        subject_summaries[subject] = summary
        rows, aggregates = collect_subject_rows(subject, summary)
        all_rows.extend(rows)
        subject_aggregates.extend(aggregates)

    baseline_aggregate = aggregate_baselines(all_rows)
    width_aggregate = aggregate_subject_widths(subject_aggregates, all_rows)

    summary = {
        "run_id": args.run_id,
        "subjects": subjects,
        "baseline_aggregate": baseline_aggregate,
        "width_aggregate": width_aggregate,
        "all_rows": all_rows,
        "subject_aggregates": subject_aggregates,
    }
    (args.output_dir / "bci2a_all_subjects_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "bci2a_all_subjects_rows.csv", all_rows)
    write_csv(args.output_dir / "bci2a_all_subjects_subject_aggregates.csv", subject_aggregates)
    write_csv(args.output_dir / "bci2a_all_subjects_baseline_aggregate.csv", baseline_aggregate)
    write_csv(args.output_dir / "bci2a_all_subjects_width_aggregate.csv", width_aggregate)
    write_markdown(
        args.output_dir / "bci2a_all_subjects_summary.md",
        subjects=subjects,
        baseline_aggregate=baseline_aggregate,
        width_aggregate=width_aggregate,
    )
    print(f"[written] {args.output_dir / 'bci2a_all_subjects_summary.md'}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
