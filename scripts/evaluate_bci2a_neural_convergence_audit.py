#!/usr/bin/env python3
"""Focused long-budget Deep4Net/EEGConformer downstream convergence audit."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from evaluate_bci2a_all_subjects_braindecode_classifiers import (  # noqa: E402
    METRIC_KEYS,
    DEFAULT_SUBJECTS,
    fit_eval_row,
    mean_std,
    parse_csv_list,
    write_csv,
)
from evaluate_bci2a_downstream_csp_lda import (  # noqa: E402
    FS_MODEL,
    bandpass_epochs,
    checkpoint_seed,
    denoise_epochs,
    expand_checkpoints,
    load_bci2a_trials,
    load_checkpoint_model,
    make_noisy_epochs,
)


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
    parser.add_argument("--denoiser-batch-size", type=int, default=256)
    parser.add_argument("--classifier-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--snr-min-db", type=float, default=-6.0)
    parser.add_argument("--snr-max-db", type=float, default=2.0)
    parser.add_argument("--trial-start-sec", type=float, default=2.5)
    parser.add_argument("--trial-stop-sec", type=float, default=6.0)
    parser.add_argument("--bandpass-low-hz", type=float, default=8.0)
    parser.add_argument("--bandpass-high-hz", type=float, default=30.0)
    parser.add_argument("--eog-index", type=int, default=22)
    parser.add_argument("--include-artifact-trials", action="store_true")
    parser.add_argument(
        "--classifier-bases",
        default="deep4net:16,eegconformer:8",
        help="Comma-separated classifier:base pairs, e.g. deep4net:16,eegconformer:8.",
    )
    return parser.parse_args()


def parse_classifier_bases(raw: str) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    for item in parse_csv_list(raw):
        if ":" not in item:
            raise ValueError(f"Expected classifier:base item, got {item!r}")
        classifier, base_text = item.split(":", 1)
        classifier = classifier.strip().lower()
        bases = {int(value.strip()) for value in base_text.split("+") if value.strip()}
        if not bases:
            raise ValueError(f"No bases specified in {item!r}")
        out[classifier] = bases
    return out


def exact_wilcoxon_one_sided_lower(deltas: list[float]) -> float:
    """Exact one-sided Wilcoxon signed-rank p-value for H1: median(delta) < 0."""

    nonzero = [(abs(delta), -1 if delta < 0 else 1) for delta in deltas if abs(delta) > 1e-12]
    n = len(nonzero)
    if n == 0:
        return 1.0
    ranks = list(range(1, n + 1))
    observed = sum(rank for rank, (_abs_delta, sign) in zip(ranks, sorted(nonzero)) if sign < 0)
    total = 0
    extreme = 0
    for signs in itertools.product([-1, 1], repeat=n):
        stat = sum(rank for rank, sign in zip(ranks, signs) if sign < 0)
        total += 1
        if stat >= observed:
            extreme += 1
    return float(extreme / total)


def bh_fdr(p_values: list[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    m = len(p_values)
    adjusted = [1.0] * m
    running = 1.0
    for rank, (idx, p_value) in reversed(list(enumerate(indexed, start=1))):
        running = min(running, p_value * m / rank)
        adjusted[idx] = min(running, 1.0)
    return adjusted


def checkpoint_base(path: Path) -> int | None:
    for part in path.parts:
        if "base" in part:
            tail = part.split("base")[-1].split("_")[0]
            if tail.isdigit():
                return int(tail)
    return None


def filter_checkpoints(paths: list[Path], bases: set[int]) -> list[Path]:
    return [path for path in paths if checkpoint_base(path) in bases]


def aggregate_checkpoint_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row["condition"] != "denoised_denoised":
            continue
        key = (str(row["classifier"]), str(row["subject"]), str(row["variant"]), str(row["base"]))
        grouped.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for (classifier, subject, variant, base), items in sorted(grouped.items()):
        result: dict[str, Any] = {
            "classifier": classifier,
            "subject": subject,
            "condition": "denoised_denoised_aggregate",
            "variant": variant,
            "base": base,
            "n": len(items),
            "train_seeds": " ".join(str(int(item["train_seed"])) for item in sorted(items, key=lambda x: int(x["train_seed"]))),
            "trainable_parameters": int(items[0]["trainable_parameters"]),
        }
        for key in METRIC_KEYS:
            result[f"{key}_mean"], result[f"{key}_std"] = mean_std([float(item[key]) for item in items])
        out.append(result)
    return out


def aggregate_across_subjects(
    rows: list[dict[str, Any]],
    subject_aggregates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_rows: list[dict[str, Any]] = []
    for classifier in sorted({str(row["classifier"]) for row in rows}):
        items = [row for row in rows if row["classifier"] == classifier and row["condition"] == "noisy_noisy"]
        result: dict[str, Any] = {"classifier": classifier, "condition": "noisy_noisy", "n_subjects": len(items)}
        for key in METRIC_KEYS:
            result[f"{key}_mean"], result[f"{key}_std"] = mean_std([float(item[key]) for item in items])
        baseline_rows.append(result)

    noisy = {
        (str(row["classifier"]), str(row["subject"])): float(row["accuracy"])
        for row in rows
        if row["condition"] == "noisy_noisy"
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in subject_aggregates:
        grouped.setdefault((str(row["classifier"]), str(row["base"])), []).append(row)

    width_rows: list[dict[str, Any]] = []
    p_values: list[float] = []
    for (classifier, base), items in sorted(grouped.items()):
        deltas = [float(item["accuracy_mean"]) - noisy[(classifier, str(item["subject"]))] for item in items]
        result: dict[str, Any] = {
            "classifier": classifier,
            "condition": "denoised_denoised_aggregate",
            "variant": items[0]["variant"],
            "base": base,
            "n_subjects": len(items),
            "subjects": " ".join(sorted(str(item["subject"]) for item in items)),
            "n_checkpoints_per_subject": int(items[0]["n"]),
            "trainable_parameters": int(items[0]["trainable_parameters"]),
            "subjects_below_noisy_noisy": int(sum(delta < 0 for delta in deltas)),
            "subjects_above_noisy_noisy": int(sum(delta > 0 for delta in deltas)),
            "wilcoxon_p_denoised_lt_noisy_noisy": exact_wilcoxon_one_sided_lower(deltas),
        }
        for key in METRIC_KEYS:
            result[f"{key}_subject_mean"], result[f"{key}_subject_std"] = mean_std(
                [float(item[f"{key}_mean"]) for item in items]
            )
        result["delta_accuracy_vs_noisy_noisy_subject_mean"], result[
            "delta_accuracy_vs_noisy_noisy_subject_std"
        ] = mean_std(deltas)
        width_rows.append(result)
        p_values.append(float(result["wilcoxon_p_denoised_lt_noisy_noisy"]))

    adjusted = bh_fdr(p_values)
    for row, p_adj in zip(width_rows, adjusted):
        row["bh_fdr_p_denoised_lt_noisy_noisy"] = p_adj
    return baseline_rows, width_rows


def write_markdown(path: Path, baseline_rows: list[dict[str, Any]], width_rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# BCI IV-2a Neural Decoder Convergence Audit",
        "",
        f"Epochs: `{args.epochs}`. Optimizer: AdamW, lr `{args.lr}`, weight decay `{args.weight_decay}`.",
        "",
        "## Noisy/Noisy Baselines",
        "",
        "| Classifier | n subjects | Accuracy | Balanced accuracy | Kappa | Macro F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in baseline_rows:
        lines.append(
            f"| {row['classifier']} | {row['n_subjects']} | "
            f"{row['accuracy_mean']:.6f} +/- {row['accuracy_std']:.6f} | "
            f"{row['balanced_accuracy_mean']:.6f} +/- {row['balanced_accuracy_std']:.6f} | "
            f"{row['cohen_kappa_mean']:.6f} +/- {row['cohen_kappa_std']:.6f} | "
            f"{row['macro_f1_mean']:.6f} +/- {row['macro_f1_std']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Matched Denoised/Denoised Results",
            "",
            "| Classifier | Base | Params | n subjects | Accuracy | Delta vs noisy/noisy | Below noisy/noisy | Wilcoxon p | BH-FDR p |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in width_rows:
        lines.append(
            f"| {row['classifier']} | {row['base']} | {row['trainable_parameters']} | {row['n_subjects']} | "
            f"{row['accuracy_subject_mean']:.6f} +/- {row['accuracy_subject_std']:.6f} | "
            f"{row['delta_accuracy_vs_noisy_noisy_subject_mean']:+.6f} +/- "
            f"{row['delta_accuracy_vs_noisy_noisy_subject_std']:.6f} | "
            f"{row['subjects_below_noisy_noisy']}/{row['n_subjects']} | "
            f"{row['wilcoxon_p_denoised_lt_noisy_noisy']:.6f} | "
            f"{row['bh_fdr_p_denoised_lt_noisy_noisy']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    subjects = [subject.strip().upper() for subject in parse_csv_list(args.subjects)]
    classifier_bases = parse_classifier_bases(args.classifier_bases)
    classifiers = sorted(classifier_bases)
    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_checkpoints = expand_checkpoints(args.checkpoint_glob)
    checkpoints_by_classifier = {
        classifier: filter_checkpoints(all_checkpoints, bases) for classifier, bases in classifier_bases.items()
    }
    for classifier, paths in checkpoints_by_classifier.items():
        if not paths:
            raise FileNotFoundError(f"No checkpoints found for classifier={classifier}, bases={classifier_bases[classifier]}")

    print(
        f"[start] run_id={args.run_id} device={device} subjects={' '.join(subjects)} "
        f"classifiers={classifier_bases} epochs={args.epochs}",
        flush=True,
    )
    rows: list[dict[str, Any]] = []

    for subject in subjects:
        print(f"[subject_start] {subject}", flush=True)
        train_mat = args.bci_dir / f"{subject}T.mat"
        test_mat = args.bci_dir / f"{subject}E.mat"
        train_clean, train_eog, y_train, _train_info = load_bci2a_trials(
            train_mat,
            trial_start_sec=args.trial_start_sec,
            trial_stop_sec=args.trial_stop_sec,
            include_artifact_trials=args.include_artifact_trials,
            eog_index=args.eog_index,
        )
        test_clean, test_eog, y_test, _test_info = load_bci2a_trials(
            test_mat,
            trial_start_sec=args.trial_start_sec,
            trial_stop_sec=args.trial_stop_sec,
            include_artifact_trials=args.include_artifact_trials,
            eog_index=args.eog_index,
        )
        train_noisy, _train_artifact, _train_noise_info = make_noisy_epochs(
            train_clean,
            train_eog,
            seed=args.seed + 1000,
            snr_min_db=args.snr_min_db,
            snr_max_db=args.snr_max_db,
        )
        test_noisy, _test_artifact, _test_noise_info = make_noisy_epochs(
            test_clean,
            test_eog,
            seed=args.seed,
            snr_min_db=args.snr_min_db,
            snr_max_db=args.snr_max_db,
        )
        train_noisy_bp = bandpass_epochs(train_noisy, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz)
        test_noisy_bp = bandpass_epochs(test_noisy, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz)

        for classifier_name in classifiers:
            baseline_row = fit_eval_row(
                classifier_name=classifier_name,
                subject=subject,
                condition="noisy_noisy",
                train_epochs=train_noisy_bp,
                y_train=y_train,
                test_epochs=test_noisy_bp,
                y_test=y_test,
                device=device,
                seed=args.seed,
                args=args,
            )
            rows.append(baseline_row)
            print(
                f"[baseline] subject={subject} classifier={classifier_name} condition=noisy_noisy "
                f"acc={baseline_row['accuracy']:.6f}",
                flush=True,
            )

            for checkpoint_path in checkpoints_by_classifier[classifier_name]:
                model, cfg, n_params = load_checkpoint_model(checkpoint_path, device)
                base = cfg.get("base", checkpoint_base(checkpoint_path))
                variant = str(cfg.get("variant") or cfg.get("model_type") or f"base{base}")
                train_seed = checkpoint_seed(checkpoint_path, cfg)
                train_denoised = denoise_epochs(model, train_noisy, device=device, batch_size=args.denoiser_batch_size)
                test_denoised = denoise_epochs(model, test_noisy, device=device, batch_size=args.denoiser_batch_size)
                train_denoised = bandpass_epochs(train_denoised, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz)
                test_denoised = bandpass_epochs(test_denoised, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz)
                row = fit_eval_row(
                    classifier_name=classifier_name,
                    subject=subject,
                    condition="denoised_denoised",
                    train_epochs=train_denoised,
                    y_train=y_train,
                    test_epochs=test_denoised,
                    y_test=y_test,
                    device=device,
                    seed=args.seed,
                    args=args,
                    variant=variant,
                    base=base,
                    train_seed=train_seed,
                    trainable_parameters=n_params,
                    checkpoint=str(checkpoint_path),
                )
                rows.append(row)
                print(
                    f"[result] subject={subject} classifier={classifier_name} base={base} seed={train_seed} "
                    f"denoised_denoised_acc={row['accuracy']:.6f}",
                    flush=True,
                )
        print(f"[subject_done] {subject}", flush=True)

    subject_aggregates = aggregate_checkpoint_rows(rows)
    baseline_aggregate, width_aggregate = aggregate_across_subjects(rows, subject_aggregates)
    summary = {
        "run_id": args.run_id,
        "subjects": subjects,
        "classifier_bases": {key: sorted(value) for key, value in classifier_bases.items()},
        "info": {
            "epochs": args.epochs,
            "classifier_batch_size": args.classifier_batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "bandpass_low_hz": args.bandpass_low_hz,
            "bandpass_high_hz": args.bandpass_high_hz,
            "snr_min_db": args.snr_min_db,
            "snr_max_db": args.snr_max_db,
        },
        "rows": rows,
        "subject_aggregates": subject_aggregates,
        "baseline_aggregate": baseline_aggregate,
        "width_aggregate": width_aggregate,
    }
    (args.output_dir / "bci2a_neural_convergence_audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "bci2a_neural_convergence_audit_rows.csv", rows)
    write_csv(args.output_dir / "bci2a_neural_convergence_audit_subject_aggregates.csv", subject_aggregates)
    write_csv(args.output_dir / "bci2a_neural_convergence_audit_baseline_aggregate.csv", baseline_aggregate)
    write_csv(args.output_dir / "bci2a_neural_convergence_audit_width_aggregate.csv", width_aggregate)
    write_markdown(args.output_dir / "bci2a_neural_convergence_audit_summary.md", baseline_aggregate, width_aggregate, args)
    print(f"[written] {args.output_dir / 'bci2a_neural_convergence_audit_summary.md'}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
