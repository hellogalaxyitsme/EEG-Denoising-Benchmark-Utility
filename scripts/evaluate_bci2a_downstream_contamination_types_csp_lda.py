#!/usr/bin/env python3
"""BCI IV-2a CSP+LDA downstream evaluation across contamination types."""

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
from scipy.signal import resample_poly

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
DEFAULT_RECIPES = ["eog", "emg", "eog_emg_line"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bci-dir",
        type=Path,
        default=Path("data/bci_iv_2a"),
    )
    parser.add_argument(
        "--emg-pool",
        type=Path,
        default=Path("data/eegdenoisenet/raw/EMG_all_epochs_512hz.npy"),
    )
    parser.add_argument("--subjects", default=",".join(DEFAULT_SUBJECTS))
    parser.add_argument("--recipes", default=",".join(DEFAULT_RECIPES))
    parser.add_argument("--checkpoint-glob", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cpu")
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
    return parser.parse_args()


def parse_subjects(raw: str) -> list[str]:
    subjects = [item.strip().upper() for item in raw.split(",") if item.strip()]
    if not subjects:
        raise ValueError("Expected at least one subject.")
    return subjects


def parse_recipes(raw: str) -> list[str]:
    recipes = [item.strip().lower() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(recipes) - set(DEFAULT_RECIPES))
    if unknown:
        raise ValueError(f"Unknown recipes: {unknown}; valid={DEFAULT_RECIPES}")
    if not recipes:
        raise ValueError("Expected at least one recipe.")
    return recipes


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


def load_emg_pool(path: Path) -> np.ndarray:
    emg = np.load(path).astype(np.float32)
    if emg.ndim > 2:
        emg = emg.reshape(emg.shape[0], -1)
    if emg.shape[-1] > 700:
        emg = resample_poly(emg, 1, 2, axis=-1).astype(np.float32)
    emg = emg - np.mean(emg, axis=-1, keepdims=True)
    return emg.astype(np.float32)


def fit_length(template: np.ndarray, n_times: int, rng: np.random.Generator) -> np.ndarray:
    template = np.asarray(template, dtype=np.float32).reshape(-1)
    if template.shape[0] == n_times:
        out = template
    elif template.shape[0] > n_times:
        start = int(rng.integers(0, template.shape[0] - n_times + 1))
        out = template[start : start + n_times]
    else:
        reps = int(math.ceil(n_times / template.shape[0]))
        out = np.tile(template, reps)[:n_times]
    out = out - float(np.mean(out))
    return out.astype(np.float32)


def make_line_template(n_times: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n_times, dtype=np.float32) / float(FS_MODEL)
    freq = 50.0 if rng.random() < 0.85 else 60.0
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    slow_phase = float(rng.uniform(0.0, 2.0 * np.pi))
    amp_mod = 1.0 + 0.2 * np.sin(2.0 * np.pi * 0.25 * t + slow_phase)
    line = amp_mod * np.sin(2.0 * np.pi * freq * t + phase)
    if freq * 2.0 < FS_MODEL / 2:
        line += 0.35 * np.sin(2.0 * np.pi * 2.0 * freq * t + phase / 2.0)
    line = line - float(np.mean(line))
    return line.astype(np.float32)


def make_recipe_noisy_epochs(
    clean_epochs: np.ndarray,
    eog_epochs: np.ndarray,
    emg_pool: np.ndarray,
    *,
    recipe: str,
    seed: int,
    snr_min_db: float,
    snr_max_db: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    noisy = np.zeros_like(clean_epochs, dtype=np.float32)
    artifact = np.zeros_like(clean_epochs, dtype=np.float32)
    snrs = rng.uniform(snr_min_db, snr_max_db, size=clean_epochs.shape[0]).astype(np.float32)
    n_times = clean_epochs.shape[-1]

    for trial_index, snr_db in enumerate(snrs):
        eog_template = eog_epochs[trial_index] - float(np.mean(eog_epochs[trial_index]))
        emg_template = fit_length(emg_pool[int(rng.integers(0, emg_pool.shape[0]))], n_times, rng)
        line_template = make_line_template(n_times, rng)

        if recipe == "eog":
            base_artifact = eog_template
        elif recipe == "emg":
            base_artifact = rng.uniform(0.6, 1.5) * emg_template
        elif recipe == "eog_emg_line":
            eog_scale = rng.uniform(0.7, 1.3) * (-1.0 if rng.random() < 0.5 else 1.0)
            emg_scale = rng.uniform(0.6, 1.5)
            line_scale = rng.uniform(0.05, 0.20) * (np.std(eog_template) + 1e-8)
            base_artifact = eog_scale * eog_template + emg_scale * emg_template + line_scale * line_template
        else:
            raise ValueError(recipe)

        base_artifact = np.asarray(base_artifact, dtype=np.float32)
        base_artifact = base_artifact - float(np.mean(base_artifact))
        p_artifact = float(np.mean(base_artifact * base_artifact) + 1e-12)
        for channel_index in range(clean_epochs.shape[1]):
            clean = clean_epochs[trial_index, channel_index]
            p_clean = float(np.mean(clean * clean) + 1e-12)
            scale = math.sqrt(p_clean / (p_artifact * (10 ** (float(snr_db) / 10.0))))
            a = (scale * base_artifact).astype(np.float32)
            artifact[trial_index, channel_index] = a
            noisy[trial_index, channel_index] = clean + a

    info = {
        "recipe": recipe,
        "snr_min_db": float(snr_min_db),
        "snr_max_db": float(snr_max_db),
        "snr_mean_db": float(np.mean(snrs)),
        "snr_std_db": float(np.std(snrs)),
    }
    return noisy, artifact, info


def load_subject_clean(args: argparse.Namespace, subject: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
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
    info = {"subject": subject, "train": train_info, "test": test_info}
    return train_clean, train_eog, test_clean, test_eog, y_train, y_test, info


def evaluate_subject_recipe(
    args: argparse.Namespace,
    *,
    subject: str,
    recipe: str,
    checkpoints: list[Path],
    emg_pool: np.ndarray,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_clean, train_eog, test_clean, test_eog, y_train, y_test, info = load_subject_clean(args, subject)
    train_noisy, _train_artifact, train_noise = make_recipe_noisy_epochs(
        train_clean,
        train_eog,
        emg_pool,
        recipe=recipe,
        seed=args.seed + 1000,
        snr_min_db=args.snr_min_db,
        snr_max_db=args.snr_max_db,
    )
    test_noisy, _test_artifact, test_noise = make_recipe_noisy_epochs(
        test_clean,
        test_eog,
        emg_pool,
        recipe=recipe,
        seed=args.seed,
        snr_min_db=args.snr_min_db,
        snr_max_db=args.snr_max_db,
    )

    train_features = bandpass_epochs(train_noisy, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz)
    test_features = bandpass_epochs(test_noisy, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz)
    baseline_classifier = fit_classifier(args, train_features, y_train)
    baseline_row = evaluate_classifier(baseline_classifier, test_features, y_test, condition="noisy_noisy")
    baseline_row.update({"subject": subject, "recipe": recipe, "base": "", "train_seed": ""})
    rows: list[dict[str, Any]] = [baseline_row]
    print(f"[baseline] subject={subject} recipe={recipe} acc={baseline_row['accuracy']:.6f}", flush=True)

    for checkpoint_path in checkpoints:
        model, cfg, n_params = load_checkpoint_model(checkpoint_path, device)
        base = cfg.get("base", "")
        variant = str(cfg.get("variant") or cfg.get("model_type") or f"base{base}")
        train_seed = checkpoint_seed(checkpoint_path, cfg)
        print(f"[denoise] subject={subject} recipe={recipe} base={base} seed={train_seed}", flush=True)
        train_den = denoise_epochs(model, train_noisy, device=device, batch_size=args.batch_size)
        test_den = denoise_epochs(model, test_noisy, device=device, batch_size=args.batch_size)
        train_den_features = bandpass_epochs(train_den, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz)
        test_den_features = bandpass_epochs(test_den, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz)
        classifier = fit_classifier(args, train_den_features, y_train)
        row = evaluate_classifier(
            classifier,
            test_den_features,
            y_test,
            condition="denoised_denoised",
            variant=variant,
            base=base,
            train_seed=train_seed,
            trainable_parameters=n_params,
            checkpoint=str(checkpoint_path),
        )
        row["subject"] = subject
        row["recipe"] = recipe
        rows.append(row)
        print(f"[result] subject={subject} recipe={recipe} base={base} seed={train_seed} acc={row['accuracy']:.6f}", flush=True)

    subject_aggregates = aggregate_subject(rows)
    info.update({"recipe": recipe, "train_noise": train_noise, "test_noise": test_noise})
    return rows, subject_aggregates, info


def aggregate_subject(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = next(row for row in rows if row["condition"] == "noisy_noisy")
    baseline_acc = float(baseline["accuracy"])
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["condition"] == "noisy_noisy":
            continue
        grouped[(str(row["variant"]), str(row["base"]), str(row["recipe"]))].append(row)

    out: list[dict[str, Any]] = []
    for (variant, base, recipe), items in sorted(grouped.items(), key=lambda kv: (kv[0][2], int(kv[0][1]))):
        entry: dict[str, Any] = {
            "subject": str(items[0]["subject"]),
            "recipe": recipe,
            "condition": "denoised_denoised",
            "variant": variant,
            "base": base,
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
    baseline_grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["condition"] == "noisy_noisy":
            baseline_grouped[str(row["recipe"])].append(row)
    baseline_rows: list[dict[str, Any]] = []
    for recipe, items in sorted(baseline_grouped.items()):
        acc_mean, acc_sd = mean_std([float(item["accuracy"]) for item in items])
        baseline_rows.append({"recipe": recipe, "condition": "noisy_noisy", "n_subjects": len(items), "accuracy_mean": acc_mean, "accuracy_std": acc_sd})

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in subject_aggregates:
        grouped[(str(row["recipe"]), str(row["variant"]), str(row["base"]))].append(row)

    out: list[dict[str, Any]] = []
    for (recipe, variant, base), items in sorted(grouped.items(), key=lambda kv: (kv[0][0], int(kv[0][2]))):
        acc = [float(item["accuracy_mean"]) for item in items]
        deltas = [float(item["delta_accuracy_vs_noisy_noisy"]) for item in items]
        acc_mean, acc_sd = mean_std(acc)
        delta_mean, delta_sd = mean_std(deltas)
        out.append({
            "recipe": recipe,
            "condition": "denoised_denoised",
            "variant": variant,
            "base": base,
            "n_subjects": len(items),
            "subjects": " ".join(sorted(str(item["subject"]) for item in items)),
            "n_checkpoints_per_subject": int(items[0]["n"]),
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
    return baseline_rows, out


def write_markdown(path: Path, baseline_rows: list[dict[str, Any]], aggregate_rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = ["# BCI IV-2a Downstream CSP+LDA Contamination-Type Robustness", ""]
    lines.append("This run evaluates matched denoised/denoised CSP+LDA under EOG, EMG, and EOG+EMG+LINE synthetic contaminants.")
    lines.append("")
    lines.append("## Baselines")
    lines.append("")
    lines.append("| Recipe | Condition | n subjects | Accuracy |")
    lines.append("|---|---|---:|---:|")
    for row in baseline_rows:
        lines.append(f"| {row['recipe']} | {row['condition']} | {row['n_subjects']} | {row['accuracy_mean']:.6f} +/- {row['accuracy_std']:.6f} |")
    lines.append("")
    lines.append("## Matched Denoised/Denoised Results")
    lines.append("")
    lines.append("| Recipe | Base | n subjects | Params | Accuracy | Delta vs noisy/noisy | Below noisy | Wilcoxon p lower | BH-FDR p |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in aggregate_rows:
        lines.append(
            f"| {row['recipe']} | {row['base']} | {row['n_subjects']} | "
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
    recipes = parse_recipes(args.recipes)
    checkpoints = expand_checkpoints(args.checkpoint_glob)
    emg_pool = load_emg_pool(args.emg_pool)
    print(
        f"[start] run_id={args.run_id} subjects={subjects} recipes={recipes} "
        f"checkpoints={len(checkpoints)} emg_pool={emg_pool.shape} device={device}",
        flush=True,
    )

    all_rows: list[dict[str, Any]] = []
    subject_aggregates: list[dict[str, Any]] = []
    subject_info: list[dict[str, Any]] = []
    for recipe in recipes:
        for subject in subjects:
            print(f"[subject_start] recipe={recipe} subject={subject}", flush=True)
            rows, aggregates, info = evaluate_subject_recipe(
                args,
                subject=subject,
                recipe=recipe,
                checkpoints=checkpoints,
                emg_pool=emg_pool,
                device=device,
            )
            all_rows.extend(rows)
            subject_aggregates.extend(aggregates)
            subject_info.append(info)
            print(f"[subject_done] recipe={recipe} subject={subject}", flush=True)

    baseline_rows, aggregate_rows = aggregate_all(all_rows, subject_aggregates)
    summary = {
        "info": {
            "run_id": args.run_id,
            "subjects": subjects,
            "recipes": recipes,
            "checkpoint_glob": args.checkpoint_glob,
            "emg_pool": str(args.emg_pool),
            "n_checkpoints": len(checkpoints),
            "device": str(device),
            "include_artifact_trials": bool(args.include_artifact_trials),
        },
        "subject_info": subject_info,
        "baseline_aggregate": baseline_rows,
        "width_aggregate": aggregate_rows,
    }
    (args.output_dir / "bci2a_downstream_contamination_types_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "bci2a_downstream_contamination_types_rows.csv", all_rows)
    write_csv(args.output_dir / "bci2a_downstream_contamination_types_subject_aggregates.csv", subject_aggregates)
    write_csv(args.output_dir / "bci2a_downstream_contamination_types_baseline_aggregate.csv", baseline_rows)
    write_csv(args.output_dir / "bci2a_downstream_contamination_types_width_aggregate.csv", aggregate_rows)
    write_markdown(args.output_dir / "bci2a_downstream_contamination_types_summary.md", baseline_rows, aggregate_rows, args)
    print(f"[written] {args.output_dir / 'bci2a_downstream_contamination_types_summary.md'}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
