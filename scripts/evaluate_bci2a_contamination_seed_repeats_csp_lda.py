#!/usr/bin/env python3
"""BCI IV-2a CSP+LDA evaluation across repeated contamination realizations."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
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

from scripts.evaluate_bci2a_downstream_contamination_types_csp_lda import (  # noqa: E402
    DEFAULT_RECIPES,
    DEFAULT_SUBJECTS,
    load_emg_pool,
    load_subject_clean,
    make_recipe_noisy_epochs,
    parse_recipes,
    parse_subjects,
)
from scripts.evaluate_bci2a_downstream_csp_lda import (  # noqa: E402
    FS_MODEL,
    METRIC_KEYS,
    bandpass_epochs,
    checkpoint_seed,
    denoise_epochs,
    evaluate_classifier,
    expand_checkpoints,
    fit_classifier,
    load_checkpoint_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bci-dir",
        type=Path,
        required=True, help="Path to the licensed input data.",
    )
    parser.add_argument(
        "--emg-pool",
        type=Path,
        required=True, help="Path to the licensed input data.",
    )
    parser.add_argument("--subjects", default=",".join(DEFAULT_SUBJECTS))
    parser.add_argument("--recipes", default=",".join(DEFAULT_RECIPES))
    parser.add_argument(
        "--checkpoint-glob",
        action="append",
        required=True,
        help="Checkpoint glob pattern. Repeat this option to evaluate multiple checkpoint families.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--test-seeds", default="42,43,44,45,46")
    parser.add_argument("--train-seeds", default="1042,1043,1044,1045,1046")
    parser.add_argument("--snr-min-db", type=float, default=-6.0)
    parser.add_argument("--snr-max-db", type=float, default=2.0)
    parser.add_argument("--trial-start-sec", type=float, default=2.5)
    parser.add_argument("--trial-stop-sec", type=float, default=6.0)
    parser.add_argument("--bandpass-low-hz", type=float, default=8.0)
    parser.add_argument("--bandpass-high-hz", type=float, default=30.0)
    parser.add_argument("--csp-components", type=int, default=8)
    parser.add_argument("--eog-index", type=int, default=22)
    parser.add_argument("--include-artifact-trials", action="store_true")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260813)
    parser.add_argument("--resume", action="store_true", help="Reuse existing raw CSV rows if present.")
    return parser.parse_args()


def parse_int_list(raw: str, *, name: str) -> list[int]:
    values = [int(part.strip()) for part in raw.replace(" ", ",").split(",") if part.strip()]
    if not values:
        raise ValueError(f"--{name} must contain at least one integer seed")
    return values


def expand_checkpoint_patterns(patterns: list[str]) -> list[Path]:
    checkpoints: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for checkpoint in expand_checkpoints(pattern):
            resolved = checkpoint.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            checkpoints.append(checkpoint)
    checkpoints.sort(key=lambda path: str(path))
    return checkpoints


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    preferred = [
        "run_id",
        "subject",
        "recipe",
        "seed_pair_index",
        "train_contamination_seed",
        "test_contamination_seed",
        "condition",
        "base",
        "checkpoint_seed",
        "checkpoint",
        "trainable_parameters",
        "accuracy",
        "delta_accuracy_vs_noisy_noisy",
        "balanced_accuracy",
        "cohen_kappa",
        "macro_f1",
    ]
    for key in preferred:
        for row in rows:
            if key in row and key not in seen:
                fieldnames.append(key)
                seen.add(key)
                break
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else float("nan")


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def sample_sd(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def bootstrap_mean_ci(values: list[float], *, n_bootstrap: int, seed: int) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(values)
    boot = []
    for _ in range(n_bootstrap):
        boot.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    return percentile(boot, 0.025), percentile(boot, 0.975)


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
    nz = [float(delta) for delta in deltas if abs(float(delta)) > 1e-12]
    if not nz:
        return 1.0
    ranks = _average_tied_ranks([abs(delta) for delta in nz])
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


def bh_fdr_adjust(rows: list[dict[str, Any]], p_key: str, out_key: str) -> None:
    indexed = []
    for index, row in enumerate(rows):
        value = row.get(p_key, "")
        if value != "":
            indexed.append((float(value), index))
    indexed.sort()
    m = len(indexed)
    adjusted: dict[int, float] = {}
    prev = 1.0
    for rank_from_end, (p_value, index) in enumerate(reversed(indexed), start=1):
        rank = m - rank_from_end + 1
        q_value = min(prev, p_value * m / rank)
        prev = q_value
        adjusted[index] = min(1.0, q_value)
    for index, row in enumerate(rows):
        row[out_key] = adjusted.get(index, "")


def holm_adjust(rows: list[dict[str, Any]], p_key: str, out_key: str) -> None:
    indexed = []
    for index, row in enumerate(rows):
        value = row.get(p_key, "")
        if value != "":
            indexed.append((float(value), index))
    indexed.sort()
    m = len(indexed)
    adjusted: dict[int, float] = {}
    prev = 0.0
    for rank, (p_value, index) in enumerate(indexed):
        value = min(1.0, (m - rank) * p_value)
        value = max(value, prev)
        prev = value
        adjusted[index] = value
    for index, row in enumerate(rows):
        row[out_key] = adjusted.get(index, "")


def load_models(checkpoints: list[Path], device: torch.device) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for checkpoint_path in checkpoints:
        model, cfg, n_params = load_checkpoint_model(checkpoint_path, device)
        base = cfg.get("base", "")
        train_seed = checkpoint_seed(checkpoint_path, cfg)
        models.append(
            {
                "checkpoint": str(checkpoint_path),
                "model": model,
                "cfg": cfg,
                "base": int(base) if str(base) != "" else "",
                "checkpoint_seed": train_seed,
                "trainable_parameters": int(n_params),
            }
        )
    return models


def evaluate_one_seed_pair(
    args: argparse.Namespace,
    *,
    run_id: str,
    subject: str,
    recipe: str,
    seed_pair_index: int,
    train_seed: int,
    test_seed: int,
    subject_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]],
    emg_pool: np.ndarray,
    models: list[dict[str, Any]],
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_clean, train_eog, test_clean, test_eog, y_train, y_test, info = subject_data
    train_noisy, _train_artifact, train_noise = make_recipe_noisy_epochs(
        train_clean,
        train_eog,
        emg_pool,
        recipe=recipe,
        seed=train_seed,
        snr_min_db=args.snr_min_db,
        snr_max_db=args.snr_max_db,
    )
    test_noisy, _test_artifact, test_noise = make_recipe_noisy_epochs(
        test_clean,
        test_eog,
        emg_pool,
        recipe=recipe,
        seed=test_seed,
        snr_min_db=args.snr_min_db,
        snr_max_db=args.snr_max_db,
    )

    train_features = bandpass_epochs(train_noisy, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz)
    test_features = bandpass_epochs(test_noisy, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz)
    baseline_classifier = fit_classifier(args, train_features, y_train)
    baseline = evaluate_classifier(baseline_classifier, test_features, y_test, condition="noisy_noisy")
    baseline.update(
        {
            "run_id": run_id,
            "subject": subject,
            "recipe": recipe,
            "seed_pair_index": seed_pair_index,
            "train_contamination_seed": train_seed,
            "test_contamination_seed": test_seed,
        }
    )
    baseline_rows = [baseline]
    result_rows: list[dict[str, Any]] = []
    baseline_acc = float(baseline["accuracy"])
    print(
        f"[baseline] subject={subject} recipe={recipe} seed_pair={seed_pair_index} "
        f"train_seed={train_seed} test_seed={test_seed} acc={baseline_acc:.6f}",
        flush=True,
    )

    for entry in models:
        model = entry["model"]
        base = entry["base"]
        checkpoint_train_seed = entry["checkpoint_seed"]
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
            base=base,
            train_seed=checkpoint_train_seed,
            trainable_parameters=entry["trainable_parameters"],
            checkpoint=entry["checkpoint"],
        )
        row.update(
            {
                "run_id": run_id,
                "subject": subject,
                "recipe": recipe,
                "seed_pair_index": seed_pair_index,
                "train_contamination_seed": train_seed,
                "test_contamination_seed": test_seed,
                "checkpoint_seed": checkpoint_train_seed,
                "delta_accuracy_vs_noisy_noisy": float(row["accuracy"]) - baseline_acc,
                "baseline_accuracy": baseline_acc,
            }
        )
        result_rows.append(row)
        print(
            f"[result] subject={subject} recipe={recipe} seed_pair={seed_pair_index} "
            f"base={base} checkpoint_seed={checkpoint_train_seed} "
            f"acc={row['accuracy']:.6f} delta={row['delta_accuracy_vs_noisy_noisy']:+.6f}",
            flush=True,
        )

    noise_info = {
        "subject": subject,
        "recipe": recipe,
        "seed_pair_index": seed_pair_index,
        "train_contamination_seed": train_seed,
        "test_contamination_seed": test_seed,
        "train_noise": train_noise,
        "test_noise": test_noise,
        "train_source": info["train"]["source"],
        "test_source": info["test"]["source"],
    }
    return baseline_rows, result_rows, noise_info


def subject_width_summary(
    baseline_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline_group: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in baseline_rows:
        baseline_group[(str(row["subject"]), str(row["recipe"]))].append(float(row["accuracy"]))

    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in result_rows:
        grouped[(str(row["subject"]), str(row["recipe"]), int(row["base"]))].append(row)

    rows: list[dict[str, Any]] = []
    for (subject, recipe, base), items in sorted(grouped.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
        baseline_values = baseline_group[(subject, recipe)]
        denoised_acc = [float(item["accuracy"]) for item in items]
        deltas = [float(item["delta_accuracy_vs_noisy_noisy"]) for item in items]
        checkpoint_seeds = sorted({str(item["checkpoint_seed"]) for item in items})
        seed_pairs = sorted({int(item["seed_pair_index"]) for item in items})
        entry: dict[str, Any] = {
            "subject": subject,
            "recipe": recipe,
            "base": base,
            "n_seed_pairs": len(seed_pairs),
            "seed_pair_indices": " ".join(str(value) for value in seed_pairs),
            "n_checkpoints": len(checkpoint_seeds),
            "checkpoint_seeds": " ".join(checkpoint_seeds),
            "n_observations_averaged": len(items),
            "trainable_parameters": int(items[0]["trainable_parameters"]),
            "baseline_accuracy_mean": mean(baseline_values),
            "baseline_accuracy_sd_over_seed_pairs": sample_sd(baseline_values),
            "denoised_accuracy_mean": mean(denoised_acc),
            "denoised_accuracy_sd_over_nuisance": sample_sd(denoised_acc),
            "delta_accuracy": mean(deltas),
            "delta_accuracy_sd_over_nuisance": sample_sd(deltas),
        }
        rows.append(entry)
    return rows


def width_inference_rows(
    subject_rows: list[dict[str, Any]],
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in subject_rows:
        grouped[(str(row["recipe"]), int(row["base"]))].append(row)

    rows: list[dict[str, Any]] = []
    for (recipe, base), items in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        deltas = [float(item["delta_accuracy"]) for item in items]
        acc = [float(item["denoised_accuracy_mean"]) for item in items]
        baseline = [float(item["baseline_accuracy_mean"]) for item in items]
        ci_low, ci_high = bootstrap_mean_ci(
            deltas,
            n_bootstrap=n_bootstrap,
            seed=bootstrap_seed + 1000 * len(rows),
        )
        rows.append(
            {
                "recipe": recipe,
                "base": base,
                "n_subjects": len(items),
                "subjects": " ".join(str(item["subject"]) for item in items),
                "trainable_parameters": int(items[0]["trainable_parameters"]),
                "baseline_accuracy_subject_mean": mean(baseline),
                "denoised_accuracy_subject_mean": mean(acc),
                "mean_delta_accuracy": mean(deltas),
                "median_delta_accuracy": median(deltas),
                "sd_delta_accuracy_across_subjects": sample_sd(deltas),
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
                "n_bootstrap": n_bootstrap,
                "subjects_below_noisy_noisy": sum(1 for delta in deltas if delta < 0),
                "subjects_above_noisy_noisy": sum(1 for delta in deltas if delta > 0),
                "wilcoxon_p_denoised_lt_noisy": exact_wilcoxon_p(deltas, "less"),
                "wilcoxon_p_denoised_gt_noisy": exact_wilcoxon_p(deltas, "greater"),
            }
        )
    bh_fdr_adjust(rows, "wilcoxon_p_denoised_lt_noisy", "bh_fdr_q_denoised_lt_noisy")
    holm_adjust(rows, "wilcoxon_p_denoised_lt_noisy", "holm_p_denoised_lt_noisy")
    return rows


def best_width_rows(width_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_recipe: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in width_rows:
        by_recipe[str(row["recipe"])].append(row)
    out = []
    for recipe, items in sorted(by_recipe.items()):
        best = max(items, key=lambda row: float(row["denoised_accuracy_subject_mean"]))
        copied = dict(best)
        copied["selection_rule"] = "highest_subject_mean_denoised_accuracy_within_recipe"
        out.append(copied)
    return out


def write_markdown(
    path: Path,
    *,
    args: argparse.Namespace,
    subjects: list[str],
    recipes: list[str],
    train_seeds: list[int],
    test_seeds: list[int],
    width_rows: list[dict[str, Any]],
    best_rows: list[dict[str, Any]],
) -> None:
    lines = ["# A1 CSP+LDA Contamination-Seed Repetition Summary", ""]
    lines.append("This experiment repeats the downstream CSP+LDA evaluation across independent synthetic contamination realizations.")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append(f"- Subjects: `{', '.join(subjects)}`.")
    lines.append(f"- Recipes: `{', '.join(recipes)}`.")
    lines.append(f"- Train contamination seeds: `{', '.join(str(seed) for seed in train_seeds)}`.")
    lines.append(f"- Test contamination seeds: `{', '.join(str(seed) for seed in test_seeds)}`.")
    lines.append(f"- SNR distribution: uniform from `{args.snr_min_db:.1f}` to `{args.snr_max_db:.1f}` dB.")
    lines.append(f"- Bootstrap resamples: `{args.n_bootstrap}` over subjects.")
    lines.append("- Inferential unit: subject. Contamination seeds and checkpoint seeds are nuisance repetitions averaged within each subject.")
    lines.append("")
    lines.append("## All Widths")
    lines.append("")
    lines.append("| Recipe | Base | n subjects | Baseline acc | Matched acc | Mean delta | Median delta | Subject SD | 95% subject-bootstrap CI | Below noisy | Wilcoxon p lower | BH-FDR q | Holm p |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in width_rows:
        lines.append(
            f"| {row['recipe']} | {row['base']} | {row['n_subjects']} | "
            f"{row['baseline_accuracy_subject_mean']:.6f} | {row['denoised_accuracy_subject_mean']:.6f} | "
            f"{row['mean_delta_accuracy']:+.6f} | {row['median_delta_accuracy']:+.6f} | "
            f"{row['sd_delta_accuracy_across_subjects']:.6f} | "
            f"[{row['bootstrap_ci95_low']:+.6f}, {row['bootstrap_ci95_high']:+.6f}] | "
            f"{row['subjects_below_noisy_noisy']} | {row['wilcoxon_p_denoised_lt_noisy']:.6f} | "
            f"{row['bh_fdr_q_denoised_lt_noisy']:.6f} | {row['holm_p_denoised_lt_noisy']:.6f} |"
        )
    lines.append("")
    lines.append("## Best Fixed Width Per Recipe")
    lines.append("")
    lines.append("Best rows are descriptive selections by highest subject-mean matched accuracy within each recipe. Primary inference should still consider the all-width table and correction.")
    lines.append("")
    lines.append("| Recipe | Base | Matched acc | Mean delta | 95% subject-bootstrap CI | Wilcoxon p lower | BH-FDR q |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in best_rows:
        lines.append(
            f"| {row['recipe']} | {row['base']} | {row['denoised_accuracy_subject_mean']:.6f} | "
            f"{row['mean_delta_accuracy']:+.6f} | "
            f"[{row['bootstrap_ci95_low']:+.6f}, {row['bootstrap_ci95_high']:+.6f}] | "
            f"{row['wilcoxon_p_denoised_lt_noisy']:.6f} | {row['bh_fdr_q_denoised_lt_noisy']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def maybe_reuse_existing(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]] | None:
    baseline_path = args.output_dir / "a1_baseline_seed_rows.csv"
    result_path = args.output_dir / "a1_checkpoint_seed_rows.csv"
    noise_path = args.output_dir / "a1_noise_metadata_rows.csv"
    if not args.resume:
        return None
    if not baseline_path.exists() or not result_path.exists() or not noise_path.exists():
        return None
    print("[resume] Reusing existing raw CSV rows", flush=True)
    return read_csv(baseline_path), read_csv(result_path), read_csv(noise_path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    subjects = parse_subjects(args.subjects)
    recipes = parse_recipes(args.recipes)
    test_seeds = parse_int_list(args.test_seeds, name="test-seeds")
    train_seeds = parse_int_list(args.train_seeds, name="train-seeds")
    if len(train_seeds) != len(test_seeds):
        raise ValueError("--train-seeds and --test-seeds must have the same length")
    if args.n_bootstrap < 10000:
        raise ValueError("--n-bootstrap must be at least 10000 for the A1 protocol")

    reused = maybe_reuse_existing(args)
    if reused is None:
        device_name = args.device
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            device_name = "cpu"
        device = torch.device(device_name)
        checkpoints = expand_checkpoint_patterns(args.checkpoint_glob)
        models = load_models(checkpoints, device)
        emg_pool = load_emg_pool(args.emg_pool)
        print(
            f"[start] run_id={args.run_id} subjects={subjects} recipes={recipes} "
            f"seed_pairs={list(zip(train_seeds, test_seeds))} checkpoints={len(models)} "
            f"emg_pool={emg_pool.shape} device={device}",
            flush=True,
        )

        baseline_rows: list[dict[str, Any]] = []
        result_rows: list[dict[str, Any]] = []
        noise_rows: list[dict[str, Any]] = []
        subject_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]] = {}
        for subject in subjects:
            print(f"[load_subject] {subject}", flush=True)
            subject_cache[subject] = load_subject_clean(args, subject)

        for recipe in recipes:
            for subject in subjects:
                subject_data = subject_cache[subject]
                for seed_pair_index, (train_seed, test_seed) in enumerate(zip(train_seeds, test_seeds), start=1):
                    print(
                        f"[seed_pair_start] recipe={recipe} subject={subject} "
                        f"seed_pair={seed_pair_index} train={train_seed} test={test_seed}",
                        flush=True,
                    )
                    base_rows, rows, noise_info = evaluate_one_seed_pair(
                        args,
                        run_id=args.run_id,
                        subject=subject,
                        recipe=recipe,
                        seed_pair_index=seed_pair_index,
                        train_seed=train_seed,
                        test_seed=test_seed,
                        subject_data=subject_data,
                        emg_pool=emg_pool,
                        models=models,
                        device=device,
                    )
                    baseline_rows.extend(base_rows)
                    result_rows.extend(rows)
                    noise_rows.append(noise_info)
                    write_csv(args.output_dir / "a1_baseline_seed_rows.csv", baseline_rows)
                    write_csv(args.output_dir / "a1_checkpoint_seed_rows.csv", result_rows)
                    write_csv(args.output_dir / "a1_noise_metadata_rows.csv", noise_rows)
                    print(
                        f"[seed_pair_done] recipe={recipe} subject={subject} seed_pair={seed_pair_index}",
                        flush=True,
                    )
    else:
        baseline_rows, result_rows, noise_rows = reused

    subject_rows = subject_width_summary(baseline_rows, result_rows)
    width_rows = width_inference_rows(
        subject_rows,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )
    best_rows = best_width_rows(width_rows)
    summary = {
        "info": {
            "run_id": args.run_id,
            "subjects": subjects,
            "recipes": recipes,
            "train_contamination_seeds": train_seeds,
            "test_contamination_seeds": test_seeds,
            "checkpoint_glob": args.checkpoint_glob,
            "emg_pool": str(args.emg_pool),
            "snr_min_db": args.snr_min_db,
            "snr_max_db": args.snr_max_db,
            "n_bootstrap": args.n_bootstrap,
            "bootstrap_seed": args.bootstrap_seed,
            "inferential_unit": "subject",
            "nuisance_repetitions": "contamination_seed_pairs and checkpoint_seeds averaged within subject for each recipe and width",
        },
        "width_inference": width_rows,
        "best_fixed_width_per_recipe": best_rows,
    }
    write_csv(args.output_dir / "a1_subject_width_summary.csv", subject_rows)
    write_csv(args.output_dir / "a1_width_inference.csv", width_rows)
    write_csv(args.output_dir / "a1_best_fixed_width_per_recipe.csv", best_rows)
    (args.output_dir / "a1_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(
        args.output_dir / "a1_summary.md",
        args=args,
        subjects=subjects,
        recipes=recipes,
        train_seeds=train_seeds,
        test_seeds=test_seeds,
        width_rows=width_rows,
        best_rows=best_rows,
    )
    print(f"[written] {args.output_dir / 'a1_summary.md'}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
