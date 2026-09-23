#!/usr/bin/env python3
"""Checkpoint-uncertainty analysis for repeated BCI IV-2a CSP+LDA evaluations."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260813)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else float("nan")


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def sample_sd(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def rms(values: list[float]) -> float:
    return math.sqrt(mean([value * value for value in values])) if values else float("nan")


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
    indexed = sorted((float(row[p_key]), index) for index, row in enumerate(rows) if row.get(p_key) != "")
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
    indexed = sorted((float(row[p_key]), index) for index, row in enumerate(rows) if row.get(p_key) != "")
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


def field_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def summarize_subjects(
    baseline_rows: list[dict[str, str]],
    checkpoint_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    baseline_by_subject_recipe: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in baseline_rows:
        baseline_by_subject_recipe[(row["subject"], row["recipe"])].append(field_float(row, "accuracy"))

    grouped: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in checkpoint_rows:
        grouped[(row["subject"], row["recipe"], int(row["base"]))].append(row)

    out: list[dict[str, Any]] = []
    for (subject, recipe, base), items in sorted(grouped.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
        deltas = [field_float(item, "delta_accuracy_vs_noisy_noisy") for item in items]
        acc = [field_float(item, "accuracy") for item in items]
        contamination_ids = sorted({int(item["seed_pair_index"]) for item in items})
        checkpoint_ids = sorted({str(item["checkpoint_seed"]) for item in items})

        by_checkpoint: dict[str, list[float]] = defaultdict(list)
        by_contamination: dict[int, list[float]] = defaultdict(list)
        for item in items:
            delta = field_float(item, "delta_accuracy_vs_noisy_noisy")
            by_checkpoint[str(item["checkpoint_seed"])].append(delta)
            by_contamination[int(item["seed_pair_index"])].append(delta)
        checkpoint_means = [mean(values) for _, values in sorted(by_checkpoint.items())]
        contamination_means = [mean(values) for _, values in sorted(by_contamination.items())]

        out.append(
            {
                "subject": subject,
                "recipe": recipe,
                "base": base,
                "n_contamination_seed_pairs": len(contamination_ids),
                "contamination_seed_pair_indices": " ".join(str(value) for value in contamination_ids),
                "n_checkpoint_seeds": len(checkpoint_ids),
                "checkpoint_seeds": " ".join(checkpoint_ids),
                "n_observations_averaged": len(items),
                "baseline_accuracy_mean": mean(baseline_by_subject_recipe[(subject, recipe)]),
                "processed_accuracy_mean": mean(acc),
                "delta_accuracy": mean(deltas),
                "delta_accuracy_sd_over_all_nuisance": sample_sd(deltas),
                "checkpoint_delta_sd_within_subject": sample_sd(checkpoint_means),
                "checkpoint_delta_range_within_subject": max(checkpoint_means) - min(checkpoint_means),
                "contamination_delta_sd_within_subject": sample_sd(contamination_means),
                "contamination_delta_range_within_subject": max(contamination_means) - min(contamination_means),
            }
        )
    return out


def primary_inference_rows(
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
        baseline = [float(item["baseline_accuracy_mean"]) for item in items]
        processed = [float(item["processed_accuracy_mean"]) for item in items]
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
                "n_contamination_seed_pairs_min": min(int(item["n_contamination_seed_pairs"]) for item in items),
                "n_contamination_seed_pairs_max": max(int(item["n_contamination_seed_pairs"]) for item in items),
                "n_checkpoint_seeds_min": min(int(item["n_checkpoint_seeds"]) for item in items),
                "n_checkpoint_seeds_max": max(int(item["n_checkpoint_seeds"]) for item in items),
                "baseline_accuracy_subject_mean": mean(baseline),
                "processed_accuracy_subject_mean": mean(processed),
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


def checkpoint_variability_rows(subject_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in subject_rows:
        grouped[(str(row["recipe"]), int(row["base"]))].append(row)

    out: list[dict[str, Any]] = []
    for (recipe, base), items in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        checkpoint_sds = [float(item["checkpoint_delta_sd_within_subject"]) for item in items]
        checkpoint_ranges = [float(item["checkpoint_delta_range_within_subject"]) for item in items]
        contamination_sds = [float(item["contamination_delta_sd_within_subject"]) for item in items]
        contamination_ranges = [float(item["contamination_delta_range_within_subject"]) for item in items]
        out.append(
            {
                "recipe": recipe,
                "base": base,
                "n_subjects": len(items),
                "mean_within_subject_checkpoint_delta_sd": mean(checkpoint_sds),
                "median_within_subject_checkpoint_delta_sd": median(checkpoint_sds),
                "rms_within_subject_checkpoint_delta_sd": rms(checkpoint_sds),
                "mean_within_subject_checkpoint_delta_range": mean(checkpoint_ranges),
                "mean_within_subject_contamination_delta_sd": mean(contamination_sds),
                "median_within_subject_contamination_delta_sd": median(contamination_sds),
                "rms_within_subject_contamination_delta_sd": rms(contamination_sds),
                "mean_within_subject_contamination_delta_range": mean(contamination_ranges),
            }
        )
    return out


def variance_decomposition_rows(checkpoint_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in checkpoint_rows:
        grouped[(row["recipe"], int(row["base"]))].append(row)

    out: list[dict[str, Any]] = []
    for (recipe, base), items in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        cell_values: dict[tuple[str, int, str], list[float]] = defaultdict(list)
        for item in items:
            key = (item["subject"], int(item["seed_pair_index"]), str(item["checkpoint_seed"]))
            cell_values[key].append(field_float(item, "delta_accuracy_vs_noisy_noisy"))
        observations = [
            {
                "subject": key[0],
                "contamination": key[1],
                "checkpoint": key[2],
                "delta": mean(values),
                "n_cell_rows": len(values),
            }
            for key, values in cell_values.items()
        ]
        values = [float(obs["delta"]) for obs in observations]
        grand = mean(values)
        total_ss = sum((value - grand) ** 2 for value in values)

        def factor_ss(factor: str) -> float:
            by_level: dict[str, list[float]] = defaultdict(list)
            for obs in observations:
                by_level[str(obs[factor])].append(float(obs["delta"]))
            return sum(len(level_values) * (mean(level_values) - grand) ** 2 for level_values in by_level.values())

        subject_ss = factor_ss("subject")
        contamination_ss = factor_ss("contamination")
        checkpoint_ss = factor_ss("checkpoint")
        residual_ss = max(0.0, total_ss - subject_ss - contamination_ss - checkpoint_ss)
        denom = total_ss if total_ss > 0 else float("nan")
        subjects = sorted({str(obs["subject"]) for obs in observations})
        contaminations = sorted({int(obs["contamination"]) for obs in observations})
        checkpoints = sorted({str(obs["checkpoint"]) for obs in observations})
        expected_cells = len(subjects) * len(contaminations) * len(checkpoints)
        duplicate_cells = sum(1 for values_for_cell in cell_values.values() if len(values_for_cell) > 1)
        out.append(
            {
                "recipe": recipe,
                "base": base,
                "n_subjects": len(subjects),
                "n_contamination_seed_pairs": len(contaminations),
                "n_checkpoint_seeds": len(checkpoints),
                "n_cells_observed": len(observations),
                "n_cells_expected_balanced": expected_cells,
                "is_balanced": len(observations) == expected_cells and duplicate_cells == 0,
                "n_duplicate_cells_averaged": duplicate_cells,
                "total_ss": total_ss,
                "subject_main_effect_ss": subject_ss,
                "contamination_main_effect_ss": contamination_ss,
                "checkpoint_main_effect_ss": checkpoint_ss,
                "residual_interaction_ss": residual_ss,
                "subject_main_effect_prop": subject_ss / denom,
                "contamination_main_effect_prop": contamination_ss / denom,
                "checkpoint_main_effect_prop": checkpoint_ss / denom,
                "residual_interaction_prop": residual_ss / denom,
                "note": "Fixed-effect descriptive decomposition of paired deltas; residual includes interactions and unexplained cell variation.",
            }
        )
    return out


def best_fixed_width_rows(inference_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in inference_rows:
        grouped[str(row["recipe"])].append(row)
    out: list[dict[str, Any]] = []
    for recipe, items in sorted(grouped.items()):
        best = max(items, key=lambda row: float(row["processed_accuracy_subject_mean"]))
        copied = dict(best)
        copied["selection_rule"] = "highest_subject_mean_processed_accuracy_within_recipe"
        out.append(copied)
    return out


def write_markdown(
    path: Path,
    *,
    run_id: str,
    inference_rows: list[dict[str, Any]],
    variability_rows: list[dict[str, Any]],
    decomposition_rows: list[dict[str, Any]],
    best_rows: list[dict[str, Any]],
) -> None:
    lines = ["# Checkpoint-uncertainty summary", ""]
    lines.append("This analysis keeps checkpoint identity in the CSP+LDA rows while retaining the human subject as the inferential unit.")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    if run_id:
        lines.append(f"- Run ID: `{run_id}`.")
    lines.append("- Primary inference: paired processed/processed minus noisy/noisy deltas are averaged over contamination realizations and checkpoint seeds within each subject before testing.")
    lines.append("- Bootstrap confidence intervals resample subjects only.")
    lines.append("- Checkpoint and contamination repetitions are reported as nuisance variability, not as additional subjects.")
    lines.append("- Variance decomposition is descriptive: subject, contamination, and checkpoint main-effect sums of squares are computed on cell-level paired deltas; the residual contains interactions.")
    lines.append("")
    lines.append("## Primary Subject-Level Inference")
    lines.append("")
    lines.append("| Recipe | Base | n subjects | n contam | n checkpoints | Baseline acc | Processed acc | Mean delta | Median delta | Subject SD | 95% subject-bootstrap CI | Below noisy | Wilcoxon p lower | BH-FDR q | Holm p |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in inference_rows:
        n_contam = f"{row['n_contamination_seed_pairs_min']}-{row['n_contamination_seed_pairs_max']}"
        n_checkpoints = f"{row['n_checkpoint_seeds_min']}-{row['n_checkpoint_seeds_max']}"
        lines.append(
            f"| {row['recipe']} | {row['base']} | {row['n_subjects']} | {n_contam} | {n_checkpoints} | "
            f"{row['baseline_accuracy_subject_mean']:.6f} | {row['processed_accuracy_subject_mean']:.6f} | "
            f"{row['mean_delta_accuracy']:+.6f} | {row['median_delta_accuracy']:+.6f} | "
            f"{row['sd_delta_accuracy_across_subjects']:.6f} | "
            f"[{row['bootstrap_ci95_low']:+.6f}, {row['bootstrap_ci95_high']:+.6f}] | "
            f"{row['subjects_below_noisy_noisy']} | {row['wilcoxon_p_denoised_lt_noisy']:.6f} | "
            f"{row['bh_fdr_q_denoised_lt_noisy']:.6f} | {row['holm_p_denoised_lt_noisy']:.6f} |"
        )
    lines.append("")
    lines.append("## Checkpoint And Contamination Variability")
    lines.append("")
    lines.append("| Recipe | Base | Checkpoint SD mean | Checkpoint range mean | Contamination SD mean | Contamination range mean |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in variability_rows:
        lines.append(
            f"| {row['recipe']} | {row['base']} | "
            f"{row['mean_within_subject_checkpoint_delta_sd']:.6f} | "
            f"{row['mean_within_subject_checkpoint_delta_range']:.6f} | "
            f"{row['mean_within_subject_contamination_delta_sd']:.6f} | "
            f"{row['mean_within_subject_contamination_delta_range']:.6f} |"
        )
    lines.append("")
    lines.append("## Variance Decomposition")
    lines.append("")
    lines.append("| Recipe | Base | Balanced | Subject % | Contamination % | Checkpoint % | Residual/interactions % |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in decomposition_rows:
        lines.append(
            f"| {row['recipe']} | {row['base']} | {row['is_balanced']} | "
            f"{100.0 * row['subject_main_effect_prop']:.2f} | "
            f"{100.0 * row['contamination_main_effect_prop']:.2f} | "
            f"{100.0 * row['checkpoint_main_effect_prop']:.2f} | "
            f"{100.0 * row['residual_interaction_prop']:.2f} |"
        )
    lines.append("")
    lines.append("## Best Fixed Width Per Recipe")
    lines.append("")
    lines.append("Best rows are descriptive selections by highest subject-mean processed accuracy within each recipe.")
    lines.append("")
    lines.append("| Recipe | Base | Processed acc | Mean delta | 95% subject-bootstrap CI | Wilcoxon p lower | BH-FDR q |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in best_rows:
        lines.append(
            f"| {row['recipe']} | {row['base']} | {row['processed_accuracy_subject_mean']:.6f} | "
            f"{row['mean_delta_accuracy']:+.6f} | "
            f"[{row['bootstrap_ci95_low']:+.6f}, {row['bootstrap_ci95_high']:+.6f}] | "
            f"{row['wilcoxon_p_denoised_lt_noisy']:.6f} | {row['bh_fdr_q_denoised_lt_noisy']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.n_bootstrap < 10000:
        raise ValueError("--n-bootstrap must be at least 10000")

    baseline_rows = read_csv(args.input_dir / "contamination-seed_baseline_seed_rows.csv")
    checkpoint_rows = read_csv(args.input_dir / "contamination-seed_checkpoint_seed_rows.csv")
    subject_rows = summarize_subjects(baseline_rows, checkpoint_rows)
    inference_rows = primary_inference_rows(
        subject_rows,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )
    variability_rows = checkpoint_variability_rows(subject_rows)
    decomposition_rows = variance_decomposition_rows(checkpoint_rows)
    best_rows = best_fixed_width_rows(inference_rows)

    write_csv(output_dir / "checkpoint_subject_primary_rows.csv", subject_rows)
    write_csv(output_dir / "checkpoint_primary_subject_inference.csv", inference_rows)
    write_csv(output_dir / "checkpoint_contamination_variability.csv", variability_rows)
    write_csv(output_dir / "checkpoint_variance_decomposition.csv", decomposition_rows)
    write_csv(output_dir / "checkpoint_best_fixed_width_per_recipe.csv", best_rows)

    summary = {
        "info": {
            "run_id": args.run_id,
            "input_dir": str(args.input_dir),
            "n_bootstrap": args.n_bootstrap,
            "bootstrap_seed": args.bootstrap_seed,
            "primary_inferential_unit": "subject",
            "nuisance_repetitions": "contamination seed pairs and checkpoint seeds",
        },
        "primary_subject_inference": inference_rows,
        "checkpoint_contamination_variability": variability_rows,
        "variance_decomposition": decomposition_rows,
        "best_fixed_width_per_recipe": best_rows,
    }
    (output_dir / "checkpoint_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(
        output_dir / "checkpoint_summary.md",
        run_id=args.run_id,
        inference_rows=inference_rows,
        variability_rows=variability_rows,
        decomposition_rows=decomposition_rows,
        best_rows=best_rows,
    )
    print(f"[written] {output_dir / 'checkpoint_summary.md'}", flush=True)


if __name__ == "__main__":
    main()
