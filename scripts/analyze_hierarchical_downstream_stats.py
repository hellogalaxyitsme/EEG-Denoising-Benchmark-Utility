#!/usr/bin/env python3
"""Hierarchical downstream statistics with subject as the inferential unit."""

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

import numpy as np
from scipy import stats


DEFAULT_SUBJECTS = [f"A{idx:02d}" for idx in range(1, 10)]
DEFAULT_RECIPES = ["eog", "emg", "eog_emg_line"]
METRIC = "accuracy"
PRIMARY_P_KEY = "primary_t_p_two_sided"
WILCOXON_P_KEY = "wilcoxon_p_two_sided"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a2-dir", type=Path, required=True, help="A2 CSP+LDA result directory.")
    parser.add_argument(
        "--a3-dir",
        type=Path,
        action="append",
        required=True,
        help="One or more A3 neural-decoder result directories. Comma-separated entries are accepted.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260815)
    parser.add_argument("--confirmatory-csp-base", type=int, default=16)
    parser.add_argument("--reference-width-base", type=int, default=16)
    return parser.parse_args()


def expand_dirs(values: list[Path]) -> list[Path]:
    out: list[Path] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                out.append(Path(part))
    return out


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    preferred = [
        "run_id",
        "analysis_family",
        "analysis_role",
        "classifier",
        "recipe",
        "base",
        "reference_base",
        "contrast_base",
        "subject",
        "n_subjects",
        "n_subjects_for_inference",
        "mean_delta_accuracy",
        "bootstrap_ci95_low",
        "bootstrap_ci95_high",
        "primary_t_p_two_sided",
        "bh_q_primary_t_two_sided_within_family",
        "holm_p_primary_t_two_sided_within_family",
        "bh_q_primary_t_two_sided_all_effects",
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
    order = sorted(range(len(abs_values)), key=lambda idx: abs_values[idx])
    ranks = [0.0] * len(abs_values)
    pos = 0
    while pos < len(order):
        end = pos + 1
        while end < len(order) and abs(abs_values[order[end]] - abs_values[order[pos]]) < 1e-12:
            end += 1
        rank = (pos + 1 + end) / 2.0
        for idx in range(pos, end):
            ranks[order[idx]] = rank
        pos = end
    return ranks


def exact_wilcoxon_p(deltas: list[float], alternative: str = "two-sided") -> float:
    nz = [float(delta) for delta in deltas if abs(float(delta)) > 1e-12]
    if not nz:
        return 1.0
    ranks = _average_tied_ranks([abs(delta) for delta in nz])
    observed = sum(rank for rank, delta in zip(ranks, nz) if delta > 0)
    null_sums = []
    for signs in itertools.product([0, 1], repeat=len(nz)):
        null_sums.append(sum(rank for rank, sign in zip(ranks, signs) if sign))
    lower = sum(1 for value in null_sums if value <= observed + 1e-12) / len(null_sums)
    upper = sum(1 for value in null_sums if value >= observed - 1e-12) / len(null_sums)
    if alternative == "less":
        return lower
    if alternative == "greater":
        return upper
    return min(1.0, 2.0 * min(lower, upper))


def bh_fdr(rows: list[dict[str, Any]], p_key: str, q_key: str, *, subset_key: str | None = None) -> None:
    groups: dict[str, list[int]] = defaultdict(list)
    if subset_key is None:
        groups["__all__"] = list(range(len(rows)))
    else:
        for idx, row in enumerate(rows):
            groups[str(row[subset_key])].append(idx)
    for indices in groups.values():
        indexed = sorted((float(rows[idx][p_key]), idx) for idx in indices if is_finite_number(rows[idx].get(p_key)))
        m = len(indexed)
        adjusted: dict[int, float] = {}
        prev = 1.0
        for rank_from_end, (p_value, idx) in enumerate(reversed(indexed), start=1):
            rank = m - rank_from_end + 1
            q = min(prev, p_value * m / rank)
            prev = q
            adjusted[idx] = min(1.0, q)
        for idx in indices:
            rows[idx][q_key] = adjusted.get(idx, "")


def holm(rows: list[dict[str, Any]], p_key: str, out_key: str, *, subset_key: str | None = None) -> None:
    groups: dict[str, list[int]] = defaultdict(list)
    if subset_key is None:
        groups["__all__"] = list(range(len(rows)))
    else:
        for idx, row in enumerate(rows):
            groups[str(row[subset_key])].append(idx)
    for indices in groups.values():
        indexed = sorted((float(rows[idx][p_key]), idx) for idx in indices if is_finite_number(rows[idx].get(p_key)))
        m = len(indexed)
        adjusted: dict[int, float] = {}
        prev = 0.0
        for rank, (p_value, idx) in enumerate(indexed):
            value = min(1.0, (m - rank) * p_value)
            value = max(prev, value)
            prev = value
            adjusted[idx] = value
        for idx in indices:
            rows[idx][out_key] = adjusted.get(idx, "")


def is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def one_sample_stats(values: list[float]) -> dict[str, float]:
    n = len(values)
    mu = mean(values)
    sd = sample_sd(values)
    if n > 1 and sd > 0:
        se = sd / math.sqrt(n)
        t_stat = mu / se
        p_two = float(2.0 * stats.t.sf(abs(t_stat), df=n - 1))
        p_less = float(stats.t.cdf(t_stat, df=n - 1))
        p_greater = float(stats.t.sf(t_stat, df=n - 1))
        cohen_dz = mu / sd
    else:
        t_stat = 0.0 if abs(mu) < 1e-12 else math.copysign(float("inf"), mu)
        p_two = 1.0 if abs(mu) < 1e-12 else 0.0
        p_less = 0.5 if abs(mu) < 1e-12 else (0.0 if mu < 0 else 1.0)
        p_greater = 0.5 if abs(mu) < 1e-12 else (1.0 if mu < 0 else 0.0)
        cohen_dz = float("nan")
    return {
        "primary_t_statistic": t_stat,
        "primary_t_df": n - 1,
        "primary_t_p_two_sided": p_two,
        "primary_t_p_denoised_lt_noisy": p_less,
        "primary_t_p_denoised_gt_noisy": p_greater,
        "cohen_dz_subject_paired": cohen_dz,
    }


def format_float(value: Any, digits: int = 6) -> str:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(value_f):
        return ""
    return f"{value_f:.{digits}f}"


def normalized_a2_rows(a2_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_raw = read_csv(a2_dir / "a1_baseline_seed_rows.csv")
    processed_raw = read_csv(a2_dir / "a1_checkpoint_seed_rows.csv")
    baseline = []
    for row in baseline_raw:
        baseline.append(
            {
                "source": "a2_csp_lda",
                "classifier": "csp_lda",
                "classifier_seed": "",
                "subject": row["subject"],
                "recipe": row["recipe"],
                "seed_pair_index": int(row["seed_pair_index"]),
                "base": "",
                "checkpoint_seed": "",
                "condition": row["condition"],
                "accuracy": float(row["accuracy"]),
            }
        )
    processed_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in processed_raw:
        if row.get("condition") != "denoised_denoised":
            continue
        item = {
            "source": "a2_csp_lda",
            "classifier": "csp_lda",
            "classifier_seed": "",
            "subject": row["subject"],
            "recipe": row["recipe"],
            "seed_pair_index": int(row["seed_pair_index"]),
            "base": int(row["base"]),
            "checkpoint_seed": str(row["checkpoint_seed"]),
            "condition": row["condition"],
            "accuracy": float(row["accuracy"]),
            "delta_accuracy_vs_noisy_noisy": float(row["delta_accuracy_vs_noisy_noisy"]),
            "baseline_accuracy": float(row.get("baseline_accuracy", "nan")),
        }
        key = (
            item["classifier"],
            item["subject"],
            item["recipe"],
            item["seed_pair_index"],
            item["base"],
            item["checkpoint_seed"],
        )
        processed_by_key.setdefault(key, item)
    return baseline, list(processed_by_key.values())


def normalized_a3_rows(a3_dirs: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    processed_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for a3_dir in a3_dirs:
        for row in read_csv(a3_dir / "a3_baseline_classifier_seed_rows.csv"):
            item = {
                "source": str(a3_dir),
                "classifier": row["classifier"],
                "classifier_seed": str(row["classifier_seed"]),
                "subject": row["subject"],
                "recipe": row["recipe"],
                "seed_pair_index": int(row["seed_pair_index"]),
                "base": "",
                "checkpoint_seed": "",
                "condition": row["condition"],
                "accuracy": float(row["accuracy"]),
            }
            key = (
                item["classifier"],
                item["classifier_seed"],
                item["subject"],
                item["recipe"],
                item["seed_pair_index"],
            )
            baseline_by_key.setdefault(key, item)
        for row in read_csv(a3_dir / "a3_checkpoint_classifier_seed_rows.csv"):
            if row.get("condition") != "denoised_denoised":
                continue
            item = {
                "source": str(a3_dir),
                "classifier": row["classifier"],
                "classifier_seed": str(row["classifier_seed"]),
                "subject": row["subject"],
                "recipe": row["recipe"],
                "seed_pair_index": int(row["seed_pair_index"]),
                "base": int(row["base"]),
                "checkpoint_seed": str(row["checkpoint_seed"]),
                "condition": row["condition"],
                "accuracy": float(row["accuracy"]),
                "delta_accuracy_vs_noisy_noisy": float(row["delta_accuracy_vs_noisy_noisy"]),
                "baseline_accuracy": float(row["baseline_accuracy"]),
            }
            key = (
                item["classifier"],
                item["classifier_seed"],
                item["subject"],
                item["recipe"],
                item["seed_pair_index"],
                item["base"],
                item["checkpoint_seed"],
            )
            processed_by_key.setdefault(key, item)
    return list(baseline_by_key.values()), list(processed_by_key.values())


def aggregate_subject_rows(
    *,
    run_id: str,
    baseline_rows: list[dict[str, Any]],
    processed_rows: list[dict[str, Any]],
    n_bootstrap: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    baseline_by_group: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in baseline_rows:
        baseline_by_group[(row["classifier"], row["subject"], row["recipe"])].append(float(row["accuracy"]))

    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in processed_rows:
        grouped[(row["classifier"], row["subject"], row["recipe"], int(row["base"]))].append(row)

    out: list[dict[str, Any]] = []
    for index, ((classifier, subject, recipe, base), items) in enumerate(sorted(grouped.items())):
        deltas = [float(item["delta_accuracy_vs_noisy_noisy"]) for item in items]
        processed_acc = [float(item["accuracy"]) for item in items]
        contamination_levels = sorted({str(item["seed_pair_index"]) for item in items})
        checkpoint_levels = sorted({str(item["checkpoint_seed"]) for item in items if str(item["checkpoint_seed"]) != ""})
        classifier_seed_levels = sorted({str(item["classifier_seed"]) for item in items if str(item["classifier_seed"]) != ""})
        by_contamination = group_means(items, "seed_pair_index")
        by_checkpoint = group_means(items, "checkpoint_seed")
        by_classifier_seed = group_means(items, "classifier_seed") if classifier != "csp_lda" else []
        within_low, within_high = bootstrap_mean_ci(
            deltas,
            n_bootstrap=n_bootstrap,
            seed=bootstrap_seed + 37 * index,
        )
        out.append(
            {
                "run_id": run_id,
                "primary_independent_unit": "subject",
                "subject": subject,
                "classifier": classifier,
                "recipe": recipe,
                "base": base,
                "n_subjects_for_inference": 9,
                "n_contamination_seed_pairs": len(contamination_levels),
                "contamination_seed_pair_indices": " ".join(contamination_levels),
                "n_checkpoint_seeds": len(checkpoint_levels),
                "checkpoint_seeds": " ".join(checkpoint_levels),
                "n_classifier_seeds": len(classifier_seed_levels),
                "classifier_seeds": " ".join(classifier_seed_levels),
                "n_technical_observations_aggregated": len(items),
                "baseline_accuracy_mean": mean(baseline_by_group[(classifier, subject, recipe)]),
                "processed_accuracy_mean": mean(processed_acc),
                "delta_accuracy": mean(deltas),
                "delta_accuracy_median_over_nuisance": median(deltas),
                "delta_accuracy_sd_over_nuisance": sample_sd(deltas),
                "delta_accuracy_descriptive_nuisance_bootstrap_ci95_low": within_low,
                "delta_accuracy_descriptive_nuisance_bootstrap_ci95_high": within_high,
                "contamination_delta_sd_within_subject": sample_sd(by_contamination),
                "checkpoint_delta_sd_within_subject": sample_sd(by_checkpoint),
                "classifier_seed_delta_sd_within_subject": sample_sd(by_classifier_seed) if by_classifier_seed else "",
                "note": "Subject row aggregates technical/nuisance repeats before inferential testing; within-subject CI is descriptive and not used as n for p-values.",
            }
        )
    return out


def group_means(items: list[dict[str, Any]], key: str) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for item in items:
        value = str(item.get(key, ""))
        if value == "":
            continue
        grouped[value].append(float(item["delta_accuracy_vs_noisy_noisy"]))
    return [mean(values) for _, values in sorted(grouped.items())]


def effect_analysis_role(classifier: str, base: int, confirmatory_csp_base: int) -> tuple[str, str]:
    if classifier == "csp_lda":
        if base == confirmatory_csp_base:
            return "confirmatory_downstream_effects", "confirmatory_csp_fixed_width"
        return "exploratory_width_effects", "exploratory_csp_width"
    return "confirmatory_downstream_effects", "confirmatory_neural_decoder_repeated_seed"


def build_effect_rows(
    subject_rows: list[dict[str, Any]],
    *,
    run_id: str,
    n_bootstrap: int,
    bootstrap_seed: int,
    confirmatory_csp_base: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in subject_rows:
        grouped[(str(row["classifier"]), str(row["recipe"]), int(row["base"]))].append(row)

    out: list[dict[str, Any]] = []
    for idx, ((classifier, recipe, base), items) in enumerate(sorted(grouped.items())):
        deltas = [float(item["delta_accuracy"]) for item in items]
        baseline = [float(item["baseline_accuracy_mean"]) for item in items]
        processed = [float(item["processed_accuracy_mean"]) for item in items]
        ci_low, ci_high = bootstrap_mean_ci(deltas, n_bootstrap=n_bootstrap, seed=bootstrap_seed + 1000 * idx)
        analysis_family, analysis_role = effect_analysis_role(classifier, base, confirmatory_csp_base)
        row: dict[str, Any] = {
            "run_id": run_id,
            "analysis_family": analysis_family,
            "analysis_role": analysis_role,
            "classifier": classifier,
            "recipe": recipe,
            "base": base,
            "primary_independent_unit": "subject",
            "n_subjects": len(items),
            "n_subjects_for_inference": len(items),
            "subjects": " ".join(str(item["subject"]) for item in items),
            "n_technical_repeats_used_as_sample_size": 0,
            "n_contamination_seed_pairs_min": min(int(item["n_contamination_seed_pairs"]) for item in items),
            "n_contamination_seed_pairs_max": max(int(item["n_contamination_seed_pairs"]) for item in items),
            "n_checkpoint_seeds_min": min(int(item["n_checkpoint_seeds"]) for item in items),
            "n_checkpoint_seeds_max": max(int(item["n_checkpoint_seeds"]) for item in items),
            "n_classifier_seeds_min": min(int(item["n_classifier_seeds"]) for item in items),
            "n_classifier_seeds_max": max(int(item["n_classifier_seeds"]) for item in items),
            "baseline_accuracy_subject_mean": mean(baseline),
            "processed_accuracy_subject_mean": mean(processed),
            "mean_delta_accuracy": mean(deltas),
            "median_delta_accuracy": median(deltas),
            "sd_delta_accuracy_across_subjects": sample_sd(deltas),
            "bootstrap_ci95_low": ci_low,
            "bootstrap_ci95_high": ci_high,
            "n_bootstrap_subject_resamples": n_bootstrap,
            "subjects_below_noisy_noisy": sum(1 for delta in deltas if delta < 0),
            "subjects_above_noisy_noisy": sum(1 for delta in deltas if delta > 0),
            "wilcoxon_p_two_sided": exact_wilcoxon_p(deltas, "two-sided"),
            "wilcoxon_p_denoised_lt_noisy": exact_wilcoxon_p(deltas, "less"),
            "wilcoxon_p_denoised_gt_noisy": exact_wilcoxon_p(deltas, "greater"),
            "statistical_note": "Primary p-value uses n=9 subject-level deltas after nuisance aggregation; Wilcoxon is robustness only.",
        }
        row.update(one_sample_stats(deltas))
        out.append(row)
    bh_fdr(out, PRIMARY_P_KEY, "bh_q_primary_t_two_sided_within_family", subset_key="analysis_family")
    holm(out, PRIMARY_P_KEY, "holm_p_primary_t_two_sided_within_family", subset_key="analysis_family")
    bh_fdr(out, WILCOXON_P_KEY, "bh_q_wilcoxon_two_sided_within_family", subset_key="analysis_family")
    bh_fdr(out, PRIMARY_P_KEY, "bh_q_primary_t_two_sided_all_effects")
    return out


def build_nuisance_rows(subject_rows: list[dict[str, Any]], *, run_id: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in subject_rows:
        grouped[(str(row["classifier"]), str(row["recipe"]), int(row["base"]))].append(row)
    out: list[dict[str, Any]] = []
    for (classifier, recipe, base), items in sorted(grouped.items()):
        out.append(
            {
                "run_id": run_id,
                "classifier": classifier,
                "recipe": recipe,
                "base": base,
                "n_subjects": len(items),
                "primary_independent_unit": "subject",
                "mean_within_subject_delta_sd_over_all_nuisance": mean([float(item["delta_accuracy_sd_over_nuisance"]) for item in items]),
                "median_within_subject_delta_sd_over_all_nuisance": median([float(item["delta_accuracy_sd_over_nuisance"]) for item in items]),
                "mean_within_subject_contamination_delta_sd": mean([float(item["contamination_delta_sd_within_subject"]) for item in items]),
                "mean_within_subject_checkpoint_delta_sd": mean([float(item["checkpoint_delta_sd_within_subject"]) for item in items]),
                "mean_within_subject_classifier_seed_delta_sd": mean(
                    [
                        float(item["classifier_seed_delta_sd_within_subject"])
                        for item in items
                        if is_finite_number(item.get("classifier_seed_delta_sd_within_subject"))
                    ]
                )
                if classifier != "csp_lda"
                else "",
                "note": "Nuisance variability is summarized separately and is not pooled into the inferential sample size.",
            }
        )
    return out


def build_width_contrasts(
    subject_rows: list[dict[str, Any]],
    *,
    run_id: str,
    n_bootstrap: int,
    bootstrap_seed: int,
    reference_base: int,
) -> list[dict[str, Any]]:
    by_subject_cell = {
        (str(row["classifier"]), str(row["recipe"]), str(row["subject"]), int(row["base"])): float(row["delta_accuracy"])
        for row in subject_rows
    }
    classifiers = sorted({str(row["classifier"]) for row in subject_rows})
    recipes = sorted({str(row["recipe"]) for row in subject_rows})
    bases_by_classifier_recipe: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in subject_rows:
        bases_by_classifier_recipe[(str(row["classifier"]), str(row["recipe"]))].add(int(row["base"]))
    out: list[dict[str, Any]] = []
    for classifier in classifiers:
        for recipe in recipes:
            bases = sorted(bases_by_classifier_recipe.get((classifier, recipe), set()))
            if reference_base not in bases:
                continue
            for base in bases:
                if base == reference_base:
                    continue
                subjects = []
                contrasts = []
                for subject in DEFAULT_SUBJECTS:
                    key = (classifier, recipe, subject, base)
                    ref_key = (classifier, recipe, subject, reference_base)
                    if key in by_subject_cell and ref_key in by_subject_cell:
                        subjects.append(subject)
                        contrasts.append(by_subject_cell[key] - by_subject_cell[ref_key])
                if not contrasts:
                    continue
                ci_low, ci_high = bootstrap_mean_ci(
                    contrasts,
                    n_bootstrap=n_bootstrap,
                    seed=bootstrap_seed + 50000 + 1000 * len(out),
                )
                row: dict[str, Any] = {
                    "run_id": run_id,
                    "analysis_family": "repeated_measures_width_contrasts",
                    "analysis_role": "sensitivity_width_fixed_effect_subject_intercept",
                    "classifier": classifier,
                    "recipe": recipe,
                    "reference_base": reference_base,
                    "contrast_base": base,
                    "contrast": f"base{base}_minus_base{reference_base}",
                    "primary_independent_unit": "subject",
                    "n_subjects": len(subjects),
                    "subjects": " ".join(subjects),
                    "mean_delta_accuracy_contrast": mean(contrasts),
                    "median_delta_accuracy_contrast": median(contrasts),
                    "sd_contrast_across_subjects": sample_sd(contrasts),
                    "bootstrap_ci95_low": ci_low,
                    "bootstrap_ci95_high": ci_high,
                    "n_bootstrap_subject_resamples": n_bootstrap,
                    "wilcoxon_p_two_sided": exact_wilcoxon_p(contrasts, "two-sided"),
                    "model_note": "Subject-intercept repeated-measures width contrast on subject-level deltas; positive means contrast_base has larger denoised-minus-noisy delta than reference_base.",
                }
                row.update(one_sample_stats(contrasts))
                out.append(row)
    bh_fdr(out, PRIMARY_P_KEY, "bh_q_primary_t_two_sided_within_family")
    holm(out, PRIMARY_P_KEY, "holm_p_primary_t_two_sided_within_family")
    bh_fdr(out, WILCOXON_P_KEY, "bh_q_wilcoxon_two_sided_within_family")
    return out


def write_markdown(
    path: Path,
    *,
    run_id: str,
    effects: list[dict[str, Any]],
    width_contrasts: list[dict[str, Any]],
    nuisance_rows: list[dict[str, Any]],
    a2_dir: Path,
    a3_dirs: list[Path],
) -> None:
    confirmatory = [row for row in effects if row["analysis_family"] == "confirmatory_downstream_effects"]
    exploratory = [row for row in effects if row["analysis_family"] == "exploratory_width_effects"]
    lines = ["# Hierarchical Downstream Statistics", ""]
    lines.append("## Protocol")
    lines.append("")
    lines.append(f"- Run ID: `{run_id}`.")
    lines.append(f"- A2 CSP+LDA input: `{a2_dir}`.")
    lines.append(f"- A3 neural inputs: `{', '.join(str(path) for path in a3_dirs)}`.")
    lines.append("- Primary independent statistical unit: human subject.")
    lines.append("- BCI IV-2a subject-level sample size remains `n=9`.")
    lines.append("- Contamination seeds, denoiser checkpoints, and classifier seeds are nuisance repetitions aggregated within subject before inference.")
    lines.append("- Primary confidence intervals are 95% bootstrap intervals obtained by resampling subjects only.")
    lines.append("- Primary p-values are one-sample paired t-tests on subject-level deltas; Wilcoxon signed-rank tests are reported as non-parametric robustness checks.")
    lines.append("- Multiple-comparison correction is explicit: confirmatory downstream effects and exploratory width effects are corrected as separate families, with an additional all-effect BH column in the CSV.")
    lines.append("- Width sensitivity uses subject-intercept repeated-measures contrasts on subject-level deltas, with `base16` as the reference width.")
    lines.append("")
    lines.append("## Confirmatory Effects")
    lines.append("")
    lines.append(effect_table(confirmatory))
    lines.append("")
    lines.append("## Exploratory CSP Width Effects")
    lines.append("")
    lines.append(effect_table(exploratory))
    lines.append("")
    lines.append("## Repeated-Measures Width Contrasts")
    lines.append("")
    lines.append(width_table(width_contrasts))
    lines.append("")
    lines.append("## Nuisance Variability")
    lines.append("")
    lines.append("| Classifier | Recipe | Base | n | All nuisance SD | Contamination SD | Checkpoint SD | Classifier-seed SD |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in nuisance_rows:
        lines.append(
            f"| {row['classifier']} | {row['recipe']} | {row['base']} | {row['n_subjects']} | "
            f"{float(row['mean_within_subject_delta_sd_over_all_nuisance']):.6f} | "
            f"{float(row['mean_within_subject_contamination_delta_sd']):.6f} | "
            f"{float(row['mean_within_subject_checkpoint_delta_sd']):.6f} | "
            f"{format_float(row['mean_within_subject_classifier_seed_delta_sd'])} |"
        )
    lines.append("")
    lines.append("## Interpretation Note")
    lines.append("")
    lines.append("The statistical sample size for BCI IV-2a downstream inference is the number of subjects, not the number of contamination/checkpoint/classifier-seed repetitions. Technical repetitions are used to stabilize each subject's paired delta and to characterize nuisance variability, but they are not allowed to create artificially small p-values.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def effect_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Classifier | Recipe | Base | n | Mean delta | 95% subject CI | Subject SD | t p | BH q | Wilcoxon p | Wilcoxon q |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['classifier']} | {row['recipe']} | {row['base']} | {row['n_subjects']} | "
            f"{float(row['mean_delta_accuracy']):+.6f} | "
            f"[{float(row['bootstrap_ci95_low']):+.6f}, {float(row['bootstrap_ci95_high']):+.6f}] | "
            f"{float(row['sd_delta_accuracy_across_subjects']):.6f} | "
            f"{float(row['primary_t_p_two_sided']):.6f} | "
            f"{float(row['bh_q_primary_t_two_sided_within_family']):.6f} | "
            f"{float(row['wilcoxon_p_two_sided']):.6f} | "
            f"{float(row['bh_q_wilcoxon_two_sided_within_family']):.6f} |"
        )
    return "\n".join(lines)


def width_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Classifier | Recipe | Contrast | n | Mean contrast | 95% subject CI | t p | BH q | Wilcoxon p |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['classifier']} | {row['recipe']} | {row['contrast']} | {row['n_subjects']} | "
            f"{float(row['mean_delta_accuracy_contrast']):+.6f} | "
            f"[{float(row['bootstrap_ci95_low']):+.6f}, {float(row['bootstrap_ci95_high']):+.6f}] | "
            f"{float(row['primary_t_p_two_sided']):.6f} | "
            f"{float(row['bh_q_primary_t_two_sided_within_family']):.6f} | "
            f"{float(row['wilcoxon_p_two_sided']):.6f} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.n_bootstrap < 10000:
        raise ValueError("--n-bootstrap must be at least 10000")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    a3_dirs = expand_dirs(args.a3_dir)

    a2_baseline, a2_processed = normalized_a2_rows(args.a2_dir)
    a3_baseline, a3_processed = normalized_a3_rows(a3_dirs)
    baseline_rows = a2_baseline + a3_baseline
    processed_rows = a2_processed + a3_processed

    subject_rows = aggregate_subject_rows(
        run_id=args.run_id,
        baseline_rows=baseline_rows,
        processed_rows=processed_rows,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )
    effects = build_effect_rows(
        subject_rows,
        run_id=args.run_id,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
        confirmatory_csp_base=args.confirmatory_csp_base,
    )
    nuisance_rows = build_nuisance_rows(subject_rows, run_id=args.run_id)
    width_contrasts = build_width_contrasts(
        subject_rows,
        run_id=args.run_id,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
        reference_base=args.reference_width_base,
    )

    confirmatory = [row for row in effects if row["analysis_family"] == "confirmatory_downstream_effects"]
    exploratory = [row for row in effects if row["analysis_family"] == "exploratory_width_effects"]

    write_csv(args.output_dir / "hierarchical_subject_deltas.csv", subject_rows)
    write_csv(args.output_dir / "hierarchical_all_effects.csv", effects)
    write_csv(args.output_dir / "hierarchical_confirmatory_effects.csv", confirmatory)
    write_csv(args.output_dir / "hierarchical_exploratory_width_effects.csv", exploratory)
    write_csv(args.output_dir / "hierarchical_nuisance_variability.csv", nuisance_rows)
    write_csv(args.output_dir / "hierarchical_repeated_measures_width_contrasts.csv", width_contrasts)

    summary = {
        "info": {
            "run_id": args.run_id,
            "a2_dir": str(args.a2_dir),
            "a3_dirs": [str(path) for path in a3_dirs],
            "primary_independent_statistical_unit": "subject",
            "bci_iv_2a_n_subjects": 9,
            "technical_repeats_as_inferential_n": False,
            "n_bootstrap_subject_resamples": args.n_bootstrap,
            "confirmatory_family_size": len(confirmatory),
            "exploratory_width_family_size": len(exploratory),
            "width_contrast_family_size": len(width_contrasts),
        },
        "confirmatory_effects": confirmatory,
        "exploratory_width_effects": exploratory,
        "repeated_measures_width_contrasts": width_contrasts,
        "nuisance_variability": nuisance_rows,
    }
    (args.output_dir / "hierarchical_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(
        args.output_dir / "hierarchical_summary.md",
        run_id=args.run_id,
        effects=effects,
        width_contrasts=width_contrasts,
        nuisance_rows=nuisance_rows,
        a2_dir=args.a2_dir,
        a3_dirs=a3_dirs,
    )
    print(f"[written] {args.output_dir / 'hierarchical_summary.md'}", flush=True)
    print(
        f"[done] subject_rows={len(subject_rows)} confirmatory={len(confirmatory)} "
        f"exploratory={len(exploratory)} width_contrasts={len(width_contrasts)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
