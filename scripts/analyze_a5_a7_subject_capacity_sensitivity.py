#!/usr/bin/env python3
"""A5 sensitivity: formal BCI zero-shot capacity analysis from A7 all-subject rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


WIDTHS = [2, 4, 6, 8, 16]
ADJACENT = [(2, 4), (4, 6), (6, 8), (8, 16)]
COMPACT_VS_LARGE = [(4, 16), (6, 16)]
METRICS = ["CC", "T_RRMSE", "S_RRMSE", "SDR"]
HIGHER_BETTER = {"CC", "SDR"}
MARGINS = {"CC": 0.005, "T_RRMSE": 0.005, "S_RRMSE": 0.005, "SDR": 0.10}
MARGIN_JUSTIFICATION = {
    "CC": "Absolute correlation changes below 0.005 are treated as practically negligible.",
    "T_RRMSE": "Absolute RRMSE changes below 0.005 are treated as practically negligible.",
    "S_RRMSE": "Absolute RRMSE changes below 0.005 are treated as practically negligible.",
    "SDR": "SDR changes below 0.10 dB are treated as practically negligible.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a7-subject-checkpoint-rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260816)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    preferred = [
        "run_id",
        "dataset",
        "metric",
        "comparison",
        "base_low",
        "base_high",
        "n_subjects",
        "n_matched_seed_subject_cells",
        "matched_seeds",
        "mean_incremental_improvement",
        "median_incremental_improvement",
        "sd_incremental_improvement",
        "ci95_low_subject_bootstrap",
        "ci95_high_subject_bootstrap",
        "equivalence_margin",
        "formal_status",
        "n_positive_subjects",
        "n_negative_subjects",
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def orient_increment(metric: str, low: float, high: float) -> float:
    return high - low if metric in HIGHER_BETTER else low - high


def bootstrap_ci(values: list[float], n_resamples: int, rng: np.random.Generator) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1 or n_resamples <= 0:
        return float(arr[0]), float(arr[0])
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else float("nan")


def sd(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def status(mean_value: float, ci_low: float, ci_high: float, margin: float) -> str:
    if ci_low >= 0.0 and ci_high <= margin:
        return "formal_saturation"
    if ci_high < 0.0 and abs(ci_high) <= margin and abs(ci_low) <= margin:
        return "noninferior_practically_negligible_regression"
    if ci_high < 0.0:
        return "larger_width_worse_transfer_inversion"
    if ci_low >= 0.0 and ci_high > margin:
        return "diminishing_returns_not_formal_saturation"
    if mean_value >= 0.0:
        return "diminishing_returns_not_formal_saturation"
    return "uncertain_or_no_consistent_gain"


def index_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str, int, int], dict[str, str]]:
    indexed: dict[tuple[str, str, int, int], dict[str, str]] = {}
    for row in rows:
        if row.get("kind") != "model":
            continue
        indexed[(row["dataset"], row["subject"], int(row["base"]), int(row["train_seed"]))] = row
    return indexed


def comparison_rows(
    rows: list[dict[str, str]],
    comparisons: list[tuple[int, int]],
    args: argparse.Namespace,
    *,
    family: str,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    indexed = index_rows(rows)
    datasets = sorted({row["dataset"] for row in rows if row.get("kind") == "model"})
    out: list[dict[str, Any]] = []
    for dataset in datasets:
        subjects = sorted({row["subject"] for row in rows if row.get("kind") == "model" and row["dataset"] == dataset})
        for metric in METRICS:
            for low, high in comparisons:
                subject_values: list[float] = []
                subject_seed_rows: list[dict[str, Any]] = []
                matched_seed_set: set[int] = set()
                for subject in subjects:
                    seeds = sorted(
                        {
                            seed
                            for ds, sub, base, seed in indexed
                            if ds == dataset and sub == subject and base in {low, high}
                        }
                    )
                    increments = []
                    matched = []
                    for seed in seeds:
                        low_row = indexed.get((dataset, subject, low, seed))
                        high_row = indexed.get((dataset, subject, high, seed))
                        if low_row is None or high_row is None:
                            continue
                        inc = orient_increment(metric, float(low_row[metric]), float(high_row[metric]))
                        increments.append(inc)
                        matched.append(seed)
                        matched_seed_set.add(seed)
                        subject_seed_rows.append(
                            {
                                "run_id": args.run_id,
                                "dataset": dataset,
                                "subject": subject,
                                "metric": metric,
                                "comparison": f"base{low}_to_base{high}",
                                "base_low": low,
                                "base_high": high,
                                "train_seed": seed,
                                "incremental_improvement": inc,
                            }
                        )
                    if increments:
                        subject_values.append(mean(increments))
                ci_low, ci_high = bootstrap_ci(subject_values, args.bootstrap_resamples, rng)
                margin = MARGINS[metric]
                out.append(
                    {
                        "run_id": args.run_id,
                        "family": family,
                        "dataset": dataset,
                        "artifact": "bci_eog_transfer",
                        "metric": metric,
                        "comparison": f"base{low}_to_base{high}",
                        "base_low": low,
                        "base_high": high,
                        "n_subjects": len(subject_values),
                        "subjects": " ".join(subjects),
                        "n_matched_seed_subject_cells": len(subject_seed_rows),
                        "matched_seeds": " ".join(str(seed) for seed in sorted(matched_seed_set)),
                        "mean_incremental_improvement": mean(subject_values),
                        "median_incremental_improvement": float(np.median(subject_values)) if subject_values else float("nan"),
                        "sd_incremental_improvement": sd(subject_values),
                        "ci95_low_subject_bootstrap": ci_low,
                        "ci95_high_subject_bootstrap": ci_high,
                        "equivalence_margin": margin,
                        "formal_status": status(mean(subject_values), ci_low, ci_high, margin),
                        "n_positive_subjects": sum(1 for value in subject_values if value > 0),
                        "n_negative_subjects": sum(1 for value in subject_values if value < 0),
                        "subject_increments": " ".join(f"{subject}:{value:+.6f}" for subject, value in zip(subjects, subject_values)),
                    }
                )
    return out


def write_markdown(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# A5/A7 Subject-Preserving BCI Capacity Sensitivity",
        "",
        "## Protocol",
        "",
        "- This analysis fills the post-A7 gap in the original A5 capacity analysis.",
        "- It uses A7 all-subject BCI zero-shot rows rather than the older single-subject zero-shot summary.",
        "- Checkpoint seeds are matched across widths within each subject.",
        "- Seed-level increments are averaged within subject before subject-bootstrap confidence intervals.",
        "- The inferential unit is the subject: `n=9` for IV-2a and `n=9` for IV-2b.",
        "- Formal saturation uses the same predefined A5 practical margins.",
        "",
        "## Adjacent Widths",
        "",
        "| dataset | metric | comparison | n | mean increment | 95% subject CI | margin | status | positive subjects |",
        "|---|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        if row["family"] != "adjacent":
            continue
        lines.append(
            f"| {row['dataset']} | {row['metric']} | {row['comparison']} | {row['n_subjects']} | "
            f"{float(row['mean_incremental_improvement']):+.6f} | "
            f"[{float(row['ci95_low_subject_bootstrap']):+.6f}, {float(row['ci95_high_subject_bootstrap']):+.6f}] | "
            f"{float(row['equivalence_margin']):.6f} | {row['formal_status']} | "
            f"{row['n_positive_subjects']}/{row['n_subjects']} |"
        )
    lines.extend(
        [
            "",
            "## Compact Versus Large",
            "",
            "| dataset | metric | comparison | n | mean increment | 95% subject CI | margin | status | positive subjects |",
            "|---|---|---|---:|---:|---:|---:|---|---:|",
        ]
    )
    for row in rows:
        if row["family"] != "compact_vs_large":
            continue
        lines.append(
            f"| {row['dataset']} | {row['metric']} | {row['comparison']} | {row['n_subjects']} | "
            f"{float(row['mean_incremental_improvement']):+.6f} | "
            f"[{float(row['ci95_low_subject_bootstrap']):+.6f}, {float(row['ci95_high_subject_bootstrap']):+.6f}] | "
            f"{float(row['equivalence_margin']):.6f} | {row['formal_status']} | "
            f"{row['n_positive_subjects']}/{row['n_subjects']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- BCI transfer capacity claims should use this all-subject sensitivity rather than the older single-subject BCI input in the original A5 run.",
            "- IV-2a shows strong base2-to-base4 gains but mixed/near-zero behavior after base6 for CC/SDR/T-RRMSE; S-RRMSE continues improving for more subjects.",
            "- IV-2b shows strong base2-to-base4 gains and a subject-consistent base8-to-base16 inversion for CC, SDR, and T-RRMSE.",
            "- Formal saturation and diminishing returns are metric- and dataset-specific; IV-2b larger-width regression is dataset-specific transfer behavior.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.a7_subject_checkpoint_rows)
    rng = np.random.default_rng(args.bootstrap_seed)
    adjacent = comparison_rows(rows, ADJACENT, args, family="adjacent", rng=rng)
    compact = comparison_rows(rows, COMPACT_VS_LARGE, args, family="compact_vs_large", rng=rng)
    out = adjacent + compact
    write_csv(args.output_dir / "a5_a7_subject_capacity_rows.csv", out)
    summary = {
        "info": {
            "run_id": args.run_id,
            "input": str(args.a7_subject_checkpoint_rows),
            "bootstrap_resamples": args.bootstrap_resamples,
            "bootstrap_seed": args.bootstrap_seed,
            "margins": MARGINS,
            "margin_justification": MARGIN_JUSTIFICATION,
            "method": "matched checkpoint seeds within subject; average seed increments within subject; resample subjects for CI",
        },
        "rows": out,
    }
    (args.output_dir / "a5_a7_subject_capacity_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.output_dir / "a5_a7_subject_capacity_summary.md", out, args)
    print(f"[done] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
