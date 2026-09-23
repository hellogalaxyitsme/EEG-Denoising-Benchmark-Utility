#!/usr/bin/env python3
"""metric-utility direct same-BCI reconstruction metric versus downstream utility analysis."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from eeg_denoise_benchmark.eval.metrics import compute_denoising_metrics  # noqa: E402
from scripts.evaluate_bci2a_downstream_contamination_types_csp_lda import (  # noqa: E402
    DEFAULT_RECIPES,
    DEFAULT_SUBJECTS,
    load_emg_pool,
    make_recipe_noisy_epochs,
    parse_recipes,
    parse_subjects,
)
from scripts.evaluate_bci2a_downstream_csp_lda import (  # noqa: E402
    FS_MODEL,
    checkpoint_seed,
    denoise_epochs,
    load_bci2a_trials,
    load_checkpoint_model,
)


METRICS = ["CC", "RMSE", "SDR", "T_RRMSE", "S_RRMSE"]
HIGHER_IS_BETTER = {"CC", "SDR"}


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
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--classifier-seed-dir",
        type=Path,
        action="append",
        required=True,
        help="One or more classifier_seed result directories. Comma-separated entries are also accepted.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--subjects", default=",".join(DEFAULT_SUBJECTS))
    parser.add_argument("--recipes", default=",".join(DEFAULT_RECIPES))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-conditions", type=int, default=0)
    parser.add_argument("--trial-start-sec", type=float, default=2.5)
    parser.add_argument("--trial-stop-sec", type=float, default=6.0)
    parser.add_argument("--eog-index", type=int, default=22)
    parser.add_argument("--include-artifact-trials", action="store_true")
    parser.add_argument("--snr-min-db", type=float, default=-6.0)
    parser.add_argument("--snr-max-db", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def expand_dirs(values: list[Path]) -> list[Path]:
    out: list[Path] = []
    for value in values:
        parts = [part.strip() for part in str(value).split(",")]
        out.extend(Path(part) for part in parts if part)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    preferred = [
        "run_id",
        "analysis",
        "classifier",
        "subject",
        "recipe",
        "seed_pair_index",
        "train_contamination_seed",
        "test_contamination_seed",
        "base",
        "checkpoint_seed",
        "checkpoint",
        "metric",
        "metric_value",
        "fidelity_oriented_metric_value",
        "delta_accuracy_vs_noisy_noisy",
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


def sample_sd(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def condition_key(row: dict[str, Any] | dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        str(row["subject"]),
        str(row["recipe"]),
        str(row["seed_pair_index"]),
        str(row["base"]),
        str(row["checkpoint_seed"]),
    )


def checkpoint_path_key(path: str) -> str:
    return str(Path(path))


def checkpoint_base(path: Path, cfg: dict[str, Any] | None = None) -> int | None:
    if cfg and str(cfg.get("base", "")) != "":
        return int(cfg["base"])
    match = re.search(r"base(\d+)", str(path))
    return int(match.group(1)) if match else None


def load_subject_clean(args: argparse.Namespace, subject: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    return train_clean, train_eog, test_clean, test_eog, y_train, y_test


def flatten_trials_channels(x: np.ndarray) -> np.ndarray:
    n_trials, n_channels, n_times = x.shape
    return x.reshape(n_trials * n_channels, n_times)


def metric_condition_id(row: dict[str, Any] | dict[str, str]) -> tuple[str, str, str, str, str, str]:
    return (*condition_key(row), checkpoint_path_key(str(row["checkpoint"])))


def build_metric_worklist(
    checkpoint_rows: list[dict[str, str]],
    classifier_seed_rows: list[dict[str, str]],
    *,
    subjects: set[str],
    recipes: set[str],
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    for source, rows in [("csp_lda", checkpoint_rows), ("neural", classifier_seed_rows)]:
        for row in rows:
            if row.get("condition") != "denoised_denoised":
                continue
            if str(row["subject"]) not in subjects:
                continue
            if str(row["recipe"]) not in recipes:
                continue
            key = metric_condition_id(row)
            if key in by_key:
                continue
            by_key[key] = {
                "subject": row["subject"],
                "recipe": row["recipe"],
                "seed_pair_index": int(row["seed_pair_index"]),
                "train_contamination_seed": int(row["train_contamination_seed"]),
                "test_contamination_seed": int(row["test_contamination_seed"]),
                "base": int(row["base"]),
                "checkpoint_seed": int(row["checkpoint_seed"]),
                "checkpoint": row["checkpoint"],
                "source": source,
            }
    return sorted(by_key.values(), key=lambda row: (row["recipe"], row["subject"], row["seed_pair_index"], row["base"], row["checkpoint_seed"], row["checkpoint"]))


def compute_reconstruction_metric_rows(
    args: argparse.Namespace,
    *,
    worklist: list[dict[str, Any]],
    subjects: list[str],
    recipes: list[str],
    device: torch.device,
) -> list[dict[str, Any]]:
    output_path = args.output_dir / "metric-utility_bci_reconstruction_metric_rows.csv"
    rows: list[dict[str, Any]] = read_csv(output_path) if args.resume and output_path.exists() else []
    done = {metric_condition_id(row) for row in rows}
    emg_pool = load_emg_pool(args.emg_pool)
    subject_cache = {subject: load_subject_clean(args, subject) for subject in subjects}
    model_cache: dict[str, tuple[torch.nn.Module, dict[str, Any], int]] = {}

    for index, item in enumerate(worklist, start=1):
        key = metric_condition_id(item)
        if key in done:
            continue
        checkpoint_path = Path(str(item["checkpoint"]))
        checkpoint_string = checkpoint_path_key(str(checkpoint_path))
        if checkpoint_string not in model_cache:
            model_cache[checkpoint_string] = load_checkpoint_model(checkpoint_path, device)
        model, cfg, n_params = model_cache[checkpoint_string]
        _train_clean, _train_eog, test_clean, test_eog, _y_train, _y_test = subject_cache[str(item["subject"])]
        test_noisy, _test_artifact, test_noise = make_recipe_noisy_epochs(
            test_clean,
            test_eog,
            emg_pool,
            recipe=str(item["recipe"]),
            seed=int(item["test_contamination_seed"]),
            snr_min_db=args.snr_min_db,
            snr_max_db=args.snr_max_db,
        )
        test_denoised = denoise_epochs(model, test_noisy, device=device, batch_size=args.batch_size)
        metrics = compute_denoising_metrics(
            target=flatten_trials_channels(test_clean),
            prediction=flatten_trials_channels(test_denoised),
            fs=float(FS_MODEL),
        )
        row = {
            "run_id": args.run_id,
            "subject": item["subject"],
            "recipe": item["recipe"],
            "seed_pair_index": item["seed_pair_index"],
            "train_contamination_seed": item["train_contamination_seed"],
            "test_contamination_seed": item["test_contamination_seed"],
            "split": "test",
            "base": checkpoint_base(checkpoint_path, cfg),
            "checkpoint_seed": checkpoint_seed(checkpoint_path, cfg),
            "checkpoint": checkpoint_string,
            "trainable_parameters": int(n_params),
            "n_trials": int(test_clean.shape[0]),
            "n_channels": int(test_clean.shape[1]),
            "n_trial_channel_samples": int(test_clean.shape[0] * test_clean.shape[1]),
            "snr_mean_db": test_noise["snr_mean_db"],
            "snr_std_db": test_noise["snr_std_db"],
        }
        for metric in METRICS:
            row[metric] = metrics[metric]
        rows.append(row)
        done.add(key)
        write_csv(output_path, rows)
        print(
            f"[metric] {index}/{len(worklist)} subject={item['subject']} recipe={item['recipe']} "
            f"seed_pair={item['seed_pair_index']} base={row['base']} checkpoint_seed={row['checkpoint_seed']} "
            f"CC={row['CC']:.6f} SDR={row['SDR']:.6f}",
            flush=True,
        )
    return rows


def aggregate_neural_utility_rows(classifier_seed_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    unique_rows: dict[tuple[str, str, str, str, str, str, str], dict[str, str]] = {}
    for row in classifier_seed_rows:
        if row.get("condition") != "denoised_denoised":
            continue
        key = (
            row["classifier"],
            row["classifier_seed"],
            row["subject"],
            row["recipe"],
            row["seed_pair_index"],
            row["base"],
            row["checkpoint_seed"],
        )
        unique_rows.setdefault(key, row)
    grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in unique_rows.values():
        grouped[
            (
                row["classifier"],
                row["subject"],
                row["recipe"],
                row["seed_pair_index"],
                row["base"],
                row["checkpoint_seed"],
            )
        ].append(row)
    out: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        classifier, subject, recipe, seed_pair_index, base, checkpoint_seed = key
        first = items[0]
        classifier_seeds = sorted({str(item["classifier_seed"]) for item in items})
        out.append(
            {
                "classifier": classifier,
                "subject": subject,
                "recipe": recipe,
                "seed_pair_index": int(seed_pair_index),
                "train_contamination_seed": int(first["train_contamination_seed"]),
                "test_contamination_seed": int(first["test_contamination_seed"]),
                "base": int(base),
                "checkpoint_seed": int(checkpoint_seed),
                "checkpoint": first["checkpoint"],
                "n_classifier_seeds": len(classifier_seeds),
                "classifier_seeds": " ".join(classifier_seeds),
                "accuracy": mean([float(item["accuracy"]) for item in items]),
                "delta_accuracy_vs_noisy_noisy": mean([float(item["delta_accuracy_vs_noisy_noisy"]) for item in items]),
                "delta_accuracy_sd_over_classifier_seeds": sample_sd([float(item["delta_accuracy_vs_noisy_noisy"]) for item in items]),
            }
        )
    return out


def build_joined_rows(
    *,
    checkpoint_rows: list[dict[str, str]],
    neural_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    metric_by_key = {condition_key(row): row for row in metric_rows}
    joined: list[dict[str, Any]] = []

    def append_metric_rows(classifier: str, utility_row: dict[str, Any] | dict[str, str], analysis: str) -> None:
        metric_row = metric_by_key.get(condition_key(utility_row))
        if metric_row is None:
            return
        for metric in METRICS:
            raw_value = float(metric_row[metric])
            oriented = raw_value if metric in HIGHER_IS_BETTER else -raw_value
            joined.append(
                {
                    "run_id": metric_row["run_id"],
                    "analysis": analysis,
                    "classifier": classifier,
                    "subject": utility_row["subject"],
                    "recipe": utility_row["recipe"],
                    "seed_pair_index": int(utility_row["seed_pair_index"]),
                    "train_contamination_seed": int(utility_row["train_contamination_seed"]),
                    "test_contamination_seed": int(utility_row["test_contamination_seed"]),
                    "base": int(utility_row["base"]),
                    "checkpoint_seed": int(utility_row["checkpoint_seed"]),
                    "checkpoint": utility_row["checkpoint"],
                    "metric": metric,
                    "metric_value": raw_value,
                    "fidelity_oriented_metric_value": oriented,
                    "delta_accuracy_vs_noisy_noisy": float(utility_row["delta_accuracy_vs_noisy_noisy"]),
                    "accuracy": float(utility_row["accuracy"]),
                    "n_classifier_seeds": utility_row.get("n_classifier_seeds", ""),
                    "delta_accuracy_sd_over_classifier_seeds": utility_row.get("delta_accuracy_sd_over_classifier_seeds", ""),
                }
            )

    for row in checkpoint_rows:
        if row.get("condition") == "denoised_denoised":
            append_metric_rows("csp_lda", row, "csp_lda")
    for row in neural_rows:
        append_metric_rows(str(row["classifier"]), row, "neural")
    return joined


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(order):
        end = pos + 1
        while end < len(order) and abs(values[order[end]] - values[order[pos]]) < 1e-12:
            end += 1
        rank = (pos + 1 + end) / 2.0
        for idx in range(pos, end):
            ranks[order[idx]] = rank
        pos = end
    return ranks


def subject_center(values: list[float], subjects: list[str]) -> list[float]:
    by_subject: dict[str, list[float]] = defaultdict(list)
    for subject, value in zip(subjects, values):
        by_subject[subject].append(value)
    means = {subject: mean(vals) for subject, vals in by_subject.items()}
    return [value - means[subject] for subject, value in zip(subjects, values)]


def subject_intercept_regression(rows: list[dict[str, Any]]) -> dict[str, Any]:
    subjects = [str(row["subject"]) for row in rows]
    x = [float(row["fidelity_oriented_metric_value"]) for row in rows]
    y = [float(row["delta_accuracy_vs_noisy_noisy"]) for row in rows]
    xc = np.asarray(subject_center(x, subjects), dtype=np.float64)
    yc = np.asarray(subject_center(y, subjects), dtype=np.float64)
    denom = float(np.sum(xc * xc))
    if denom <= 1e-20:
        beta = float("nan")
        se = float("nan")
        p_value = float("nan")
    else:
        beta = float(np.sum(xc * yc) / denom)
        residual = yc - beta * xc
        cluster_scores: dict[str, float] = defaultdict(float)
        for subject, xi, ei in zip(subjects, xc, residual):
            cluster_scores[subject] += float(xi * ei)
        meat = sum(score * score for score in cluster_scores.values())
        g = len(cluster_scores)
        n = len(rows)
        finite = (g / max(g - 1, 1)) * ((n - 1) / max(n - 1, 1))
        se = math.sqrt(max(0.0, finite * meat / (denom * denom)))
        t_stat = beta / se if se > 0 else float("inf") if beta != 0 else 0.0
        p_value = float(2.0 * stats.t.sf(abs(t_stat), df=max(g - 1, 1)))
    centered_pearson = float(np.corrcoef(xc, yc)[0, 1]) if len(rows) > 2 and np.std(xc) > 0 and np.std(yc) > 0 else float("nan")
    x_rank_center = np.asarray(subject_center(rankdata(x), subjects), dtype=np.float64)
    y_rank_center = np.asarray(subject_center(rankdata(y), subjects), dtype=np.float64)
    centered_spearman = (
        float(np.corrcoef(x_rank_center, y_rank_center)[0, 1])
        if len(rows) > 2 and np.std(x_rank_center) > 0 and np.std(y_rank_center) > 0
        else float("nan")
    )
    pearson = stats.pearsonr(x, y) if len(rows) > 2 else (float("nan"), float("nan"))
    spearman = stats.spearmanr(x, y) if len(rows) > 2 else (float("nan"), float("nan"))
    return {
        "n_observations": len(rows),
        "n_subjects": len(set(subjects)),
        "subject_intercept_slope_oriented": beta,
        "subject_intercept_cluster_se": se,
        "subject_intercept_p_two_sided": p_value,
        "subject_centered_pearson_r": centered_pearson,
        "subject_centered_spearman_r": centered_spearman,
        "pooled_pearson_r_descriptive": float(pearson.statistic) if hasattr(pearson, "statistic") else float(pearson[0]),
        "pooled_pearson_p_descriptive": float(pearson.pvalue) if hasattr(pearson, "pvalue") else float(pearson[1]),
        "pooled_spearman_r_sensitivity": float(spearman.statistic) if hasattr(spearman, "statistic") else float(spearman[0]),
        "pooled_spearman_p_sensitivity": float(spearman.pvalue) if hasattr(spearman, "pvalue") else float(spearman[1]),
    }


def bh_fdr(rows: list[dict[str, Any]], p_key: str, q_key: str) -> None:
    indexed = sorted((float(row[p_key]), idx) for idx, row in enumerate(rows) if row.get(p_key) == row.get(p_key))
    m = len(indexed)
    adjusted: dict[int, float] = {}
    prev = 1.0
    for rank_from_end, (p_value, idx) in enumerate(reversed(indexed), start=1):
        rank = m - rank_from_end + 1
        q = min(prev, p_value * m / rank)
        prev = q
        adjusted[idx] = min(1.0, q)
    for idx, row in enumerate(rows):
        row[q_key] = adjusted.get(idx, float("nan"))


def analysis_rows(joined_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in joined_rows:
        grouped[(str(row["classifier"]), str(row["recipe"]), str(row["metric"]))].append(row)
    out: list[dict[str, Any]] = []
    for (classifier, recipe, metric), items in sorted(grouped.items()):
        result = {
            "classifier": classifier,
            "recipe": recipe,
            "metric": metric,
            "metric_direction": "higher_better" if metric in HIGHER_IS_BETTER else "lower_better",
            "raw_metric_slope_sign_note": "positive oriented slope means better reconstruction is associated with better downstream delta",
            "bases": " ".join(str(value) for value in sorted({int(item["base"]) for item in items})),
            "checkpoint_seeds": " ".join(str(value) for value in sorted({int(item["checkpoint_seed"]) for item in items})),
            "contamination_seed_pairs": " ".join(str(value) for value in sorted({int(item["seed_pair_index"]) for item in items})),
        }
        result.update(subject_intercept_regression(items))
        out.append(result)
    bh_fdr(out, "subject_intercept_p_two_sided", "bh_fdr_q_subject_intercept")
    return out


def write_markdown(path: Path, *, analysis: list[dict[str, Any]]) -> None:
    lines = ["# Same-BCI metric--utility analysis", ""]
    lines.append("Reconstruction metrics are computed directly on the held-out contaminated BCI IV-2a evaluation trials, using the clean pre-contamination BCI trial as target.")
    lines.append("")
    lines.append("## Subject-Intercept Regression")
    lines.append("")
    lines.append("Positive oriented slopes mean that better reconstruction is associated with a better downstream delta after accounting for subject identity. For error metrics, the oriented predictor is the negative error value.")
    lines.append("")
    lines.append("| Classifier | Recipe | Metric | n | Slope | p | BH-FDR q | Centered Pearson | Centered Spearman |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for row in analysis:
        lines.append(
            f"| {row['classifier']} | {row['recipe']} | {row['metric']} | {row['n_observations']} | "
            f"{row['subject_intercept_slope_oriented']:+.6f} | "
            f"{row['subject_intercept_p_two_sided']:.6f} | "
            f"{row['bh_fdr_q_subject_intercept']:.6f} | "
            f"{row['subject_centered_pearson_r']:+.6f} | "
            f"{row['subject_centered_spearman_r']:+.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subjects = parse_subjects(args.subjects)
    recipes = parse_recipes(args.recipes)
    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    checkpoint_rows = read_csv(args.checkpoint_dir / "contamination-seed_checkpoint_seed_rows.csv")
    classifier_seed_dirs = expand_dirs(args.classifier_seed_dir)
    classifier_seed_rows = []
    for classifier_seed_dir in classifier_seed_dirs:
        classifier_seed_rows.extend(read_csv(classifier_seed_dir / "classifier_seed_checkpoint_rows.csv"))
    worklist = build_metric_worklist(checkpoint_rows, classifier_seed_rows, subjects=set(subjects), recipes=set(recipes))
    if args.max_conditions > 0:
        worklist = worklist[: args.max_conditions]
    print(f"[start] run_id={args.run_id} metric_conditions={len(worklist)} device={device}", flush=True)
    metric_rows = compute_reconstruction_metric_rows(args, worklist=worklist, subjects=subjects, recipes=recipes, device=device)
    neural_utility = aggregate_neural_utility_rows(classifier_seed_rows)
    write_csv(args.output_dir / "metric-utility_neural_utility_classifier_seed_aggregated_rows.csv", neural_utility)
    joined = build_joined_rows(checkpoint_rows=checkpoint_rows, neural_rows=neural_utility, metric_rows=metric_rows)
    write_csv(args.output_dir / "metric-utility_metric_utility_joined_long_rows.csv", joined)
    analysis = analysis_rows(joined)
    write_csv(args.output_dir / "metric-utility_subject_intercept_metric_tests.csv", analysis)
    summary = {
        "info": {
            "run_id": args.run_id,
            "checkpoint_dir": str(args.checkpoint_dir),
            "classifier_seed_dirs": [str(path) for path in classifier_seed_dirs],
            "reconstruction_metric_source": "same held-out BCI IV-2a synthetic-contamination evaluation trials",
            "main_model": "subject-intercept regression on condition-level rows",
            "descriptive_secondary": "pooled Pearson correlations",
            "nonparametric_sensitivity": "Spearman correlations",
            "metrics": METRICS,
            "p_value_family": "all classifier x recipe x metric subject-intercept tests",
        },
        "analysis": analysis,
    }
    (args.output_dir / "metric-utility_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.output_dir / "metric-utility_summary.md", analysis=analysis)
    print(f"[written] {args.output_dir / 'metric-utility_summary.md'}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
