#!/usr/bin/env python3
"""classifier_seed neural-decoder classifier_seed uncertainty for BCI IV-2a downstream evaluation."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import re
import statistics
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "src"))

from evaluate_bci2a_all_subjects_braindecode_classifiers import (  # noqa: E402
    METRIC_KEYS,
    build_classifier_model,
    standardize_train_test,
)
from scripts.evaluate_bci2a_downstream_contamination_types_csp_lda import (  # noqa: E402
    DEFAULT_RECIPES,
    DEFAULT_SUBJECTS,
    load_emg_pool,
    make_recipe_noisy_epochs,
    parse_recipes,
    parse_subjects,
)
from scripts.evaluate_bci2a_downstream_csp_lda import (  # noqa: E402
    CLASS_NAMES,
    FS_MODEL,
    bandpass_epochs,
    checkpoint_seed,
    denoise_epochs,
    expand_checkpoints,
    load_bci2a_trials,
    load_checkpoint_model,
)
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402


DEFAULT_CLASSIFIERS = ["eegnet", "shallowfbcsp", "deep4net", "eegconformer"]


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
    parser.add_argument("--recipes", default="eog")
    parser.add_argument("--classifiers", default=",".join(DEFAULT_CLASSIFIERS))
    parser.add_argument(
        "--checkpoint-glob",
        action="append",
        required=True,
        help="Checkpoint glob pattern. Repeat this option to combine checkpoint families.",
    )
    parser.add_argument("--bases", default="6,16")
    parser.add_argument("--checkpoint-seeds", default="42,43,44")
    parser.add_argument("--classifier-seeds", default="501,502,503")
    parser.add_argument("--test-seeds", default="42,43,44")
    parser.add_argument("--train-seeds", default="1042,1043,1044")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--denoiser-batch-size", type=int, default=256)
    parser.add_argument("--classifier-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--snr-min-db", type=float, default=-6.0)
    parser.add_argument("--snr-max-db", type=float, default=2.0)
    parser.add_argument("--trial-start-sec", type=float, default=2.5)
    parser.add_argument("--trial-stop-sec", type=float, default=6.0)
    parser.add_argument("--bandpass-low-hz", type=float, default=8.0)
    parser.add_argument("--bandpass-high-hz", type=float, default=30.0)
    parser.add_argument("--eog-index", type=int, default=22)
    parser.add_argument("--include-artifact-trials", action="store_true")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260814)
    parser.add_argument("--resume", action="store_true", help="Reuse existing raw rows and skip completed cells.")
    return parser.parse_args()


def parse_csv_list(raw: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated value")
    return values


def parse_int_list(raw: str, *, name: str) -> list[int]:
    values = [int(part.strip()) for part in raw.replace(" ", ",").split(",") if part.strip()]
    if not values:
        raise ValueError(f"--{name} must contain at least one integer")
    return values


def checkpoint_base(path: Path, cfg: dict[str, Any] | None = None) -> int | None:
    if cfg and str(cfg.get("base", "")) != "":
        return int(cfg["base"])
    match = re.search(r"base(\d+)", str(path))
    return int(match.group(1)) if match else None


def expand_checkpoint_patterns(patterns: list[str], *, bases: set[int], seeds: set[int]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in expand_checkpoints(pattern):
            base = checkpoint_base(path)
            seed_match = re.search(r"seed(\d+)", str(path))
            seed = int(seed_match.group(1)) if seed_match else None
            if base not in bases or seed not in seeds:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(path)
    paths.sort(key=lambda path: (checkpoint_base(path) or 10**9, int(re.search(r"seed(\d+)", str(path)).group(1)) if re.search(r"seed(\d+)", str(path)) else 10**9, str(path)))
    expected = len(bases) * len(seeds)
    if len(paths) != expected:
        found = [(checkpoint_base(path), re.search(r"seed(\d+)", str(path)).group(1) if re.search(r"seed(\d+)", str(path)) else "") for path in paths]
        raise ValueError(f"Expected {expected} checkpoints for bases={sorted(bases)} seeds={sorted(seeds)}, found {len(paths)}: {found}")
    return paths


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    preferred = [
        "run_id",
        "classifier",
        "classifier_seed",
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
        "baseline_accuracy",
        "accuracy",
        "delta_accuracy_vs_noisy_noisy",
        "balanced_accuracy",
        "cohen_kappa",
        "macro_f1",
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


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else float("nan")


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def sample_sd(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def percentile(values: list[float], p: float) -> float:
    xs = sorted(values)
    if not xs:
        return float("nan")
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def bootstrap_mean_ci(values: list[float], *, n_bootstrap: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    boot = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_bootstrap)]
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


def set_classifier_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


def train_classifier_predictions(
    *,
    classifier_name: str,
    train_epochs: np.ndarray,
    y_train: np.ndarray,
    test_epochs: np.ndarray,
    device: torch.device,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    set_classifier_seed(seed)
    train_x, test_x = standardize_train_test(train_epochs, test_epochs)
    model = build_classifier_model(
        classifier_name,
        n_chans=int(train_x.shape[1]),
        n_times=int(train_x.shape[2]),
        n_outputs=int(len(CLASS_NAMES)),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(y_train).long())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    start_time = time.time()
    last_loss = float("nan")
    model.train()
    for _epoch in range(epochs):
        total_loss = 0.0
        n_seen = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * int(xb.shape[0])
            n_seen += int(xb.shape[0])
        last_loss = total_loss / max(n_seen, 1)

    predictions: list[np.ndarray] = []
    test_tensor = torch.from_numpy(test_x).float()
    model.eval()
    with torch.no_grad():
        for start in range(0, len(test_tensor), batch_size):
            xb = test_tensor[start : start + batch_size].to(device)
            logits = model(xb)
            predictions.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
    info = {
        "classifier_epochs": int(epochs),
        "classifier_batch_size": int(batch_size),
        "classifier_lr": float(lr),
        "classifier_weight_decay": float(weight_decay),
        "classifier_final_train_loss": float(last_loss),
        "classifier_train_time_sec": float(time.time() - start_time),
    }
    return np.concatenate(predictions).astype(np.int64), info


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    return {
        "n_trials": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": json.dumps(cm.tolist()),
    }


def row_key(row: dict[str, Any] | dict[str, str]) -> tuple[str, ...]:
    return (
        str(row["classifier"]),
        str(row["classifier_seed"]),
        str(row["subject"]),
        str(row["recipe"]),
        str(row["seed_pair_index"]),
        str(row.get("base", "")),
        str(row.get("checkpoint_seed", "")),
        str(row["condition"]),
    )


def load_subject_clean(args: argparse.Namespace, subject: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
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
    return train_clean, train_eog, test_clean, test_eog, y_train, y_test, {"train": train_info, "test": test_info}


def load_models(checkpoints: list[Path], device: torch.device) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for checkpoint_path in checkpoints:
        model, cfg, n_params = load_checkpoint_model(checkpoint_path, device)
        base = checkpoint_base(checkpoint_path, cfg)
        train_seed = checkpoint_seed(checkpoint_path, cfg)
        models.append(
            {
                "checkpoint": str(checkpoint_path),
                "model": model,
                "cfg": cfg,
                "base": int(base) if base is not None else "",
                "checkpoint_seed": train_seed,
                "trainable_parameters": int(n_params),
            }
        )
    return models


def completed_keys(rows: list[dict[str, Any]] | list[dict[str, str]]) -> set[tuple[str, ...]]:
    return {row_key(row) for row in rows}


def summarize_subjects(
    baseline_rows: list[dict[str, Any]] | list[dict[str, str]],
    result_rows: list[dict[str, Any]] | list[dict[str, str]],
) -> list[dict[str, Any]]:
    baseline_group: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in baseline_rows:
        baseline_group[(str(row["classifier"]), str(row["subject"]), str(row["recipe"]))].append(float(row["accuracy"]))

    grouped: dict[tuple[str, str, str, int], list[dict[str, Any] | dict[str, str]]] = defaultdict(list)
    for row in result_rows:
        grouped[(str(row["classifier"]), str(row["subject"]), str(row["recipe"]), int(row["base"]))].append(row)

    out: list[dict[str, Any]] = []
    for (classifier, subject, recipe, base), items in sorted(grouped.items()):
        deltas = [float(item["delta_accuracy_vs_noisy_noisy"]) for item in items]
        acc = [float(item["accuracy"]) for item in items]
        by_classifier_seed: dict[str, list[float]] = defaultdict(list)
        by_checkpoint_seed: dict[str, list[float]] = defaultdict(list)
        by_contamination: dict[str, list[float]] = defaultdict(list)
        for item in items:
            delta = float(item["delta_accuracy_vs_noisy_noisy"])
            by_classifier_seed[str(item["classifier_seed"])].append(delta)
            by_checkpoint_seed[str(item["checkpoint_seed"])].append(delta)
            by_contamination[str(item["seed_pair_index"])].append(delta)
        classifier_seed_means = [mean(values) for _, values in sorted(by_classifier_seed.items())]
        checkpoint_seed_means = [mean(values) for _, values in sorted(by_checkpoint_seed.items())]
        contamination_means = [mean(values) for _, values in sorted(by_contamination.items())]
        out.append(
            {
                "classifier": classifier,
                "subject": subject,
                "recipe": recipe,
                "base": base,
                "n_contamination_seed_pairs": len(by_contamination),
                "n_checkpoint_seeds": len(by_checkpoint_seed),
                "n_classifier_seeds": len(by_classifier_seed),
                "n_observations_averaged": len(items),
                "baseline_accuracy_mean": mean(baseline_group[(classifier, subject, recipe)]),
                "processed_accuracy_mean": mean(acc),
                "delta_accuracy": mean(deltas),
                "delta_accuracy_sd_over_all_nuisance": sample_sd(deltas),
                "classifier_seed_delta_sd_within_subject": sample_sd(classifier_seed_means),
                "classifier_seed_delta_range_within_subject": max(classifier_seed_means) - min(classifier_seed_means),
                "checkpoint_seed_delta_sd_within_subject": sample_sd(checkpoint_seed_means),
                "checkpoint_seed_delta_range_within_subject": max(checkpoint_seed_means) - min(checkpoint_seed_means),
                "contamination_delta_sd_within_subject": sample_sd(contamination_means),
                "contamination_delta_range_within_subject": max(contamination_means) - min(contamination_means),
            }
        )
    return out


def primary_inference_rows(subject_rows: list[dict[str, Any]], *, n_bootstrap: int, bootstrap_seed: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in subject_rows:
        grouped[(str(row["classifier"]), str(row["recipe"]), int(row["base"]))].append(row)
    out: list[dict[str, Any]] = []
    for (classifier, recipe, base), items in sorted(grouped.items()):
        deltas = [float(item["delta_accuracy"]) for item in items]
        ci_low, ci_high = bootstrap_mean_ci(deltas, n_bootstrap=n_bootstrap, seed=bootstrap_seed + 1000 * len(out))
        out.append(
            {
                "classifier": classifier,
                "recipe": recipe,
                "base": base,
                "n_subjects": len(items),
                "subjects": " ".join(str(item["subject"]) for item in items),
                "n_contamination_seed_pairs_min": min(int(item["n_contamination_seed_pairs"]) for item in items),
                "n_contamination_seed_pairs_max": max(int(item["n_contamination_seed_pairs"]) for item in items),
                "n_checkpoint_seeds_min": min(int(item["n_checkpoint_seeds"]) for item in items),
                "n_checkpoint_seeds_max": max(int(item["n_checkpoint_seeds"]) for item in items),
                "n_classifier_seeds_min": min(int(item["n_classifier_seeds"]) for item in items),
                "n_classifier_seeds_max": max(int(item["n_classifier_seeds"]) for item in items),
                "baseline_accuracy_subject_mean": mean([float(item["baseline_accuracy_mean"]) for item in items]),
                "processed_accuracy_subject_mean": mean([float(item["processed_accuracy_mean"]) for item in items]),
                "mean_delta_accuracy": mean(deltas),
                "median_delta_accuracy": median(deltas),
                "sd_delta_accuracy_across_subjects": sample_sd(deltas),
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
                "n_bootstrap": n_bootstrap,
                "subjects_below_noisy_noisy": sum(1 for delta in deltas if delta < 0),
                "subjects_above_noisy_noisy": sum(1 for delta in deltas if delta > 0),
                "wilcoxon_p_processed_lt_noisy": exact_wilcoxon_p(deltas, "less"),
                "wilcoxon_p_processed_gt_noisy": exact_wilcoxon_p(deltas, "greater"),
            }
        )
    bh_fdr_adjust(out, "wilcoxon_p_processed_lt_noisy", "bh_fdr_q_processed_lt_noisy")
    return out


def nuisance_variability_rows(subject_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in subject_rows:
        grouped[(str(row["classifier"]), str(row["recipe"]), int(row["base"]))].append(row)
    out: list[dict[str, Any]] = []
    for (classifier, recipe, base), items in sorted(grouped.items()):
        out.append(
            {
                "classifier": classifier,
                "recipe": recipe,
                "base": base,
                "n_subjects": len(items),
                "mean_within_subject_classifier_seed_delta_sd": mean([float(item["classifier_seed_delta_sd_within_subject"]) for item in items]),
                "median_within_subject_classifier_seed_delta_sd": median([float(item["classifier_seed_delta_sd_within_subject"]) for item in items]),
                "mean_within_subject_classifier_seed_delta_range": mean([float(item["classifier_seed_delta_range_within_subject"]) for item in items]),
                "mean_within_subject_checkpoint_seed_delta_sd": mean([float(item["checkpoint_seed_delta_sd_within_subject"]) for item in items]),
                "mean_within_subject_checkpoint_seed_delta_range": mean([float(item["checkpoint_seed_delta_range_within_subject"]) for item in items]),
                "mean_within_subject_contamination_delta_sd": mean([float(item["contamination_delta_sd_within_subject"]) for item in items]),
                "mean_within_subject_contamination_delta_range": mean([float(item["contamination_delta_range_within_subject"]) for item in items]),
            }
        )
    return out


def variance_decomposition_rows(result_rows: list[dict[str, Any]] | list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any] | dict[str, str]]] = defaultdict(list)
    for row in result_rows:
        grouped[(str(row["classifier"]), str(row["recipe"]), int(row["base"]))].append(row)
    out: list[dict[str, Any]] = []
    for (classifier, recipe, base), items in sorted(grouped.items()):
        cell_values: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
        for row in items:
            key = (str(row["subject"]), str(row["seed_pair_index"]), str(row["checkpoint_seed"]), str(row["classifier_seed"]))
            cell_values[key].append(float(row["delta_accuracy_vs_noisy_noisy"]))
        observations = [
            {
                "subject": key[0],
                "contamination": key[1],
                "checkpoint": key[2],
                "classifier_seed": key[3],
                "delta": mean(values),
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
        classifier_seed_ss = factor_ss("classifier_seed")
        residual_ss = max(0.0, total_ss - subject_ss - contamination_ss - checkpoint_ss - classifier_seed_ss)
        denom = total_ss if total_ss > 0 else float("nan")
        subjects = sorted({str(obs["subject"]) for obs in observations})
        contaminations = sorted({str(obs["contamination"]) for obs in observations})
        checkpoints = sorted({str(obs["checkpoint"]) for obs in observations})
        classifier_seeds = sorted({str(obs["classifier_seed"]) for obs in observations})
        expected = len(subjects) * len(contaminations) * len(checkpoints) * len(classifier_seeds)
        out.append(
            {
                "classifier": classifier,
                "recipe": recipe,
                "base": base,
                "n_subjects": len(subjects),
                "n_contamination_seed_pairs": len(contaminations),
                "n_checkpoint_seeds": len(checkpoints),
                "n_classifier_seeds": len(classifier_seeds),
                "n_cells_observed": len(observations),
                "n_cells_expected_balanced": expected,
                "is_balanced": len(observations) == expected and all(len(values_for_cell) == 1 for values_for_cell in cell_values.values()),
                "total_ss": total_ss,
                "subject_main_effect_ss": subject_ss,
                "contamination_main_effect_ss": contamination_ss,
                "checkpoint_main_effect_ss": checkpoint_ss,
                "classifier_seed_main_effect_ss": classifier_seed_ss,
                "residual_interaction_ss": residual_ss,
                "subject_main_effect_prop": subject_ss / denom,
                "contamination_main_effect_prop": contamination_ss / denom,
                "checkpoint_main_effect_prop": checkpoint_ss / denom,
                "classifier_seed_main_effect_prop": classifier_seed_ss / denom,
                "residual_interaction_prop": residual_ss / denom,
                "note": "Fixed-effect descriptive decomposition of paired deltas; residual includes interactions and unexplained cell variation.",
            }
        )
    return out


def write_markdown(
    path: Path,
    *,
    args: argparse.Namespace,
    inference_rows: list[dict[str, Any]],
    variability_rows: list[dict[str, Any]],
    decomposition_rows: list[dict[str, Any]],
) -> None:
    lines = ["# Classifier-seed uncertainty summary", ""]
    lines.append("CSP+LDA is deterministic once the input data and configuration are fixed; this classifier_seed confirmatory run therefore targets neural-decoder initialization uncertainty.")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append(f"- Recipes: `{args.recipes}`.")
    lines.append(f"- Widths: `{args.bases}`.")
    lines.append(f"- Checkpoint seeds: `{args.checkpoint_seeds}`.")
    lines.append(f"- Train contamination seeds: `{args.train_seeds}`.")
    lines.append(f"- Test contamination seeds: `{args.test_seeds}`.")
    lines.append(f"- Classifier seeds: `{args.classifier_seeds}`.")
    lines.append(f"- Neural decoder epochs: `{args.epochs}`.")
    lines.append("- Primary inference averages contamination, checkpoint, and classifier_seed repetitions within each subject before testing.")
    lines.append("- Bootstrap confidence intervals resample subjects only.")
    lines.append("")
    lines.append("## Primary Subject-Level Inference")
    lines.append("")
    lines.append("| Classifier | Recipe | Base | n subjects | n contam | n checkpoints | n clf seeds | Baseline acc | Processed acc | Mean delta | 95% subject-bootstrap CI | Below noisy | Wilcoxon p lower | BH-FDR q |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in inference_rows:
        lines.append(
            f"| {row['classifier']} | {row['recipe']} | {row['base']} | {row['n_subjects']} | "
            f"{row['n_contamination_seed_pairs_min']}-{row['n_contamination_seed_pairs_max']} | "
            f"{row['n_checkpoint_seeds_min']}-{row['n_checkpoint_seeds_max']} | "
            f"{row['n_classifier_seeds_min']}-{row['n_classifier_seeds_max']} | "
            f"{row['baseline_accuracy_subject_mean']:.6f} | {row['processed_accuracy_subject_mean']:.6f} | "
            f"{row['mean_delta_accuracy']:+.6f} | "
            f"[{row['bootstrap_ci95_low']:+.6f}, {row['bootstrap_ci95_high']:+.6f}] | "
            f"{row['subjects_below_noisy_noisy']} | {row['wilcoxon_p_processed_lt_noisy']:.6f} | "
            f"{row['bh_fdr_q_processed_lt_noisy']:.6f} |"
        )
    lines.append("")
    lines.append("## Nuisance Variability")
    lines.append("")
    lines.append("| Classifier | Recipe | Base | Classifier-seed SD | Checkpoint-seed SD | Contamination SD |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in variability_rows:
        lines.append(
            f"| {row['classifier']} | {row['recipe']} | {row['base']} | "
            f"{row['mean_within_subject_classifier_seed_delta_sd']:.6f} | "
            f"{row['mean_within_subject_checkpoint_seed_delta_sd']:.6f} | "
            f"{row['mean_within_subject_contamination_delta_sd']:.6f} |"
        )
    lines.append("")
    lines.append("## Variance Decomposition")
    lines.append("")
    lines.append("| Classifier | Recipe | Base | Balanced | Subject % | Contamination % | Checkpoint % | Classifier seed % | Residual/interactions % |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in decomposition_rows:
        lines.append(
            f"| {row['classifier']} | {row['recipe']} | {row['base']} | {row['is_balanced']} | "
            f"{100.0 * row['subject_main_effect_prop']:.2f} | "
            f"{100.0 * row['contamination_main_effect_prop']:.2f} | "
            f"{100.0 * row['checkpoint_main_effect_prop']:.2f} | "
            f"{100.0 * row['classifier_seed_main_effect_prop']:.2f} | "
            f"{100.0 * row['residual_interaction_prop']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_and_write(
    args: argparse.Namespace,
    *,
    baseline_rows: list[dict[str, Any]] | list[dict[str, str]],
    result_rows: list[dict[str, Any]] | list[dict[str, str]],
) -> None:
    subject_rows = summarize_subjects(baseline_rows, result_rows)
    inference_rows = primary_inference_rows(subject_rows, n_bootstrap=args.n_bootstrap, bootstrap_seed=args.bootstrap_seed)
    variability_rows = nuisance_variability_rows(subject_rows)
    decomposition_rows = variance_decomposition_rows(result_rows)
    write_csv(args.output_dir / "classifier_seed_subject_primary_rows.csv", subject_rows)
    write_csv(args.output_dir / "classifier_seed_primary_subject_inference.csv", inference_rows)
    write_csv(args.output_dir / "classifier_seed_nuisance_variability.csv", variability_rows)
    write_csv(args.output_dir / "classifier_seed_variance_decomposition.csv", decomposition_rows)
    summary = {
        "info": {
            "run_id": args.run_id,
            "subjects": parse_subjects(args.subjects),
            "recipes": parse_recipes(args.recipes),
            "classifiers": parse_csv_list(args.classifiers),
            "bases": parse_int_list(args.bases, name="bases"),
            "checkpoint_seeds": parse_int_list(args.checkpoint_seeds, name="checkpoint-seeds"),
            "classifier_seeds": parse_int_list(args.classifier_seeds, name="classifier_seeds"),
            "train_contamination_seeds": parse_int_list(args.train_seeds, name="train-seeds"),
            "test_contamination_seeds": parse_int_list(args.test_seeds, name="test-seeds"),
            "primary_inferential_unit": "subject",
            "csp_lda_classifier_seed_note": "CSP+LDA is deterministic once input data and configuration are fixed.",
        },
        "primary_subject_inference": inference_rows,
        "nuisance_variability": variability_rows,
        "variance_decomposition": decomposition_rows,
    }
    (args.output_dir / "classifier_seed_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(
        args.output_dir / "classifier_seed_summary.md",
        args=args,
        inference_rows=inference_rows,
        variability_rows=variability_rows,
        decomposition_rows=decomposition_rows,
    )
    print(f"[written] {args.output_dir / 'classifier_seed_summary.md'}", flush=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.n_bootstrap < 10000:
        raise ValueError("--n-bootstrap must be at least 10000")

    subjects = parse_subjects(args.subjects)
    recipes = parse_recipes(args.recipes)
    classifiers = [value.lower() for value in parse_csv_list(args.classifiers)]
    unknown_classifiers = sorted(set(classifiers) - set(DEFAULT_CLASSIFIERS))
    if unknown_classifiers:
        raise ValueError(f"Unknown classifiers: {unknown_classifiers}; valid={DEFAULT_CLASSIFIERS}")
    bases = set(parse_int_list(args.bases, name="bases"))
    checkpoint_seeds = set(parse_int_list(args.checkpoint_seeds, name="checkpoint-seeds"))
    classifier_seeds = parse_int_list(args.classifier_seeds, name="classifier_seeds")
    train_seeds = parse_int_list(args.train_seeds, name="train-seeds")
    test_seeds = parse_int_list(args.test_seeds, name="test-seeds")
    if len(train_seeds) != len(test_seeds):
        raise ValueError("--train-seeds and --test-seeds must have the same length")

    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    checkpoints = expand_checkpoint_patterns(args.checkpoint_glob, bases=bases, seeds=checkpoint_seeds)
    models = load_models(checkpoints, device)
    models_by_base = sorted(models, key=lambda item: (int(item["base"]), int(item["checkpoint_seed"])))
    emg_pool = load_emg_pool(args.emg_pool)

    baseline_path = args.output_dir / "classifier_seed_baseline_rows.csv"
    result_path = args.output_dir / "classifier_seed_checkpoint_rows.csv"
    baseline_rows: list[dict[str, Any]] = list(read_csv(baseline_path)) if args.resume else []
    result_rows: list[dict[str, Any]] = list(read_csv(result_path)) if args.resume else []
    done_baselines = completed_keys(baseline_rows)
    done_results = completed_keys(result_rows)

    print(
        f"[start] run_id={args.run_id} subjects={subjects} recipes={recipes} classifiers={classifiers} "
        f"bases={sorted(bases)} checkpoint_seeds={sorted(checkpoint_seeds)} "
        f"contamination_pairs={list(zip(train_seeds, test_seeds))} classifier_seeds={classifier_seeds} "
        f"checkpoints={len(models_by_base)} device={device}",
        flush=True,
    )

    subject_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]] = {}
    for subject in subjects:
        print(f"[load_subject] {subject}", flush=True)
        subject_cache[subject] = load_subject_clean(args, subject)

    for recipe in recipes:
        for subject in subjects:
            train_clean, train_eog, test_clean, test_eog, y_train, y_test, _info = subject_cache[subject]
            for seed_pair_index, (train_seed, test_seed) in enumerate(zip(train_seeds, test_seeds), start=1):
                print(
                    f"[seed_pair_start] recipe={recipe} subject={subject} seed_pair={seed_pair_index} "
                    f"train={train_seed} test={test_seed}",
                    flush=True,
                )
                train_noisy, _train_artifact, _train_noise = make_recipe_noisy_epochs(
                    train_clean,
                    train_eog,
                    emg_pool,
                    recipe=recipe,
                    seed=train_seed,
                    snr_min_db=args.snr_min_db,
                    snr_max_db=args.snr_max_db,
                )
                test_noisy, _test_artifact, _test_noise = make_recipe_noisy_epochs(
                    test_clean,
                    test_eog,
                    emg_pool,
                    recipe=recipe,
                    seed=test_seed,
                    snr_min_db=args.snr_min_db,
                    snr_max_db=args.snr_max_db,
                )
                train_noisy_bp = bandpass_epochs(train_noisy, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz)
                test_noisy_bp = bandpass_epochs(test_noisy, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz)

                baseline_acc: dict[tuple[str, int], float] = {}
                for classifier in classifiers:
                    for classifier_seed in classifier_seeds:
                        key_row = {
                            "classifier": classifier,
                            "classifier_seed": classifier_seed,
                            "subject": subject,
                            "recipe": recipe,
                            "seed_pair_index": seed_pair_index,
                            "base": "",
                            "checkpoint_seed": "",
                            "condition": "noisy_noisy",
                        }
                        key = row_key(key_row)
                        existing = next((row for row in baseline_rows if row_key(row) == key), None)
                        if existing is not None:
                            baseline_acc[(classifier, classifier_seed)] = float(existing["accuracy"])
                            continue
                        pred, train_info = train_classifier_predictions(
                            classifier_name=classifier,
                            train_epochs=train_noisy_bp,
                            y_train=y_train,
                            test_epochs=test_noisy_bp,
                            device=device,
                            seed=classifier_seed,
                            epochs=args.epochs,
                            batch_size=args.classifier_batch_size,
                            lr=args.lr,
                            weight_decay=args.weight_decay,
                        )
                        row = metric_row(y_test, pred)
                        row.update(train_info)
                        row.update(
                            {
                                "run_id": args.run_id,
                                "classifier": classifier,
                                "classifier_seed": classifier_seed,
                                "subject": subject,
                                "recipe": recipe,
                                "seed_pair_index": seed_pair_index,
                                "train_contamination_seed": train_seed,
                                "test_contamination_seed": test_seed,
                                "condition": "noisy_noisy",
                                "base": "",
                                "checkpoint_seed": "",
                                "checkpoint": "",
                                "trainable_parameters": "",
                                "baseline_accuracy": row["accuracy"],
                                "delta_accuracy_vs_noisy_noisy": 0.0,
                            }
                        )
                        baseline_rows.append(row)
                        done_baselines.add(key)
                        baseline_acc[(classifier, classifier_seed)] = float(row["accuracy"])
                        write_csv(baseline_path, baseline_rows)
                        print(
                            f"[baseline] subject={subject} recipe={recipe} seed_pair={seed_pair_index} "
                            f"classifier={classifier} classifier_seed={classifier_seed} acc={row['accuracy']:.6f}",
                            flush=True,
                        )

                for entry in models_by_base:
                    train_den = denoise_epochs(entry["model"], train_noisy, device=device, batch_size=args.denoiser_batch_size)
                    test_den = denoise_epochs(entry["model"], test_noisy, device=device, batch_size=args.denoiser_batch_size)
                    train_den_bp = bandpass_epochs(train_den, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz)
                    test_den_bp = bandpass_epochs(test_den, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz)
                    for classifier in classifiers:
                        for classifier_seed in classifier_seeds:
                            key_row = {
                                "classifier": classifier,
                                "classifier_seed": classifier_seed,
                                "subject": subject,
                                "recipe": recipe,
                                "seed_pair_index": seed_pair_index,
                                "base": entry["base"],
                                "checkpoint_seed": entry["checkpoint_seed"],
                                "condition": "denoised_denoised",
                            }
                            key = row_key(key_row)
                            if key in done_results:
                                continue
                            pred, train_info = train_classifier_predictions(
                                classifier_name=classifier,
                                train_epochs=train_den_bp,
                                y_train=y_train,
                                test_epochs=test_den_bp,
                                device=device,
                                seed=classifier_seed,
                                epochs=args.epochs,
                                batch_size=args.classifier_batch_size,
                                lr=args.lr,
                                weight_decay=args.weight_decay,
                            )
                            row = metric_row(y_test, pred)
                            row.update(train_info)
                            baseline = baseline_acc[(classifier, classifier_seed)]
                            row.update(
                                {
                                    "run_id": args.run_id,
                                    "classifier": classifier,
                                    "classifier_seed": classifier_seed,
                                    "subject": subject,
                                    "recipe": recipe,
                                    "seed_pair_index": seed_pair_index,
                                    "train_contamination_seed": train_seed,
                                    "test_contamination_seed": test_seed,
                                    "condition": "denoised_denoised",
                                    "base": entry["base"],
                                    "checkpoint_seed": entry["checkpoint_seed"],
                                    "checkpoint": entry["checkpoint"],
                                    "trainable_parameters": entry["trainable_parameters"],
                                    "baseline_accuracy": baseline,
                                    "delta_accuracy_vs_noisy_noisy": float(row["accuracy"]) - baseline,
                                }
                            )
                            result_rows.append(row)
                            done_results.add(key)
                            write_csv(result_path, result_rows)
                            print(
                                f"[result] subject={subject} recipe={recipe} seed_pair={seed_pair_index} "
                                f"classifier={classifier} classifier_seed={classifier_seed} "
                                f"base={entry['base']} checkpoint_seed={entry['checkpoint_seed']} "
                                f"acc={row['accuracy']:.6f} delta={row['delta_accuracy_vs_noisy_noisy']:+.6f}",
                                flush=True,
                            )
                print(f"[seed_pair_done] recipe={recipe} subject={subject} seed_pair={seed_pair_index}", flush=True)

    analyze_and_write(args, baseline_rows=baseline_rows, result_rows=result_rows)
    print("[done]", flush=True)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
