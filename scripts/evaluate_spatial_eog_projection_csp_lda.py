#!/usr/bin/env python3
"""spatial-projection BCI IV-2a CSP+LDA control for spatial EOG contamination geometry."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
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

from scripts.evaluate_bci2a_contamination_seed_repeats_csp_lda import (  # noqa: E402
    bh_fdr_adjust,
    bootstrap_mean_ci,
    exact_wilcoxon_p,
    holm_adjust,
    median,
    sample_sd,
)
from scripts.evaluate_bci2a_downstream_contamination_types_csp_lda import (  # noqa: E402
    DEFAULT_SUBJECTS,
    parse_subjects,
)
from scripts.evaluate_bci2a_downstream_csp_lda import (  # noqa: E402
    FS_MODEL,
    bandpass_epochs,
    checkpoint_seed,
    denoise_epochs,
    evaluate_classifier,
    expand_checkpoints,
    fit_classifier,
    load_bci2a_trials,
    load_checkpoint_model,
)


GEOMETRIES = ["simple_equal_scaling", "train_projection"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bci-dir",
        type=Path,
        required=True, help="Path to the licensed input data.",
    )
    parser.add_argument("--subjects", default=",".join(DEFAULT_SUBJECTS))
    parser.add_argument(
        "--checkpoint-glob",
        action="append",
        required=True,
        help="External denoiser checkpoint glob. Repeat to combine checkpoint families.",
    )
    parser.add_argument("--bases", default="2,4,6,8,16")
    parser.add_argument("--checkpoint-seeds", default="42,43,44")
    parser.add_argument("--train-seeds", default="1042,1043,1044,1045,1046")
    parser.add_argument("--test-seeds", default="42,43,44,45,46")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
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
    parser.add_argument("--bootstrap-seed", type=int, default=20260816)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


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


def checkpoint_seed_from_path(path: Path) -> int | None:
    match = re.search(r"seed(\d+)", str(path))
    return int(match.group(1)) if match else None


def expand_checkpoint_patterns(patterns: list[str], *, bases: set[int], seeds: set[int]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in expand_checkpoints(pattern):
            base = checkpoint_base(path)
            seed = checkpoint_seed_from_path(path)
            if base not in bases or seed not in seeds:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(path)
    paths.sort(key=lambda path: (checkpoint_base(path) or 10**9, checkpoint_seed_from_path(path) or 10**9, str(path)))
    missing = sorted((base, seed) for base in bases for seed in seeds if not any(checkpoint_base(path) == base and checkpoint_seed_from_path(path) == seed for path in paths))
    if missing:
        raise ValueError(f"Missing requested checkpoint base/seed pairs: {missing}")
    return paths


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    preferred = [
        "run_id",
        "subject",
        "recipe",
        "contamination_geometry",
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


def estimate_eog_projection_coefficients(train_clean: np.ndarray, train_eog: np.ndarray) -> dict[str, Any]:
    eog = train_eog - np.mean(train_eog, axis=1, keepdims=True)
    denom = float(np.sum(eog * eog) + 1e-12)
    raw = []
    for channel in range(train_clean.shape[1]):
        eeg = train_clean[:, channel, :] - np.mean(train_clean[:, channel, :], axis=1, keepdims=True)
        raw.append(float(np.sum(eeg * eog) / denom))
    raw_coeff = np.asarray(raw, dtype=np.float32)
    rms = float(np.sqrt(np.mean(raw_coeff * raw_coeff)) + 1e-12)
    if not np.isfinite(rms) or rms <= 1e-10:
        normalized = np.ones_like(raw_coeff, dtype=np.float32)
        rms = 1.0
    else:
        normalized = (raw_coeff / rms).astype(np.float32)
    return {
        "raw_coefficients": raw_coeff.tolist(),
        "normalized_coefficients": normalized.tolist(),
        "raw_rms": float(rms),
        "normalized_min": float(np.min(normalized)),
        "normalized_max": float(np.max(normalized)),
        "normalized_mean": float(np.mean(normalized)),
        "normalized_sd": float(np.std(normalized)),
        "estimation_source": "training_session_only",
    }


def _center_template(template: np.ndarray) -> np.ndarray:
    centered = np.asarray(template, dtype=np.float32)
    centered = centered - float(np.mean(centered))
    return centered.astype(np.float32)


def make_simple_eog_noisy_epochs(
    clean_epochs: np.ndarray,
    eog_epochs: np.ndarray,
    *,
    seed: int,
    snr_min_db: float,
    snr_max_db: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    noisy = np.zeros_like(clean_epochs, dtype=np.float32)
    artifact = np.zeros_like(clean_epochs, dtype=np.float32)
    snrs = rng.uniform(snr_min_db, snr_max_db, size=clean_epochs.shape[0]).astype(np.float32)
    for trial_index, snr_db in enumerate(snrs):
        template = _center_template(eog_epochs[trial_index])
        p_template = float(np.mean(template * template) + 1e-12)
        for channel_index in range(clean_epochs.shape[1]):
            clean = clean_epochs[trial_index, channel_index]
            p_clean = float(np.mean(clean * clean) + 1e-12)
            scale = math.sqrt(p_clean / (p_template * (10 ** (float(snr_db) / 10.0))))
            a = (scale * template).astype(np.float32)
            artifact[trial_index, channel_index] = a
            noisy[trial_index, channel_index] = clean + a
    return noisy, artifact, {
        "contamination_geometry": "simple_equal_scaling",
        "snr_min_db": float(snr_min_db),
        "snr_max_db": float(snr_max_db),
        "snr_mean_db": float(np.mean(snrs)),
        "snr_std_db": float(np.std(snrs)),
        "scaling_rule": "same EOG template per channel, independently scaled to each channel power",
    }


def make_projected_eog_noisy_epochs(
    clean_epochs: np.ndarray,
    eog_epochs: np.ndarray,
    coefficients: np.ndarray,
    *,
    seed: int,
    snr_min_db: float,
    snr_max_db: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    noisy = np.zeros_like(clean_epochs, dtype=np.float32)
    artifact = np.zeros_like(clean_epochs, dtype=np.float32)
    snrs = rng.uniform(snr_min_db, snr_max_db, size=clean_epochs.shape[0]).astype(np.float32)
    coeff = np.asarray(coefficients, dtype=np.float32).reshape(1, -1, 1)
    for trial_index, snr_db in enumerate(snrs):
        template = _center_template(eog_epochs[trial_index]).reshape(1, -1)
        base_artifact = coeff[0] * template
        base_artifact = base_artifact - np.mean(base_artifact, axis=1, keepdims=True)
        p_clean = float(np.mean(clean_epochs[trial_index] * clean_epochs[trial_index]) + 1e-12)
        p_artifact = float(np.mean(base_artifact * base_artifact) + 1e-12)
        scale = math.sqrt(p_clean / (p_artifact * (10 ** (float(snr_db) / 10.0))))
        a = (scale * base_artifact).astype(np.float32)
        artifact[trial_index] = a
        noisy[trial_index] = clean_epochs[trial_index] + a
    return noisy, artifact, {
        "contamination_geometry": "train_projection",
        "snr_min_db": float(snr_min_db),
        "snr_max_db": float(snr_max_db),
        "snr_mean_db": float(np.mean(snrs)),
        "snr_std_db": float(np.std(snrs)),
        "scaling_rule": "training-estimated channel coefficients with per-trial global SNR scaling",
    }


def make_geometry_noisy(
    clean_epochs: np.ndarray,
    eog_epochs: np.ndarray,
    *,
    geometry: str,
    projection_coefficients: np.ndarray,
    seed: int,
    snr_min_db: float,
    snr_max_db: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if geometry == "simple_equal_scaling":
        return make_simple_eog_noisy_epochs(
            clean_epochs,
            eog_epochs,
            seed=seed,
            snr_min_db=snr_min_db,
            snr_max_db=snr_max_db,
        )
    if geometry == "train_projection":
        return make_projected_eog_noisy_epochs(
            clean_epochs,
            eog_epochs,
            projection_coefficients,
            seed=seed,
            snr_min_db=snr_min_db,
            snr_max_db=snr_max_db,
        )
    raise ValueError(f"Unknown contamination geometry: {geometry}")


def load_models(checkpoints: list[Path], device: torch.device) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for checkpoint_path in checkpoints:
        model, cfg, n_params = load_checkpoint_model(checkpoint_path, device)
        base = int(checkpoint_base(checkpoint_path, cfg) or -1)
        seed_value = int(checkpoint_seed(checkpoint_path, cfg))
        models.append(
            {
                "checkpoint": str(checkpoint_path),
                "model": model,
                "cfg": cfg,
                "base": base,
                "checkpoint_seed": seed_value,
                "trainable_parameters": int(n_params),
            }
        )
    return models


def evaluate_csp(args: argparse.Namespace, train_epochs: np.ndarray, y_train: np.ndarray, test_epochs: np.ndarray, y_test: np.ndarray, *, condition: str) -> dict[str, Any]:
    train_features = bandpass_epochs(train_epochs, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz)
    test_features = bandpass_epochs(test_epochs, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz)
    classifier = fit_classifier(args, train_features, y_train)
    return evaluate_classifier(classifier, test_features, y_test, condition=condition)


def evaluate_one_cell(
    args: argparse.Namespace,
    *,
    subject: str,
    geometry: str,
    seed_pair_index: int,
    train_seed: int,
    test_seed: int,
    subject_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]],
    projection: dict[str, Any],
    models: list[dict[str, Any]],
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_clean, train_eog, test_clean, test_eog, y_train, y_test, info = subject_data
    coefficients = np.asarray(projection["normalized_coefficients"], dtype=np.float32)
    train_noisy, _train_artifact, train_noise = make_geometry_noisy(
        train_clean,
        train_eog,
        geometry=geometry,
        projection_coefficients=coefficients,
        seed=train_seed,
        snr_min_db=args.snr_min_db,
        snr_max_db=args.snr_max_db,
    )
    test_noisy, _test_artifact, test_noise = make_geometry_noisy(
        test_clean,
        test_eog,
        geometry=geometry,
        projection_coefficients=coefficients,
        seed=test_seed,
        snr_min_db=args.snr_min_db,
        snr_max_db=args.snr_max_db,
    )
    baseline = evaluate_csp(args, train_noisy, y_train, test_noisy, y_test, condition="noisy_noisy")
    baseline.update(
        {
            "run_id": args.run_id,
            "subject": subject,
            "recipe": "eog",
            "contamination_geometry": geometry,
            "seed_pair_index": seed_pair_index,
            "train_contamination_seed": train_seed,
            "test_contamination_seed": test_seed,
            "base": "",
            "checkpoint_seed": "",
            "checkpoint": "",
            "trainable_parameters": "",
            "delta_accuracy_vs_noisy_noisy": "",
        }
    )
    result_rows: list[dict[str, Any]] = []
    baseline_acc = float(baseline["accuracy"])
    print(
        f"[baseline] subject={subject} geometry={geometry} seed_pair={seed_pair_index} acc={baseline_acc:.6f}",
        flush=True,
    )
    for entry in models:
        train_den = denoise_epochs(entry["model"], train_noisy, device=device, batch_size=args.batch_size)
        test_den = denoise_epochs(entry["model"], test_noisy, device=device, batch_size=args.batch_size)
        row = evaluate_csp(args, train_den, y_train, test_den, y_test, condition="denoised_denoised")
        row.update(
            {
                "run_id": args.run_id,
                "subject": subject,
                "recipe": "eog",
                "contamination_geometry": geometry,
                "seed_pair_index": seed_pair_index,
                "train_contamination_seed": train_seed,
                "test_contamination_seed": test_seed,
                "base": int(entry["base"]),
                "checkpoint_seed": int(entry["checkpoint_seed"]),
                "checkpoint": entry["checkpoint"],
                "trainable_parameters": int(entry["trainable_parameters"]),
                "baseline_accuracy": baseline_acc,
                "delta_accuracy_vs_noisy_noisy": float(row["accuracy"]) - baseline_acc,
            }
        )
        result_rows.append(row)
        print(
            f"[result] subject={subject} geometry={geometry} seed_pair={seed_pair_index} "
            f"base={entry['base']} checkpoint_seed={entry['checkpoint_seed']} "
            f"acc={row['accuracy']:.6f} delta={row['delta_accuracy_vs_noisy_noisy']:+.6f}",
            flush=True,
        )
    noise_info = {
        "subject": subject,
        "recipe": "eog",
        "contamination_geometry": geometry,
        "seed_pair_index": seed_pair_index,
        "train_contamination_seed": train_seed,
        "test_contamination_seed": test_seed,
        "projection_estimation_source": "A0xT_only",
        "projection": projection,
        "train_noise": train_noise,
        "test_noise": test_noise,
        "train_source": info["train"]["source"],
        "test_source": info["test"]["source"],
    }
    return [baseline], result_rows, noise_info


def subject_width_summary(
    baseline_rows: list[dict[str, Any] | dict[str, str]],
    result_rows: list[dict[str, Any] | dict[str, str]],
) -> list[dict[str, Any]]:
    baseline_group: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in baseline_rows:
        baseline_group[(str(row["subject"]), str(row["contamination_geometry"]))].append(float(row["accuracy"]))
    grouped: dict[tuple[str, str, int], list[dict[str, Any] | dict[str, str]]] = defaultdict(list)
    for row in result_rows:
        grouped[(str(row["subject"]), str(row["contamination_geometry"]), int(row["base"]))].append(row)
    rows: list[dict[str, Any]] = []
    for (subject, geometry, base), items in sorted(grouped.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
        deltas = [float(item["delta_accuracy_vs_noisy_noisy"]) for item in items]
        acc = [float(item["accuracy"]) for item in items]
        by_checkpoint: dict[str, list[float]] = defaultdict(list)
        by_contamination: dict[str, list[float]] = defaultdict(list)
        for item in items:
            by_checkpoint[str(item["checkpoint_seed"])].append(float(item["delta_accuracy_vs_noisy_noisy"]))
            by_contamination[str(item["seed_pair_index"])].append(float(item["delta_accuracy_vs_noisy_noisy"]))
        rows.append(
            {
                "subject": subject,
                "recipe": "eog",
                "contamination_geometry": geometry,
                "base": base,
                "n_contamination_seed_pairs": len(by_contamination),
                "n_checkpoint_seeds": len(by_checkpoint),
                "n_observations_averaged": len(items),
                "baseline_accuracy_mean": mean(baseline_group[(subject, geometry)]),
                "baseline_accuracy_sd_over_seed_pairs": sample_sd(baseline_group[(subject, geometry)]),
                "denoised_accuracy_mean": mean(acc),
                "delta_accuracy": mean(deltas),
                "delta_accuracy_sd_over_nuisance": sample_sd(deltas),
                "checkpoint_delta_sd_within_subject": sample_sd([mean(values) for _, values in sorted(by_checkpoint.items())]),
                "contamination_delta_sd_within_subject": sample_sd([mean(values) for _, values in sorted(by_contamination.items())]),
            }
        )
    return rows


def inference_rows(subject_rows: list[dict[str, Any]], *, n_bootstrap: int, bootstrap_seed: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in subject_rows:
        grouped[(str(row["contamination_geometry"]), int(row["base"]))].append(row)
    rows: list[dict[str, Any]] = []
    for (geometry, base), items in sorted(grouped.items()):
        deltas = [float(item["delta_accuracy"]) for item in items]
        ci_low, ci_high = bootstrap_mean_ci(deltas, n_bootstrap=n_bootstrap, seed=bootstrap_seed + 1000 * len(rows))
        rows.append(
            {
                "recipe": "eog",
                "contamination_geometry": geometry,
                "base": base,
                "n_subjects": len(items),
                "subjects": " ".join(str(item["subject"]) for item in items),
                "n_contamination_seed_pairs": int(items[0]["n_contamination_seed_pairs"]),
                "n_checkpoint_seeds": int(items[0]["n_checkpoint_seeds"]),
                "baseline_accuracy_subject_mean": mean([float(item["baseline_accuracy_mean"]) for item in items]),
                "denoised_accuracy_subject_mean": mean([float(item["denoised_accuracy_mean"]) for item in items]),
                "mean_delta_accuracy": mean(deltas),
                "median_delta_accuracy": median(deltas),
                "sd_delta_accuracy_across_subjects": sample_sd(deltas),
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
                "n_bootstrap_subject_resamples": n_bootstrap,
                "subjects_below_noisy_noisy": sum(1 for delta in deltas if delta < 0),
                "subjects_above_noisy_noisy": sum(1 for delta in deltas if delta > 0),
                "wilcoxon_p_denoised_lt_noisy": exact_wilcoxon_p(deltas, "less"),
                "wilcoxon_p_denoised_gt_noisy": exact_wilcoxon_p(deltas, "greater"),
                "wilcoxon_p_two_sided": exact_wilcoxon_p(deltas, "two-sided"),
            }
        )
    bh_fdr_adjust(rows, "wilcoxon_p_denoised_lt_noisy", "bh_fdr_q_denoised_lt_noisy")
    holm_adjust(rows, "wilcoxon_p_denoised_lt_noisy", "holm_p_denoised_lt_noisy")
    return rows


def geometry_contrast_rows(subject_rows: list[dict[str, Any]], *, n_bootstrap: int, bootstrap_seed: int) -> list[dict[str, Any]]:
    by_key = {
        (str(row["subject"]), int(row["base"]), str(row["contamination_geometry"])): row
        for row in subject_rows
    }
    rows: list[dict[str, Any]] = []
    for base in sorted({int(row["base"]) for row in subject_rows}):
        contrasts: list[float] = []
        subjects: list[str] = []
        for subject in sorted({str(row["subject"]) for row in subject_rows}):
            simple = by_key.get((subject, base, "simple_equal_scaling"))
            projected = by_key.get((subject, base, "train_projection"))
            if simple is None or projected is None:
                continue
            contrasts.append(float(projected["delta_accuracy"]) - float(simple["delta_accuracy"]))
            subjects.append(subject)
        if not contrasts:
            continue
        ci_low, ci_high = bootstrap_mean_ci(contrasts, n_bootstrap=n_bootstrap, seed=bootstrap_seed + 50000 + 1000 * len(rows))
        rows.append(
            {
                "recipe": "eog",
                "base": base,
                "contrast": "train_projection_minus_simple_equal_scaling_delta_accuracy",
                "n_subjects": len(subjects),
                "subjects": " ".join(subjects),
                "mean_contrast": mean(contrasts),
                "median_contrast": median(contrasts),
                "sd_contrast_across_subjects": sample_sd(contrasts),
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
                "wilcoxon_p_projection_gt_simple": exact_wilcoxon_p(contrasts, "greater"),
                "wilcoxon_p_projection_lt_simple": exact_wilcoxon_p(contrasts, "less"),
                "subjects_projection_less_negative": sum(1 for value in contrasts if value > 0),
                "subjects_projection_more_negative": sum(1 for value in contrasts if value < 0),
            }
        )
    bh_fdr_adjust(rows, "wilcoxon_p_projection_lt_simple", "bh_fdr_q_projection_lt_simple")
    return rows


def write_markdown(path: Path, *, args: argparse.Namespace, inference: list[dict[str, Any]], contrasts: list[dict[str, Any]]) -> None:
    lines = ["# spatial-projection Spatial EOG Projection CSP+LDA Robustness Control", ""]
    lines.append("This experiment compares the original simple EOG contamination geometry against a channel-specific EOG projection estimated from the training session only.")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append(f"- Subjects: `{args.subjects}`.")
    lines.append(f"- Bases: `{args.bases}`.")
    lines.append(f"- Checkpoint seeds: `{args.checkpoint_seeds}`.")
    lines.append(f"- Train contamination seeds: `{args.train_seeds}`.")
    lines.append(f"- Test contamination seeds: `{args.test_seeds}`.")
    lines.append("- Projection coefficients are estimated from `A0xT` only and applied to both train and held-out `A0xE` synthetic contamination.")
    lines.append("- This is a robustness control for contamination geometry, not a complete physiological ocular propagation model.")
    lines.append("")
    lines.append("## Subject-Level Effects")
    lines.append("")
    lines.append("| Geometry | Base | n subjects | Noisy acc | Denoised acc | Mean delta | 95% subject-bootstrap CI | Below noisy | Wilcoxon p lower | BH q |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in inference:
        lines.append(
            f"| {row['contamination_geometry']} | {row['base']} | {row['n_subjects']} | "
            f"{row['baseline_accuracy_subject_mean']:.6f} | {row['denoised_accuracy_subject_mean']:.6f} | "
            f"{row['mean_delta_accuracy']:+.6f} | [{row['bootstrap_ci95_low']:+.6f}, {row['bootstrap_ci95_high']:+.6f}] | "
            f"{row['subjects_below_noisy_noisy']} | {row['wilcoxon_p_denoised_lt_noisy']:.6f} | "
            f"{row['bh_fdr_q_denoised_lt_noisy']:.6f} |"
        )
    lines.append("")
    lines.append("## Geometry Contrast")
    lines.append("")
    lines.append("| Base | Mean projection-minus-simple delta | 95% subject-bootstrap CI | Projection less negative | Wilcoxon p projection<simple | BH q |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for row in contrasts:
        lines.append(
            f"| {row['base']} | {row['mean_contrast']:+.6f} | "
            f"[{row['bootstrap_ci95_low']:+.6f}, {row['bootstrap_ci95_high']:+.6f}] | "
            f"{row['subjects_projection_less_negative']}/{row['n_subjects']} | "
            f"{row['wilcoxon_p_projection_lt_simple']:.6f} | {row['bh_fdr_q_projection_lt_simple']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.n_bootstrap < 10000:
        raise ValueError("--n-bootstrap must be at least 10000")
    subjects = parse_subjects(args.subjects)
    bases = set(parse_int_list(args.bases, name="bases"))
    checkpoint_seeds = set(parse_int_list(args.checkpoint_seeds, name="checkpoint-seeds"))
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
    print(
        f"[start] run_id={args.run_id} subjects={subjects} geometries={GEOMETRIES} "
        f"bases={sorted(bases)} checkpoint_seeds={sorted(checkpoint_seeds)} "
        f"seed_pairs={list(zip(train_seeds, test_seeds))} models={len(models)} device={device}",
        flush=True,
    )

    baseline_path = args.output_dir / "spatial-projection_baseline_seed_rows.csv"
    result_path = args.output_dir / "spatial-projection_checkpoint_seed_rows.csv"
    noise_path = args.output_dir / "spatial-projection_noise_projection_metadata_rows.csv"
    baseline_rows: list[dict[str, Any] | dict[str, str]] = read_csv(baseline_path) if args.resume else []
    result_rows: list[dict[str, Any] | dict[str, str]] = read_csv(result_path) if args.resume else []
    noise_rows: list[dict[str, Any] | dict[str, str]] = read_csv(noise_path) if args.resume else []

    subject_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]] = {}
    projection_rows: list[dict[str, Any]] = []
    for subject in subjects:
        print(f"[load_subject] {subject}", flush=True)
        subject_cache[subject] = load_subject_clean(args, subject)
        projection = estimate_eog_projection_coefficients(subject_cache[subject][0], subject_cache[subject][1])
        projection_rows.append({"subject": subject, **projection})

    for subject in subjects:
        projection = next(row for row in projection_rows if row["subject"] == subject)
        for geometry in GEOMETRIES:
            for seed_pair_index, (train_seed, test_seed) in enumerate(zip(train_seeds, test_seeds), start=1):
                print(
                    f"[cell_start] subject={subject} geometry={geometry} seed_pair={seed_pair_index} train={train_seed} test={test_seed}",
                    flush=True,
                )
                base_rows, rows, noise_info = evaluate_one_cell(
                    args,
                    subject=subject,
                    geometry=geometry,
                    seed_pair_index=seed_pair_index,
                    train_seed=train_seed,
                    test_seed=test_seed,
                    subject_data=subject_cache[subject],
                    projection=projection,
                    models=models,
                    device=device,
                )
                baseline_rows.extend(base_rows)
                result_rows.extend(rows)
                noise_rows.append(noise_info)
                write_csv(baseline_path, [dict(row) for row in baseline_rows])
                write_csv(result_path, [dict(row) for row in result_rows])
                write_csv(noise_path, [dict(row) for row in noise_rows])
                print(
                    f"[cell_done] subject={subject} geometry={geometry} seed_pair={seed_pair_index} "
                    f"baseline_rows={len(baseline_rows)} result_rows={len(result_rows)}",
                    flush=True,
                )

    subject_rows = subject_width_summary(baseline_rows, result_rows)
    inference = inference_rows(subject_rows, n_bootstrap=args.n_bootstrap, bootstrap_seed=args.bootstrap_seed)
    contrasts = geometry_contrast_rows(subject_rows, n_bootstrap=args.n_bootstrap, bootstrap_seed=args.bootstrap_seed)
    write_csv(args.output_dir / "spatial-projection_projection_coefficients.csv", projection_rows)
    write_csv(args.output_dir / "spatial-projection_subject_width_summary.csv", subject_rows)
    write_csv(args.output_dir / "spatial-projection_geometry_inference.csv", inference)
    write_csv(args.output_dir / "spatial-projection_geometry_contrasts.csv", contrasts)
    summary = {
        "info": {
            "run_id": args.run_id,
            "subjects": subjects,
            "recipe": "eog",
            "geometries": GEOMETRIES,
            "bases": sorted(bases),
            "checkpoint_seeds": sorted(checkpoint_seeds),
            "train_contamination_seeds": train_seeds,
            "test_contamination_seeds": test_seeds,
            "projection_estimation": "subject-specific least-squares EOG-to-EEG coefficients estimated from A0xT only",
            "projection_application": "estimated coefficients applied to A0xE synthetic contamination with trial timing preserved via each trial EOG template",
            "robustness_scope": "contamination geometry control, not complete physiological ocular propagation model",
            "inferential_unit": "subject",
            "n_bootstrap_subject_resamples": args.n_bootstrap,
        },
        "projection_coefficients": projection_rows,
        "geometry_inference": inference,
        "geometry_contrasts": contrasts,
    }
    (args.output_dir / "spatial-projection_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.output_dir / "spatial-projection_summary.md", args=args, inference=inference, contrasts=contrasts)
    print(f"[written] {args.output_dir / 'spatial-projection_summary.md'}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
