#!/usr/bin/env python3
"""real-recording real-recording BCI IV-2a downstream analysis stratified by EOG energy."""

from __future__ import annotations

import argparse
import csv
import glob
import itertools
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.evaluate_bci2a_downstream_csp_lda import (  # noqa: E402
    CLASS_NAMES,
    FS_MODEL,
    METRIC_KEYS,
    bandpass_epochs,
    checkpoint_seed,
    denoise_epochs,
    expand_checkpoints,
    fit_classifier,
    load_bci2a_trials,
    load_checkpoint_model,
)


DEFAULT_SUBJECTS = [f"A{index:02d}" for index in range(1, 10)]
STRATA = ("all", "quiet_low_eog", "artifact_heavy_high_eog")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bci-dir", type=Path, required=True, help="Path to the licensed input data.")
    parser.add_argument("--subjects", default=",".join(DEFAULT_SUBJECTS))
    parser.add_argument("--checkpoint-glob", required=True)
    parser.add_argument("--bases", default="2,4,6,8,16")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--trial-start-sec", type=float, default=2.5)
    parser.add_argument("--trial-stop-sec", type=float, default=6.0)
    parser.add_argument("--bandpass-low-hz", type=float, default=8.0)
    parser.add_argument("--bandpass-high-hz", type=float, default=30.0)
    parser.add_argument("--csp-components", type=int, default=8)
    parser.add_argument("--eog-index", type=int, default=22)
    parser.add_argument("--eog-energy-low-quantile", type=float, default=1.0 / 3.0)
    parser.add_argument("--eog-energy-high-quantile", type=float, default=2.0 / 3.0)
    parser.add_argument("--eog-energy-band-low-hz", type=float, default=0.5)
    parser.add_argument("--eog-energy-band-high-hz", type=float, default=10.0)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260816)
    parser.add_argument(
        "--include-artifact-trials",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include BCI IV-2a artifact-flagged trials. Default true because this is a real-recording downstream check.",
    )
    return parser.parse_args()


def parse_subjects(raw: str) -> list[str]:
    subjects = [part.strip().upper() for part in raw.replace(",", " ").split() if part.strip()]
    if not subjects:
        raise ValueError("At least one subject is required.")
    return subjects


def parse_bases(raw: str) -> set[int]:
    bases = {int(part) for part in raw.replace(",", " ").split() if part.strip()}
    if not bases:
        raise ValueError("At least one base width is required.")
    return bases


def checkpoint_base(path: Path) -> int | None:
    match = re.search(r"base(\d+)", str(path))
    return int(match.group(1)) if match else None


def filtered_checkpoints(pattern: str, bases: set[int]) -> list[Path]:
    checkpoints = [path for path in expand_checkpoints(pattern) if checkpoint_base(path) in bases]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints matched requested bases {sorted(bases)} from {pattern}")
    return checkpoints


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.stdev(values))


def bootstrap_ci(values: list[float], *, n_resamples: int, rng: np.random.Generator) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1 or n_resamples <= 0:
        return float(arr[0]), float(arr[0])
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


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
    indexed = []
    for index, row in enumerate(rows):
        value = row.get(p_key, "")
        if value == "":
            continue
        indexed.append((float(value), index))
    indexed.sort()
    m = len(indexed)
    prev = 1.0
    adjusted: dict[int, float] = {}
    for rank_from_end, (p_value, index) in enumerate(reversed(indexed), start=1):
        rank = m - rank_from_end + 1
        q_value = min(prev, p_value * m / rank)
        prev = q_value
        adjusted[index] = min(1.0, q_value)
    for index, row in enumerate(rows):
        row[out_key] = adjusted.get(index, "")


def eog_energy(eog_epochs: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    shaped = eog_epochs[:, None, :].astype(np.float32)
    filtered = bandpass_epochs(shaped, FS_MODEL, args.eog_energy_band_low_hz, args.eog_energy_band_high_hz)[:, 0, :]
    return np.mean(np.square(filtered), axis=1).astype(np.float64)


def stratum_masks(test_energy: np.ndarray, low_threshold: float, high_threshold: float) -> dict[str, np.ndarray]:
    return {
        "all": np.ones(test_energy.shape[0], dtype=bool),
        "quiet_low_eog": test_energy <= low_threshold,
        "artifact_heavy_high_eog": test_energy >= high_threshold,
    }


def score_predictions(labels: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

    cm = confusion_matrix(labels, prediction, labels=list(range(len(CLASS_NAMES))))
    return {
        "n_trials": int(len(labels)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "cohen_kappa": float(cohen_kappa_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro")),
        "confusion_matrix": json.dumps(cm.tolist()),
    }


def evaluate_by_stratum(
    classifier: Any,
    features: np.ndarray,
    labels: np.ndarray,
    masks: dict[str, np.ndarray],
    *,
    condition: str,
    subject: str,
    variant: str = "",
    base: int | str = "",
    train_seed: int | str = "",
    trainable_parameters: int | str = "",
    checkpoint: str = "",
) -> list[dict[str, Any]]:
    prediction = classifier.predict(features)
    rows: list[dict[str, Any]] = []
    for stratum, mask in masks.items():
        if int(mask.sum()) == 0:
            continue
        row = {
            "subject": subject,
            "condition": condition,
            "stratum": stratum,
            "variant": variant,
            "base": base,
            "train_seed": train_seed,
            "trainable_parameters": trainable_parameters,
            "checkpoint": checkpoint,
        }
        row.update(score_predictions(labels[mask], prediction[mask]))
        rows.append(row)
    return rows


def class_counts(labels: np.ndarray) -> str:
    counts = {CLASS_NAMES[index]: int(np.sum(labels == index)) for index in range(len(CLASS_NAMES))}
    return json.dumps(counts, sort_keys=True)


def evaluate_subject(
    args: argparse.Namespace,
    subject: str,
    checkpoints: list[Path],
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_mat = args.bci_dir / f"{subject}T.mat"
    test_mat = args.bci_dir / f"{subject}E.mat"
    train_raw, train_eog, y_train, train_info = load_bci2a_trials(
        train_mat,
        trial_start_sec=args.trial_start_sec,
        trial_stop_sec=args.trial_stop_sec,
        include_artifact_trials=args.include_artifact_trials,
        eog_index=args.eog_index,
    )
    test_raw, test_eog, y_test, test_info = load_bci2a_trials(
        test_mat,
        trial_start_sec=args.trial_start_sec,
        trial_stop_sec=args.trial_stop_sec,
        include_artifact_trials=args.include_artifact_trials,
        eog_index=args.eog_index,
    )

    train_energy = eog_energy(train_eog, args)
    test_energy = eog_energy(test_eog, args)
    low_threshold = float(np.quantile(train_energy, args.eog_energy_low_quantile))
    high_threshold = float(np.quantile(train_energy, args.eog_energy_high_quantile))
    masks = stratum_masks(test_energy, low_threshold, high_threshold)
    stratum_rows = []
    for stratum, mask in masks.items():
        stratum_rows.append(
            {
                "subject": subject,
                "stratum": stratum,
                "n_test_trials": int(mask.sum()),
                "test_class_counts": class_counts(y_test[mask]),
                "train_eog_energy_low_threshold": low_threshold,
                "train_eog_energy_high_threshold": high_threshold,
                "test_eog_energy_mean": float(np.mean(test_energy[mask])) if int(mask.sum()) else "",
                "test_eog_energy_median": float(np.median(test_energy[mask])) if int(mask.sum()) else "",
                "eog_energy_low_quantile": float(args.eog_energy_low_quantile),
                "eog_energy_high_quantile": float(args.eog_energy_high_quantile),
                "eog_energy_band_hz": f"{args.eog_energy_band_low_hz:g}-{args.eog_energy_band_high_hz:g}",
            }
        )

    train_features = bandpass_epochs(train_raw, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz)
    test_features = bandpass_epochs(test_raw, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz)
    raw_classifier = fit_classifier(args, train_features, y_train)

    rows = evaluate_by_stratum(raw_classifier, test_features, y_test, masks, condition="raw_raw", subject=subject)
    raw_all = next(row for row in rows if row["stratum"] == "all")
    print(
        f"[baseline] subject={subject} raw_raw_all={raw_all['accuracy']:.6f} "
        f"quiet_n={int(masks['quiet_low_eog'].sum())} heavy_n={int(masks['artifact_heavy_high_eog'].sum())}",
        flush=True,
    )

    for checkpoint_path in checkpoints:
        model, cfg, n_params = load_checkpoint_model(checkpoint_path, device)
        base = int(cfg.get("base", checkpoint_base(checkpoint_path) or -1))
        variant = str(cfg.get("variant") or cfg.get("model_type") or f"base{base}")
        train_seed = checkpoint_seed(checkpoint_path, cfg)
        print(f"[eval_start] subject={subject} base={base} seed={train_seed}", flush=True)

        train_denoised = denoise_epochs(model, train_raw, device=device, batch_size=args.batch_size)
        test_denoised = denoise_epochs(model, test_raw, device=device, batch_size=args.batch_size)
        train_denoised_features = bandpass_epochs(train_denoised, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz)
        test_denoised_features = bandpass_epochs(test_denoised, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz)
        denoised_classifier = fit_classifier(args, train_denoised_features, y_train)

        rows.extend(
            evaluate_by_stratum(
                raw_classifier,
                test_denoised_features,
                y_test,
                masks,
                condition="raw_denoised",
                subject=subject,
                variant=variant,
                base=base,
                train_seed=train_seed,
                trainable_parameters=n_params,
                checkpoint=str(checkpoint_path),
            )
        )
        rows.extend(
            evaluate_by_stratum(
                denoised_classifier,
                test_denoised_features,
                y_test,
                masks,
                condition="denoised_denoised",
                subject=subject,
                variant=variant,
                base=base,
                train_seed=train_seed,
                trainable_parameters=n_params,
                checkpoint=str(checkpoint_path),
            )
        )
        matched_all = [
            row
            for row in rows
            if row["condition"] == "denoised_denoised"
            and str(row["base"]) == str(base)
            and str(row["train_seed"]) == str(train_seed)
            and row["stratum"] == "all"
        ][-1]
        print(f"[result] subject={subject} base={base} seed={train_seed} denoised_all={matched_all['accuracy']:.6f}", flush=True)

    subject_info = {
        "subject": subject,
        "train": train_info,
        "test": test_info,
        "train_eog_energy_low_threshold": low_threshold,
        "train_eog_energy_high_threshold": high_threshold,
    }
    return rows, aggregate_subject(rows), stratum_rows, subject_info


def aggregate_subject(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_lookup = {
        str(row["stratum"]): row
        for row in rows
        if row["condition"] == "raw_raw"
    }
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["condition"] not in {"raw_denoised", "denoised_denoised"}:
            continue
        grouped[(str(row["condition"]), str(row["stratum"]), str(row["variant"]), str(row["base"]))].append(row)

    out: list[dict[str, Any]] = []
    for (condition, stratum, variant, base), items in sorted(grouped.items()):
        raw_row = raw_lookup[stratum]
        entry: dict[str, Any] = {
            "subject": str(items[0]["subject"]),
            "condition": condition,
            "stratum": stratum,
            "variant": variant,
            "base": base,
            "n_checkpoints": len(items),
            "train_seeds": " ".join(str(item["train_seed"]) for item in sorted(items, key=lambda r: str(r["train_seed"]))),
            "trainable_parameters": int(items[0]["trainable_parameters"]),
            "raw_raw_accuracy": float(raw_row["accuracy"]),
            "n_trials": int(raw_row["n_trials"]),
        }
        for key in METRIC_KEYS:
            vals = [float(item[key]) for item in items]
            m, sd = mean_std(vals)
            entry[f"{key}_mean"] = m
            entry[f"{key}_std"] = sd
        entry["delta_accuracy_vs_raw_raw"] = float(entry["accuracy_mean"]) - float(raw_row["accuracy"])
        out.append(entry)
    return out


def aggregate_inference(subject_aggregates: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in subject_aggregates:
        grouped[(str(row["condition"]), str(row["stratum"]), str(row["variant"]), str(row["base"]))].append(row)

    rng = np.random.default_rng(args.bootstrap_seed)
    inference: list[dict[str, Any]] = []
    for (condition, stratum, variant, base), items in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][3], kv[0][1])):
        deltas = [float(item["delta_accuracy_vs_raw_raw"]) for item in items]
        acc = [float(item["accuracy_mean"]) for item in items]
        raw_acc = [float(item["raw_raw_accuracy"]) for item in items]
        mean_delta, sd_delta = mean_std(deltas)
        lo, hi = bootstrap_ci(deltas, n_resamples=args.n_bootstrap, rng=rng)
        inference.append(
            {
                "test_family": "real_recording_downstream_delta",
                "condition": condition,
                "stratum": stratum,
                "variant": variant,
                "base": base,
                "n_subjects": len(items),
                "subjects": " ".join(sorted(str(item["subject"]) for item in items)),
                "n_checkpoints_per_subject": int(items[0]["n_checkpoints"]),
                "trainable_parameters": int(items[0]["trainable_parameters"]),
                "raw_raw_accuracy_subject_mean": float(statistics.mean(raw_acc)),
                "denoised_accuracy_subject_mean": float(statistics.mean(acc)),
                "delta_accuracy_mean": mean_delta,
                "delta_accuracy_median": float(np.median(deltas)),
                "delta_accuracy_std": sd_delta,
                "delta_accuracy_ci95_low_subject_bootstrap": lo,
                "delta_accuracy_ci95_high_subject_bootstrap": hi,
                "subjects_below_raw_raw": sum(1 for value in deltas if value < 0),
                "subjects_above_raw_raw": sum(1 for value in deltas if value > 0),
                "wilcoxon_p_denoised_lt_raw": exact_wilcoxon_p(deltas, "less"),
            }
        )

    primary = [row for row in inference if row["condition"] == "denoised_denoised"]
    bh_fdr(primary, "wilcoxon_p_denoised_lt_raw", "bh_fdr_q_denoised_lt_raw")

    contrasts: list[dict[str, Any]] = []
    by_base_condition: dict[tuple[str, str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in subject_aggregates:
        if row["stratum"] in {"quiet_low_eog", "artifact_heavy_high_eog"}:
            by_base_condition[(str(row["condition"]), str(row["variant"]), str(row["base"]))][str(row["stratum"])].append(row)
    for (condition, variant, base), strata in sorted(by_base_condition.items(), key=lambda kv: (kv[0][0], kv[0][2])):
        quiet = {str(row["subject"]): row for row in strata.get("quiet_low_eog", [])}
        heavy = {str(row["subject"]): row for row in strata.get("artifact_heavy_high_eog", [])}
        subjects = sorted(set(quiet) & set(heavy))
        if not subjects:
            continue
        values = [
            float(quiet[subject]["delta_accuracy_vs_raw_raw"]) - float(heavy[subject]["delta_accuracy_vs_raw_raw"])
            for subject in subjects
        ]
        mean_value, sd_value = mean_std(values)
        lo, hi = bootstrap_ci(values, n_resamples=args.n_bootstrap, rng=rng)
        contrasts.append(
            {
                "test_family": "quiet_minus_heavy_delta_concentration",
                "condition": condition,
                "variant": variant,
                "base": base,
                "n_subjects": len(subjects),
                "subjects": " ".join(subjects),
                "contrast": "quiet_low_eog_delta_minus_artifact_heavy_high_eog_delta",
                "mean_contrast": mean_value,
                "median_contrast": float(np.median(values)),
                "std_contrast": sd_value,
                "ci95_low_subject_bootstrap": lo,
                "ci95_high_subject_bootstrap": hi,
                "subjects_quiet_more_negative": sum(1 for value in values if value < 0),
                "subjects_heavy_more_negative": sum(1 for value in values if value > 0),
                "wilcoxon_p_quiet_more_negative": exact_wilcoxon_p(values, "less"),
            }
        )
    primary_contrasts = [row for row in contrasts if row["condition"] == "denoised_denoised"]
    bh_fdr(primary_contrasts, "wilcoxon_p_quiet_more_negative", "bh_fdr_q_quiet_more_negative")
    return inference, contrasts


def write_figures(output_dir: Path, subject_aggregates: list[dict[str, Any]], inference: list[dict[str, Any]]) -> list[str]:
    paths: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        (output_dir / "real-recording_figure_error.txt").write_text(f"matplotlib unavailable: {exc}\n", encoding="utf-8")
        return paths

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    base_values = sorted({int(row["base"]) for row in subject_aggregates if row["condition"] == "denoised_denoised"})
    colors = {"all": "#111827", "quiet_low_eog": "#2563eb", "artifact_heavy_high_eog": "#dc2626"}
    for base in base_values:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True, constrained_layout=True)
        for ax, stratum in zip(axes, STRATA):
            rows = [
                row
                for row in subject_aggregates
                if row["condition"] == "denoised_denoised" and int(row["base"]) == base and row["stratum"] == stratum
            ]
            rows = sorted(rows, key=lambda row: str(row["subject"]))
            for index, row in enumerate(rows):
                raw = float(row["raw_raw_accuracy"])
                den = float(row["accuracy_mean"])
                ax.plot([0, 1], [raw, den], color="#9ca3af", linewidth=1.0, alpha=0.8)
                ax.scatter([0, 1], [raw, den], color=[colors[stratum], colors[stratum]], s=18, zorder=3)
                ax.text(1.03, den, str(row["subject"]), fontsize=7, va="center")
            ax.set_xticks([0, 1], ["raw/raw", "denoised/denoised"])
            ax.set_title(stratum.replace("_", " "))
            ax.set_xlim(-0.15, 1.35)
            ax.set_ylim(0.0, 1.0)
            ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        axes[0].set_ylabel("A0xE accuracy")
        fig.suptitle(f"Real IV-2a subject-paired downstream accuracy, base{base}")
        path = fig_dir / f"real-recording_subject_pairs_base{base}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(str(path))

    den_inf = [row for row in inference if row["condition"] == "denoised_denoised"]
    if den_inf:
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        offsets = {"all": -0.18, "quiet_low_eog": 0.0, "artifact_heavy_high_eog": 0.18}
        for row in den_inf:
            base = int(row["base"])
            stratum = str(row["stratum"])
            x = base + offsets[stratum]
            mean = float(row["delta_accuracy_mean"])
            lo = float(row["delta_accuracy_ci95_low_subject_bootstrap"])
            hi = float(row["delta_accuracy_ci95_high_subject_bootstrap"])
            ax.errorbar(
                [x],
                [mean],
                yerr=[[mean - lo], [hi - mean]],
                fmt="o",
                color=colors[stratum],
                capsize=3,
                label=stratum.replace("_", " "),
            )
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.axhline(0, color="#111827", linewidth=0.9)
        ax.set_xticks(base_values, [f"base{base}" for base in base_values])
        ax.set_ylabel("Delta accuracy vs raw/raw")
        ax.set_title("Subject-level mean deltas with subject-bootstrap 95% CI")
        ax.legend(unique.values(), unique.keys(), frameon=False, fontsize=8)
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        path = fig_dir / "real-recording_delta_by_eog_stratum.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(str(path))
    return paths


def write_markdown(
    path: Path,
    args: argparse.Namespace,
    inference: list[dict[str, Any]],
    contrasts: list[dict[str, Any]],
    stratum_rows: list[dict[str, Any]],
    figure_paths: list[str],
) -> None:
    den_inf = [row for row in inference if row["condition"] == "denoised_denoised"]
    den_contrasts = [row for row in contrasts if row["condition"] == "denoised_denoised"]
    lines = [
        "# real-recording Real-Recording BCI IV-2a Downstream Analysis",
        "",
        "## Protocol",
        "",
        "- Denoisers are applied directly to naturally recorded BCI IV-2a A0xT/A0xE trials.",
        "- CSP+LDA is trained and evaluated as raw/raw or denoised/denoised.",
        "- This is a downstream utility experiment only; it is not a reconstruction experiment because no artifact-free clean target exists for these real recordings.",
        "- The analysis cannot determine precisely which removed components were neural versus artifactual.",
        f"- Artifact-labeled trials included: `{args.include_artifact_trials}`.",
        f"- EOG-energy criterion: mean squared `{args.eog_energy_band_low_hz:g}-{args.eog_energy_band_high_hz:g}` Hz EOG signal over the trial window.",
        f"- Quiet threshold: A0xT subject-specific quantile `{args.eog_energy_low_quantile:.3f}`; artifact-heavy threshold: A0xT subject-specific quantile `{args.eog_energy_high_quantile:.3f}`.",
        "- The A0xT thresholds are applied unchanged to A0xE, preserving the official train/evaluation separation.",
        "",
        "## Subject-Level Deltas",
        "",
        "| base | stratum | raw/raw acc | denoised acc | mean delta | median delta | 95% CI | below raw | Wilcoxon p | BH q |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in den_inf:
        lines.append(
            f"| {row['base']} | {row['stratum']} | {float(row['raw_raw_accuracy_subject_mean']):.6f} | "
            f"{float(row['denoised_accuracy_subject_mean']):.6f} | {float(row['delta_accuracy_mean']):+.6f} | "
            f"{float(row['delta_accuracy_median']):+.6f} | "
            f"[{float(row['delta_accuracy_ci95_low_subject_bootstrap']):+.6f}, {float(row['delta_accuracy_ci95_high_subject_bootstrap']):+.6f}] | "
            f"{row['subjects_below_raw_raw']}/{row['n_subjects']} | "
            f"{float(row['wilcoxon_p_denoised_lt_raw']):.6f} | {row.get('bh_fdr_q_denoised_lt_raw', '')} |"
        )
    lines.extend(
        [
            "",
            "## Quiet-versus-heavy concentration test",
            "",
            "Negative contrast means the denoising penalty is more negative in quiet/low-EOG trials than artifact-heavy/high-EOG trials.",
            "",
            "| base | mean quiet-heavy contrast | median | 95% CI | quiet more negative | Wilcoxon p | BH q |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in den_contrasts:
        lines.append(
            f"| {row['base']} | {float(row['mean_contrast']):+.6f} | {float(row['median_contrast']):+.6f} | "
            f"[{float(row['ci95_low_subject_bootstrap']):+.6f}, {float(row['ci95_high_subject_bootstrap']):+.6f}] | "
            f"{row['subjects_quiet_more_negative']}/{row['n_subjects']} | "
            f"{float(row['wilcoxon_p_quiet_more_negative']):.6f} | {row.get('bh_fdr_q_quiet_more_negative', '')} |"
        )
    lines.extend(["", "## A0xE stratum sizes", ""])
    lines.extend(["| subject | stratum | n test trials | EOG energy median |", "|---|---|---:|---:|"])
    for row in sorted(stratum_rows, key=lambda item: (str(item["subject"]), str(item["stratum"]))):
        lines.append(
            f"| {row['subject']} | {row['stratum']} | {row['n_test_trials']} | "
            f"{float(row['test_eog_energy_median']):.6g} |"
        )
    lines.extend(["", "## Figures", ""])
    for fig in figure_paths:
        lines.append(f"- `{fig}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not (0.0 < args.eog_energy_low_quantile < args.eog_energy_high_quantile < 1.0):
        raise ValueError("Expected 0 < low quantile < high quantile < 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    subjects = parse_subjects(args.subjects)
    bases = parse_bases(args.bases)
    checkpoints = filtered_checkpoints(args.checkpoint_glob, bases)
    print(
        f"[start] run_id={args.run_id} subjects={subjects} bases={sorted(bases)} "
        f"checkpoints={len(checkpoints)} device={device}",
        flush=True,
    )

    all_rows: list[dict[str, Any]] = []
    all_subject_aggregates: list[dict[str, Any]] = []
    all_stratum_rows: list[dict[str, Any]] = []
    infos: list[dict[str, Any]] = []
    for subject in subjects:
        rows, aggregates, stratum_rows, info = evaluate_subject(args, subject, checkpoints, device)
        all_rows.extend(rows)
        all_subject_aggregates.extend(aggregates)
        all_stratum_rows.extend(stratum_rows)
        infos.append(info)

    inference, contrasts = aggregate_inference(all_subject_aggregates, args)
    figure_paths = write_figures(args.output_dir, all_subject_aggregates, inference)

    write_csv(args.output_dir / "real-recording_real_downstream_rows.csv", all_rows)
    write_csv(args.output_dir / "real-recording_real_downstream_subject_aggregates.csv", all_subject_aggregates)
    write_csv(args.output_dir / "real-recording_eog_stratum_rows.csv", all_stratum_rows)
    write_csv(args.output_dir / "real-recording_real_downstream_inference.csv", inference)
    write_csv(args.output_dir / "real-recording_quiet_heavy_contrasts.csv", contrasts)
    summary = {
        "info": {
            "run_id": args.run_id,
            "subjects": subjects,
            "checkpoint_glob": args.checkpoint_glob,
            "bases": sorted(bases),
            "include_artifact_trials": bool(args.include_artifact_trials),
            "trial_start_sec": float(args.trial_start_sec),
            "trial_stop_sec": float(args.trial_stop_sec),
            "classifier_bandpass_hz": [float(args.bandpass_low_hz), float(args.bandpass_high_hz)],
            "eog_energy_band_hz": [float(args.eog_energy_band_low_hz), float(args.eog_energy_band_high_hz)],
            "eog_energy_quantiles": [float(args.eog_energy_low_quantile), float(args.eog_energy_high_quantile)],
            "n_bootstrap": int(args.n_bootstrap),
            "guardrails": [
                "real-recording downstream utility only",
                "no artifact-free clean target; do not frame as reconstruction",
                "cannot determine precisely which removed components were neural versus artifactual",
            ],
        },
        "subject_info": infos,
        "inference": inference,
        "quiet_heavy_contrasts": contrasts,
        "figures": figure_paths,
    }
    (args.output_dir / "real-recording_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.output_dir / "real-recording_summary.md", args, inference, contrasts, all_stratum_rows, figure_paths)
    print(f"[written] {args.output_dir / 'real-recording_summary.md'}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
