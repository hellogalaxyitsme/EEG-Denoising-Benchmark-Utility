#!/usr/bin/env python3
"""Sleep-EDF downstream denoising pilot.

This helper provides Sleep-EDF epoch loading and downstream feature functions.
It compares raw/raw, noisy/noisy, and denoised/denoised sleep-stage decoding on
Sleep-EDF Sleep Cassette subjects using night-to-night train/test splits.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import random
import re
import statistics
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.signal import resample_poly, welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from eeg_denoise_benchmark.checkpoints import load_model_from_checkpoint  # noqa: E402
from eeg_denoise_benchmark.models import count_trainable_parameters  # noqa: E402
from scripts.evaluate_bci2a_downstream_csp_lda import (  # noqa: E402
    FS_MODEL,
    denoise_epochs,
    expand_checkpoints,
)


STAGE_TO_LABEL = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
LABEL_TO_STAGE = {value: key for key, value in STAGE_TO_LABEL.items()}
FEATURE_BANDS = [
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 12.0),
    ("sigma", 12.0, 16.0),
    ("beta", 16.0, 30.0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sleep-edf-dir",
        type=Path,
        required=True,
        help="Path to the licensed Sleep-EDF sleep-cassette directory.",
    )
    parser.add_argument("--subjects", default="auto5", help="Comma list like SC400,SC401 or autoN.")
    parser.add_argument("--checkpoint-glob", action="append", default=[])
    parser.add_argument(
        "--model-spec",
        action="append",
        default=[],
        help="Optional labelled checkpoint glob, e.g. base4=runs/...base4.../best.pt. Repeatable. If set, supersedes --bases/--checkpoint-seeds filtering.",
    )
    parser.add_argument("--bases", default="6,16")
    parser.add_argument("--checkpoint-seeds", default="42,43,44")
    parser.add_argument("--train-seeds", default="1042,1043,1044")
    parser.add_argument("--test-seeds", default="42,43,44")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--snr-min-db", type=float, default=-6.0)
    parser.add_argument("--snr-max-db", type=float, default=2.0)
    parser.add_argument("--trim-wake-min", type=float, default=30.0)
    parser.add_argument("--max-train-epochs", type=int, default=1600)
    parser.add_argument("--max-test-epochs", type=int, default=1600)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260816)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.replace(" ", ",").split(",") if part.strip()]


def checkpoint_base(path: Path) -> int | None:
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
            paths.append(path)
            seen.add(resolved)
    paths.sort(key=lambda path: (checkpoint_base(path) or 10**9, checkpoint_seed_from_path(path) or 10**9, str(path)))
    missing = sorted((base, seed) for base in bases for seed in seeds if not any(checkpoint_base(path) == base and checkpoint_seed_from_path(path) == seed for path in paths))
    if missing:
        raise ValueError(f"Missing checkpoints for base/seed pairs: {missing}")
    return paths


def infer_model_label(path: Path) -> str:
    text = str(path).lower()
    base = checkpoint_base(path)
    if base is not None:
        return f"base{base}"
    if "eegdn_cnn" in text or "complex_cnn" in text:
        return "EEGDN_CNN"
    if "eegdn_rnn" in text or "rnn_lstm" in text or "lstm" in text:
        return "EEGDN_RNN_LSTM"
    if "microwavenet" in text:
        return "MicroWaveNet"
    return path.parent.name


def expand_model_specs(specs: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for raw in specs:
        if "=" in raw:
            label, pattern = raw.split("=", 1)
            label = label.strip()
        else:
            pattern = raw
            label = ""
        paths = sorted((Path(path) for path in glob.glob(pattern)), key=lambda path: (infer_model_label(path), checkpoint_seed_from_path(path) or 10**9, str(path)))
        if not paths:
            raise FileNotFoundError(f"No checkpoints matched model spec: {raw}")
        for path in paths:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            out.append({"label": label or infer_model_label(path), "path": path})
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    preferred = [
        "run_id",
        "dataset",
        "subject",
        "train_recording",
        "test_recording",
        "condition",
        "denoiser_label",
        "base",
        "checkpoint_seed",
        "checkpoint",
        "trainable_parameters",
        "train_contamination_seed",
        "test_contamination_seed",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "cohen_kappa",
        "delta_accuracy_vs_noisy_noisy",
        "delta_balanced_accuracy_vs_noisy_noisy",
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


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def stage_from_description(description: str) -> str | None:
    text = description.strip().lower()
    if "sleep stage w" in text or text in {"w", "wake"}:
        return "W"
    if "sleep stage 1" in text or text in {"1", "n1"}:
        return "N1"
    if "sleep stage 2" in text or text in {"2", "n2"}:
        return "N2"
    if "sleep stage 3" in text or "sleep stage 4" in text or text in {"3", "4", "n3", "n4"}:
        return "N3"
    if "sleep stage r" in text or text in {"r", "rem"}:
        return "REM"
    return None


def normalize_channel(name: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", name.upper())


def choose_channels(ch_names: list[str]) -> tuple[list[int], int]:
    normalized = [normalize_channel(name) for name in ch_names]
    eeg_targets = ["EEGFPZCZ", "EEGPZOZ", "FPZCZ", "PZOZ"]
    eeg_indices: list[int] = []
    for target in eeg_targets:
        if target in normalized:
            idx = normalized.index(target)
            if idx not in eeg_indices:
                eeg_indices.append(idx)
    if not eeg_indices:
        eeg_indices = [idx for idx, name in enumerate(normalized) if name.startswith("EEG")][:1]
    eog_candidates = [idx for idx, name in enumerate(normalized) if "EOG" in name]
    if not eeg_indices or not eog_candidates:
        raise ValueError(f"Could not select EEG/EOG channels from {ch_names}")
    return eeg_indices[:2], eog_candidates[0]


def resample_to_model_fs(x: np.ndarray, fs_in: float) -> np.ndarray:
    fs_in_i = int(round(fs_in))
    if fs_in_i == FS_MODEL:
        return x.astype(np.float32)
    gcd = math.gcd(FS_MODEL, fs_in_i)
    return resample_poly(x, FS_MODEL // gcd, fs_in_i // gcd, axis=-1).astype(np.float32)


def discover_recordings(root: Path) -> dict[str, list[dict[str, Path | str]]]:
    psg_paths = sorted(root.glob("*-PSG.edf"))
    hyp_paths = {path.name[:6]: path for path in root.glob("*Hypnogram.edf")}
    grouped: dict[str, list[dict[str, Path | str]]] = defaultdict(list)
    for psg in psg_paths:
        rec_id = psg.name[:6]
        subject = psg.name[:5]
        hyp = hyp_paths.get(rec_id)
        if hyp is None:
            continue
        grouped[subject].append({"subject": subject, "recording": rec_id, "psg": psg, "hyp": hyp})
    return {subject: sorted(items, key=lambda item: str(item["recording"])) for subject, items in grouped.items() if len(items) >= 2}


def select_subjects(raw: str, grouped: dict[str, list[dict[str, Path | str]]]) -> list[str]:
    available = sorted(grouped)
    if raw.lower().startswith("auto"):
        n = int(raw[4:] or "5")
        return available[:n]
    subjects = [part.strip().upper() for part in raw.replace(" ", ",").split(",") if part.strip()]
    missing = sorted(set(subjects) - set(available))
    if missing:
        raise ValueError(f"Missing Sleep-EDF subjects with two paired recordings: {missing}")
    return subjects


def load_recording(
    psg_path: Path,
    hyp_path: Path,
    *,
    trim_wake_min: float,
    max_epochs: int,
    rng_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    try:
        import mne
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("mne is required for Sleep-EDF evaluation") from exc

    mne.set_log_level("WARNING")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = mne.io.read_raw_edf(str(psg_path), preload=True, verbose=False)
        annotations = mne.read_annotations(str(hyp_path))
    fs = float(raw.info["sfreq"])
    data = np.nan_to_num(raw.get_data() * 1e6).astype(np.float32)
    eeg_idx, eog_idx = choose_channels(raw.ch_names)

    epoch_rows: list[tuple[float, int]] = []
    for onset, duration, description in zip(annotations.onset, annotations.duration, annotations.description):
        stage = stage_from_description(str(description))
        if stage is None:
            continue
        label = STAGE_TO_LABEL[stage]
        n_epochs = int(math.floor(float(duration) / 30.0))
        for offset in range(n_epochs):
            epoch_rows.append((float(onset) + 30.0 * offset, label))
    if not epoch_rows:
        raise ValueError(f"No sleep-stage epochs found for {psg_path}")

    sleep_onsets = [onset for onset, label in epoch_rows if LABEL_TO_STAGE[label] != "W"]
    if sleep_onsets and trim_wake_min >= 0:
        first_sleep = min(sleep_onsets)
        last_sleep = max(sleep_onsets)
        lo = max(0.0, first_sleep - trim_wake_min * 60.0)
        hi = last_sleep + (trim_wake_min + 0.5) * 60.0
        epoch_rows = [(onset, label) for onset, label in epoch_rows if lo <= onset <= hi]

    eeg_epochs: list[np.ndarray] = []
    eog_epochs: list[np.ndarray] = []
    labels: list[int] = []
    n_samples_30s = int(round(30.0 * fs))
    for onset, label in epoch_rows:
        start = int(round(onset * fs))
        stop = start + n_samples_30s
        if start < 0 or stop > data.shape[1]:
            continue
        eeg = data[eeg_idx, start:stop]
        eog = data[eog_idx, start:stop]
        eeg_epochs.append(resample_to_model_fs(eeg, fs))
        eog_epochs.append(resample_to_model_fs(eog[None, :], fs)[0])
        labels.append(label)

    if not eeg_epochs:
        raise ValueError(f"No valid 30-second epochs extracted from {psg_path}")

    x = np.stack(eeg_epochs).astype(np.float32)
    eog = np.stack(eog_epochs).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    if max_epochs > 0 and len(y) > max_epochs:
        rng = np.random.default_rng(rng_seed)
        keep = np.sort(rng.choice(np.arange(len(y)), size=max_epochs, replace=False))
        x = x[keep]
        eog = eog[keep]
        y = y[keep]
    info = {
        "psg": str(psg_path),
        "hypnogram": str(hyp_path),
        "fs_original": fs,
        "fs_model": FS_MODEL,
        "channels": [raw.ch_names[idx] for idx in eeg_idx],
        "eog_channel": raw.ch_names[eog_idx],
        "n_epochs": int(len(y)),
        "n_channels": int(x.shape[1]),
        "n_times": int(x.shape[2]),
        "class_counts": {LABEL_TO_STAGE[index]: int(np.sum(y == index)) for index in sorted(LABEL_TO_STAGE)},
    }
    return x, eog, y, info


def make_noisy_epochs(
    clean_epochs: np.ndarray,
    eog_epochs: np.ndarray,
    *,
    seed: int,
    snr_min_db: float,
    snr_max_db: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    noisy = np.zeros_like(clean_epochs, dtype=np.float32)
    snrs = rng.uniform(snr_min_db, snr_max_db, size=clean_epochs.shape[0]).astype(np.float32)
    for epoch_index, snr_db in enumerate(snrs):
        template = eog_epochs[epoch_index]
        template = template - float(np.mean(template))
        p_template = float(np.mean(template * template) + 1e-12)
        for channel_index in range(clean_epochs.shape[1]):
            clean = clean_epochs[epoch_index, channel_index]
            p_clean = float(np.mean(clean * clean) + 1e-12)
            scale = math.sqrt(p_clean / (p_template * (10 ** (float(snr_db) / 10.0))))
            noisy[epoch_index, channel_index] = clean + (scale * template).astype(np.float32)
    info = {
        "snr_min_db": float(snr_min_db),
        "snr_max_db": float(snr_max_db),
        "snr_mean_db": float(np.mean(snrs)),
        "snr_std_db": float(np.std(snrs)),
    }
    return noisy, info


def bandpower_features(signal: np.ndarray, fs: int) -> list[float]:
    freqs, psd = welch(signal, fs=fs, nperseg=min(len(signal), 4 * fs), noverlap=min(len(signal) // 2, 2 * fs), scaling="density")
    total_mask = (freqs >= 0.5) & (freqs <= 30.0)
    total = float(np.trapezoid(psd[total_mask], freqs[total_mask]) + 1e-12)
    feats: list[float] = [math.log(total)]
    for _name, lo, hi in FEATURE_BANDS:
        mask = (freqs >= lo) & (freqs < hi)
        power = float(np.trapezoid(psd[mask], freqs[mask]) + 1e-12)
        feats.extend([math.log(power), power / total])
    return feats


def extract_features(epochs: np.ndarray) -> np.ndarray:
    rows: list[list[float]] = []
    for epoch in epochs:
        feats: list[float] = []
        for channel in epoch:
            centered = channel - float(np.mean(channel))
            feats.extend(bandpower_features(centered, FS_MODEL))
            diff = np.diff(centered)
            feats.extend(
                [
                    float(np.mean(centered)),
                    float(np.std(centered)),
                    float(np.percentile(centered, 95) - np.percentile(centered, 5)),
                    float(np.sqrt(np.mean(diff * diff) + 1e-12)),
                    float(np.mean(np.abs(centered))),
                ]
            )
        rows.append(feats)
    return np.asarray(rows, dtype=np.float32)


def build_classifier() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=0,
                ),
            ),
        ]
    )


def evaluate(train_epochs: np.ndarray, y_train: np.ndarray, test_epochs: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    clf = build_classifier()
    clf.fit(extract_features(train_epochs), y_train)
    pred = clf.predict(extract_features(test_epochs))
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, labels=sorted(STAGE_TO_LABEL.values()), average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_test, pred)),
    }


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


def bootstrap_ci(values: list[float], *, n_resamples: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    boot = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples)]
    return percentile(boot, 0.025), percentile(boot, 0.975)


def exact_wilcoxon_less(values: list[float]) -> float:
    nz = [float(v) for v in values if abs(float(v)) > 1e-12]
    if not nz:
        return 1.0
    ranks = list(range(1, len(nz) + 1))
    order = sorted(range(len(nz)), key=lambda i: abs(nz[i]))
    ordered = [0.0] * len(nz)
    for rank, idx in zip(ranks, order):
        ordered[idx] = float(rank)
    observed = sum(rank for rank, value in zip(ordered, nz) if value > 0)
    null = []
    for mask in range(1 << len(nz)):
        null.append(sum(rank for bit, rank in enumerate(ordered) if mask & (1 << bit)))
    return sum(1 for value in null if value <= observed + 1e-12) / len(null)


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    baseline = [row for row in rows if row["condition"] == "noisy_noisy"]
    processed = [row for row in rows if row["condition"] == "denoised_denoised"]
    noisy_by_subject = defaultdict(list)
    for row in baseline:
        noisy_by_subject[row["subject"]].append(float(row["balanced_accuracy"]))
    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in processed:
        key = (str(row["denoiser_label"]), str(row["checkpoint_seed"]))
        rows_by_key.setdefault(
            key,
            {
                "denoiser_label": str(row["denoiser_label"]),
                "base": row.get("base", ""),
                "checkpoint_seed": str(row["checkpoint_seed"]),
                "subject_values": defaultdict(list),
            },
        )
        rows_by_key[key]["subject_values"][row["subject"]].append(float(row["delta_balanced_accuracy_vs_noisy_noisy"]))

    summary_rows: list[dict[str, Any]] = []
    by_label: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for item in rows_by_key.values():
        subject_deltas = []
        for subject, values in item["subject_values"].items():
            subject_deltas.append(float(statistics.mean(values)))
        stable_offset = sum(ord(ch) for ch in f"{item['denoiser_label']}:{item['checkpoint_seed']}")
        lo, hi = bootstrap_ci(subject_deltas, n_resamples=args.bootstrap_resamples, seed=args.bootstrap_seed + stable_offset)
        summary_rows.append(
            {
                "run_id": args.run_id,
                "scope": "checkpoint",
                "denoiser_label": item["denoiser_label"],
                "base": item.get("base", ""),
                "checkpoint_seed": item["checkpoint_seed"],
                "n_subjects": len(subject_deltas),
                "mean_delta_balanced_accuracy": float(statistics.mean(subject_deltas)),
                "median_delta_balanced_accuracy": float(statistics.median(subject_deltas)),
                "subject_sd": float(statistics.stdev(subject_deltas)) if len(subject_deltas) > 1 else 0.0,
                "ci95_low": lo,
                "ci95_high": hi,
                "below_noisy": int(sum(1 for value in subject_deltas if value < 0)),
                "wilcoxon_p_less": exact_wilcoxon_less(subject_deltas),
            }
        )
        for subject, values in item["subject_values"].items():
            by_label[str(item["denoiser_label"])][subject].append(float(statistics.mean(values)))

    for label, subj_map in sorted(by_label.items()):
        subject_deltas = [float(statistics.mean(values)) for _subject, values in sorted(subj_map.items())]
        lo, hi = bootstrap_ci(subject_deltas, n_resamples=args.bootstrap_resamples, seed=args.bootstrap_seed + sum(ord(ch) for ch in label))
        summary_rows.append(
            {
                "run_id": args.run_id,
                "scope": "denoiser_all_checkpoints",
                "denoiser_label": label,
                "base": label[4:] if label.startswith("base") and label[4:].isdigit() else "",
                "checkpoint_seed": "all",
                "n_subjects": len(subject_deltas),
                "mean_delta_balanced_accuracy": float(statistics.mean(subject_deltas)),
                "median_delta_balanced_accuracy": float(statistics.median(subject_deltas)),
                "subject_sd": float(statistics.stdev(subject_deltas)) if len(subject_deltas) > 1 else 0.0,
                "ci95_low": lo,
                "ci95_high": hi,
                "below_noisy": int(sum(1 for value in subject_deltas if value < 0)),
                "wilcoxon_p_less": exact_wilcoxon_less(subject_deltas),
            }
        )
    return summary_rows


def metric_row(
    *,
    args: argparse.Namespace,
    subject: str,
    train_recording: str,
    test_recording: str,
    condition: str,
    metrics: dict[str, Any],
    train_seed: int | str = "",
    test_seed: int | str = "",
    denoiser_label: str = "",
    base: int | str = "",
    ckpt_seed: int | str = "",
    checkpoint: str = "",
    trainable_parameters: int | str = "",
    noisy_reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = dict(metrics)
    row.update(
        {
            "run_id": args.run_id,
            "dataset": "Sleep_EDF_SC",
            "subject": subject,
            "train_recording": train_recording,
            "test_recording": test_recording,
            "condition": condition,
            "denoiser_label": denoiser_label,
            "base": base,
            "checkpoint_seed": ckpt_seed,
            "checkpoint": checkpoint,
            "trainable_parameters": trainable_parameters,
            "train_contamination_seed": train_seed,
            "test_contamination_seed": test_seed,
        }
    )
    if noisy_reference is not None:
        row["delta_accuracy_vs_noisy_noisy"] = float(row["accuracy"]) - float(noisy_reference["accuracy"])
        row["delta_balanced_accuracy_vs_noisy_noisy"] = float(row["balanced_accuracy"]) - float(noisy_reference["balanced_accuracy"])
    else:
        row["delta_accuracy_vs_noisy_noisy"] = ""
        row["delta_balanced_accuracy_vs_noisy_noisy"] = ""
    return row


def write_summary_md(args: argparse.Namespace, subjects: list[str], subject_info: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    lines = ["# Sleep-EDF Downstream Denoising Pilot", ""]
    lines.extend(
        [
            "",
            "## Protocol",
            "",
            f"- Subjects: `{', '.join(subjects)}`.",
            "- Dataset: Sleep-EDF Expanded Sleep Cassette.",
            "- Split: first paired recording/night for training, second paired recording/night for testing.",
            "- Task: 5-class sleep staging with N3 formed by merging stages 3 and 4.",
            "- Classifier: deterministic bandpower/statistical features + balanced logistic regression.",
            "- Comparison: raw/raw, noisy/noisy, denoised/denoised.",
            f"- SNR distribution: uniform `{args.snr_min_db}` to `{args.snr_max_db}` dB.",
            "- Synthetic EOG contamination uses the recording's EOG channel.",
            "- Denoising is applied channel-wise to 30-second epochs using fixed-window overlap-add.",
            "- Inference unit for this pilot summary: subject.",
            "",
            "## Base-Level Summary",
            "",
            "| scope | denoiser | base | checkpoint | n | mean delta balanced acc | median | 95% CI | below noisy | Wilcoxon p less |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        lines.append(
            f"| {row['scope']} | {row.get('denoiser_label', '')} | {row['base']} | {row['checkpoint_seed']} | {row['n_subjects']} | "
            f"{float(row['mean_delta_balanced_accuracy']):+.6f} | {float(row['median_delta_balanced_accuracy']):+.6f} | "
            f"[{float(row['ci95_low']):+.6f}, {float(row['ci95_high']):+.6f}] | {row['below_noisy']} | {float(row['wilcoxon_p_less']):.6f} |"
        )
    lines.extend(["", "## Subject/Recording Info", ""])
    for item in subject_info:
        lines.append(f"- {item['subject']}: train `{item['train']['recording']}` n={item['train']['n_epochs']} counts={item['train']['class_counts']}; test `{item['test']['recording']}` n={item['test']['n_epochs']} counts={item['test']['class_counts']}.")
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- This is not a BCI replication; it is an external downstream sanity check on sleep-stage decoding.",
            "- This exploratory subject count is enough to inspect directionality, not to make a polished claim.",
            "- Sleep-EDF raw EEG is not artifact-free clean ground truth.",
        ]
    )
    (args.output_dir / "sleep_edf_pilot_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bases = set(parse_int_list(args.bases))
    checkpoint_seeds = set(parse_int_list(args.checkpoint_seeds))
    train_seeds = parse_int_list(args.train_seeds)
    test_seeds = parse_int_list(args.test_seeds)
    if len(train_seeds) != len(test_seeds):
        raise ValueError("--train-seeds and --test-seeds must have equal length")
    grouped = discover_recordings(args.sleep_edf_dir)
    subjects = select_subjects(args.subjects, grouped)
    if args.model_spec:
        checkpoint_entries = expand_model_specs(args.model_spec)
    else:
        if not args.checkpoint_glob:
            raise ValueError("Provide either --model-spec or --checkpoint-glob")
        checkpoint_entries = [{"label": infer_model_label(path), "path": path} for path in expand_checkpoint_patterns(args.checkpoint_glob, bases=bases, seeds=checkpoint_seeds)]
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    models = []
    for entry in checkpoint_entries:
        path = Path(entry["path"])
        model, checkpoint = load_model_from_checkpoint(path, map_location="cpu", strict=True)
        cfg = dict(checkpoint.get("cfg", {})) if isinstance(checkpoint, dict) else {}
        model.to(device)
        model.eval()
        seed = cfg.get("seed", checkpoint_seed_from_path(path))
        models.append(
            {
                "label": str(entry["label"]),
                "path": path,
                "model": model,
                "base": int(checkpoint_base(path) or cfg.get("base")) if (checkpoint_base(path) is not None or str(cfg.get("base", "")).isdigit()) else "",
                "checkpoint_seed": int(seed) if str(seed).isdigit() else str(seed or ""),
                "params": int(count_trainable_parameters(model)),
            }
        )
    row_path = args.output_dir / "sleep_edf_pilot_rows.csv"
    rows: list[dict[str, Any]] = read_csv(row_path) if args.resume else []
    completed = {
        (
            row.get("subject"),
            row.get("condition"),
            row.get("denoiser_label", ""),
            row.get("base", ""),
            row.get("checkpoint_seed", ""),
            row.get("train_contamination_seed", ""),
            row.get("test_contamination_seed", ""),
        )
        for row in rows
    }
    subject_info: list[dict[str, Any]] = []
    for subject in subjects:
        recordings = grouped[subject]
        train_rec, test_rec = recordings[0], recordings[1]
        print(f"[subject] {subject} train={train_rec['recording']} test={test_rec['recording']}", flush=True)
        train_clean, train_eog, y_train, train_info = load_recording(
            Path(train_rec["psg"]),
            Path(train_rec["hyp"]),
            trim_wake_min=args.trim_wake_min,
            max_epochs=args.max_train_epochs,
            rng_seed=args.bootstrap_seed,
        )
        test_clean, test_eog, y_test, test_info = load_recording(
            Path(test_rec["psg"]),
            Path(test_rec["hyp"]),
            trim_wake_min=args.trim_wake_min,
            max_epochs=args.max_test_epochs,
            rng_seed=args.bootstrap_seed + 1,
        )
        train_info["recording"] = str(train_rec["recording"])
        test_info["recording"] = str(test_rec["recording"])
        subject_info.append({"subject": subject, "train": train_info, "test": test_info})
        raw_key = (subject, "raw_raw", "", "", "", "", "")
        if raw_key not in completed:
            rows.append(
                metric_row(
                    args=args,
                    subject=subject,
                    train_recording=str(train_rec["recording"]),
                    test_recording=str(test_rec["recording"]),
                    condition="raw_raw",
                    metrics=evaluate(train_clean, y_train, test_clean, y_test),
                )
            )
            write_csv(row_path, rows)
        for pair_index, (train_seed, test_seed) in enumerate(zip(train_seeds, test_seeds), start=1):
            print(f"[seed_pair] {subject} {pair_index}/{len(train_seeds)}", flush=True)
            train_noisy, train_noise_info = make_noisy_epochs(
                train_clean,
                train_eog,
                seed=train_seed,
                snr_min_db=args.snr_min_db,
                snr_max_db=args.snr_max_db,
            )
            test_noisy, test_noise_info = make_noisy_epochs(
                test_clean,
                test_eog,
                seed=test_seed,
                snr_min_db=args.snr_min_db,
                snr_max_db=args.snr_max_db,
            )
            noisy_key = (subject, "noisy_noisy", "", "", "", str(train_seed), str(test_seed))
            noisy_metrics = None
            if noisy_key not in completed:
                noisy_metrics = evaluate(train_noisy, y_train, test_noisy, y_test)
                rows.append(
                    metric_row(
                        args=args,
                        subject=subject,
                        train_recording=str(train_rec["recording"]),
                        test_recording=str(test_rec["recording"]),
                        condition="noisy_noisy",
                        metrics=noisy_metrics,
                        train_seed=train_seed,
                        test_seed=test_seed,
                    )
                )
                write_csv(row_path, rows)
            else:
                matches = [row for row in rows if (row.get("subject"), row.get("condition"), row.get("train_contamination_seed"), row.get("test_contamination_seed")) == (subject, "noisy_noisy", str(train_seed), str(test_seed))]
                noisy_metrics = matches[0] if matches else None
            if noisy_metrics is None:
                raise RuntimeError(f"Missing noisy reference for {subject} {train_seed}/{test_seed}")
            for item in models:
                key = (subject, "denoised_denoised", str(item["label"]), str(item["base"]), str(item["checkpoint_seed"]), str(train_seed), str(test_seed))
                if key in completed:
                    continue
                print(f"[denoise] {subject} model={item['label']} base={item['base']} seed={item['checkpoint_seed']} pair={pair_index}", flush=True)
                train_den = denoise_epochs(item["model"], train_noisy, device=device, batch_size=args.batch_size)
                test_den = denoise_epochs(item["model"], test_noisy, device=device, batch_size=args.batch_size)
                rows.append(
                    metric_row(
                        args=args,
                        subject=subject,
                        train_recording=str(train_rec["recording"]),
                        test_recording=str(test_rec["recording"]),
                        condition="denoised_denoised",
                        metrics=evaluate(train_den, y_train, test_den, y_test),
                        train_seed=train_seed,
                        test_seed=test_seed,
                        denoiser_label=item["label"],
                        base=item["base"],
                        ckpt_seed=item["checkpoint_seed"],
                        checkpoint=str(item["path"]),
                        trainable_parameters=item["params"],
                        noisy_reference=noisy_metrics,
                    )
                )
                write_csv(row_path, rows)

    summary_rows = summarize(rows, args)
    write_csv(args.output_dir / "sleep_edf_pilot_summary_rows.csv", summary_rows)
    (args.output_dir / "sleep_edf_pilot_subject_info.json").write_text(json.dumps(subject_info, indent=2) + "\n", encoding="utf-8")
    write_summary_md(args, subjects, subject_info, summary_rows)
    print(f"[done] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
