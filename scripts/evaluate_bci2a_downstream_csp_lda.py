#!/usr/bin/env python3
"""BCI IV-2a downstream CSP+LDA validation for denoised EEG."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import statistics
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, resample_poly
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eeg_denoise_benchmark.models import TinyDenoiser, build_eegdn_baseline, count_trainable_parameters  # noqa: E402


FS_MODEL = 256
N_EEG_CHANNELS = 22
CLASS_NAMES = ["left hand", "right hand", "feet", "tongue"]
METRIC_KEYS = ["accuracy", "balanced_accuracy", "cohen_kappa", "macro_f1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-mat", type=Path, required=True)
    parser.add_argument("--test-mat", type=Path, required=True)
    parser.add_argument("--checkpoint-glob", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--subject", default="", help="BCI IV-2a subject label, e.g. A01.")
    parser.add_argument("--device", default="cuda")
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


def natural_key(path: Path) -> tuple[int, int, str]:
    base_match = re.search(r"base(\d+)", str(path))
    seed_match = re.search(r"seed(\d+)", str(path))
    base = int(base_match.group(1)) if base_match else 10**9
    seed = int(seed_match.group(1)) if seed_match else 10**9
    return base, seed, str(path)


def expand_checkpoints(pattern: str) -> list[Path]:
    checkpoints = sorted((Path(path) for path in glob.glob(pattern)), key=natural_key)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints matched: {pattern}")
    return checkpoints


def _mat_struct_to_dict(obj: Any) -> dict[str, Any]:
    return {name: getattr(obj, name) for name in getattr(obj, "_fieldnames", [])}


def load_bci2a_trials(
    mat_path: Path,
    *,
    trial_start_sec: float,
    trial_stop_sec: float,
    include_artifact_trials: bool,
    eog_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    mat = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    if "data" not in mat:
        raise KeyError(f"'data' not found in {mat_path}; keys={list(mat)}")

    data = mat["data"]
    runs = data if isinstance(data, (list, np.ndarray)) else [data]

    eeg_trials: list[np.ndarray] = []
    eog_trials: list[np.ndarray] = []
    labels: list[int] = []
    artifact_flags: list[int] = []
    run_indices: list[int] = []
    fs_values: list[int] = []
    n_labeled_total = 0
    n_artifact_labeled_total = 0

    for run_index, run in enumerate(runs):
        fields = _mat_struct_to_dict(run) if hasattr(run, "_fieldnames") else dict(run)
        trial = np.asarray(fields.get("trial", []), dtype=np.int64).reshape(-1)
        y = np.asarray(fields.get("y", []), dtype=np.int64).reshape(-1)
        artifacts = np.asarray(fields.get("artifacts", np.zeros_like(y)), dtype=np.int64).reshape(-1)
        if trial.size == 0 or y.size == 0:
            continue

        x = np.asarray(fields["X"], dtype=np.float32)
        fs = int(np.asarray(fields.get("fs", 250)).item())
        start_offset = int(round(trial_start_sec * fs))
        stop_offset = int(round(trial_stop_sec * fs))
        n_labeled_total += int(y.size)
        n_artifact_labeled_total += int(np.sum(artifacts != 0))

        for trial_start, label, artifact_flag in zip(trial, y, artifacts):
            if artifact_flag and not include_artifact_trials:
                continue
            # MATLAB exports are 1-indexed.
            start = int(trial_start) - 1 + start_offset
            stop = int(trial_start) - 1 + stop_offset
            if start < 0 or stop > x.shape[0]:
                continue
            eeg = x[start:stop, :N_EEG_CHANNELS].T
            eog = x[start:stop, eog_index]
            eeg = resample_to_model_fs(eeg, fs)
            eog = resample_to_model_fs(eog[None, :], fs)[0]
            eeg_trials.append(eeg.astype(np.float32))
            eog_trials.append(eog.astype(np.float32))
            labels.append(int(label) - 1)
            artifact_flags.append(int(artifact_flag))
            run_indices.append(run_index)
            fs_values.append(fs)

    if not eeg_trials:
        raise ValueError(f"No labeled trials extracted from {mat_path}")

    eeg_array = np.stack(eeg_trials).astype(np.float32)
    eog_array = np.stack(eog_trials).astype(np.float32)
    label_array = np.asarray(labels, dtype=np.int64)
    info = {
        "source": str(mat_path),
        "n_trials": int(eeg_array.shape[0]),
        "n_labeled_total": int(n_labeled_total),
        "n_artifact_labeled_total": int(n_artifact_labeled_total),
        "n_artifact_labeled_excluded": int(n_artifact_labeled_total) if not include_artifact_trials else 0,
        "n_channels": int(eeg_array.shape[1]),
        "n_times": int(eeg_array.shape[2]),
        "fs_original_values": sorted(set(int(value) for value in fs_values)),
        "fs_model": FS_MODEL,
        "trial_start_sec": float(trial_start_sec),
        "trial_stop_sec": float(trial_stop_sec),
        "include_artifact_trials": bool(include_artifact_trials),
        "kept_artifact_labeled_trials": int(np.sum(artifact_flags)),
        "run_indices": sorted(set(int(value) for value in run_indices)),
        "class_counts": {
            CLASS_NAMES[index]: int(np.sum(label_array == index)) for index in range(len(CLASS_NAMES))
        },
    }
    return eeg_array, eog_array, label_array, info


def resample_to_model_fs(x: np.ndarray, fs_in: float) -> np.ndarray:
    fs_in_i = int(round(fs_in))
    if fs_in_i == FS_MODEL:
        return x.astype(np.float32)
    gcd = math.gcd(FS_MODEL, fs_in_i)
    return resample_poly(x, FS_MODEL // gcd, fs_in_i // gcd, axis=-1).astype(np.float32)


def bandpass_epochs(epochs: np.ndarray, fs: int, lo: float, hi: float, order: int = 4) -> np.ndarray:
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, epochs, axis=-1).astype(np.float32)


def make_noisy_epochs(
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
        template = eog_epochs[trial_index]
        template = template - float(np.mean(template))
        p_template = float(np.mean(template * template) + 1e-12)
        for channel_index in range(clean_epochs.shape[1]):
            clean = clean_epochs[trial_index, channel_index]
            p_clean = float(np.mean(clean * clean) + 1e-12)
            scale = math.sqrt(p_clean / (p_template * (10 ** (float(snr_db) / 10.0))))
            a = (scale * template).astype(np.float32)
            artifact[trial_index, channel_index] = a
            noisy[trial_index, channel_index] = clean + a

    info = {
        "snr_min_db": float(snr_min_db),
        "snr_max_db": float(snr_max_db),
        "snr_mean_db": float(np.mean(snrs)),
        "snr_std_db": float(np.std(snrs)),
    }
    return noisy, artifact, info


def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any], int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError(f"Checkpoint does not contain model_state: {checkpoint_path}")
    cfg = dict(checkpoint.get("cfg", {}))
    cfg.setdefault("seed", checkpoint.get("seed", ""))
    model_type = str(cfg.get("model_type", "controlled_backbone")).strip().lower()
    if model_type in {"controlled_backbone", "tinydenoiser", "tiny_denoiser"}:
        model = TinyDenoiser(
            base=int(cfg.get("base", 16)),
            extra_bottleneck_blocks=int(cfg.get("extra_blocks", 2)),
            use_dwt=bool(cfg.get("use_dwt", True)),
            use_gate=bool(cfg.get("use_gate", True)),
            use_attn=bool(cfg.get("use_attn", True)),
            use_artifact_head=bool(cfg.get("use_artifact_head", True)),
            conv_block=str(cfg.get("conv_block", "ds")),
        )
    else:
        datanum = int(cfg.get("datanum", cfg.get("length", 512)))
        model = build_eegdn_baseline(model_type=model_type, datanum=datanum)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model, cfg, count_trainable_parameters(model)


def checkpoint_seed(checkpoint_path: Path, cfg: dict[str, Any]) -> int | str:
    if "seed" in cfg:
        return int(cfg["seed"])
    match = re.search(r"seed(\d+)", str(checkpoint_path))
    return int(match.group(1)) if match else ""


@torch.no_grad()
def denoise_epochs(
    model: torch.nn.Module,
    noisy_epochs: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    n_trials, n_channels, n_times = noisy_epochs.shape
    flat = noisy_epochs.reshape(n_trials * n_channels, n_times)
    sigmas = np.std(flat, axis=1, keepdims=True).astype(np.float32) + 1e-8
    normalized = (flat / sigmas).astype(np.float32)
    datanum = int(getattr(model, "datanum", n_times))
    if datanum != n_times:
        denoised = denoise_fixed_length_array(
            model,
            normalized,
            datanum=datanum,
            device=device,
            batch_size=batch_size,
        )
        return (denoised * sigmas).reshape(n_trials, n_channels, n_times).astype(np.float32)

    denoised = np.zeros_like(normalized, dtype=np.float32)
    for start in range(0, normalized.shape[0], batch_size):
        batch = torch.from_numpy(normalized[start : start + batch_size]).float().unsqueeze(1).to(device)
        output = model(batch)
        denoised[start : start + batch_size] = output[:, 0, :].detach().cpu().numpy()
    return (denoised * sigmas).reshape(n_trials, n_channels, n_times).astype(np.float32)


@torch.no_grad()
def denoise_fixed_length_array(
    model: torch.nn.Module,
    signals: np.ndarray,
    *,
    datanum: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Apply fixed-window EEGDN models to longer/shorter signals by overlap-add."""

    n_signals, n_times = signals.shape
    if n_times < datanum:
        pad_width = datanum - n_times
        padded = np.pad(signals, ((0, 0), (0, pad_width)), mode="edge").astype(np.float32)
        denoised = np.zeros((n_signals, datanum), dtype=np.float32)
        for start in range(0, n_signals, batch_size):
            batch = torch.from_numpy(padded[start : start + batch_size]).float().unsqueeze(1).to(device)
            output = model(batch)
            denoised[start : start + batch_size] = output[:, 0, :].detach().cpu().numpy()
        return denoised[:, :n_times]

    stride = max(1, datanum // 2)
    starts = list(range(0, max(1, n_times - datanum + 1), stride))
    final_start = n_times - datanum
    if starts[-1] != final_start:
        starts.append(final_start)

    windows: list[np.ndarray] = []
    window_meta: list[tuple[int, int]] = []
    for signal_index in range(n_signals):
        for window_start in starts:
            windows.append(signals[signal_index, window_start : window_start + datanum])
            window_meta.append((signal_index, window_start))

    output_sum = np.zeros_like(signals, dtype=np.float32)
    output_count = np.zeros_like(signals, dtype=np.float32)
    window_array = np.asarray(windows, dtype=np.float32)
    for start in range(0, len(window_array), batch_size):
        batch = torch.from_numpy(window_array[start : start + batch_size]).float().unsqueeze(1).to(device)
        output = model(batch)[:, 0, :].detach().cpu().numpy()
        for offset, denoised_window in enumerate(output):
            signal_index, window_start = window_meta[start + offset]
            window_stop = window_start + datanum
            output_sum[signal_index, window_start:window_stop] += denoised_window
            output_count[signal_index, window_start:window_stop] += 1.0

    return output_sum / np.maximum(output_count, 1.0)


def build_classifier(args: argparse.Namespace) -> Pipeline:
    try:
        import mne
        from mne.decoding import CSP
    except Exception as exc:  # pragma: no cover - depends on remote env
        raise RuntimeError("mne is required for CSP+LDA downstream evaluation") from exc

    mne.set_log_level("WARNING")
    csp = CSP(
        n_components=args.csp_components,
        reg="oas",
        log=True,
        norm_trace=False,
        cov_est="concat",
    )
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    return Pipeline([("csp", csp), ("lda", lda)])


def evaluate_classifier(
    classifier: Pipeline,
    epochs: np.ndarray,
    labels: np.ndarray,
    *,
    condition: str,
    variant: str = "",
    base: int | str = "",
    train_seed: int | str = "",
    trainable_parameters: int | str = "",
    checkpoint: str = "",
) -> dict[str, Any]:
    prediction = classifier.predict(epochs)
    cm = confusion_matrix(labels, prediction, labels=list(range(len(CLASS_NAMES))))
    return {
        "subject": "",
        "condition": condition,
        "variant": variant,
        "base": base,
        "train_seed": train_seed,
        "trainable_parameters": trainable_parameters,
        "checkpoint": checkpoint,
        "n_trials": int(len(labels)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "cohen_kappa": float(cohen_kappa_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro")),
        "confusion_matrix": json.dumps(cm.tolist()),
    }


def fit_classifier(args: argparse.Namespace, train_epochs: np.ndarray, labels: np.ndarray) -> Pipeline:
    classifier = build_classifier(args)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        classifier.fit(train_epochs, labels)
    return classifier


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row["condition"] not in {"clean_denoised", "denoised_denoised"}:
            continue
        variant = str(row.get("variant") or f"base{row['base']}")
        grouped.setdefault((str(row["condition"]), variant, str(row["base"])), []).append(row)

    aggregates: list[dict[str, Any]] = []
    for (condition, variant, base), items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        seeds = sorted(int(item["train_seed"]) for item in items if item["train_seed"] != "")
        aggregate: dict[str, Any] = {
            "condition": f"{condition}_aggregate",
            "variant": variant,
            "base": base,
            "n": len(items),
            "train_seeds": " ".join(str(seed) for seed in seeds),
            "trainable_parameters": int(items[0]["trainable_parameters"]),
        }
        for key in METRIC_KEYS:
            values = [float(item[key]) for item in items]
            aggregate[f"{key}_mean"] = float(statistics.mean(values))
            aggregate[f"{key}_std"] = float(statistics.stdev(values)) if len(values) > 1 else 0.0
        aggregates.append(aggregate)
    return aggregates


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    info: dict[str, Any],
) -> None:
    subject = str(info.get("subject") or "")
    subject_title = f" {subject}" if subject else ""
    clean_clean = next(row for row in rows if row["condition"] == "clean_clean")
    clean_noisy = next(row for row in rows if row["condition"] == "clean_noisy")
    noisy_noisy = next(row for row in rows if row["condition"] == "noisy_noisy")
    lines = [f"# BCI IV-2a{subject_title} Downstream CSP+LDA Validation", ""]
    lines.append("Protocol: CSP+LDA on BCI IV-2a train/evaluation sessions with clean, noisy, strict denoised, and matched denoised train/test conditions.")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append(f"- Train file: `{info['train']['source']}`.")
    lines.append(f"- Test file: `{info['test']['source']}`.")
    lines.append(f"- Trial window: `{info['trial_start_sec']:.2f}` to `{info['trial_stop_sec']:.2f}` s.")
    lines.append(f"- CSP band: `{info['bandpass_low_hz']:.1f}-{info['bandpass_high_hz']:.1f}` Hz.")
    lines.append(f"- CSP components: `{info['csp_components']}`; CSP covariance regularization: `oas`; LDA shrinkage: `auto`.")
    lines.append(f"- Artifact-labeled trials included: `{info['include_artifact_trials']}`.")
    lines.append(f"- Train contamination SNR: `{info['snr_min_db']:.1f}` to `{info['snr_max_db']:.1f}` dB, mean `{info['train_noise']['snr_mean_db']:.4f}` dB.")
    lines.append(f"- Test contamination SNR: `{info['snr_min_db']:.1f}` to `{info['snr_max_db']:.1f}` dB, mean `{info['test_noise']['snr_mean_db']:.4f}` dB.")
    lines.append("- Classifier policy: strict rows train CSP+LDA on the clean training session; matched rows train CSP+LDA on the same preprocessing condition used at test.")
    lines.append("")
    lines.append("## Baselines")
    lines.append("")
    lines.append("| Condition | Accuracy | Balanced accuracy | Kappa | Macro F1 |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in [clean_clean, clean_noisy, noisy_noisy]:
        lines.append(
            f"| {row['condition']} | {row['accuracy']:.6f} | {row['balanced_accuracy']:.6f} | "
            f"{row['cohen_kappa']:.6f} | {row['macro_f1']:.6f} |"
        )
    lines.append("")
    lines.append("## Denoised Aggregate By Width")
    lines.append("")
    lines.append("| Condition | Variant | Base | n | Seeds | Params | Accuracy | Balanced accuracy | Kappa | Macro F1 | Delta acc vs clean-clean | Delta acc vs clean-noisy | Delta acc vs noisy-noisy |")
    lines.append("|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in aggregates:
        delta_clean_clean = row["accuracy_mean"] - clean_clean["accuracy"]
        delta_clean_noisy = row["accuracy_mean"] - clean_noisy["accuracy"]
        delta_noisy_noisy = row["accuracy_mean"] - noisy_noisy["accuracy"]
        lines.append(
            f"| {row['condition']} | {row.get('variant', '')} | {row['base']} | {row['n']} | {row['train_seeds']} | {row['trainable_parameters']} | "
            f"{row['accuracy_mean']:.6f} +/- {row['accuracy_std']:.6f} | "
            f"{row['balanced_accuracy_mean']:.6f} +/- {row['balanced_accuracy_std']:.6f} | "
            f"{row['cohen_kappa_mean']:.6f} +/- {row['cohen_kappa_std']:.6f} | "
            f"{row['macro_f1_mean']:.6f} +/- {row['macro_f1_std']:.6f} | "
            f"{delta_clean_clean:+.6f} | {delta_clean_noisy:+.6f} | {delta_noisy_noisy:+.6f} |"
        )
    lines.append("")
    lines.append("## Per-Checkpoint Results")
    lines.append("")
    lines.append("| Condition | Variant | Base | Train seed | Params | Accuracy | Balanced accuracy | Kappa | Macro F1 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        if row["condition"] not in {"clean_denoised", "denoised_denoised"}:
            continue
        lines.append(
            f"| {row['condition']} | {row.get('variant', '')} | {row['base']} | {row['train_seed']} | {row['trainable_parameters']} | "
            f"{row['accuracy']:.6f} | {row['balanced_accuracy']:.6f} | "
            f"{row['cohen_kappa']:.6f} | {row['macro_f1']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[start] run_id={args.run_id} device={device}")
    checkpoints = expand_checkpoints(args.checkpoint_glob)
    print(f"[checkpoints] {len(checkpoints)}")
    for checkpoint in checkpoints:
        print(f"  - {checkpoint}")

    train_clean, train_eog, y_train, train_info = load_bci2a_trials(
        args.train_mat,
        trial_start_sec=args.trial_start_sec,
        trial_stop_sec=args.trial_stop_sec,
        include_artifact_trials=args.include_artifact_trials,
        eog_index=args.eog_index,
    )
    test_clean, test_eog, y_test, test_info = load_bci2a_trials(
        args.test_mat,
        trial_start_sec=args.trial_start_sec,
        trial_stop_sec=args.trial_stop_sec,
        include_artifact_trials=args.include_artifact_trials,
        eog_index=args.eog_index,
    )
    train_noisy, _train_artifact, train_noise_info = make_noisy_epochs(
        train_clean,
        train_eog,
        seed=args.seed + 1000,
        snr_min_db=args.snr_min_db,
        snr_max_db=args.snr_max_db,
    )
    test_noisy, _test_artifact, test_noise_info = make_noisy_epochs(
        test_clean,
        test_eog,
        seed=args.seed,
        snr_min_db=args.snr_min_db,
        snr_max_db=args.snr_max_db,
    )

    train_features = bandpass_epochs(train_clean, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz)
    train_noisy_features = bandpass_epochs(train_noisy, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz)
    test_clean_features = bandpass_epochs(test_clean, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz)
    test_noisy_features = bandpass_epochs(test_noisy, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz)

    clean_classifier = fit_classifier(args, train_features, y_train)
    noisy_classifier = fit_classifier(args, train_noisy_features, y_train)

    rows: list[dict[str, Any]] = []
    rows.append(evaluate_classifier(clean_classifier, test_clean_features, y_test, condition="clean_clean"))
    rows.append(evaluate_classifier(clean_classifier, test_noisy_features, y_test, condition="clean_noisy"))
    rows.append(evaluate_classifier(noisy_classifier, test_noisy_features, y_test, condition="noisy_noisy"))
    for row in rows:
        row["subject"] = args.subject
    print(
        f"[baseline] clean_clean_acc={rows[0]['accuracy']:.6f} "
        f"clean_noisy_acc={rows[1]['accuracy']:.6f} noisy_noisy_acc={rows[2]['accuracy']:.6f} "
        f"clean_clean_kappa={rows[0]['cohen_kappa']:.6f} "
        f"clean_noisy_kappa={rows[1]['cohen_kappa']:.6f} noisy_noisy_kappa={rows[2]['cohen_kappa']:.6f}"
    )

    for checkpoint_path in checkpoints:
        model, cfg, n_params = load_checkpoint_model(checkpoint_path, device)
        base = cfg.get("base", "")
        variant = str(cfg.get("variant") or cfg.get("model_type") or f"base{base}")
        train_seed = checkpoint_seed(checkpoint_path, cfg)
        print(f"[eval_start] base={base} seed={train_seed} params={n_params} checkpoint={checkpoint_path}")
        train_denoised = denoise_epochs(model, train_noisy, device=device, batch_size=args.batch_size)
        test_denoised = denoise_epochs(model, test_noisy, device=device, batch_size=args.batch_size)
        train_denoised_features = bandpass_epochs(train_denoised, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz)
        test_denoised_features = bandpass_epochs(test_denoised, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz)
        strict_row = evaluate_classifier(
            clean_classifier,
            test_denoised_features,
            y_test,
            condition="clean_denoised",
            base=base,
            variant=variant,
            train_seed=train_seed,
            trainable_parameters=n_params,
            checkpoint=str(checkpoint_path),
        )
        denoised_classifier = fit_classifier(args, train_denoised_features, y_train)
        matched_row = evaluate_classifier(
            denoised_classifier,
            test_denoised_features,
            y_test,
            condition="denoised_denoised",
            base=base,
            variant=variant,
            train_seed=train_seed,
            trainable_parameters=n_params,
            checkpoint=str(checkpoint_path),
        )
        rows.extend([strict_row, matched_row])
        strict_row["subject"] = args.subject
        matched_row["subject"] = args.subject
        print(
            f"[result] base={base} seed={train_seed} "
            f"clean_denoised_acc={strict_row['accuracy']:.6f} "
            f"denoised_denoised_acc={matched_row['accuracy']:.6f} "
            f"clean_denoised_kappa={strict_row['cohen_kappa']:.6f} "
            f"denoised_denoised_kappa={matched_row['cohen_kappa']:.6f}"
        )

    aggregates = aggregate_rows(rows)
    info = {
        "run_id": args.run_id,
        "subject": args.subject,
        "protocol": "BCI IV-2a CSP+LDA condition matrix: clean-clean, clean-noisy, noisy-noisy, clean-denoised, denoised-denoised",
        "train": train_info,
        "test": test_info,
        "train_noise": train_noise_info,
        "test_noise": test_noise_info,
        "trial_start_sec": float(args.trial_start_sec),
        "trial_stop_sec": float(args.trial_stop_sec),
        "bandpass_low_hz": float(args.bandpass_low_hz),
        "bandpass_high_hz": float(args.bandpass_high_hz),
        "csp_components": int(args.csp_components),
        "include_artifact_trials": bool(args.include_artifact_trials),
        "snr_min_db": float(args.snr_min_db),
        "snr_max_db": float(args.snr_max_db),
        "class_names": CLASS_NAMES,
    }
    summary = {"info": info, "rows": rows, "aggregates": aggregates}

    json_path = args.output_dir / "bci2a_csp_lda_summary.json"
    rows_csv_path = args.output_dir / "bci2a_csp_lda_rows.csv"
    aggregate_csv_path = args.output_dir / "bci2a_csp_lda_aggregate_by_width.csv"
    md_path = args.output_dir / "bci2a_csp_lda_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(rows_csv_path, rows)
    write_csv(aggregate_csv_path, aggregates)
    write_markdown(md_path, rows=rows, aggregates=aggregates, info=info)
    print(f"[written] {json_path}")
    print(f"[written] {rows_csv_path}")
    print(f"[written] {aggregate_csv_path}")
    print(f"[written] {md_path}")
    print("[done]")


if __name__ == "__main__":
    main()
