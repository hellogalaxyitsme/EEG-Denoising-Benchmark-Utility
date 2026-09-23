"""Shared Sleep-EDF recording, contamination, feature, and evaluation helpers."""

from __future__ import annotations

import glob
import math
import re
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

from scripts.evaluate_bci2a_downstream_csp_lda import FS_MODEL, denoise_epochs, expand_checkpoints  # noqa: E402

STAGE_TO_LABEL = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
LABEL_TO_STAGE = {value: key for key, value in STAGE_TO_LABEL.items()}
FEATURE_BANDS = [("delta", 0.5, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 12.0), ("sigma", 12.0, 16.0), ("beta", 16.0, 30.0)]

def parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.replace(" ", ",").split(",") if part.strip()]


def checkpoint_base(path: Path) -> int | None:
    match = re.search(r"base(\d+)", str(path))
    return int(match.group(1)) if match else None


def checkpoint_seed_from_path(path: Path) -> int | None:
    match = re.search(r"seed(\d+)", str(path))
    return int(match.group(1)) if match else None

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
