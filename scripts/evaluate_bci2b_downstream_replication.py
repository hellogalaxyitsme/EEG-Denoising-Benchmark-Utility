#!/usr/bin/env python3
"""IV-2b BCI IV-2b downstream denoising replication with subject-level inference."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
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
from scipy.signal import resample_poly
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "src"))

from evaluate_bci2a_all_subjects_braindecode_classifiers import (  # noqa: E402
    build_classifier_model,
    standardize_train_test,
)
from scripts.evaluate_bci2a_downstream_contamination_types_csp_lda import (  # noqa: E402
    DEFAULT_RECIPES,
    load_emg_pool,
    make_recipe_noisy_epochs,
    parse_recipes,
)
from scripts.evaluate_bci2a_downstream_csp_lda import (  # noqa: E402
    FS_MODEL,
    bandpass_epochs,
    checkpoint_seed,
    denoise_epochs,
    expand_checkpoints,
    load_checkpoint_model,
)


DEFAULT_SUBJECTS = [f"B{index:02d}" for index in range(1, 10)]
CLASSIFIERS = ["csp_lda", "eegnet"]
CLASS_LABELS = [0, 1]
CLASS_NAMES = ["left hand", "right hand"]
METRIC_KEYS = ["accuracy", "balanced_accuracy", "cohen_kappa", "macro_f1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bci2b-dir",
        type=Path,
        required=True, help="Path to the licensed input data.",
    )
    parser.add_argument(
        "--emg-pool",
        type=Path,
        required=True, help="Path to the licensed input data.",
    )
    parser.add_argument("--subjects", default=",".join(DEFAULT_SUBJECTS))
    parser.add_argument("--recipes", default=",".join(DEFAULT_RECIPES))
    parser.add_argument("--classifiers", default="csp_lda,eegnet")
    parser.add_argument(
        "--checkpoint-glob",
        action="append",
        required=True,
        help="Checkpoint glob pattern. Repeat this option to combine checkpoint families.",
    )
    parser.add_argument("--csp-bases", default="2,4,6,8,16")
    parser.add_argument("--neural-bases", default="6")
    parser.add_argument("--checkpoint-seeds", default="42,43,44")
    parser.add_argument("--classifier-seeds", default="501,502,503")
    parser.add_argument("--test-seeds", default="42,43,44,45,46")
    parser.add_argument("--train-seeds", default="1042,1043,1044,1045,1046")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--denoiser-batch-size", type=int, default=256)
    parser.add_argument("--classifier-batch-size", type=int, default=64)
    parser.add_argument("--classifier-epochs", type=int, default=80)
    parser.add_argument("--classifier-lr", type=float, default=1e-3)
    parser.add_argument("--classifier-weight-decay", type=float, default=1e-4)
    parser.add_argument("--snr-min-db", type=float, default=-6.0)
    parser.add_argument("--snr-max-db", type=float, default=2.0)
    parser.add_argument("--trial-start-sec", type=float, default=0.5, help="Seconds after cue event.")
    parser.add_argument("--trial-stop-sec", type=float, default=4.0, help="Seconds after cue event.")
    parser.add_argument("--bandpass-low-hz", type=float, default=8.0)
    parser.add_argument("--bandpass-high-hz", type=float, default=30.0)
    parser.add_argument("--csp-components", type=int, default=2)
    parser.add_argument("--include-artifact-trials", action="store_true")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260815)
    parser.add_argument("--resume", action="store_true", help="Reuse existing raw rows and skip completed cells.")
    return parser.parse_args()


def parse_csv_list(raw: str) -> list[str]:
    values = [item.strip().lower() for item in raw.replace(" ", ",").split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated value")
    return values


def parse_subjects(raw: str) -> list[str]:
    subjects = [item.strip().upper() for item in raw.replace(" ", ",").split(",") if item.strip()]
    if not subjects:
        raise ValueError("Expected at least one subject")
    unknown = sorted(set(subjects) - set(DEFAULT_SUBJECTS))
    if unknown:
        raise ValueError(f"Unknown BCI IV-2b subjects: {unknown}")
    return subjects


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
        raise ValueError(f"Missing checkpoints for requested base/seed pairs: {missing}")
    return paths


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    preferred = [
        "run_id",
        "dataset",
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


def normalize_name(name: str) -> str:
    return name.strip().replace("EEG-", "").replace("EEG:", "").replace(" ", "").upper()


def eeg_channel_indices(ch_names: list[str]) -> list[int]:
    normalized = [normalize_name(name) for name in ch_names]
    out = []
    for target in ["C3", "CZ", "C4"]:
        if target not in normalized:
            raise ValueError(f"Required IV-2b EEG channel {target!r} missing; available={ch_names}")
        out.append(normalized.index(target))
    return out


def eog_channel_indices(ch_names: list[str]) -> list[int]:
    indices = [index for index, name in enumerate(ch_names) if "EOG" in normalize_name(name)]
    if not indices:
        raise ValueError(f"No EOG channels found; available={ch_names}")
    return indices


def resample_to_model_fs(x: np.ndarray, fs_in: float) -> np.ndarray:
    fs_in_i = int(round(fs_in))
    if fs_in_i == FS_MODEL:
        return x.astype(np.float32)
    gcd = math.gcd(FS_MODEL, fs_in_i)
    return resample_poly(x, FS_MODEL // gcd, fs_in_i // gcd, axis=-1).astype(np.float32)


def load_bci2b_trials(
    gdf_paths: list[Path],
    *,
    trial_start_sec: float,
    trial_stop_sec: float,
    include_artifact_trials: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    try:
        import mne
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("mne is required for BCI IV-2b .gdf evaluation") from exc

    mne.set_log_level("WARNING")
    eeg_trials: list[np.ndarray] = []
    eog_trials: list[np.ndarray] = []
    labels: list[int] = []
    session_infos: list[dict[str, Any]] = []

    for gdf_path in gdf_paths:
        if not gdf_path.exists():
            raise FileNotFoundError(gdf_path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_gdf(str(gdf_path), preload=True, verbose=False)
            events, event_id = mne.events_from_annotations(raw, verbose=False)
        fs = float(raw.info["sfreq"])
        data = np.nan_to_num(raw.get_data() * 1e6).astype(np.float32)
        eeg_idx = eeg_channel_indices(raw.ch_names)
        eog_idx = eog_channel_indices(raw.ch_names)
        code_to_label = {}
        for key, value in event_id.items():
            key_s = str(key)
            if key_s.endswith("769") or key_s == "769":
                code_to_label[int(value)] = 0
            if key_s.endswith("770") or key_s == "770":
                code_to_label[int(value)] = 1
        artifact_codes = {int(value) for key, value in event_id.items() if str(key).endswith("1023") or str(key) == "1023"}
        if not code_to_label:
            raise ValueError(f"No labeled 769/770 cue events found in {gdf_path}; event_id={event_id}")

        start_offset = int(round(trial_start_sec * fs))
        stop_offset = int(round(trial_stop_sec * fs))
        n_labeled = 0
        n_artifact_excluded = 0
        n_out_of_bounds = 0
        session_labels: list[int] = []
        for sample, _previous, code in events:
            if int(code) not in code_to_label:
                continue
            n_labeled += 1
            start = int(sample) + start_offset
            stop = int(sample) + stop_offset
            if start < 0 or stop > data.shape[1]:
                n_out_of_bounds += 1
                continue
            has_artifact = any(
                int(event_sample) >= int(sample) - int(round(0.5 * fs))
                and int(event_sample) <= stop
                and int(event_code) in artifact_codes
                for event_sample, _prev, event_code in events
            )
            if has_artifact and not include_artifact_trials:
                n_artifact_excluded += 1
                continue
            eeg = data[eeg_idx, start:stop]
            eog = np.mean(data[eog_idx, start:stop], axis=0)
            eeg_trials.append(resample_to_model_fs(eeg, fs))
            eog_trials.append(resample_to_model_fs(eog[None, :], fs)[0])
            label = code_to_label[int(code)]
            labels.append(label)
            session_labels.append(label)

        session_infos.append(
            {
                "source": str(gdf_path),
                "fs_original": fs,
                "ch_names": list(raw.ch_names),
                "eeg_channels": [raw.ch_names[index] for index in eeg_idx],
                "eog_channels": [raw.ch_names[index] for index in eog_idx],
                "n_labeled_cue_events": n_labeled,
                "n_artifact_labeled_excluded": n_artifact_excluded,
                "n_out_of_bounds_labeled": n_out_of_bounds,
                "n_trials_kept": len(session_labels),
                "class_counts": {CLASS_NAMES[index]: int(np.sum(np.asarray(session_labels) == index)) for index in CLASS_LABELS},
                "event_id": {str(key): int(value) for key, value in event_id.items()},
            }
        )

    if not eeg_trials:
        raise ValueError(f"No labeled trials extracted from {gdf_paths}")
    eeg_array = np.stack(eeg_trials).astype(np.float32)
    eog_array = np.stack(eog_trials).astype(np.float32)
    label_array = np.asarray(labels, dtype=np.int64)
    info = {
        "sources": [str(path) for path in gdf_paths],
        "sessions": session_infos,
        "n_trials": int(eeg_array.shape[0]),
        "n_channels": int(eeg_array.shape[1]),
        "n_times": int(eeg_array.shape[2]),
        "fs_model": FS_MODEL,
        "trial_window_reference": "cue_event",
        "trial_start_sec_after_cue": float(trial_start_sec),
        "trial_stop_sec_after_cue": float(trial_stop_sec),
        "include_artifact_trials": bool(include_artifact_trials),
        "class_counts": {CLASS_NAMES[index]: int(np.sum(label_array == index)) for index in CLASS_LABELS},
    }
    return eeg_array, eog_array, label_array, info


def load_subject_clean(args: argparse.Namespace, subject: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    train_paths = [args.bci2b_dir / f"{subject}01T.gdf", args.bci2b_dir / f"{subject}02T.gdf"]
    test_paths = [args.bci2b_dir / f"{subject}03T.gdf"]
    train_clean, train_eog, y_train, train_info = load_bci2b_trials(
        train_paths,
        trial_start_sec=args.trial_start_sec,
        trial_stop_sec=args.trial_stop_sec,
        include_artifact_trials=args.include_artifact_trials,
    )
    test_clean, test_eog, y_test, test_info = load_bci2b_trials(
        test_paths,
        trial_start_sec=args.trial_start_sec,
        trial_stop_sec=args.trial_stop_sec,
        include_artifact_trials=args.include_artifact_trials,
    )
    return train_clean, train_eog, test_clean, test_eog, y_train, y_test, {"train": train_info, "test": test_info}


def load_models(checkpoints: list[Path], device: torch.device) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for checkpoint_path in checkpoints:
        model, cfg, n_params = load_checkpoint_model(checkpoint_path, device)
        base = checkpoint_base(checkpoint_path, cfg)
        seed = checkpoint_seed(checkpoint_path, cfg)
        models.append(
            {
                "checkpoint": str(checkpoint_path),
                "model": model,
                "cfg": cfg,
                "base": int(base) if base is not None else "",
                "checkpoint_seed": int(seed) if str(seed) != "" else "",
                "trainable_parameters": int(n_params),
            }
        )
    return models


def build_csp_lda(args: argparse.Namespace) -> Pipeline:
    try:
        import mne
        from mne.decoding import CSP
    except Exception as exc:  # pragma: no cover
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


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    return {
        "n_trials": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASS_LABELS, average="macro")),
        "confusion_matrix": json.dumps(cm.tolist()),
    }


def evaluate_csp_lda(args: argparse.Namespace, train_epochs: np.ndarray, y_train: np.ndarray, test_epochs: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    train_features = bandpass_epochs(train_epochs, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz)
    test_features = bandpass_epochs(test_epochs, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz)
    classifier = build_csp_lda(args)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        classifier.fit(train_features, y_train)
    return metric_row(y_test, classifier.predict(test_features))


def set_classifier_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


def evaluate_neural(
    args: argparse.Namespace,
    *,
    classifier_name: str,
    classifier_seed: int,
    train_epochs: np.ndarray,
    y_train: np.ndarray,
    test_epochs: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    set_classifier_seed(classifier_seed)
    train_x, test_x = standardize_train_test(train_epochs, test_epochs)
    model = build_classifier_model(
        classifier_name,
        n_chans=int(train_x.shape[1]),
        n_times=int(train_x.shape[2]),
        n_outputs=len(CLASS_LABELS),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.classifier_lr, weight_decay=args.classifier_weight_decay)
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator()
    generator.manual_seed(classifier_seed)
    dataset = TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(y_train).long())
    loader = DataLoader(dataset, batch_size=args.classifier_batch_size, shuffle=True, generator=generator)
    start_time = time.time()
    last_loss = float("nan")
    model.train()
    for _epoch in range(args.classifier_epochs):
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
        for start in range(0, len(test_tensor), args.classifier_batch_size):
            xb = test_tensor[start : start + args.classifier_batch_size].to(device)
            logits = model(xb)
            predictions.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
    row = metric_row(y_test, np.concatenate(predictions).astype(np.int64))
    row.update(
        {
            "classifier_epochs": int(args.classifier_epochs),
            "classifier_batch_size": int(args.classifier_batch_size),
            "classifier_lr": float(args.classifier_lr),
            "classifier_weight_decay": float(args.classifier_weight_decay),
            "classifier_final_train_loss": float(last_loss),
            "classifier_train_time_sec": float(time.time() - start_time),
        }
    )
    return row


def base_row(
    *,
    args: argparse.Namespace,
    classifier: str,
    classifier_seed: int | str,
    subject: str,
    recipe: str,
    seed_pair_index: int,
    train_seed: int,
    test_seed: int,
    condition: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    row = dict(metrics)
    row.update(
        {
            "run_id": args.run_id,
            "dataset": "BCI_IV2b",
            "classifier": classifier,
            "classifier_seed": classifier_seed,
            "subject": subject,
            "recipe": recipe,
            "seed_pair_index": seed_pair_index,
            "train_contamination_seed": train_seed,
            "test_contamination_seed": test_seed,
            "condition": condition,
            "base": "",
            "checkpoint_seed": "",
            "checkpoint": "",
            "trainable_parameters": "",
            "delta_accuracy_vs_noisy_noisy": "",
        }
    )
    return row


def result_row(
    *,
    args: argparse.Namespace,
    classifier: str,
    classifier_seed: int | str,
    subject: str,
    recipe: str,
    seed_pair_index: int,
    train_seed: int,
    test_seed: int,
    model_entry: dict[str, Any],
    baseline_accuracy: float,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    row = dict(metrics)
    row.update(
        {
            "run_id": args.run_id,
            "dataset": "BCI_IV2b",
            "classifier": classifier,
            "classifier_seed": classifier_seed,
            "subject": subject,
            "recipe": recipe,
            "seed_pair_index": seed_pair_index,
            "train_contamination_seed": train_seed,
            "test_contamination_seed": test_seed,
            "condition": "denoised_denoised",
            "base": int(model_entry["base"]),
            "checkpoint_seed": model_entry["checkpoint_seed"],
            "checkpoint": model_entry["checkpoint"],
            "trainable_parameters": int(model_entry["trainable_parameters"]),
            "baseline_accuracy": float(baseline_accuracy),
            "delta_accuracy_vs_noisy_noisy": float(row["accuracy"]) - float(baseline_accuracy),
        }
    )
    return row


def row_key(row: dict[str, Any] | dict[str, str]) -> tuple[str, ...]:
    return (
        str(row["classifier"]),
        str(row.get("classifier_seed", "")),
        str(row["subject"]),
        str(row["recipe"]),
        str(row["seed_pair_index"]),
        str(row.get("condition", "")),
        str(row.get("base", "")),
        str(row.get("checkpoint_seed", "")),
    )


def summarize_subjects(
    baseline_rows: list[dict[str, Any] | dict[str, str]],
    result_rows: list[dict[str, Any] | dict[str, str]],
) -> list[dict[str, Any]]:
    noisy_baselines: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    raw_baselines: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in baseline_rows:
        key = (str(row["classifier"]), str(row["subject"]), str(row["recipe"]))
        if row["condition"] == "noisy_noisy":
            noisy_baselines[key].append(float(row["accuracy"]))
        elif row["condition"] == "raw_raw":
            raw_baselines[key].append(float(row["accuracy"]))

    grouped: dict[tuple[str, str, str, int], list[dict[str, Any] | dict[str, str]]] = defaultdict(list)
    for row in result_rows:
        grouped[(str(row["classifier"]), str(row["subject"]), str(row["recipe"]), int(row["base"]))].append(row)

    out: list[dict[str, Any]] = []
    for (classifier, subject, recipe, base), items in sorted(grouped.items()):
        deltas = [float(item["delta_accuracy_vs_noisy_noisy"]) for item in items]
        acc = [float(item["accuracy"]) for item in items]
        by_checkpoint_seed: dict[str, list[float]] = defaultdict(list)
        by_contamination: dict[str, list[float]] = defaultdict(list)
        by_classifier_seed: dict[str, list[float]] = defaultdict(list)
        for item in items:
            delta = float(item["delta_accuracy_vs_noisy_noisy"])
            by_checkpoint_seed[str(item["checkpoint_seed"])].append(delta)
            by_contamination[str(item["seed_pair_index"])].append(delta)
            if str(item.get("classifier_seed", "")):
                by_classifier_seed[str(item["classifier_seed"])].append(delta)
        checkpoint_means = [mean(values) for _, values in sorted(by_checkpoint_seed.items())]
        contamination_means = [mean(values) for _, values in sorted(by_contamination.items())]
        classifier_seed_means = [mean(values) for _, values in sorted(by_classifier_seed.items())]
        out.append(
            {
                "dataset": "BCI_IV2b",
                "classifier": classifier,
                "subject": subject,
                "recipe": recipe,
                "base": base,
                "n_contamination_seed_pairs": len(by_contamination),
                "n_checkpoint_seeds": len(by_checkpoint_seed),
                "n_classifier_seeds": len(by_classifier_seed) if classifier_seed_means else 0,
                "n_observations_averaged": len(items),
                "raw_accuracy_mean": mean(raw_baselines[(classifier, subject, recipe)]),
                "noisy_accuracy_mean": mean(noisy_baselines[(classifier, subject, recipe)]),
                "denoised_accuracy_mean": mean(acc),
                "delta_accuracy": mean(deltas),
                "delta_accuracy_sd_over_nuisance": sample_sd(deltas),
                "checkpoint_seed_delta_sd_within_subject": sample_sd(checkpoint_means),
                "contamination_delta_sd_within_subject": sample_sd(contamination_means),
                "classifier_seed_delta_sd_within_subject": sample_sd(classifier_seed_means),
            }
        )
    return out


def inference_rows(subject_rows: list[dict[str, Any]], *, n_bootstrap: int, bootstrap_seed: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in subject_rows:
        grouped[(str(row["classifier"]), str(row["recipe"]), int(row["base"]))].append(row)

    rows: list[dict[str, Any]] = []
    for (classifier, recipe, base), items in sorted(grouped.items()):
        deltas = [float(item["delta_accuracy"]) for item in items]
        raw_acc = [float(item["raw_accuracy_mean"]) for item in items]
        noisy_acc = [float(item["noisy_accuracy_mean"]) for item in items]
        den_acc = [float(item["denoised_accuracy_mean"]) for item in items]
        ci_low, ci_high = bootstrap_mean_ci(deltas, n_bootstrap=n_bootstrap, seed=bootstrap_seed + 1000 * len(rows))
        rows.append(
            {
                "dataset": "BCI_IV2b",
                "classifier": classifier,
                "recipe": recipe,
                "base": base,
                "n_subjects": len(items),
                "subjects": " ".join(str(item["subject"]) for item in items),
                "n_contamination_seed_pairs": int(items[0]["n_contamination_seed_pairs"]),
                "n_checkpoint_seeds": int(items[0]["n_checkpoint_seeds"]),
                "n_classifier_seeds": int(items[0]["n_classifier_seeds"]),
                "raw_accuracy_subject_mean": mean(raw_acc),
                "noisy_accuracy_subject_mean": mean(noisy_acc),
                "denoised_accuracy_subject_mean": mean(den_acc),
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


def write_markdown(path: Path, *, args: argparse.Namespace, subjects: list[str], recipes: list[str], classifiers: list[str], width_rows: list[dict[str, Any]]) -> None:
    lines = ["# BCI IV-2b downstream replication", ""]
    lines.append("This experiment evaluates matched raw/raw, noisy/noisy, and denoised/denoised downstream classification on BCI IV-2b.")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append(f"- Subjects: `{', '.join(subjects)}`.")
    lines.append("- Sessions: train on labeled `01T` and `02T`; test on labeled `03T`. The local `E` sessions contain unknown labels and are not used for downstream inference.")
    lines.append("- EEG channels: fixed IV-2b channels `C3`, `Cz`, and `C4`.")
    lines.append(f"- Trial window: `{args.trial_start_sec:.2f}` to `{args.trial_stop_sec:.2f}` seconds after the cue event.")
    lines.append(f"- Classifiers: `{', '.join(classifiers)}`.")
    lines.append(f"- Recipes: `{', '.join(recipes)}`.")
    lines.append(f"- Train contamination seeds: `{args.train_seeds}`.")
    lines.append(f"- Test contamination seeds: `{args.test_seeds}`.")
    lines.append(f"- SNR distribution: uniform from `{args.snr_min_db:.1f}` to `{args.snr_max_db:.1f}` dB.")
    lines.append(f"- Subject bootstrap resamples: `{args.n_bootstrap}`.")
    lines.append("- Inferential unit: subject. Contamination seeds, denoiser checkpoints, and classifier seeds are nuisance repetitions averaged within each subject.")
    lines.append("")
    lines.append("## Subject-Level Inference")
    lines.append("")
    lines.append("| Classifier | Recipe | Base | n subjects | Raw acc | Noisy acc | Denoised acc | Mean delta | Median delta | Subject SD | 95% subject-bootstrap CI | Below noisy | Wilcoxon p lower | BH-FDR q | Holm p |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in width_rows:
        lines.append(
            f"| {row['classifier']} | {row['recipe']} | {row['base']} | {row['n_subjects']} | "
            f"{row['raw_accuracy_subject_mean']:.6f} | {row['noisy_accuracy_subject_mean']:.6f} | "
            f"{row['denoised_accuracy_subject_mean']:.6f} | {row['mean_delta_accuracy']:+.6f} | "
            f"{row['median_delta_accuracy']:+.6f} | {row['sd_delta_accuracy_across_subjects']:.6f} | "
            f"[{row['bootstrap_ci95_low']:+.6f}, {row['bootstrap_ci95_high']:+.6f}] | "
            f"{row['subjects_below_noisy_noisy']} | {row['wilcoxon_p_denoised_lt_noisy']:.6f} | "
            f"{row['bh_fdr_q_denoised_lt_noisy']:.6f} | {row['holm_p_denoised_lt_noisy']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_subject_seed_pair(
    args: argparse.Namespace,
    *,
    subject: str,
    recipe: str,
    seed_pair_index: int,
    train_seed: int,
    test_seed: int,
    subject_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]],
    emg_pool: np.ndarray,
    models: list[dict[str, Any]],
    classifiers: list[str],
    csp_bases: set[int],
    neural_bases: set[int],
    classifier_seeds: list[int],
    baseline_rows: list[dict[str, Any] | dict[str, str]],
    result_rows: list[dict[str, Any] | dict[str, str]],
    device: torch.device,
) -> dict[str, Any]:
    train_clean, train_eog, test_clean, test_eog, y_train, y_test, info = subject_data
    train_noisy, _train_artifact, train_noise = make_recipe_noisy_epochs(
        train_clean,
        train_eog,
        emg_pool,
        recipe=recipe,
        seed=train_seed,
        snr_min_db=args.snr_min_db,
        snr_max_db=args.snr_max_db,
    )
    test_noisy, _test_artifact, test_noise = make_recipe_noisy_epochs(
        test_clean,
        test_eog,
        emg_pool,
        recipe=recipe,
        seed=test_seed,
        snr_min_db=args.snr_min_db,
        snr_max_db=args.snr_max_db,
    )

    existing_baseline_keys = {row_key(row) for row in baseline_rows}
    existing_result_keys = {row_key(row) for row in result_rows}

    if "csp_lda" in classifiers:
        raw_metrics = evaluate_csp_lda(args, train_clean, y_train, test_clean, y_test)
        row = base_row(
            args=args,
            classifier="csp_lda",
            classifier_seed="",
            subject=subject,
            recipe=recipe,
            seed_pair_index=seed_pair_index,
            train_seed=train_seed,
            test_seed=test_seed,
            condition="raw_raw",
            metrics=raw_metrics,
        )
        if row_key(row) not in existing_baseline_keys:
            baseline_rows.append(row)
        noisy_metrics = evaluate_csp_lda(args, train_noisy, y_train, test_noisy, y_test)
        noisy_row = base_row(
            args=args,
            classifier="csp_lda",
            classifier_seed="",
            subject=subject,
            recipe=recipe,
            seed_pair_index=seed_pair_index,
            train_seed=train_seed,
            test_seed=test_seed,
            condition="noisy_noisy",
            metrics=noisy_metrics,
        )
        if row_key(noisy_row) not in existing_baseline_keys:
            baseline_rows.append(noisy_row)
        baseline_accuracy = float(noisy_metrics["accuracy"])
        print(
            f"[baseline] classifier=csp_lda subject={subject} recipe={recipe} seed_pair={seed_pair_index} "
            f"raw={raw_metrics['accuracy']:.6f} noisy={baseline_accuracy:.6f}",
            flush=True,
        )

        for model_entry in models:
            if int(model_entry["base"]) not in csp_bases:
                continue
            key_probe = {
                "classifier": "csp_lda",
                "classifier_seed": "",
                "subject": subject,
                "recipe": recipe,
                "seed_pair_index": seed_pair_index,
                "condition": "denoised_denoised",
                "base": model_entry["base"],
                "checkpoint_seed": model_entry["checkpoint_seed"],
            }
            if row_key(key_probe) in existing_result_keys:
                continue
            train_den = denoise_epochs(model_entry["model"], train_noisy, device=device, batch_size=args.denoiser_batch_size)
            test_den = denoise_epochs(model_entry["model"], test_noisy, device=device, batch_size=args.denoiser_batch_size)
            metrics = evaluate_csp_lda(args, train_den, y_train, test_den, y_test)
            row = result_row(
                args=args,
                classifier="csp_lda",
                classifier_seed="",
                subject=subject,
                recipe=recipe,
                seed_pair_index=seed_pair_index,
                train_seed=train_seed,
                test_seed=test_seed,
                model_entry=model_entry,
                baseline_accuracy=baseline_accuracy,
                metrics=metrics,
            )
            result_rows.append(row)
            print(
                f"[result] classifier=csp_lda subject={subject} recipe={recipe} seed_pair={seed_pair_index} "
                f"base={model_entry['base']} checkpoint_seed={model_entry['checkpoint_seed']} "
                f"acc={row['accuracy']:.6f} delta={row['delta_accuracy_vs_noisy_noisy']:+.6f}",
                flush=True,
            )

    neural_classifiers = [classifier for classifier in classifiers if classifier != "csp_lda"]
    for classifier in neural_classifiers:
        for classifier_seed in classifier_seeds:
            raw_metrics = evaluate_neural(
                args,
                classifier_name=classifier,
                classifier_seed=classifier_seed,
                train_epochs=train_clean,
                y_train=y_train,
                test_epochs=test_clean,
                y_test=y_test,
                device=device,
            )
            row = base_row(
                args=args,
                classifier=classifier,
                classifier_seed=classifier_seed,
                subject=subject,
                recipe=recipe,
                seed_pair_index=seed_pair_index,
                train_seed=train_seed,
                test_seed=test_seed,
                condition="raw_raw",
                metrics=raw_metrics,
            )
            if row_key(row) not in existing_baseline_keys:
                baseline_rows.append(row)
            noisy_metrics = evaluate_neural(
                args,
                classifier_name=classifier,
                classifier_seed=classifier_seed,
                train_epochs=train_noisy,
                y_train=y_train,
                test_epochs=test_noisy,
                y_test=y_test,
                device=device,
            )
            noisy_row = base_row(
                args=args,
                classifier=classifier,
                classifier_seed=classifier_seed,
                subject=subject,
                recipe=recipe,
                seed_pair_index=seed_pair_index,
                train_seed=train_seed,
                test_seed=test_seed,
                condition="noisy_noisy",
                metrics=noisy_metrics,
            )
            if row_key(noisy_row) not in existing_baseline_keys:
                baseline_rows.append(noisy_row)
            baseline_accuracy = float(noisy_metrics["accuracy"])
            print(
                f"[baseline] classifier={classifier} classifier_seed={classifier_seed} subject={subject} "
                f"recipe={recipe} seed_pair={seed_pair_index} raw={raw_metrics['accuracy']:.6f} noisy={baseline_accuracy:.6f}",
                flush=True,
            )

            for model_entry in models:
                if int(model_entry["base"]) not in neural_bases:
                    continue
                key_probe = {
                    "classifier": classifier,
                    "classifier_seed": classifier_seed,
                    "subject": subject,
                    "recipe": recipe,
                    "seed_pair_index": seed_pair_index,
                    "condition": "denoised_denoised",
                    "base": model_entry["base"],
                    "checkpoint_seed": model_entry["checkpoint_seed"],
                }
                if row_key(key_probe) in existing_result_keys:
                    continue
                train_den = denoise_epochs(model_entry["model"], train_noisy, device=device, batch_size=args.denoiser_batch_size)
                test_den = denoise_epochs(model_entry["model"], test_noisy, device=device, batch_size=args.denoiser_batch_size)
                metrics = evaluate_neural(
                    args,
                    classifier_name=classifier,
                    classifier_seed=classifier_seed,
                    train_epochs=train_den,
                    y_train=y_train,
                    test_epochs=test_den,
                    y_test=y_test,
                    device=device,
                )
                row = result_row(
                    args=args,
                    classifier=classifier,
                    classifier_seed=classifier_seed,
                    subject=subject,
                    recipe=recipe,
                    seed_pair_index=seed_pair_index,
                    train_seed=train_seed,
                    test_seed=test_seed,
                    model_entry=model_entry,
                    baseline_accuracy=baseline_accuracy,
                    metrics=metrics,
                )
                result_rows.append(row)
                print(
                    f"[result] classifier={classifier} classifier_seed={classifier_seed} subject={subject} "
                    f"recipe={recipe} seed_pair={seed_pair_index} base={model_entry['base']} "
                    f"checkpoint_seed={model_entry['checkpoint_seed']} acc={row['accuracy']:.6f} "
                    f"delta={row['delta_accuracy_vs_noisy_noisy']:+.6f}",
                    flush=True,
                )

    return {
        "subject": subject,
        "recipe": recipe,
        "seed_pair_index": seed_pair_index,
        "train_contamination_seed": train_seed,
        "test_contamination_seed": test_seed,
        "train_noise": train_noise,
        "test_noise": test_noise,
        "train_source": info["train"]["sources"],
        "test_source": info["test"]["sources"],
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.n_bootstrap < 10000:
        raise ValueError("--n-bootstrap must be at least 10000")

    subjects = parse_subjects(args.subjects)
    recipes = parse_recipes(args.recipes)
    classifiers = parse_csv_list(args.classifiers)
    unknown_classifiers = sorted(set(classifiers) - set(CLASSIFIERS))
    if unknown_classifiers:
        raise ValueError(f"Unknown classifiers: {unknown_classifiers}; valid={CLASSIFIERS}")
    train_seeds = parse_int_list(args.train_seeds, name="train-seeds")
    test_seeds = parse_int_list(args.test_seeds, name="test-seeds")
    checkpoint_seeds = set(parse_int_list(args.checkpoint_seeds, name="checkpoint-seeds"))
    classifier_seeds = parse_int_list(args.classifier_seeds, name="classifier_seeds")
    csp_bases = set(parse_int_list(args.csp_bases, name="csp-bases")) if "csp_lda" in classifiers else set()
    neural_bases = set(parse_int_list(args.neural_bases, name="neural-bases")) if any(classifier != "csp_lda" for classifier in classifiers) else set()
    if len(train_seeds) != len(test_seeds):
        raise ValueError("--train-seeds and --test-seeds must have the same length")

    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    requested_bases = csp_bases | neural_bases
    checkpoints = expand_checkpoint_patterns(args.checkpoint_glob, bases=requested_bases, seeds=checkpoint_seeds)
    models = load_models(checkpoints, device)
    emg_pool = load_emg_pool(args.emg_pool)
    print(
        f"[start] run_id={args.run_id} subjects={subjects} recipes={recipes} classifiers={classifiers} "
        f"seed_pairs={list(zip(train_seeds, test_seeds))} csp_bases={sorted(csp_bases)} "
        f"neural_bases={sorted(neural_bases)} checkpoint_seeds={sorted(checkpoint_seeds)} "
        f"models={len(models)} emg_pool={emg_pool.shape} device={device}",
        flush=True,
    )

    baseline_path = args.output_dir / "IV-2b_bci2b_baseline_seed_rows.csv"
    result_path = args.output_dir / "IV-2b_bci2b_checkpoint_seed_rows.csv"
    noise_path = args.output_dir / "IV-2b_bci2b_noise_metadata_rows.csv"
    baseline_rows: list[dict[str, Any] | dict[str, str]] = read_csv(baseline_path) if args.resume else []
    result_rows: list[dict[str, Any] | dict[str, str]] = read_csv(result_path) if args.resume else []
    noise_rows: list[dict[str, Any] | dict[str, str]] = read_csv(noise_path) if args.resume else []

    subject_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]] = {}
    subject_info: list[dict[str, Any]] = []
    for subject in subjects:
        print(f"[load_subject] {subject}", flush=True)
        subject_cache[subject] = load_subject_clean(args, subject)
        subject_info.append({"subject": subject, "train": subject_cache[subject][-1]["train"], "test": subject_cache[subject][-1]["test"]})

    for recipe in recipes:
        for subject in subjects:
            subject_data = subject_cache[subject]
            for seed_pair_index, (train_seed, test_seed) in enumerate(zip(train_seeds, test_seeds), start=1):
                print(
                    f"[seed_pair_start] subject={subject} recipe={recipe} seed_pair={seed_pair_index} "
                    f"train_seed={train_seed} test_seed={test_seed}",
                    flush=True,
                )
                noise_info = evaluate_subject_seed_pair(
                    args,
                    subject=subject,
                    recipe=recipe,
                    seed_pair_index=seed_pair_index,
                    train_seed=train_seed,
                    test_seed=test_seed,
                    subject_data=subject_data,
                    emg_pool=emg_pool,
                    models=models,
                    classifiers=classifiers,
                    csp_bases=csp_bases,
                    neural_bases=neural_bases,
                    classifier_seeds=classifier_seeds,
                    baseline_rows=baseline_rows,
                    result_rows=result_rows,
                    device=device,
                )
                noise_rows.append(noise_info)
                write_csv(baseline_path, [dict(row) for row in baseline_rows])
                write_csv(result_path, [dict(row) for row in result_rows])
                write_csv(noise_path, [dict(row) for row in noise_rows])
                print(
                    f"[seed_pair_done] subject={subject} recipe={recipe} seed_pair={seed_pair_index} "
                    f"baseline_rows={len(baseline_rows)} result_rows={len(result_rows)}",
                    flush=True,
                )

    subject_rows = summarize_subjects(baseline_rows, result_rows)
    width_rows = inference_rows(subject_rows, n_bootstrap=args.n_bootstrap, bootstrap_seed=args.bootstrap_seed)
    write_csv(args.output_dir / "IV-2b_bci2b_subject_width_summary.csv", subject_rows)
    write_csv(args.output_dir / "IV-2b_bci2b_width_inference.csv", width_rows)
    summary = {
        "info": {
            "run_id": args.run_id,
            "dataset": "BCI_IV2b",
            "subjects": subjects,
            "recipes": recipes,
            "classifiers": classifiers,
            "protocol": "train on labeled 01T+02T sessions; test on labeled 03T session; E sessions not used because local labels are unknown",
            "eeg_channels": ["C3", "Cz", "C4"],
            "trial_window_reference": "cue_event",
            "trial_start_sec_after_cue": args.trial_start_sec,
            "trial_stop_sec_after_cue": args.trial_stop_sec,
            "train_contamination_seeds": train_seeds,
            "test_contamination_seeds": test_seeds,
            "checkpoint_seeds": sorted(checkpoint_seeds),
            "classifier_seeds": classifier_seeds,
            "csp_bases": sorted(csp_bases),
            "neural_bases": sorted(neural_bases),
            "snr_min_db": args.snr_min_db,
            "snr_max_db": args.snr_max_db,
            "inferential_unit": "subject",
            "n_bootstrap_subject_resamples": args.n_bootstrap,
        },
        "subject_info": subject_info,
        "width_inference": width_rows,
    }
    (args.output_dir / "IV-2b_bci2b_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(
        args.output_dir / "IV-2b_bci2b_summary.md",
        args=args,
        subjects=subjects,
        recipes=recipes,
        classifiers=classifiers,
        width_rows=width_rows,
    )
    print(f"[written] {args.output_dir / 'IV-2b_bci2b_summary.md'}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
