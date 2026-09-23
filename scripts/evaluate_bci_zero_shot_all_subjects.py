#!/usr/bin/env python3
"""zero-shot all-subject BCI zero-shot reconstruction audit."""

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
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.io import loadmat  # noqa: E402
from scipy.signal import butter, filtfilt, resample_poly  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eeg_denoise_benchmark.eval.metrics import compute_denoising_metrics  # noqa: E402
from eeg_denoise_benchmark.models import TinyDenoiser, count_trainable_parameters  # noqa: E402


FS_TARGET = 256
SEG_SEC = 2.0
LENGTH = int(FS_TARGET * SEG_SEC)
METRIC_KEYS = ["CC", "MSE", "RMSE", "T_RRMSE", "S_RRMSE", "SDR", "PSD_KLD", "PSD_WD"]
PRIMARY_METRICS = ["CC", "SDR", "T_RRMSE", "S_RRMSE"]
HIGHER_BETTER = {"CC", "SDR"}
WIDTHS = [2, 4, 6, 8, 16]
ADJACENT = [(2, 4), (4, 6), (6, 8), (8, 16)]
BCI2A_EEG_CHANNELS = {"C3": 7, "Cz": 9, "C4": 11}
BCI2A_EOG_CHANNELS = {"EOG-left": 22, "EOG-central": 23, "EOG-right": 24}


@dataclass(frozen=True)
class ContinuousRecording:
    source: str
    data: np.ndarray
    fs: float
    ch_names: tuple[str, ...]


@dataclass(frozen=True)
class ChannelPayload:
    dataset: str
    subject: str
    channel: str
    noisy: np.ndarray
    clean: np.ndarray
    info: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bci2a-dir", type=Path, required=True)
    parser.add_argument("--bci2b-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-glob", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-test-per-channel", type=int, default=500)
    parser.add_argument("--min-windows-per-pool", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--snr-min-db", type=float, default=-6.0)
    parser.add_argument("--snr-max-db", type=float, default=2.0)
    parser.add_argument("--low-quantile", type=float, default=0.20)
    parser.add_argument("--high-quantile", type=float, default=0.90)
    parser.add_argument("--bci2a-subjects", nargs="*", default=None)
    parser.add_argument("--bci2b-subjects", nargs="*", default=None)
    parser.add_argument("--bci2a-eeg-channels", nargs="+", default=["C3", "Cz", "C4"])
    parser.add_argument("--bci2b-eeg-channels", nargs="+", default=["C3", "Cz", "C4"])
    parser.add_argument("--bci2b-eog-substr", default="EOG")
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260815)
    return parser.parse_args()


def natural_key(path: Path) -> tuple[int, int, str]:
    base_match = re.search(r"base(\d+)", str(path))
    seed_match = re.search(r"seed(\d+)", str(path))
    base = int(base_match.group(1)) if base_match else 10**9
    seed = int(seed_match.group(1)) if seed_match else 10**9
    return base, seed, str(path)


def expand_checkpoints(pattern: str) -> list[Path]:
    checkpoints = sorted((Path(match) for match in glob.glob(pattern)), key=natural_key)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints matched: {pattern}")
    return checkpoints


def stable_int(text: str) -> int:
    total = 0
    for char in text:
        total = (total * 131 + ord(char)) % 2_147_483_647
    return total


def _mat_struct_to_dict(obj: Any) -> dict[str, Any]:
    return {name: getattr(obj, name) for name in getattr(obj, "_fieldnames", [])}


def load_bci2a_mat(mat_path: Path) -> ContinuousRecording:
    mat = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    if "data" not in mat:
        raise KeyError(f"'data' not found in {mat_path}; keys={list(mat)}")
    data = mat["data"]
    runs = data if isinstance(data, (list, np.ndarray)) else [data]
    arrays: list[np.ndarray] = []
    fs: int | None = None
    for run in runs:
        if hasattr(run, "_fieldnames"):
            fields = _mat_struct_to_dict(run)
        elif isinstance(run, dict):
            fields = run
        else:
            continue
        if "X" in fields:
            arrays.append(np.asarray(fields["X"], dtype=np.float32))
        if fs is None and "fs" in fields:
            fs = int(np.asarray(fields["fs"]).item())
    if not arrays:
        raise ValueError(f"Could not find field X in {mat_path}")
    data_arr = np.concatenate(arrays, axis=0)
    data_arr = np.nan_to_num(data_arr, copy=False).astype(np.float32)
    ch_names = tuple([f"EEG{idx:02d}" for idx in range(22)] + ["EOG-left", "EOG-central", "EOG-right"])
    return ContinuousRecording(source=str(mat_path), data=data_arr, fs=float(fs or 250), ch_names=ch_names)


def load_bci2b_gdf(gdf_path: Path) -> ContinuousRecording:
    try:
        import mne
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("mne is required for BCI IV-2b .gdf evaluation") from exc
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = mne.io.read_raw_gdf(str(gdf_path), preload=True, verbose=False)
    data_uv = np.nan_to_num(raw.get_data() * 1e6).T.astype(np.float32)
    return ContinuousRecording(
        source=str(gdf_path),
        data=data_uv,
        fs=float(raw.info["sfreq"]),
        ch_names=tuple(raw.ch_names),
    )


def resample_to_target(x: np.ndarray, fs_in: float, fs_out: int = FS_TARGET) -> np.ndarray:
    fs_in_i = int(round(fs_in))
    if fs_in_i == fs_out:
        return x.astype(np.float32)
    gcd = math.gcd(fs_out, fs_in_i)
    return resample_poly(x, fs_out // gcd, fs_in_i // gcd).astype(np.float32)


def bandpass(x: np.ndarray, fs: int = FS_TARGET, lo: float = 0.5, hi: float = 10.0, order: int = 4) -> np.ndarray:
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x).astype(np.float32)


def normalize_name(name: str) -> str:
    return name.strip().replace("EEG-", "").replace("EEG:", "").replace(" ", "").upper()


def channel_index_by_name(ch_names: tuple[str, ...], wanted: str) -> int:
    normalized = [normalize_name(name) for name in ch_names]
    target = normalize_name(wanted)
    if target in normalized:
        return normalized.index(target)
    raise ValueError(f"Channel {wanted!r} not found; available={list(ch_names)}")


def eog_indices_by_substr(ch_names: tuple[str, ...], substr: str) -> list[int]:
    target = substr.upper()
    indices = [idx for idx, name in enumerate(ch_names) if target in normalize_name(name)]
    if not indices:
        raise ValueError(f"No EOG channels found with substring {substr!r}; available={list(ch_names)}")
    return indices


def subject_from_path(dataset: str, path: Path) -> str:
    if dataset == "BCI_IV2a":
        match = re.search(r"(A\d{2})[TE]\.mat$", path.name)
    else:
        match = re.search(r"(B\d{2})\d{2}[TE]\.gdf$", path.name)
    if not match:
        raise ValueError(f"Could not infer subject from {path}")
    return match.group(1)


def discover_subject_files(
    *,
    dataset: str,
    root: Path,
    subjects: list[str] | None,
) -> dict[str, list[Path]]:
    pattern = "A0*[TE].mat" if dataset == "BCI_IV2a" else "B0*.gdf"
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(root.glob(pattern)):
        subject = subject_from_path(dataset, path)
        if subjects is not None and subject not in subjects:
            continue
        grouped[subject].append(path)
    if not grouped:
        raise FileNotFoundError(f"No {dataset} files found in {root}")
    return dict(sorted(grouped.items()))


def load_recordings(dataset: str, files: list[Path]) -> list[ContinuousRecording]:
    loader = load_bci2a_mat if dataset == "BCI_IV2a" else load_bci2b_gdf
    return [loader(path) for path in files]


def make_eog_composite(recording: ContinuousRecording, eog_indices: list[int]) -> np.ndarray:
    eog = recording.data[:, eog_indices].astype(np.float32)
    return np.mean(eog, axis=1).astype(np.float32)


def selected_indices_for_dataset(
    dataset: str,
    recordings: list[ContinuousRecording],
    channel: str,
    bci2b_eog_substr: str,
) -> tuple[int, list[int], str, str]:
    if dataset == "BCI_IV2a":
        eeg_index = BCI2A_EEG_CHANNELS[channel]
        eog_indices = [BCI2A_EOG_CHANNELS[name] for name in BCI2A_EOG_CHANNELS]
        return eeg_index, eog_indices, channel, "mean(EOG-left,EOG-central,EOG-right)"
    eeg_index = channel_index_by_name(recordings[0].ch_names, channel)
    eog_indices = eog_indices_by_substr(recordings[0].ch_names, bci2b_eog_substr)
    eog_names = ",".join(recordings[0].ch_names[idx] for idx in eog_indices)
    return eeg_index, eog_indices, channel, f"mean({eog_names})"


def build_channel_payload(
    *,
    dataset: str,
    subject: str,
    channel: str,
    recordings: list[ContinuousRecording],
    bci2b_eog_substr: str,
    n_test: int,
    min_windows_per_pool: int,
    seed: int,
    snr_min_db: float,
    snr_max_db: float,
    low_quantile: float,
    high_quantile: float,
) -> ChannelPayload | None:
    eeg_index, eog_indices, eeg_source, artifact_source = selected_indices_for_dataset(
        dataset,
        recordings,
        channel,
        bci2b_eog_substr,
    )
    prepared: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    scored: list[tuple[int, int, float]] = []
    for file_index, recording in enumerate(recordings):
        if eeg_index >= recording.data.shape[1] or max(eog_indices) >= recording.data.shape[1]:
            raise ValueError(f"Channel index out of range for {recording.source}: shape={recording.data.shape}")
        eeg = resample_to_target(recording.data[:, eeg_index].astype(np.float32), recording.fs)
        eog = resample_to_target(make_eog_composite(recording, eog_indices), recording.fs)
        n = min(len(eeg), len(eog))
        eeg = eeg[:n]
        eog = eog[:n]
        if n < LENGTH:
            continue
        starts = np.arange(0, n - LENGTH + 1, LENGTH, dtype=np.int64)
        eog_bp = bandpass(eog)
        for start in starts:
            segment = eog_bp[start : start + LENGTH]
            scored.append((file_index, int(start), float(np.mean(segment * segment))))
        prepared.append((eeg, eog, starts, eog_bp))

    if not scored:
        return None
    scores = np.array([item[2] for item in scored], dtype=np.float32)
    low_thr = float(np.quantile(scores, low_quantile))
    high_thr = float(np.quantile(scores, high_quantile))
    clean_pool = [(file_index, start) for file_index, start, score in scored if score <= low_thr]
    artifact_pool = [(file_index, start) for file_index, start, score in scored if score >= high_thr]
    if len(clean_pool) < min_windows_per_pool or len(artifact_pool) < min_windows_per_pool:
        print(
            f"[skip] dataset={dataset} subject={subject} channel={channel} "
            f"clean_pool={len(clean_pool)} artifact_pool={len(artifact_pool)}",
            flush=True,
        )
        return None

    n_eval = min(n_test, len(clean_pool), len(artifact_pool))
    rng = np.random.default_rng(seed + stable_int(f"{dataset}:{subject}:{channel}"))
    clean_indices = rng.choice(len(clean_pool), size=n_eval, replace=False)
    artifact_indices = rng.choice(len(artifact_pool), size=n_eval, replace=False)
    rng.shuffle(artifact_indices)

    noisy = np.zeros((n_eval, LENGTH), dtype=np.float32)
    clean = np.zeros((n_eval, LENGTH), dtype=np.float32)
    snrs = np.zeros((n_eval,), dtype=np.float32)
    sigmas = np.zeros((n_eval,), dtype=np.float32)
    for row_index, (clean_i, artifact_i) in enumerate(zip(clean_indices, artifact_indices)):
        clean_file, clean_start = clean_pool[int(clean_i)]
        artifact_file, artifact_start = artifact_pool[int(artifact_i)]
        eeg_clean = prepared[clean_file][0][clean_start : clean_start + LENGTH].copy()
        eog_artifact = prepared[artifact_file][1][artifact_start : artifact_start + LENGTH].copy()
        snr_db = float(rng.uniform(snr_min_db, snr_max_db))
        y, _artifact = mix_at_snr(eeg_clean, eog_artifact, snr_db)
        sigma_y = float(np.std(y) + 1e-8)
        noisy[row_index] = y / sigma_y
        clean[row_index] = eeg_clean / sigma_y
        snrs[row_index] = snr_db
        sigmas[row_index] = sigma_y

    return ChannelPayload(
        dataset=dataset,
        subject=subject,
        channel=channel,
        noisy=noisy,
        clean=clean,
        info={
            "dataset": dataset,
            "subject": subject,
            "channel": channel,
            "n_files": len(recordings),
            "sources": [recording.source for recording in recordings],
            "fs_target": FS_TARGET,
            "window_length_samples": LENGTH,
            "window_length_seconds": SEG_SEC,
            "hop_samples": LENGTH,
            "overlap_policy": "non_overlapping_2s_windows_hop_equals_window_length",
            "clean_artifact_pool_policy": "assumed_clean_low_EOG_energy_windows_and_artifact_high_EOG_energy_windows_are_quantile_disjoint_non_overlapping_windows",
            "target_policy": "BCI clean windows are assumed-clean/low-artifact windows, not guaranteed artifact-free neural ground truth",
            "eeg_source": eeg_source,
            "artifact_source": artifact_source,
            "n_candidate_windows": len(scored),
            "clean_pool": len(clean_pool),
            "artifact_pool": len(artifact_pool),
            "n_eval": int(n_eval),
            "low_quantile": low_quantile,
            "high_quantile": high_quantile,
            "low_energy_threshold": low_thr,
            "high_energy_threshold": high_thr,
            "snr_min_db": snr_min_db,
            "snr_max_db": snr_max_db,
            "snr_mean_db": float(np.mean(snrs)),
            "sigma_y_mean": float(np.mean(sigmas)),
            "sigma_y_std": float(np.std(sigmas)),
        },
    )


def mix_at_snr(clean: np.ndarray, artifact_template: np.ndarray, snr_db: float) -> tuple[np.ndarray, np.ndarray]:
    p_clean = float(np.mean(clean * clean) + 1e-12)
    p_artifact = float(np.mean(artifact_template * artifact_template) + 1e-12)
    scale = math.sqrt(p_clean / (p_artifact * (10 ** (snr_db / 10))))
    artifact = (scale * artifact_template).astype(np.float32)
    return (clean + artifact).astype(np.float32), artifact


def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> tuple[TinyDenoiser, dict[str, Any], int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError(f"Checkpoint does not contain model_state: {checkpoint_path}")
    cfg = dict(checkpoint.get("cfg", {}))
    model = TinyDenoiser(
        base=int(cfg.get("base", 16)),
        extra_bottleneck_blocks=int(cfg.get("extra_blocks", 2)),
        use_dwt=bool(cfg.get("use_dwt", True)),
        use_gate=bool(cfg.get("use_gate", True)),
        use_attn=bool(cfg.get("use_attn", True)),
        use_artifact_head=bool(cfg.get("use_artifact_head", True)),
        conv_block=str(cfg.get("conv_block", "ds")),
    )
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
def infer_clean(model: TinyDenoiser, noisy: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    predictions = np.zeros_like(noisy, dtype=np.float32)
    for start in range(0, len(noisy), batch_size):
        batch = torch.from_numpy(noisy[start : start + batch_size]).float().unsqueeze(1).to(device)
        output = model(batch)
        predictions[start : start + batch_size] = output[:, 0, :].detach().cpu().numpy()
    return predictions


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    preferred = [
        "run_id",
        "kind",
        "dataset",
        "subject",
        "channel",
        "base",
        "train_seed",
        "metric",
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


def weighted_metric_mean(items: list[dict[str, Any]], metric: str) -> float:
    weights = [float(item.get("n_eval", item.get("n_samples", 1))) for item in items]
    values = [float(item[metric]) for item in items]
    total = sum(weights)
    return float(sum(value * weight for value, weight in zip(values, weights)) / total) if total else float("nan")


def subject_rows(channel_rows: list[dict[str, Any]], run_id: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in channel_rows:
        base = int(row["base"]) if row["base"] != "" else -1
        grouped[(row["kind"], row["dataset"], row["subject"], base, str(row["train_seed"]))].append(row)
    out = []
    for (kind, dataset, subject, base, train_seed), items in sorted(grouped.items()):
        merged: dict[str, Any] = {
            "run_id": run_id,
            "kind": kind,
            "dataset": dataset,
            "subject": subject,
            "base": "" if base < 0 else base,
            "train_seed": train_seed,
            "n_channels": len(items),
            "channels": " ".join(item["channel"] for item in items),
            "n_eval_total": int(sum(int(item["n_eval"]) for item in items)),
        }
        if kind == "model":
            merged["checkpoint"] = items[0]["checkpoint"]
            merged["trainable_parameters"] = items[0]["trainable_parameters"]
        for metric in METRIC_KEYS:
            merged[metric] = weighted_metric_mean(items, metric)
        out.append(merged)
    return out


def subject_width_rows(rows: list[dict[str, Any]], run_id: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["kind"] == "model":
            grouped[(row["dataset"], row["subject"], int(row["base"]))].append(row)
    out = []
    for (dataset, subject, base), items in sorted(grouped.items()):
        merged = {
            "run_id": run_id,
            "dataset": dataset,
            "subject": subject,
            "base": base,
            "n_checkpoints": len(items),
            "train_seeds": " ".join(str(item["train_seed"]) for item in sorted(items, key=lambda row: int(row["train_seed"]))),
            "n_eval_total_per_checkpoint": items[0]["n_eval_total"],
        }
        for metric in METRIC_KEYS:
            merged[metric] = float(statistics.mean(float(item[metric]) for item in items))
        out.append(merged)
    return out


def bootstrap_ci(values: list[float], *, n_resamples: int, rng: np.random.Generator) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])
    indices = rng.integers(0, len(arr), size=(n_resamples, len(arr)))
    means = arr[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def aggregate_subject_stats(
    subject_width: list[dict[str, Any]],
    run_id: str,
    n_resamples: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in subject_width:
        grouped[(row["dataset"], int(row["base"]))].append(row)
    out = []
    for (dataset, base), items in sorted(grouped.items()):
        row = {
            "run_id": run_id,
            "dataset": dataset,
            "base": base,
            "n_subjects": len(items),
            "subjects": " ".join(item["subject"] for item in items),
        }
        for metric in METRIC_KEYS:
            values = [float(item[metric]) for item in items]
            ci_low, ci_high = bootstrap_ci(values, n_resamples=n_resamples, rng=rng)
            row[f"{metric}_mean"] = float(statistics.mean(values))
            row[f"{metric}_median"] = float(statistics.median(values))
            row[f"{metric}_sd"] = float(statistics.stdev(values)) if len(values) > 1 else 0.0
            row[f"{metric}_ci95_low"] = ci_low
            row[f"{metric}_ci95_high"] = ci_high
        out.append(row)
    return out


def oriented_increment(metric: str, low: float, high: float) -> float:
    return high - low if metric in HIGHER_BETTER else low - high


def diminishing_rows(subject_width: list[dict[str, Any]], run_id: str) -> list[dict[str, Any]]:
    by_key = {(row["dataset"], row["subject"], int(row["base"])): row for row in subject_width}
    datasets = sorted({row["dataset"] for row in subject_width})
    out = []
    for dataset in datasets:
        subjects = sorted({row["subject"] for row in subject_width if row["dataset"] == dataset})
        for metric in PRIMARY_METRICS:
            for low, high in ADJACENT:
                increments = []
                present_subjects = []
                for subject in subjects:
                    low_row = by_key.get((dataset, subject, low))
                    high_row = by_key.get((dataset, subject, high))
                    if low_row is None or high_row is None:
                        continue
                    increments.append(oriented_increment(metric, float(low_row[metric]), float(high_row[metric])))
                    present_subjects.append(subject)
                if not increments:
                    continue
                out.append(
                    {
                        "run_id": run_id,
                        "dataset": dataset,
                        "metric": metric,
                        "comparison": f"base{low}_to_base{high}",
                        "base_low": low,
                        "base_high": high,
                        "n_subjects": len(increments),
                        "subjects": " ".join(present_subjects),
                        "mean_incremental_improvement": float(statistics.mean(increments)),
                        "median_incremental_improvement": float(statistics.median(increments)),
                        "sd_incremental_improvement": float(statistics.stdev(increments)) if len(increments) > 1 else 0.0,
                        "n_positive_subjects": int(sum(value > 0 for value in increments)),
                        "n_negative_subjects": int(sum(value < 0 for value in increments)),
                        "subject_increments": " ".join(f"{subject}:{value:+.6f}" for subject, value in zip(present_subjects, increments)),
                    }
                )
    return out


def plot_subject_points(subject_width: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    plot_dir = output_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for dataset in sorted({row["dataset"] for row in subject_width}):
        for metric in PRIMARY_METRICS:
            rows = [row for row in subject_width if row["dataset"] == dataset]
            fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=160)
            subjects = sorted({row["subject"] for row in rows})
            for subject in subjects:
                items = sorted([row for row in rows if row["subject"] == subject], key=lambda row: int(row["base"]))
                xs = [int(row["base"]) for row in items]
                ys = [float(row[metric]) for row in items]
                ax.plot(xs, ys, color="#9ca3af", alpha=0.55, linewidth=0.9)
                ax.scatter(xs, ys, s=22, color="#111827", alpha=0.75)
            by_base: dict[int, list[float]] = defaultdict(list)
            for row in rows:
                by_base[int(row["base"])].append(float(row[metric]))
            xs = sorted(by_base)
            means = [float(statistics.mean(by_base[x])) for x in xs]
            ax.plot(xs, means, color="#dc2626", linewidth=2.2, marker="o", label="subject mean")
            ax.set_xticks(WIDTHS)
            ax.set_xlabel("Base width")
            ax.set_ylabel(metric)
            ax.set_title(f"zero-shot {dataset} subject-level zero-shot {metric}")
            ax.grid(True, color="#e5e7eb", linewidth=0.8)
            ax.legend(frameon=False)
            fig.tight_layout()
            path = plot_dir / f"zero-shot_{dataset}_{metric}_subject_points.png"
            fig.savefig(path)
            plt.close(fig)
            paths.append(path)
    return paths


def write_markdown(
    path: Path,
    *,
    run_id: str,
    channel_payloads: list[ChannelPayload],
    aggregate_stats: list[dict[str, Any]],
    diminishing: list[dict[str, Any]],
    figures: list[Path],
) -> None:
    lines = ["# zero-shot All-Subject BCI Zero-Shot Reconstruction", ""]
    lines.append(f"Run id: `{run_id}`")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append("- BCI windows are treated as assumed-clean/low-artifact targets, not guaranteed artifact-free neural ground truth.")
    lines.append("- Evaluation windows are non-overlapping 2 s windows at 256 Hz; hop equals the 512-sample window length.")
    lines.append("- Clean and artifact pools are quantile-disjoint within subject/channel, so the same time window cannot contribute to both pools.")
    lines.append("- Subject identity is preserved in all summary statistics.")
    lines.append("- Fixed EEG channels are used across widths: `C3`, `Cz`, and `C4`.")
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    for dataset in sorted({payload.dataset for payload in channel_payloads}):
        subjects = sorted({payload.subject for payload in channel_payloads if payload.dataset == dataset})
        lines.append(f"- {dataset}: `{len(subjects)}` subjects, subjects `{subjects}`.")
    lines.append("")
    lines.append("## Subject-Level Width Summaries")
    lines.append("")
    lines.append("| Dataset | Base | n subjects | CC mean [CI] | SDR mean [CI] | T_RRMSE mean [CI] | S_RRMSE mean [CI] |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in aggregate_stats:
        lines.append(
            f"| {row['dataset']} | {row['base']} | {row['n_subjects']} | "
            f"{row['CC_mean']:.6f} [{row['CC_ci95_low']:.6f}, {row['CC_ci95_high']:.6f}] | "
            f"{row['SDR_mean']:.6f} [{row['SDR_ci95_low']:.6f}, {row['SDR_ci95_high']:.6f}] | "
            f"{row['T_RRMSE_mean']:.6f} [{row['T_RRMSE_ci95_low']:.6f}, {row['T_RRMSE_ci95_high']:.6f}] | "
            f"{row['S_RRMSE_mean']:.6f} [{row['S_RRMSE_ci95_low']:.6f}, {row['S_RRMSE_ci95_high']:.6f}] |"
        )
    lines.append("")
    lines.append("## Adjacent-Width Subject Heterogeneity")
    lines.append("")
    lines.append("| Dataset | Metric | Comparison | Mean increment | Median increment | Positive subjects | Negative subjects |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for row in diminishing:
        lines.append(
            f"| {row['dataset']} | {row['metric']} | {row['comparison']} | "
            f"{row['mean_incremental_improvement']:+.6f} | {row['median_incremental_improvement']:+.6f} | "
            f"{row['n_positive_subjects']}/{row['n_subjects']} | {row['n_negative_subjects']}/{row['n_subjects']} |"
        )
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    for figure in figures:
        lines.append(f"- `{figure}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    checkpoints = expand_checkpoints(args.checkpoint_glob)

    print(f"[start] run_id={args.run_id} device={device}", flush=True)
    print(f"[checkpoints] {len(checkpoints)}", flush=True)
    for checkpoint in checkpoints:
        print(f"  - {checkpoint}", flush=True)

    subject_files = {
        "BCI_IV2a": discover_subject_files(dataset="BCI_IV2a", root=args.bci2a_dir, subjects=args.bci2a_subjects),
        "BCI_IV2b": discover_subject_files(dataset="BCI_IV2b", root=args.bci2b_dir, subjects=args.bci2b_subjects),
    }

    payloads: list[ChannelPayload] = []
    dataset_info: list[dict[str, Any]] = []
    for dataset, grouped_files in subject_files.items():
        channels = args.bci2a_eeg_channels if dataset == "BCI_IV2a" else args.bci2b_eeg_channels
        for subject, files in grouped_files.items():
            print(f"[load_subject] dataset={dataset} subject={subject} n_files={len(files)}", flush=True)
            recordings = load_recordings(dataset, files)
            for channel in channels:
                payload = build_channel_payload(
                    dataset=dataset,
                    subject=subject,
                    channel=channel,
                    recordings=recordings,
                    bci2b_eog_substr=args.bci2b_eog_substr,
                    n_test=args.n_test_per_channel,
                    min_windows_per_pool=args.min_windows_per_pool,
                    seed=args.seed,
                    snr_min_db=args.snr_min_db,
                    snr_max_db=args.snr_max_db,
                    low_quantile=args.low_quantile,
                    high_quantile=args.high_quantile,
                )
                if payload is None:
                    continue
                payloads.append(payload)
                dataset_info.append(payload.info)
                print(
                    f"[payload] dataset={dataset} subject={subject} channel={channel} "
                    f"n_eval={payload.info['n_eval']} clean_pool={payload.info['clean_pool']} "
                    f"artifact_pool={payload.info['artifact_pool']}",
                    flush=True,
                )

    channel_rows: list[dict[str, Any]] = []
    for payload in payloads:
        baseline = compute_denoising_metrics(payload.clean, payload.noisy, fs=FS_TARGET)
        channel_rows.append(
            {
                "run_id": args.run_id,
                "kind": "noisy_baseline",
                "dataset": payload.dataset,
                "subject": payload.subject,
                "channel": payload.channel,
                "base": "",
                "train_seed": "",
                "checkpoint": "",
                "trainable_parameters": "",
                "n_eval": payload.info["n_eval"],
                **{key: baseline[key] for key in METRIC_KEYS},
            }
        )

    for checkpoint_path in checkpoints:
        model, cfg, n_params = load_checkpoint_model(checkpoint_path, device)
        base = int(cfg.get("base", -1))
        train_seed = checkpoint_seed(checkpoint_path, cfg)
        print(f"[eval_start] base={base} seed={train_seed} params={n_params}", flush=True)
        for payload in payloads:
            prediction = infer_clean(model, payload.noisy, device=device, batch_size=args.batch_size)
            metrics = compute_denoising_metrics(payload.clean, prediction, fs=FS_TARGET)
            channel_rows.append(
                {
                    "run_id": args.run_id,
                    "kind": "model",
                    "dataset": payload.dataset,
                    "subject": payload.subject,
                    "channel": payload.channel,
                    "base": base,
                    "train_seed": train_seed,
                    "checkpoint": str(checkpoint_path),
                    "trainable_parameters": n_params,
                    "n_eval": payload.info["n_eval"],
                    "use_dwt": bool(cfg.get("use_dwt", True)),
                    "use_attn": bool(cfg.get("use_attn", True)),
                    "use_gate": bool(cfg.get("use_gate", True)),
                    "use_artifact_head": bool(cfg.get("use_artifact_head", True)),
                    **{key: metrics[key] for key in METRIC_KEYS},
                }
            )
        print(f"[eval_done] base={base} seed={train_seed}", flush=True)

    subj_rows = subject_rows(channel_rows, args.run_id)
    subj_width = subject_width_rows(subj_rows, args.run_id)
    aggregate_stats = aggregate_subject_stats(
        subj_width,
        args.run_id,
        n_resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )
    diminishing = diminishing_rows(subj_width, args.run_id)
    figures = plot_subject_points(subj_width, args.output_dir)

    write_csv(args.output_dir / "zero-shot_channel_reconstruction_rows.csv", channel_rows)
    write_csv(args.output_dir / "zero-shot_subject_checkpoint_rows.csv", subj_rows)
    write_csv(args.output_dir / "zero-shot_subject_width_rows.csv", subj_width)
    write_csv(args.output_dir / "zero-shot_subject_width_summary.csv", aggregate_stats)
    write_csv(args.output_dir / "zero-shot_adjacent_width_subject_heterogeneity.csv", diminishing)
    write_csv(args.output_dir / "zero-shot_dataset_window_pool_info.csv", dataset_info)
    summary = {
        "run_id": args.run_id,
        "protocol": {
            "target_policy": "BCI clean windows are assumed-clean/low-artifact windows, not guaranteed artifact-free neural ground truth.",
            "window_policy": "Non-overlapping 2 s windows; hop equals window length.",
            "pool_policy": "Clean and artifact pools are low/high EOG-energy quantile-disjoint within subject/channel.",
            "fixed_channels": {"BCI_IV2a": args.bci2a_eeg_channels, "BCI_IV2b": args.bci2b_eeg_channels},
            "snr_min_db": args.snr_min_db,
            "snr_max_db": args.snr_max_db,
            "n_test_per_channel": args.n_test_per_channel,
            "bootstrap_resamples": args.bootstrap_resamples,
        },
        "n_payloads": len(payloads),
        "n_channel_rows": len(channel_rows),
        "n_subject_checkpoint_rows": len(subj_rows),
        "n_subject_width_rows": len(subj_width),
        "n_aggregate_rows": len(aggregate_stats),
        "n_diminishing_rows": len(diminishing),
        "figures": [str(path) for path in figures],
    }
    (args.output_dir / "zero-shot_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(
        args.output_dir / "zero-shot_summary.md",
        run_id=args.run_id,
        channel_payloads=payloads,
        aggregate_stats=aggregate_stats,
        diminishing=diminishing,
        figures=figures,
    )
    print(f"[done] payloads={len(payloads)} channel_rows={len(channel_rows)} subject_width_rows={len(subj_width)}", flush=True)


if __name__ == "__main__":
    main()
