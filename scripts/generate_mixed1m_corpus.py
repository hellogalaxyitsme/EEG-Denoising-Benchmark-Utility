#!/usr/bin/env python3
"""Generate the Mixed-1M synthetic artifact corpus from EEGDenoiseNet source pools."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import io, signal


RECIPE_NAMES = [
    "EOG",
    "EMG",
    "EOG+EMG",
    "EMG+LINE",
    "EOG+LINE",
    "EOG+EMG+LINE",
    "EOG+EMG+LINE+ECG",
]
RECIPE_PROBABILITIES = np.asarray([0.25, 0.25, 0.20, 0.10, 0.10, 0.05, 0.05], dtype=np.float64)
RECIPE_COMPONENTS = [
    ("EOG",),
    ("EMG",),
    ("EOG", "EMG"),
    ("EMG", "LINE"),
    ("EOG", "LINE"),
    ("EOG", "EMG", "LINE"),
    ("EOG", "EMG", "LINE", "ECG"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-eeg", type=Path, required=True, help="Clean EEG source pool (.npy, .npz, or .mat).")
    parser.add_argument("--eog", type=Path, required=True, help="EOG artifact source pool (.npy, .npz, or .mat).")
    parser.add_argument("--emg", type=Path, required=True, help="EMG artifact source pool (.npy, .npz, or .mat).")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clean-key", default="auto")
    parser.add_argument("--eog-key", default="auto")
    parser.add_argument("--emg-key", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fs", type=float, default=256.0)
    parser.add_argument("--emg-fs", type=float, default=512.0)
    parser.add_argument("--segment-length", type=int, default=512)
    parser.add_argument("--n-train", type=int, default=800_000)
    parser.add_argument("--n-val", type=int, default=100_000)
    parser.add_argument("--n-test", type=int, default=100_000)
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_array(path: Path, key: str, preferred_lengths: tuple[int, ...]) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
    elif path.suffix == ".npz":
        z = np.load(path, allow_pickle=False)
        arr = z[choose_key(z.files, z, key)]
    elif path.suffix == ".mat":
        mat = io.loadmat(path)
        keys = [name for name in mat if not name.startswith("__")]
        arr = mat[choose_key(keys, mat, key)]
    else:
        raise ValueError(f"Unsupported source format: {path}")
    arr = np.asarray(arr, dtype=np.float32).squeeze()
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D source array in {path}, got {arr.shape}")
    return orient_segments(arr, preferred_lengths)


def choose_key(keys: list[str], container: Any, requested: str) -> str:
    if requested != "auto":
        if requested not in keys:
            raise KeyError(f"Requested key '{requested}' not found. Available keys: {keys}")
        return requested
    candidates: list[tuple[int, str]] = []
    for name in keys:
        value = np.asarray(container[name])
        if np.issubdtype(value.dtype, np.number) and value.squeeze().ndim == 2:
            candidates.append((int(value.size), name))
    if not candidates:
        raise KeyError(f"No 2D numeric array found. Available keys: {keys}")
    return max(candidates)[1]


def orient_segments(arr: np.ndarray, preferred_lengths: tuple[int, ...]) -> np.ndarray:
    if arr.shape[1] in preferred_lengths:
        return np.ascontiguousarray(arr, dtype=np.float32)
    if arr.shape[0] in preferred_lengths:
        return np.ascontiguousarray(arr.T, dtype=np.float32)
    raise ValueError(f"Could not infer segment axis for shape {arr.shape}; expected one axis in {preferred_lengths}")


def source_split_indices(n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    indices = rng.permutation(n)
    n_train = int(round(0.8 * n))
    n_val = int(round(0.1 * n))
    return {
        "train": np.sort(indices[:n_train]),
        "val": np.sort(indices[n_train : n_train + n_val]),
        "test": np.sort(indices[n_train + n_val :]),
    }


def crop_or_pad(x: np.ndarray, length: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).squeeze()
    if x.shape[0] == length:
        return x
    if x.shape[0] > length:
        start = (x.shape[0] - length) // 2
        return x[start : start + length]
    out = np.zeros(length, dtype=np.float32)
    start = (length - x.shape[0]) // 2
    out[start : start + x.shape[0]] = x
    return out


def standardize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x - float(np.mean(x))) / (float(np.std(x)) + eps)


def resample_if_needed(x: np.ndarray, source_fs: float, target_fs: float, target_length: int) -> np.ndarray:
    if abs(source_fs - target_fs) < 1e-6:
        return crop_or_pad(x, target_length)
    scale = 1000
    up = int(round(target_fs * scale))
    down = int(round(source_fs * scale))
    divisor = math.gcd(up, down)
    y = signal.resample_poly(x, up // divisor, down // divisor).astype(np.float32)
    return crop_or_pad(y, target_length)


def line_noise(length: int, fs: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(length, dtype=np.float32) / float(fs)
    base = 50.0 if rng.random() < 0.85 else 60.0
    phase = rng.uniform(0.0, 2.0 * np.pi)
    slow_phase = rng.uniform(0.0, 2.0 * np.pi)
    envelope = 1.0 + 0.20 * np.sin(2.0 * np.pi * rng.uniform(0.05, 0.35) * t + slow_phase)
    y = np.sin(2.0 * np.pi * base * t + phase)
    if 2.0 * base < fs / 2.0:
        y += 0.35 * np.sin(2.0 * np.pi * 2.0 * base * t + rng.uniform(0.0, 2.0 * np.pi))
    return standardize(envelope * y)


def synthetic_ecg(length: int, fs: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(length, dtype=np.float32) / float(fs)
    heart_rate = rng.uniform(55.0, 95.0)
    rr = 60.0 / heart_rate
    first = rng.uniform(0.0, rr)
    y = np.zeros(length, dtype=np.float32)
    center = first
    while center < t[-1] + rr:
        qrs_width = rng.uniform(0.018, 0.035)
        y += np.exp(-0.5 * ((t - center) / qrs_width) ** 2).astype(np.float32)
        if rng.random() < 0.75:
            t_center = center + rng.uniform(0.18, 0.32)
            t_width = rng.uniform(0.055, 0.095)
            y += (0.20 * rng.uniform(0.7, 1.3) * np.exp(-0.5 * ((t - t_center) / t_width) ** 2)).astype(np.float32)
        center += rr * rng.uniform(0.92, 1.08)
    return standardize(y)


def electrode_noise(length: int, fs: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(length, dtype=np.float32) / float(fs)
    drift = np.sin(2.0 * np.pi * rng.uniform(0.05, 0.45) * t + rng.uniform(0.0, 2.0 * np.pi))
    burst = np.zeros(length, dtype=np.float32)
    if rng.random() < 0.65:
        center = rng.integers(0, length)
        width = rng.uniform(4.0, 18.0)
        idx = np.arange(length, dtype=np.float32)
        burst = rng.choice([-1.0, 1.0]) * np.exp(-0.5 * ((idx - float(center)) / width) ** 2).astype(np.float32)
    hiss = rng.normal(0.0, 0.15, size=length).astype(np.float32)
    return standardize(0.6 * drift + 0.8 * burst + hiss)


def target_snr(rng: np.random.Generator) -> float:
    if rng.random() < 0.70:
        return float(rng.uniform(-7.0, 2.0))
    return float(rng.uniform(-12.0, -7.0))


def scale_to_snr(clean: np.ndarray, artifact: np.ndarray, snr_db: float, eps: float = 1e-8) -> tuple[np.ndarray, float]:
    px = float(np.mean(np.square(clean)))
    pa = float(np.mean(np.square(artifact)))
    lam = math.sqrt((px + eps) / ((pa + eps) * (10.0 ** (snr_db / 10.0))))
    return (artifact * lam).astype(np.float32), float(lam)


def generate_one(
    *,
    split: str,
    clean_pool: np.ndarray,
    eog_pool: np.ndarray,
    emg_pool: np.ndarray,
    splits: dict[str, dict[str, np.ndarray]],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> dict[str, Any]:
    recipe_id = int(rng.choice(len(RECIPE_NAMES), p=RECIPE_PROBABILITIES))
    components = set(RECIPE_COMPONENTS[recipe_id])
    clean_source = int(rng.choice(splits["clean"][split]))
    eog_source = -1
    emg_source = -1
    clean = crop_or_pad(clean_pool[clean_source], args.segment_length).astype(np.float32)
    artifact = np.zeros(args.segment_length, dtype=np.float32)

    if "EOG" in components:
        eog_source = int(rng.choice(splits["eog"][split]))
        eog = crop_or_pad(eog_pool[eog_source], args.segment_length)
        if rng.random() < 0.5:
            eog = -eog
        artifact += rng.uniform(0.7, 1.3) * eog

    if "EMG" in components:
        emg_source = int(rng.choice(splits["emg"][split]))
        emg = resample_if_needed(emg_pool[emg_source], args.emg_fs, args.fs, args.segment_length)
        artifact += rng.uniform(0.6, 1.5) * emg

    line_used = int("LINE" in components)
    ecg_used = int("ECG" in components)
    if line_used:
        artifact += line_noise(args.segment_length, args.fs, rng)
    if ecg_used:
        artifact += synthetic_ecg(args.segment_length, args.fs, rng)

    elec_used = int(rng.random() < 0.35)
    if elec_used:
        artifact += 0.5 * electrode_noise(args.segment_length, args.fs, rng)

    if float(np.std(artifact)) < 1e-8:
        artifact += rng.normal(0.0, 1.0, size=args.segment_length).astype(np.float32)

    snr_db = target_snr(rng)
    scaled_artifact, lam = scale_to_snr(clean, artifact, snr_db)
    noisy = clean + scaled_artifact
    sigma_y = float(np.std(noisy) + 1e-8)

    return {
        "Y": (noisy / sigma_y).astype(np.float32),
        "X": (clean / sigma_y).astype(np.float32),
        "A": (scaled_artifact / sigma_y).astype(np.float32),
        "sigma_y": sigma_y,
        "snr_db": snr_db,
        "lambda": lam,
        "recipe_id": recipe_id,
        "eog_used": int("EOG" in components),
        "emg_used": int("EMG" in components),
        "line_used": line_used,
        "ecg_used": ecg_used,
        "elec_used": elec_used,
        "clean_source_index": clean_source,
        "eog_source_index": eog_source,
        "emg_source_index": emg_source,
    }


def write_split(
    *,
    split: str,
    n_samples: int,
    clean_pool: np.ndarray,
    eog_pool: np.ndarray,
    emg_pool: np.ndarray,
    splits: dict[str, dict[str, np.ndarray]],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> None:
    split_dir = args.output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, n_samples, args.chunk_size):
        n_chunk = min(args.chunk_size, n_samples - start)
        rows = [
            generate_one(
                split=split,
                clean_pool=clean_pool,
                eog_pool=eog_pool,
                emg_pool=emg_pool,
                splits=splits,
                args=args,
                rng=rng,
            )
            for _ in range(n_chunk)
        ]
        chunk_index = start // args.chunk_size
        out: dict[str, np.ndarray] = {}
        for key in ["Y", "X", "A"]:
            out[key] = np.stack([row[key] for row in rows]).astype(np.float32)
        for key in ["sigma_y", "snr_db", "lambda"]:
            out[key] = np.asarray([row[key] for row in rows], dtype=np.float32)
        for key in [
            "recipe_id",
            "eog_used",
            "emg_used",
            "line_used",
            "ecg_used",
            "elec_used",
            "clean_source_index",
            "eog_source_index",
            "emg_source_index",
        ]:
            out[key] = np.asarray([row[key] for row in rows], dtype=np.int32)
        path = split_dir / f"chunk_{chunk_index:04d}.npz"
        np.savez_compressed(path, **out)
        print(f"[chunk] {split} {chunk_index:04d} n={n_chunk} path={path}", flush=True)


def write_meta(
    args: argparse.Namespace,
    clean_pool: np.ndarray,
    eog_pool: np.ndarray,
    emg_pool: np.ndarray,
    splits: dict[str, dict[str, np.ndarray]],
) -> None:
    meta = {
        "name": "Mixed-1M",
        "seed": args.seed,
        "sampling_rate_hz": args.fs,
        "segment_length": args.segment_length,
        "segment_duration_s": args.segment_length / args.fs,
        "counts": {"train": args.n_train, "val": args.n_val, "test": args.n_test},
        "chunk_size": args.chunk_size,
        "source_paths": {
            "clean_eeg": str(args.clean_eeg),
            "eog": str(args.eog),
            "emg": str(args.emg),
        },
        "source_shapes": {
            "clean_eeg": list(clean_pool.shape),
            "eog": list(eog_pool.shape),
            "emg": list(emg_pool.shape),
        },
        "source_splits": {
            source: {split: values.astype(int).tolist() for split, values in source_splits.items()}
            for source, source_splits in splits.items()
        },
        "recipes": [
            {"id": idx, "name": name, "probability": float(prob), "components": list(RECIPE_COMPONENTS[idx])}
            for idx, (name, prob) in enumerate(zip(RECIPE_NAMES, RECIPE_PROBABILITIES))
        ],
        "snr_distribution": {
            "main": {"probability": 0.70, "uniform_db": [-7.0, 2.0]},
            "low_snr_tail": {"probability": 0.30, "uniform_db": [-12.0, -7.0]},
        },
        "augmentation": {
            "eog_polarity_flip_probability": 0.5,
            "eog_pre_scale_uniform": [0.7, 1.3],
            "emg_pre_scale_uniform": [0.6, 1.5],
            "electrode_noise_probability": 0.35,
            "line_frequency": {"50_hz_probability": 0.85, "60_hz_probability": 0.15},
            "normalization": "Scale artifact mixture to target SNR, form y=x+a, then normalize Y, X, and A by std(y).",
        },
    }
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def initialize_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{path} is not empty. Use --overwrite to continue.")
        for split in ["train", "val", "test"]:
            split_dir = path / split
            if split_dir.exists():
                for chunk in split_dir.glob("chunk_*.npz"):
                    chunk.unlink()
        meta_path = path / "meta.json"
        if meta_path.exists():
            meta_path.unlink()
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    initialize_output_dir(args.output_dir, args.overwrite)

    clean_pool = load_array(args.clean_eeg, args.clean_key, (args.segment_length,))
    eog_pool = load_array(args.eog, args.eog_key, (args.segment_length,))
    emg_pool = load_array(args.emg, args.emg_key, (args.segment_length, int(round(args.segment_length * args.emg_fs / args.fs))))

    rng = np.random.default_rng(args.seed)
    splits = {
        "clean": source_split_indices(clean_pool.shape[0], rng),
        "eog": source_split_indices(eog_pool.shape[0], rng),
        "emg": source_split_indices(emg_pool.shape[0], rng),
    }

    print(f"[source] clean={clean_pool.shape} eog={eog_pool.shape} emg={emg_pool.shape}")
    write_meta(args, clean_pool, eog_pool, emg_pool, splits)
    for split, count in [("train", args.n_train), ("val", args.n_val), ("test", args.n_test)]:
        write_split(
            split=split,
            n_samples=count,
            clean_pool=clean_pool,
            eog_pool=eog_pool,
            emg_pool=emg_pool,
            splits=splits,
            args=args,
            rng=rng,
        )
    print(f"[done] wrote {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise
