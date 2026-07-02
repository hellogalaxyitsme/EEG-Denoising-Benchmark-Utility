#!/usr/bin/env python3
"""Evaluate zero/few-parameter classical EEGDenoiseNet baselines."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eeg_denoise_benchmark.data import load_synthetic_split  # noqa: E402
from eeg_denoise_benchmark.eval.metrics import compute_denoising_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eog-data", type=Path, required=True)
    parser.add_argument("--emg-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--include-oracle-regression", action="store_true")
    return parser.parse_args()


def fft_filter(x: np.ndarray, fs: float, low: float | None = None, high: float | None = None) -> np.ndarray:
    freqs = np.fft.rfftfreq(x.shape[-1], d=1.0 / fs)
    mask = np.ones_like(freqs, dtype=np.float32)
    if low is not None:
        mask *= freqs >= low
    if high is not None:
        mask *= freqs <= high
    y = np.fft.irfft(np.fft.rfft(x, axis=-1) * mask[None, :], n=x.shape[-1], axis=-1)
    return y.astype(np.float32)


def haar_soft_threshold(x: np.ndarray, levels: int = 4) -> np.ndarray:
    """Small fixed Haar wavelet shrinkage baseline using only NumPy."""

    original_len = x.shape[-1]
    pad_len = int(np.ceil(original_len / (2**levels)) * (2**levels))
    if pad_len != original_len:
        x_work = np.pad(x, ((0, 0), (0, pad_len - original_len)), mode="edge")
    else:
        x_work = x.copy()

    approx = x_work.astype(np.float32)
    details: list[np.ndarray] = []
    scale = np.float32(2.0**0.5)
    for _ in range(levels):
        even = approx[:, 0::2]
        odd = approx[:, 1::2]
        details.append((even - odd) / scale)
        approx = (even + odd) / scale

    sigma = np.median(np.abs(details[0]), axis=-1, keepdims=True) / 0.6745
    thresh = sigma * np.sqrt(2.0 * np.log(x_work.shape[-1]))
    shrunk = [np.sign(d) * np.maximum(np.abs(d) - thresh, 0.0) for d in details]

    rec = approx
    for detail in reversed(shrunk):
        even = (rec + detail) / scale
        odd = (rec - detail) / scale
        merged = np.empty((x.shape[0], rec.shape[-1] * 2), dtype=np.float32)
        merged[:, 0::2] = even
        merged[:, 1::2] = odd
        rec = merged
    return rec[:, :original_len].astype(np.float32)


def oracle_artifact_regression(train_pack: dict[str, np.ndarray], test_pack: dict[str, np.ndarray]) -> np.ndarray:
    """Fit one scalar beta on train X ~= Y - beta*A and apply it to test.

    This is oracle-assisted because the benchmark artifact component A is known.
    It is useful as an upper anchor but should not be presented as a deployable
    classical baseline.
    """

    y_train = np.asarray(train_pack["Y"], dtype=np.float32)
    x_train = np.asarray(train_pack["X"], dtype=np.float32)
    a_train = np.asarray(train_pack["A"], dtype=np.float32)
    numerator = np.sum(a_train * (y_train - x_train))
    denominator = np.sum(a_train * a_train) + 1e-12
    beta = float(numerator / denominator)
    return np.asarray(test_pack["Y"], dtype=np.float32) - beta * np.asarray(test_pack["A"], dtype=np.float32)


def write_summary(
    out_dir: Path,
    *,
    run_id: str,
    task: str,
    method: str,
    data_path: Path,
    fs: float,
    prediction: np.ndarray,
    target: np.ndarray,
    trainable_parameters: int,
    notes: str,
) -> dict[str, object]:
    metrics = compute_denoising_metrics(target=target, prediction=prediction, fs=fs)
    summary: dict[str, object] = {
        "output_dir": str(out_dir),
        "run_id": run_id,
        "data": str(data_path),
        "device": "none",
        "seed": "deterministic",
        "task": task,
        "method": method,
        "model_type": "classical",
        "variant": method,
        "trainable_parameters": trainable_parameters,
        "best_epoch": "",
        "best_val_sdr": "",
        "length": int(target.shape[-1]),
        "notes": notes,
        "test": metrics,
        "config": {
            "seed": "deterministic",
            "task": task,
            "method": method,
            "fs": fs,
            "trainable_parameters": trainable_parameters,
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def task_methods(task: str, fs: float) -> list[tuple[str, int, str, Callable[[np.ndarray], np.ndarray]]]:
    methods: list[tuple[str, int, str, Callable[[np.ndarray], np.ndarray]]] = [
        ("noisy_passthrough", 0, "Contaminated signal Y used directly as prediction.", lambda y: y),
        ("bandpass_0p5_45", 0, "Fixed FFT bandpass retaining 0.5-45 Hz.", lambda y: fft_filter(y, fs, 0.5, 45.0)),
        ("haar_wavelet_soft", 0, "Fixed four-level Haar soft-thresholding.", lambda y: haar_soft_threshold(y, levels=4)),
    ]
    if task == "eog":
        methods.append(("eog_highpass_1_45", 0, "Task-specific fixed FFT high-pass/bandpass retaining 1-45 Hz.", lambda y: fft_filter(y, fs, 1.0, 45.0)))
    elif task == "emg":
        methods.append(("emg_lowpass_40", 0, "Task-specific fixed FFT low-pass retaining <=40 Hz.", lambda y: fft_filter(y, fs, None, 40.0)))
    return methods


def evaluate_task(
    *,
    task: str,
    data_path: Path,
    fs: float,
    output_dir: Path,
    run_id: str,
    split: str,
    include_oracle_regression: bool,
) -> list[dict[str, object]]:
    test_pack = load_synthetic_split(data_path, split=split)
    y = np.asarray(test_pack["Y"], dtype=np.float32)
    x = np.asarray(test_pack["X"], dtype=np.float32)

    summaries: list[dict[str, object]] = []
    for method, params, notes, fn in task_methods(task, fs):
        pred = fn(y)
        out_dir = output_dir / f"{run_id}_{task}_{method}"
        summaries.append(
            write_summary(
                out_dir,
                run_id=run_id,
                task=task,
                method=method,
                data_path=data_path,
                fs=fs,
                prediction=pred,
                target=x,
                trainable_parameters=params,
                notes=notes,
            )
        )
        print(f"[result] task={task} method={method} CC={summaries[-1]['test']['CC']:.6f}")

    if include_oracle_regression:
        train_pack = load_synthetic_split(data_path, split="train")
        pred = oracle_artifact_regression(train_pack, test_pack)
        method = "oracle_artifact_regression"
        out_dir = output_dir / f"{run_id}_{task}_{method}"
        summaries.append(
            write_summary(
                out_dir,
                run_id=run_id,
                task=task,
                method=method,
                data_path=data_path,
                fs=fs,
                prediction=pred,
                target=x,
                trainable_parameters=1,
                notes="Oracle-assisted one-scalar regression using the stored synthetic artifact component A.",
            )
        )
        print(f"[result] task={task} method={method} CC={summaries[-1]['test']['CC']:.6f}")

    return summaries


def write_index(path: Path, summaries: list[dict[str, object]]) -> None:
    metric_keys = ["CC", "MSE", "RMSE", "T_RRMSE", "S_RRMSE", "SDR", "PSD_KLD", "PSD_WD"]
    fields = [
        "task",
        "method",
        "trainable_parameters",
        "data",
        "length",
        "n_samples",
        *[f"test_{key}" for key in metric_keys],
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            test = summary["test"]
            row = {
                "task": summary["task"],
                "method": summary["method"],
                "trainable_parameters": summary["trainable_parameters"],
                "data": summary["data"],
                "length": summary["length"],
                "n_samples": test["n_samples"],
                "notes": summary["notes"],
            }
            for key in metric_keys:
                row[f"test_{key}"] = test[key]
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    summaries.extend(
        evaluate_task(
            task="eog",
            data_path=args.eog_data,
            fs=256.0,
            output_dir=args.output_dir,
            run_id=args.run_id,
            split=args.split,
            include_oracle_regression=args.include_oracle_regression,
        )
    )
    summaries.extend(
        evaluate_task(
            task="emg",
            data_path=args.emg_data,
            fs=512.0,
            output_dir=args.output_dir,
            run_id=args.run_id,
            split=args.split,
            include_oracle_regression=args.include_oracle_regression,
        )
    )
    (args.output_dir / f"{args.run_id}_classical_summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_index(args.output_dir / f"{args.run_id}_classical_summary_index.csv", summaries)
    print(f"[written] {args.output_dir}")


if __name__ == "__main__":
    main()
