#!/usr/bin/env python3
"""A9 BCI IV-2a domain-adapted single-channel denoiser control."""

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
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from eeg_denoise_benchmark.eval.metrics import compute_denoising_metrics, sdr_db_np  # noqa: E402
from eeg_denoise_benchmark.models import count_trainable_parameters  # noqa: E402
from eeg_denoise_benchmark.training.losses import DenoiseLossConfig, denoise_loss, make_band_weights_rfft  # noqa: E402
from scripts.evaluate_bci2a_contamination_seed_repeats_csp_lda import (  # noqa: E402
    bh_fdr_adjust,
    bootstrap_mean_ci,
    exact_wilcoxon_p,
    holm_adjust,
    median,
    sample_sd,
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
    FS_MODEL,
    METRIC_KEYS,
    bandpass_epochs,
    checkpoint_seed,
    denoise_epochs,
    evaluate_classifier,
    expand_checkpoints,
    fit_classifier,
    load_bci2a_trials,
    load_checkpoint_model,
)


RECON_METRICS = ["CC", "RMSE", "T_RRMSE", "S_RRMSE", "SDR"]


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
    parser.add_argument("--recipes", default="eog", help=f"Comma-separated subset of {DEFAULT_RECIPES}.")
    parser.add_argument(
        "--checkpoint-glob",
        action="append",
        required=True,
        help="External single-channel checkpoint glob. Repeat to combine checkpoint families.",
    )
    parser.add_argument("--bases", default="6,16")
    parser.add_argument("--checkpoint-seeds", default="42,43,44")
    parser.add_argument("--train-seeds", default="1042,1043,1044,1045,1046")
    parser.add_argument("--test-seeds", default="42,43,44,45,46")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--denoise-batch-size", type=int, default=256)
    parser.add_argument("--adapt-batch-size", type=int, default=256)
    parser.add_argument("--adapt-epochs", type=int, default=15)
    parser.add_argument("--adapt-lr", type=float, default=3e-4)
    parser.add_argument("--adapt-weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
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
    parser.add_argument("--bootstrap-seed", type=int, default=20260815)
    parser.add_argument(
        "--multichannel-control-dir",
        type=Path,
        default=Path("runs/bci2a_multichannel_denoiser_csp_control_20260602_203000"),
        help="Existing BCI-trained multichannel control directory to ingest when available.",
    )
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
        "seed_pair_index",
        "train_contamination_seed",
        "test_contamination_seed",
        "condition",
        "domain_condition",
        "base",
        "checkpoint_seed",
        "checkpoint",
        "adapted_checkpoint",
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


def flat_single_channel_arrays(
    noisy_epochs: np.ndarray,
    clean_epochs: np.ndarray,
    artifact_epochs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    noisy = noisy_epochs.reshape(-1, noisy_epochs.shape[-1]).astype(np.float32)
    clean = clean_epochs.reshape(-1, clean_epochs.shape[-1]).astype(np.float32)
    artifact = artifact_epochs.reshape(-1, artifact_epochs.shape[-1]).astype(np.float32)
    sigma = np.std(noisy, axis=1, keepdims=True).astype(np.float32) + 1e-8
    return (noisy / sigma).astype(np.float32), (clean / sigma).astype(np.float32), (artifact / sigma).astype(np.float32), sigma


def metric_stats(clean_epochs: np.ndarray, denoised_epochs: np.ndarray) -> dict[str, float]:
    target = clean_epochs.reshape(-1, clean_epochs.shape[-1]).astype(np.float32)
    prediction = denoised_epochs.reshape(-1, denoised_epochs.shape[-1]).astype(np.float32)
    metrics = compute_denoising_metrics(target=target, prediction=prediction, fs=float(FS_MODEL))
    return {key: float(metrics[key]) for key in RECON_METRICS}


def loss_config_from_cfg(cfg: dict[str, Any], args: argparse.Namespace) -> DenoiseLossConfig:
    return DenoiseLossConfig(
        fs=float(FS_MODEL),
        alpha=float(cfg.get("alpha", 0.0)),
        beta=float(cfg.get("beta", 0.0)),
        gamma=float(cfg.get("gamma", 0.0)),
        eta=float(cfg.get("eta", 0.0)),
        snr_min=float(args.snr_min_db),
        snr_max=float(args.snr_max_db),
    )


def run_adapt_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    loss_config: DenoiseLossConfig,
    fft_weights: torch.Tensor,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    losses: list[float] = []
    sdrs: list[float] = []
    for xb, yb, ab in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        ab = ab.to(device)
        snr = torch.zeros((xb.shape[0],), dtype=torch.float32, device=device)
        with torch.set_grad_enabled(train):
            pred = model(xb)
            loss, _parts = denoise_loss(
                mixture=xb,
                clean=yb,
                artifact=ab,
                prediction=pred,
                snr_db=snr,
                config=loss_config,
                fft_weights=fft_weights,
            )
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        clean_hat = pred[:, 0, :].detach().cpu().numpy()
        target = yb[:, 0, :].detach().cpu().numpy()
        losses.append(float(loss.detach().cpu()))
        sdrs.append(float(np.mean(sdr_db_np(target, clean_hat))))
    return {"loss": mean(losses), "SDR": mean(sdrs)}


def fine_tune_bci_adapted(
    *,
    checkpoint_path: Path,
    train_noisy: np.ndarray,
    train_clean: np.ndarray,
    train_artifact: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    subject: str,
    recipe: str,
    seed_pair_index: int,
    base: int,
    checkpoint_seed_value: int,
) -> tuple[nn.Module, dict[str, Any], Path]:
    model, cfg, n_params = load_checkpoint_model(checkpoint_path, device)
    rng = np.random.default_rng(100000 + 1000 * seed_pair_index + 37 * int(checkpoint_seed_value) + int(base))
    x, y, artifact, _sigma = flat_single_channel_arrays(train_noisy, train_clean, train_artifact)
    order = rng.permutation(x.shape[0])
    n_val = max(1, int(round(args.val_fraction * x.shape[0])))
    val_idx = order[:n_val]
    train_idx = order[n_val:]
    train_ds = TensorDataset(
        torch.from_numpy(x[train_idx, None, :]).float(),
        torch.from_numpy(y[train_idx, None, :]).float(),
        torch.from_numpy(artifact[train_idx, None, :]).float(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(x[val_idx, None, :]).float(),
        torch.from_numpy(y[val_idx, None, :]).float(),
        torch.from_numpy(artifact[val_idx, None, :]).float(),
    )
    generator = torch.Generator()
    generator.manual_seed(int(checkpoint_seed_value) + 1000 * seed_pair_index)
    train_loader = DataLoader(train_ds, batch_size=args.adapt_batch_size, shuffle=True, generator=generator, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.adapt_batch_size, shuffle=False, drop_last=False)
    loss_config = loss_config_from_cfg(cfg, args)
    fft_weights = make_band_weights_rfft(length=train_noisy.shape[-1], fs=float(FS_MODEL), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.adapt_lr, weight_decay=args.adapt_weight_decay)
    best_state: dict[str, torch.Tensor] | None = None
    best_val_sdr = -1e30
    best_epoch = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.adapt_epochs + 1):
        train_stats = run_adapt_epoch(
            model,
            train_loader,
            device=device,
            loss_config=loss_config,
            fft_weights=fft_weights,
            optimizer=optimizer,
        )
        val_stats = run_adapt_epoch(
            model,
            val_loader,
            device=device,
            loss_config=loss_config,
            fft_weights=fft_weights,
            optimizer=None,
        )
        if val_stats["SDR"] > best_val_sdr:
            best_val_sdr = float(val_stats["SDR"])
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        history.append({"epoch": epoch, "train": train_stats, "val": val_stats, "best_epoch": best_epoch, "best_val_sdr": best_val_sdr})

    if best_state is not None:
        model.load_state_dict(best_state)
    adapted_dir = args.output_dir / "adapted_checkpoints"
    adapted_dir.mkdir(parents=True, exist_ok=True)
    adapted_path = adapted_dir / f"{subject}_{recipe}_pair{seed_pair_index}_base{base}_seed{checkpoint_seed_value}.pt"
    save_cfg = dict(cfg)
    save_cfg.update(
        {
            "domain_condition": "bci_adapted_single_channel",
            "adapt_source": "BCI_IV2a_A0xT",
            "adapt_subject": subject,
            "adapt_recipe": recipe,
            "adapt_seed_pair_index": int(seed_pair_index),
            "adapt_epochs": int(args.adapt_epochs),
            "adapt_lr": float(args.adapt_lr),
            "adapt_weight_decay": float(args.adapt_weight_decay),
        }
    )
    torch.save(
        {
            "model_state": model.state_dict(),
            "cfg": save_cfg,
            "seed": int(checkpoint_seed_value),
            "adaptation": {
                "best_epoch": int(best_epoch),
                "best_val_sdr": float(best_val_sdr),
                "history": history,
                "n_train_examples": int(len(train_idx)),
                "n_val_examples": int(len(val_idx)),
                "source_checkpoint": str(checkpoint_path),
            },
        },
        adapted_path,
    )
    meta = {
        "adapted_checkpoint": str(adapted_path),
        "adapt_best_epoch": int(best_epoch),
        "adapt_best_val_sdr": float(best_val_sdr),
        "adapt_final_train_loss": float(history[-1]["train"]["loss"]),
        "adapt_final_val_loss": float(history[-1]["val"]["loss"]),
        "adapt_n_train_examples": int(len(train_idx)),
        "adapt_n_val_examples": int(len(val_idx)),
        "trainable_parameters": int(n_params),
    }
    model.to(device)
    model.eval()
    return model, meta, adapted_path


def evaluate_csp(
    args: argparse.Namespace,
    train_epochs: np.ndarray,
    y_train: np.ndarray,
    test_epochs: np.ndarray,
    y_test: np.ndarray,
    *,
    condition: str,
) -> dict[str, Any]:
    train_features = bandpass_epochs(train_epochs, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz)
    test_features = bandpass_epochs(test_epochs, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        classifier = fit_classifier(args, train_features, y_train)
    return evaluate_classifier(classifier, test_features, y_test, condition=condition)


def row_key(row: dict[str, Any] | dict[str, str]) -> tuple[str, ...]:
    return (
        str(row["subject"]),
        str(row["recipe"]),
        str(row["seed_pair_index"]),
        str(row["condition"]),
        str(row.get("domain_condition", "")),
        str(row.get("base", "")),
        str(row.get("checkpoint_seed", "")),
    )


def evaluate_one_seed_pair(
    args: argparse.Namespace,
    *,
    subject: str,
    recipe: str,
    seed_pair_index: int,
    train_seed: int,
    test_seed: int,
    subject_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]],
    emg_pool: np.ndarray,
    checkpoints: list[Path],
    device: torch.device,
    baseline_rows: list[dict[str, Any] | dict[str, str]],
    result_rows: list[dict[str, Any] | dict[str, str]],
    recon_rows: list[dict[str, Any] | dict[str, str]],
) -> dict[str, Any]:
    train_clean, train_eog, test_clean, test_eog, y_train, y_test, info = subject_data
    train_noisy, train_artifact, train_noise = make_recipe_noisy_epochs(
        train_clean,
        train_eog,
        emg_pool,
        recipe=recipe,
        seed=train_seed,
        snr_min_db=args.snr_min_db,
        snr_max_db=args.snr_max_db,
    )
    test_noisy, test_artifact, test_noise = make_recipe_noisy_epochs(
        test_clean,
        test_eog,
        emg_pool,
        recipe=recipe,
        seed=test_seed,
        snr_min_db=args.snr_min_db,
        snr_max_db=args.snr_max_db,
    )

    existing_baselines = {row_key(row) for row in baseline_rows}
    existing_results = {row_key(row) for row in result_rows}

    raw_metrics = evaluate_csp(args, train_clean, y_train, test_clean, y_test, condition="raw_raw")
    raw_row = dict(raw_metrics)
    raw_row.update(
        {
            "run_id": args.run_id,
            "subject": subject,
            "recipe": recipe,
            "seed_pair_index": seed_pair_index,
            "train_contamination_seed": train_seed,
            "test_contamination_seed": test_seed,
            "domain_condition": "raw",
            "base": "",
            "checkpoint_seed": "",
            "checkpoint": "",
            "adapted_checkpoint": "",
            "trainable_parameters": "",
            "delta_accuracy_vs_noisy_noisy": "",
        }
    )
    if row_key(raw_row) not in existing_baselines:
        baseline_rows.append(raw_row)

    noisy_metrics = evaluate_csp(args, train_noisy, y_train, test_noisy, y_test, condition="noisy_noisy")
    noisy_row = dict(noisy_metrics)
    noisy_row.update(
        {
            "run_id": args.run_id,
            "subject": subject,
            "recipe": recipe,
            "seed_pair_index": seed_pair_index,
            "train_contamination_seed": train_seed,
            "test_contamination_seed": test_seed,
            "domain_condition": "noisy",
            "base": "",
            "checkpoint_seed": "",
            "checkpoint": "",
            "adapted_checkpoint": "",
            "trainable_parameters": "",
            "delta_accuracy_vs_noisy_noisy": "",
        }
    )
    if row_key(noisy_row) not in existing_baselines:
        baseline_rows.append(noisy_row)
    baseline_acc = float(noisy_metrics["accuracy"])
    print(
        f"[baseline] subject={subject} recipe={recipe} seed_pair={seed_pair_index} raw={raw_metrics['accuracy']:.6f} noisy={baseline_acc:.6f}",
        flush=True,
    )

    for checkpoint_path in checkpoints:
        zero_model, cfg, n_params = load_checkpoint_model(checkpoint_path, device)
        base = int(checkpoint_base(checkpoint_path, cfg) or -1)
        seed_value = int(checkpoint_seed(checkpoint_path, cfg))
        for domain_condition in ["zero_shot_external", "bci_adapted_single_channel"]:
            key_probe = {
                "subject": subject,
                "recipe": recipe,
                "seed_pair_index": seed_pair_index,
                "condition": "denoised_denoised",
                "domain_condition": domain_condition,
                "base": base,
                "checkpoint_seed": seed_value,
            }
            if row_key(key_probe) in existing_results:
                continue
            if domain_condition == "zero_shot_external":
                model = zero_model
                meta: dict[str, Any] = {
                    "adapted_checkpoint": "",
                    "adapt_best_epoch": "",
                    "adapt_best_val_sdr": "",
                    "adapt_final_train_loss": "",
                    "adapt_final_val_loss": "",
                    "adapt_n_train_examples": "",
                    "adapt_n_val_examples": "",
                    "trainable_parameters": int(n_params),
                }
            else:
                model, meta, _adapted_path = fine_tune_bci_adapted(
                    checkpoint_path=checkpoint_path,
                    train_noisy=train_noisy,
                    train_clean=train_clean,
                    train_artifact=train_artifact,
                    args=args,
                    device=device,
                    subject=subject,
                    recipe=recipe,
                    seed_pair_index=seed_pair_index,
                    base=base,
                    checkpoint_seed_value=seed_value,
                )

            train_den = denoise_epochs(model, train_noisy, device=device, batch_size=args.denoise_batch_size)
            test_den = denoise_epochs(model, test_noisy, device=device, batch_size=args.denoise_batch_size)
            metrics = evaluate_csp(args, train_den, y_train, test_den, y_test, condition="denoised_denoised")
            row = dict(metrics)
            row.update(
                {
                    "run_id": args.run_id,
                    "subject": subject,
                    "recipe": recipe,
                    "seed_pair_index": seed_pair_index,
                    "train_contamination_seed": train_seed,
                    "test_contamination_seed": test_seed,
                    "domain_condition": domain_condition,
                    "base": base,
                    "checkpoint_seed": seed_value,
                    "checkpoint": str(checkpoint_path),
                    "adapted_checkpoint": meta["adapted_checkpoint"],
                    "trainable_parameters": int(meta["trainable_parameters"]),
                    "baseline_accuracy": baseline_acc,
                    "delta_accuracy_vs_noisy_noisy": float(row["accuracy"]) - baseline_acc,
                    "adapt_best_epoch": meta["adapt_best_epoch"],
                    "adapt_best_val_sdr": meta["adapt_best_val_sdr"],
                    "adapt_final_train_loss": meta["adapt_final_train_loss"],
                    "adapt_final_val_loss": meta["adapt_final_val_loss"],
                    "adapt_n_train_examples": meta["adapt_n_train_examples"],
                    "adapt_n_val_examples": meta["adapt_n_val_examples"],
                }
            )
            result_rows.append(row)
            recon = metric_stats(test_clean, test_den)
            recon_row = {
                "run_id": args.run_id,
                "subject": subject,
                "recipe": recipe,
                "seed_pair_index": seed_pair_index,
                "train_contamination_seed": train_seed,
                "test_contamination_seed": test_seed,
                "domain_condition": domain_condition,
                "base": base,
                "checkpoint_seed": seed_value,
                "checkpoint": str(checkpoint_path),
                "adapted_checkpoint": meta["adapted_checkpoint"],
                **recon,
            }
            recon_rows.append(recon_row)
            print(
                f"[result] subject={subject} recipe={recipe} seed_pair={seed_pair_index} "
                f"domain={domain_condition} base={base} checkpoint_seed={seed_value} "
                f"acc={row['accuracy']:.6f} delta={row['delta_accuracy_vs_noisy_noisy']:+.6f} "
                f"CC={recon['CC']:.4f} SDR={recon['SDR']:.2f}",
                flush=True,
            )
        del zero_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "subject": subject,
        "recipe": recipe,
        "seed_pair_index": seed_pair_index,
        "train_contamination_seed": train_seed,
        "test_contamination_seed": test_seed,
        "train_noise": train_noise,
        "test_noise": test_noise,
        "train_source": info["train"]["source"],
        "test_source": info["test"]["source"],
        "train_artifact_shape": list(train_artifact.shape),
        "test_artifact_shape": list(test_artifact.shape),
    }


def subject_summary(
    baseline_rows: list[dict[str, Any] | dict[str, str]],
    result_rows: list[dict[str, Any] | dict[str, str]],
    recon_rows: list[dict[str, Any] | dict[str, str]],
) -> list[dict[str, Any]]:
    noisy_group: dict[tuple[str, str], list[float]] = defaultdict(list)
    raw_group: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in baseline_rows:
        key = (str(row["subject"]), str(row["recipe"]))
        if row["condition"] == "noisy_noisy":
            noisy_group[key].append(float(row["accuracy"]))
        elif row["condition"] == "raw_raw":
            raw_group[key].append(float(row["accuracy"]))

    recon_group: dict[tuple[str, str, str, int], list[dict[str, Any] | dict[str, str]]] = defaultdict(list)
    for row in recon_rows:
        recon_group[(str(row["subject"]), str(row["recipe"]), str(row["domain_condition"]), int(row["base"]))].append(row)

    grouped: dict[tuple[str, str, str, int], list[dict[str, Any] | dict[str, str]]] = defaultdict(list)
    for row in result_rows:
        grouped[(str(row["subject"]), str(row["recipe"]), str(row["domain_condition"]), int(row["base"]))].append(row)

    rows: list[dict[str, Any]] = []
    for (subject, recipe, domain_condition, base), items in sorted(grouped.items()):
        deltas = [float(item["delta_accuracy_vs_noisy_noisy"]) for item in items]
        acc = [float(item["accuracy"]) for item in items]
        by_checkpoint: dict[str, list[float]] = defaultdict(list)
        by_contamination: dict[str, list[float]] = defaultdict(list)
        for item in items:
            by_checkpoint[str(item["checkpoint_seed"])].append(float(item["delta_accuracy_vs_noisy_noisy"]))
            by_contamination[str(item["seed_pair_index"])].append(float(item["delta_accuracy_vs_noisy_noisy"]))
        entry: dict[str, Any] = {
            "subject": subject,
            "recipe": recipe,
            "domain_condition": domain_condition,
            "base": base,
            "n_contamination_seed_pairs": len(by_contamination),
            "n_checkpoint_seeds": len(by_checkpoint),
            "n_observations_averaged": len(items),
            "raw_accuracy_mean": mean(raw_group[(subject, recipe)]),
            "noisy_accuracy_mean": mean(noisy_group[(subject, recipe)]),
            "denoised_accuracy_mean": mean(acc),
            "delta_accuracy": mean(deltas),
            "delta_accuracy_sd_over_nuisance": sample_sd(deltas),
            "checkpoint_delta_sd_within_subject": sample_sd([mean(values) for _, values in sorted(by_checkpoint.items())]),
            "contamination_delta_sd_within_subject": sample_sd([mean(values) for _, values in sorted(by_contamination.items())]),
        }
        ritems = recon_group[(subject, recipe, domain_condition, base)]
        for key in RECON_METRICS:
            entry[f"{key}_mean"] = mean([float(item[key]) for item in ritems])
            entry[f"{key}_sd_over_nuisance"] = sample_sd([float(item[key]) for item in ritems])
        rows.append(entry)
    return rows


def inference_rows(subject_rows: list[dict[str, Any]], *, n_bootstrap: int, bootstrap_seed: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in subject_rows:
        grouped[(str(row["recipe"]), str(row["domain_condition"]), int(row["base"]))].append(row)
    rows: list[dict[str, Any]] = []
    for (recipe, domain_condition, base), items in sorted(grouped.items()):
        deltas = [float(item["delta_accuracy"]) for item in items]
        ci_low, ci_high = bootstrap_mean_ci(deltas, n_bootstrap=n_bootstrap, seed=bootstrap_seed + 1000 * len(rows))
        entry: dict[str, Any] = {
            "recipe": recipe,
            "domain_condition": domain_condition,
            "base": base,
            "n_subjects": len(items),
            "subjects": " ".join(str(item["subject"]) for item in items),
            "n_contamination_seed_pairs": int(items[0]["n_contamination_seed_pairs"]),
            "n_checkpoint_seeds": int(items[0]["n_checkpoint_seeds"]),
            "raw_accuracy_subject_mean": mean([float(item["raw_accuracy_mean"]) for item in items]),
            "noisy_accuracy_subject_mean": mean([float(item["noisy_accuracy_mean"]) for item in items]),
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
        for key in RECON_METRICS:
            entry[f"{key}_subject_mean"] = mean([float(item[f"{key}_mean"]) for item in items])
        rows.append(entry)
    bh_fdr_adjust(rows, "wilcoxon_p_denoised_lt_noisy", "bh_fdr_q_denoised_lt_noisy")
    holm_adjust(rows, "wilcoxon_p_denoised_lt_noisy", "holm_p_denoised_lt_noisy")
    return rows


def adaptation_contrast_rows(subject_rows: list[dict[str, Any]], *, n_bootstrap: int, bootstrap_seed: int) -> list[dict[str, Any]]:
    by_key = {
        (str(row["subject"]), str(row["recipe"]), int(row["base"]), str(row["domain_condition"])): row
        for row in subject_rows
    }
    rows: list[dict[str, Any]] = []
    groups = sorted({(str(row["recipe"]), int(row["base"])) for row in subject_rows})
    for recipe, base in groups:
        contrasts: list[float] = []
        subjects: list[str] = []
        for subject in sorted({str(row["subject"]) for row in subject_rows}):
            zero = by_key.get((subject, recipe, base, "zero_shot_external"))
            adapted = by_key.get((subject, recipe, base, "bci_adapted_single_channel"))
            if zero is None or adapted is None:
                continue
            contrasts.append(float(adapted["delta_accuracy"]) - float(zero["delta_accuracy"]))
            subjects.append(subject)
        if not contrasts:
            continue
        ci_low, ci_high = bootstrap_mean_ci(contrasts, n_bootstrap=n_bootstrap, seed=bootstrap_seed + 50000 + 1000 * len(rows))
        rows.append(
            {
                "recipe": recipe,
                "base": base,
                "contrast": "bci_adapted_minus_zero_shot_delta_accuracy",
                "n_subjects": len(subjects),
                "subjects": " ".join(subjects),
                "mean_contrast": mean(contrasts),
                "median_contrast": median(contrasts),
                "sd_contrast_across_subjects": sample_sd(contrasts),
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
                "wilcoxon_p_adapted_gt_zero": exact_wilcoxon_p(contrasts, "greater"),
                "wilcoxon_p_adapted_lt_zero": exact_wilcoxon_p(contrasts, "less"),
                "subjects_adaptation_reduced_degradation": sum(1 for value in contrasts if value > 0),
                "subjects_adaptation_worsened_delta": sum(1 for value in contrasts if value < 0),
            }
        )
    bh_fdr_adjust(rows, "wilcoxon_p_adapted_gt_zero", "bh_fdr_q_adapted_gt_zero")
    return rows


def load_multichannel_control_rows(path: Path, *, recipes: list[str]) -> list[dict[str, Any]]:
    agg_path = path / "bci2a_multichannel_denoiser_csp_control_scope_aggregate.csv"
    if not agg_path.exists() or "eog" not in recipes:
        return []
    rows = []
    for row in read_csv(agg_path):
        if str(row.get("denoiser_scope")) != "multichannel":
            continue
        rows.append(
            {
                "recipe": "eog",
                "domain_condition": "bci_trained_multichannel_control_existing",
                "base": int(row["base"]),
                "n_subjects": int(row["n_subjects"]),
                "subjects": row["subjects"],
                "n_contamination_seed_pairs": 1,
                "n_checkpoint_seeds": int(row["n_train_seeds_per_subject"]),
                "raw_accuracy_subject_mean": "",
                "noisy_accuracy_subject_mean": "",
                "denoised_accuracy_subject_mean": float(row["accuracy_subject_mean"]),
                "mean_delta_accuracy": float(row["delta_accuracy_vs_noisy_noisy_subject_mean"]),
                "median_delta_accuracy": "",
                "sd_delta_accuracy_across_subjects": float(row["delta_accuracy_vs_noisy_noisy_subject_std"]),
                "bootstrap_ci95_low": "",
                "bootstrap_ci95_high": "",
                "n_bootstrap_subject_resamples": "",
                "subjects_below_noisy_noisy": int(row["subjects_below_noisy_noisy"]),
                "subjects_above_noisy_noisy": int(row["subjects_above_noisy_noisy"]),
                "wilcoxon_p_denoised_lt_noisy": float(row["wilcoxon_p_denoised_lt_noisy"]),
                "wilcoxon_p_denoised_gt_noisy": float(row["wilcoxon_p_denoised_gt_noisy"]),
                "wilcoxon_p_two_sided": "",
                "bh_fdr_q_denoised_lt_noisy": float(row["bh_fdr_p_denoised_lt_noisy"]),
                "holm_p_denoised_lt_noisy": "",
                "source": str(path),
                "note": "Existing multichannel control used fixed EOG contamination and is included as an external comparator.",
            }
        )
    return rows


def write_markdown(
    path: Path,
    *,
    args: argparse.Namespace,
    inference: list[dict[str, Any]],
    contrasts: list[dict[str, Any]],
    multichannel_rows: list[dict[str, Any]],
) -> None:
    lines = ["# A9 BCI IV-2a Domain-Shift Control", ""]
    lines.append("This experiment compares external zero-shot single-channel denoising with subject-specific BCI-adapted single-channel denoising.")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append(f"- Subjects: `{args.subjects}`.")
    lines.append(f"- Recipes: `{args.recipes}`.")
    lines.append(f"- Bases: `{args.bases}`.")
    lines.append(f"- Checkpoint seeds: `{args.checkpoint_seeds}`.")
    lines.append(f"- Train contamination seeds: `{args.train_seeds}`.")
    lines.append(f"- Test contamination seeds: `{args.test_seeds}`.")
    lines.append(f"- Adaptation source: subject-matched `A0xT` synthetic contaminated/target pairs only.")
    lines.append(f"- Evaluation: subject-matched `A0xE` synthetic contaminated trials and CSP+LDA.")
    lines.append(f"- Adaptation epochs: `{args.adapt_epochs}` with best checkpoint selected by held-out `A0xT` validation SDR.")
    lines.append("- Inference unit: subject. Contamination and checkpoint repetitions are averaged within subject.")
    lines.append("")
    lines.append("## Subject-Level Effects")
    lines.append("")
    lines.append("| Recipe | Domain condition | Base | n subjects | Noisy acc | Denoised acc | Mean delta | 95% subject-bootstrap CI | Below noisy | Wilcoxon p lower | BH q | CC | SDR |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in inference:
        lines.append(
            f"| {row['recipe']} | {row['domain_condition']} | {row['base']} | {row['n_subjects']} | "
            f"{row['noisy_accuracy_subject_mean']:.6f} | {row['denoised_accuracy_subject_mean']:.6f} | "
            f"{row['mean_delta_accuracy']:+.6f} | [{row['bootstrap_ci95_low']:+.6f}, {row['bootstrap_ci95_high']:+.6f}] | "
            f"{row['subjects_below_noisy_noisy']} | {row['wilcoxon_p_denoised_lt_noisy']:.6f} | "
            f"{row['bh_fdr_q_denoised_lt_noisy']:.6f} | {row['CC_subject_mean']:.6f} | {row['SDR_subject_mean']:.6f} |"
        )
    if multichannel_rows:
        lines.append("")
        lines.append("## Existing Multichannel Comparator")
        lines.append("")
        lines.append("| Recipe | Domain condition | Base | n subjects | Denoised acc | Mean delta | Below noisy | Wilcoxon p lower | BH q | Note |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for row in multichannel_rows:
            lines.append(
                f"| {row['recipe']} | {row['domain_condition']} | {row['base']} | {row['n_subjects']} | "
                f"{row['denoised_accuracy_subject_mean']:.6f} | {row['mean_delta_accuracy']:+.6f} | "
                f"{row['subjects_below_noisy_noisy']} | {row['wilcoxon_p_denoised_lt_noisy']:.6f} | "
                f"{row['bh_fdr_q_denoised_lt_noisy']:.6f} | {row['note']} |"
            )
    lines.append("")
    lines.append("## Adaptation Contrast")
    lines.append("")
    lines.append("| Recipe | Base | Mean adapted-minus-zero-shot delta | 95% subject-bootstrap CI | Subjects improved | Wilcoxon p adapted>zero | BH q |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in contrasts:
        lines.append(
            f"| {row['recipe']} | {row['base']} | {row['mean_contrast']:+.6f} | "
            f"[{row['bootstrap_ci95_low']:+.6f}, {row['bootstrap_ci95_high']:+.6f}] | "
            f"{row['subjects_adaptation_reduced_degradation']}/{row['n_subjects']} | "
            f"{row['wilcoxon_p_adapted_gt_zero']:.6f} | {row['bh_fdr_q_adapted_gt_zero']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.n_bootstrap < 10000:
        raise ValueError("--n-bootstrap must be at least 10000")
    subjects = parse_subjects(args.subjects)
    recipes = parse_recipes(args.recipes)
    bases = set(parse_int_list(args.bases, name="bases"))
    checkpoint_seeds = set(parse_int_list(args.checkpoint_seeds, name="checkpoint-seeds"))
    train_seeds = parse_int_list(args.train_seeds, name="train-seeds")
    test_seeds = parse_int_list(args.test_seeds, name="test-seeds")
    if len(train_seeds) != len(test_seeds):
        raise ValueError("--train-seeds and --test-seeds must have matching lengths")
    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    checkpoints = expand_checkpoint_patterns(args.checkpoint_glob, bases=bases, seeds=checkpoint_seeds)
    emg_pool = load_emg_pool(args.emg_pool)
    print(
        f"[start] run_id={args.run_id} subjects={subjects} recipes={recipes} bases={sorted(bases)} "
        f"checkpoint_seeds={sorted(checkpoint_seeds)} seed_pairs={list(zip(train_seeds, test_seeds))} "
        f"checkpoints={len(checkpoints)} device={device}",
        flush=True,
    )

    baseline_path = args.output_dir / "a9_baseline_seed_rows.csv"
    result_path = args.output_dir / "a9_domain_condition_seed_rows.csv"
    recon_path = args.output_dir / "a9_reconstruction_seed_rows.csv"
    noise_path = args.output_dir / "a9_noise_metadata_rows.csv"
    baseline_rows: list[dict[str, Any] | dict[str, str]] = read_csv(baseline_path) if args.resume else []
    result_rows: list[dict[str, Any] | dict[str, str]] = read_csv(result_path) if args.resume else []
    recon_rows: list[dict[str, Any] | dict[str, str]] = read_csv(recon_path) if args.resume else []
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
                    f"[seed_pair_start] subject={subject} recipe={recipe} seed_pair={seed_pair_index} train_seed={train_seed} test_seed={test_seed}",
                    flush=True,
                )
                noise_info = evaluate_one_seed_pair(
                    args,
                    subject=subject,
                    recipe=recipe,
                    seed_pair_index=seed_pair_index,
                    train_seed=train_seed,
                    test_seed=test_seed,
                    subject_data=subject_data,
                    emg_pool=emg_pool,
                    checkpoints=checkpoints,
                    device=device,
                    baseline_rows=baseline_rows,
                    result_rows=result_rows,
                    recon_rows=recon_rows,
                )
                noise_rows.append(noise_info)
                write_csv(baseline_path, [dict(row) for row in baseline_rows])
                write_csv(result_path, [dict(row) for row in result_rows])
                write_csv(recon_path, [dict(row) for row in recon_rows])
                write_csv(noise_path, [dict(row) for row in noise_rows])
                print(
                    f"[seed_pair_done] subject={subject} recipe={recipe} seed_pair={seed_pair_index} "
                    f"baseline_rows={len(baseline_rows)} result_rows={len(result_rows)} recon_rows={len(recon_rows)}",
                    flush=True,
                )

    subject_rows = subject_summary(baseline_rows, result_rows, recon_rows)
    inference = inference_rows(subject_rows, n_bootstrap=args.n_bootstrap, bootstrap_seed=args.bootstrap_seed)
    contrasts = adaptation_contrast_rows(subject_rows, n_bootstrap=args.n_bootstrap, bootstrap_seed=args.bootstrap_seed)
    multichannel_rows = load_multichannel_control_rows(args.multichannel_control_dir, recipes=recipes)
    write_csv(args.output_dir / "a9_subject_domain_summary.csv", subject_rows)
    write_csv(args.output_dir / "a9_domain_inference.csv", inference)
    write_csv(args.output_dir / "a9_adaptation_contrasts.csv", contrasts)
    write_csv(args.output_dir / "a9_existing_multichannel_comparator.csv", multichannel_rows)
    summary = {
        "info": {
            "run_id": args.run_id,
            "subjects": subjects,
            "recipes": recipes,
            "bases": sorted(bases),
            "checkpoint_seeds": sorted(checkpoint_seeds),
            "train_contamination_seeds": train_seeds,
            "test_contamination_seeds": test_seeds,
            "adaptation_source": "BCI_IV2a_A0xT_only",
            "evaluation_source": "BCI_IV2a_A0xE",
            "adapt_epochs": args.adapt_epochs,
            "adapt_lr": args.adapt_lr,
            "adapt_weight_decay": args.adapt_weight_decay,
            "n_bootstrap_subject_resamples": args.n_bootstrap,
            "inferential_unit": "subject",
            "multichannel_control_dir": str(args.multichannel_control_dir),
        },
        "subject_info": subject_info,
        "domain_inference": inference,
        "adaptation_contrasts": contrasts,
        "existing_multichannel_comparator": multichannel_rows,
    }
    (args.output_dir / "a9_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.output_dir / "a9_summary.md", args=args, inference=inference, contrasts=contrasts, multichannel_rows=multichannel_rows)
    print(f"[written] {args.output_dir / 'a9_summary.md'}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
