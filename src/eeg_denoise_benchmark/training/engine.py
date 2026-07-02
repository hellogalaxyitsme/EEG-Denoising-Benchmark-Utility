"""Training loop for controlled EEG denoising experiments."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from eeg_denoise_benchmark.config import dump_json
from eeg_denoise_benchmark.data import SyntheticArrayDataset, load_synthetic_split
from eeg_denoise_benchmark.eval.metrics import (
    cc_np,
    compute_denoising_metrics,
    mse_np,
    rmse_np,
    sdr_db_np,
    t_rrmse_np,
)
from eeg_denoise_benchmark.models import (
    LinearFIRDenoiser,
    MicroWaveNet,
    PatchTransformerDenoiser,
    TinyDenoiser,
    TinyTCNDenoiser,
    build_eegdn_baseline,
    count_trainable_parameters,
)
from eeg_denoise_benchmark.training.losses import (
    DenoiseLossConfig,
    denoise_loss,
    make_band_weights_rfft,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lr_schedule(epoch: int, base_lr: float, warmup_epochs: int, total_epochs: int) -> float:
    if epoch <= warmup_epochs:
        return base_lr * (epoch / max(1, warmup_epochs))
    t = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return base_lr * (0.5 * (1 + math.cos(math.pi * t)))


def make_loader(
    data_dir: str | Path,
    split: str,
    batch_size: int,
    num_workers: int,
    include_metadata: bool,
    device: torch.device,
) -> DataLoader:
    pack = load_synthetic_split(data_dir, split=split)
    dataset = SyntheticArrayDataset(pack, include_metadata=include_metadata)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _batch_metrics(clean: torch.Tensor, clean_hat: torch.Tensor) -> dict[str, float]:
    x = clean.detach().cpu().numpy()[:, 0, :]
    xh = clean_hat.detach().cpu().numpy()[:, 0, :]
    return {
        "CC": float(np.mean(cc_np(x, xh))),
        "MSE": float(np.mean(mse_np(x, xh))),
        "RMSE": float(np.mean(rmse_np(x, xh))),
        "T_RRMSE": float(np.mean(t_rrmse_np(x, xh))),
        "SDR": float(np.mean(sdr_db_np(x, xh))),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_config: DenoiseLossConfig,
    fft_weights: torch.Tensor,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    aggregate = {key: [] for key in ["loss", "Lx", "La", "Lc", "Ls", "Le", "CC", "MSE", "RMSE", "T_RRMSE", "SDR"]}

    for batch_index, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break
        mixture, clean, artifact, snr_db, _sigma_y = batch
        mixture = mixture.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        artifact = artifact.to(device, non_blocking=True)
        snr_db = snr_db.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            prediction = model(mixture)
            loss, parts = denoise_loss(
                mixture=mixture,
                clean=clean,
                artifact=artifact,
                prediction=prediction,
                snr_db=snr_db,
                config=loss_config,
                fft_weights=fft_weights,
            )
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        clean_hat = prediction[:, 0:1, :]
        aggregate["loss"].append(float(loss.detach().cpu()))
        for key in ["Lx", "La", "Lc", "Ls", "Le"]:
            aggregate[key].append(parts[key])
        for key, value in _batch_metrics(clean, clean_hat).items():
            aggregate[key].append(value)

    return {key: _mean(value) for key, value in aggregate.items()}


@torch.no_grad()
def predict_split(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for batch_index, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break
        mixture, clean, _artifact, _snr_db, _sigma_y = batch
        mixture = mixture.to(device, non_blocking=True)
        prediction = model(mixture)
        predictions.append(prediction[:, 0, :].detach().cpu().numpy())
        targets.append(clean.numpy()[:, 0, :])
    return np.concatenate(targets, axis=0), np.concatenate(predictions, axis=0)


def _cfg_for_checkpoint(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": config.get("name", "A0_baseline"),
        "model_type": str(config.get("model_type", "controlled_backbone")),
        "variant": str(config.get("variant", config.get("model_type", "controlled_backbone"))),
        "datanum": int(config.get("datanum", config.get("length", 512))),
        "base": int(config.get("base", 16)),
        "extra_blocks": int(config.get("extra_blocks", 2)),
        "use_dwt": bool(config.get("use_dwt", True)),
        "use_gate": bool(config.get("use_gate", True)),
        "use_attn": bool(config.get("use_attn", True)),
        "use_artifact_head": bool(config.get("use_artifact_head", True)),
        "conv_block": str(config.get("conv_block", "ds")),
        "use_spectral": bool(config.get("gamma", 0.0) > 0),
        "use_envelope": bool(config.get("eta", 0.0) > 0),
        "alpha": float(config.get("alpha", 0.5)),
        "beta": float(config.get("beta", 0.5)),
        "gamma": float(config.get("gamma", 0.1)),
        "eta": float(config.get("eta", 0.0)),
    }


def train_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """Train a controlled DSConv U-Net experiment from a flat config dictionary."""

    seed = int(config.get("seed", 42))
    set_seed(seed)
    device_name = str(config.get("device", "cuda"))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    output_dir = Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_json(output_dir / "config.resolved.json", config)

    batch_size = int(config.get("batch_size", 256))
    num_workers = int(config.get("num_workers", 2))
    data_dir = Path(str(config["data"]))
    train_loader = make_loader(data_dir, "train", batch_size, num_workers, True, device)
    val_loader = make_loader(data_dir, "val", batch_size, num_workers, True, device)
    test_loader = make_loader(data_dir, "test", batch_size, num_workers, True, device)

    sample_batch = next(iter(train_loader))
    length = int(sample_batch[0].shape[-1])

    model_type = str(config.get("model_type", "controlled_backbone")).strip().lower()
    config["datanum"] = length
    config["length"] = length
    if model_type in {"controlled_backbone", "tinydenoiser", "tiny_denoiser"}:
        model = TinyDenoiser(
            base=int(config.get("base", 16)),
            extra_bottleneck_blocks=int(config.get("extra_blocks", 2)),
            use_dwt=bool(config.get("use_dwt", True)),
            use_gate=bool(config.get("use_gate", True)),
            use_attn=bool(config.get("use_attn", True)),
            use_artifact_head=bool(config.get("use_artifact_head", True)),
            conv_block=str(config.get("conv_block", "ds")),
        ).to(device)
    elif model_type in {"linear_fir", "linear_fir_denoiser", "fir"}:
        model = LinearFIRDenoiser(kernel_size=int(config.get("kernel_size", 33))).to(device)
    elif model_type in {"tiny_tcn", "tiny_tcn_denoiser", "tcn_tiny"}:
        model = TinyTCNDenoiser(
            hidden=int(config.get("hidden", 4)),
            kernel_size=int(config.get("kernel_size", 9)),
        ).to(device)
    elif model_type in {"patch_transformer", "patch_transformer_denoiser", "tiny_transformer"}:
        model = PatchTransformerDenoiser(
            d_model=int(config.get("d_model", 32)),
            nhead=int(config.get("nhead", 4)),
            num_layers=int(config.get("num_layers", 1)),
            dim_feedforward=int(config.get("dim_feedforward", int(config.get("d_model", 32)) * 2)),
            patch_stride=int(config.get("patch_stride", 4)),
            max_tokens=int(config.get("max_tokens", 512)),
        ).to(device)
    elif model_type in {"microwavenet", "micro_wave_net"}:
        model = MicroWaveNet().to(device)
    else:
        model = build_eegdn_baseline(model_type=model_type, datanum=length).to(device)
    n_params = count_trainable_parameters(model)
    param_cap = int(config.get("param_cap", 75000))
    if n_params > param_cap:
        raise AssertionError(f"Param cap exceeded: {n_params} > {param_cap}")

    loss_config = DenoiseLossConfig(
        fs=float(config.get("fs", 256)),
        alpha=float(config.get("alpha", 0.5)),
        beta=float(config.get("beta", 0.5)),
        gamma=float(config.get("gamma", 0.1)),
        eta=float(config.get("eta", 0.0)),
        snr_min=float(config.get("snr_min", -12.0)),
        snr_max=float(config.get("snr_max", 2.0)),
    )
    fft_weights = make_band_weights_rfft(length=length, fs=loss_config.fs, device=device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("lr", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    epochs = int(config.get("epochs", 25))
    warmup_epochs = int(config.get("warmup_epochs", 3))
    max_train_batches = config.get("max_train_batches")
    max_eval_batches = config.get("max_eval_batches")
    max_train_batches = None if max_train_batches is None else int(max_train_batches)
    max_eval_batches = None if max_eval_batches is None else int(max_eval_batches)

    history_path = output_dir / "history.jsonl"
    best_score = -1e30
    best_epoch = 0
    best_state = None
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        lr = lr_schedule(epoch, float(config.get("lr", 1e-3)), warmup_epochs, epochs)
        for group in optimizer.param_groups:
            group["lr"] = lr

        train_stats = run_epoch(
            model,
            train_loader,
            device,
            loss_config,
            fft_weights,
            optimizer=optimizer,
            max_batches=max_train_batches,
        )
        val_stats = run_epoch(
            model,
            val_loader,
            device,
            loss_config,
            fft_weights,
            optimizer=None,
            max_batches=max_eval_batches,
        )
        score = float(val_stats["SDR"])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

        row = {
            "epoch": epoch,
            "lr": lr,
            "train": train_stats,
            "val": val_stats,
            "best_epoch": best_epoch,
            "best_val_sdr": best_score,
        }
        history.append(row)
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(
            f"Epoch {epoch:03d}/{epochs} | lr={lr:.2e} | "
            f"train loss={train_stats['loss']:.4f} CC={train_stats['CC']:.4f} SDR={train_stats['SDR']:.2f} | "
            f"val loss={val_stats['loss']:.4f} CC={val_stats['CC']:.4f} SDR={val_stats['SDR']:.2f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    target, prediction = predict_split(model, test_loader, device, max_batches=max_eval_batches)
    test_metrics = compute_denoising_metrics(target=target, prediction=prediction, fs=loss_config.fs)
    summary = {
        "output_dir": str(output_dir),
        "data": str(data_dir),
        "device": str(device),
        "seed": seed,
        "length": length,
        "trainable_parameters": n_params,
        "best_epoch": best_epoch,
        "best_val_sdr": best_score,
        "test": test_metrics,
        "history": history,
    }

    checkpoint = {
        "model_state": model.state_dict(),
        "cfg": _cfg_for_checkpoint(config),
        "test": test_metrics,
        "seed": seed,
    }
    torch.save(checkpoint, output_dir / "best.pt")
    dump_json(output_dir / "summary.json", summary)
    return summary
