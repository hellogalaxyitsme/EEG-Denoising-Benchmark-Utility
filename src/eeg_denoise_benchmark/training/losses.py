"""Training losses for supervised EEG denoising experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class DenoiseLossConfig:
    """Weights and frequency settings for the adaptive denoising loss."""

    fs: float
    alpha: float = 0.5
    beta: float = 0.5
    gamma: float = 0.1
    eta: float = 0.0
    snr_min: float = -12.0
    snr_max: float = 2.0
    fft_delta: float = 1e-4
    charbonnier_eps: float = 2e-3


def charbonnier(x: torch.Tensor, eps: float = 2e-3) -> torch.Tensor:
    return torch.sqrt(x * x + eps * eps)


def snr_weight(
    snr_db: torch.Tensor,
    snr_min: float = -12.0,
    snr_max: float = 2.0,
) -> torch.Tensor:
    weight = (snr_max - snr_db) / (snr_max - snr_min + 1e-12)
    return torch.clamp(weight, 0.25, 2.0)


def make_band_weights_rfft(
    length: int,
    fs: float,
    device: torch.device,
    f_low: float = 0.5,
    f_mid: float = 30.0,
    f_hi: float = 80.0,
) -> torch.Tensor:
    freqs = torch.fft.rfftfreq(length, d=1.0 / fs).to(device)
    weights = torch.ones_like(freqs)
    weights = torch.where(
        (freqs >= f_low) & (freqs <= f_mid),
        torch.tensor(2.0, device=device),
        weights,
    )
    weights = torch.where(
        (freqs > f_mid) & (freqs <= f_hi),
        torch.tensor(1.0, device=device),
        weights,
    )
    weights = torch.where(freqs > f_hi, torch.tensor(0.4, device=device), weights)
    return weights


def band_weighted_logfft_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    delta: float = 1e-4,
) -> torch.Tensor:
    pred_fft = torch.fft.rfft(predicted, dim=-1)
    target_fft = torch.fft.rfft(target, dim=-1)
    pred_mag = torch.log(torch.abs(pred_fft) + delta)
    target_mag = torch.log(torch.abs(target_fft) + delta)
    diff = torch.abs(pred_mag - target_mag)
    return (diff * weights.view(1, 1, -1)).mean()


def envelope_loss_artifact(
    predicted_artifact: torch.Tensor,
    artifact: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    pred_energy = predicted_artifact * predicted_artifact
    target_energy = artifact * artifact
    pred_smooth = F.avg_pool1d(pred_energy, kernel_size=33, stride=1, padding=16)
    target_smooth = F.avg_pool1d(target_energy, kernel_size=33, stride=1, padding=16)
    pred_env = torch.sqrt(pred_smooth + eps)
    target_env = torch.sqrt(target_smooth + eps)
    return torch.mean(torch.abs(pred_env - target_env))


def denoise_loss(
    mixture: torch.Tensor,
    clean: torch.Tensor,
    artifact: torch.Tensor,
    prediction: torch.Tensor,
    snr_db: torch.Tensor,
    config: DenoiseLossConfig,
    fft_weights: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the clean/artifact/consistency/spectral/envelope loss."""

    clean_hat = prediction[:, 0:1, :]
    if prediction.shape[1] > 1:
        artifact_hat = prediction[:, 1:2, :]
    else:
        artifact_hat = torch.zeros_like(artifact)
    weight = snr_weight(snr_db, config.snr_min, config.snr_max).view(-1, 1, 1)

    lx = torch.mean(weight * charbonnier(clean_hat - clean, config.charbonnier_eps))
    la = torch.mean(weight * charbonnier(artifact_hat - artifact, config.charbonnier_eps))
    lc = torch.mean(torch.abs((clean_hat + artifact_hat) - mixture))
    ls = band_weighted_logfft_loss(clean_hat, clean, fft_weights, delta=config.fft_delta)
    if config.eta > 0:
        le = envelope_loss_artifact(artifact_hat, artifact)
    else:
        le = torch.tensor(0.0, device=mixture.device)

    loss = lx + config.alpha * la + config.beta * lc + config.gamma * ls + config.eta * le
    parts = {
        "Lx": float(lx.detach().cpu()),
        "La": float(la.detach().cpu()),
        "Lc": float(lc.detach().cpu()),
        "Ls": float(ls.detach().cpu()),
        "Le": float(le.detach().cpu()),
    }
    return loss, parts
