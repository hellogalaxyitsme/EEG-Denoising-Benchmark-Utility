

from __future__ import annotations

from typing import Any

import numpy as np

try:  # pragma: no cover - availability differs across machines
    from scipy.signal import welch

    HAVE_SCIPY = True
except Exception:  # pragma: no cover
    welch = None
    HAVE_SCIPY = False


def cc_np(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = x - x.mean(axis=-1, keepdims=True)
    y = y - y.mean(axis=-1, keepdims=True)
    num = np.sum(x * y, axis=-1)
    den = np.sqrt(np.sum(x * x, axis=-1) * np.sum(y * y, axis=-1) + eps)
    return num / den


def mse_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.mean((x - y) ** 2, axis=-1)


def rmse_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sqrt(mse_np(x, y))


def t_rrmse_np(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.linalg.norm(x - y, axis=-1) / (np.linalg.norm(x, axis=-1) + eps)


def sdr_db_np(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    num = np.sum(x * x, axis=-1)
    den = np.sum((x - y) ** 2, axis=-1) + eps
    return 10 * np.log10((num + eps) / den)


def psd_np(sig: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    sig = np.asarray(sig)
    if HAVE_SCIPY:
        nperseg = min(256, sig.shape[-1])
        freq, power = welch(sig, fs=fs, nperseg=nperseg, axis=-1)
        return freq, power
    spectrum = np.fft.rfft(sig, axis=-1)
    power = (np.abs(spectrum) ** 2) / sig.shape[-1]
    freq = np.fft.rfftfreq(sig.shape[-1], d=1 / fs)
    return freq, power


def psd_kld_np(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = p / (np.sum(p, axis=-1, keepdims=True) + eps)
    q = q / (np.sum(q, axis=-1, keepdims=True) + eps)
    return np.sum(p * np.log((p + eps) / (q + eps)), axis=-1)


def psd_wd_np(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = p / (np.sum(p, axis=-1, keepdims=True) + eps)
    q = q / (np.sum(q, axis=-1, keepdims=True) + eps)
    cdf_p = np.cumsum(p, axis=-1)
    cdf_q = np.cumsum(q, axis=-1)
    return np.sum(np.abs(cdf_p - cdf_q), axis=-1)


def s_rrmse_from_psd_np(p_x: np.ndarray, p_y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.linalg.norm(p_x - p_y, axis=-1) / (np.linalg.norm(p_x, axis=-1) + eps)


def _mean(value: np.ndarray) -> float:
    return float(np.mean(value))


def compute_denoising_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    fs: float,
) -> dict[str, Any]:


    target = np.asarray(target, dtype=np.float32)
    prediction = np.asarray(prediction, dtype=np.float32)
    if target.shape != prediction.shape:
        raise ValueError(f"target/prediction shape mismatch: {target.shape} vs {prediction.shape}")

    _, p_target = psd_np(target, fs=fs)
    _, p_prediction = psd_np(prediction, fs=fs)

    return {
        "n_samples": int(target.shape[0]),
        "length": int(target.shape[-1]),
        "MSE": _mean(mse_np(target, prediction)),
        "RMSE": _mean(rmse_np(target, prediction)),
        "T_RRMSE": _mean(t_rrmse_np(target, prediction)),
        "S_RRMSE": _mean(s_rrmse_from_psd_np(p_target, p_prediction)),
        "CC": _mean(cc_np(target, prediction)),
        "SDR": _mean(sdr_db_np(target, prediction)),
        "PSD_KLD": _mean(psd_kld_np(p_target, p_prediction)),
        "PSD_WD": _mean(psd_wd_np(p_target, p_prediction)),
    }
