"""Checkpoint helpers for denoising model artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from eeg_denoise_benchmark.models import MicroWaveNet, TinyDenoiser, build_eegdn_baseline, count_trainable_parameters


STATE_KEYS = ("model", "model_state", "model_state_dict", "state_dict")


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    """Load a PyTorch checkpoint from disk."""

    return torch.load(Path(path), map_location=map_location)


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Return a model state dict from common checkpoint formats."""

    if isinstance(checkpoint, dict):
        for key in STATE_KEYS:
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint
    raise TypeError("Could not find a model state dict in checkpoint.")


def model_config_from_checkpoint(checkpoint: Any) -> dict[str, Any]:
    """Extract the architecture subset needed to instantiate TinyDenoiser."""

    cfg = checkpoint.get("cfg", {}) if isinstance(checkpoint, dict) else {}
    return {
        "base": int(cfg.get("base", 16)),
        "extra_bottleneck_blocks": int(cfg.get("extra_bottleneck_blocks", cfg.get("extra_blocks", 2))),
        "use_dwt": bool(cfg.get("use_dwt", True)),
        "use_gate": bool(cfg.get("use_gate", True)),
        "use_attn": bool(cfg.get("use_attn", True)),
        "use_artifact_head": bool(cfg.get("use_artifact_head", True)),
        "conv_block": str(cfg.get("conv_block", "ds")),
    }


def build_model_from_checkpoint(checkpoint: Any) -> torch.nn.Module:
    """Instantiate a denoiser using checkpoint metadata when present."""

    cfg = checkpoint.get("cfg", {}) if isinstance(checkpoint, dict) else {}
    model_type = str(cfg.get("model_type", "controlled_backbone")).strip().lower()
    if model_type in {"controlled_backbone", "tinydenoiser", "tiny_denoiser"}:
        return TinyDenoiser(**model_config_from_checkpoint(checkpoint))
    if model_type in {"microwavenet", "micro_wave_net"}:
        return MicroWaveNet()
    if model_type.startswith("eegdn") or model_type in {"complex_cnn", "rnn_lstm"}:
        return build_eegdn_baseline(model_type=model_type, datanum=int(cfg.get("datanum", cfg.get("length", 512))))
    raise ValueError(f"Unsupported checkpoint model_type={model_type!r}")


def load_model_from_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> tuple[torch.nn.Module, Any]:
    """Load a checkpoint and return the instantiated model with metadata."""

    checkpoint = load_checkpoint(path, map_location=map_location)
    model = build_model_from_checkpoint(checkpoint)
    state = extract_state_dict(checkpoint)
    model.load_state_dict(state, strict=strict)
    return model, checkpoint


def checkpoint_summary(path: str | Path) -> dict[str, Any]:
    """Return a JSON-friendly summary without running inference."""

    path = Path(path)
    checkpoint = load_checkpoint(path, map_location="cpu")
    state = extract_state_dict(checkpoint)
    tensor_numel = int(sum(v.numel() for v in state.values() if torch.is_tensor(v)))
    tensor_keys = list(state.keys())

    summary: dict[str, Any] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "checkpoint_type": type(checkpoint).__name__,
        "state_key_count": len(tensor_keys),
        "state_tensor_numel_including_buffers": tensor_numel,
        "state_key_sample": tensor_keys[:20],
    }
    if isinstance(checkpoint, dict):
        summary["top_level_keys"] = list(checkpoint.keys())
        if "cfg" in checkpoint:
            summary["cfg"] = checkpoint["cfg"]
        if "test" in checkpoint:
            summary["embedded_test"] = checkpoint["test"]
        if "seed" in checkpoint:
            summary["seed"] = checkpoint["seed"]

    try:
        model = build_model_from_checkpoint(checkpoint)
        model.load_state_dict(state, strict=True)
        summary["model_load_strict"] = True
        summary["trainable_parameters"] = count_trainable_parameters(model)
    except Exception as exc:  # pragma: no cover - diagnostic path
        summary["model_load_strict"] = False
        summary["model_load_error"] = str(exc)

    return summary
