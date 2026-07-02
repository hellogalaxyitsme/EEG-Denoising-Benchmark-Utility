"""Load synthetic EEGDenoiseNet-style split arrays."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def _as_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        obj = obj.item()
    if isinstance(obj, dict):
        return dict(obj)
    if isinstance(obj, (tuple, list)):
        keys = ["Y", "X", "A", "sigma_y", "snr_db", "lambda"]
        return {key: value for key, value in zip(keys, obj)}
    if isinstance(obj, np.ndarray):
        return {"Y": obj}
    raise TypeError(f"Unsupported split format: {type(obj)}")


def _cast_pack(pack: dict[str, Any]) -> dict[str, Any]:
    for key in ["Y", "X", "A"]:
        if key in pack and pack[key] is not None:
            pack[key] = np.asarray(pack[key], dtype=np.float32)
    for key in ["sigma_y", "snr_db", "lambda"]:
        if key in pack and pack[key] is not None:
            pack[key] = np.asarray(pack[key], dtype=np.float32)
    return pack


def _load_npy(path: Path) -> dict[str, Any]:
    return _cast_pack(_as_dict(np.load(path, allow_pickle=True)))


def _load_npz(path: Path, split: str) -> dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    if {"Y", "X", "A"}.issubset(set(z.files)):
        return _cast_pack({key: z[key] for key in z.files})
    if split in z.files:
        return _cast_pack(_as_dict(z[split]))
    raise KeyError(f"Could not find split '{split}' or Y/X/A arrays in {path}.")


def _load_chunk_dir(path: Path) -> dict[str, Any]:
    arrays: dict[str, list[np.ndarray]] = {}
    for chunk in sorted(path.glob("chunk_*.npz")):
        z = np.load(chunk, allow_pickle=True)
        for key in ["Y", "X", "A", "sigma_y", "snr_db"]:
            if key in z.files:
                arrays.setdefault(key, []).append(np.asarray(z[key], dtype=np.float32))
    if not arrays:
        raise FileNotFoundError(f"No chunk_*.npz files found in {path}.")
    return {key: np.concatenate(values, axis=0) for key, values in arrays.items()}


def load_synthetic_split(path: str | Path, split: str = "test") -> dict[str, Any]:
    """Load one split from a split directory, an `.npy`, an `.npz`, or chunk dir."""

    path = Path(path)
    if path.is_dir():
        split_npy = path / f"{split}.npy"
        split_dir = path / split
        if split_npy.exists():
            return _load_npy(split_npy)
        if split_dir.exists():
            return _load_chunk_dir(split_dir)
        if list(path.glob("chunk_*.npz")):
            return _load_chunk_dir(path)
        raise FileNotFoundError(f"No supported split files found under {path}.")
    if path.suffix == ".npy":
        return _load_npy(path)
    if path.suffix == ".npz":
        return _load_npz(path, split=split)
    raise ValueError(f"Unsupported dataset path: {path}")


class SyntheticArrayDataset(Dataset):
    """Torch dataset for normalized synthetic packs with Y/X/A arrays."""

    def __init__(self, pack: dict[str, Any], include_metadata: bool = False) -> None:
        missing = [key for key in ["Y", "X", "A"] if key not in pack]
        if missing:
            raise KeyError(f"Missing required arrays: {missing}")
        self.Y = np.asarray(pack["Y"], dtype=np.float32)
        self.X = np.asarray(pack["X"], dtype=np.float32)
        self.A = np.asarray(pack["A"], dtype=np.float32)
        if self.Y.shape != self.X.shape or self.Y.shape != self.A.shape:
            raise ValueError(f"Y/X/A shape mismatch: {self.Y.shape}, {self.X.shape}, {self.A.shape}")
        if self.Y.ndim != 2:
            raise ValueError(f"Expected Y/X/A arrays with shape (N, T), got {self.Y.shape}")
        self.snr_db = pack.get("snr_db")
        self.sigma_y = pack.get("sigma_y")
        self.include_metadata = include_metadata

    def __len__(self) -> int:
        return int(self.Y.shape[0])

    def __getitem__(self, index: int):
        y = torch.from_numpy(self.Y[index])[None, :]
        x = torch.from_numpy(self.X[index])[None, :]
        a = torch.from_numpy(self.A[index])[None, :]
        if self.include_metadata:
            snr = 0.0 if self.snr_db is None else float(self.snr_db[index])
            sigma = 1.0 if self.sigma_y is None else float(self.sigma_y[index])
            return (
                y,
                x,
                a,
                torch.tensor(snr, dtype=torch.float32),
                torch.tensor(sigma, dtype=torch.float32),
            )
        return y, x, a
