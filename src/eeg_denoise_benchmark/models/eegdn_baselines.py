"""PyTorch ports of the canonical EEGDenoiseNet benchmark baselines.

The source architectures are the `Complex_CNN` and `RNN_lstm` definitions from
NCCLab's EEGDenoiseNet benchmark code. They are intentionally kept close to the
reference Keras layer order so parameter counts match the reported baselines for
512-sample windows.
"""

from __future__ import annotations

import torch
from torch import nn


class _ResidualBlock(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size, padding=padding, bias=True),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size, padding=padding, bias=True),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size, padding=padding, bias=True),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + x


class _MultiKernelBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.branch3 = nn.Sequential(_ResidualBlock(3), _ResidualBlock(3))
        self.branch5 = nn.Sequential(_ResidualBlock(5), _ResidualBlock(5))
        self.branch7 = nn.Sequential(_ResidualBlock(7), _ResidualBlock(7))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.branch3(x), self.branch5(x), self.branch7(x)], dim=1)


class EEGDNComplexCNN(nn.Module):
    """EEGDenoiseNet `Complex_CNN(datanum)` baseline."""

    def __init__(self, datanum: int = 512) -> None:
        super().__init__()
        self.datanum = int(datanum)
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2, bias=True),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            _MultiKernelBlock(),
            nn.Conv1d(96, 32, 1, bias=True),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.head = nn.Linear(32 * self.datanum, self.datanum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.datanum:
            raise ValueError(f"Expected {self.datanum} samples, got {x.shape[-1]}")
        y = self.features(x)
        y = self.head(torch.flatten(y, start_dim=1))
        return y.unsqueeze(1)


class EEGDNRNNLSTM(nn.Module):
    """EEGDenoiseNet `RNN_lstm(datanum)` baseline."""

    def __init__(self, datanum: int = 512, dropout: float = 0.3) -> None:
        super().__init__()
        self.datanum = int(datanum)
        self.lstm = nn.LSTM(input_size=1, hidden_size=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(self.datanum, self.datanum),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.datanum, self.datanum),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.datanum, self.datanum),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.datanum:
            raise ValueError(f"Expected {self.datanum} samples, got {x.shape[-1]}")
        y = x.transpose(1, 2)
        y, _ = self.lstm(y)
        y = self.head(torch.flatten(y, start_dim=1))
        return y.unsqueeze(1)


def build_eegdn_baseline(model_type: str, datanum: int) -> nn.Module:

    normalized = model_type.strip().lower()
    if normalized in {"eegdn_cnn", "eegdn_complex_cnn", "complex_cnn"}:
        return EEGDNComplexCNN(datanum=datanum)
    if normalized in {"eegdn_rnn", "eegdn_lstm", "rnn_lstm"}:
        return EEGDNRNNLSTM(datanum=datanum)
    raise ValueError(f"Unsupported EEGDN baseline model_type={model_type!r}")
