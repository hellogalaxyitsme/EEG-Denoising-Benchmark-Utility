"""DeepSeparator architecture ported from NCCLab's official implementation.

The original network switches behavior using an indicator: 0 extracts denoised
EEG and 1 extracts artifact.  The wrapper below exposes both as channels so it
can be trained by the shared clean/artifact denoising loss.
"""

from __future__ import annotations

import torch
from torch import nn


class _DeepSeparatorCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1_1_1 = nn.Conv1d(1, 5, kernel_size=3, padding=1)
        self.conv1_1_2 = nn.Conv1d(1, 5, kernel_size=5, padding=2)
        self.conv1_1_3 = nn.Conv1d(1, 5, kernel_size=11, padding=5)
        self.conv1_1_4 = nn.Conv1d(1, 5, kernel_size=15, padding=7)

        self.conv1_2_1 = nn.Conv1d(20, 5, kernel_size=3, padding=1)
        self.conv1_2_2 = nn.Conv1d(20, 5, kernel_size=5, padding=2)
        self.conv1_2_3 = nn.Conv1d(20, 5, kernel_size=11, padding=5)
        self.conv1_2_4 = nn.Conv1d(20, 5, kernel_size=15, padding=7)

        self.conv1_3_1 = nn.Conv1d(20, 5, kernel_size=3, padding=1)
        self.conv1_3_2 = nn.Conv1d(20, 5, kernel_size=5, padding=2)
        self.conv1_3_3 = nn.Conv1d(20, 5, kernel_size=11, padding=5)
        self.conv1_3_4 = nn.Conv1d(20, 5, kernel_size=15, padding=7)

        self.conv1_4_1 = nn.Conv1d(20, 5, kernel_size=3, padding=1)
        self.conv1_4_2 = nn.Conv1d(20, 5, kernel_size=5, padding=2)
        self.conv1_4_3 = nn.Conv1d(20, 5, kernel_size=11, padding=5)
        self.conv1_4_4 = nn.Conv1d(20, 5, kernel_size=15, padding=7)

        self.conv1_squeeze1 = nn.Conv1d(20, 1, kernel_size=1)

        self.conv2_1_1 = nn.Conv1d(1, 5, kernel_size=3, padding=1)
        self.conv2_1_2 = nn.Conv1d(1, 5, kernel_size=5, padding=2)
        self.conv2_1_3 = nn.Conv1d(1, 5, kernel_size=11, padding=5)
        self.conv2_1_4 = nn.Conv1d(1, 5, kernel_size=15, padding=7)

        self.conv2_2_1 = nn.Conv1d(20, 5, kernel_size=3, padding=1)
        self.conv2_2_2 = nn.Conv1d(20, 5, kernel_size=5, padding=2)
        self.conv2_2_3 = nn.Conv1d(20, 5, kernel_size=11, padding=5)
        self.conv2_2_4 = nn.Conv1d(20, 5, kernel_size=15, padding=7)

        self.conv2_3_1 = nn.Conv1d(20, 5, kernel_size=3, padding=1)
        self.conv2_3_2 = nn.Conv1d(20, 5, kernel_size=5, padding=2)
        self.conv2_3_3 = nn.Conv1d(20, 5, kernel_size=11, padding=5)
        self.conv2_3_4 = nn.Conv1d(20, 5, kernel_size=15, padding=7)

        self.conv2_4_1 = nn.Conv1d(20, 5, kernel_size=3, padding=1)
        self.conv2_4_2 = nn.Conv1d(20, 5, kernel_size=5, padding=2)
        self.conv2_4_3 = nn.Conv1d(20, 5, kernel_size=11, padding=5)
        self.conv2_4_4 = nn.Conv1d(20, 5, kernel_size=15, padding=7)

        self.conv1_squeeze2 = nn.Conv1d(20, 1, kernel_size=1)

        self.conv3_1_1 = nn.Conv1d(1, 5, kernel_size=3, padding=1)
        self.conv3_1_2 = nn.Conv1d(1, 5, kernel_size=5, padding=2)
        self.conv3_1_3 = nn.Conv1d(1, 5, kernel_size=11, padding=5)
        self.conv3_1_4 = nn.Conv1d(1, 5, kernel_size=15, padding=7)

        self.conv3_2_1 = nn.Conv1d(20, 5, kernel_size=3, padding=1)
        self.conv3_2_2 = nn.Conv1d(20, 5, kernel_size=5, padding=2)
        self.conv3_2_3 = nn.Conv1d(20, 5, kernel_size=11, padding=5)
        self.conv3_2_4 = nn.Conv1d(20, 5, kernel_size=15, padding=7)

        self.conv3_3_1 = nn.Conv1d(20, 5, kernel_size=3, padding=1)
        self.conv3_3_2 = nn.Conv1d(20, 5, kernel_size=5, padding=2)
        self.conv3_3_3 = nn.Conv1d(20, 5, kernel_size=11, padding=5)
        self.conv3_3_4 = nn.Conv1d(20, 5, kernel_size=15, padding=7)

        self.conv3_4_1 = nn.Conv1d(20, 5, kernel_size=3, padding=1)
        self.conv3_4_2 = nn.Conv1d(20, 5, kernel_size=5, padding=2)
        self.conv3_4_3 = nn.Conv1d(20, 5, kernel_size=11, padding=5)
        self.conv3_4_4 = nn.Conv1d(20, 5, kernel_size=15, padding=7)

        self.conv1_squeeze3 = nn.Conv1d(20, 1, kernel_size=1)

    @staticmethod
    def _multi_kernel(x: torch.Tensor, *layers: nn.Conv1d) -> torch.Tensor:
        return torch.cat([layer(x) for layer in layers], dim=1)

    def forward(self, x: torch.Tensor, indicator: torch.Tensor | float | int) -> torch.Tensor:
        if x.ndim == 3:
            x = x[:, 0, :]
        if not torch.is_tensor(indicator):
            indicator = torch.full((x.shape[0], 1), float(indicator), dtype=x.dtype, device=x.device)
        indicator = indicator.to(device=x.device, dtype=x.dtype)
        if indicator.ndim == 1:
            indicator = indicator.unsqueeze(1)

        emb_x = x.unsqueeze(1)
        emb_x = self._multi_kernel(emb_x, self.conv1_1_1, self.conv1_1_2, self.conv1_1_3, self.conv1_1_4)
        emb_x = torch.relu(emb_x)
        emb_skip_connect_x = emb_x
        emb_x = self._multi_kernel(emb_x, self.conv1_2_1, self.conv1_2_2, self.conv1_2_3, self.conv1_2_4)
        emb_x = torch.sigmoid(emb_x)
        emb_x = self._multi_kernel(emb_x, self.conv1_3_1, self.conv1_3_2, self.conv1_3_3, self.conv1_3_4)
        emb_x = torch.sigmoid(emb_x)
        emb_x = emb_x + emb_skip_connect_x
        emb_x = self._multi_kernel(emb_x, self.conv1_4_1, self.conv1_4_2, self.conv1_4_3, self.conv1_4_4)
        emb_x = self.conv1_squeeze1(emb_x).squeeze(1)

        learnable_atte_x = x.unsqueeze(1)
        learnable_atte_x = self._multi_kernel(
            learnable_atte_x,
            self.conv2_1_1,
            self.conv2_1_2,
            self.conv2_1_3,
            self.conv2_1_4,
        )
        learnable_atte_x = torch.relu(learnable_atte_x)
        atte_skip_connect_x = learnable_atte_x
        learnable_atte_x = self._multi_kernel(
            learnable_atte_x,
            self.conv2_2_1,
            self.conv2_2_2,
            self.conv2_2_3,
            self.conv2_2_4,
        )
        learnable_atte_x = torch.sigmoid(learnable_atte_x)
        learnable_atte_x = self._multi_kernel(
            learnable_atte_x,
            self.conv2_3_1,
            self.conv2_3_2,
            self.conv2_3_3,
            self.conv2_3_4,
        )
        learnable_atte_x = torch.sigmoid(learnable_atte_x)
        learnable_atte_x = learnable_atte_x + atte_skip_connect_x
        learnable_atte_x = self._multi_kernel(
            learnable_atte_x,
            self.conv2_4_1,
            self.conv2_4_2,
            self.conv2_4_3,
            self.conv2_4_4,
        )
        learnable_atte_x = torch.sigmoid(self.conv1_squeeze2(learnable_atte_x)).squeeze(1)

        output = emb_x * torch.abs(indicator - learnable_atte_x)
        output = output.unsqueeze(1)
        output = self._multi_kernel(output, self.conv3_1_1, self.conv3_1_2, self.conv3_1_3, self.conv3_1_4)
        output = torch.relu(output)
        output_skip_connect_x = output
        output = self._multi_kernel(output, self.conv3_2_1, self.conv3_2_2, self.conv3_2_3, self.conv3_2_4)
        output = torch.sigmoid(output)
        output = self._multi_kernel(output, self.conv3_3_1, self.conv3_3_2, self.conv3_3_3, self.conv3_3_4)
        output = torch.sigmoid(output)
        output = output + output_skip_connect_x
        output = self._multi_kernel(output, self.conv3_4_1, self.conv3_4_2, self.conv3_4_3, self.conv3_4_4)
        output = self.conv1_squeeze3(output)
        return output.squeeze(1)


class DeepSeparator(nn.Module):
    """Shared-interface wrapper around the official DeepSeparator architecture."""

    def __init__(self, datanum: int = 512, return_artifact: bool = True) -> None:
        super().__init__()
        self.datanum = int(datanum)
        self.return_artifact = bool(return_artifact)
        self.core = _DeepSeparatorCore()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        clean = self.core(x, 0).unsqueeze(1)
        if not self.return_artifact:
            return clean
        artifact = self.core(x, 1).unsqueeze(1)
        return torch.cat([clean, artifact], dim=1)
