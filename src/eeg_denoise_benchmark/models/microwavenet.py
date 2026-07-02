"""Controlled-pipeline PyTorch port of the MicroWaveNet architecture.

The layer structure follows the public MicroWaveNet implementation:
The original repository trains a dual-head model with additional Soft-DTW and wavelet-domain losses. In this
project we expose the clean-EEG head as a standard denoiser so it can be trained
under the same loss, splits, and metrics as the other controlled baselines.
"""

from __future__ import annotations

import torch
from torch import nn


class _LayerNorm1D(nn.Module):
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.layer_norm(x)
        return x.permute(0, 2, 1)


class _ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16) -> None:
        super().__init__()
        hidden = max(1, in_planes // ratio)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(in_planes, hidden, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(hidden, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class _SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class _CBAM(nn.Module):
    def __init__(self, in_channels: int, ratio: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        self.channel_attention = _ChannelAttention(in_channels, ratio)
        self.spatial_attention = _SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.channel_attention(x) * x
        return self.spatial_attention(out) * out


class _Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.skip_states: list[torch.Tensor] = []
        self.eeg_skip_states: list[torch.Tensor] = []
        self.art_skip_states: list[torch.Tensor] = []
        self.eeg_cbam_skip_states: list[torch.Tensor] = []
        self.art_cbam_skip_states: list[torch.Tensor] = []

        self.pre_wavelet = nn.Sequential(
            nn.Conv1d(1, 3, 3, padding=1, padding_mode="reflect"),
            nn.BatchNorm1d(3),
            nn.LeakyReLU(),
        )
        self.wavelet_1_stream = nn.Sequential(
            nn.Conv1d(1, 2, 21, padding=10, padding_mode="reflect"),
            nn.BatchNorm1d(2),
            nn.LeakyReLU(),
            nn.Conv1d(2, 4, 9, padding=4, padding_mode="reflect"),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(),
        )
        self.wavelet_2_stream = nn.Sequential(
            nn.Conv1d(1, 2, 11, padding=5, padding_mode="reflect"),
            nn.BatchNorm1d(2),
            nn.LeakyReLU(),
            nn.Conv1d(2, 4, 7, padding=3, padding_mode="reflect"),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(),
        )
        self.wavelet_3_stream = nn.Sequential(
            nn.Conv1d(1, 2, 5, padding=2, padding_mode="reflect"),
            nn.BatchNorm1d(2),
            nn.LeakyReLU(),
            nn.Conv1d(2, 4, 3, padding=1, padding_mode="reflect"),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(),
        )

        self.wavelet_1_stream_post_loss = nn.Sequential(
            nn.Conv1d(4, 8, 15, stride=2, padding=7, padding_mode="reflect"),
            nn.BatchNorm1d(8),
            nn.MaxPool1d(3, stride=2, padding=1),
            nn.Conv1d(8, 8, 9, padding=4, padding_mode="reflect"),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),
            nn.MaxPool1d(3, stride=2, padding=1),
            nn.Dropout(0.1),
        )
        self.wavelet_2_stream_post_loss = nn.Sequential(
            nn.Conv1d(4, 8, 9, stride=2, padding=4, padding_mode="reflect"),
            nn.BatchNorm1d(8),
            nn.MaxPool1d(3, stride=2, padding=1),
            nn.Conv1d(8, 8, 5, padding=2, padding_mode="reflect"),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),
            nn.MaxPool1d(3, stride=2, padding=1),
            nn.Dropout(0.1),
        )
        self.wavelet_3_stream_post_loss = nn.Sequential(
            nn.Conv1d(4, 8, 5, stride=2, padding=2, padding_mode="reflect"),
            nn.BatchNorm1d(8),
            nn.MaxPool1d(3, stride=2, padding=1),
            nn.Conv1d(8, 8, 3, padding=1, padding_mode="reflect"),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),
            nn.MaxPool1d(3, stride=2, padding=1),
            nn.Dropout(0.1),
        )

        self.pre_latent0_cbam = nn.Sequential(_CBAM(24, ratio=8), _LayerNorm1D(24), nn.LeakyReLU())
        self.pre_latent0 = nn.Sequential(
            nn.Conv1d(24, 32, 5, padding=2, padding_mode="reflect"),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
        )
        self.pre_latent1_eeg_cbam = nn.Sequential(_CBAM(32, ratio=16), _LayerNorm1D(32), nn.LeakyReLU())
        self.pre_latent1_art_cbam = nn.Sequential(_CBAM(32, ratio=16), _LayerNorm1D(32), nn.LeakyReLU())
        self.pre_latent1_eeg = nn.Sequential(
            nn.Conv1d(32, 64, 5, stride=2, padding=2, padding_mode="reflect", groups=8),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.15),
        )
        self.pre_latent1_art = nn.Sequential(
            nn.Conv1d(32, 64, 5, stride=2, padding=2, padding_mode="reflect", groups=8),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.15),
        )
        self.pre_latent2_eeg_cbam = nn.Sequential(_CBAM(64, ratio=32), _LayerNorm1D(64), nn.LeakyReLU())
        self.pre_latent2_art_cbam = nn.Sequential(_CBAM(64, ratio=32), _LayerNorm1D(64), nn.LeakyReLU())
        self.pre_latent2_eeg = nn.Sequential(
            nn.Conv1d(64, 128, 5, stride=2, padding=2, padding_mode="reflect", groups=16),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
        )
        self.pre_latent2_art = nn.Sequential(
            nn.Conv1d(64, 128, 5, stride=2, padding=2, padding_mode="reflect", groups=32),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
        )
        self.pre_latent3_eeg_cbam = nn.Sequential(_CBAM(128, ratio=64), _LayerNorm1D(128), nn.LeakyReLU())
        self.pre_latent3_art_cbam = nn.Sequential(_CBAM(128, ratio=64), _LayerNorm1D(128), nn.LeakyReLU())
        self.pre_latent3_eeg = nn.Sequential(nn.MaxPool1d(3, stride=2, padding=1), nn.LeakyReLU(), nn.Dropout(0.25))
        self.pre_latent3_art = nn.Sequential(nn.MaxPool1d(3, stride=2, padding=1), nn.LeakyReLU(), nn.Dropout(0.25))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.skip_states = []
        self.eeg_skip_states = []
        self.art_skip_states = []
        self.eeg_cbam_skip_states = []
        self.art_cbam_skip_states = []

        low_in, mid_in, high_in = torch.split(self.pre_wavelet(x), 1, dim=1)
        low = self.wavelet_1_stream(low_in)
        mid = self.wavelet_2_stream(mid_in)
        high = self.wavelet_3_stream(high_in)
        self.skip_states.append(torch.cat([low, mid, high], dim=1))

        low = self.wavelet_1_stream_post_loss(low)
        mid = self.wavelet_2_stream_post_loss(mid)
        high = self.wavelet_3_stream_post_loss(high)
        self.skip_states.append(torch.cat([low, mid, high], dim=1))

        x = torch.cat([low, mid, high], dim=1)
        x = self.pre_latent0_cbam(x)
        self.eeg_cbam_skip_states.append(x.clone())
        self.art_cbam_skip_states.append(x.clone())

        x = self.pre_latent0(x)
        self.skip_states.append(x.clone())

        x_eeg = self.pre_latent1_eeg_cbam(x)
        x_art = self.pre_latent1_art_cbam(x)
        self.eeg_cbam_skip_states.append(x_eeg.clone())
        self.art_cbam_skip_states.append(x_art.clone())
        x_eeg = self.pre_latent1_eeg(x)
        x_art = self.pre_latent1_art(x)
        self.eeg_skip_states.append(x_eeg.clone())
        self.art_skip_states.append(x_art.clone())

        x_eeg = self.pre_latent2_eeg_cbam(x_eeg)
        x_art = self.pre_latent2_art_cbam(x_art)
        self.eeg_cbam_skip_states.append(x_eeg.clone())
        self.art_cbam_skip_states.append(x_art.clone())
        x_eeg = self.pre_latent2_eeg(x_eeg)
        x_art = self.pre_latent2_art(x_art)
        self.eeg_skip_states.append(x_eeg.clone())
        self.art_skip_states.append(x_art.clone())

        x_eeg = self.pre_latent3_eeg(self.pre_latent3_eeg_cbam(x_eeg))
        x_art = self.pre_latent3_art(self.pre_latent3_art_cbam(x_art))
        return x_eeg, x_art


class _Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.post_latent3 = nn.Sequential(
            nn.ConvTranspose1d(128, 128, 4, stride=2, padding=1, groups=16),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
        )
        self.post_latent3_cbam = nn.Sequential(_CBAM(128, ratio=64), _LayerNorm1D(128), nn.LeakyReLU())
        self.post_latent2 = nn.Sequential(
            nn.Conv1d(256, 128, 3, padding=1, padding_mode="reflect", groups=64),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(128, 64, 6, stride=2, padding=2, groups=32),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
        )
        self.post_latent2_cbam = nn.Sequential(
            nn.Conv1d(128, 64, 1, groups=32),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            _CBAM(64, ratio=32),
            _LayerNorm1D(64),
            nn.LeakyReLU(),
        )
        self.post_latent1 = nn.Sequential(
            nn.Conv1d(128, 64, 5, padding=2, padding_mode="reflect", groups=32),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(64, 32, 6, stride=2, padding=2, groups=16),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout(0.15),
        )
        self.post_latent1_cbam = nn.Sequential(
            nn.Conv1d(64, 32, 1, groups=16),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            _CBAM(32, ratio=16),
            _LayerNorm1D(32),
            nn.LeakyReLU(),
        )
        self.post_latent0 = nn.Sequential(
            nn.Conv1d(64, 32, 5, padding=2, padding_mode="reflect", groups=16),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Conv1d(32, 24, 5, padding=2, groups=8),
            nn.BatchNorm1d(24),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
        )
        self.post_latent0_cbam = nn.Sequential(
            nn.Conv1d(48, 24, 1, groups=8),
            nn.BatchNorm1d(24),
            nn.LeakyReLU(),
            _CBAM(24, ratio=8),
            _LayerNorm1D(24),
            nn.LeakyReLU(),
        )
        self.wavelet_1_stream_pre_loss = nn.Sequential(
            nn.ConvTranspose1d(16, 8, 6, stride=2, padding=2),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(8, 4, 14, stride=4, padding=5),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
        )
        self.wavelet_2_stream_pre_loss = nn.Sequential(
            nn.ConvTranspose1d(16, 8, 6, stride=2, padding=2),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(8, 4, 10, stride=4, padding=3),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
        )
        self.wavelet_3_stream_pre_loss = nn.Sequential(
            nn.ConvTranspose1d(16, 8, 6, stride=2, padding=2),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(8, 4, 6, stride=4, padding=1),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
        )
        self.wavelet_1_stream_post_loss = nn.Sequential(
            nn.ConvTranspose1d(8, 2, 21, padding=10),
            nn.BatchNorm1d(2),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(2, 1, 9, padding=4),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(),
        )
        self.wavelet_2_stream_post_loss = nn.Sequential(
            nn.ConvTranspose1d(8, 2, 11, padding=5),
            nn.BatchNorm1d(2),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(2, 1, 7, padding=3),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(),
        )
        self.wavelet_3_stream_post_loss = nn.Sequential(
            nn.ConvTranspose1d(8, 2, 5, padding=2),
            nn.BatchNorm1d(2),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(2, 1, 3, padding=1),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(),
        )
        self.post_wavelet = nn.ConvTranspose1d(3, 1, 3, padding=1)
        self.window_weights = nn.Sequential(
            nn.Conv1d(1, 1, 16, padding=8),
            nn.AvgPool1d(16, stride=16),
        )

    def forward(
        self,
        x: torch.Tensor,
        skip_states: list[torch.Tensor],
        eegart_skip_states: list[torch.Tensor],
        cbam_skip_states: list[torch.Tensor],
        original: torch.Tensor,
    ) -> torch.Tensor:
        x = self.post_latent3_cbam(self.post_latent3(x))
        x = self.post_latent2(torch.cat([x, eegart_skip_states[-1]], dim=1))
        x = self.post_latent2_cbam(torch.cat([x, cbam_skip_states[-1]], dim=1))
        x = self.post_latent1(torch.cat([x, eegart_skip_states[-2]], dim=1))
        x = self.post_latent1_cbam(torch.cat([x, cbam_skip_states[-2]], dim=1))
        x = self.post_latent0(torch.cat([x, skip_states[-1]], dim=1))
        x = self.post_latent0_cbam(torch.cat([x, cbam_skip_states[-3]], dim=1))

        low, mid, high = torch.split(x, 8, dim=1)
        skip_low, skip_mid, skip_high = torch.split(skip_states[-2], 8, dim=1)
        low = self.wavelet_1_stream_pre_loss(torch.cat([low, skip_low], dim=1))
        mid = self.wavelet_2_stream_pre_loss(torch.cat([mid, skip_mid], dim=1))
        high = self.wavelet_3_stream_pre_loss(torch.cat([high, skip_high], dim=1))

        skip_low, skip_mid, skip_high = torch.split(skip_states[-3], 4, dim=1)
        low = self.wavelet_1_stream_post_loss(torch.cat([low, skip_low], dim=1))
        mid = self.wavelet_2_stream_post_loss(torch.cat([mid, skip_mid], dim=1))
        high = self.wavelet_3_stream_post_loss(torch.cat([high, skip_high], dim=1))

        y = self.post_wavelet(torch.cat([low, mid, high], dim=1))
        if y.shape[-1] != original.shape[-1]:
            y = y[..., : original.shape[-1]]

        diff = original - y
        weights = self.window_weights(diff)
        target_windows = y.shape[-1] // 16
        weights = weights[..., :target_windows]
        batch, channels, samples = y.shape
        trimmed = y[..., : target_windows * 16]
        weighted = trimmed.view(batch, channels, target_windows, 16) * weights[..., None]
        y = weighted.view(batch, channels, target_windows * 16)
        if y.shape[-1] < samples:
            y = torch.nn.functional.pad(y, (0, samples - y.shape[-1]))
        return y


class MicroWaveNet(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.encoder = _Encoder()
        self.eeg_decoder = _Decoder()
        self.artifact_decoder = _Decoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original = x.clone()
        eeg_z, art_z = self.encoder(x)
        eeg = self.eeg_decoder(
            eeg_z,
            self.encoder.skip_states,
            self.encoder.eeg_skip_states,
            self.encoder.eeg_cbam_skip_states,
            original,
        )
        # Keep the artifact branch active during training so this port matches
        # the reference dual-decoder topology, even though the controlled loss
        # supervises only the clean output through the project trainer.
        _artifact = self.artifact_decoder(
            art_z,
            self.encoder.skip_states,
            self.encoder.art_skip_states,
            self.encoder.art_cbam_skip_states,
            original,
        )
        return eeg
