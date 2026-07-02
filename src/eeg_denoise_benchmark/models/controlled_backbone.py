"""Controlled depthwise-separable U-Net models for EEG denoising."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class HaarDWT1D(nn.Module):
    """Fixed one-level Haar analysis filter bank for single-channel EEG."""

    def __init__(self) -> None:
        super().__init__()
        h = torch.tensor([1.0, 1.0]) / (2**0.5)
        g = torch.tensor([1.0, -1.0]) / (2**0.5)
        self.register_buffer("h", h.view(1, 1, 2))
        self.register_buffer("g", g.view(1, 1, 2))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        low = F.conv1d(x, self.h, stride=2, padding=0)
        high = F.conv1d(x, self.g, stride=2, padding=0)
        return low, high


class ECA1D(nn.Module):
    """Efficient channel attention block."""

    def __init__(self, channels: int, k: int = 3) -> None:
        super().__init__()
        del channels
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg(x).transpose(1, 2)
        y = self.conv(y).transpose(1, 2)
        return x * torch.sigmoid(y)


class DSConv1D(nn.Module):
    """Depthwise-separable 1D convolution block with optional ECA."""

    def __init__(
        self,
        cin: int,
        cout: int,
        k: int = 9,
        dilation: int = 1,
        use_attn: bool = True,
    ) -> None:
        super().__init__()
        pad = (k // 2) * dilation
        self.dw = nn.Conv1d(
            cin,
            cin,
            k,
            padding=pad,
            dilation=dilation,
            groups=cin,
            bias=False,
        )
        self.pw = nn.Conv1d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm1d(cout)
        self.act = nn.SiLU()
        self.attn = ECA1D(cout) if use_attn else nn.Identity()
        self.res = cin == cout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.bn(self.pw(self.dw(x))))
        y = self.attn(y)
        return y + x if self.res else y


class StandardConv1D(nn.Module):
    """Standard 1D convolution block used for DSConv ablation."""

    def __init__(
        self,
        cin: int,
        cout: int,
        k: int = 9,
        dilation: int = 1,
        use_attn: bool = True,
    ) -> None:
        super().__init__()
        pad = (k // 2) * dilation
        self.conv = nn.Conv1d(
            cin,
            cout,
            k,
            padding=pad,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(cout)
        self.act = nn.SiLU()
        self.attn = ECA1D(cout) if use_attn else nn.Identity()
        self.res = cin == cout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.bn(self.conv(x)))
        y = self.attn(y)
        return y + x if self.res else y


class TinyDenoiser(nn.Module):
    """Single-channel controlled DSConv U-Net denoiser.

    Input shape is ``(B, 1, T)`` and output shape is ``(B, 2, T)`` where channel
    0 is the predicted clean EEG and channel 1 is the predicted artifact.
    """

    def __init__(
        self,
        base: int = 16,
        extra_bottleneck_blocks: int = 2,
        use_dwt: bool = True,
        use_gate: bool = True,
        use_attn: bool = True,
        use_artifact_head: bool = True,
        conv_block: str = "ds",
    ) -> None:
        super().__init__()
        if conv_block not in {"ds", "standard"}:
            raise ValueError(f"Unsupported conv_block={conv_block!r}; expected 'ds' or 'standard'")
        self.use_dwt = use_dwt
        self.use_gate = use_gate and use_artifact_head
        self.use_artifact_head = use_artifact_head
        block_cls = DSConv1D if conv_block == "ds" else StandardConv1D
        self.dwt = HaarDWT1D()
        self.stem = nn.Conv1d(3 if use_dwt else 1, base, 1, bias=False)

        self.e1 = block_cls(base, base, k=9, dilation=1, use_attn=use_attn)
        self.down1 = nn.Conv1d(base, base * 2, 4, stride=2, padding=1, bias=False)

        self.e2 = block_cls(base * 2, base * 2, k=9, dilation=2, use_attn=use_attn)
        self.down2 = nn.Conv1d(base * 2, base * 4, 4, stride=2, padding=1, bias=False)

        blocks = [block_cls(base * 4, base * 4, k=9, dilation=4, use_attn=use_attn)]
        if extra_bottleneck_blocks >= 1:
            blocks.append(block_cls(base * 4, base * 4, k=9, dilation=8, use_attn=use_attn))
        if extra_bottleneck_blocks >= 2:
            blocks.append(block_cls(base * 4, base * 4, k=9, dilation=16, use_attn=use_attn))
        self.b = nn.Sequential(*blocks)

        self.up2 = nn.ConvTranspose1d(base * 4, base * 2, 4, stride=2, padding=1, bias=False)
        self.d2 = block_cls(base * 4, base * 2, k=9, dilation=2, use_attn=use_attn)

        self.up1 = nn.ConvTranspose1d(base * 2, base, 4, stride=2, padding=1, bias=False)
        self.d1 = block_cls(base * 2, base, k=9, dilation=1, use_attn=use_attn)

        self.out = nn.Conv1d(base, 2 if use_artifact_head else 1, 1)
        self.gate = (
            nn.Sequential(
                nn.Conv1d(base, base, 1, bias=False),
                nn.SiLU(),
                nn.Conv1d(base, 1, 1, bias=True),
                nn.Sigmoid(),
            )
            if use_gate
            else nn.Identity()
        )

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_dwt:
            return x
        l1, h1 = self.dwt(x)
        _, h2 = self.dwt(l1)
        t = x.shape[-1]
        h1u = F.interpolate(h1, size=t, mode="linear", align_corners=False)
        h2u = F.interpolate(h2, size=t, mode="linear", align_corners=False)
        return torch.cat([x, h1u, h2u], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._features(x)
        x0 = self.stem(feats)
        s1 = self.e1(x0)
        x1 = self.down1(s1)

        s2 = self.e2(x1)
        x2 = self.down2(s2)

        xb = self.b(x2)

        u2 = self.up2(xb)
        d2 = self.d2(torch.cat([u2, s2], dim=1))

        u1 = self.up1(d2)
        d1 = self.d1(torch.cat([u1, s1], dim=1))

        y2 = self.out(d1)
        clean = y2[:, 0:1, :]
        if not self.use_artifact_head:
            return clean
        artifact = y2[:, 1:2, :]
        if self.use_gate:
            artifact = artifact * self.gate(d1)
        return torch.cat([clean, artifact], dim=1)


class LinearFIRDenoiser(nn.Module):
    """Single learned FIR filter baseline for clean EEG prediction."""

    def __init__(self, kernel_size: int = 33) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve temporal length")
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TinyTCNDenoiser(nn.Module):
    """Sub-1k parameter temporal convolutional denoising baseline."""

    def __init__(self, hidden: int = 4, kernel_size: int = 9) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve temporal length")
        layers: list[nn.Module] = []
        cin = 1
        for dilation in (1, 2, 4):
            padding = (kernel_size // 2) * dilation
            layers.extend(
                [
                    nn.Conv1d(cin, hidden, kernel_size, padding=padding, dilation=dilation, bias=True),
                    nn.BatchNorm1d(hidden),
                    nn.SiLU(),
                ]
            )
            cin = hidden
        layers.append(nn.Conv1d(hidden, 1, 1, bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PatchTransformerDenoiser(nn.Module):
    """Lightweight patch-Transformer denoiser for capacity-control experiments."""

    def __init__(
        self,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 1,
        dim_feedforward: int | None = None,
        patch_stride: int = 4,
        max_tokens: int = 512,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        if dim_feedforward is None:
            dim_feedforward = d_model * 2
        self.patch_stride = patch_stride
        self.stem = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
        )
        self.patch = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=patch_stride,
            stride=patch_stride,
            bias=False,
        )
        position = torch.arange(max_tokens, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pos = torch.zeros(1, max_tokens, d_model, dtype=torch.float32)
        pos[0, :, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pos[0, :, 1::2] = torch.cos(position * div_term[: pos[0, :, 1::2].shape[-1]])
        self.register_buffer("pos", pos, persistent=False)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.up = nn.ConvTranspose1d(
            d_model,
            d_model,
            kernel_size=patch_stride,
            stride=patch_stride,
            bias=False,
        )
        self.refine = nn.Sequential(
            DSConv1D(d_model, d_model, k=9, dilation=1, use_attn=False),
            nn.Conv1d(d_model, 1, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_length = x.shape[-1]
        remainder = original_length % self.patch_stride
        if remainder:
            pad = self.patch_stride - remainder
            x = F.pad(x, (0, pad), mode="replicate")
        y = self.stem(x)
        tokens = self.patch(y).transpose(1, 2)
        if tokens.shape[1] > self.pos.shape[1]:
            raise ValueError(f"Token length {tokens.shape[1]} exceeds max_tokens={self.pos.shape[1]}")
        tokens = tokens + self.pos[:, : tokens.shape[1], :]
        encoded = self.encoder(tokens).transpose(1, 2)
        up = self.up(encoded)
        clean = self.refine(up)
        return clean[..., :original_length]


def count_trainable_parameters(model: nn.Module) -> int:
    """Return trainable parameter count."""

    return sum(p.numel() for p in model.parameters() if p.requires_grad)
