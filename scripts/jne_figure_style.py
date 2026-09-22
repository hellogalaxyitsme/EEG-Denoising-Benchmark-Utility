"""Shared plotting style for JNE figures.

The helper intentionally depends only on matplotlib/numpy/pandas-compatible
objects so final figures remain easy to reproduce on the lab machine.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


WIDTH_FULL = 7.15
WIDTH_HALF = 3.45

COLORS = {
    "base2": "#4C78A8",
    "base4": "#59A14F",
    "base6": "#F28E2B",
    "base8": "#B07AA1",
    "base16": "#E15759",
    "noisy": "#6B7280",
    "raw": "#374151",
    "signal": "#6B7280",
    "block": "#4C78A8",
    "block_light": "#DCEBFA",
    "bottleneck": "#F28E2B",
    "bottleneck_light": "#FDE7CF",
    "skip": "#64748B",
    "accent": "#2A9D8F",
    "eog": "#4C78A8",
    "emg": "#E15759",
    "mixed": "#2A9D8F",
    "text": "#1F2937",
    "grid": "#E5E7EB",
}

MODEL_COLORS = {
    "base2": COLORS["base2"],
    "base4": COLORS["base4"],
    "base6": COLORS["base6"],
    "base8": COLORS["base8"],
    "base16": COLORS["base16"],
    "EEGDN_CNN": "#8CD17D",
    "EEGDN_RNN": "#B6992D",
    "MicroWaveNet": "#499894",
    "DeepSeparator": "#D37295",
}


def set_jne_style() -> None:
    """Apply a compact, white-background manuscript style."""

    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def new_figure(width: float = WIDTH_FULL, height: float = 4.2, **kwargs):
    set_jne_style()
    return plt.subplots(figsize=(width, height), constrained_layout=True, **kwargs)


def add_panel_label(ax, label: str, x: float = -0.02, y: float = 1.03) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=COLORS["text"],
    )


def light_grid(ax, axis: str = "y") -> None:
    ax.grid(True, axis=axis, color=COLORS["grid"], linewidth=0.45, alpha=0.7)
    ax.set_axisbelow(True)


def save_figure(fig, output_base: Path, *, dpi: int = 450) -> tuple[Path, Path]:
    """Save vector PDF and PNG preview using the same stem."""

    output_base.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_base.with_suffix(".pdf")
    png_path = output_base.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    return pdf_path, png_path
