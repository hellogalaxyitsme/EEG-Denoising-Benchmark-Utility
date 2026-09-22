"""Final Supplementary Fig. S2: Mixed-1M within-split source reuse ECDFs."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from jne_figure_style import COLORS, WIDTH_FULL, add_panel_label, light_grid, save_figure, set_jne_style


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "a11_mixed1m_source_reuse_20260816_140553"
ECDF_CSV = RUN_DIR / "a11_reuse_ecdf.csv"
OUT_DIR = ROOT / "Revised_Paper" / "Images"
DATA_DIR = ROOT / "Revised_Paper" / "figure_data"

SOURCE_TYPES = [
    ("clean", "Clean EEG"),
    ("eog", "EOG"),
    ("emg", "EMG"),
]
SPLITS = ["train", "val", "test"]
SPLIT_LABELS = {"train": "Train", "val": "Validation", "test": "Test"}
SPLIT_COLORS = {
    "train": "#4C78A8",
    "val": "#F28E2B",
    "test": "#2A9D8F",
}


def load_ecdf() -> pd.DataFrame:
    ecdf = pd.read_csv(ECDF_CSV)
    required_cols = {"ecdf", "reuse_count", "source_type", "split"}
    missing_cols = required_cols - set(ecdf.columns)
    if missing_cols:
        raise ValueError(f"Missing ECDF columns: {sorted(missing_cols)}")

    expected_sources = {source for source, _ in SOURCE_TYPES}
    expected_splits = set(SPLITS)
    if set(ecdf["source_type"].unique()) != expected_sources:
        raise ValueError(f"Unexpected source types: {sorted(ecdf['source_type'].unique())}")
    if set(ecdf["split"].unique()) != expected_splits:
        raise ValueError(f"Unexpected splits: {sorted(ecdf['split'].unique())}")

    ecdf = ecdf.copy()
    ecdf["source_label"] = ecdf["source_type"].map(dict(SOURCE_TYPES))
    ecdf["split_label"] = ecdf["split"].map(SPLIT_LABELS)
    ecdf["source_file"] = str(ECDF_CSV.relative_to(ROOT))
    ecdf["aggregation"] = "ECDF of within-split source reuse counts"
    return ecdf.sort_values(["source_type", "split", "reuse_count", "ecdf"]).reset_index(drop=True)


def save_source_data(ecdf: pd.DataFrame) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / "supp_s2_source_reuse.csv"
    columns = [
        "source_type",
        "source_label",
        "split",
        "split_label",
        "reuse_count",
        "ecdf",
        "source_file",
        "aggregation",
    ]
    ecdf[columns].to_csv(out, index=False)
    return out


def draw_panel(ax, ecdf: pd.DataFrame, *, source_type: str, title: str) -> None:
    for split in SPLITS:
        sub = ecdf[
            ecdf["source_type"].eq(source_type) & ecdf["split"].eq(split)
        ].sort_values("reuse_count")
        ax.step(
            sub["reuse_count"],
            sub["ecdf"],
            where="post",
            color=SPLIT_COLORS[split],
            linewidth=1.35,
            label=SPLIT_LABELS[split],
        )

    ax.set_title(title, pad=5)
    ax.set_xlabel("Reuse count")
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    light_grid(ax, axis="y")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def main() -> None:
    set_jne_style()
    ecdf = load_ecdf()
    data_path = save_source_data(ecdf)

    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_FULL, 2.55), sharey=True, constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.22, top=0.72, wspace=0.28)

    for ax, label, (source_type, title) in zip(axes, "ABC", SOURCE_TYPES, strict=True):
        draw_panel(ax, ecdf, source_type=source_type, title=title)
        add_panel_label(ax, label, x=-0.14, y=1.07)

    axes[0].set_ylabel("ECDF")
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)

    handles = [
        Line2D([0], [0], color=SPLIT_COLORS[split], lw=1.35, label=SPLIT_LABELS[split])
        for split in SPLITS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=3,
        frameon=False,
        handlelength=1.9,
        columnspacing=1.4,
        borderpad=0.1,
    )

    pdf, png = save_figure(fig, OUT_DIR / "supp_s2_source_reuse_final", dpi=400)
    plt.close(fig)

    print(f"source file(s) used: {ECDF_CSV.relative_to(ROOT)}")
    print(f"number of observations plotted: {len(ecdf)} ECDF rows")
    print("observations represent: source segments summarized as within-split ECDFs, not human subjects")
    print("aggregation performed before plotting: none beyond stored ECDF table")
    print(f"source data saved: {data_path.relative_to(ROOT)}")
    print(f"final output paths: {pdf.relative_to(ROOT)}, {png.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
