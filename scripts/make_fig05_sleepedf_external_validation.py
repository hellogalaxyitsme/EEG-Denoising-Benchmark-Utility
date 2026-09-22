#!/usr/bin/env python3
"""Generate simplified Figure 5 for Sleep-EDF external validation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from jne_figure_style import COLORS, WIDTH_FULL, add_panel_label, light_grid, save_figure, set_jne_style


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "sleep_edf_formal_all_subjects_20260817_010000"
ROWS_CSV = RUN_DIR / "sleep_edf_formal_downstream_rows.csv"
SUMMARY_CSV = RUN_DIR / "sleep_edf_formal_downstream_summary.csv"
OUT_DIR = ROOT / "Revised_Paper" / "Images"
DATA_DIR = ROOT / "Revised_Paper" / "figure_data"

MODEL_ORDER = [
    "base2",
    "base4",
    "base6",
    "base8",
    "base16",
    "EEGDN_CNN",
    "EEGDN_RNN",
    "MicroWaveNet",
    "DeepSeparator",
]

DISPLAY_LABELS = {
    "base2": "base2",
    "base4": "base4",
    "base6": "base6",
    "base8": "base8",
    "base16": "base16",
    "EEGDN_CNN": "EEGDN CNN",
    "EEGDN_RNN": "EEGDN RNN",
    "MicroWaveNet": "MicroWaveNet",
    "DeepSeparator": "DeepSeparator",
}

PANELS = [
    {
        "panel": "A",
        "title": "Synthetic-EOG evaluation",
        "condition": "denoised_denoised",
        "delta_col": "delta_balanced_accuracy_vs_noisy_noisy",
        "x_label": "Balanced-accuracy change vs noisy/noisy",
        "condition_label": "Sleep-EDF + synthetic EOG",
    },
    {
        "panel": "B",
        "title": "Unmodified-recording evaluation",
        "condition": "real_denoised_denoised",
        "delta_col": "delta_balanced_accuracy_vs_raw_raw",
        "x_label": "Balanced-accuracy change vs raw/raw",
        "condition_label": "Sleep-EDF without synthetic injection",
    },
]

SUBJECT_COLOR = "#B8BCC2"
ACCENT_COLOR = COLORS["accent"]
ZERO_COLOR = "#6B7280"


def y_positions() -> dict[str, float]:
    positions: dict[str, float] = {}
    y = len(MODEL_ORDER) - 1.0
    for idx, model in enumerate(MODEL_ORDER):
        positions[model] = y
        y -= 1.0
        if idx == 4:
            y -= 0.55
    return positions


def stable_horizontal_jitter(subjects: pd.Series, scale: float = 0.00075) -> np.ndarray:
    """Tiny deterministic x-jitter for exact-overlap visibility."""

    values: list[float] = []
    for subject in subjects.astype(str):
        code = sum((idx + 1) * ord(char) for idx, char in enumerate(subject))
        values.append(((code % 1000) / 999.0 - 0.5) * 2.0 * scale)
    return np.asarray(values)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = pd.read_csv(ROWS_CSV)
    summary = pd.read_csv(SUMMARY_CSV)

    subject_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []

    for panel in PANELS:
        sub = rows[
            rows["condition"].eq(panel["condition"])
            & rows["denoiser_label"].isin(MODEL_ORDER)
            & rows[panel["delta_col"]].notna()
        ].copy()
        if sub.empty:
            raise ValueError(f"No downstream rows found for {panel['condition']}")

        grouped = (
            sub.groupby(["subject", "condition", "denoiser_label"], as_index=False)
            .agg(
                delta_balanced_accuracy=(panel["delta_col"], "mean"),
                n_source_rows=(panel["delta_col"], "size"),
                n_checkpoint_seeds=("checkpoint_seed", "nunique"),
                n_train_contamination_seeds=("train_contamination_seed", "nunique"),
                n_test_contamination_seeds=("test_contamination_seed", "nunique"),
            )
            .reset_index(drop=True)
        )
        grouped["panel"] = panel["panel"]
        grouped["panel_title"] = panel["title"]
        grouped["condition_label"] = panel["condition_label"]
        grouped["delta_reference"] = panel["x_label"].replace("Balanced-accuracy change vs ", "")
        grouped["row_type"] = "subject"
        grouped["mean_delta_balanced_accuracy"] = np.nan
        grouped["ci95_low"] = np.nan
        grouped["ci95_high"] = np.nan
        grouped["source_file"] = str(ROWS_CSV.relative_to(ROOT))
        grouped["aggregation"] = (
            "Human subject-level effect after averaging checkpoint and contamination nuisance repetitions"
        )
        subject_frames.append(grouped)

        sm = summary[
            summary["condition"].eq(panel["condition"])
            & summary["denoiser_label"].isin(MODEL_ORDER)
        ].copy()
        if sm.empty:
            raise ValueError(f"No downstream summary rows found for {panel['condition']}")
        sm["panel"] = panel["panel"]
        sm["panel_title"] = panel["title"]
        sm["condition_label"] = panel["condition_label"]
        sm["delta_reference"] = panel["x_label"].replace("Balanced-accuracy change vs ", "")
        sm["row_type"] = "mean_ci"
        sm["subject"] = ""
        sm["delta_balanced_accuracy"] = np.nan
        sm["n_source_rows"] = np.nan
        sm["source_file"] = str(SUMMARY_CSV.relative_to(ROOT))
        sm["aggregation"] = "Mean and 95% subject-bootstrap CI across 75 subjects"
        summary_frames.append(sm)

    subject_data = pd.concat(subject_frames, ignore_index=True)
    summary_data = pd.concat(summary_frames, ignore_index=True)

    for frame in (subject_data, summary_data):
        frame["model_order"] = frame["denoiser_label"].map({m: i for i, m in enumerate(MODEL_ORDER)})
        frame["display_label"] = frame["denoiser_label"].map(DISPLAY_LABELS)
        if frame["display_label"].isna().any():
            missing = frame.loc[frame["display_label"].isna(), "denoiser_label"].unique()
            raise ValueError(f"Missing display labels for {missing}")

    expected_subjects = subject_data.groupby(["panel", "denoiser_label"])["subject"].nunique()
    if not expected_subjects.eq(75).all():
        raise ValueError(f"Expected 75 subjects per plotted row, got:\n{expected_subjects}")
    expected_summary = summary_data.groupby(["panel", "denoiser_label"]).size()
    if not expected_summary.eq(1).all():
        raise ValueError(f"Expected one summary row per plotted row, got:\n{expected_summary}")

    return subject_data.reset_index(drop=True), summary_data.reset_index(drop=True)


def write_figure_data(subject_data: pd.DataFrame, summary_data: pd.DataFrame) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    columns = [
        "row_type",
        "panel",
        "panel_title",
        "condition_label",
        "condition",
        "delta_reference",
        "subject",
        "denoiser_label",
        "display_label",
        "model_order",
        "delta_balanced_accuracy",
        "mean_delta_balanced_accuracy",
        "ci95_low",
        "ci95_high",
        "n_source_rows",
        "n_subjects",
        "n_checkpoint_seeds",
        "n_contamination_pairs",
        "n_train_contamination_seeds",
        "n_test_contamination_seeds",
        "source_file",
        "aggregation",
    ]
    for col in columns:
        if col not in subject_data:
            subject_data[col] = np.nan
        if col not in summary_data:
            summary_data[col] = np.nan
    plotted = pd.concat([subject_data[columns], summary_data[columns]], ignore_index=True)
    plotted = plotted.sort_values(["panel", "model_order", "row_type", "subject"]).reset_index(drop=True)
    out_path = DATA_DIR / "fig05_sleepedf_subject_effects.csv"
    plotted.to_csv(out_path, index=False)
    return out_path


def compute_xlim(subject_data: pd.DataFrame, summary_data: pd.DataFrame) -> tuple[float, float]:
    values = pd.concat(
        [
            subject_data["delta_balanced_accuracy"],
            summary_data["ci95_low"],
            summary_data["ci95_high"],
            pd.Series([0.0]),
        ],
        ignore_index=True,
    ).astype(float)
    xmin = float(values.min())
    xmax = float(values.max())
    pad = max(0.01, 0.08 * (xmax - xmin))
    return xmin - pad, xmax + pad


def draw_panel(
    ax,
    panel: dict[str, str],
    subject_data: pd.DataFrame,
    summary_data: pd.DataFrame,
    xlim: tuple[float, float],
    show_ylabels: bool,
) -> None:
    positions = y_positions()
    panel_subjects = subject_data[subject_data["panel"].eq(panel["panel"])].copy()
    panel_summary = summary_data[summary_data["panel"].eq(panel["panel"])].copy()

    for model in MODEL_ORDER:
        y = positions[model]
        pts = panel_subjects[panel_subjects["denoiser_label"].eq(model)].copy()
        x_plot = pts["delta_balanced_accuracy"].astype(float).to_numpy()
        x_plot = x_plot + stable_horizontal_jitter(pts["subject"])
        ax.scatter(
            x_plot,
            np.full(len(pts), y),
            s=5.5,
            color=SUBJECT_COLOR,
            alpha=0.46,
            linewidths=0,
            zorder=2,
        )

        sm = panel_summary[panel_summary["denoiser_label"].eq(model)]
        if len(sm) != 1:
            raise ValueError(f"Expected one summary row for {panel['condition']} {model}, got {len(sm)}")
        row = sm.iloc[0]
        mean = float(row["mean_delta_balanced_accuracy"])
        low = float(row["ci95_low"])
        high = float(row["ci95_high"])
        ax.hlines(y, low, high, color=ACCENT_COLOR, linewidth=1.35, zorder=3)
        ax.plot([low, low], [y - 0.10, y + 0.10], color=ACCENT_COLOR, linewidth=1.0, zorder=3)
        ax.plot([high, high], [y - 0.10, y + 0.10], color=ACCENT_COLOR, linewidth=1.0, zorder=3)
        ax.scatter(mean, y, s=31, color=ACCENT_COLOR, edgecolors="white", linewidths=0.45, zorder=4)

    gap_y = (positions["base16"] + positions["EEGDN_CNN"]) / 2.0
    ax.axhline(gap_y, color="#E5E7EB", linewidth=0.75, zorder=0)
    ax.axvline(0, color=ZERO_COLOR, linewidth=0.8, alpha=0.82, zorder=1)
    ax.set_xlim(*xlim)
    ax.set_ylim(min(positions.values()) - 0.65, max(positions.values()) + 0.55)
    ax.set_yticks([positions[m] for m in MODEL_ORDER])
    ax.set_yticklabels([DISPLAY_LABELS[m] for m in MODEL_ORDER] if show_ylabels else [])
    ax.set_xlabel(panel["x_label"])
    ax.set_title(panel["title"], pad=6)
    ax.tick_params(axis="both", labelsize=7.6)
    light_grid(ax, axis="x")
    add_panel_label(ax, panel["panel"], x=-0.16 if show_ylabels else -0.08, y=1.03)


def main() -> None:
    set_jne_style()
    subject_data, summary_data = load_data()
    data_path = write_figure_data(subject_data, summary_data)
    xlim = compute_xlim(subject_data, summary_data)

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_FULL, 3.45), sharex=True, constrained_layout=False)
    fig.subplots_adjust(left=0.20, right=0.99, top=0.86, bottom=0.25, wspace=0.20)
    for idx, (ax, panel) in enumerate(zip(axes, PANELS)):
        draw_panel(ax, panel, subject_data, summary_data, xlim, show_ylabels=(idx == 0))

    subject_handle = axes[0].scatter([], [], s=5.5, color=SUBJECT_COLOR, alpha=0.46, linewidths=0, label="Subject")
    mean_handle = axes[0].errorbar(
        [],
        [],
        xerr=[],
        fmt="o",
        markersize=4.8,
        color=ACCENT_COLOR,
        ecolor=ACCENT_COLOR,
        elinewidth=1.35,
        capsize=2,
        label="Mean + 95% CI",
    )
    fig.legend(
        [subject_handle, mean_handle],
        ["Subject", "Mean + 95% CI"],
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=6.4,
        handlelength=1.5,
        borderpad=0.12,
        bbox_to_anchor=(0.5, 0.035),
    )

    pdf_path, png_path = save_figure(fig, OUT_DIR / "fig05_sleepedf_external_validation", dpi=400)
    plt.close(fig)

    print("Sources used:")
    print(f"- {ROWS_CSV.relative_to(ROOT)}")
    print(f"- {SUMMARY_CSV.relative_to(ROOT)}")
    print(f"Observations plotted: {len(subject_data) + len(summary_data)} rows")
    print(f"- subject observations: {len(subject_data)}")
    print(f"- mean/CI overlays: {len(summary_data)}")
    print("Observation type: Sleep-EDF human subject-level balanced-accuracy effects")
    print("Aggregation before plotting: checkpoint and contamination repetitions averaged within subject/model/condition")
    print(f"Figure data: {data_path.relative_to(ROOT)}")
    print(f"PDF: {pdf_path.relative_to(ROOT)}")
    print(f"PNG: {png_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
