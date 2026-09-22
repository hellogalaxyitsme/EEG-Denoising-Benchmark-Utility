#!/usr/bin/env python3
"""Generate simplified Figure 3 for hierarchical BCI IV-2a downstream effects."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from jne_figure_style import COLORS, add_panel_label, light_grid, save_figure, set_jne_style


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "hierarchical_downstream_stats_20260815_000000"
OUT_DIR = ROOT / "Revised_Paper" / "Images"
DATA_DIR = ROOT / "Revised_Paper" / "figure_data"

SUBJECT_PATH = RUN_DIR / "hierarchical_subject_deltas.csv"
EFFECTS_PATH = RUN_DIR / "hierarchical_confirmatory_effects.csv"

CLASSIFIER_ORDER = ["csp_lda", "shallowfbcsp", "eegconformer", "eegnet", "deep4net"]
CLASSIFIER_LABELS = {
    "csp_lda": "CSP+LDA",
    "shallowfbcsp": "ShallowFBCSPNet",
    "eegconformer": "EEGConformer",
    "eegnet": "EEGNet",
    "deep4net": "Deep4Net",
}
RECIPE_ORDER = ["eog", "emg", "eog_emg_line"]
RECIPE_LABELS = {
    "eog": "EOG",
    "emg": "EMG",
    "eog_emg_line": "EOG+EMG+LINE",
}

SUBJECT_COLOR = "#B8BCC2"
ACCENT_COLOR = COLORS["accent"]
ZERO_COLOR = "#6B7280"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    subjects = pd.read_csv(SUBJECT_PATH)
    effects = pd.read_csv(EFFECTS_PATH)

    subjects = subjects[
        subjects["base"].eq(16)
        & subjects["recipe"].isin(RECIPE_ORDER)
        & subjects["classifier"].isin(CLASSIFIER_ORDER)
    ].copy()
    effects = effects[
        effects["base"].eq(16)
        & effects["recipe"].isin(RECIPE_ORDER)
        & effects["classifier"].isin(CLASSIFIER_ORDER)
    ].copy()

    expected_subject_rows = len(RECIPE_ORDER) * len(CLASSIFIER_ORDER) * 9
    expected_effect_rows = len(RECIPE_ORDER) * len(CLASSIFIER_ORDER)
    if len(subjects) != expected_subject_rows:
        raise ValueError(f"Expected {expected_subject_rows} subject rows, found {len(subjects)}")
    if len(effects) != expected_effect_rows:
        raise ValueError(f"Expected {expected_effect_rows} mean/CI rows, found {len(effects)}")
    if not subjects.groupby(["recipe", "classifier"])["subject"].nunique().eq(9).all():
        raise ValueError("Each recipe/classifier combination must contain exactly 9 subjects")

    subjects["panel"] = subjects["recipe"].map(RECIPE_LABELS)
    subjects["classifier_label"] = subjects["classifier"].map(CLASSIFIER_LABELS)
    subjects["row_type"] = "subject"
    subjects["mean_delta_accuracy"] = np.nan
    subjects["bootstrap_ci95_low"] = np.nan
    subjects["bootstrap_ci95_high"] = np.nan
    subjects["source_file"] = str(SUBJECT_PATH.relative_to(ROOT))
    subjects["aggregation"] = (
        "Subject-level delta after contamination/checkpoint/classifier-seed nuisance aggregation"
    )

    effects["panel"] = effects["recipe"].map(RECIPE_LABELS)
    effects["classifier_label"] = effects["classifier"].map(CLASSIFIER_LABELS)
    effects["row_type"] = "mean_ci"
    effects["subject"] = ""
    effects["delta_accuracy"] = np.nan
    effects["source_file"] = str(EFFECTS_PATH.relative_to(ROOT))
    effects["aggregation"] = "Mean and 95% subject-bootstrap CI across n=9 subjects"

    return subjects.reset_index(drop=True), effects.reset_index(drop=True)


def write_figure_data(subjects: pd.DataFrame, effects: pd.DataFrame) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    columns = [
        "row_type",
        "panel",
        "recipe",
        "classifier",
        "classifier_label",
        "base",
        "subject",
        "delta_accuracy",
        "mean_delta_accuracy",
        "bootstrap_ci95_low",
        "bootstrap_ci95_high",
        "n_subjects_for_inference",
        "primary_independent_unit",
        "n_contamination_seed_pairs",
        "n_checkpoint_seeds",
        "n_classifier_seeds",
        "n_technical_observations_aggregated",
        "source_file",
        "aggregation",
    ]
    for col in columns:
        if col not in subjects:
            subjects[col] = np.nan
        if col not in effects:
            effects[col] = np.nan

    plotted = pd.concat([subjects[columns], effects[columns]], ignore_index=True)
    classifier_rank = {name: i for i, name in enumerate(CLASSIFIER_ORDER)}
    recipe_rank = {name: i for i, name in enumerate(RECIPE_ORDER)}
    plotted["_recipe_rank"] = plotted["recipe"].map(recipe_rank)
    plotted["_classifier_rank"] = plotted["classifier"].map(classifier_rank)
    plotted["_row_rank"] = plotted["row_type"].map({"subject": 0, "mean_ci": 1})
    plotted = plotted.sort_values(["_recipe_rank", "_classifier_rank", "_row_rank", "subject"])
    plotted = plotted.drop(columns=["_recipe_rank", "_classifier_rank", "_row_rank"])

    out_path = DATA_DIR / "fig03_iv2a_subject_effects.csv"
    plotted.to_csv(out_path, index=False)
    return out_path


def subject_offsets(subjects: list[str], scale: float = 0.105) -> dict[str, float]:
    offsets = np.linspace(-scale, scale, len(subjects))
    return {subject: float(offsets[i]) for i, subject in enumerate(sorted(subjects))}


def compute_x_limits(subjects: pd.DataFrame, effects: pd.DataFrame) -> tuple[float, float]:
    values = pd.concat(
        [
            subjects["delta_accuracy"],
            effects["bootstrap_ci95_low"],
            effects["bootstrap_ci95_high"],
            pd.Series([0.0]),
        ],
        ignore_index=True,
    ).astype(float)
    xmin = float(values.min())
    xmax = float(values.max())
    pad = max(0.015, (xmax - xmin) * 0.08)
    return xmin - pad, xmax + pad


def draw_panel(ax, recipe: str, subjects: pd.DataFrame, effects: pd.DataFrame, show_ylabels: bool) -> None:
    y_positions = {clf: len(CLASSIFIER_ORDER) - 1 - i for i, clf in enumerate(CLASSIFIER_ORDER)}
    offsets = subject_offsets(sorted(subjects["subject"].unique()))

    for clf in CLASSIFIER_ORDER:
        y = y_positions[clf]
        sub = subjects[(subjects["recipe"] == recipe) & (subjects["classifier"] == clf)]
        eff = effects[(effects["recipe"] == recipe) & (effects["classifier"] == clf)].iloc[0]

        for _, row in sub.iterrows():
            ax.scatter(
                float(row["delta_accuracy"]),
                y + offsets[row["subject"]],
                s=13,
                color=SUBJECT_COLOR,
                alpha=0.78,
                linewidths=0,
                zorder=2,
            )

        mean = float(eff["mean_delta_accuracy"])
        lo = float(eff["bootstrap_ci95_low"])
        hi = float(eff["bootstrap_ci95_high"])
        ax.hlines(y, lo, hi, color=ACCENT_COLOR, linewidth=1.45, zorder=3)
        ax.plot([lo, lo], [y - 0.09, y + 0.09], color=ACCENT_COLOR, linewidth=1.0, zorder=3)
        ax.plot([hi, hi], [y - 0.09, y + 0.09], color=ACCENT_COLOR, linewidth=1.0, zorder=3)
        ax.scatter(mean, y, s=34, color=ACCENT_COLOR, edgecolors="white", linewidths=0.45, zorder=4)

    ax.axvline(0, color=ZERO_COLOR, linewidth=0.8, alpha=0.8, zorder=1)
    ax.set_title(RECIPE_LABELS[recipe], pad=5)
    ax.set_yticks([y_positions[c] for c in CLASSIFIER_ORDER])
    if show_ylabels:
        ax.set_yticklabels([CLASSIFIER_LABELS[c] for c in CLASSIFIER_ORDER])
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.55, len(CLASSIFIER_ORDER) - 0.45)
    ax.tick_params(axis="both", labelsize=7.6)
    light_grid(ax, axis="x")


def main() -> None:
    set_jne_style()
    subjects, effects = load_data()
    data_path = write_figure_data(subjects, effects)
    xlim = compute_x_limits(subjects, effects)

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 3.35), sharex=True, constrained_layout=False)
    fig.subplots_adjust(left=0.16, right=0.99, top=0.86, bottom=0.27, wspace=0.20)

    for idx, (ax, recipe) in enumerate(zip(axes, RECIPE_ORDER)):
        draw_panel(ax, recipe, subjects, effects, show_ylabels=(idx == 0))
        ax.set_xlim(*xlim)
        add_panel_label(ax, chr(ord("A") + idx), x=-0.17 if idx == 0 else -0.10, y=1.03)
    fig.supxlabel("Δ accuracy (denoised/denoised − noisy/noisy)", fontsize=8.5, y=0.13)

    subject_handle = axes[0].scatter([], [], s=13, color=SUBJECT_COLOR, alpha=0.78, linewidths=0, label="Subject")
    mean_handle = axes[0].errorbar(
        [],
        [],
        xerr=[],
        fmt="o",
        markersize=4.8,
        color=ACCENT_COLOR,
        ecolor=ACCENT_COLOR,
        elinewidth=1.45,
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
        borderpad=0.1,
        bbox_to_anchor=(0.5, 0.035),
    )

    pdf_path, png_path = save_figure(fig, OUT_DIR / "fig03_iv2a_hierarchical_downstream", dpi=400)
    plt.close(fig)

    print("Sources used:")
    print(f"- {SUBJECT_PATH.relative_to(ROOT)}")
    print(f"- {EFFECTS_PATH.relative_to(ROOT)}")
    print(f"Observations plotted: {len(subjects) + len(effects)} rows")
    print(f"- subject observations: {len(subjects)}")
    print(f"- mean/CI overlays: {len(effects)}")
    print("Observation type: BCI IV-2a human subjects for dots; subject-bootstrap summaries for overlays")
    print("Aggregation before plotting: technical contamination/checkpoint/classifier-seed repeats already averaged within subject")
    print(f"Figure data: {data_path.relative_to(ROOT)}")
    print(f"PDF: {pdf_path.relative_to(ROOT)}")
    print(f"PNG: {png_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
