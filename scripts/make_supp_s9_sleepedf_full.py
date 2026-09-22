"""Supplementary Figure S9: complete Sleep-EDF external validation.

All plotted values come from the formal Sleep-EDF result artifacts. For the
forest panels, checkpoint and contamination repetitions are averaged within
subject before plotting, so each dot is one Sleep-EDF subject. For the metric-
utility panel, stored within-subject slopes are oriented so positive always
means that better reconstruction predicts a more favorable downstream delta.
The Sleep-EDF pre-injection EEG is an assumed-clean/low-artifact reference, not
an artifact-free neural source.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jne_figure_style import COLORS, MODEL_COLORS, WIDTH_FULL, add_panel_label, light_grid, save_figure, set_jne_style


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = PROJECT_ROOT / "runs" / "sleep_edf_formal_all_subjects_20260817_010000"
ROWS_CSV = RUN_DIR / "sleep_edf_formal_downstream_rows.csv"
SUMMARY_CSV = RUN_DIR / "sleep_edf_formal_downstream_summary.csv"
METRIC_UTILITY_CSV = RUN_DIR / "sleep_edf_formal_metric_utility.csv"
OUT_DIR = PROJECT_ROOT / "Revised_Paper" / "Images"
DATA_DIR = PROJECT_ROOT / "Revised_Paper" / "figure_data"

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

METRIC_ORDER = ["CC", "RMSE", "T_RRMSE", "S_RRMSE", "SDR"]
METRIC_LABELS = {
    "CC": "CC",
    "RMSE": "RMSE",
    "T_RRMSE": "T-RRMSE",
    "S_RRMSE": "S-RRMSE",
    "SDR": "SDR",
}
ERROR_METRICS = {"RMSE", "T_RRMSE", "S_RRMSE"}

PANELS = [
    {
        "panel": "A",
        "title": "Synthetic EOG",
        "condition": "denoised_denoised",
        "delta_col": "delta_balanced_accuracy_vs_noisy_noisy",
        "condition_label": "Synthetic EOG",
    },
    {
        "panel": "B",
        "title": "Unmodified recordings",
        "condition": "real_denoised_denoised",
        "delta_col": "delta_balanced_accuracy_vs_raw_raw",
        "condition_label": "Unmodified recording",
    },
]


def stable_jitter(subjects: pd.Series, scale: float = 0.13) -> np.ndarray:
    vals = []
    for subject in subjects.astype(str):
        code = sum((idx + 1) * ord(char) for idx, char in enumerate(subject))
        vals.append(((code % 1000) / 999.0 - 0.5) * 2.0 * scale)
    return np.asarray(vals)


def load_downstream() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = pd.read_csv(ROWS_CSV)
    summary = pd.read_csv(SUMMARY_CSV)

    subject_frames = []
    summary_frames = []
    for panel in PANELS:
        sub = rows.loc[
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
        )
        grouped["panel"] = panel["panel"]
        grouped["condition_label"] = panel["condition_label"]
        subject_frames.append(grouped)

        sm = summary.loc[
            summary["condition"].eq(panel["condition"])
            & summary["denoiser_label"].isin(MODEL_ORDER)
        ].copy()
        if sm.empty:
            raise ValueError(f"No downstream summary rows found for {panel['condition']}")
        sm["panel"] = panel["panel"]
        sm["condition_label"] = panel["condition_label"]
        summary_frames.append(sm)

    subject_data = pd.concat(subject_frames, ignore_index=True)
    summary_data = pd.concat(summary_frames, ignore_index=True)
    for df in (subject_data, summary_data):
        df["model_order"] = df["denoiser_label"].map({m: i for i, m in enumerate(MODEL_ORDER)})
        df["display_label"] = df["denoiser_label"].map(DISPLAY_LABELS)
        if df["display_label"].isna().any():
            missing = sorted(df.loc[df["display_label"].isna(), "denoiser_label"].unique())
            raise ValueError(f"Missing display labels for {missing}")

    expected = subject_data.groupby(["panel", "denoiser_label"])["subject"].nunique()
    if not (expected == 75).all():
        raise ValueError(f"Expected 75 subjects per row, got:\n{expected}")

    return subject_data, summary_data


def load_metric_utility() -> pd.DataFrame:
    metric = pd.read_csv(METRIC_UTILITY_CSV)
    metric = metric.loc[metric["denoiser_label"].isin(MODEL_ORDER) & metric["metric"].isin(METRIC_ORDER)].copy()
    if metric.empty:
        raise ValueError("No Sleep-EDF metric-utility rows found")

    metric["model_order"] = metric["denoiser_label"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    metric["display_label"] = metric["denoiser_label"].map(DISPLAY_LABELS)
    metric["metric_order"] = metric["metric"].map({m: i for i, m in enumerate(METRIC_ORDER)})
    metric["metric_label"] = metric["metric"].map(METRIC_LABELS)
    metric["orientation"] = np.where(metric["metric"].isin(ERROR_METRICS), "lower_is_better", "higher_is_better")

    # Stored slopes are against the raw metric value. Flip error metrics so the
    # heatmap's sign answers one question: does better reconstruction predict a
    # more favorable downstream delta?
    sign = np.where(metric["metric"].isin(ERROR_METRICS), -1.0, 1.0)
    for col in ["mean_subject_slope", "median_subject_slope", "slope_ci95_low", "slope_ci95_high"]:
        metric[f"oriented_{col}"] = metric[col] * sign
    lo = np.minimum(metric["oriented_slope_ci95_low"], metric["oriented_slope_ci95_high"])
    hi = np.maximum(metric["oriented_slope_ci95_low"], metric["oriented_slope_ci95_high"])
    metric["oriented_slope_ci95_low"] = lo
    metric["oriented_slope_ci95_high"] = hi

    expected = metric.groupby("denoiser_label")["metric"].nunique()
    if not (expected == len(METRIC_ORDER)).all():
        raise ValueError(f"Expected all metrics per denoiser, got:\n{expected}")
    return metric.sort_values(["model_order", "metric_order"]).reset_index(drop=True)


def save_source_data(subject_data: pd.DataFrame, summary_data: pd.DataFrame, metric_data: pd.DataFrame) -> tuple[Path, Path, Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    subject_path = DATA_DIR / "supp_s9_sleepedf_subject_deltas.csv"
    effect_path = DATA_DIR / "supp_s9_sleepedf_effects.csv"
    metric_path = DATA_DIR / "supp_s9_sleepedf_metric_utility_oriented.csv"

    subject_cols = [
        "panel",
        "condition_label",
        "condition",
        "subject",
        "denoiser_label",
        "display_label",
        "model_order",
        "delta_balanced_accuracy",
        "n_source_rows",
        "n_checkpoint_seeds",
        "n_train_contamination_seeds",
        "n_test_contamination_seeds",
    ]
    summary_cols = [
        "panel",
        "condition_label",
        "condition",
        "denoiser_label",
        "display_label",
        "model_order",
        "n_subjects",
        "n_rows",
        "n_checkpoint_seeds",
        "n_contamination_pairs",
        "mean_delta_balanced_accuracy",
        "median_delta_balanced_accuracy",
        "subject_sd",
        "ci95_low",
        "ci95_high",
        "subjects_below_reference",
        "wilcoxon_p_less",
        "bh_fdr_q",
    ]
    metric_cols = [
        "denoiser_label",
        "display_label",
        "model_order",
        "metric",
        "metric_label",
        "metric_order",
        "orientation",
        "n_subjects_with_slope",
        "mean_subject_slope",
        "oriented_mean_subject_slope",
        "oriented_slope_ci95_low",
        "oriented_slope_ci95_high",
        "wilcoxon_p_two_sided",
        "bh_fdr_q",
        "pooled_subject_demeaned_pearson_r_descriptive",
        "pooled_subject_demeaned_spearman_r_descriptive",
    ]

    subject_data.loc[:, subject_cols].sort_values(["panel", "model_order", "subject"]).to_csv(subject_path, index=False)
    summary_data.loc[:, summary_cols].sort_values(["panel", "model_order"]).to_csv(effect_path, index=False)
    metric_data.loc[:, metric_cols].sort_values(["model_order", "metric_order"]).to_csv(metric_path, index=False)
    return subject_path, effect_path, metric_path


def draw_forest(ax, panel: dict[str, str], subject_data: pd.DataFrame, summary_data: pd.DataFrame, xlim: tuple[float, float]) -> None:
    panel_subjects = subject_data.loc[subject_data["panel"].eq(panel["panel"])].copy()
    panel_summary = summary_data.loc[summary_data["panel"].eq(panel["panel"])].copy()
    y_lookup = {model: len(MODEL_ORDER) - 1 - idx for idx, model in enumerate(MODEL_ORDER)}

    for model in MODEL_ORDER:
        pts = panel_subjects.loc[panel_subjects["denoiser_label"].eq(model)].copy()
        pts["y"] = y_lookup[model]
        color = MODEL_COLORS.get(model, COLORS["text"])
        ax.scatter(
            pts["delta_balanced_accuracy"],
            pts["y"] + stable_jitter(pts["subject"]),
            s=5.5,
            color=color,
            alpha=0.24,
            linewidths=0,
            zorder=2,
        )

        sm = panel_summary.loc[panel_summary["denoiser_label"].eq(model)]
        if len(sm) != 1:
            raise ValueError(f"Expected one summary row for {panel['condition']} {model}, got {len(sm)}")
        row = sm.iloc[0]
        y = y_lookup[model]
        mean = float(row["mean_delta_balanced_accuracy"])
        low = float(row["ci95_low"])
        high = float(row["ci95_high"])
        ax.hlines(y, low, high, color=COLORS["text"], linewidth=1.2, zorder=3)
        ax.scatter(mean, y, s=26, color=COLORS["text"], edgecolor="white", linewidth=0.5, zorder=4)

    ax.axvline(0, color="#6B7280", linewidth=0.75, zorder=1)
    ax.set_xlim(xlim)
    ax.set_ylim(-0.7, len(MODEL_ORDER) - 0.3)
    ax.set_yticks([y_lookup[m] for m in MODEL_ORDER])
    ax.set_yticklabels([DISPLAY_LABELS[m] for m in MODEL_ORDER])
    ax.set_xlabel("Delta balanced accuracy")
    ax.set_title(panel["title"], pad=7)
    light_grid(ax, axis="x")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def draw_metric_heatmap(ax, metric_data: pd.DataFrame) -> None:
    mat = np.full((len(MODEL_ORDER), len(METRIC_ORDER)), np.nan)
    qmat = np.full_like(mat, np.nan, dtype=float)
    for _, row in metric_data.iterrows():
        r = int(row["model_order"])
        c = int(row["metric_order"])
        mat[r, c] = float(row["oriented_mean_subject_slope"])
        qmat[r, c] = float(row["bh_fdr_q"])

    vmax = float(np.nanmax(np.abs(mat)))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(METRIC_ORDER)))
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRIC_ORDER])
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels([DISPLAY_LABELS[m] for m in MODEL_ORDER])
    ax.set_title("Metric-utility slopes", pad=7)

    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            val = mat[r, c]
            if not np.isfinite(val):
                continue
            text_color = "white" if abs(val) > 0.58 * vmax else COLORS["text"]
            ax.text(c, r, f"{val:+.2g}", ha="center", va="center", fontsize=6.5, color=text_color)
            if np.isfinite(qmat[r, c]) and qmat[r, c] < 0.05:
                ax.scatter(c + 0.36, r - 0.34, s=9, facecolor="none", edgecolor=text_color, linewidth=0.7)

    ax.set_xticks(np.arange(-0.5, len(METRIC_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(MODEL_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.05, pad=0.025)
    cbar.set_label("Oriented slope")
    cbar.ax.tick_params(labelsize=7)
    ax.text(
        0.0,
        -0.16,
        "Positive: better reconstruction predicts better utility; circles mark BH q<0.05.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color=COLORS["text"],
    )


def main() -> None:
    set_jne_style()
    subject_data, summary_data = load_downstream()
    metric_data = load_metric_utility()
    data_paths = save_source_data(subject_data, summary_data, metric_data)

    low = min(subject_data["delta_balanced_accuracy"].min(), summary_data["ci95_low"].min())
    high = max(subject_data["delta_balanced_accuracy"].max(), summary_data["ci95_high"].max())
    pad = 0.015
    xlim = (min(low - pad, -0.14), max(high + pad, 0.035))

    fig = plt.figure(figsize=(WIDTH_FULL, 5.25), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    draw_forest(ax_a, PANELS[0], subject_data, summary_data, xlim)
    draw_forest(ax_b, PANELS[1], subject_data, summary_data, xlim)
    ax_b.set_yticklabels([])

    draw_metric_heatmap(ax_c, metric_data)
    add_panel_label(ax_a, "A", x=-0.22, y=1.02)
    add_panel_label(ax_b, "B", x=-0.06, y=1.02)
    add_panel_label(ax_c, "C", x=-0.06, y=1.02)

    pdf, png = save_figure(fig, OUT_DIR / "supp_s9_sleepedf_full")
    plt.close(fig)

    print(f"[saved] {pdf}")
    print(f"[saved] {png}")
    for path in data_paths:
        print(f"[data] {path}")


if __name__ == "__main__":
    main()
