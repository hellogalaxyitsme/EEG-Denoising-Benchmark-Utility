"""Final Supplementary Fig. S4: complete metric-utility association matrix."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from jne_figure_style import COLORS, WIDTH_FULL, add_panel_label, save_figure, set_jne_style


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "a4_same_bci_metric_utility_all_neural_artifacts_20260814_215900"
TESTS_CSV = RUN_DIR / "a4_subject_intercept_metric_tests.csv"
OUT_DIR = ROOT / "Revised_Paper" / "Images"
DATA_DIR = ROOT / "Revised_Paper" / "figure_data"

RECIPES = [
    ("eog", "EOG"),
    ("emg", "EMG"),
    ("eog_emg_line", "EOG+EMG+LINE"),
]
CLASSIFIERS = [
    ("csp_lda", "CSP+LDA"),
    ("shallowfbcsp", "ShallowFBCSPNet"),
    ("eegconformer", "EEGConformer"),
    ("eegnet", "EEGNet"),
    ("deep4net", "Deep4Net"),
]
METRICS = ["CC", "RMSE", "T_RRMSE", "S_RRMSE", "SDR"]
METRIC_LABELS = {
    "CC": "CC",
    "RMSE": "RMSE",
    "T_RRMSE": "T-\nRRMSE",
    "S_RRMSE": "S-\nRRMSE",
    "SDR": "SDR",
}


def load_tests() -> pd.DataFrame:
    df = pd.read_csv(TESTS_CSV)
    required_cols = {
        "classifier",
        "recipe",
        "metric",
        "subject_intercept_slope_oriented",
        "bh_fdr_q_subject_intercept",
    }
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing metric-utility columns: {sorted(missing_cols)}")

    expected = {
        (recipe, classifier, metric)
        for recipe, _ in RECIPES
        for classifier, _ in CLASSIFIERS
        for metric in METRICS
    }
    have = set(zip(df["recipe"], df["classifier"], df["metric"], strict=False))
    missing_rows = sorted(expected - have)
    extra_rows = sorted(have - expected)
    if missing_rows or extra_rows:
        raise ValueError(f"Unexpected metric rows. Missing={missing_rows[:3]}, extra={extra_rows[:3]}")

    df = df.copy()
    df["recipe_label"] = df["recipe"].map(dict(RECIPES))
    df["classifier_label"] = df["classifier"].map(dict(CLASSIFIERS))
    df["metric_label"] = df["metric"].map(METRIC_LABELS)
    df["oriented_slope"] = df["subject_intercept_slope_oriented"].astype(float)
    df["bh_fdr_q"] = df["bh_fdr_q_subject_intercept"].astype(float)
    df["outlined_bh_fdr_q_lt_0_05"] = df["bh_fdr_q"] < 0.05
    df["source_file"] = str(TESTS_CSV.relative_to(ROOT))
    return df


def save_source_data(df: pd.DataFrame) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    order = {
        (recipe, classifier, metric): (r_i, c_i, m_i)
        for r_i, (recipe, _) in enumerate(RECIPES)
        for c_i, (classifier, _) in enumerate(CLASSIFIERS)
        for m_i, metric in enumerate(METRICS)
    }
    out = df.copy()
    out["_sort"] = [order[(row.recipe, row.classifier, row.metric)] for row in out.itertuples()]
    out = out.sort_values("_sort").drop(columns="_sort")
    columns = [
        "recipe",
        "recipe_label",
        "classifier",
        "classifier_label",
        "metric",
        "metric_label",
        "oriented_slope",
        "bh_fdr_q",
        "outlined_bh_fdr_q_lt_0_05",
        "n_observations",
        "n_subjects",
        "source_file",
    ]
    out_path = DATA_DIR / "supp_s4_metric_utility_matrix.csv"
    out[columns].to_csv(out_path, index=False)
    return out_path


def matrix_for_recipe(df: pd.DataFrame, recipe: str) -> tuple[np.ndarray, np.ndarray]:
    slopes = np.zeros((len(CLASSIFIERS), len(METRICS)), dtype=float)
    qs = np.zeros_like(slopes)
    for i, (classifier, _) in enumerate(CLASSIFIERS):
        for j, metric in enumerate(METRICS):
            row = df[df["recipe"].eq(recipe) & df["classifier"].eq(classifier) & df["metric"].eq(metric)]
            if len(row) != 1:
                raise ValueError(f"Expected one row for {recipe}/{classifier}/{metric}, found {len(row)}")
            slopes[i, j] = float(row.iloc[0]["oriented_slope"])
            qs[i, j] = float(row.iloc[0]["bh_fdr_q"])
    return slopes, qs


def draw_heatmap(ax, df: pd.DataFrame, *, recipe: str, title: str, norm: TwoSlopeNorm):
    slopes, qs = matrix_for_recipe(df, recipe)
    image = ax.imshow(slopes, cmap="RdBu_r", norm=norm, aspect="equal")

    ax.set_title(title, pad=5)
    ax.set_xticks(np.arange(len(METRICS)))
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], rotation=0, fontsize=7.2)
    ax.set_yticks(np.arange(len(CLASSIFIERS)))
    ax.set_yticklabels([label for _, label in CLASSIFIERS])
    ax.tick_params(length=0, pad=2)

    ax.set_xticks(np.arange(-0.5, len(METRICS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(CLASSIFIERS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7, zorder=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    threshold = 0.58 * max(abs(norm.vmin), abs(norm.vmax))
    for i in range(len(CLASSIFIERS)):
        for j in range(len(METRICS)):
            value = slopes[i, j]
            if qs[i, j] < 0.05:
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor=COLORS["text"],
                        linewidth=1.5,
                        zorder=4,
                        clip_on=False,
                    )
                )
            color = "white" if abs(value) >= threshold else COLORS["text"]
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=6.2, color=color, zorder=5)

    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def main() -> None:
    set_jne_style()
    df = load_tests()
    data_path = save_source_data(df)

    max_abs = float(np.nanmax(np.abs(df["oriented_slope"])))
    vmax = np.ceil(max_abs * 10.0) / 10.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_FULL, 3.0), constrained_layout=False)
    fig.subplots_adjust(left=0.16, right=0.91, bottom=0.31, top=0.82, wspace=0.20)

    images = []
    for ax, label, (recipe, title) in zip(axes, "ABC", RECIPES, strict=True):
        images.append(draw_heatmap(ax, df, recipe=recipe, title=title, norm=norm))
        add_panel_label(ax, label, x=-0.28, y=1.06)

    for ax in axes[1:]:
        ax.set_yticklabels([])

    colorbar = fig.colorbar(images[0], ax=axes, location="right", fraction=0.032, pad=0.020)
    colorbar.set_label("Oriented slope")
    colorbar.outline.set_linewidth(0.5)

    outline_handle = Line2D(
        [0],
        [0],
        marker="s",
        linestyle="none",
        markersize=6,
        markerfacecolor="white",
        markeredgecolor=COLORS["text"],
        markeredgewidth=0.9,
        label="Outlined cell: BH-FDR q < 0.05",
    )
    fig.legend(
        handles=[outline_handle],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        frameon=False,
        borderpad=0.1,
    )

    pdf, png = save_figure(fig, OUT_DIR / "supp_s4_metric_utility_matrix_final", dpi=400)
    plt.close(fig)

    print(f"source file(s) used: {TESTS_CSV.relative_to(ROOT)}")
    print(f"number of observations plotted: {len(df)} classifier x artifact x metric tests")
    print("observations represent: formal subject-intercept metric-utility test results, not individual subjects")
    print("aggregation performed before plotting: none beyond stored subject-intercept test table")
    print(f"source data saved: {data_path.relative_to(ROOT)}")
    print(f"final output paths: {pdf.relative_to(ROOT)}, {png.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
