#!/usr/bin/env python3
"""Generate simplified Figure 4 for same-condition metric-utility associations."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from jne_figure_style import COLORS, add_panel_label, light_grid, save_figure, set_jne_style


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "a4_same_bci_metric_utility_all_neural_artifacts_20260814_215900"
OUT_DIR = ROOT / "Revised_Paper" / "Images"
DATA_DIR = ROOT / "Revised_Paper" / "figure_data"

JOINED_PATH = RUN_DIR / "a4_metric_utility_joined_long_rows.csv"
TESTS_PATH = RUN_DIR / "a4_subject_intercept_metric_tests.csv"

PANELS = [
    ("A", "csp_lda", "eog", "CSP+LDA \u2014 EOG"),
    ("B", "csp_lda", "emg", "CSP+LDA \u2014 EMG"),
    ("C", "shallowfbcsp", "eog_emg_line", "ShallowFBCSPNet\nEOG+EMG+LINE"),
]

POINT_COLOR = "#B8BCC2"
FIT_COLOR = "#1F2937"
ZERO_COLOR = "#CBD0D6"


def format_q(q: float) -> str:
    if pd.isna(q):
        return "NA"
    if q < 0.001:
        return f"{q:.1e}"
    return f"{q:.3f}"


def load_and_prepare() -> tuple[pd.DataFrame, pd.DataFrame]:
    joined = pd.read_csv(JOINED_PATH)
    tests = pd.read_csv(TESTS_PATH)
    panel_keys = {(classifier, recipe) for _, classifier, recipe, _ in PANELS}

    sdr = joined[
        joined["metric"].eq("SDR")
        & joined.set_index(["classifier", "recipe"]).index.isin(panel_keys)
    ].copy()
    if sdr.empty:
        raise ValueError("No SDR rows found for requested Figure 4 panels")

    display = (
        sdr.groupby(["classifier", "recipe", "subject", "base"], as_index=False)
        .agg(
            sdr_db=("fidelity_oriented_metric_value", "mean"),
            delta_accuracy=("delta_accuracy_vs_noisy_noisy", "mean"),
            n_condition_rows=("delta_accuracy_vs_noisy_noisy", "size"),
            n_contamination_seed_pairs=("seed_pair_index", "nunique"),
            n_train_contamination_seeds=("train_contamination_seed", "nunique"),
            n_test_contamination_seeds=("test_contamination_seed", "nunique"),
            n_checkpoint_seeds=("checkpoint_seed", "nunique"),
            n_classifier_seed_values=(
                "n_classifier_seeds",
                lambda x: int(np.nanmax(x)) if np.isfinite(pd.to_numeric(x, errors="coerce")).any() else 0,
            ),
        )
        .sort_values(["classifier", "recipe", "subject", "base"])
        .reset_index(drop=True)
    )

    display["subject_mean_sdr_db"] = display.groupby(["classifier", "recipe", "subject"])[
        "sdr_db"
    ].transform("mean")
    display["subject_mean_delta_accuracy"] = display.groupby(["classifier", "recipe", "subject"])[
        "delta_accuracy"
    ].transform("mean")
    display["centered_sdr_db"] = display["sdr_db"] - display["subject_mean_sdr_db"]
    display["centered_delta_accuracy"] = (
        display["delta_accuracy"] - display["subject_mean_delta_accuracy"]
    )
    display["source_file"] = str(JOINED_PATH.relative_to(ROOT))
    display["aggregation"] = (
        "Display aggregation: SDR and downstream delta averaged to subject x width, "
        "then centered within subject"
    )

    for _, classifier, recipe, _ in PANELS:
        sub = display[(display["classifier"] == classifier) & (display["recipe"] == recipe)]
        bases = sorted(sub["base"].unique())
        expected_rows = 9 * len(bases)
        if len(sub) != expected_rows:
            raise ValueError(
                f"Expected {expected_rows} rows for {classifier}/{recipe}, found {len(sub)}"
            )
        if not sub.groupby("subject")["base"].nunique().eq(len(bases)).all():
            raise ValueError(
                f"Each subject must have all available widths for {classifier}/{recipe}: {bases}"
            )

    stats = tests[
        tests["metric"].eq("SDR")
        & tests.set_index(["classifier", "recipe"]).index.isin(panel_keys)
    ].copy()
    stats["source_file"] = str(TESTS_PATH.relative_to(ROOT))
    stats["statistical_source"] = (
        "Full condition-level subject-intercept model; values are not recomputed "
        "from display-aggregated points"
    )
    stats = stats[
        [
            "classifier",
            "recipe",
            "metric",
            "metric_direction",
            "bases",
            "checkpoint_seeds",
            "contamination_seed_pairs",
            "n_observations",
            "n_subjects",
            "subject_intercept_slope_oriented",
            "subject_intercept_cluster_se",
            "subject_intercept_p_two_sided",
            "subject_centered_pearson_r",
            "subject_centered_spearman_r",
            "bh_fdr_q_subject_intercept",
            "source_file",
            "statistical_source",
        ]
    ].reset_index(drop=True)
    if len(stats) != len(PANELS):
        raise ValueError(f"Expected {len(PANELS)} SDR model rows, found {len(stats)}")

    return display, stats


def write_figure_data(display: pd.DataFrame, stats: pd.DataFrame) -> tuple[Path, Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    display_path = DATA_DIR / "fig04_centered_display.csv"
    stats_path = DATA_DIR / "fig04_model_statistics.csv"
    display.to_csv(display_path, index=False)
    stats.to_csv(stats_path, index=False)
    return display_path, stats_path


def common_limits(display: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    x_values = pd.concat([display["centered_sdr_db"], pd.Series([0.0])], ignore_index=True)
    y_values = pd.concat([display["centered_delta_accuracy"], pd.Series([0.0])], ignore_index=True)
    x_pad = max(0.08, 0.08 * (float(x_values.max()) - float(x_values.min())))
    y_pad = max(0.008, 0.10 * (float(y_values.max()) - float(y_values.min())))
    return (
        (float(x_values.min()) - x_pad, float(x_values.max()) + x_pad),
        (float(y_values.min()) - y_pad, float(y_values.max()) + y_pad),
    )


def draw_panel(
    ax,
    display: pd.DataFrame,
    stats: pd.DataFrame,
    classifier: str,
    recipe: str,
    title: str,
    xlim: tuple[float, float],
) -> None:
    sub = display[(display["classifier"] == classifier) & (display["recipe"] == recipe)].copy()
    stat = stats[(stats["classifier"] == classifier) & (stats["recipe"] == recipe)].iloc[0]
    slope = float(stat["subject_intercept_slope_oriented"])
    q_value = float(stat["bh_fdr_q_subject_intercept"])

    ax.scatter(
        sub["centered_sdr_db"],
        sub["centered_delta_accuracy"],
        s=15,
        color=POINT_COLOR,
        alpha=0.78,
        linewidths=0,
        zorder=2,
    )

    ax.axhline(0, color=ZERO_COLOR, linewidth=0.65, alpha=0.90, linestyle=(0, (3, 2)), zorder=0)
    ax.axvline(0, color=ZERO_COLOR, linewidth=0.65, alpha=0.90, linestyle=(0, (3, 2)), zorder=0)
    xs = np.asarray(xlim)
    ax.plot(xs, slope * xs, color=FIT_COLOR, linewidth=1.35, zorder=3)

    ax.text(
        0.045,
        0.955,
        f"slope = {slope:+.3f}\nBH-FDR q = {format_q(q_value)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
        color=COLORS["text"],
    )
    ax.set_title(title, pad=5, fontsize=9.0)
    ax.tick_params(axis="both", labelsize=7.6)
    light_grid(ax)


def main() -> None:
    set_jne_style()
    display, stats = load_and_prepare()
    display_path, stats_path = write_figure_data(display, stats)
    xlim, ylim = common_limits(display)

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.75), sharex=True, sharey=True, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.025, wspace=0.06, hspace=0.02)

    for ax, (panel, classifier, recipe, title) in zip(axes, PANELS):
        draw_panel(ax, display, stats, classifier, recipe, title, xlim)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        add_panel_label(ax, panel, x=-0.16 if panel == "A" else -0.10, y=1.03)

    axes[0].set_ylabel("Within-subject centered Δ accuracy")
    axes[1].set_xlabel("Within-subject centered SDR (dB)")

    pdf_path, png_path = save_figure(fig, OUT_DIR / "fig04_same_condition_metric_utility", dpi=400)
    plt.close(fig)

    print("Sources used:")
    print(f"- {JOINED_PATH.relative_to(ROOT)}")
    print(f"- {TESTS_PATH.relative_to(ROOT)}")
    print(f"Observations plotted: {len(display)} subject-width display observations")
    print("Observation type: subject x width centered display aggregates")
    print("Aggregation before plotting: nuisance contamination/checkpoint/classifier repetitions averaged within subject x width; then centered within subject")
    print("Model annotations: exact SDR slope and BH-FDR q from full formal subject-intercept A4 model")
    print(f"Figure data: {display_path.relative_to(ROOT)}")
    print(f"Model statistics: {stats_path.relative_to(ROOT)}")
    print(f"PDF: {pdf_path.relative_to(ROOT)}")
    print(f"PNG: {png_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
