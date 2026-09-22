"""Final Supplementary Fig. S5: IV-2b replication and real-recording controls."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from jne_figure_style import COLORS, WIDTH_FULL, add_panel_label, light_grid, save_figure, set_jne_style


ROOT = Path(__file__).resolve().parents[1]
A8_DIR = ROOT / "runs" / "a8_bci2b_downstream_replication_20260815_170804"
A12_DIR = ROOT / "runs" / "a12_real_bci2a_eog_stratified_downstream_20260816_142449"
A8_SUBJECT_CSV = A8_DIR / "a8_bci2b_subject_width_summary.csv"
A8_INFERENCE_CSV = A8_DIR / "a8_bci2b_width_inference.csv"
A12_SUBJECT_CSV = A12_DIR / "a12_real_downstream_subject_aggregates.csv"
A12_INFERENCE_CSV = A12_DIR / "a12_real_downstream_inference.csv"
OUT_DIR = ROOT / "Revised_Paper" / "Images"
DATA_DIR = ROOT / "Revised_Paper" / "figure_data"

ACCENT = COLORS["accent"]
SUBJECT_COLOR = COLORS["signal"]
X_LIM = (-35.0, 25.0)
X_TICKS = [-30, -20, -10, 0, 10, 20]

A8_ROWS = [
    ("csp_lda", "eog", 4, "CSP+LDA — EOG"),
    ("csp_lda", "emg", 4, "CSP+LDA — EMG"),
    ("csp_lda", "eog_emg_line", 16, "CSP+LDA — EOG+EMG+LINE"),
    ("eegnet", "eog", 6, "EEGNet — EOG"),
    ("eegnet", "emg", 6, "EEGNet — EMG"),
    ("eegnet", "eog_emg_line", 6, "EEGNet — EOG+EMG+LINE"),
]
A12_ROWS = [
    ("all", "All recorded trials"),
    ("quiet_low_eog", "Low-EOG / quiet trials"),
    ("artifact_heavy_high_eog", "High-EOG / artifact-heavy trials"),
]


def load_a8() -> tuple[pd.DataFrame, pd.DataFrame]:
    subjects = pd.read_csv(A8_SUBJECT_CSV)
    inference = pd.read_csv(A8_INFERENCE_CSV)

    subject_rows = []
    inference_rows = []
    for row_order, (classifier, recipe, base, row_label) in enumerate(A8_ROWS):
        sub = subjects[
            subjects["classifier"].eq(classifier)
            & subjects["recipe"].eq(recipe)
            & subjects["base"].eq(base)
        ].copy()
        if len(sub) != 9:
            raise ValueError(f"Expected 9 IV-2b subject rows for {classifier}/{recipe}/base{base}, found {len(sub)}")
        sub["row_order"] = row_order
        sub["row_label"] = row_label
        sub["delta_accuracy_pp"] = 100.0 * sub["delta_accuracy"]
        sub["source_file"] = str(A8_SUBJECT_CSV.relative_to(ROOT))
        subject_rows.append(sub)

        inf = inference[
            inference["classifier"].eq(classifier)
            & inference["recipe"].eq(recipe)
            & inference["base"].eq(base)
        ].copy()
        if len(inf) != 1:
            raise ValueError(f"Expected one IV-2b inference row for {classifier}/{recipe}/base{base}, found {len(inf)}")
        inf["row_order"] = row_order
        inf["row_label"] = row_label
        inf["mean_delta_accuracy_pp"] = 100.0 * inf["mean_delta_accuracy"]
        inf["ci95_low_pp"] = 100.0 * inf["bootstrap_ci95_low"]
        inf["ci95_high_pp"] = 100.0 * inf["bootstrap_ci95_high"]
        inf["source_file"] = str(A8_INFERENCE_CSV.relative_to(ROOT))
        inference_rows.append(inf)

    return pd.concat(subject_rows, ignore_index=True), pd.concat(inference_rows, ignore_index=True)


def load_a12() -> tuple[pd.DataFrame, pd.DataFrame]:
    subjects = pd.read_csv(A12_SUBJECT_CSV)
    inference = pd.read_csv(A12_INFERENCE_CSV)

    wanted_strata = [stratum for stratum, _ in A12_ROWS]
    subjects = subjects[
        subjects["condition"].eq("denoised_denoised")
        & subjects["base"].eq(16)
        & subjects["stratum"].isin(wanted_strata)
    ].copy()
    if len(subjects) != 27:
        raise ValueError(f"Expected 27 base16 real-recording subject rows, found {len(subjects)}")

    inference = inference[
        inference["condition"].eq("denoised_denoised")
        & inference["base"].eq(16)
        & inference["stratum"].isin(wanted_strata)
    ].copy()
    if len(inference) != 3:
        raise ValueError(f"Expected 3 base16 real-recording inference rows, found {len(inference)}")

    labels = dict(A12_ROWS)
    order = {stratum: idx for idx, (stratum, _) in enumerate(A12_ROWS)}
    subjects["row_order"] = subjects["stratum"].map(order)
    subjects["row_label"] = subjects["stratum"].map(labels)
    subjects["delta_accuracy_pp"] = 100.0 * subjects["delta_accuracy_vs_raw_raw"]
    subjects["source_file"] = str(A12_SUBJECT_CSV.relative_to(ROOT))

    inference["row_order"] = inference["stratum"].map(order)
    inference["row_label"] = inference["stratum"].map(labels)
    inference["mean_delta_accuracy_pp"] = 100.0 * inference["delta_accuracy_mean"]
    inference["ci95_low_pp"] = 100.0 * inference["delta_accuracy_ci95_low_subject_bootstrap"]
    inference["ci95_high_pp"] = 100.0 * inference["delta_accuracy_ci95_high_subject_bootstrap"]
    inference["source_file"] = str(A12_INFERENCE_CSV.relative_to(ROOT))

    for stratum, _ in A12_ROWS:
        n_subjects = subjects.loc[subjects["stratum"].eq(stratum), "subject"].nunique()
        if n_subjects != 9:
            raise ValueError(f"Expected 9 subjects for real-recording stratum {stratum}, found {n_subjects}")

    return subjects, inference


def save_source_data(
    a8_subjects: pd.DataFrame,
    a8_inference: pd.DataFrame,
    a12_subjects: pd.DataFrame,
    a12_inference: pd.DataFrame,
) -> tuple[Path, Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    a8_out = DATA_DIR / "supp_s5_iv2b_subject_effects.csv"
    a12_out = DATA_DIR / "supp_s5_real_recording_subject_effects.csv"

    a8_subject_export = a8_subjects[
        [
            "row_order",
            "row_label",
            "dataset",
            "classifier",
            "recipe",
            "base",
            "subject",
            "delta_accuracy_pp",
            "n_contamination_seed_pairs",
            "n_checkpoint_seeds",
            "n_classifier_seeds",
            "source_file",
        ]
    ].copy()
    a8_subject_export["row_type"] = "subject"
    a8_inference_export = a8_inference[
        [
            "row_order",
            "row_label",
            "dataset",
            "classifier",
            "recipe",
            "base",
            "n_subjects",
            "mean_delta_accuracy_pp",
            "ci95_low_pp",
            "ci95_high_pp",
            "source_file",
        ]
    ].copy()
    a8_inference_export["row_type"] = "mean_ci"
    pd.concat([a8_subject_export, a8_inference_export], ignore_index=True, sort=False).to_csv(a8_out, index=False)

    a12_subject_export = a12_subjects[
        [
            "row_order",
            "row_label",
            "condition",
            "stratum",
            "base",
            "subject",
            "delta_accuracy_pp",
            "n_checkpoints",
            "train_seeds",
            "source_file",
        ]
    ].copy()
    a12_subject_export["row_type"] = "subject"
    a12_inference_export = a12_inference[
        [
            "row_order",
            "row_label",
            "condition",
            "stratum",
            "base",
            "n_subjects",
            "mean_delta_accuracy_pp",
            "ci95_low_pp",
            "ci95_high_pp",
            "source_file",
        ]
    ].copy()
    a12_inference_export["row_type"] = "mean_ci"
    pd.concat([a12_subject_export, a12_inference_export], ignore_index=True, sort=False).to_csv(a12_out, index=False)
    return a8_out, a12_out


def draw_forest(
    ax,
    subject_rows: pd.DataFrame,
    inference_rows: pd.DataFrame,
    *,
    n_rows: int,
    row_labels: list[str],
    separator_after_row: int | None = None,
) -> None:
    rng = np.random.default_rng(20260919)
    y_by_order = {row_order: n_rows - 1 - row_order for row_order in range(n_rows)}

    ax.axvline(0, color=COLORS["text"], linewidth=0.75, alpha=0.70, zorder=0)
    if separator_after_row is not None:
        ax.axhline(y_by_order[separator_after_row] - 0.5, color=COLORS["grid"], linewidth=0.9, zorder=0)

    for row_order in range(n_rows):
        y = y_by_order[row_order]
        sub = subject_rows[subject_rows["row_order"].eq(row_order)].copy()
        inf = inference_rows[inference_rows["row_order"].eq(row_order)].iloc[0]
        jitter = rng.uniform(-0.13, 0.13, size=len(sub))
        ax.scatter(
            sub["delta_accuracy_pp"],
            y + jitter,
            s=13,
            color=SUBJECT_COLOR,
            alpha=0.45,
            linewidths=0,
            zorder=2,
        )
        mean = float(inf["mean_delta_accuracy_pp"])
        lo = float(inf["ci95_low_pp"])
        hi = float(inf["ci95_high_pp"])
        ax.plot([lo, hi], [y, y], color=ACCENT, linewidth=1.15, zorder=3)
        ax.plot(mean, y, marker="o", color=ACCENT, markersize=4.2, zorder=4)

    ax.set_yticks([y_by_order[i] for i in range(n_rows)])
    ax.set_yticklabels(row_labels)
    ax.tick_params(axis="y", labelsize=7.8)
    ax.set_ylim(-0.7, n_rows - 0.3)
    ax.set_xlim(*X_LIM)
    ax.set_xticks(X_TICKS)
    ax.set_xlabel("Δ accuracy (percentage points)")
    light_grid(ax, axis="x")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def main() -> None:
    set_jne_style()
    a8_subjects, a8_inference = load_a8()
    a12_subjects, a12_inference = load_a12()
    a8_data_path, a12_data_path = save_source_data(a8_subjects, a8_inference, a12_subjects, a12_inference)

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_FULL, 3.45), constrained_layout=False)
    fig.subplots_adjust(left=0.235, right=0.995, bottom=0.24, top=0.84, wspace=0.48)

    draw_forest(
        axes[0],
        a8_subjects,
        a8_inference,
        n_rows=len(A8_ROWS),
        row_labels=[label for *_, label in A8_ROWS],
        separator_after_row=2,
    )
    draw_forest(
        axes[1],
        a12_subjects,
        a12_inference,
        n_rows=len(A12_ROWS),
        row_labels=[label for _, label in A12_ROWS],
    )

    axes[0].set_title("BCI IV-2b replication", pad=5)
    axes[1].set_title("IV-2a unmodified recordings", pad=5)
    axes[1].yaxis.tick_right()
    axes[1].tick_params(axis="y", labelleft=False, labelright=True, length=0, pad=5)
    add_panel_label(axes[0], "A", x=-0.32, y=1.06)
    add_panel_label(axes[1], "B", x=-0.32, y=1.06)

    handles = [
        Line2D([0], [0], marker="o", color=SUBJECT_COLOR, lw=0, markersize=4.0, alpha=0.45, label="Subject"),
        Line2D([0], [0], marker="o", color=ACCENT, lw=1.15, markersize=4.2, label="Mean + 95% CI"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=2,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.5,
        borderpad=0.1,
    )

    pdf, png = save_figure(fig, OUT_DIR / "supp_s5_replication_real_controls_final", dpi=400)
    plt.close(fig)

    print(f"source file(s) used: {A8_SUBJECT_CSV.relative_to(ROOT)}, {A8_INFERENCE_CSV.relative_to(ROOT)}, {A12_SUBJECT_CSV.relative_to(ROOT)}, {A12_INFERENCE_CSV.relative_to(ROOT)}")
    print(f"number of observations plotted: {len(a8_subjects)} IV-2b subject rows plus {len(a12_subjects)} real-recording subject rows")
    print("observations represent: human subjects; nuisance contamination/checkpoint/classifier repetitions are already averaged within subject")
    print("aggregation performed before plotting: used stored subject-level deltas and subject-bootstrap CIs")
    print(f"source data saved: {a8_data_path.relative_to(ROOT)}, {a12_data_path.relative_to(ROOT)}")
    print(f"final output paths: {pdf.relative_to(ROOT)}, {png.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
