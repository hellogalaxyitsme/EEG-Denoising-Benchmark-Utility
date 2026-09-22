"""Supplementary Fig. S7: IV-2b replication and IV-2a domain-control deltas."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jne_figure_style import COLORS, WIDTH_FULL, add_panel_label, light_grid, save_figure, set_jne_style


ROOT = Path(__file__).resolve().parents[1]
A8_DIR = ROOT / "runs" / "a8_bci2b_downstream_replication_20260815_170804"
A9_DIR = ROOT / "runs" / "a9_bci_domain_shift_single_channel_20260815_211702"
A8_SUBJECT_CSV = A8_DIR / "a8_bci2b_subject_width_summary.csv"
A8_INFERENCE_CSV = A8_DIR / "a8_bci2b_width_inference.csv"
A9_SUBJECT_CSV = A9_DIR / "a9_subject_domain_summary.csv"
A9_INFERENCE_CSV = A9_DIR / "a9_domain_inference.csv"
A9_CONTRAST_CSV = A9_DIR / "a9_adaptation_contrasts.csv"
OUT_DIR = ROOT / "Revised_Paper" / "Images"
DATA_DIR = ROOT / "Revised_Paper" / "figure_data"

A8_ROWS = [
    ("csp_lda", "eog", 4, "CSP+LDA EOG base4"),
    ("csp_lda", "emg", 4, "CSP+LDA EMG base4"),
    ("csp_lda", "eog_emg_line", 16, "CSP+LDA EOG+EMG+LINE base16"),
    ("eegnet", "eog", 6, "EEGNet EOG base6"),
    ("eegnet", "emg", 6, "EEGNet EMG base6"),
    ("eegnet", "eog_emg_line", 6, "EEGNet EOG+EMG+LINE base6"),
]
DOMAIN_LABELS = {
    "zero_shot_external": "Zero-shot",
    "bci_adapted_single_channel": "Adapted",
}
WIDTH_COLORS = {6: COLORS["base6"], 16: COLORS["base16"]}


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    a8_subjects = pd.read_csv(A8_SUBJECT_CSV)
    a8_inference = pd.read_csv(A8_INFERENCE_CSV)
    a9_subjects = pd.read_csv(A9_SUBJECT_CSV)
    a9_inference = pd.read_csv(A9_INFERENCE_CSV)
    a9_contrasts = pd.read_csv(A9_CONTRAST_CSV)

    plotted_a8_subjects = []
    plotted_a8_effects = []
    for row_order, (classifier, recipe, base, label) in enumerate(A8_ROWS):
        s = a8_subjects[
            a8_subjects["classifier"].eq(classifier)
            & a8_subjects["recipe"].eq(recipe)
            & a8_subjects["base"].eq(base)
        ].copy()
        if len(s) != 9:
            raise ValueError(f"Expected 9 IV-2b subject rows for {classifier}/{recipe}/base{base}, found {len(s)}")
        s["row_order"] = row_order
        s["row_label"] = label
        s["delta_accuracy_pp"] = 100.0 * s["delta_accuracy"]
        plotted_a8_subjects.append(s)

        e = a8_inference[
            a8_inference["classifier"].eq(classifier)
            & a8_inference["recipe"].eq(recipe)
            & a8_inference["base"].eq(base)
        ].copy()
        if len(e) != 1:
            raise ValueError(f"Expected one IV-2b inference row for {classifier}/{recipe}/base{base}, found {len(e)}")
        e["row_order"] = row_order
        e["row_label"] = label
        e["mean_delta_accuracy_pp"] = 100.0 * e["mean_delta_accuracy"]
        e["bootstrap_ci95_low_pp"] = 100.0 * e["bootstrap_ci95_low"]
        e["bootstrap_ci95_high_pp"] = 100.0 * e["bootstrap_ci95_high"]
        plotted_a8_effects.append(e)

    a8_sub = pd.concat(plotted_a8_subjects, ignore_index=True)
    a8_eff = pd.concat(plotted_a8_effects, ignore_index=True)

    a9_sub = a9_subjects[
        a9_subjects["recipe"].eq("eog")
        & a9_subjects["base"].isin([6, 16])
        & a9_subjects["domain_condition"].isin(DOMAIN_LABELS)
    ].copy()
    if len(a9_sub) != 36:
        raise ValueError(f"Expected 36 A9 subject-domain rows, found {len(a9_sub)}")
    a9_sub["delta_accuracy_pp"] = 100.0 * a9_sub["delta_accuracy"]

    a9_inf = a9_inference[
        a9_inference["recipe"].eq("eog")
        & a9_inference["base"].isin([6, 16])
        & a9_inference["domain_condition"].isin(DOMAIN_LABELS)
    ].copy()
    if len(a9_inf) != 4:
        raise ValueError(f"Expected 4 A9 inference rows, found {len(a9_inf)}")
    a9_inf["mean_delta_accuracy_pp"] = 100.0 * a9_inf["mean_delta_accuracy"]
    a9_inf["bootstrap_ci95_low_pp"] = 100.0 * a9_inf["bootstrap_ci95_low"]
    a9_inf["bootstrap_ci95_high_pp"] = 100.0 * a9_inf["bootstrap_ci95_high"]

    a9_con = a9_contrasts[a9_contrasts["recipe"].eq("eog") & a9_contrasts["base"].isin([6, 16])].copy()
    if len(a9_con) != 2:
        raise ValueError(f"Expected 2 A9 adaptation contrasts, found {len(a9_con)}")

    a8_sub["source_file"] = str(A8_SUBJECT_CSV.relative_to(ROOT))
    a8_eff["source_file"] = str(A8_INFERENCE_CSV.relative_to(ROOT))
    a9_sub["source_file"] = str(A9_SUBJECT_CSV.relative_to(ROOT))
    a9_inf["source_file"] = str(A9_INFERENCE_CSV.relative_to(ROOT))
    a9_con["source_file"] = str(A9_CONTRAST_CSV.relative_to(ROOT))
    return a8_sub, a8_eff, a9_sub, a9_inf, a9_con


def save_source_data(
    a8_subjects: pd.DataFrame,
    a8_effects: pd.DataFrame,
    a9_subjects: pd.DataFrame,
    a9_inference: pd.DataFrame,
    a9_contrasts: pd.DataFrame,
) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    a8_subjects.to_csv(DATA_DIR / "supp_s7_iv2b_subject_deltas.csv", index=False)
    a8_effects.to_csv(DATA_DIR / "supp_s7_iv2b_effects.csv", index=False)
    a9_subjects.to_csv(DATA_DIR / "supp_s7_a9_subject_domain_summary.csv", index=False)
    a9_inference.to_csv(DATA_DIR / "supp_s7_a9_domain_inference.csv", index=False)
    a9_contrasts.to_csv(DATA_DIR / "supp_s7_a9_adaptation_contrasts.csv", index=False)


def draw_panel_a(ax, a8_subjects: pd.DataFrame, a8_effects: pd.DataFrame) -> None:
    y_by_row = {row_order: len(A8_ROWS) - 1 - row_order for row_order in range(len(A8_ROWS))}
    rng = np.random.default_rng(20260824)
    colors = {"csp_lda": COLORS["base4"], "eegnet": "#F28E2B"}

    ax.axvline(0, color=COLORS["text"], linewidth=0.75, alpha=0.75, zorder=0)
    ax.axhline(y_by_row[2] - 0.5, color=COLORS["grid"], linewidth=0.9, zorder=0)

    for row_order, (classifier, _, _, _) in enumerate(A8_ROWS):
        y = y_by_row[row_order]
        s = a8_subjects[a8_subjects["row_order"].eq(row_order)]
        jitter = rng.uniform(-0.13, 0.13, size=len(s))
        ax.scatter(
            s["delta_accuracy_pp"],
            y + jitter,
            s=13,
            color=colors[classifier],
            alpha=0.48,
            linewidths=0,
            zorder=2,
        )
        e = a8_effects[a8_effects["row_order"].eq(row_order)].iloc[0]
        lo = float(e["bootstrap_ci95_low_pp"])
        hi = float(e["bootstrap_ci95_high_pp"])
        mean = float(e["mean_delta_accuracy_pp"])
        ax.plot([lo, hi], [y, y], color=COLORS["text"], linewidth=1.0, zorder=3)
        ax.plot(mean, y, marker="D", color=COLORS["text"], markersize=3.5, zorder=4)

    ax.set_title("BCI IV-2b replication", loc="left", pad=4)
    ax.set_yticks([y_by_row[i] for i in range(len(A8_ROWS))])
    ax.set_yticklabels([label for *_, label in A8_ROWS])
    ax.set_ylim(-0.7, len(A8_ROWS) - 0.3)
    ax.set_xlim(-12, 12)
    ax.set_xlabel("Delta accuracy (pp)")
    light_grid(ax, axis="x")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def draw_panel_b(ax, a9_subjects: pd.DataFrame, a9_inference: pd.DataFrame, a9_contrasts: pd.DataFrame) -> None:
    x_positions = {
        (6, "zero_shot_external"): 0.0,
        (6, "bci_adapted_single_channel"): 0.78,
        (16, "zero_shot_external"): 2.0,
        (16, "bci_adapted_single_channel"): 2.78,
    }
    rng = np.random.default_rng(20260824)
    subjects = sorted(a9_subjects["subject"].unique())

    ax.axhline(0, color=COLORS["text"], linewidth=0.75, alpha=0.75, zorder=0)

    for base in [6, 16]:
        wide = a9_subjects[a9_subjects["base"].eq(base)].pivot(
            index="subject",
            columns="domain_condition",
            values="delta_accuracy_pp",
        )
        for subject in subjects:
            row = wide.loc[subject]
            xs = [x_positions[(base, "zero_shot_external")], x_positions[(base, "bci_adapted_single_channel")]]
            ys = [row["zero_shot_external"], row["bci_adapted_single_channel"]]
            ax.plot(xs, ys, color=COLORS["signal"], linewidth=0.7, alpha=0.27, zorder=1)
            ax.scatter(xs, ys, s=12, color=WIDTH_COLORS[base], alpha=0.52, linewidths=0, zorder=2)

    for base in [6, 16]:
        for domain in ["zero_shot_external", "bci_adapted_single_channel"]:
            row = a9_inference[
                a9_inference["base"].eq(base) & a9_inference["domain_condition"].eq(domain)
            ].iloc[0]
            x = x_positions[(base, domain)]
            x = x + rng.uniform(-0.015, 0.015)
            mean = float(row["mean_delta_accuracy_pp"])
            lo = float(row["bootstrap_ci95_low_pp"])
            hi = float(row["bootstrap_ci95_high_pp"])
            ax.errorbar(
                x,
                mean,
                yerr=[[mean - lo], [hi - mean]],
                fmt="D",
                color=COLORS["text"],
                ecolor=COLORS["text"],
                markersize=3.6,
                elinewidth=0.9,
                capsize=2.0,
                capthick=0.75,
                zorder=4,
            )

    contrast6 = a9_contrasts[a9_contrasts["base"].eq(6)].iloc[0]
    contrast16 = a9_contrasts[a9_contrasts["base"].eq(16)].iloc[0]
    ax.text(
        0.39,
        0.05,
        f"+{100 * float(contrast6['mean_contrast']):.1f} pp\n7/9 improved",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=6.0,
        color=COLORS["text"],
        bbox=dict(boxstyle="round,pad=0.13", facecolor="white", edgecolor=COLORS["grid"], linewidth=0.45, alpha=0.92),
    )
    ax.text(
        2.39,
        0.05,
        f"+{100 * float(contrast16['mean_contrast']):.1f} pp\n4/9 improved",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=6.0,
        color=COLORS["text"],
        bbox=dict(boxstyle="round,pad=0.13", facecolor="white", edgecolor=COLORS["grid"], linewidth=0.45, alpha=0.92),
    )

    ax.set_title("IV-2a domain adaptation", loc="left", pad=4)
    ax.set_xticks([0.0, 0.78, 2.0, 2.78])
    ax.set_xticklabels(["Zero", "Adapt", "Zero", "Adapt"])
    ax.text(0.39, -0.22, "base6", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7.0)
    ax.text(2.39, -0.22, "base16", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7.0)
    ax.set_xlim(-0.42, 3.2)
    ax.set_ylim(-18, 3)
    ax.set_ylabel("Delta accuracy (pp)")
    light_grid(ax, axis="y")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def main() -> None:
    set_jne_style()
    a8_subjects, a8_effects, a9_subjects, a9_inference, a9_contrasts = load_inputs()
    save_source_data(a8_subjects, a8_effects, a9_subjects, a9_inference, a9_contrasts)

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_FULL, 3.25), constrained_layout=False, gridspec_kw={"width_ratios": [1.12, 1.0]})
    fig.subplots_adjust(left=0.22, right=0.985, bottom=0.2, top=0.88, wspace=0.32)

    draw_panel_a(axes[0], a8_subjects, a8_effects)
    draw_panel_b(axes[1], a9_subjects, a9_inference, a9_contrasts)
    add_panel_label(axes[0], "A", x=-0.47, y=1.04)
    add_panel_label(axes[1], "B", x=-0.17, y=1.04)

    pdf, png = save_figure(fig, OUT_DIR / "supp_s7_iv2b_and_domain_controls", dpi=500)
    plt.close(fig)

    print(f"[saved] {pdf}")
    print(f"[saved] {png}")
    print(f"[data] {DATA_DIR / 'supp_s7_iv2b_subject_deltas.csv'}")
    print(f"[data] {DATA_DIR / 'supp_s7_a9_subject_domain_summary.csv'}")


if __name__ == "__main__":
    main()
