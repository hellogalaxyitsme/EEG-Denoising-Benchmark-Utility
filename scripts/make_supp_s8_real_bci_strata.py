"""Supplementary Fig. S8: real BCI IV-2a paired raw/denoised strata."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jne_figure_style import COLORS, WIDTH_FULL, add_panel_label, light_grid, save_figure, set_jne_style


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "a12_real_bci2a_eog_stratified_downstream_20260816_142449"
SUBJECT_CSV = RUN_DIR / "a12_real_downstream_subject_aggregates.csv"
INFERENCE_CSV = RUN_DIR / "a12_real_downstream_inference.csv"
CONTRAST_CSV = RUN_DIR / "a12_quiet_heavy_contrasts.csv"
STRATUM_CSV = RUN_DIR / "a12_eog_stratum_rows.csv"
OUT_DIR = ROOT / "Revised_Paper" / "Images"
DATA_DIR = ROOT / "Revised_Paper" / "figure_data"

BASE = 16
STRATA = [
    ("all", "All A0xE trials"),
    ("quiet_low_eog", "Low-EOG / quiet trials"),
    ("artifact_heavy_high_eog", "High-EOG / artifact-heavy trials"),
]
STRATUM_COLORS = {
    "all": "#4C78A8",
    "quiet_low_eog": "#2A9D8F",
    "artifact_heavy_high_eog": "#E15759",
}


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subjects = pd.read_csv(SUBJECT_CSV)
    inference = pd.read_csv(INFERENCE_CSV)
    contrasts = pd.read_csv(CONTRAST_CSV)
    strata = pd.read_csv(STRATUM_CSV)

    subjects = subjects[
        subjects["condition"].eq("denoised_denoised")
        & subjects["base"].eq(BASE)
        & subjects["stratum"].isin([s for s, _ in STRATA])
    ].copy()
    if len(subjects) != 27:
        raise ValueError(f"Expected 27 base16 subject rows, found {len(subjects)}")
    for stratum, _ in STRATA:
        n_subjects = subjects.loc[subjects["stratum"].eq(stratum), "subject"].nunique()
        if n_subjects != 9:
            raise ValueError(f"Expected 9 subjects for {stratum}, found {n_subjects}")

    inference = inference[
        inference["condition"].eq("denoised_denoised")
        & inference["base"].eq(BASE)
        & inference["stratum"].isin([s for s, _ in STRATA])
    ].copy()
    if len(inference) != 3:
        raise ValueError(f"Expected 3 base16 inference rows, found {len(inference)}")

    contrasts = contrasts[
        contrasts["condition"].eq("denoised_denoised")
        & contrasts["base"].eq(BASE)
        & contrasts["contrast"].eq("quiet_low_eog_delta_minus_artifact_heavy_high_eog_delta")
    ].copy()
    if len(contrasts) != 1:
        raise ValueError(f"Expected one base16 quiet-heavy contrast, found {len(contrasts)}")

    subjects["raw_accuracy_pct"] = 100.0 * subjects["raw_raw_accuracy"]
    subjects["denoised_accuracy_pct"] = 100.0 * subjects["accuracy_mean"]
    subjects["delta_accuracy_pp"] = 100.0 * subjects["delta_accuracy_vs_raw_raw"]

    inference["raw_accuracy_subject_mean_pct"] = 100.0 * inference["raw_raw_accuracy_subject_mean"]
    inference["denoised_accuracy_subject_mean_pct"] = 100.0 * inference["denoised_accuracy_subject_mean"]
    inference["delta_accuracy_mean_pp"] = 100.0 * inference["delta_accuracy_mean"]
    inference["delta_ci95_low_pp"] = 100.0 * inference["delta_accuracy_ci95_low_subject_bootstrap"]
    inference["delta_ci95_high_pp"] = 100.0 * inference["delta_accuracy_ci95_high_subject_bootstrap"]

    contrasts["mean_contrast_pp"] = 100.0 * contrasts["mean_contrast"]
    contrasts["ci95_low_pp"] = 100.0 * contrasts["ci95_low_subject_bootstrap"]
    contrasts["ci95_high_pp"] = 100.0 * contrasts["ci95_high_subject_bootstrap"]

    subjects["source_file"] = str(SUBJECT_CSV.relative_to(ROOT))
    inference["source_file"] = str(INFERENCE_CSV.relative_to(ROOT))
    contrasts["source_file"] = str(CONTRAST_CSV.relative_to(ROOT))
    strata["source_file"] = str(STRATUM_CSV.relative_to(ROOT))
    return subjects, inference, contrasts, strata


def save_source_data(
    subjects: pd.DataFrame,
    inference: pd.DataFrame,
    contrasts: pd.DataFrame,
    strata: pd.DataFrame,
) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    subjects.to_csv(DATA_DIR / "supp_s8_base16_subject_pairs.csv", index=False)
    inference.to_csv(DATA_DIR / "supp_s8_base16_inference.csv", index=False)
    contrasts.to_csv(DATA_DIR / "supp_s8_quiet_heavy_contrast.csv", index=False)
    strata.to_csv(DATA_DIR / "supp_s8_eog_stratum_rows.csv", index=False)


def draw_panel(ax, subjects: pd.DataFrame, inference: pd.DataFrame, stratum: str, title: str, ylim: tuple[float, float]) -> None:
    sub = subjects[subjects["stratum"].eq(stratum)].sort_values("subject")
    inf = inference[inference["stratum"].eq(stratum)].iloc[0]
    color = STRATUM_COLORS[stratum]

    x_raw = 0.0
    x_den = 1.0
    rng = np.random.default_rng(20260824)
    for row in sub.itertuples():
        jitter = rng.uniform(-0.035, 0.035)
        xs = [x_raw + jitter, x_den + jitter]
        ys = [row.raw_accuracy_pct, row.denoised_accuracy_pct]
        ax.plot(xs, ys, color=COLORS["signal"], linewidth=0.75, alpha=0.32, zorder=1)
        ax.scatter(xs, ys, s=14, color=color, alpha=0.62, linewidths=0, zorder=2)

    ax.plot(
        [x_raw, x_den],
        [inf["raw_accuracy_subject_mean_pct"], inf["denoised_accuracy_subject_mean_pct"]],
        color=COLORS["text"],
        linewidth=1.25,
        zorder=4,
    )
    ax.scatter(
        [x_raw, x_den],
        [inf["raw_accuracy_subject_mean_pct"], inf["denoised_accuracy_subject_mean_pct"]],
        marker="D",
        s=24,
        color=COLORS["text"],
        zorder=5,
    )

    delta = inf["delta_accuracy_mean_pp"]
    ci_low = inf["delta_ci95_low_pp"]
    ci_high = inf["delta_ci95_high_pp"]
    ax.text(
        0.5,
        0.035,
        f"Delta {delta:+.1f} pp\n95% CI [{ci_low:+.1f}, {ci_high:+.1f}]",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.0,
        color=COLORS["text"],
        bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor=COLORS["grid"], linewidth=0.45, alpha=0.94),
    )

    ax.set_title(title, pad=4)
    ax.set_xticks([x_raw, x_den])
    ax.set_xticklabels(["Raw", "Denoised"])
    ax.set_xlim(-0.32, 1.32)
    ax.set_ylim(*ylim)
    light_grid(ax, axis="y")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def main() -> None:
    set_jne_style()
    subjects, inference, contrasts, strata = load_inputs()
    save_source_data(subjects, inference, contrasts, strata)

    all_acc = pd.concat([subjects["raw_accuracy_pct"], subjects["denoised_accuracy_pct"]], ignore_index=True)
    lo = float(np.floor((all_acc.min() - 4.0) / 5.0) * 5.0)
    hi = float(np.ceil((all_acc.max() + 4.0) / 5.0) * 5.0)

    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_FULL, 3.05), constrained_layout=False, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.22, top=0.78, wspace=0.26)

    for ax, panel, (stratum, title) in zip(axes, "ABC", STRATA, strict=True):
        draw_panel(ax, subjects, inference, stratum, title, (lo, hi))
        add_panel_label(ax, panel, x=-0.22, y=1.06)
    axes[0].set_ylabel("CSP+LDA accuracy (%)")

    contrast = contrasts.iloc[0]
    fig.text(
        0.08,
        0.93,
        (
            "Base16 paired real-recording evaluation. Quiet-heavy delta contrast: "
            f"{contrast['mean_contrast_pp']:+.1f} pp, 95% CI "
            f"[{contrast['ci95_low_pp']:+.1f}, {contrast['ci95_high_pp']:+.1f}], "
            f"BH q={contrast['bh_fdr_q_quiet_more_negative']:.3f}; not corrected-significant."
        ),
        ha="left",
        va="center",
        fontsize=6.3,
        color=COLORS["text"],
    )

    pdf, png = save_figure(fig, OUT_DIR / "supp_s8_real_bci_strata", dpi=500)
    plt.close(fig)

    print(f"[saved] {pdf}")
    print(f"[saved] {png}")
    print(f"[data] {DATA_DIR / 'supp_s8_base16_subject_pairs.csv'}")
    print(f"[data] {DATA_DIR / 'supp_s8_base16_inference.csv'}")
    print(f"[data] {DATA_DIR / 'supp_s8_quiet_heavy_contrast.csv'}")


if __name__ == "__main__":
    main()
