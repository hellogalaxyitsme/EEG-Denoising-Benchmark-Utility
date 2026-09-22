"""Final Supplementary Fig. S3: BCI zero-shot all-subject capacity trends."""

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
RUN_DIR = ROOT / "runs" / "a7_bci_zero_shot_all_subjects_20260815_000000"
ROWS_CSV = RUN_DIR / "a7_subject_width_rows.csv"
SUMMARY_CSV = RUN_DIR / "a7_subject_width_summary.csv"
OUT_DIR = ROOT / "Revised_Paper" / "Images"
DATA_DIR = ROOT / "Revised_Paper" / "figure_data"

DATASETS = [("BCI_IV2a", "IV-2a"), ("BCI_IV2b", "IV-2b")]
METRICS = [("CC", "CC"), ("SDR", "SDR (dB)")]
WIDTHS = [2, 4, 6, 8, 16]
WIDTH_LABELS = ["b2", "b4", "b6", "b8", "b16"]


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = pd.read_csv(ROWS_CSV)
    summary = pd.read_csv(SUMMARY_CSV)
    rows = rows[rows["dataset"].isin([d for d, _ in DATASETS]) & rows["base"].isin(WIDTHS)].copy()
    summary = summary[summary["dataset"].isin([d for d, _ in DATASETS]) & summary["base"].isin(WIDTHS)].copy()

    expected_subject_rows = len(DATASETS) * 9 * len(WIDTHS)
    if len(rows) != expected_subject_rows:
        raise ValueError(f"Expected {expected_subject_rows} subject-width rows, found {len(rows)}")
    for dataset, _ in DATASETS:
        subjects = sorted(rows.loc[rows["dataset"].eq(dataset), "subject"].unique())
        if len(subjects) != 9:
            raise ValueError(f"Expected 9 subjects for {dataset}, found {len(subjects)}")
        for subject in subjects:
            bases = sorted(rows.loc[rows["dataset"].eq(dataset) & rows["subject"].eq(subject), "base"].astype(int))
            if bases != WIDTHS:
                raise ValueError(f"{dataset} {subject} has bases {bases}, expected {WIDTHS}")

    rows["source_file"] = str(ROWS_CSV.relative_to(ROOT))
    rows["row_type"] = "subject"
    summary["source_file"] = str(SUMMARY_CSV.relative_to(ROOT))
    summary["row_type"] = "mean_ci"
    return rows, summary


def save_source_data(rows: pd.DataFrame, summary: pd.DataFrame) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    subject_rows = rows[
        ["row_type", "dataset", "subject", "base", "CC", "SDR", "n_checkpoints", "train_seeds", "source_file"]
    ].copy()
    for metric, _ in METRICS:
        subject_rows[f"{metric}_mean"] = pd.NA
        subject_rows[f"{metric}_ci95_low"] = pd.NA
        subject_rows[f"{metric}_ci95_high"] = pd.NA
    subject_rows["n_subjects"] = pd.NA

    summary_rows = pd.DataFrame(
        {
            "row_type": summary["row_type"],
            "dataset": summary["dataset"],
            "subject": pd.NA,
            "base": summary["base"],
            "CC": pd.NA,
            "SDR": pd.NA,
            "n_checkpoints": pd.NA,
            "train_seeds": pd.NA,
            "source_file": summary["source_file"],
            "CC_mean": summary["CC_mean"],
            "CC_ci95_low": summary["CC_ci95_low"],
            "CC_ci95_high": summary["CC_ci95_high"],
            "SDR_mean": summary["SDR_mean"],
            "SDR_ci95_low": summary["SDR_ci95_low"],
            "SDR_ci95_high": summary["SDR_ci95_high"],
            "n_subjects": summary["n_subjects"],
        }
    )

    output_columns = [
        "row_type",
        "dataset",
        "subject",
        "base",
        "CC",
        "SDR",
        "n_checkpoints",
        "train_seeds",
        "CC_mean",
        "CC_ci95_low",
        "CC_ci95_high",
        "SDR_mean",
        "SDR_ci95_low",
        "SDR_ci95_high",
        "n_subjects",
        "source_file",
    ]

    out = DATA_DIR / "supp_s3_bci_zeroshot_subjects.csv"
    pd.concat([subject_rows[output_columns], summary_rows[output_columns]], ignore_index=True).to_csv(out, index=False)
    return out


def metric_limits(rows: pd.DataFrame, summary: pd.DataFrame, metric: str) -> tuple[float, float]:
    values = list(rows[metric].dropna())
    values += list(summary[f"{metric}_ci95_low"].dropna())
    values += list(summary[f"{metric}_ci95_high"].dropna())
    lo = float(np.min(values))
    hi = float(np.max(values))
    pad = 0.08 * (hi - lo) if hi > lo else 0.05
    return lo - pad, hi + pad


def draw_panel(
    ax,
    rows: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    dataset: str,
    metric: str,
    y_limits: tuple[float, float],
) -> None:
    sub = rows[rows["dataset"].eq(dataset)].copy()
    summ = summary[summary["dataset"].eq(dataset)].copy()
    x_by_width = {width: idx for idx, width in enumerate(WIDTHS)}

    for subject in sorted(sub["subject"].unique()):
        subject_rows = sub[sub["subject"].eq(subject)].sort_values("base")
        x = subject_rows["base"].map(x_by_width).to_numpy(dtype=float)
        y = subject_rows[metric].to_numpy(dtype=float)
        ax.plot(x, y, color=COLORS["signal"], linewidth=0.55, alpha=0.30, zorder=1)
        ax.scatter(x, y, s=10, color=COLORS["signal"], alpha=0.55, linewidths=0, zorder=2)

    means = []
    ci_low = []
    ci_high = []
    for width in WIDTHS:
        row = summ[summ["base"].eq(width)]
        if len(row) != 1:
            raise ValueError(f"Expected one summary row for {dataset} base{width}, found {len(row)}")
        row = row.iloc[0]
        mean = float(row[f"{metric}_mean"])
        means.append(mean)
        ci_low.append(mean - float(row[f"{metric}_ci95_low"]))
        ci_high.append(float(row[f"{metric}_ci95_high"]) - mean)

    ax.errorbar(
        np.arange(len(WIDTHS)),
        means,
        yerr=np.vstack([ci_low, ci_high]),
        fmt="o",
        markersize=4.0,
        color=COLORS["accent"],
        ecolor=COLORS["accent"],
        elinewidth=0.9,
        capsize=2.2,
        capthick=0.8,
        zorder=4,
    )

    ax.set_xticks(np.arange(len(WIDTHS)))
    ax.set_xticklabels(WIDTH_LABELS, rotation=0)
    ax.set_xlim(-0.35, len(WIDTHS) - 0.65)
    ax.set_ylim(*y_limits)
    light_grid(ax, axis="y")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def main() -> None:
    set_jne_style()
    rows, summary = load_inputs()
    data_path = save_source_data(rows, summary)
    limits = {metric: metric_limits(rows, summary, metric) for metric, _ in METRICS}

    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_FULL, 4.1), constrained_layout=False)
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.14, top=0.82, wspace=0.26, hspace=0.42)

    panels = [
        ("A", "BCI_IV2a", "CC", "IV-2a CC"),
        ("B", "BCI_IV2a", "SDR", "IV-2a SDR"),
        ("C", "BCI_IV2b", "CC", "IV-2b CC"),
        ("D", "BCI_IV2b", "SDR", "IV-2b SDR"),
    ]
    for ax, (label, dataset, metric, title) in zip(axes.ravel(), panels, strict=True):
        draw_panel(ax, rows, summary, dataset=dataset, metric=metric, y_limits=limits[metric])
        ax.set_title(title, pad=5)
        add_panel_label(ax, label, x=-0.12, y=1.06)

    axes[0, 0].set_ylabel("CC")
    axes[1, 0].set_ylabel("CC")
    axes[0, 1].set_ylabel("SDR (dB)")
    axes[1, 1].set_ylabel("SDR (dB)")
    handles = [
        Line2D([0], [0], color=COLORS["signal"], marker="o", lw=0.6, markersize=3.5, alpha=0.55, label="Subject"),
        Line2D([0], [0], color=COLORS["accent"], marker="o", lw=0.9, markersize=4.0, label="Mean + 95% CI"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=2,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.4,
        borderpad=0.1,
    )

    pdf, png = save_figure(fig, OUT_DIR / "supp_s3_bci_zeroshot_final", dpi=400)
    plt.close(fig)

    print(f"source file(s) used: {ROWS_CSV.relative_to(ROOT)}, {SUMMARY_CSV.relative_to(ROOT)}")
    print(f"number of observations plotted: {len(rows)} subject-width rows plus {len(summary)} mean/CI rows")
    print("observations represent: BCI human subjects; checkpoint/channel repetitions are already aggregated within subject")
    print("aggregation performed before plotting: used stored subject-width rows and subject-bootstrap CI summaries")
    print(f"source data saved: {data_path.relative_to(ROOT)}")
    print(f"final output paths: {pdf.relative_to(ROOT)}, {png.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
