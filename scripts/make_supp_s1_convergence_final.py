"""Final Supplementary Fig. S1: A6 convergence/optimization audit."""

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

from jne_figure_style import COLORS, WIDTH_FULL, light_grid, save_figure, set_jne_style


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "a6_optimization_convergence_audit_20260815_000000"
CURVES_CSV = RUN_DIR / "a6_seed_mean_curves.csv"
OUT_DIR = ROOT / "Revised_Paper" / "Images"
DATA_DIR = ROOT / "Revised_Paper" / "figure_data"

TASKS = [
    ("eog", "EOG", 75, 25),
    ("emg", "EMG", 75, 25),
    ("mixed1m", "Mixed-1M", 30, 10),
]
METRICS = [
    ("train_loss", "Training loss"),
    ("val_SDR", "Validation SDR (dB)"),
]
BASES = [4, 6, 16]
BASE_LABELS = {4: "base4", 6: "base6", 16: "base16"}
WIDTH_COLORS = {
    4: "#4C78A8",
    6: "#59A14F",
    16: "#B55A55",
}


def load_curves() -> pd.DataFrame:
    curves = pd.read_csv(CURVES_CSV)
    curves = curves[
        curves["budget"].eq("extended")
        & curves["base"].isin(BASES)
        & np.isclose(curves["lr"].astype(float), 0.001)
        & curves["metric"].isin([metric for metric, _ in METRICS])
        & curves["task"].isin([task for task, *_ in TASKS])
    ].copy()
    if curves.empty:
        raise ValueError("No extended-budget convergence curves found.")

    curves["base"] = curves["base"].astype(int)
    curves["epoch"] = curves["epoch"].astype(int)
    curves["mean"] = curves["mean"].astype(float)
    curves["se"] = curves["se"].fillna(0.0).astype(float)
    curves["sd"] = curves["sd"].fillna(0.0).astype(float)
    curves["n"] = curves["n"].astype(int)
    curves["base_label"] = curves["base"].map(BASE_LABELS)
    curves["task_label"] = curves["task"].map({task: label for task, label, *_ in TASKS})
    curves["metric_label"] = curves["metric"].map(dict(METRICS))
    curves["source_file"] = str(CURVES_CSV.relative_to(ROOT))
    curves["aggregation"] = "training-seed mean +/- standard error from matched extended-budget runs"

    keys = ["task", "base", "metric", "epoch"]
    duplicate_keys = curves.duplicated(keys, keep=False)
    if duplicate_keys.any():
        curves = (
            curves.groupby(keys, as_index=False)
            .agg(
                budget=("budget", "first"),
                lr=("lr", "first"),
                mean=("mean", "mean"),
                sd=("sd", "mean"),
                se=("se", "mean"),
                n=("n", "max"),
                seeds=("seeds", lambda s: " ".join(sorted(set(map(str, s))))),
                base_label=("base_label", "first"),
                task_label=("task_label", "first"),
                metric_label=("metric_label", "first"),
                source_file=("source_file", "first"),
                aggregation=("aggregation", "first"),
            )
        )

    expected = {
        (task, base, metric): max_epoch
        for task, _, max_epoch, _ in TASKS
        for base in BASES
        for metric, _ in METRICS
    }
    missing = []
    for key, max_epoch in expected.items():
        task, base, metric = key
        sub = curves[(curves["task"].eq(task)) & (curves["base"].eq(base)) & (curves["metric"].eq(metric))]
        epochs = set(sub["epoch"].astype(int))
        expected_epochs = set(range(1, max_epoch + 1))
        if epochs != expected_epochs:
            missing.append((task, base, metric, sorted(expected_epochs - epochs), sorted(epochs - expected_epochs)))
    if missing:
        raise ValueError(f"Incomplete or unexpected epoch coverage: {missing[:3]}")

    return curves.sort_values(["task", "metric", "base", "epoch"]).reset_index(drop=True)


def save_source_data(curves: pd.DataFrame) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / "supp_s1_convergence.csv"
    columns = [
        "task",
        "task_label",
        "metric",
        "metric_label",
        "base",
        "base_label",
        "epoch",
        "mean",
        "se",
        "sd",
        "n",
        "seeds",
        "budget",
        "lr",
        "source_file",
        "aggregation",
    ]
    curves[columns].to_csv(out, index=False)
    return out


def draw_panel(ax, curves: pd.DataFrame, *, task: str, metric: str, max_epoch: int, original_budget: int) -> None:
    ax.axvline(original_budget, color="#6B7280", linestyle=(0, (3, 2)), linewidth=0.75, zorder=1)
    for base in BASES:
        sub = curves[
            curves["task"].eq(task) & curves["metric"].eq(metric) & curves["base"].eq(base)
        ].sort_values("epoch")
        color = WIDTH_COLORS[base]
        x = sub["epoch"].to_numpy(dtype=float)
        y = sub["mean"].to_numpy(dtype=float)
        se = sub["se"].to_numpy(dtype=float)
        ax.plot(x, y, color=color, linewidth=1.25, zorder=3)
        ax.fill_between(x, y - se, y + se, color=color, alpha=0.14, linewidth=0, zorder=2)

    ax.set_xlim(1, max_epoch)
    if max_epoch == 75:
        ax.set_xticks([1, 25, 50, 75])
    else:
        ax.set_xticks([1, 10, 20, 30])
    light_grid(ax, axis="y")
    ax.tick_params(axis="both", length=2.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def main() -> None:
    set_jne_style()
    curves = load_curves()
    data_path = save_source_data(curves)

    fig, axes = plt.subplots(2, 3, figsize=(WIDTH_FULL, 3.95), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.145, top=0.815, wspace=0.18, hspace=0.24)

    for col, (task, task_label, max_epoch, original_budget) in enumerate(TASKS):
        axes[0, col].set_title(task_label, pad=5)
        for row, (metric, metric_label) in enumerate(METRICS):
            ax = axes[row, col]
            draw_panel(ax, curves, task=task, metric=metric, max_epoch=max_epoch, original_budget=original_budget)
            if row == 0:
                ax.tick_params(axis="x", labelbottom=False)
            if col != 0:
                ax.tick_params(axis="y", labelleft=False)

    axes[0, 0].set_ylabel("Training loss")
    axes[1, 0].set_ylabel("Validation SDR (dB)")
    fig.supxlabel("Epoch", fontsize=8.5, y=0.035)

    handles = [
        Line2D([0], [0], color=WIDTH_COLORS[base], lw=1.35, label=BASE_LABELS[base])
        for base in BASES
    ]
    handles.append(
        Line2D([0], [0], color="#6B7280", lw=0.8, linestyle=(0, (3, 2)), label="Original budget")
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=4,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.2,
        borderpad=0.1,
    )

    pdf, png = save_figure(fig, OUT_DIR / "supp_s1_convergence_final", dpi=400)
    plt.close(fig)

    print(f"source file(s) used: {CURVES_CSV.relative_to(ROOT)}")
    print(f"number of observations plotted: {len(curves)} curve summary rows")
    print("observations represent: training-seed mean curves, not human subjects")
    print("aggregation performed before plotting: stored matched-seed means with +/-1 SE ribbons; no subject inference")
    print(f"source data saved: {data_path.relative_to(ROOT)}")
    print(f"final output paths: {pdf.relative_to(ROOT)}, {png.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
