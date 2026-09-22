#!/usr/bin/env python3
"""Regenerate Figure 2 capacity curves from stored experiment summaries."""

from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from jne_figure_style import COLORS, WIDTH_FULL, add_panel_label, light_grid, save_figure, set_jne_style


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT_DIR = ROOT / "Revised_Paper" / "Images"
DATA_DIR = ROOT / "Revised_Paper" / "figure_data"

MANUSCRIPT_SOURCES = [
    ROOT / "Revised_Paper" / "main.tex",
    ROOT / "Revised_Paper" / "supplementary_material.tex",
    ROOT / "new_experiments.md",
]
RESULT_SOURCES = [
    RUNS / "attention_width_sweep_20260520_212826_summary_index.csv",
    RUNS / "jne_revision_high_leverage_20260601_150050_width_saturation_extra_seeds_summary_index.csv",
    RUNS / "mixed1m_pareto_multiseed_20260521_141627_summary_index.csv",
    RUNS / "a7_bci_zero_shot_all_subjects_20260815_000000" / "a7_subject_width_rows.csv",
    RUNS / "a7_bci_zero_shot_all_subjects_20260815_000000" / "a7_subject_width_summary.csv",
]

BASES = [2, 4, 6, 8, 16]
PARAMS = {2: 1048, 4: 3218, 6: 6532, 8: 10990, 16: 40262}
BASE_LABELS = {base: f"b{base}" for base in BASES}
X_TICKS = [1e3, 3e3, 1e4, 4e4]
X_TICK_LABELS = ["1K", "3K", "10K", "40K"]

EOG_COLOR = "#4C78A8"
EMG_COLOR = "#B55A55"
ACCENT = "#2A9D8F"
GRAY_POINT = "#8A9099"
GRAY_LINE = "#B9BDC5"
TEXT = COLORS["text"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sample_mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sample_mean(values)
    return float(math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1)))


def parse_artifact_base(row: dict[str, str]) -> tuple[str, int] | None:
    run_dir = row.get("run_dir", "")
    match = re.search(r"_(eog|emg)_base(\d+)_seed\d+", run_dir)
    if not match:
        return None
    artifact, base = match.group(1).upper(), int(match.group(2))
    if base not in BASES:
        return None
    return artifact, base


def load_eegdenoisenet() -> list[dict[str, object]]:
    observations: list[dict[str, object]] = []
    for source in RESULT_SOURCES[:2]:
        for row in read_csv(source):
            parsed = parse_artifact_base(row)
            if parsed is None:
                continue
            artifact, base = parsed
            observations.append(
                {
                    "panel": "A",
                    "panel_title": "EEGDenoiseNet",
                    "row_type": "seed_observation",
                    "condition": artifact,
                    "base": base,
                    "width_label": BASE_LABELS[base],
                    "parameters": PARAMS[base],
                    "seed": int(row["seed"]),
                    "subject": "",
                    "cc": float(row["test_CC"]),
                    "mean_cc": "",
                    "sd_cc": "",
                    "ci95_low": "",
                    "ci95_high": "",
                    "source_file": str(source.relative_to(ROOT)),
                    "aggregation": "none; independent training seed",
                }
            )
    expected = {(artifact, base): 0 for artifact in ["EOG", "EMG"] for base in BASES}
    for row in observations:
        expected[(str(row["condition"]), int(row["base"]))] += 1
    missing = {key: count for key, count in expected.items() if count != 5}
    if missing:
        raise ValueError(f"Expected five EEGDenoiseNet seeds per artifact/base, got {missing}")
    return sorted(observations, key=lambda r: (str(r["condition"]), int(r["base"]), int(r["seed"])))


def load_mixed1m() -> list[dict[str, object]]:
    source = RESULT_SOURCES[2]
    observations: list[dict[str, object]] = []
    for row in read_csv(source):
        match = re.search(r"_mixed_base(\d+)_seed\d+", row.get("run_dir", ""))
        if not match:
            continue
        base = int(match.group(1))
        if base not in BASES:
            continue
        observations.append(
            {
                "panel": "B",
                "panel_title": "Mixed-1M",
                "row_type": "seed_observation",
                "condition": "Mixed-1M",
                "base": base,
                "width_label": BASE_LABELS[base],
                "parameters": PARAMS[base],
                "seed": int(row["seed"]),
                "subject": "",
                "cc": float(row["test_CC"]),
                "mean_cc": "",
                "sd_cc": "",
                "ci95_low": "",
                "ci95_high": "",
                "source_file": str(source.relative_to(ROOT)),
                "aggregation": "none; independent training seed",
            }
        )
    counts = defaultdict(int)
    for row in observations:
        counts[int(row["base"])] += 1
    missing = {base: counts[base] for base in BASES if counts[base] != 3}
    if missing:
        raise ValueError(f"Expected three Mixed-1M seeds per base, got {missing}")
    return sorted(observations, key=lambda r: (int(r["base"]), int(r["seed"])))


def load_bci_iv2a() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    row_source = RESULT_SOURCES[3]
    summary_source = RESULT_SOURCES[4]
    subject_rows: list[dict[str, object]] = []
    for row in read_csv(row_source):
        if row["dataset"] != "BCI_IV2a":
            continue
        base = int(row["base"])
        if base not in BASES:
            continue
        subject_rows.append(
            {
                "panel": "C",
                "panel_title": "BCI IV-2a zero-shot",
                "row_type": "subject_observation",
                "condition": "BCI_IV2a",
                "base": base,
                "width_label": BASE_LABELS[base],
                "parameters": PARAMS[base],
                "seed": "",
                "subject": row["subject"],
                "cc": float(row["CC"]),
                "mean_cc": "",
                "sd_cc": "",
                "ci95_low": "",
                "ci95_high": "",
                "source_file": str(row_source.relative_to(ROOT)),
                "aggregation": "subject-level mean after checkpoint/channel aggregation in A7",
            }
        )
    counts = defaultdict(set)
    for row in subject_rows:
        counts[int(row["base"])].add(str(row["subject"]))
    missing = {base: len(counts[base]) for base in BASES if len(counts[base]) != 9}
    if missing:
        raise ValueError(f"Expected nine IV-2a subjects per base, got {missing}")

    summary_rows: list[dict[str, object]] = []
    for row in read_csv(summary_source):
        if row["dataset"] != "BCI_IV2a":
            continue
        base = int(row["base"])
        if base not in BASES:
            continue
        summary_rows.append(
            {
                "panel": "C",
                "panel_title": "BCI IV-2a zero-shot",
                "row_type": "subject_summary",
                "condition": "BCI_IV2a",
                "base": base,
                "width_label": BASE_LABELS[base],
                "parameters": PARAMS[base],
                "seed": "",
                "subject": "",
                "cc": "",
                "mean_cc": float(row["CC_mean"]),
                "sd_cc": float(row["CC_sd"]),
                "ci95_low": float(row["CC_ci95_low"]),
                "ci95_high": float(row["CC_ci95_high"]),
                "source_file": str(summary_source.relative_to(ROOT)),
                "aggregation": "subject-bootstrap CI over nine subject-level means",
            }
        )
    if len(summary_rows) != len(BASES):
        raise ValueError(f"Expected five IV-2a summary rows, got {len(summary_rows)}")
    return (
        sorted(subject_rows, key=lambda r: (str(r["subject"]), int(r["base"]))),
        sorted(summary_rows, key=lambda r: int(r["base"])),
    )


def add_seed_summaries(observations: list[dict[str, object]], panel: str, panel_title: str) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    groups = sorted({str(row["condition"]) for row in observations})
    for condition in groups:
        for base in BASES:
            values = [float(row["cc"]) for row in observations if row["condition"] == condition and int(row["base"]) == base]
            if not values:
                continue
            summaries.append(
                {
                    "panel": panel,
                    "panel_title": panel_title,
                    "row_type": "seed_summary",
                    "condition": condition,
                    "base": base,
                    "width_label": BASE_LABELS[base],
                    "parameters": PARAMS[base],
                    "seed": "",
                    "subject": "",
                    "cc": "",
                    "mean_cc": sample_mean(values),
                    "sd_cc": sample_sd(values),
                    "ci95_low": "",
                    "ci95_high": "",
                    "source_file": "derived from plotted seed observations",
                    "aggregation": "mean and sample SD across independent training seeds",
                }
            )
    return summaries


def write_figure_data(rows: list[dict[str, object]]) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / "fig02_capacity.csv"
    fieldnames = [
        "panel",
        "panel_title",
        "row_type",
        "condition",
        "base",
        "width_label",
        "parameters",
        "seed",
        "subject",
        "cc",
        "mean_cc",
        "sd_cc",
        "ci95_low",
        "ci95_high",
        "source_file",
        "aggregation",
    ]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out


def set_capacity_xaxis(ax) -> None:
    ax.set_xscale("log")
    ax.set_xlim(PARAMS[2] / 1.35, PARAMS[16] * 1.35)
    ax.set_xticks(X_TICKS)
    ax.set_xticklabels(X_TICK_LABELS, rotation=0)
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def set_auto_ylim(ax, values: list[float], *, pad_frac: float = 0.10) -> None:
    low = min(values)
    high = max(values)
    span = max(high - low, 1e-4)
    ax.set_ylim(low - span * pad_frac, high + span * pad_frac)


def log_jitter(base: int, index: int, count: int, spread: float = 0.026) -> float:
    if count <= 1:
        return float(PARAMS[base])
    offsets = np.linspace(-spread, spread, count)
    return float(PARAMS[base] * math.exp(float(offsets[index])))


def draw_panel_a(ax, observations: list[dict[str, object]], summaries: list[dict[str, object]]) -> None:
    style = {
        "EOG": {"color": EOG_COLOR, "offset": -0.030},
        "EMG": {"color": EMG_COLOR, "offset": 0.030},
    }
    for condition in ["EOG", "EMG"]:
        color = style[condition]["color"]
        for base in BASES:
            vals = sorted(
                [row for row in observations if row["condition"] == condition and int(row["base"]) == base],
                key=lambda r: int(r["seed"]),
            )
            for idx, row in enumerate(vals):
                x = log_jitter(base, idx, len(vals), spread=0.018) * math.exp(style[condition]["offset"])
                ax.scatter(x, float(row["cc"]), s=11, color=color, alpha=0.45, linewidths=0, zorder=2)
        summary = [row for row in summaries if row["condition"] == condition]
        xs = [float(row["parameters"]) * math.exp(style[condition]["offset"]) for row in summary]
        ys = [float(row["mean_cc"]) for row in summary]
        yerr = [float(row["sd_cc"]) for row in summary]
        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            color=color,
            marker="o",
            markersize=3.5,
            linewidth=1.25,
            elinewidth=0.85,
            capsize=2.2,
            label=condition,
            zorder=4,
        )
    all_values = [float(row["cc"]) for row in observations]
    all_values += [float(row["mean_cc"]) + float(row["sd_cc"]) for row in summaries]
    all_values += [float(row["mean_cc"]) - float(row["sd_cc"]) for row in summaries]
    set_auto_ylim(ax, all_values)
    set_capacity_xaxis(ax)
    ax.set_ylabel("CC")
    ax.set_title("EEGDenoiseNet", pad=5)
    light_grid(ax, axis="y")


def draw_panel_b(ax, observations: list[dict[str, object]], summaries: list[dict[str, object]]) -> None:
    for base in BASES:
        vals = sorted([row for row in observations if int(row["base"]) == base], key=lambda r: int(r["seed"]))
        for idx, row in enumerate(vals):
            ax.scatter(log_jitter(base, idx, len(vals), spread=0.018), float(row["cc"]), s=11, color=GRAY_POINT, alpha=0.48, linewidths=0, zorder=2)
    xs = [float(row["parameters"]) for row in summaries]
    ys = [float(row["mean_cc"]) for row in summaries]
    yerr = [float(row["sd_cc"]) for row in summaries]
    ax.errorbar(
        xs,
        ys,
        yerr=yerr,
        color=ACCENT,
        marker="o",
        markersize=3.7,
        linewidth=1.35,
        elinewidth=0.9,
        capsize=2.3,
        zorder=4,
    )
    all_values = [float(row["cc"]) for row in observations]
    all_values += [float(row["mean_cc"]) + float(row["sd_cc"]) for row in summaries]
    all_values += [float(row["mean_cc"]) - float(row["sd_cc"]) for row in summaries]
    set_auto_ylim(ax, all_values)
    set_capacity_xaxis(ax)
    ax.set_ylabel("CC")
    ax.set_title("Mixed-1M", pad=5)
    light_grid(ax, axis="y")


def draw_panel_c(ax, observations: list[dict[str, object]], summaries: list[dict[str, object]]) -> None:
    by_subject: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in observations:
        by_subject[str(row["subject"])].append(row)
    for subject in sorted(by_subject):
        rows = sorted(by_subject[subject], key=lambda r: int(r["base"]))
        xs = [float(row["parameters"]) for row in rows]
        ys = [float(row["cc"]) for row in rows]
        ax.plot(xs, ys, color=GRAY_LINE, linewidth=0.55, alpha=0.62, zorder=1)
        ax.scatter(xs, ys, s=9, color=GRAY_POINT, alpha=0.58, linewidths=0, zorder=2)

    summaries = sorted(summaries, key=lambda r: int(r["base"]))
    xs = np.asarray([float(row["parameters"]) for row in summaries], dtype=float)
    ys = np.asarray([float(row["mean_cc"]) for row in summaries], dtype=float)
    lows = np.asarray([float(row["ci95_low"]) for row in summaries], dtype=float)
    highs = np.asarray([float(row["ci95_high"]) for row in summaries], dtype=float)
    ax.errorbar(
        xs,
        ys,
        yerr=np.vstack([ys - lows, highs - ys]),
        color=ACCENT,
        marker="o",
        markersize=3.8,
        linewidth=1.35,
        elinewidth=0.9,
        capsize=2.4,
        zorder=4,
    )
    all_values = [float(row["cc"]) for row in observations] + list(lows) + list(highs)
    set_auto_ylim(ax, all_values, pad_frac=0.08)
    set_capacity_xaxis(ax)
    ax.set_ylabel("CC")
    ax.set_title("BCI IV-2a zero-shot", pad=5)
    light_grid(ax, axis="y")


def ensure_panel_c_not_clipped(ax, observations: list[dict[str, object]], summaries: list[dict[str, object]]) -> bool:
    values = [float(row["cc"]) for row in observations]
    values += [float(row["ci95_low"]) for row in summaries]
    values += [float(row["ci95_high"]) for row in summaries]
    data_low = min(values)
    data_high = max(values)
    axis_low, axis_high = ax.get_ylim()
    if data_low >= axis_low and data_high <= axis_high:
        return False
    span = max(data_high - data_low, 1e-4)
    pad = span * 0.06
    ax.set_ylim(min(axis_low, data_low - pad), max(axis_high, data_high + pad))
    return True


def main() -> None:
    set_jne_style()
    eeg = load_eegdenoisenet()
    mixed = load_mixed1m()
    bci_subjects, bci_summary = load_bci_iv2a()
    eeg_summary = add_seed_summaries(eeg, "A", "EEGDenoiseNet")
    mixed_summary = add_seed_summaries(mixed, "B", "Mixed-1M")

    plotted_rows = eeg + eeg_summary + mixed + mixed_summary + bci_subjects + bci_summary
    data_path = write_figure_data(plotted_rows)

    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_FULL, 3.10), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.86, bottom=0.24, wspace=0.32)

    draw_panel_a(axes[0], eeg, eeg_summary)
    draw_panel_b(axes[1], mixed, mixed_summary)
    draw_panel_c(axes[2], bci_subjects, bci_summary)
    panel_c_ylim_expanded = ensure_panel_c_not_clipped(axes[2], bci_subjects, bci_summary)

    axes[0].legend(loc="lower right", ncol=1, frameon=False, handlelength=1.5, borderpad=0.2)

    for label, ax in zip(["A", "B", "C"], axes, strict=True):
        add_panel_label(ax, label, x=-0.16, y=1.04)

    fig.text(
        0.5,
        0.07,
        "Trainable parameters (log scale)",
        ha="center",
        va="center",
        fontsize=8.5,
        color=TEXT,
    )

    pdf_path, png_path = save_figure(fig, OUT_DIR / "fig02_capacity_subject_points", dpi=400)
    plt.close(fig)

    raw_seed_count = sum(row["row_type"] == "seed_observation" for row in plotted_rows)
    subject_count = sum(row["row_type"] == "subject_observation" for row in plotted_rows)
    summary_count = len(plotted_rows) - raw_seed_count - subject_count

    print("source file(s) used:")
    for source in MANUSCRIPT_SOURCES + RESULT_SOURCES:
        print(f"- {source}")
    print(f"number of observations plotted: {len(plotted_rows)} CSV rows ({raw_seed_count} training-seed observations, {subject_count} subject observations, {summary_count} summary elements)")
    print("observation type: training seeds in panels A/B; human subjects in panel C; summary rows are mean/SD or mean/subject-bootstrap CI overlays")
    print("aggregation performed before plotting: A/B summarize independent training seeds by width; C uses A7 subject-level means after checkpoint/channel aggregation, then subject-bootstrap CI")
    print(f"panel C clipping check: {'expanded y-axis modestly' if panel_c_ylim_expanded else 'no clipping detected'}")
    print(f"source data: {data_path}")
    print(f"final PDF: {pdf_path}")
    print(f"final PNG: {png_path}")


if __name__ == "__main__":
    main()
