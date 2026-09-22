#!/usr/bin/env python3
"""A6 optimization/convergence audit for width-capacity experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PRIMARY_BASES = [2, 4, 6, 8, 16]
REPRESENTATIVE_BASES = [4, 6, 16]
METRIC_SPECS = [
    ("train_loss", "Training loss"),
    ("val_SDR", "Validation SDR"),
    ("val_CC", "Validation CC"),
]
COLORS = {
    2: "#4b5563",
    4: "#2563eb",
    6: "#059669",
    8: "#d97706",
    16: "#dc2626",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--current-eeg-prefix", default="attention_width_sweep_20260520_212826")
    parser.add_argument("--current-mixed-prefix", default="mixed1m_pareto_multiseed_20260521_141627")
    parser.add_argument("--extended-prefix", default="a6_optimization_convergence")
    parser.add_argument("--primary-bases", nargs="+", type=int, default=PRIMARY_BASES)
    parser.add_argument("--representative-bases", nargs="+", type=int, default=REPRESENTATIVE_BASES)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_history(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    preferred = [
        "run_id",
        "budget",
        "task",
        "base",
        "seed",
        "lr",
        "epoch",
        "metric",
        "mean",
        "sd",
        "se",
        "n",
        "run_dir",
    ]
    for key in preferred:
        if any(key in row for row in rows):
            fieldnames.append(key)
            seen.add(key)
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def sem(values: list[float]) -> float:
    return sd(values) / math.sqrt(len(values)) if values else float("nan")


def parse_base_seed(run_dir: Path) -> tuple[int | None, int | None]:
    text = run_dir.name
    base = re.search(r"base(\d+)", text)
    seed = re.search(r"seed(\d+)", text)
    return (int(base.group(1)) if base else None, int(seed.group(1)) if seed else None)


def lr_label(lr: float) -> str:
    return f"{lr:.6g}"


def task_from_config(config: dict[str, Any], run_dir: Path) -> str:
    text = (str(config.get("data", "")) + " " + run_dir.name).lower()
    if "mixed" in text:
        return "mixed1m"
    if "emg" in text:
        return "emg"
    if "eog" in text:
        return "eog"
    return "unknown"


def discover_current(args: argparse.Namespace) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for task in ["eog", "emg"]:
        for base in args.primary_bases:
            out.extend(("current", path) for path in sorted(args.runs_dir.glob(f"{args.current_eeg_prefix}_{task}_base{base}_seed*")))
    for base in args.primary_bases:
        out.extend(("current", path) for path in sorted(args.runs_dir.glob(f"{args.current_mixed_prefix}_mixed_base{base}_seed*")))
    return out


def discover_extended(args: argparse.Namespace) -> list[tuple[str, Path]]:
    return [("extended", path) for path in sorted(args.runs_dir.glob(f"{args.extended_prefix}_*"))]


def run_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    epoch_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    seen_dirs: set[Path] = set()
    for budget, run_dir in discover_current(args) + discover_extended(args):
        run_dir = run_dir.resolve()
        if run_dir in seen_dirs:
            continue
        seen_dirs.add(run_dir)
        history_path = run_dir / "history.jsonl"
        config_path = run_dir / "config.resolved.json"
        summary_path = run_dir / "summary.json"
        if not history_path.exists() or not config_path.exists():
            continue
        config = read_json(config_path)
        summary = read_json(summary_path) if summary_path.exists() else {}
        history = read_history(history_path)
        if not history:
            continue
        parsed_base, parsed_seed = parse_base_seed(run_dir)
        base = int(config.get("base", parsed_base))
        seed = int(config.get("seed", parsed_seed))
        lr = float(config.get("lr", history[0].get("lr", float("nan"))))
        task = task_from_config(config, run_dir)
        epochs = len(history)
        best_epoch = int(summary.get("best_epoch", history[-1].get("best_epoch", 0)))
        best_val_sdr = float(summary.get("best_val_sdr", history[-1].get("best_val_sdr", float("nan"))))
        best_row = history[max(0, best_epoch - 1)] if best_epoch else history[-1]
        tail_start = max(1, math.floor(0.8 * epochs))
        tail = [row for row in history if int(row["epoch"]) >= tail_start]
        tail_sdr = [float(row["val"]["SDR"]) for row in tail]
        tail_cc = [float(row["val"]["CC"]) for row in tail]
        tail_loss = [float(row["train"]["loss"]) for row in tail]
        tail_sdr_mean = mean(tail_sdr)
        tail_sdr_sd = sd(tail_sdr)
        selected_near_tail = abs(best_val_sdr - tail_sdr_mean) <= max(0.10, tail_sdr_sd)
        selected_in_tail = best_epoch >= tail_start

        test = summary.get("test", {})
        summary_rows.append(
            {
                "run_id": args.run_id,
                "budget": budget,
                "task": task,
                "base": base,
                "seed": seed,
                "lr": lr_label(lr),
                "epochs": epochs,
                "best_epoch": best_epoch,
                "best_epoch_fraction": best_epoch / epochs if epochs else float("nan"),
                "best_val_sdr": best_val_sdr,
                "best_val_cc": float(best_row["val"]["CC"]),
                "final_val_sdr": float(history[-1]["val"]["SDR"]),
                "final_val_cc": float(history[-1]["val"]["CC"]),
                "tail_start_epoch": tail_start,
                "tail_val_sdr_mean": tail_sdr_mean,
                "tail_val_sdr_sd": tail_sdr_sd,
                "best_minus_tail_val_sdr": best_val_sdr - tail_sdr_mean,
                "selected_in_final_20pct": selected_in_tail,
                "selected_near_tail_stable_sdr": selected_near_tail,
                "tail_train_loss_mean": mean(tail_loss),
                "tail_val_cc_mean": mean(tail_cc),
                "test_CC": test.get("CC", ""),
                "test_SDR": test.get("SDR", ""),
                "test_T_RRMSE": test.get("T_RRMSE", ""),
                "test_S_RRMSE": test.get("S_RRMSE", ""),
                "trainable_parameters": summary.get("trainable_parameters", ""),
                "run_dir": str(run_dir),
            }
        )
        for row in history:
            epoch_rows.append(
                {
                    "run_id": args.run_id,
                    "budget": budget,
                    "task": task,
                    "base": base,
                    "seed": seed,
                    "lr": lr_label(lr),
                    "epoch": int(row["epoch"]),
                    "scheduler_lr": float(row["lr"]),
                    "train_loss": float(row["train"]["loss"]),
                    "train_CC": float(row["train"]["CC"]),
                    "train_SDR": float(row["train"]["SDR"]),
                    "val_loss": float(row["val"]["loss"]),
                    "val_CC": float(row["val"]["CC"]),
                    "val_SDR": float(row["val"]["SDR"]),
                    "best_epoch_so_far": int(row["best_epoch"]),
                    "best_val_sdr_so_far": float(row["best_val_sdr"]),
                    "run_dir": str(run_dir),
                }
            )
    return epoch_rows, summary_rows


def curve_rows(epoch_rows: list[dict[str, Any]], run_id: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in epoch_rows:
        grouped[(row["budget"], row["task"], int(row["base"]), str(row["lr"]), int(row["epoch"]))].append(row)
    out: list[dict[str, Any]] = []
    for (budget, task, base, lr, epoch), rows in sorted(grouped.items()):
        for metric, _label in METRIC_SPECS:
            values = [float(row[metric]) for row in rows]
            out.append(
                {
                    "run_id": run_id,
                    "budget": budget,
                    "task": task,
                    "base": base,
                    "lr": lr,
                    "epoch": epoch,
                    "metric": metric,
                    "mean": mean(values),
                    "sd": sd(values),
                    "se": sem(values),
                    "n": len(values),
                    "seeds": " ".join(str(row["seed"]) for row in sorted(rows, key=lambda item: int(item["seed"]))),
                }
            )
    return out


def comparison_rows(summary_rows: list[dict[str, Any]], run_id: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        grouped[(row["budget"], row["task"], int(row["base"]), str(row["lr"]))].append(row)
    out: list[dict[str, Any]] = []
    for budget in sorted({row["budget"] for row in summary_rows}):
        for task in sorted({row["task"] for row in summary_rows}):
            lr_values = sorted({str(row["lr"]) for row in summary_rows if row["budget"] == budget and row["task"] == task})
            for lr in lr_values:
                for low_base, high_base in [(4, 16), (6, 16)]:
                    low = grouped.get((budget, task, low_base, lr), [])
                    high = grouped.get((budget, task, high_base, lr), [])
                    low_seeds = {int(row["seed"]) for row in low}
                    high_seeds = {int(row["seed"]) for row in high}
                    seeds = sorted(low_seeds & high_seeds)
                    if not seeds:
                        continue
                    low_by_seed = {int(row["seed"]): row for row in low}
                    high_by_seed = {int(row["seed"]): row for row in high}
                    for metric in ["best_val_sdr", "best_val_cc", "test_SDR", "test_CC"]:
                        diffs = []
                        for seed in seeds:
                            low_value = low_by_seed[seed].get(metric, "")
                            high_value = high_by_seed[seed].get(metric, "")
                            if low_value == "" or high_value == "":
                                continue
                            diffs.append(float(high_value) - float(low_value))
                        if not diffs:
                            continue
                        out.append(
                            {
                                "run_id": run_id,
                                "budget": budget,
                                "task": task,
                                "comparison": f"base{low_base}_to_base{high_base}",
                                "lr": lr,
                                "metric": metric,
                                "n_matched_seeds": len(diffs),
                                "matched_seeds": " ".join(str(seed) for seed in seeds),
                                "mean_high_minus_low": mean(diffs),
                                "sd_high_minus_low": sd(diffs),
                                "se_high_minus_low": sem(diffs),
                            }
                        )
    return out


def budget_lr_sensitivity_rows(summary_rows: list[dict[str, Any]], run_id: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    current = [row for row in summary_rows if row["budget"] == "current" and str(row["lr"]) == "0.001"]
    extended = [row for row in summary_rows if row["budget"] == "extended"]
    current_index = {(row["task"], int(row["base"]), int(row["seed"])): row for row in current}
    extended_current_lr_index = {
        (row["task"], int(row["base"]), int(row["seed"])): row
        for row in extended
        if str(row["lr"]) == "0.001"
    }
    extended_lower_lr_index = {
        (row["task"], int(row["base"]), int(row["seed"])): row
        for row in extended
        if str(row["lr"]) == "0.0003"
    }
    metrics = ["best_val_sdr", "test_SDR", "best_val_cc", "test_CC"]

    for task in sorted({row["task"] for row in summary_rows}):
        for base in REPRESENTATIVE_BASES:
            seeds = sorted(
                {seed for task_i, base_i, seed in current_index if task_i == task and base_i == base}
                & {seed for task_i, base_i, seed in extended_current_lr_index if task_i == task and base_i == base}
            )
            for metric in metrics:
                diffs = [
                    float(extended_current_lr_index[(task, base, seed)][metric])
                    - float(current_index[(task, base, seed)][metric])
                    for seed in seeds
                    if current_index[(task, base, seed)].get(metric, "") != ""
                    and extended_current_lr_index[(task, base, seed)].get(metric, "") != ""
                ]
                if diffs:
                    out.append(
                        {
                            "run_id": run_id,
                            "analysis": "extended_minus_current_budget",
                            "task": task,
                            "base": base,
                            "lr": "0.001",
                            "metric": metric,
                            "n_matched_seeds": len(diffs),
                            "matched_seeds": " ".join(str(seed) for seed in seeds),
                            "mean_delta": mean(diffs),
                            "sd_delta": sd(diffs),
                            "se_delta": sem(diffs),
                            "seed_deltas": " ".join(f"{value:+.6f}" for value in diffs),
                        }
                    )

        seeds = sorted(
            {seed for task_i, base_i, seed in extended_current_lr_index if task_i == task and base_i == 16}
            & {seed for task_i, base_i, seed in extended_lower_lr_index if task_i == task and base_i == 16}
        )
        for metric in metrics:
            diffs = [
                float(extended_lower_lr_index[(task, 16, seed)][metric])
                - float(extended_current_lr_index[(task, 16, seed)][metric])
                for seed in seeds
                if extended_current_lr_index[(task, 16, seed)].get(metric, "") != ""
                and extended_lower_lr_index[(task, 16, seed)].get(metric, "") != ""
            ]
            if diffs:
                out.append(
                    {
                        "run_id": run_id,
                        "analysis": "base16_lower_lr_minus_current_lr",
                        "task": task,
                        "base": 16,
                        "lr": "0.0003_minus_0.001",
                        "metric": metric,
                        "n_matched_seeds": len(diffs),
                        "matched_seeds": " ".join(str(seed) for seed in seeds),
                        "mean_delta": mean(diffs),
                        "sd_delta": sd(diffs),
                        "se_delta": sem(diffs),
                        "seed_deltas": " ".join(f"{value:+.6f}" for value in diffs),
                    }
                )
    return out


def plot_curves(curves: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    plot_dir = output_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    budgets = sorted({row["budget"] for row in curves})
    for budget in budgets:
        tasks = sorted({row["task"] for row in curves if row["budget"] == budget})
        for task in tasks:
            for metric, label in METRIC_SPECS:
                rows = [row for row in curves if row["budget"] == budget and row["task"] == task and row["metric"] == metric]
                if not rows:
                    continue
                fig, ax = plt.subplots(figsize=(7.0, 4.3), dpi=160)
                keys = sorted({(int(row["base"]), str(row["lr"])) for row in rows})
                for base, lr in keys:
                    items = sorted(
                        [row for row in rows if int(row["base"]) == base and str(row["lr"]) == lr],
                        key=lambda item: int(item["epoch"]),
                    )
                    xs = [int(row["epoch"]) for row in items]
                    ys = [float(row["mean"]) for row in items]
                    ses = [float(row["se"]) for row in items]
                    color = COLORS.get(base, "#111827")
                    label_text = f"base{base}, lr={lr}" if len({str(row["lr"]) for row in rows}) > 1 else f"base{base}"
                    ax.plot(xs, ys, color=color, linewidth=1.8, label=label_text)
                    ax.fill_between(
                        xs,
                        [y - se for y, se in zip(ys, ses)],
                        [y + se for y, se in zip(ys, ses)],
                        color=color,
                        alpha=0.16,
                        linewidth=0,
                    )
                ax.set_title(f"A6 {budget} {task}: {label}")
                ax.set_xlabel("Epoch")
                ax.set_ylabel(label)
                ax.grid(True, color="#e5e7eb", linewidth=0.8)
                ax.legend(frameon=False, fontsize=8)
                fig.tight_layout()
                path = plot_dir / f"a6_{budget}_{task}_{metric}.png"
                fig.savefig(path)
                plt.close(fig)
                paths.append(path)
    return paths


def compact_summary(comparisons: list[dict[str, Any]]) -> list[str]:
    lines = []
    for row in comparisons:
        if row["metric"] != "best_val_sdr":
            continue
        lines.append(
            f"- {row['budget']} {row['task']} {row['comparison']} lr={row['lr']}: "
            f"mean high-low validation SDR `{float(row['mean_high_minus_low']):+.6f}` "
            f"(n={row['n_matched_seeds']})."
        )
    return lines


def sensitivity_summary(rows: list[dict[str, Any]]) -> list[str]:
    lines = []
    for row in rows:
        if row["metric"] != "test_SDR":
            continue
        if row["analysis"] == "extended_minus_current_budget":
            lines.append(
                f"- {row['task']} base{row['base']} extended-current test SDR: "
                f"`{float(row['mean_delta']):+.6f}` (n={row['n_matched_seeds']})."
            )
        elif row["analysis"] == "base16_lower_lr_minus_current_lr":
            lines.append(
                f"- {row['task']} base16 lower-LR minus current-LR test SDR: "
                f"`{float(row['mean_delta']):+.6f}` (n={row['n_matched_seeds']})."
            )
    return lines


def write_markdown(
    path: Path,
    *,
    run_id: str,
    epoch_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    sensitivity: list[dict[str, Any]],
    plot_paths: list[Path],
) -> None:
    lines = ["# A6 Optimization/Convergence Audit", ""]
    lines.append(f"Run id: `{run_id}`")
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append(f"- Epoch rows: `{len(epoch_rows)}`")
    lines.append(f"- Run summaries: `{len(summary_rows)}`")
    lines.append(f"- Capacity comparison rows: `{len(comparisons)}`")
    lines.append(f"- Budget/LR sensitivity rows: `{len(sensitivity)}`")
    coverage = defaultdict(set)
    for row in summary_rows:
        coverage[(row["budget"], row["task"])].add((row["base"], row["lr"], row["seed"], row["epochs"]))
    for key, values in sorted(coverage.items()):
        budgets = sorted(values)
        bases = sorted({value[0] for value in budgets})
        lrs = sorted({value[1] for value in budgets})
        seeds = sorted({value[2] for value in budgets})
        epochs = sorted({value[3] for value in budgets})
        lines.append(f"- {key[0]} {key[1]}: bases `{bases}`, lrs `{lrs}`, seeds `{seeds}`, epochs `{epochs}`")
    lines.append("")
    lines.append("## Selection Stability")
    lines.append("")
    stable = [row for row in summary_rows if row["selected_near_tail_stable_sdr"]]
    in_tail = [row for row in summary_rows if row["selected_in_final_20pct"]]
    lines.append(f"- Runs whose selected checkpoint is near the final validation-SDR tail: `{len(stable)}/{len(summary_rows)}`")
    lines.append(f"- Runs whose best epoch falls in the final 20 percent of training: `{len(in_tail)}/{len(summary_rows)}`")
    lines.append("")
    lines.append("## Compact-Vs-Large Validation SDR")
    lines.append("")
    lines.extend(compact_summary(comparisons) or ["- No matched compact-vs-large comparisons available yet."])
    lines.append("")
    lines.append("## Extended Budget And Learning-Rate Sensitivity")
    lines.append("")
    lines.extend(sensitivity_summary(sensitivity) or ["- No matched budget/LR sensitivity rows available yet."])
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    for path_item in plot_paths:
        lines.append(f"- `{path_item}`")
    lines.append("")
    lines.append("## Interpretation Guardrails")
    lines.append("")
    lines.append("- Use seed-mean curves with uncertainty ribbons for the paper, not every raw trajectory.")
    lines.append("- Treat extended-budget rows as the decisive optimization sensitivity check once all launched controls finish.")
    lines.append("- Interpret capacity behavior jointly with the extended-training and lower-learning-rate controls.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    epoch_rows, summary_rows = run_rows(args)
    curves = curve_rows(epoch_rows, args.run_id)
    comparisons = comparison_rows(summary_rows, args.run_id)
    sensitivity = budget_lr_sensitivity_rows(summary_rows, args.run_id)
    plot_paths = plot_curves(curves, args.output_dir)
    write_csv(args.output_dir / "a6_epoch_history_rows.csv", epoch_rows)
    write_csv(args.output_dir / "a6_seed_mean_curves.csv", curves)
    write_csv(args.output_dir / "a6_run_selection_stability.csv", summary_rows)
    write_csv(args.output_dir / "a6_capacity_optimization_comparisons.csv", comparisons)
    write_csv(args.output_dir / "a6_budget_and_lr_sensitivity_deltas.csv", sensitivity)
    summary = {
        "run_id": args.run_id,
        "n_epoch_rows": len(epoch_rows),
        "n_run_summaries": len(summary_rows),
        "n_curve_rows": len(curves),
        "n_comparison_rows": len(comparisons),
        "n_budget_lr_sensitivity_rows": len(sensitivity),
        "figures": [str(path) for path in plot_paths],
    }
    (args.output_dir / "a6_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(
        args.output_dir / "a6_summary.md",
        run_id=args.run_id,
        epoch_rows=epoch_rows,
        summary_rows=summary_rows,
        comparisons=comparisons,
        sensitivity=sensitivity,
        plot_paths=plot_paths,
    )
    print(f"[done] epoch_rows={len(epoch_rows)} run_summaries={len(summary_rows)} figures={len(plot_paths)}", flush=True)


if __name__ == "__main__":
    main()
