#!/usr/bin/env python3
"""capacity formal capacity/diminishing-return analysis for reconstruction metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from scipy import stats


WIDTHS = [2, 4, 6, 8, 16]
ADJACENT = [(2, 4), (4, 6), (6, 8), (8, 16)]
COMPACT_VS_LARGE = [(4, 16), (6, 16)]
METRICS = ["CC", "T_RRMSE", "S_RRMSE", "SDR"]
HIGHER_BETTER = {"CC", "SDR"}
MARGINS = {
    "CC": 0.005,
    "T_RRMSE": 0.005,
    "S_RRMSE": 0.005,
    "SDR": 0.10,
}
MARGIN_JUSTIFICATION = {
    "CC": "Absolute correlation changes below 0.005 are treated as practically negligible because they are below one-half percentage point on a bounded [-1,1] agreement metric and far smaller than the observed noisy-to-denoised gains.",
    "T_RRMSE": "Absolute RRMSE changes below 0.005 are treated as practically negligible because they correspond to less than 0.5 percentage points of normalized temporal error.",
    "S_RRMSE": "Absolute RRMSE changes below 0.005 are treated as practically negligible because they correspond to less than 0.5 percentage points of normalized spectral error.",
    "SDR": "SDR changes below 0.10 dB are treated as practically negligible because they are below the resolution normally interpreted as a meaningful denoising gain and far smaller than the multi-dB noisy-to-denoised improvements.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width-index", type=Path, required=True)
    parser.add_argument("--bci-zero-shot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--elbow-threshold", type=float, default=0.95)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    preferred = [
        "run_id",
        "analysis",
        "dataset",
        "artifact",
        "metric",
        "comparison",
        "base_low",
        "base_high",
        "n_matched_seeds",
        "matched_seeds",
        "mean_incremental_improvement",
        "ci95_low",
        "ci95_high",
        "equivalence_margin",
        "formal_status",
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
    return float(statistics.mean(values)) if values else float("nan")


def sample_sd(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def ci_t(values: list[float], confidence: float = 0.95) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], values[0]
    mu = mean(values)
    se = sample_sd(values) / math.sqrt(len(values))
    q = float(stats.t.ppf(0.5 + confidence / 2.0, df=len(values) - 1))
    return mu - q * se, mu + q * se


def paired_t_p(values: list[float]) -> float:
    if len(values) < 2 or sample_sd(values) == 0:
        return 1.0 if abs(mean(values)) < 1e-12 else 0.0
    result = stats.ttest_1samp(values, popmean=0.0)
    return float(result.pvalue)


def tost_p(values: list[float], margin: float) -> tuple[float, float, float]:
    """TOST p-value for equivalence of improvement to zero within +/- margin."""
    if len(values) < 2 or sample_sd(values) == 0:
        mu = mean(values)
        p_lower = 0.0 if mu > -margin else 1.0
        p_upper = 0.0 if mu < margin else 1.0
        return p_lower, p_upper, max(p_lower, p_upper)
    mu = mean(values)
    se = sample_sd(values) / math.sqrt(len(values))
    df = len(values) - 1
    t_lower = (mu - (-margin)) / se
    t_upper = (mu - margin) / se
    p_lower = float(stats.t.sf(t_lower, df=df))
    p_upper = float(stats.t.cdf(t_upper, df=df))
    return p_lower, p_upper, max(p_lower, p_upper)


def base_from_text(text: str) -> int | None:
    match = re.search(r"base(\d+)", text)
    return int(match.group(1)) if match else None


def normalize_width_rows(rows: list[dict[str, str]], run_id: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        data = row["data"].lower()
        if "eog contaminated" in data:
            artifact = "eog"
        elif "emg contaminated" in data:
            artifact = "emg"
        else:
            continue
        base = base_from_text(row.get("run_dir", "") + " " + row.get("output_dir", ""))
        if base not in WIDTHS:
            continue
        for metric in METRICS:
            out.append(
                {
                    "run_id": run_id,
                    "analysis": "synthetic_width_benchmark",
                    "dataset": "synthetic_eegdenoisenet",
                    "artifact": artifact,
                    "base": int(base),
                    "seed": int(row["seed"]),
                    "trainable_parameters": int(float(row["trainable_parameters"])),
                    "metric": metric,
                    "metric_value": float(row[f"test_{metric}"]),
                    "oriented_value": orient_metric(metric, float(row[f"test_{metric}"])),
                }
            )
    return out


def normalize_bci_rows(rows: list[dict[str, str]], run_id: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if row.get("kind") != "model":
            continue
        base = int(row["base"])
        if base not in WIDTHS:
            continue
        for metric in METRICS:
            out.append(
                {
                    "run_id": run_id,
                    "analysis": "bci_zero_shot_transfer",
                    "dataset": row["dataset"],
                    "artifact": "bci_eog_transfer",
                    "base": base,
                    "seed": int(row["train_seed"]),
                    "trainable_parameters": int(float(row["trainable_parameters"])),
                    "metric": metric,
                    "metric_value": float(row[metric]),
                    "oriented_value": orient_metric(metric, float(row[metric])),
                }
            )
    return out


def orient_metric(metric: str, value: float) -> float:
    return value if metric in HIGHER_BETTER else -value


def raw_increment(metric: str, low_value: float, high_value: float) -> float:
    return high_value - low_value if metric in HIGHER_BETTER else low_value - high_value


def index_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, int, int], dict[str, Any]]:
    return {
        (str(row["dataset"]), str(row["artifact"]), str(row["metric"]), int(row["base"]), int(row["seed"])): row
        for row in rows
    }


def comparison_rows(
    rows: list[dict[str, Any]],
    *,
    run_id: str,
    comparisons: list[tuple[int, int]],
    analysis: str,
    alpha: float,
) -> list[dict[str, Any]]:
    indexed = index_rows(rows)
    datasets = sorted({str(row["dataset"]) for row in rows})
    artifacts = sorted({str(row["artifact"]) for row in rows})
    metrics = sorted({str(row["metric"]) for row in rows}, key=METRICS.index)
    out = []
    for dataset in datasets:
        for artifact in artifacts:
            if not any(row["dataset"] == dataset and row["artifact"] == artifact for row in rows):
                continue
            for metric in metrics:
                for base_low, base_high in comparisons:
                    seeds_low = {int(row["seed"]) for row in rows if row["dataset"] == dataset and row["artifact"] == artifact and row["metric"] == metric and int(row["base"]) == base_low}
                    seeds_high = {int(row["seed"]) for row in rows if row["dataset"] == dataset and row["artifact"] == artifact and row["metric"] == metric and int(row["base"]) == base_high}
                    seeds = sorted(seeds_low & seeds_high)
                    if not seeds:
                        continue
                    increments = []
                    low_values = []
                    high_values = []
                    for seed in seeds:
                        low = indexed[(dataset, artifact, metric, base_low, seed)]
                        high = indexed[(dataset, artifact, metric, base_high, seed)]
                        low_values.append(float(low["metric_value"]))
                        high_values.append(float(high["metric_value"]))
                        increments.append(raw_increment(metric, float(low["metric_value"]), float(high["metric_value"])))
                    ci_low, ci_high = ci_t(increments)
                    p_equiv_lower, p_equiv_upper, p_equiv = tost_p(increments, MARGINS[metric])
                    status = formal_status(increments, ci_low, ci_high, MARGINS[metric], alpha)
                    out.append(
                        {
                            "run_id": run_id,
                            "analysis": analysis,
                            "dataset": dataset,
                            "artifact": artifact,
                            "metric": metric,
                            "metric_direction": "higher_better" if metric in HIGHER_BETTER else "lower_better",
                            "comparison": f"base{base_low}_to_base{base_high}",
                            "base_low": base_low,
                            "base_high": base_high,
                            "n_matched_seeds": len(seeds),
                            "matched_seeds": " ".join(str(seed) for seed in seeds),
                            "base_low_metric_mean": mean(low_values),
                            "base_high_metric_mean": mean(high_values),
                            "mean_incremental_improvement": mean(increments),
                            "sd_incremental_improvement": sample_sd(increments),
                            "ci95_low": ci_low,
                            "ci95_high": ci_high,
                            "paired_t_p_increment_ne_0": paired_t_p(increments),
                            "equivalence_margin": MARGINS[metric],
                            "equivalence_margin_justification": MARGIN_JUSTIFICATION[metric],
                            "tost_p_lower_bound_gt_minus_margin": p_equiv_lower,
                            "tost_p_upper_bound_lt_margin": p_equiv_upper,
                            "tost_p_equivalence": p_equiv,
                            "formal_status": status,
                            "status_note": status_note(status),
                        }
                    )
    return out


def formal_status(values: list[float], ci_low: float, ci_high: float, margin: float, alpha: float) -> str:
    mu = mean(values)
    # The equivalence band is symmetric: a negative signed interval inside it is
    # practically equivalent, rather than evidence of a transfer inversion.
    if ci_low >= -margin and ci_high <= margin:
        return "formal_saturation"
    if ci_high < 0:
        return "larger_width_worse_transfer_inversion"
    if ci_high <= margin:
        return "noninferior_practically_negligible_gain"
    if mu > 0:
        return "diminishing_returns_not_formal_saturation"
    return "uncertain_or_no_consistent_gain"


def status_note(status: str) -> str:
    return {
        "larger_width_worse_transfer_inversion": "The larger width is worse on this matched comparison; treat as dataset-specific transfer behavior, not saturation.",
        "formal_saturation": "The full 95% CI for incremental improvement lies inside the predeclared equivalence margin.",
        "noninferior_practically_negligible_gain": "The upper 95% CI is within the practical margin, but the lower CI is outside the symmetric equivalence band.",
        "diminishing_returns_not_formal_saturation": "The point estimate is positive but the CI still permits a practically meaningful gain.",
        "uncertain_or_no_consistent_gain": "The matched-seed estimate is not a clear positive improvement and does not satisfy the formal saturation criterion.",
    }[status]


def aggregate_rows(rows: list[dict[str, Any]], *, run_id: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["analysis"], row["dataset"], row["artifact"], row["metric"], row["base"])].append(row)
    out = []
    for (analysis, dataset, artifact, metric, base), items in sorted(grouped.items()):
        values = [float(item["metric_value"]) for item in items]
        out.append(
            {
                "run_id": run_id,
                "analysis": analysis,
                "dataset": dataset,
                "artifact": artifact,
                "metric": metric,
                "base": base,
                "n_seeds": len(items),
                "seeds": " ".join(str(item["seed"]) for item in sorted(items, key=lambda row: row["seed"])),
                "metric_mean": mean(values),
                "metric_sd": sample_sd(values),
                "trainable_parameters": int(items[0]["trainable_parameters"]),
            }
        )
    return out


def descriptive_elbows(rows: list[dict[str, Any]], *, run_id: str, threshold: float) -> list[dict[str, Any]]:
    agg = aggregate_rows(rows, run_id=run_id)
    by_key = {(row["analysis"], row["dataset"], row["artifact"], row["metric"], int(row["base"])): row for row in agg}
    out = []
    keys = sorted({(row["analysis"], row["dataset"], row["artifact"], row["metric"]) for row in agg})
    for analysis, dataset, artifact, metric in keys:
        if not all((analysis, dataset, artifact, metric, base) in by_key for base in [2, 16]):
            continue
        value2 = float(by_key[(analysis, dataset, artifact, metric, 2)]["metric_mean"])
        value16 = float(by_key[(analysis, dataset, artifact, metric, 16)]["metric_mean"])
        total_gain = raw_increment(metric, value2, value16)
        if abs(total_gain) < 1e-12:
            elbow = ""
        else:
            elbow = ""
            for base in WIDTHS:
                value = float(by_key[(analysis, dataset, artifact, metric, base)]["metric_mean"])
                gain = raw_increment(metric, value2, value)
                frac = gain / total_gain
                if frac >= threshold:
                    elbow = base
                    break
        out.append(
            {
                "run_id": run_id,
                "analysis": analysis,
                "dataset": dataset,
                "artifact": artifact,
                "metric": metric,
                "elbow_threshold_fraction_of_base2_to_base16_gain": threshold,
                "descriptive_elbow_base": elbow,
                "base2_to_base16_total_gain": total_gain,
                "note": "Descriptive elbow only; not used as proof of saturation.",
            }
        )
    return out


def write_markdown(
    path: Path,
    *,
    run_id: str,
    adjacent: list[dict[str, Any]],
    compact: list[dict[str, Any]],
    bci_adjacent: list[dict[str, Any]],
    elbows: list[dict[str, Any]],
) -> None:
    lines = ["# Capacity and diminishing-return analysis", ""]
    lines.append("## Formal Criterion")
    lines.append("")
    lines.append("Saturation is used only when the full matched-seed 95% CI for the incremental improvement lies inside the predeclared practical equivalence band. Positive increments whose CIs still allow a meaningful gain are labelled diminishing returns rather than saturation. Larger-width regressions are labelled transfer inversion/regression, not saturation.")
    lines.append("")
    lines.append("| Metric | Margin | Justification |")
    lines.append("|---|---:|---|")
    for metric in METRICS:
        lines.append(f"| {metric} | {MARGINS[metric]:.6f} | {MARGIN_JUSTIFICATION[metric]} |")
    lines.append("")
    lines.append("## Synthetic EOG/EMG Adjacent Widths")
    lines.append("")
    lines.append(comparison_table(adjacent))
    lines.append("")
    lines.append("## Synthetic Compact-Vs-Large")
    lines.append("")
    lines.append(comparison_table(compact))
    lines.append("")
    lines.append("## BCI Zero-Shot Transfer Sensitivity")
    lines.append("")
    lines.append(comparison_table(bci_adjacent))
    lines.append("")
    lines.append("## Descriptive Elbows")
    lines.append("")
    lines.append("| Analysis | Dataset | Artifact | Metric | Elbow base | Base2-to-base16 gain | Note |")
    lines.append("|---|---|---|---|---:|---:|---|")
    for row in elbows:
        lines.append(
            f"| {row['analysis']} | {row['dataset']} | {row['artifact']} | {row['metric']} | "
            f"{row['descriptive_elbow_base']} | {float(row['base2_to_base16_total_gain']):+.6f} | {row['note']} |"
        )
    lines.append("")
    lines.append("## Required Interpretation Checks")
    lines.append("")
    lines.append("- EOG CC continues to improve from base4 through base16 in the synthetic EOG width benchmark; this should be acknowledged explicitly rather than described as saturated at base4.")
    lines.append("- BCI IV-2a zero-shot reconstruction continues improving through larger widths for the main reconstruction metrics; this should be acknowledged explicitly.")
    lines.append("- BCI IV-2b shows a larger-width inversion/regression after the compact widths for several metrics. This should be described as dataset-specific transfer behavior, not proof of universal saturation.")
    lines.append("- Bootstrap/descriptive elbows remain supporting evidence only; formal saturation language is reserved for rows labelled `formal_saturation`.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def comparison_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Dataset | Artifact | Metric | Comparison | n | Mean increment | 95% CI | Margin | Status |",
        "|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['artifact']} | {row['metric']} | {row['comparison']} | "
            f"{row['n_matched_seeds']} | {float(row['mean_incremental_improvement']):+.6f} | "
            f"[{float(row['ci95_low']):+.6f}, {float(row['ci95_high']):+.6f}] | "
            f"{float(row['equivalence_margin']):.6f} | {row['formal_status']} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    width_rows = normalize_width_rows(read_csv(args.width_index), args.run_id)
    bci_rows = normalize_bci_rows(read_csv(args.bci_zero_shot), args.run_id)

    synthetic_adjacent = comparison_rows(
        width_rows,
        run_id=args.run_id,
        comparisons=ADJACENT,
        analysis="synthetic_adjacent_width_increment",
        alpha=args.alpha,
    )
    synthetic_compact = comparison_rows(
        width_rows,
        run_id=args.run_id,
        comparisons=COMPACT_VS_LARGE,
        analysis="synthetic_compact_vs_large",
        alpha=args.alpha,
    )
    bci_adjacent = comparison_rows(
        bci_rows,
        run_id=args.run_id,
        comparisons=ADJACENT,
        analysis="bci_zero_shot_adjacent_width_increment",
        alpha=args.alpha,
    )
    bci_compact = comparison_rows(
        bci_rows,
        run_id=args.run_id,
        comparisons=COMPACT_VS_LARGE,
        analysis="bci_zero_shot_compact_vs_large",
        alpha=args.alpha,
    )
    elbows = descriptive_elbows(width_rows + bci_rows, run_id=args.run_id, threshold=args.elbow_threshold)
    aggregates = aggregate_rows(width_rows + bci_rows, run_id=args.run_id)

    write_csv(args.output_dir / "capacity_reconstruction_metric_rows.csv", width_rows + bci_rows)
    write_csv(args.output_dir / "capacity_metric_aggregates.csv", aggregates)
    write_csv(args.output_dir / "capacity_synthetic_adjacent_increment_tests.csv", synthetic_adjacent)
    write_csv(args.output_dir / "capacity_synthetic_compact_vs_large_tests.csv", synthetic_compact)
    write_csv(args.output_dir / "capacity_bci_zero_shot_adjacent_increment_tests.csv", bci_adjacent)
    write_csv(args.output_dir / "capacity_bci_zero_shot_compact_vs_large_tests.csv", bci_compact)
    write_csv(args.output_dir / "capacity_descriptive_elbows.csv", elbows)
    summary = {
        "info": {
            "run_id": args.run_id,
            "width_index": str(args.width_index),
            "bci_zero_shot": str(args.bci_zero_shot),
            "widths": WIDTHS,
            "adjacent_comparisons": ADJACENT,
            "compact_vs_large_comparisons": COMPACT_VS_LARGE,
            "metrics": METRICS,
            "margins": MARGINS,
            "margin_justification": MARGIN_JUSTIFICATION,
            "formal_saturation_rule": "95% CI for incremental improvement must lie within +/- the predeclared metric-specific margin.",
            "descriptive_elbow_rule": f"First width reaching {args.elbow_threshold:.2f} of base2-to-base16 gain; supporting evidence only.",
        },
        "synthetic_adjacent": synthetic_adjacent,
        "synthetic_compact_vs_large": synthetic_compact,
        "bci_zero_shot_adjacent": bci_adjacent,
        "bci_zero_shot_compact_vs_large": bci_compact,
        "descriptive_elbows": elbows,
    }
    (args.output_dir / "capacity_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(
        args.output_dir / "capacity_summary.md",
        run_id=args.run_id,
        adjacent=synthetic_adjacent,
        compact=synthetic_compact,
        bci_adjacent=bci_adjacent,
        elbows=elbows,
    )
    print(f"[written] {args.output_dir / 'capacity_summary.md'}", flush=True)
    print(
        f"[done] rows={len(width_rows) + len(bci_rows)} synthetic_adjacent={len(synthetic_adjacent)} "
        f"synthetic_compact={len(synthetic_compact)} bci_adjacent={len(bci_adjacent)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
