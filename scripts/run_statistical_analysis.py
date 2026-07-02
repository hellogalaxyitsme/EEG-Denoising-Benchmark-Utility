#!/usr/bin/env python3
"""Statistical summaries for EEG denoising benchmark experiments.

This script uses only the Python standard library. It computes statistics at
the strongest unit available in the existing result files:

- reconstruction sweeps: paired seed-level comparisons and seed-bootstrap CIs;
- downstream BCI: paired subject-level comparisons and exact signed-rank tests.

The outputs are written under results/statistical_analysis/ by default.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]

METRICS_HIGHER_BETTER = {"CC", "SDR", "accuracy", "balanced_accuracy", "cohen_kappa", "macro_f1"}
METRICS_LOWER_BETTER = {"RMSE", "MSE", "T_RRMSE", "S_RRMSE", "PSD_KLD", "PSD_WD"}
CORE_RECON_METRICS = ["CC", "RMSE", "SDR", "T_RRMSE", "S_RRMSE", "PSD_KLD"]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def write_union_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen = set()
    preferred = [
        "analysis",
        "source",
        "dataset",
        "classifier",
        "condition",
        "metric",
        "stratum_type",
        "stratum_label",
        "base",
        "candidate_base",
        "reference_base",
        "n_units",
        "n_subjects",
        "unit",
        "mean",
        "mean_accuracy",
        "delta_mean",
        "delta_vs_noisy_mean",
        "p_value",
        "p_holm",
        "p_bh_fdr",
        "p_value_denoised_lt_noisy",
        "p_holm_denoised_lt_noisy",
        "p_bh_fdr_denoised_lt_noisy",
    ]
    for key in preferred:
        for row in rows:
            if key in row and key not in seen:
                fieldnames.append(key)
                seen.add(key)
                break
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def as_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def safe_int(value: object) -> int | None:
    x = as_float(value)
    return None if x is None else int(x)


def fmt(x: float | None, nd: int = 6) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{x:.{nd}f}"


def sample_sd(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def bootstrap_ci(values: list[float], n_boot: int, rng: random.Random) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], values[0]
    n = len(values)
    boots = []
    for _ in range(n_boot):
        boots.append(mean(values[rng.randrange(n)] for _ in range(n)))
    return percentile(boots, 0.025), percentile(boots, 0.975)


def paired_bootstrap_ci(deltas: list[float], n_boot: int, rng: random.Random) -> tuple[float, float]:
    return bootstrap_ci(deltas, n_boot, rng)


def _average_tied_ranks(abs_values: list[float]) -> list[float]:
    order = sorted(range(len(abs_values)), key=lambda i: abs_values[i])
    ranks = [0.0] * len(abs_values)
    pos = 0
    while pos < len(order):
        end = pos + 1
        while end < len(order) and abs(abs_values[order[end]] - abs_values[order[pos]]) < 1e-12:
            end += 1
        avg_rank = (pos + 1 + end) / 2.0
        for j in range(pos, end):
            ranks[order[j]] = avg_rank
        pos = end
    return ranks


def exact_wilcoxon_p(deltas: list[float], alternative: str) -> float | None:
    """Exact signed-rank p-value.

    alternative:
      - 'greater': median delta > 0
      - 'less': median delta < 0
      - 'two-sided': two-sided probability by doubling the smaller tail
    """
    nz = [d for d in deltas if abs(d) > 1e-12]
    if not nz:
        return 1.0
    if len(nz) > 20:
        # Avoid an exponential blow-up for small paired-sample analyses.
        return None
    ranks = _average_tied_ranks([abs(d) for d in nz])
    observed_wplus = sum(r for r, d in zip(ranks, nz) if d > 0)
    null_sums = []
    for signs in itertools.product([0, 1], repeat=len(nz)):
        null_sums.append(sum(r for r, s in zip(ranks, signs) if s))
    if alternative == "greater":
        return sum(1 for s in null_sums if s >= observed_wplus - 1e-12) / len(null_sums)
    if alternative == "less":
        return sum(1 for s in null_sums if s <= observed_wplus + 1e-12) / len(null_sums)
    lower = sum(1 for s in null_sums if s <= observed_wplus + 1e-12) / len(null_sums)
    upper = sum(1 for s in null_sums if s >= observed_wplus - 1e-12) / len(null_sums)
    return min(1.0, 2.0 * min(lower, upper))


def holm_adjust(rows: list[dict[str, object]], p_key: str = "p_value", out_key: str = "p_holm") -> None:
    indexed = []
    for i, row in enumerate(rows):
        p = as_float(row.get(p_key))
        if p is not None:
            indexed.append((p, i))
    indexed.sort()
    m = len(indexed)
    prev = 0.0
    adjusted = {}
    for rank, (p, i) in enumerate(indexed):
        adj = min(1.0, (m - rank) * p)
        adj = max(adj, prev)
        prev = adj
        adjusted[i] = adj
    for i, row in enumerate(rows):
        if i in adjusted:
            row[out_key] = adjusted[i]
        else:
            row[out_key] = ""


def bh_fdr_adjust(rows: list[dict[str, object]], p_key: str = "p_value", out_key: str = "p_bh_fdr") -> None:
    indexed = []
    for i, row in enumerate(rows):
        p = as_float(row.get(p_key))
        if p is not None:
            indexed.append((p, i))
    indexed.sort()
    m = len(indexed)
    adjusted = {}
    prev = 1.0
    for reverse_rank, (p, i) in enumerate(reversed(indexed), start=1):
        rank = m - reverse_rank + 1
        q = min(prev, p * m / rank)
        prev = q
        adjusted[i] = min(1.0, q)
    for i, row in enumerate(rows):
        if i in adjusted:
            row[out_key] = adjusted[i]
        else:
            row[out_key] = ""


def metric_alternative(metric: str, target: str) -> str:
    """Return signed-rank alternative for candidate - reference deltas."""
    if target == "candidate_better":
        return "greater" if metric in METRICS_HIGHER_BETTER else "less"
    if target == "candidate_worse":
        return "less" if metric in METRICS_HIGHER_BETTER else "greater"
    return "two-sided"


def parse_base(row: dict[str, str]) -> int | None:
    if row.get("base"):
        return safe_int(row["base"])
    text = " ".join(str(row.get(k, "")) for k in ("run_dir", "output_dir", "source", "checkpoint"))
    m = re.search(r"base(\d+)", text)
    return int(m.group(1)) if m else None


def parse_dataset_kind(row: dict[str, str]) -> str:
    data = row.get("data", "")
    run_dir = row.get("run_dir", "")
    if "EOG Contaminated" in data or "_eog_" in run_dir:
        return "EEGDenoiseNet_EOG"
    if "EMG Contaminated" in data or "_emg_" in run_dir:
        return "EEGDenoiseNet_EMG"
    if "synthetic_1M_mixed" in data or "_mixed_" in run_dir:
        return "Mixed1M"
    return row.get("dataset", "") or "unknown"


def collect_reconstruction_rows() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []

    sources = [
        (ROOT / "runs/attention_width_sweep_20260520_212826_summary_index.csv", "EEGDenoiseNet width sweep"),
        (ROOT / "runs/mixed1m_pareto_multiseed_20260521_141627_summary_index.csv", "Mixed-1M Pareto"),
    ]
    extra_seed_paths = {
        *ROOT.glob("runs/width_saturation_extra_seeds_*_summary_index.csv"),
        *ROOT.glob("runs/*width_saturation_extra_seeds*_summary_index.csv"),
    }
    for path in sorted(extra_seed_paths):
        sources.append((path, "EEGDenoiseNet width sweep extra seeds"))
    seen_reconstruction_units: set[tuple[str, str, int, int, str]] = set()
    for path, source_name in sources:
        for row in read_csv(path):
            base = parse_base(row)
            seed = safe_int(row.get("seed"))
            if base is None or seed is None:
                continue
            dataset = parse_dataset_kind(row)
            for metric in CORE_RECON_METRICS:
                value = as_float(row.get(f"test_{metric}"))
                if value is None:
                    continue
                dedupe_key = (source_name, dataset, base, seed, metric)
                if dedupe_key in seen_reconstruction_units:
                    continue
                seen_reconstruction_units.add(dedupe_key)
                records.append(
                    {
                        "analysis_family": "reconstruction",
                        "source": source_name,
                        "dataset": dataset,
                        "metric": metric,
                        "unit_id": seed,
                        "base": base,
                        "value": value,
                        "params": safe_int(row.get("trainable_parameters")),
                    }
                )

    bci_path = ROOT / "runs/bci_zero_shot_pareto_multiseed_20260521_154053/bci_zero_shot_summary.csv"
    for row in read_csv(bci_path):
        if row.get("kind") != "model":
            continue
        base = parse_base(row)
        seed = safe_int(row.get("train_seed"))
        dataset = row.get("dataset", "")
        if base is None or seed is None or not dataset:
            continue
        for metric in CORE_RECON_METRICS:
            value = as_float(row.get(metric))
            if value is None:
                continue
            records.append(
                {
                    "analysis_family": "reconstruction",
                    "source": "BCI zero-shot Pareto",
                    "dataset": f"ZeroShot_{dataset}",
                    "metric": metric,
                    "unit_id": seed,
                    "base": base,
                    "value": value,
                    "params": safe_int(row.get("trainable_parameters")),
                }
            )
    return records


def summarize_reconstruction(records: list[dict[str, object]], n_boot: int, rng: random.Random) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for r in records:
        grouped[(str(r["dataset"]), str(r["metric"]), int(r["base"]))].append(r)

    summary_rows: list[dict[str, object]] = []
    for (dataset, metric, base), rs in sorted(grouped.items()):
        vals = [float(r["value"]) for r in rs]
        lo, hi = bootstrap_ci(vals, n_boot, rng)
        params = next((r.get("params") for r in rs if r.get("params") is not None), "")
        sources = "; ".join(sorted({str(r.get("source", "")) for r in rs if r.get("source")}))
        summary_rows.append(
            {
                "analysis": "reconstruction_summary",
                "source": sources,
                "dataset": dataset,
                "metric": metric,
                "base": base,
                "params": params,
                "n_units": len(vals),
                "unit": "seed",
                "mean": mean(vals),
                "sd": sample_sd(vals),
                "ci95_low": lo,
                "ci95_high": hi,
            }
        )

    # Paired comparisons: adjacent widths and compact widths vs base16.
    by_dm: dict[tuple[str, str], dict[int, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    params_by: dict[tuple[str, str, int], int | None] = {}
    for r in records:
        key = (str(r["dataset"]), str(r["metric"]))
        base = int(r["base"])
        by_dm[key][base][int(r["unit_id"])] = float(r["value"])
        params_by[(key[0], key[1], base)] = r.get("params")  # type: ignore[assignment]

    comp_rows: list[dict[str, object]] = []
    for (dataset, metric), base_map in sorted(by_dm.items()):
        bases = sorted(base_map)
        pairs = [(a, b, "adjacent") for a, b in zip(bases, bases[1:])]
        if 16 in base_map:
            pairs.extend((b, 16, "vs_base16") for b in bases if b != 16)
        seen = set()
        for candidate, reference, comparison in pairs:
            marker = (candidate, reference, comparison)
            if marker in seen:
                continue
            seen.add(marker)
            common = sorted(set(base_map[candidate]) & set(base_map[reference]))
            if not common:
                continue
            cand_vals = [base_map[candidate][u] for u in common]
            ref_vals = [base_map[reference][u] for u in common]
            deltas = [c - r for c, r in zip(cand_vals, ref_vals)]
            lo, hi = paired_bootstrap_ci(deltas, n_boot, rng)
            p_two = exact_wilcoxon_p(deltas, "two-sided")
            p_better = exact_wilcoxon_p(deltas, metric_alternative(metric, "candidate_better"))
            comp_rows.append(
                {
                    "analysis": "reconstruction_pairwise",
                    "dataset": dataset,
                    "metric": metric,
                    "comparison": comparison,
                    "candidate_base": candidate,
                    "reference_base": reference,
                    "n_pairs": len(common),
                    "unit": "paired_seed",
                    "candidate_mean": mean(cand_vals),
                    "reference_mean": mean(ref_vals),
                    "delta_mean": mean(deltas),
                    "delta_sd": sample_sd(deltas),
                    "delta_ci95_low": lo,
                    "delta_ci95_high": hi,
                    "p_value": p_two,
                    "p_candidate_better": p_better,
                    "candidate_params": params_by.get((dataset, metric, candidate), ""),
                    "reference_params": params_by.get((dataset, metric, reference), ""),
                }
            )
    holm_adjust(comp_rows, "p_value", "p_holm")
    holm_adjust(comp_rows, "p_candidate_better", "p_candidate_better_holm")
    return summary_rows, comp_rows


def load_downstream_rows() -> list[dict[str, object]]:
    specs = [
        (
            "CSP+LDA",
            ROOT / "runs/bci2a_all_subjects_csp_lda_downstream_20260523_204332/bci2a_all_subjects_rows.csv",
            "csp_lda",
        ),
        (
            "Braindecode EEGNet/ShallowFBCSPNet",
            ROOT / "runs/bci2a_braindecode_downstream_20260523_210024/bci2a_braindecode_downstream_rows.csv",
            "",
        ),
        (
            "Braindecode Deep4Net/EEGConformer",
            ROOT / "runs/bci2a_deep4_conformer_downstream_20260524_103830/bci2a_braindecode_downstream_rows.csv",
            "",
        ),
    ]
    records = []
    for source_name, path, default_classifier in specs:
        for row in read_csv(path):
            condition = row.get("condition", "")
            subject = row.get("subject", "")
            acc = as_float(row.get("accuracy"))
            if not condition or not subject or acc is None:
                continue
            classifier = row.get("classifier") or default_classifier
            base = parse_base(row)
            seed = safe_int(row.get("train_seed") or row.get("seed"))
            records.append(
                {
                    "source": source_name,
                    "classifier": classifier,
                    "condition": condition,
                    "subject": subject,
                    "base": base,
                    "seed": seed,
                    "accuracy": acc,
                    "balanced_accuracy": as_float(row.get("balanced_accuracy")),
                    "cohen_kappa": as_float(row.get("cohen_kappa")),
                    "macro_f1": as_float(row.get("macro_f1")),
                    "params": safe_int(row.get("trainable_parameters")),
                }
            )
    return records


def summarize_downstream(records: list[dict[str, object]], n_boot: int, rng: random.Random) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    # Baselines are one row per subject/condition/classifier.
    baseline_rows: list[dict[str, object]] = []
    baseline_map: dict[str, dict[str, float]] = defaultdict(dict)
    for clf in sorted({str(r["classifier"]) for r in records}):
        for cond in ["clean_clean", "clean_noisy", "noisy_noisy"]:
            vals = [
                float(r["accuracy"])
                for r in records
                if str(r["classifier"]) == clf and str(r["condition"]) == cond
            ]
            if not vals:
                continue
            lo, hi = bootstrap_ci(vals, n_boot, rng)
            baseline_rows.append(
                {
                    "analysis": "downstream_baseline",
                    "classifier": clf,
                    "condition": cond,
                    "n_subjects": len(vals),
                    "unit": "subject",
                    "mean_accuracy": mean(vals),
                    "sd_accuracy": sample_sd(vals),
                    "ci95_low": lo,
                    "ci95_high": hi,
                }
            )
            if cond == "noisy_noisy":
                for r in records:
                    if str(r["classifier"]) == clf and str(r["condition"]) == cond:
                        baseline_map[clf][str(r["subject"])] = float(r["accuracy"])

    denoised_rows: list[dict[str, object]] = []
    for clf in sorted({str(r["classifier"]) for r in records}):
        noisy = baseline_map.get(clf, {})
        if not noisy:
            continue
        for condition in ["clean_denoised", "denoised_denoised"]:
            bases = sorted(
                {
                    int(r["base"])
                    for r in records
                    if str(r["classifier"]) == clf
                    and str(r["condition"]) == condition
                    and r.get("base") is not None
                }
            )
            for base in bases:
                by_subject: dict[str, list[float]] = defaultdict(list)
                params = ""
                for r in records:
                    if str(r["classifier"]) == clf and str(r["condition"]) == condition and r.get("base") == base:
                        by_subject[str(r["subject"])].append(float(r["accuracy"]))
                        if r.get("params") is not None:
                            params = r["params"]
                subjects = sorted(set(by_subject) & set(noisy))
                den = [mean(by_subject[s]) for s in subjects]
                ref = [noisy[s] for s in subjects]
                deltas = [d - n for d, n in zip(den, ref)]
                lo, hi = bootstrap_ci(den, n_boot, rng)
                dlo, dhi = paired_bootstrap_ci(deltas, n_boot, rng)
                p = exact_wilcoxon_p(deltas, "less")
                denoised_rows.append(
                    {
                        "analysis": "downstream_denoised",
                        "classifier": clf,
                        "condition": condition,
                        "base": base,
                        "params": params,
                        "n_subjects": len(subjects),
                        "unit": "subject_mean_over_seeds",
                        "mean_accuracy": mean(den),
                        "sd_accuracy": sample_sd(den),
                        "ci95_low": lo,
                        "ci95_high": hi,
                        "delta_vs_noisy_mean": mean(deltas),
                        "delta_vs_noisy_sd": sample_sd(deltas),
                        "delta_ci95_low": dlo,
                        "delta_ci95_high": dhi,
                        "subjects_below_noisy": sum(1 for d in deltas if d < 0),
                        "subjects_above_noisy": sum(1 for d in deltas if d > 0),
                        "p_value_denoised_lt_noisy": p,
                    }
                )

        # Best per subject over all matched denoised checkpoints.
        for condition in ["clean_denoised", "denoised_denoised"]:
            by_subject = defaultdict(list)
            for r in records:
                if str(r["classifier"]) == clf and str(r["condition"]) == condition:
                    by_subject[str(r["subject"])].append(float(r["accuracy"]))
            subjects = sorted(set(by_subject) & set(noisy))
            if not subjects:
                continue
            den = [max(by_subject[s]) for s in subjects]
            ref = [noisy[s] for s in subjects]
            deltas = [d - n for d, n in zip(den, ref)]
            dlo, dhi = paired_bootstrap_ci(deltas, n_boot, rng)
            denoised_rows.append(
                {
                    "analysis": "downstream_best_per_subject",
                    "classifier": clf,
                    "condition": condition,
                    "base": "best_any",
                    "params": "",
                    "n_subjects": len(subjects),
                    "unit": "subject_best_checkpoint",
                    "mean_accuracy": mean(den),
                    "sd_accuracy": sample_sd(den),
                    "ci95_low": bootstrap_ci(den, n_boot, rng)[0],
                    "ci95_high": bootstrap_ci(den, n_boot, rng)[1],
                    "delta_vs_noisy_mean": mean(deltas),
                    "delta_vs_noisy_sd": sample_sd(deltas),
                    "delta_ci95_low": dlo,
                    "delta_ci95_high": dhi,
                    "subjects_below_noisy": sum(1 for d in deltas if d < 0),
                    "subjects_above_noisy": sum(1 for d in deltas if d > 0),
                    "p_value_denoised_lt_noisy": exact_wilcoxon_p(deltas, "less"),
                }
            )

    holm_adjust(
        [r for r in denoised_rows if r["analysis"] == "downstream_denoised" and r["condition"] == "denoised_denoised"],
        "p_value_denoised_lt_noisy",
        "p_holm_denoised_lt_noisy",
    )
    bh_fdr_adjust(
        [r for r in denoised_rows if r["analysis"] == "downstream_denoised" and r["condition"] == "denoised_denoised"],
        "p_value_denoised_lt_noisy",
        "p_bh_fdr_denoised_lt_noisy",
    )
    # The list comprehensions above create references to the same row dicts, so rows are updated in-place.
    return baseline_rows, denoised_rows


def summarize_stratified(n_boot: int, rng: random.Random) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = read_csv(ROOT / "runs/mixed1m_stratified_pareto_multiseed_20260521_220500/mixed1m_stratified_rows.csv")
    if not rows:
        return [], []
    summary: list[dict[str, object]] = []
    pairwise: list[dict[str, object]] = []

    grouped: dict[tuple[str, str, str, int, str], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("kind") != "model":
            continue
        base = parse_base(row)
        seed = safe_int(row.get("train_seed"))
        if base is None or seed is None:
            continue
        for metric in ["CC", "SDR"]:
            val = as_float(row.get(metric))
            if val is None:
                continue
            key = (row.get("stratum_type", ""), row.get("stratum_label", ""), metric, base, str(seed))
            grouped[key].append(val)

    # One value per seed/key in practice. Collapse in case duplicated.
    collapsed = {k: mean(v) for k, v in grouped.items()}
    by_smb: dict[tuple[str, str, str, int], dict[str, float]] = defaultdict(dict)
    for (stype, label, metric, base, seed), val in collapsed.items():
        by_smb[(stype, label, metric, base)][seed] = val

    for (stype, label, metric, base), seed_vals in sorted(by_smb.items()):
        vals = list(seed_vals.values())
        lo, hi = bootstrap_ci(vals, n_boot, rng)
        summary.append(
            {
                "analysis": "stratified_summary",
                "stratum_type": stype,
                "stratum_label": label,
                "metric": metric,
                "base": base,
                "n_units": len(vals),
                "unit": "seed",
                "mean": mean(vals),
                "sd": sample_sd(vals),
                "ci95_low": lo,
                "ci95_high": hi,
            }
        )

    for (stype, label, metric), _ in sorted({((k[0], k[1], k[2]), None) for k in by_smb}):
        base_map = {}
        for key, val in by_smb.items():
            stype2, label2, metric2, base = key
            if stype2 == stype and label2 == label and metric2 == metric:
                base_map[base] = val
        if 16 not in base_map:
            continue
        for base in sorted(b for b in base_map if b != 16):
            common = sorted(set(base_map[base]) & set(base_map[16]))
            if not common:
                continue
            deltas = [base_map[base][s] - base_map[16][s] for s in common]
            lo, hi = paired_bootstrap_ci(deltas, n_boot, rng)
            pairwise.append(
                {
                    "analysis": "stratified_vs_base16",
                    "stratum_type": stype,
                    "stratum_label": label,
                    "metric": metric,
                    "candidate_base": base,
                    "reference_base": 16,
                    "n_pairs": len(common),
                    "unit": "paired_seed",
                    "delta_mean": mean(deltas),
                    "delta_sd": sample_sd(deltas),
                    "delta_ci95_low": lo,
                    "delta_ci95_high": hi,
                    "p_value": exact_wilcoxon_p(deltas, "two-sided"),
                    "p_candidate_better": exact_wilcoxon_p(deltas, metric_alternative(metric, "candidate_better")),
                }
            )
    holm_adjust(pairwise, "p_value", "p_holm")
    return summary, pairwise


def write_markdown(
    out_path: Path,
    recon_summary: list[dict[str, object]],
    recon_pairwise: list[dict[str, object]],
    down_base: list[dict[str, object]],
    down_den: list[dict[str, object]],
    strat_summary: list[dict[str, object]],
    strat_pairwise: list[dict[str, object]],
) -> None:
    lines: list[str] = []
    lines.append("# Statistical Analysis Summary")
    lines.append("")
    lines.append("This analysis was generated from the current result files. Reconstruction statistics use seed-level aggregates because per-sample predictions were not retained for all sweeps. Downstream BCI statistics use subject-level paired comparisons across all 9 BCI IV-2a subjects.")
    lines.append("")

    lines.append("## Reconstruction Width Sweeps")
    lines.append("")
    lines.append("Bootstrap confidence intervals are computed over training seeds. With `n=3`, these intervals quantify training-run variability only; they are not test-set uncertainty intervals.")
    lines.append("")
    for dataset in sorted({str(r["dataset"]) for r in recon_summary}):
        lines.append(f"### {dataset}")
        cc_rows = [r for r in recon_summary if r["dataset"] == dataset and r["metric"] == "CC"]
        if cc_rows:
            lines.append("")
            lines.append("| Base | Mean CC | 95% bootstrap CI | n seeds |")
            lines.append("|---:|---:|---:|---:|")
            for r in sorted(cc_rows, key=lambda x: int(x["base"])):
                lines.append(f"| {r['base']} | {fmt(float(r['mean']))} | [{fmt(float(r['ci95_low']))}, {fmt(float(r['ci95_high']))}] | {r['n_units']} |")
        vs16 = [
            r
            for r in recon_pairwise
            if r["dataset"] == dataset and r["metric"] == "CC" and r["comparison"] == "vs_base16"
        ]
        if vs16:
            lines.append("")
            lines.append("| Candidate vs base16 | Delta CC | 95% paired-bootstrap CI | two-sided p | Holm p |")
            lines.append("|---:|---:|---:|---:|---:|")
            for r in sorted(vs16, key=lambda x: int(x["candidate_base"])):
                lines.append(
                    f"| base{r['candidate_base']} | {fmt(float(r['delta_mean']))} | "
                    f"[{fmt(float(r['delta_ci95_low']))}, {fmt(float(r['delta_ci95_high']))}] | "
                    f"{fmt(as_float(r.get('p_value')))} | {fmt(as_float(r.get('p_holm')))} |"
                )
        lines.append("")

    lines.append("## Downstream BCI IV-2a Classifier Statistics")
    lines.append("")
    lines.append("The main hypothesis test is matched denoised/denoised accuracy vs noisy/noisy accuracy across subjects, using a one-sided exact Wilcoxon signed-rank test with alternative denoised < noisy.")
    lines.append("")
    lines.append("### Baselines")
    lines.append("")
    lines.append("| Classifier | Clean/Clean | Clean/Noisy | Noisy/Noisy |")
    lines.append("|---|---:|---:|---:|")
    by_clf_cond = {(r["classifier"], r["condition"]): r for r in down_base}
    for clf in sorted({str(r["classifier"]) for r in down_base}):
        vals = []
        for cond in ["clean_clean", "clean_noisy", "noisy_noisy"]:
            r = by_clf_cond.get((clf, cond))
            if r:
                vals.append(f"{fmt(float(r['mean_accuracy']))} ± {fmt(float(r['sd_accuracy']))}")
            else:
                vals.append("NA")
        lines.append(f"| {clf} | {vals[0]} | {vals[1]} | {vals[2]} |")
    lines.append("")

    lines.append("### Matched Denoised/Denoised")
    lines.append("")
    lines.append("| Classifier | Base | Accuracy | Delta vs noisy/noisy | 95% delta CI | Below noisy | p | Holm p |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    matched = [r for r in down_den if r["analysis"] == "downstream_denoised" and r["condition"] == "denoised_denoised"]
    for r in sorted(matched, key=lambda x: (str(x["classifier"]), int(x["base"]))):
        lines.append(
            f"| {r['classifier']} | {r['base']} | {fmt(float(r['mean_accuracy']))} ± {fmt(float(r['sd_accuracy']))} | "
            f"{fmt(float(r['delta_vs_noisy_mean']))} ± {fmt(float(r['delta_vs_noisy_sd']))} | "
            f"[{fmt(float(r['delta_ci95_low']))}, {fmt(float(r['delta_ci95_high']))}] | "
            f"{r['subjects_below_noisy']}/{r['n_subjects']} | "
            f"{fmt(as_float(r.get('p_value_denoised_lt_noisy')))} | {fmt(as_float(r.get('p_holm_denoised_lt_noisy')))} |"
        )
    lines.append("")

    lines.append("### Best Fixed Matched Width By Classifier")
    lines.append("")
    lines.append("| Classifier | Best fixed width | Accuracy | Delta vs noisy/noisy | Interpretation |")
    lines.append("|---|---:|---:|---:|---|")
    for clf in sorted({str(r["classifier"]) for r in matched}):
        rows = [r for r in matched if r["classifier"] == clf]
        best = max(rows, key=lambda r: float(r["mean_accuracy"]))
        delta = float(best["delta_vs_noisy_mean"])
        if delta < -0.02:
            interp = "supports utility gap"
        elif delta > 0.01:
            interp = "does not support degradation"
        else:
            interp = "approximately neutral"
        lines.append(
            f"| {clf} | base{best['base']} | {fmt(float(best['mean_accuracy']))} ± {fmt(float(best['sd_accuracy']))} | "
            f"{fmt(delta)} | {interp} |"
        )
    lines.append("")

    lines.append("## Mixed-1M Stratified Statistics")
    lines.append("")
    lines.append(f"Computed stratified summaries: {len(strat_summary)} rows. Computed base-vs-base16 stratified comparisons: {len(strat_pairwise)} rows.")
    lines.append("The detailed CSV files provide the corresponding effect estimates and multiple-comparison corrections.")
    lines.append("")
    out_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "statistical_analysis"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260524)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    recon_records = collect_reconstruction_rows()
    recon_summary, recon_pairwise = summarize_reconstruction(recon_records, args.bootstrap, rng)

    down_records = load_downstream_rows()
    down_base, down_den = summarize_downstream(down_records, args.bootstrap, rng)

    strat_summary, strat_pairwise = summarize_stratified(args.bootstrap, rng)

    write_csv(
        out_dir / "reconstruction_summary.csv",
        recon_summary,
        [
            "analysis",
            "source",
            "dataset",
            "metric",
            "base",
            "params",
            "n_units",
            "unit",
            "mean",
            "sd",
            "ci95_low",
            "ci95_high",
        ],
    )
    write_csv(
        out_dir / "reconstruction_pairwise.csv",
        recon_pairwise,
        [
            "analysis",
            "dataset",
            "metric",
            "comparison",
            "candidate_base",
            "reference_base",
            "n_pairs",
            "unit",
            "candidate_mean",
            "reference_mean",
            "delta_mean",
            "delta_sd",
            "delta_ci95_low",
            "delta_ci95_high",
            "p_value",
            "p_holm",
            "p_candidate_better",
            "p_candidate_better_holm",
            "candidate_params",
            "reference_params",
        ],
    )
    write_csv(
        out_dir / "downstream_baseline_stats.csv",
        down_base,
        ["analysis", "classifier", "condition", "n_subjects", "unit", "mean_accuracy", "sd_accuracy", "ci95_low", "ci95_high"],
    )
    write_csv(
        out_dir / "downstream_denoised_stats.csv",
        down_den,
        [
            "analysis",
            "classifier",
            "condition",
            "base",
            "params",
            "n_subjects",
            "unit",
            "mean_accuracy",
            "sd_accuracy",
            "ci95_low",
            "ci95_high",
            "delta_vs_noisy_mean",
            "delta_vs_noisy_sd",
            "delta_ci95_low",
            "delta_ci95_high",
            "subjects_below_noisy",
            "subjects_above_noisy",
            "p_value_denoised_lt_noisy",
            "p_holm_denoised_lt_noisy",
            "p_bh_fdr_denoised_lt_noisy",
        ],
    )
    write_csv(
        out_dir / "mixed1m_stratified_summary_stats.csv",
        strat_summary,
        ["analysis", "stratum_type", "stratum_label", "metric", "base", "n_units", "unit", "mean", "sd", "ci95_low", "ci95_high"],
    )
    write_csv(
        out_dir / "mixed1m_stratified_pairwise_stats.csv",
        strat_pairwise,
        [
            "analysis",
            "stratum_type",
            "stratum_label",
            "metric",
            "candidate_base",
            "reference_base",
            "n_pairs",
            "unit",
            "delta_mean",
            "delta_sd",
            "delta_ci95_low",
            "delta_ci95_high",
            "p_value",
            "p_holm",
            "p_candidate_better",
        ],
    )
    write_markdown(
        out_dir / "statistical_summary.md",
        recon_summary,
        recon_pairwise,
        down_base,
        down_den,
        strat_summary,
        strat_pairwise,
    )
    write_union_csv(
        out_dir / "statistical_summary.csv",
        recon_summary + recon_pairwise + down_base + down_den + strat_summary + strat_pairwise,
    )

    print(f"Wrote statistical outputs to {out_dir}")
    print(f"Reconstruction records: {len(recon_records)}")
    print(f"Downstream records: {len(down_records)}")
    print(f"Stratified summary rows: {len(strat_summary)}")


if __name__ == "__main__":
    main()
