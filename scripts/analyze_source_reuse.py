#!/usr/bin/env python3
"""Audit Mixed-1M source reuse and effective sample size."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


RECIPE_NAMES = {
    0: "EOG",
    1: "EMG",
    2: "EOG+EMG",
    3: "EMG+LINE",
    4: "EOG+LINE",
    5: "EOG+EMG+LINE",
    6: "EOG+EMG+LINE+ECG",
}
SOURCE_KEY_CANDIDATES = {
    "clean": ("clean_source_index", "eeg_id"),
    "eog": ("eog_source_index", "eog_id"),
    "emg": ("emg_source_index", "emg_id"),
}
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Mixed-1M corpus root containing train/val/test chunks.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260816)
    return parser.parse_args()


def split_list(raw: str) -> list[str]:
    values = [part.strip() for part in raw.replace(",", " ").split() if part.strip()]
    unknown = sorted(set(values) - set(SPLITS))
    if unknown:
        raise ValueError(f"Unknown split(s): {unknown}; expected some of {SPLITS}")
    return values


def chunk_paths(root: Path, split: str, max_chunks: int | None) -> list[Path]:
    split_dir = root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")
    paths = sorted(split_dir.glob("chunk_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No chunk_*.npz files found in {split_dir}")
    return paths[:max_chunks] if max_chunks is not None else paths


def choose_key(files: list[str], source_type: str) -> str:
    for key in SOURCE_KEY_CANDIDATES[source_type]:
        if key in files:
            return key
    raise KeyError(f"No source-index key found for {source_type}; tried {SOURCE_KEY_CANDIDATES[source_type]}")


def load_split_metadata(root: Path, split: str, max_chunks: int | None) -> dict[str, np.ndarray]:
    paths = chunk_paths(root, split, max_chunks)
    recipes: list[np.ndarray] = []
    ids: dict[str, list[np.ndarray]] = {"clean": [], "eog": [], "emg": []}
    schema_keys: dict[str, str] | None = None
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            if "recipe_id" not in z.files:
                raise KeyError(f"Missing recipe_id in {path}")
            if schema_keys is None:
                schema_keys = {source: choose_key(z.files, source) for source in ids}
            recipes.append(np.asarray(z["recipe_id"], dtype=np.int16))
            for source, key in schema_keys.items():
                ids[source].append(np.asarray(z[key], dtype=np.int64))
    out: dict[str, np.ndarray] = {
        "recipe_id": np.concatenate(recipes),
        "clean": np.concatenate(ids["clean"]),
        "eog": np.concatenate(ids["eog"]),
        "emg": np.concatenate(ids["emg"]),
    }
    out["n_chunks"] = np.asarray([len(paths)], dtype=np.int64)
    return out


def finite_float(value: float | int | None) -> float | str:
    if value is None:
        return ""
    value = float(value)
    if math.isnan(value):
        return ""
    return value


def reuse_counts(source_ids: np.ndarray) -> np.ndarray:
    used = np.asarray(source_ids, dtype=np.int64)
    used = used[used >= 0]
    if used.size == 0:
        return np.asarray([], dtype=np.int64)
    _, counts = np.unique(used, return_counts=True)
    return counts.astype(np.int64)


def count_stats(counts: np.ndarray) -> dict[str, Any]:
    if counts.size == 0:
        return {
            "n_unique_sources": 0,
            "reuse_min": "",
            "reuse_median": "",
            "reuse_mean": "",
            "reuse_std": "",
            "reuse_max": "",
        }
    return {
        "n_unique_sources": int(counts.size),
        "reuse_min": int(np.min(counts)),
        "reuse_median": float(np.median(counts)),
        "reuse_mean": float(np.mean(counts)),
        "reuse_std": float(np.std(counts, ddof=1)) if counts.size > 1 else 0.0,
        "reuse_max": int(np.max(counts)),
    }


def source_summary_row(split: str, source_type: str, source_ids: np.ndarray, n_mixtures: int) -> dict[str, Any]:
    used = source_ids[source_ids >= 0]
    row = {
        "split": split,
        "source_type": source_type,
        "n_mixtures_in_split": n_mixtures,
        "n_mixtures_using_source_type": int(used.size),
    }
    row.update(count_stats(reuse_counts(source_ids)))
    return row


def recipe_summary_rows(split: str, pack: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    recipes = pack["recipe_id"]
    for recipe_id in sorted(int(x) for x in np.unique(recipes)):
        mask = recipes == recipe_id
        recipe_name = RECIPE_NAMES.get(recipe_id, f"recipe_{recipe_id}")
        for source_type in ("clean", "eog", "emg"):
            source_ids = pack[source_type][mask]
            used = source_ids[source_ids >= 0]
            row = {
                "split": split,
                "recipe_id": recipe_id,
                "recipe": recipe_name,
                "source_type": source_type,
                "n_mixtures_in_recipe": int(mask.sum()),
                "n_mixtures_using_source_type": int(used.size),
            }
            row.update(count_stats(reuse_counts(source_ids)))
            rows.append(row)
    return rows


def combination_rows(split: str, pack: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    recipes = pack["recipe_id"]
    for recipe_id in sorted(int(x) for x in np.unique(recipes)):
        mask = recipes == recipe_id
        recipe_name = RECIPE_NAMES.get(recipe_id, f"recipe_{recipe_id}")
        clean = pack["clean"][mask]
        eog = pack["eog"][mask]
        emg = pack["emg"][mask]
        combos = {
            "clean_only": [(int(c),) for c in clean],
            "clean_eog_pair": [(int(c), int(e)) for c, e in zip(clean, eog) if e >= 0],
            "clean_emg_pair": [(int(c), int(m)) for c, m in zip(clean, emg) if m >= 0],
            "eog_emg_pair": [(int(e), int(m)) for e, m in zip(eog, emg) if e >= 0 and m >= 0],
            "clean_eog_emg_tuple": [
                (int(c), int(e), int(m)) for c, e, m in zip(clean, eog, emg) if e >= 0 and m >= 0
            ],
        }
        for combination_type, values in combos.items():
            rows.append(
                {
                    "split": split,
                    "recipe_id": recipe_id,
                    "recipe": recipe_name,
                    "combination_type": combination_type,
                    "n_mixtures_applicable": len(values),
                    "n_unique_combinations": len(set(values)),
                }
            )
    return rows


def histogram_rows(split: str, source_type: str, counts: np.ndarray) -> list[dict[str, Any]]:
    counter = Counter(int(x) for x in counts.tolist())
    total = int(counts.size)
    return [
        {
            "split": split,
            "source_type": source_type,
            "reuse_count": count,
            "n_sources": n_sources,
            "fraction_sources": float(n_sources / total) if total else "",
        }
        for count, n_sources in sorted(counter.items())
    ]


def ecdf_rows(split: str, source_type: str, counts: np.ndarray) -> list[dict[str, Any]]:
    if counts.size == 0:
        return []
    sorted_counts = np.sort(counts)
    n = sorted_counts.size
    rows: list[dict[str, Any]] = []
    last_count = None
    for idx, count in enumerate(sorted_counts, start=1):
        count = int(count)
        if count == last_count:
            rows[-1]["ecdf"] = float(idx / n)
        else:
            rows.append({"split": split, "source_type": source_type, "reuse_count": count, "ecdf": float(idx / n)})
            last_count = count
    return rows


def source_bootstrap_row(
    split: str,
    source_type: str,
    counts: np.ndarray,
    n_resamples: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    row = {
        "split": split,
        "source_type": source_type,
        "estimand": "mean_reuse_count_over_source_segments",
        "n_unique_sources": int(counts.size),
        "n_bootstrap_resamples": n_resamples,
        "point_estimate": finite_float(np.mean(counts)) if counts.size else "",
        "ci95_low_source_clustered": "",
        "ci95_high_source_clustered": "",
    }
    if counts.size == 0 or n_resamples <= 0:
        return row
    idx = rng.integers(0, counts.size, size=(n_resamples, counts.size))
    means = counts[idx].mean(axis=1)
    row["ci95_low_source_clustered"] = float(np.quantile(means, 0.025))
    row["ci95_high_source_clustered"] = float(np.quantile(means, 0.975))
    return row


def split_source_sets(packs: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, set[int]]]:
    sets: dict[str, dict[str, set[int]]] = defaultdict(dict)
    for split, pack in packs.items():
        for source_type in ("clean", "eog", "emg"):
            ids = pack[source_type]
            sets[source_type][split] = set(int(x) for x in ids[ids >= 0].tolist())
    return sets


def overlap_rows(packs: dict[str, dict[str, np.ndarray]], meta: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sets = split_source_sets(packs)
    source_name_map = {"clean": ("clean", "EEG"), "eog": ("eog", "EOG"), "emg": ("emg", "EMG")}
    for source_type, split_sets in sets.items():
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            if a in split_sets and b in split_sets:
                rows.append(
                    {
                        "source_type": source_type,
                        "split_a": a,
                        "split_b": b,
                        "n_unique_a_represented": len(split_sets[a]),
                        "n_unique_b_represented": len(split_sets[b]),
                        "n_intersection_represented": len(split_sets[a] & split_sets[b]),
                        "n_intersection_from_meta": meta_overlap(meta, source_name_map[source_type], a, b),
                    }
                )
    return rows


def meta_overlap(meta: dict[str, Any] | None, names: tuple[str, str], a: str, b: str) -> int | str:
    if not meta:
        return ""
    for outer in ("source_splits", "split_indices"):
        block = meta.get(outer, {})
        for name in names:
            if name in block and a in block[name] and b in block[name]:
                return len(set(int(x) for x in block[name][a]) & set(int(x) for x in block[name][b]))
    return ""


def source_pool_rows(meta: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not meta:
        return []
    rows: list[dict[str, Any]] = []
    block = meta.get("source_splits") or meta.get("split_indices") or {}
    for source_name, source_splits in block.items():
        if not isinstance(source_splits, dict):
            continue
        for split, values in source_splits.items():
            rows.append({"source_pool": source_name, "split": split, "n_source_segments_in_split_pool": len(values)})
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def maybe_write_figures(output_dir: Path, counts_by_split_source: dict[tuple[str, str], np.ndarray]) -> list[str]:
    figure_paths: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on remote environment
        (output_dir / "source-reuse_figure_error.txt").write_text(f"matplotlib unavailable: {exc}\n", encoding="utf-8")
        return figure_paths

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = {"clean": "#2563eb", "eog": "#dc2626", "emg": "#059669"}
    for split in SPLITS:
        present = [(source, counts_by_split_source[(split, source)]) for source in ("clean", "eog", "emg") if (split, source) in counts_by_split_source]
        if not present:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
        for source_type, counts in present:
            if counts.size == 0:
                continue
            axes[0].hist(counts, bins=40, alpha=0.45, color=colors[source_type], label=source_type)
            sorted_counts = np.sort(counts)
            axes[1].step(sorted_counts, np.arange(1, sorted_counts.size + 1) / sorted_counts.size, where="post", color=colors[source_type], label=source_type)
        axes[0].set_title(f"{split} reuse histogram")
        axes[0].set_xlabel("reuse count per source segment")
        axes[0].set_ylabel("number of source segments")
        axes[0].legend(frameon=False)
        axes[1].set_title(f"{split} reuse ECDF")
        axes[1].set_xlabel("reuse count per source segment")
        axes[1].set_ylabel("empirical cumulative probability")
        axes[1].legend(frameon=False)
        path = fig_dir / f"source-reuse_{split}_reuse_hist_ecdf.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        figure_paths.append(str(path))
    return figure_paths


def write_markdown(
    path: Path,
    args: argparse.Namespace,
    source_rows: list[dict[str, Any]],
    overlap: list[dict[str, Any]],
    bootstrap: list[dict[str, Any]],
    figure_paths: list[str],
) -> None:
    lines = [
        "# source-reuse Mixed-1M source-reuse/effective-sample-size audit",
        "",
        f"- Run ID: `{args.run_id}`",
        f"- Corpus: `{args.data}`",
        f"- Splits audited: `{args.splits}`",
        f"- Max chunks per split: `{args.max_chunks if args.max_chunks is not None else 'all'}`",
        f"- Source-level bootstrap resamples: `{args.bootstrap_resamples}`",
        "",
        "## Interpretation guardrails",
        "",
        "- The mixture count is a Monte-Carlo count from fixed source pools, not a count of biologically independent EEG observations.",
        "- Train/validation/test source disjointness prevents direct source reuse leakage across splits, but generated mixtures sharing source segments remain technical resamples of those fixed pools.",
        "- Larger generated test sets should be described as denser Monte-Carlo sampling of the mixture-generating process conditional on the fixed source pools.",
        "- Source-clustered bootstrap intervals below summarize uncertainty over represented source segments; they are distinct from Monte-Carlo variability over generated mixtures.",
        "",
        "## Split-level reuse summary",
        "",
        "| split | source | mixtures using source | unique sources represented | min | median | mean | sd | max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in source_rows:
        lines.append(
            "| {split} | {source_type} | {n_mixtures_using_source_type} | {n_unique_sources} | {reuse_min} | {reuse_median} | {reuse_mean:.3f} | {reuse_std:.3f} | {reuse_max} |".format(
                **row
            )
        )
    lines.extend(["", "## Split-overlap check", ""])
    if overlap:
        lines.extend(
            [
                "| source | split A | split B | unique A | unique B | represented overlap | metadata overlap |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in overlap:
            lines.append(
                "| {source_type} | {split_a} | {split_b} | {n_unique_a_represented} | {n_unique_b_represented} | {n_intersection_represented} | {n_intersection_from_meta} |".format(
                    **row
                )
            )
    lines.extend(["", "## Source-clustered bootstrap", ""])
    if bootstrap:
        lines.extend(
            [
                "| split | source | mean reuse | source-clustered 95% CI | unique sources |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in bootstrap:
            lines.append(
                f"| {row['split']} | {row['source_type']} | {float(row['point_estimate']):.3f} | [{float(row['ci95_low_source_clustered']):.3f}, {float(row['ci95_high_source_clustered']):.3f}] | {row['n_unique_sources']} |"
            )
    lines.extend(["", "## Figures", ""])
    if figure_paths:
        for fig in figure_paths:
            lines.append(f"- `{fig}`")
    else:
        lines.append("- Figure generation unavailable; use `source-reuse_reuse_count_histogram.csv` and `source-reuse_reuse_ecdf.csv`.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = split_list(args.splits)
    meta_path = args.data / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None

    packs: dict[str, dict[str, np.ndarray]] = {}
    for split in splits:
        print(f"[load] split={split}", flush=True)
        packs[split] = load_split_metadata(args.data, split, args.max_chunks)

    source_rows: list[dict[str, Any]] = []
    recipe_rows: list[dict[str, Any]] = []
    combo_rows: list[dict[str, Any]] = []
    hist_rows: list[dict[str, Any]] = []
    ecdf: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    counts_by_split_source: dict[tuple[str, str], np.ndarray] = {}
    rng = np.random.default_rng(args.bootstrap_seed)

    for split, pack in packs.items():
        n_mixtures = int(pack["recipe_id"].size)
        for source_type in ("clean", "eog", "emg"):
            source_rows.append(source_summary_row(split, source_type, pack[source_type], n_mixtures))
            counts = reuse_counts(pack[source_type])
            counts_by_split_source[(split, source_type)] = counts
            hist_rows.extend(histogram_rows(split, source_type, counts))
            ecdf.extend(ecdf_rows(split, source_type, counts))
            bootstrap_rows.append(source_bootstrap_row(split, source_type, counts, args.bootstrap_resamples, rng))
        recipe_rows.extend(recipe_summary_rows(split, pack))
        combo_rows.extend(combination_rows(split, pack))

    overlap = overlap_rows(packs, meta)
    pool_rows = source_pool_rows(meta)
    figure_paths = maybe_write_figures(args.output_dir, counts_by_split_source)

    write_csv(args.output_dir / "source-reuse_split_source_reuse_summary.csv", source_rows)
    write_csv(args.output_dir / "source-reuse_recipe_source_reuse_summary.csv", recipe_rows)
    write_csv(args.output_dir / "source-reuse_unique_combinations_summary.csv", combo_rows)
    write_csv(args.output_dir / "source-reuse_split_source_overlap.csv", overlap)
    write_csv(args.output_dir / "source-reuse_source_pool_split_sizes.csv", pool_rows)
    write_csv(args.output_dir / "source-reuse_reuse_count_histogram.csv", hist_rows)
    write_csv(args.output_dir / "source-reuse_reuse_ecdf.csv", ecdf)
    write_csv(args.output_dir / "source-reuse_source_clustered_bootstrap.csv", bootstrap_rows)
    write_json(
        args.output_dir / "source-reuse_summary.json",
        {
            "run_id": args.run_id,
            "data": str(args.data),
            "splits": splits,
            "max_chunks": args.max_chunks,
            "bootstrap_resamples": args.bootstrap_resamples,
            "source_reuse_summary": source_rows,
            "source_overlap": overlap,
            "source_clustered_bootstrap": bootstrap_rows,
            "figure_paths": figure_paths,
            "language_guardrails": {
                "avoid": "100,000 independent EEG observations",
                "preferred": "denser Monte-Carlo sampling of the mixture-generating process conditional on the fixed source pools",
                "leakage_note": "source-level train/validation/test disjointness prevents direct source leakage but does not make every generated mixture biologically independent",
            },
        },
    )
    write_markdown(args.output_dir / "source-reuse_summary.md", args, source_rows, overlap, bootstrap_rows, figure_paths)
    print(f"[done] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
