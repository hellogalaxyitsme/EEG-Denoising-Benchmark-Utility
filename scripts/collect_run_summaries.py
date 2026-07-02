#!/usr/bin/env python3
"""Collect controlled DSConv U-Net run summaries into JSON and CSV files."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any


METRIC_KEYS = ["CC", "MSE", "RMSE", "T_RRMSE", "S_RRMSE", "SDR", "PSD_KLD", "PSD_WD"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Run directories, summary/metrics files, or glob patterns.")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        paths.extend(Path(match) for match in matches)
    return sorted(set(paths))


def candidate_files(path: Path) -> list[Path]:
    if path.is_dir():
        return [path / "summary.json", path / "metrics.json"]
    return [path]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def row_from_summary(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    config = data.get("config", {})
    test = data.get("test", {})
    row: dict[str, Any] = {
        "source": str(path),
        "run_dir": str(path.parent),
        "kind": "summary",
        "output_dir": data.get("output_dir", str(path.parent)),
        "data": data.get("data", ""),
        "device": data.get("device", ""),
        "seed": data.get("seed", config.get("seed", "")),
        "trainable_parameters": data.get("trainable_parameters", ""),
        "best_epoch": data.get("best_epoch", ""),
        "best_val_sdr": data.get("best_val_sdr", ""),
        "length": data.get("length", test.get("length", "")),
        "n_samples": test.get("n_samples", ""),
    }
    for key in METRIC_KEYS:
        row[f"test_{key}"] = test.get(key, "")
    return row


def row_from_metrics(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    metrics = data.get("metrics", {})
    row: dict[str, Any] = {
        "source": str(path),
        "run_dir": str(path.parent),
        "kind": "metrics",
        "output_dir": str(path.parent),
        "data": data.get("data", ""),
        "device": data.get("device", ""),
        "seed": data.get("checkpoint_cfg", {}).get("seed", ""),
        "trainable_parameters": data.get("trainable_parameters", ""),
        "best_epoch": "",
        "best_val_sdr": "",
        "length": metrics.get("length", ""),
        "n_samples": metrics.get("n_samples", ""),
    }
    for key in METRIC_KEYS:
        row[f"test_{key}"] = metrics.get(key, "")
    return row


def collect(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for candidate in candidate_files(path):
            if not candidate.exists():
                continue
            data = load_json(candidate)
            if candidate.name == "summary.json":
                rows.append(row_from_summary(candidate, data))
            elif candidate.name == "metrics.json":
                rows.append(row_from_metrics(candidate, data))
    return rows


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = collect(expand_inputs(args.paths))
    text = json.dumps(rows, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        write_json(args.output_json, rows)
    if args.output_csv is not None:
        write_csv(args.output_csv, rows)


if __name__ == "__main__":
    main()
