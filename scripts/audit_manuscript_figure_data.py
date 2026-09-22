#!/usr/bin/env python3
"""Validate released manuscript aggregate data without rerunning inference."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "results" / "manuscript_figure_data"
REQUIRED_COLUMNS = {
    "fig02_capacity.csv": {"panel", "base", "parameters"},
    "fig03_iv2a_subject_effects.csv": {"row_type", "classifier", "recipe", "n_subjects_for_inference"},
    "fig04_centered_display.csv": {"classifier", "recipe", "subject", "sdr_db", "delta_accuracy"},
    "fig05_sleepedf_subject_effects.csv": {"row_type", "condition", "n_subjects", "subject"},
}
EXPECTED_ROWS = {"fig02_capacity.csv": 130, "fig03_iv2a_subject_effects.csv": 150, "fig04_centered_display.csv": 108, "fig05_sleepedf_subject_effects.csv": 1368}
FORBIDDEN_PATH_TOKENS = ("/home/", "c:\\users\\", "desktop/litedenoisenet", "/data/neuro_data/")
def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--data-dir",type=Path,default=DEFAULT_DATA_DIR); return p.parse_args()
def read_rows(path: Path) -> list[dict[str,str]]:
    with path.open(newline="",encoding="utf-8-sig") as h: return list(csv.DictReader(h))
def subjects(rows): return {r["subject"] for r in rows if r.get("subject", "")}
def main() -> None:
    data_dir=parse_args().data_dir.resolve(); failures=[]; tables={}
    for name, req in REQUIRED_COLUMNS.items():
        path=data_dir/name
        if not path.is_file(): failures.append(f"missing {path}"); continue
        rows=read_rows(path); tables[name]=rows; cols=set(rows[0]) if rows else set()
        if not rows: failures.append(f"{name}: no rows")
        if req-cols: failures.append(f"{name}: missing columns {sorted(req-cols)}")
        if len(rows)!=EXPECTED_ROWS[name]: failures.append(f"{name}: expected {EXPECTED_ROWS[name]} rows, found {len(rows)}")
        if any(tok in path.read_text(encoding="utf-8-sig").lower() for tok in FORBIDDEN_PATH_TOKENS): failures.append(f"{name}: contains a machine-local path")
    if "fig03_iv2a_subject_effects.csv" in tables:
        r=tables["fig03_iv2a_subject_effects.csv"]
        if subjects(r)!={f"A{i:02d}" for i in range(1,10)}: failures.append("fig03: expected exactly nine IV-2a subjects")
        if {x.get("n_subjects_for_inference") for x in r}!={"9"}: failures.append("fig03: inferential subject count must be 9")
    if "fig04_centered_display.csv" in tables:
        r=tables["fig04_centered_display.csv"]; groups={}
        for x in r: groups.setdefault((x["classifier"],x["recipe"]),set()).add(x["subject"])
        if any(len(v)!=9 for v in groups.values()) or len(groups)!=3: failures.append("fig04: each classifier/recipe display group must contain nine subjects")
    if "fig05_sleepedf_subject_effects.csv" in tables:
        r=[x for x in tables["fig05_sleepedf_subject_effects.csv"] if x.get("row_type")=="subject"]; groups={}
        for x in r: groups.setdefault((x["condition"],x["denoiser_label"]),set()).add(x["subject"])
        if not groups or any(len(v)!=75 for v in groups.values()): failures.append("fig05: each condition/model group must contain 75 subjects")
        if {x.get("n_subjects") for x in tables["fig05_sleepedf_subject_effects.csv"] if x.get("row_type")=="mean_ci"}!={"75.0"}: failures.append("fig05: summary subject count must be 75")
    if failures: raise SystemExit("\n".join(failures))
if __name__=="__main__": main()
