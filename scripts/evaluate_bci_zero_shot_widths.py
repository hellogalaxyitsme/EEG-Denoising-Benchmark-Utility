#!/usr/bin/env python3
"""Zero-shot BCI IV-2a/IV-2b semi-synthetic evaluation for controlled DSConv U-Net widths."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import statistics
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eeg_denoise_benchmark.eval.metrics import compute_denoising_metrics  # noqa: E402
from eeg_denoise_benchmark.models import TinyDenoiser, count_trainable_parameters  # noqa: E402


FS_TRAIN = 256
SEG_SEC = 2.0
LENGTH = int(FS_TRAIN * SEG_SEC)
METRIC_KEYS = ["CC", "MSE", "RMSE", "T_RRMSE", "S_RRMSE", "SDR", "PSD_KLD", "PSD_WD"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bci2a-mat", type=Path, required=True)
    parser.add_argument("--bci2b-gdf", type=Path, required=True)
    parser.add_argument("--checkpoint-glob", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--snr-min-db", type=float, default=-6.0)
    parser.add_argument("--snr-max-db", type=float, default=2.0)
    parser.add_argument("--low-quantile", type=float, default=0.20)
    parser.add_argument("--high-quantile", type=float, default=0.90)
    parser.add_argument("--hop", type=int, default=64)
    parser.add_argument("--bci2a-eeg-index", type=int, default=9)
    parser.add_argument("--bci2a-eog-index", type=int, default=22)
    parser.add_argument("--bci2b-eeg-name", default="C3")
    parser.add_argument("--bci2b-eeg-index", type=int, default=0)
    parser.add_argument("--bci2b-eog-substr", default="EOG")
    return parser.parse_args()


def natural_key(path: Path) -> tuple[int, str]:
    match = re.search(r"base(\d+)", str(path))
    base = int(match.group(1)) if match else 10**9
    seed_match = re.search(r"seed(\d+)", str(path))
    seed = int(seed_match.group(1)) if seed_match else 10**9
    return base, seed, str(path)


def expand_checkpoints(pattern: str) -> list[Path]:
    checkpoints = sorted((Path(p) for p in glob.glob(pattern)), key=natural_key)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints matched: {pattern}")
    return checkpoints


def _mat_struct_to_dict(obj: Any) -> dict[str, Any]:
    return {name: getattr(obj, name) for name in getattr(obj, "_fieldnames", [])}


def load_bci2a_mat(mat_path: Path) -> tuple[np.ndarray, int]:
    """Load BCI IV-2a MATLAB export into continuous samples x channels."""

    mat = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    if "data" not in mat:
        raise KeyError(f"'data' not found in {mat_path}; keys={list(mat)}")

    data = mat["data"]
    runs = data if isinstance(data, (list, np.ndarray)) else [data]
    arrays: list[np.ndarray] = []
    fs: int | None = None

    for run in runs:
        if hasattr(run, "_fieldnames"):
            fields = _mat_struct_to_dict(run)
        elif isinstance(run, dict):
            fields = run
        else:
            continue
        if "X" in fields:
            arrays.append(np.asarray(fields["X"], dtype=np.float32))
        if fs is None and "fs" in fields:
            fs = int(np.asarray(fields["fs"]).item())

    if not arrays:
        raise ValueError(f"Could not find field X in {mat_path}")

    xcat = np.concatenate(arrays, axis=0)
    if xcat.ndim != 2 or xcat.shape[1] < 23:
        raise ValueError(f"Unexpected BCI IV-2a shape {xcat.shape}; expected samples x channels.")
    return xcat, fs or 250


def load_bci2b_gdf(gdf_path: Path) -> tuple[np.ndarray, float, list[str]]:
    try:
        import mne
    except Exception as exc:  # pragma: no cover - depends on remote env
        raise RuntimeError("mne is required for BCI IV-2b .gdf evaluation") from exc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = mne.io.read_raw_gdf(str(gdf_path), preload=True, verbose=False)
    sfreq = float(raw.info["sfreq"])
    ch_names = list(raw.ch_names)
    data_uv = (raw.get_data() * 1e6).astype(np.float32)
    return data_uv.T, sfreq, ch_names


def resample_to_256(x: np.ndarray, fs_in: float, fs_out: int = FS_TRAIN) -> np.ndarray:
    fs_in_i = int(round(fs_in))
    if fs_in_i == fs_out:
        return x.astype(np.float32)
    gcd = math.gcd(fs_out, fs_in_i)
    return resample_poly(x, fs_out // gcd, fs_in_i // gcd).astype(np.float32)


def bandpass(x: np.ndarray, fs: int, lo: float = 0.5, hi: float = 10.0, order: int = 4) -> np.ndarray:
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x).astype(np.float32)


def build_pool_starts(
    eeg: np.ndarray,
    eog: np.ndarray,
    *,
    length: int,
    hop: int,
    low_quantile: float,
    high_quantile: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    starts = np.arange(0, min(len(eeg), len(eog)) - length + 1, hop)
    if starts.size == 0:
        raise ValueError("Signal is shorter than the requested segment length")

    eog_bp = bandpass(eog, FS_TRAIN)
    scores = np.empty(starts.shape[0], dtype=np.float32)
    for index, start in enumerate(starts):
        seg = eog_bp[start : start + length]
        scores[index] = float(np.mean(seg * seg))

    low_thr = float(np.quantile(scores, low_quantile))
    high_thr = float(np.quantile(scores, high_quantile))
    clean = starts[scores <= low_thr]
    artifact = starts[scores >= high_thr]
    if len(clean) <= 100 or len(artifact) <= 100:
        raise ValueError(
            f"Pools too small: clean={len(clean)}, artifact={len(artifact)}; "
            "relax quantiles or hop."
        )
    return clean, artifact, {
        "n_starts": int(len(starts)),
        "clean_pool": int(len(clean)),
        "artifact_pool": int(len(artifact)),
        "low_energy_threshold": low_thr,
        "high_energy_threshold": high_thr,
    }


def mix_at_snr(clean: np.ndarray, artifact_template: np.ndarray, snr_db: float) -> tuple[np.ndarray, np.ndarray]:
    p_clean = float(np.mean(clean * clean) + 1e-12)
    p_artifact = float(np.mean(artifact_template * artifact_template) + 1e-12)
    scale = math.sqrt(p_clean / (p_artifact * (10 ** (snr_db / 10))))
    artifact = (scale * artifact_template).astype(np.float32)
    noisy = (clean + artifact).astype(np.float32)
    return noisy, artifact


def make_semisynthetic_set(
    *,
    eeg: np.ndarray,
    eog: np.ndarray,
    n_test: int,
    seed: int,
    snr_min_db: float,
    snr_max_db: float,
    hop: int,
    low_quantile: float,
    high_quantile: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    clean_starts, artifact_starts, pool_info = build_pool_starts(
        eeg,
        eog,
        length=LENGTH,
        hop=hop,
        low_quantile=low_quantile,
        high_quantile=high_quantile,
    )
    rng = np.random.default_rng(seed)
    noisy = np.zeros((n_test, LENGTH), dtype=np.float32)
    clean = np.zeros((n_test, LENGTH), dtype=np.float32)
    artifact = np.zeros((n_test, LENGTH), dtype=np.float32)
    snrs = np.zeros((n_test,), dtype=np.float32)
    sigmas = np.zeros((n_test,), dtype=np.float32)

    for index in range(n_test):
        clean_start = int(rng.choice(clean_starts))
        artifact_start = int(rng.choice(artifact_starts))
        x = eeg[clean_start : clean_start + LENGTH].copy()
        a0 = eog[artifact_start : artifact_start + LENGTH].copy()
        snr_db = float(rng.uniform(snr_min_db, snr_max_db))
        y, a = mix_at_snr(x, a0, snr_db)
        sigma_y = float(np.std(y) + 1e-8)

        noisy[index] = y / sigma_y
        clean[index] = x / sigma_y
        artifact[index] = a / sigma_y
        snrs[index] = snr_db
        sigmas[index] = sigma_y

    info = {
        **pool_info,
        "n_test": int(n_test),
        "seed": int(seed),
        "snr_min_db": float(snr_min_db),
        "snr_max_db": float(snr_max_db),
        "snr_mean_db": float(np.mean(snrs)),
        "sigma_y_mean": float(np.mean(sigmas)),
        "sigma_y_std": float(np.std(sigmas)),
    }
    return noisy, clean, artifact, snrs, sigmas, info


def bci2b_channel_indices(
    ch_names: list[str],
    eeg_name: str,
    eeg_fallback: int,
    eog_substr: str,
) -> tuple[int, int]:
    norm = [name.strip().replace("EEG-", "").replace(" ", "").upper() for name in ch_names]
    target = eeg_name.strip().replace("EEG-", "").replace(" ", "").upper()
    eeg_index = norm.index(target) if target in norm else eeg_fallback
    eog_candidates = [idx for idx, name in enumerate(norm) if eog_substr.upper() in name]
    if not eog_candidates:
        raise ValueError(f"No BCI IV-2b EOG channel found with substring {eog_substr!r}")
    return eeg_index, eog_candidates[0]


def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> tuple[TinyDenoiser, dict[str, Any], int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError(f"Checkpoint does not contain model_state: {checkpoint_path}")
    cfg = dict(checkpoint.get("cfg", {}))
    model = TinyDenoiser(
        base=int(cfg.get("base", 16)),
        extra_bottleneck_blocks=int(cfg.get("extra_blocks", 2)),
        use_dwt=bool(cfg.get("use_dwt", True)),
        use_gate=bool(cfg.get("use_gate", True)),
        use_attn=bool(cfg.get("use_attn", True)),
        use_artifact_head=bool(cfg.get("use_artifact_head", True)),
        conv_block=str(cfg.get("conv_block", "ds")),
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model, cfg, count_trainable_parameters(model)


@torch.no_grad()
def infer_clean(model: TinyDenoiser, noisy: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    predictions = np.zeros_like(noisy, dtype=np.float32)
    for start in range(0, len(noisy), batch_size):
        batch = torch.from_numpy(noisy[start : start + batch_size]).float().unsqueeze(1).to(device)
        output = model(batch)
        predictions[start : start + batch_size] = output[:, 0, :].detach().cpu().numpy()
    return predictions


def checkpoint_seed(checkpoint_path: Path, cfg: dict[str, Any]) -> int | str:
    if "seed" in cfg:
        return int(cfg["seed"])
    match = re.search(r"seed(\d+)", str(checkpoint_path))
    return int(match.group(1)) if match else ""


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        if row["kind"] != "model":
            continue
        grouped.setdefault((str(row["dataset"]), int(row["base"])), []).append(row)

    aggregates: list[dict[str, Any]] = []
    for (dataset, base), items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        seeds = sorted(int(item["train_seed"]) for item in items if item["train_seed"] != "")
        aggregate: dict[str, Any] = {
            "kind": "aggregate",
            "dataset": dataset,
            "base": base,
            "n": len(items),
            "train_seeds": " ".join(str(seed) for seed in seeds),
            "trainable_parameters": int(items[0]["trainable_parameters"]),
        }
        for key in METRIC_KEYS:
            values = [float(item[key]) for item in items]
            aggregate[f"{key}_mean"] = float(statistics.mean(values))
            aggregate[f"{key}_std"] = float(statistics.stdev(values)) if len(values) > 1 else 0.0
        aggregates.append(aggregate)
    return aggregates


def write_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    datasets: dict[str, dict[str, Any]],
) -> None:
    lines = ["# BCI Zero-Shot Width Evaluation", ""]
    lines.append("Protocol: Option-A semi-synthetic BCI EEG + BCI EOG, no fine-tuning.")
    lines.append("")
    for dataset_name, info in datasets.items():
        lines.append(f"## {dataset_name}")
        lines.append("")
        lines.append(
            f"- Source: `{info['source']}`; EEG: `{info['eeg_source']}`; "
            f"EOG: `{info['artifact_source']}`; clean pool: `{info['clean_pool']}`; "
            f"artifact pool: `{info['artifact_pool']}`."
        )
        lines.append("")
        lines.append("### Aggregate By Width")
        lines.append("")
        lines.append("| base | n | seeds | params | CC | RMSE | T_RRMSE | S_RRMSE | SDR | PSD_KLD |")
        lines.append("| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in aggregates:
            if row["dataset"] != dataset_name:
                continue
            lines.append(
                f"| {row['base']} | {row['n']} | {row['train_seeds']} | {row['trainable_parameters']} | "
                f"{row['CC_mean']:.6f} +/- {row['CC_std']:.6f} | "
                f"{row['RMSE_mean']:.6f} +/- {row['RMSE_std']:.6f} | "
                f"{row['T_RRMSE_mean']:.6f} +/- {row['T_RRMSE_std']:.6f} | "
                f"{row['S_RRMSE_mean']:.6f} +/- {row['S_RRMSE_std']:.6f} | "
                f"{row['SDR_mean']:.6f} +/- {row['SDR_std']:.6f} | "
                f"{row['PSD_KLD_mean']:.6f} +/- {row['PSD_KLD_std']:.6f} |"
            )
        lines.append("")
        lines.append("### Per-Checkpoint Results")
        lines.append("")
        lines.append("| base | train seed | params | CC | RMSE | T_RRMSE | S_RRMSE | SDR | PSD_KLD |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in rows:
            if row["dataset"] != dataset_name or row["kind"] != "model":
                continue
            lines.append(
                f"| {row['base']} | {row['train_seed']} | {row['trainable_parameters']} | "
                f"{row['CC']:.6f} | {row['RMSE']:.6f} | {row['T_RRMSE']:.6f} | "
                f"{row['S_RRMSE']:.6f} | {row['SDR']:.6f} | {row['PSD_KLD']:.6f} |"
            )
        baseline = next((row for row in rows if row["dataset"] == dataset_name and row["kind"] == "noisy_baseline"), None)
        if baseline is not None:
            lines.append("")
            lines.append(
                f"Noisy baseline: CC `{baseline['CC']:.6f}`, RMSE `{baseline['RMSE']:.6f}`, "
                f"SDR `{baseline['SDR']:.6f}`, PSD_KLD `{baseline['PSD_KLD']:.6f}`."
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[start] run_id={args.run_id} device={device}")
    checkpoints = expand_checkpoints(args.checkpoint_glob)
    print(f"[checkpoints] {len(checkpoints)}")
    for checkpoint in checkpoints:
        print(f"  - {checkpoint}")

    x2a, fs2a = load_bci2a_mat(args.bci2a_mat)
    eeg2a = resample_to_256(x2a[:, args.bci2a_eeg_index].astype(np.float32), fs2a)
    eog2a = resample_to_256(x2a[:, args.bci2a_eog_index].astype(np.float32), fs2a)
    y2a, clean2a, _artifact2a, _snr2a, _sigma2a, info2a = make_semisynthetic_set(
        eeg=eeg2a,
        eog=eog2a,
        n_test=args.n_test,
        seed=args.seed,
        snr_min_db=args.snr_min_db,
        snr_max_db=args.snr_max_db,
        hop=args.hop,
        low_quantile=args.low_quantile,
        high_quantile=args.high_quantile,
    )
    info2a.update(
        {
            "source": str(args.bci2a_mat),
            "fs_in": float(fs2a),
            "eeg_source": f"index_{args.bci2a_eeg_index}",
            "artifact_source": f"index_{args.bci2a_eog_index}",
        }
    )
    print(f"[dataset] BCI_IV2a {info2a}")

    x2b, fs2b, ch_names2b = load_bci2b_gdf(args.bci2b_gdf)
    eeg_idx2b, eog_idx2b = bci2b_channel_indices(
        ch_names2b,
        args.bci2b_eeg_name,
        args.bci2b_eeg_index,
        args.bci2b_eog_substr,
    )
    eeg2b = resample_to_256(x2b[:, eeg_idx2b].astype(np.float32), fs2b)
    eog2b = resample_to_256(x2b[:, eog_idx2b].astype(np.float32), fs2b)
    y2b, clean2b, _artifact2b, _snr2b, _sigma2b, info2b = make_semisynthetic_set(
        eeg=eeg2b,
        eog=eog2b,
        n_test=args.n_test,
        seed=args.seed,
        snr_min_db=args.snr_min_db,
        snr_max_db=args.snr_max_db,
        hop=args.hop,
        low_quantile=args.low_quantile,
        high_quantile=args.high_quantile,
    )
    info2b.update(
        {
            "source": str(args.bci2b_gdf),
            "fs_in": float(fs2b),
            "eeg_source": ch_names2b[eeg_idx2b],
            "artifact_source": ch_names2b[eog_idx2b],
        }
    )
    print(f"[dataset] BCI_IV2b {info2b}")

    datasets = {
        "BCI_IV2a": {"noisy": y2a, "clean": clean2a, "info": info2a},
        "BCI_IV2b": {"noisy": y2b, "clean": clean2b, "info": info2b},
    }

    rows: list[dict[str, Any]] = []
    for dataset_name, payload in datasets.items():
        baseline_metrics = compute_denoising_metrics(payload["clean"], payload["noisy"], fs=FS_TRAIN)
        rows.append(
            {
                "kind": "noisy_baseline",
                "dataset": dataset_name,
                "checkpoint": "",
                "base": "",
                "train_seed": "",
                "trainable_parameters": "",
                **{key: baseline_metrics[key] for key in METRIC_KEYS},
            }
        )

    for checkpoint_path in checkpoints:
        model, cfg, n_params = load_checkpoint_model(checkpoint_path, device)
        base = int(cfg.get("base", -1))
        train_seed = checkpoint_seed(checkpoint_path, cfg)
        print(f"[eval_start] base={base} seed={train_seed} params={n_params} checkpoint={checkpoint_path}")
        for dataset_name, payload in datasets.items():
            prediction = infer_clean(model, payload["noisy"], device=device, batch_size=args.batch_size)
            metrics = compute_denoising_metrics(payload["clean"], prediction, fs=FS_TRAIN)
            row = {
                "kind": "model",
                "dataset": dataset_name,
                "checkpoint": str(checkpoint_path),
                "base": base,
                "train_seed": train_seed,
                "trainable_parameters": n_params,
                "use_dwt": bool(cfg.get("use_dwt", True)),
                "use_attn": bool(cfg.get("use_attn", True)),
                "use_gate": bool(cfg.get("use_gate", True)),
                "use_artifact_head": bool(cfg.get("use_artifact_head", True)),
                **{key: metrics[key] for key in METRIC_KEYS},
            }
            rows.append(row)
            print(
                f"[result] dataset={dataset_name} base={base} seed={train_seed} "
                f"CC={row['CC']:.6f} RMSE={row['RMSE']:.6f} SDR={row['SDR']:.6f} "
                f"PSD_KLD={row['PSD_KLD']:.6f}"
            )

    aggregates = aggregate_rows(rows)
    summary = {
        "run_id": args.run_id,
        "protocol": "BCI Option-A semi-synthetic zero-shot, sigma_y normalized, no fine-tuning",
        "fs": FS_TRAIN,
        "length": LENGTH,
        "datasets": {name: payload["info"] for name, payload in datasets.items()},
        "aggregates": aggregates,
        "rows": rows,
    }
    json_path = args.output_dir / "bci_zero_shot_summary.json"
    csv_path = args.output_dir / "bci_zero_shot_summary.csv"
    aggregate_csv_path = args.output_dir / "bci_zero_shot_aggregate_by_width.csv"
    md_path = args.output_dir / "bci_zero_shot_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(csv_path, rows)
    write_csv(aggregate_csv_path, aggregates)
    write_markdown(md_path, rows, aggregates, summary["datasets"])
    print(f"[written] {json_path}")
    print(f"[written] {csv_path}")
    print(f"[written] {aggregate_csv_path}")
    print(f"[written] {md_path}")
    print("[done]")


if __name__ == "__main__":
    main()
