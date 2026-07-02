#!/usr/bin/env python3
"""BCI IV-2a downstream evaluation with Braindecode EEGNet/ShallowFBCSPNet."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from evaluate_bci2a_downstream_csp_lda import (  # noqa: E402
    CLASS_NAMES,
    FS_MODEL,
    bandpass_epochs,
    checkpoint_seed,
    denoise_epochs,
    expand_checkpoints,
    load_bci2a_trials,
    load_checkpoint_model,
    make_noisy_epochs,
)


METRIC_KEYS = ["accuracy", "balanced_accuracy", "cohen_kappa", "macro_f1"]
DEFAULT_SUBJECTS = [f"A{index:02d}" for index in range(1, 10)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bci-dir",
        type=Path,
        default=Path("data/bci_iv_2a"),
    )
    parser.add_argument("--subjects", default=",".join(DEFAULT_SUBJECTS))
    parser.add_argument("--classifiers", default="eegnet,shallowfbcsp")
    parser.add_argument("--checkpoint-glob", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--denoiser-batch-size", type=int, default=256)
    parser.add_argument("--classifier-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--snr-min-db", type=float, default=-6.0)
    parser.add_argument("--snr-max-db", type=float, default=2.0)
    parser.add_argument("--trial-start-sec", type=float, default=2.5)
    parser.add_argument("--trial-stop-sec", type=float, default=6.0)
    parser.add_argument("--bandpass-low-hz", type=float, default=8.0)
    parser.add_argument("--bandpass-high-hz", type=float, default=30.0)
    parser.add_argument("--eog-index", type=int, default=22)
    parser.add_argument("--include-artifact-trials", action="store_true")
    parser.add_argument("--max-checkpoints", type=int, default=0, help="Optional smoke-test limit.")
    return parser.parse_args()


def import_braindecode_models() -> dict[str, type[nn.Module]]:
    """Import Braindecode models while tolerating the remote MOABB 1.5 namespace change."""

    try:
        import moabb.datasets

        if not hasattr(moabb.datasets, "BNCI2014001"):
            dummy = type("BNCI2014001", (), {"__doc__": "Compatibility placeholder."})
            setattr(moabb.datasets, "BNCI2014001", dummy)
    except Exception:
        pass

    from braindecode.models import Deep4Net, EEGConformer, EEGNet, ShallowFBCSPNet

    return {
        "deep4net": Deep4Net,
        "eegconformer": EEGConformer,
        "eegnet": EEGNet,
        "shallowfbcsp": ShallowFBCSPNet,
    }


def parse_csv_list(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one comma-separated item.")
    return items


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


def standardize_train_test(train_epochs: np.ndarray, test_epochs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_epochs.mean(axis=(0, 2), keepdims=True)
    std = train_epochs.std(axis=(0, 2), keepdims=True) + 1e-6
    return ((train_epochs - mean) / std).astype(np.float32), ((test_epochs - mean) / std).astype(np.float32)


def build_classifier_model(name: str, n_chans: int, n_times: int, n_outputs: int) -> nn.Module:
    models = import_braindecode_models()
    key = name.strip().lower()
    if key == "eegnet":
        return models["eegnet"](
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
            final_conv_length="auto",
            drop_prob=0.25,
        )
    if key in {"fbcsp", "shallowfbcsp", "shallow_fbcsp"}:
        return models["shallowfbcsp"](
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
            final_conv_length="auto",
            n_filters_time=40,
            n_filters_spat=40,
            filter_time_length=25,
            pool_time_length=75,
            pool_time_stride=15,
            drop_prob=0.5,
        )
    if key in {"deep4", "deep4net", "deep"}:
        return models["deep4net"](
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
            final_conv_length="auto",
            drop_prob=0.5,
        )
    if key in {"eegconformer", "conformer"}:
        return models["eegconformer"](
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
            n_filters_time=40,
            filter_time_length=25,
            pool_time_length=75,
            pool_time_stride=15,
            drop_prob=0.5,
            att_depth=6,
            att_heads=10,
            att_drop_prob=0.5,
            final_fc_length="auto",
        )
    raise ValueError(f"Unsupported classifier: {name}")


def train_classifier(
    *,
    classifier_name: str,
    train_epochs: np.ndarray,
    y_train: np.ndarray,
    test_epochs: np.ndarray,
    device: torch.device,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    set_seed(seed)
    train_x, test_x = standardize_train_test(train_epochs, test_epochs)
    n_chans = int(train_x.shape[1])
    n_times = int(train_x.shape[2])
    n_outputs = int(len(CLASS_NAMES))
    model = build_classifier_model(classifier_name, n_chans, n_times, n_outputs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(y_train).long())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

    model.train()
    start_time = time.time()
    last_loss = float("nan")
    for _epoch in range(epochs):
        total_loss = 0.0
        n_seen = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * int(xb.shape[0])
            n_seen += int(xb.shape[0])
        last_loss = total_loss / max(n_seen, 1)

    model.eval()
    predictions: list[np.ndarray] = []
    test_tensor = torch.from_numpy(test_x).float()
    with torch.no_grad():
        for start in range(0, len(test_tensor), batch_size):
            xb = test_tensor[start : start + batch_size].to(device)
            logits = model(xb)
            predictions.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
    pred = np.concatenate(predictions).astype(np.int64)
    train_info = {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "final_train_loss": float(last_loss),
        "train_time_sec": float(time.time() - start_time),
    }
    return pred, train_info


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    classifier: str,
    subject: str,
    condition: str,
    variant: str = "",
    base: int | str = "",
    train_seed: int | str = "",
    trainable_parameters: int | str = "",
    checkpoint: str = "",
    classifier_seed: int | str = "",
    train_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    row: dict[str, Any] = {
        "classifier": classifier,
        "subject": subject,
        "condition": condition,
        "variant": variant,
        "base": base,
        "train_seed": train_seed,
        "trainable_parameters": trainable_parameters,
        "checkpoint": checkpoint,
        "classifier_seed": classifier_seed,
        "n_trials": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": json.dumps(cm.tolist()),
    }
    if train_info:
        row.update({f"classifier_{key}": value for key, value in train_info.items()})
    return row


def fit_eval_row(
    *,
    classifier_name: str,
    subject: str,
    condition: str,
    train_epochs: np.ndarray,
    y_train: np.ndarray,
    test_epochs: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    seed: int,
    args: argparse.Namespace,
    variant: str = "",
    base: int | str = "",
    train_seed: int | str = "",
    trainable_parameters: int | str = "",
    checkpoint: str = "",
) -> dict[str, Any]:
    pred, train_info = train_classifier(
        classifier_name=classifier_name,
        train_epochs=train_epochs,
        y_train=y_train,
        test_epochs=test_epochs,
        device=device,
        seed=seed,
        epochs=args.epochs,
        batch_size=args.classifier_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    return evaluate_predictions(
        y_test,
        pred,
        classifier=classifier_name,
        subject=subject,
        condition=condition,
        variant=variant,
        base=base,
        train_seed=train_seed,
        trainable_parameters=trainable_parameters,
        checkpoint=checkpoint,
        classifier_seed=seed,
        train_info=train_info,
    )


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.stdev(values))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_checkpoint_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row["condition"] not in {"clean_denoised", "denoised_denoised"}:
            continue
        key = (
            str(row["classifier"]),
            str(row["subject"]),
            str(row["condition"]),
            str(row.get("variant") or f"base{row['base']}"),
            str(row["base"]),
        )
        grouped.setdefault(key, []).append(row)

    aggregates: list[dict[str, Any]] = []
    for (classifier, subject, condition, variant, base), items in sorted(grouped.items()):
        seeds = sorted(int(item["train_seed"]) for item in items if item["train_seed"] != "")
        row: dict[str, Any] = {
            "classifier": classifier,
            "subject": subject,
            "condition": f"{condition}_aggregate",
            "variant": variant,
            "base": base,
            "n": len(items),
            "train_seeds": " ".join(str(seed) for seed in seeds),
            "trainable_parameters": int(items[0]["trainable_parameters"]),
        }
        for key in METRIC_KEYS:
            values = [float(item[key]) for item in items]
            row[f"{key}_mean"], row[f"{key}_std"] = mean_std(values)
        aggregates.append(row)
    return aggregates


def aggregate_across_subjects(rows: list[dict[str, Any]], subject_aggregates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_out: list[dict[str, Any]] = []
    for classifier in sorted({str(row["classifier"]) for row in rows}):
        for condition in ["clean_clean", "clean_noisy", "noisy_noisy"]:
            items = [row for row in rows if row["classifier"] == classifier and row["condition"] == condition]
            out: dict[str, Any] = {"classifier": classifier, "condition": condition, "n_subjects": len(items)}
            for key in METRIC_KEYS:
                out[f"{key}_mean"], out[f"{key}_std"] = mean_std([float(item[key]) for item in items])
            baseline_out.append(out)

    noisy = {
        (str(row["classifier"]), str(row["subject"])): float(row["accuracy"])
        for row in rows
        if row["condition"] == "noisy_noisy"
    }
    clean = {
        (str(row["classifier"]), str(row["subject"])): float(row["accuracy"])
        for row in rows
        if row["condition"] == "clean_clean"
    }
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in subject_aggregates:
        key = (str(row["classifier"]), str(row["condition"]), str(row["variant"]), str(row["base"]))
        grouped.setdefault(key, []).append(row)

    width_out: list[dict[str, Any]] = []
    for (classifier, condition, variant, base), items in sorted(grouped.items()):
        out: dict[str, Any] = {
            "classifier": classifier,
            "condition": condition,
            "variant": variant,
            "base": base,
            "n_subjects": len(items),
            "subjects": " ".join(sorted(str(item["subject"]) for item in items)),
            "n_checkpoints_per_subject": int(items[0]["n"]),
            "trainable_parameters": int(items[0]["trainable_parameters"]),
        }
        for key in METRIC_KEYS:
            values = [float(item[f"{key}_mean"]) for item in items]
            out[f"{key}_subject_mean"], out[f"{key}_subject_std"] = mean_std(values)
        deltas_noisy = [
            float(item["accuracy_mean"]) - noisy[(classifier, str(item["subject"]))]
            for item in items
        ]
        deltas_clean = [
            float(item["accuracy_mean"]) - clean[(classifier, str(item["subject"]))]
            for item in items
        ]
        out["delta_accuracy_vs_noisy_noisy_subject_mean"], out["delta_accuracy_vs_noisy_noisy_subject_std"] = mean_std(deltas_noisy)
        out["delta_accuracy_vs_clean_clean_subject_mean"], out["delta_accuracy_vs_clean_clean_subject_std"] = mean_std(deltas_clean)
        out["subjects_below_noisy_noisy"] = int(sum(delta < 0 for delta in deltas_noisy))
        out["subjects_above_noisy_noisy"] = int(sum(delta > 0 for delta in deltas_noisy))
        width_out.append(out)
    return baseline_out, width_out


def write_markdown(path: Path, baseline_rows: list[dict[str, Any]], width_rows: list[dict[str, Any]]) -> None:
    lines = ["# BCI IV-2a All-Subject Braindecode Downstream Evaluation", ""]
    classifier_names = ", ".join(sorted(summary["metadata"]["classifiers"]))
    lines.append(f"Classifiers: {classifier_names} from Braindecode.")
    lines.append("")
    lines.append("## Baselines")
    lines.append("")
    lines.append("| Classifier | Condition | n | Accuracy | Balanced accuracy | Kappa | Macro F1 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in baseline_rows:
        lines.append(
            f"| {row['classifier']} | {row['condition']} | {row['n_subjects']} | "
            f"{row['accuracy_mean']:.6f} +/- {row['accuracy_std']:.6f} | "
            f"{row['balanced_accuracy_mean']:.6f} +/- {row['balanced_accuracy_std']:.6f} | "
            f"{row['cohen_kappa_mean']:.6f} +/- {row['cohen_kappa_std']:.6f} | "
            f"{row['macro_f1_mean']:.6f} +/- {row['macro_f1_std']:.6f} |"
        )
    lines.append("")
    lines.append("## Denoised Conditions")
    lines.append("")
    lines.append("| Classifier | Condition | Base | Subjects | Accuracy | Delta vs noisy/noisy | Subjects below noisy/noisy |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in width_rows:
        lines.append(
            f"| {row['classifier']} | {row['condition']} | {row['base']} | {row['n_subjects']} | "
            f"{row['accuracy_subject_mean']:.6f} +/- {row['accuracy_subject_std']:.6f} | "
            f"{row['delta_accuracy_vs_noisy_noisy_subject_mean']:+.6f} +/- {row['delta_accuracy_vs_noisy_noisy_subject_std']:.6f} | "
            f"{row['subjects_below_noisy_noisy']}/{row['n_subjects']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    subjects = [subject.strip().upper() for subject in parse_csv_list(args.subjects)]
    classifiers = [classifier.strip().lower() for classifier in parse_csv_list(args.classifiers)]
    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = expand_checkpoints(args.checkpoint_glob)
    if args.max_checkpoints > 0:
        checkpoints = checkpoints[: args.max_checkpoints]

    print(f"[start] run_id={args.run_id} device={device} subjects={' '.join(subjects)} classifiers={' '.join(classifiers)} checkpoints={len(checkpoints)}", flush=True)
    rows: list[dict[str, Any]] = []

    for subject in subjects:
        train_mat = args.bci_dir / f"{subject}T.mat"
        test_mat = args.bci_dir / f"{subject}E.mat"
        print(f"[subject_start] {subject}", flush=True)
        train_clean, train_eog, y_train, train_info = load_bci2a_trials(
            train_mat,
            trial_start_sec=args.trial_start_sec,
            trial_stop_sec=args.trial_stop_sec,
            include_artifact_trials=args.include_artifact_trials,
            eog_index=args.eog_index,
        )
        test_clean, test_eog, y_test, test_info = load_bci2a_trials(
            test_mat,
            trial_start_sec=args.trial_start_sec,
            trial_stop_sec=args.trial_stop_sec,
            include_artifact_trials=args.include_artifact_trials,
            eog_index=args.eog_index,
        )
        train_noisy, _train_artifact, train_noise_info = make_noisy_epochs(
            train_clean,
            train_eog,
            seed=args.seed + 1000,
            snr_min_db=args.snr_min_db,
            snr_max_db=args.snr_max_db,
        )
        test_noisy, _test_artifact, test_noise_info = make_noisy_epochs(
            test_clean,
            test_eog,
            seed=args.seed,
            snr_min_db=args.snr_min_db,
            snr_max_db=args.snr_max_db,
        )

        signals = {
            "clean": (
                bandpass_epochs(train_clean, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz),
                bandpass_epochs(test_clean, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz),
            ),
            "noisy": (
                bandpass_epochs(train_noisy, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz),
                bandpass_epochs(test_noisy, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz),
            ),
        }

        for classifier_name in classifiers:
            print(f"[baseline_start] subject={subject} classifier={classifier_name}", flush=True)
            baseline_specs = [
                ("clean_clean", "clean", "clean"),
                ("clean_noisy", "clean", "noisy"),
                ("noisy_noisy", "noisy", "noisy"),
            ]
            for condition, train_key, test_key in baseline_specs:
                train_epochs, _ = signals[train_key]
                _, test_epochs = signals[test_key]
                row = fit_eval_row(
                    classifier_name=classifier_name,
                    subject=subject,
                    condition=condition,
                    train_epochs=train_epochs,
                    y_train=y_train,
                    test_epochs=test_epochs,
                    y_test=y_test,
                    device=device,
                    seed=args.seed,
                    args=args,
                )
                rows.append(row)
                print(f"[baseline] subject={subject} classifier={classifier_name} condition={condition} acc={row['accuracy']:.6f}", flush=True)

        for checkpoint_path in checkpoints:
            model, cfg, n_params = load_checkpoint_model(checkpoint_path, device)
            base = cfg.get("base", "")
            variant = str(cfg.get("variant") or cfg.get("model_type") or f"base{base}")
            train_seed = checkpoint_seed(checkpoint_path, cfg)
            train_denoised = denoise_epochs(model, train_noisy, device=device, batch_size=args.denoiser_batch_size)
            test_denoised = denoise_epochs(model, test_noisy, device=device, batch_size=args.denoiser_batch_size)
            train_denoised = bandpass_epochs(train_denoised, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz)
            test_denoised = bandpass_epochs(test_denoised, FS_MODEL, args.bandpass_low_hz, args.bandpass_high_hz)
            train_clean_bp, _test_clean_bp = signals["clean"]

            for classifier_name in classifiers:
                strict_seed = args.seed
                matched_seed = args.seed
                strict_row = fit_eval_row(
                    classifier_name=classifier_name,
                    subject=subject,
                    condition="clean_denoised",
                    train_epochs=train_clean_bp,
                    y_train=y_train,
                    test_epochs=test_denoised,
                    y_test=y_test,
                    device=device,
                    seed=strict_seed,
                    args=args,
                    variant=variant,
                    base=base,
                    train_seed=train_seed,
                    trainable_parameters=n_params,
                    checkpoint=str(checkpoint_path),
                )
                matched_row = fit_eval_row(
                    classifier_name=classifier_name,
                    subject=subject,
                    condition="denoised_denoised",
                    train_epochs=train_denoised,
                    y_train=y_train,
                    test_epochs=test_denoised,
                    y_test=y_test,
                    device=device,
                    seed=matched_seed,
                    args=args,
                    variant=variant,
                    base=base,
                    train_seed=train_seed,
                    trainable_parameters=n_params,
                    checkpoint=str(checkpoint_path),
                )
                rows.extend([strict_row, matched_row])
                print(
                    f"[result] subject={subject} classifier={classifier_name} base={base} seed={train_seed} "
                    f"clean_denoised_acc={strict_row['accuracy']:.6f} denoised_denoised_acc={matched_row['accuracy']:.6f}",
                    flush=True,
                )

        print(f"[subject_done] {subject}", flush=True)

    subject_aggregates = aggregate_checkpoint_rows(rows)
    baseline_aggregate, width_aggregate = aggregate_across_subjects(rows, subject_aggregates)
    summary = {
        "run_id": args.run_id,
        "subjects": subjects,
        "classifiers": classifiers,
        "info": {
            "epochs": args.epochs,
            "classifier_batch_size": args.classifier_batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "bandpass_low_hz": args.bandpass_low_hz,
            "bandpass_high_hz": args.bandpass_high_hz,
            "snr_min_db": args.snr_min_db,
            "snr_max_db": args.snr_max_db,
            "class_names": CLASS_NAMES,
        },
        "rows": rows,
        "subject_aggregates": subject_aggregates,
        "baseline_aggregate": baseline_aggregate,
        "width_aggregate": width_aggregate,
    }
    (args.output_dir / "bci2a_braindecode_downstream_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "bci2a_braindecode_downstream_rows.csv", rows)
    write_csv(args.output_dir / "bci2a_braindecode_downstream_subject_aggregates.csv", subject_aggregates)
    write_csv(args.output_dir / "bci2a_braindecode_downstream_baseline_aggregate.csv", baseline_aggregate)
    write_csv(args.output_dir / "bci2a_braindecode_downstream_width_aggregate.csv", width_aggregate)
    write_markdown(args.output_dir / "bci2a_braindecode_downstream_summary.md", baseline_aggregate, width_aggregate)
    print(f"[written] {args.output_dir / 'bci2a_braindecode_downstream_summary.md'}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
