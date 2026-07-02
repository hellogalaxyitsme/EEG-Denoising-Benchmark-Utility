# EEG Denoising Benchmark Utility

This repository contains the analysis code for controlled-capacity EEG denoising benchmarks, baseline retraining, downstream BCI utility evaluation, deployment profiling, and statistical aggregation. Large datasets, trained checkpoints, and raw run directories are not included

## Repository Layout

```text
src/eeg_denoise_benchmark/   Core models, losses, metrics, data loaders, checkpoints
scripts/                     Training, evaluation, downstream BCI, profiling, statistics
configs/templates/           Minimal local-path templates for reproducible runs
data/README.md               Dataset preparation notes and expected array layout
results/README.md            Expected location for regenerated summaries
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

Install the optional downstream dependencies when running Braindecode classifiers:

```bash
python -m pip install ".[bci]"
```

The original experiments were run with PyTorch/CUDA on an NVIDIA RTX A5000 workstation. 

## Datasets

- EEGDenoiseNet synthetic splits for EOG and EMG.
- The Mixed-1M corpus generated from disjoint EEGDenoiseNet source pools.
- BCI Competition IV-2a and IV-2b files for zero-shot and downstream experiments.

See `data/README.md` for expected split keys and directory conventions. 

## Common Commands

Train one controlled-backbone width after editing paths in a config:

```bash
python scripts/train_denoiser.py configs/templates/controlled_backbone_eegdenoisenet.yaml \
  --set data=/path/to/eegdenoisenet/eog \
  --set output_dir=runs/eog_base6_seed42 \
  --set base=6 \
  --set seed=42
```

Evaluate a checkpoint on a held-out synthetic split:

```bash
python scripts/evaluate_checkpoint.py \
  --checkpoint runs/eog_base6_seed42/best.pt \
  --data /path/to/eegdenoisenet/eog \
  --output runs/eog_base6_seed42/eval.json
```

Run controlled classical baselines:

```bash
python scripts/evaluate_classical_eegdenoisenet_baselines.py \
  --eog-data /path/to/eegdenoisenet/eog \
  --emg-data /path/to/eegdenoisenet/emg \
  --output-dir runs/classical_baselines
```

Generate the Mixed-1M corpus from downloaded EEGDenoiseNet source pools:

```bash
python scripts/generate_mixed1m_corpus.py \
  --clean-eeg /path/to/eegdenoisenet/clean_eeg.npy \
  --eog /path/to/eegdenoisenet/eog.npy \
  --emg /path/to/eegdenoisenet/emg.npy \
  --output-dir data/mixed1m \
  --seed 42
```

Profile deployment cost:

```bash
python scripts/profile_model_complexity.py --output-dir runs/complexity_profile
```

## Main Experiment Scripts

- `scripts/train_denoiser.py`: supervised denoiser training from flat YAML/JSON configs.
- `scripts/generate_mixed1m_corpus.py`: deterministic Mixed-1M construction from EEGDenoiseNet source pools.
- `scripts/evaluate_bci_zero_shot_widths.py`: IV-2a/IV-2b zero-shot reconstruction evaluation.
- `scripts/evaluate_mixed1m_stratified.py`: Mixed-1M SNR/artifact stratification.
- `scripts/evaluate_classical_eegdenoisenet_baselines.py`: zero-parameter and tiny lower anchors.
- `scripts/profile_model_complexity.py`: parameter, FLOP, size, latency, throughput, and memory profiling.
- `scripts/evaluate_bci2a_all_subjects_downstream_csp_lda.py`: all-subject CSP+LDA downstream utility.
- `scripts/evaluate_bci2a_all_subjects_braindecode_classifiers.py`: EEGNet, ShallowFBCSPNet, Deep4Net, EEGConformer.
- `scripts/evaluate_bci2a_downstream_contamination_types_csp_lda.py`: EOG/EMG/mixed artifact robustness.
- `scripts/evaluate_bci2a_real_downstream_csp_lda.py`: real non-synthetic IV-2a downstream evaluation.
- `scripts/evaluate_bci2a_csp_lda_calibration_sensitivity.py`: post-denoising calibration controls.
- `scripts/evaluate_bci2a_multichannel_denoiser_csp_control.py`: multichannel covariance-aware control.
- `scripts/evaluate_bci2a_neural_convergence_audit.py`: extended neural decoder training audit.

## Reproducibility Notes

The primary analysis uses seeds `42, 43, 44, 45, 46` for n=5 width sweeps and `42, 43, 44` for exploratory ablations and controlled baselines unless stated otherwise. Scripts write JSON/CSV summaries under the requested `output_dir`; statistical aggregation scripts consume those summaries for reported analyses.
