# Reproducibility guide

This repository provides reusable training, evaluation, and statistical-analysis software for controlled EEG denoising studies. Recordings, checkpoints, and run outputs are not distributed.

## Analysis principles

- Treat human participants as inferential units for downstream analyses.
- Treat contamination realizations, checkpoints, channels, and classifier initializations as technical repetitions.
- Report reconstruction and downstream outcomes separately; their association is task- and model-dependent.
- Interpret capacity increments with explicit metric-specific tolerances and their full uncertainty intervals.

Run the scripts only with data and checkpoints that you are licensed and authorized to use. Each evaluation script writes its requested scientific summary to its output directory.
