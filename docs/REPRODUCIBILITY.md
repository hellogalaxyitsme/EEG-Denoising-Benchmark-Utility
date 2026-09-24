# Reproducibility guide

This repository provides reusable training, evaluation, and statistical-analysis software for controlled EEG denoising studies. Recordings, checkpoints, and run outputs are not distributed. Re-running an analysis therefore requires authorized source data, the relevant checkpoint set, and the result directories expected by each aggregation script.

## Primary and secondary reconstruction comparisons

The controlled EEGDenoiseNet sweep uses five available training seeds (42 through 46) for two primary correlation-coefficient (CC) contrasts: EOG base8-to-base16 CC and EMG base6-to-base8 CC. Both are paired across all five seeds. Other metric and adjacent-width comparisons use the three matched seeds 42, 43, and 44 and are secondary sensitivity analyses.

scripts/analyze_capacity_diminishing_returns.py computes every metric and comparison available in its input and uses the intersection of seed labels for each output row. It does not label an output as published primary or secondary. For the retained primary rows, require n_matched_seeds = 5 and matched_seeds = "42 43 44 45 46"; for the secondary EEGDenoiseNet rows, require n_matched_seeds = 3 and matched_seeds = "42 43 44". Retain only the designated rows below.

Use two input passes. First collect the raw controlled summaries, adjusting the glob to the directory containing the EEGDenoiseNet width-sweep runs:

    python scripts/collect_run_summaries.py "runs/*/summary.json" --output-csv results/controlled_runs.csv

Create selected inputs with this standard-library script, saved for example as results/filter_capacity_inputs.py:

    import csv, re, sys
    src, dst, mode = sys.argv[1:]
    def base(row):
        match = re.search(r"base(\d+)", row["run_dir"] + " " + row["output_dir"])
        return int(match.group(1)) if match else None
    with open(src, newline="", encoding="utf-8") as inp:
        rows = list(csv.DictReader(inp))
    keep = []
    for row in rows:
        task = row["data"].lower()
        seed = int(row["seed"])
        if mode == "primary":
            selected = seed in {42, 43, 44, 45, 46} and (
                ("eog contaminated" in task and base(row) in {8, 16}) or
                ("emg contaminated" in task and base(row) in {6, 8}))
        else:
            selected = seed in {42, 43, 44}
        if selected:
            keep.append(row)
    with open(dst, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(keep)

    python results/filter_capacity_inputs.py results/controlled_runs.csv results/capacity_primary.csv primary
    python results/filter_capacity_inputs.py results/controlled_runs.csv results/capacity_secondary.csv secondary
    python scripts/analyze_capacity_diminishing_returns.py --width-index results/capacity_primary.csv --bci-zero-shot results/zero_shot/zero-shot_channel_reconstruction_rows.csv --output-dir results/capacity_primary --run-id local_primary
    python scripts/analyze_capacity_diminishing_returns.py --width-index results/capacity_secondary.csv --bci-zero-shot results/zero_shot/zero-shot_channel_reconstruction_rows.csv --output-dir results/capacity_secondary --run-id local_secondary

From the primary run, retain only the synthetic EOG CC base8-to-base16 and EMG CC base6-to-base8 rows. From the secondary run, retain all other metric/comparison entries. The BCI zero-shot argument is required by the current capacity CLI; it is not part of either published primary EEGDenoiseNet CC contrast.

## Inferential units and multiplicity

For downstream BCI effects, the independent unit is the human subject (n=9); Sleep-EDF downstream effects likewise use the subject (n=75). Contamination realizations, denoiser checkpoints, channels, and neural-classifier initializations are technical repetitions. They are averaged within subject or represented as descriptive variability, never pooled as additional human observations.

The BCI IV-2a primary downstream effects use two-sided one-sample t-tests of subject-level denoised-minus-reference accuracy deltas. Their 95% confidence intervals are unadjusted subject-bootstrap intervals from 10,000 subject-only resamples, and exact Wilcoxon signed-rank tests are non-parametric sensitivity analyses. BH-FDR is applied across the 27-effect primary family: three base16 CSP+LDA artifact-recipe effects plus 24 neural effects (four classifiers by two widths by three recipes). The three CSP+LDA effects meet BH-FDR, but none meets Holm adjustment calculated from the same primary-test p-values. These are distinct multiplicity summaries, not interchangeable decisions.

The same-condition IV-2a metric-utility analysis is a separate BH-FDR family of 75 classifier by artifact by metric subject-intercept tests. scripts/analyze_hierarchical_downstream_stats.py writes both within-family BH and Holm columns, plus an all-effect BH column; scripts/analyze_metric_utility.py corrects the association tests separately.

Sleep-EDF uses two further, separate families specified in Supplementary Section S6.1. The 18 synthetic/real model-condition downstream effects use lower-tail Wilcoxon signed-rank tests of subject-level balanced-accuracy deltas and BH-FDR. The 45 model by metric association tests use one slope per subject, two-sided Wilcoxon signed-rank tests against zero, and a separate BH-FDR adjustment. Do not combine either Sleep-EDF family with the BCI effect or association families.

## Distinct BCI protocols

The downstream IV-2a scripts use their defaults --trial-start-sec 2.5 and --trial-stop-sec 6.0, measured from the stored trial-start index. Because the cue is 2 s after trial start, this is the 0.5-to-4.0 s post-cue interval. The zero-shot reconstruction script is a separate protocol: it uses non-overlapping 2 s windows (512 samples after resampling to 256 Hz), with low- and high-EOG-energy pools selected from disjoint quantiles. Do not substitute one protocol's windowing for the other.

## Dataset boundaries and uncertainty

EEGDenoiseNet reference targets are benchmark reference EEG, not direct artifact-free neural ground truth. In its EMG preparation, EMG artifact segments are split across train, validation, and test, but clean EEG sources can recur across those splits. That benchmark result consequently does not establish generalization to unseen clean EEG sources.

Mixed-1M is constructed differently. scripts/generate_mixed1m_corpus.py splits each clean EEG, EOG, and EMG source pool before mixture generation, producing source-disjoint train/validation/test pools. It deliberately reuses represented sources within each split to generate many mixtures. Treat mixture rows as conditional Monte-Carlo draws from those fixed source pools, not as biologically independent observations. This release does not implement source-cluster confidence intervals for Mixed-1M; do not infer such intervals from row-level mixture counts.

Sleep-EDF uses within-subject cross-recording training and testing: the first available Sleep Cassette recording supplies training and the second supplies held-out testing. It does not establish generalization to unseen subjects. In the published comparison, controlled widths use Mixed-1M-trained checkpoints and external architectures use EEGDenoiseNet EOG checkpoints. This dataset and checkpoint provenance means those comparisons cannot isolate architecture alone.

## Resource and export claims

scripts/profile_model_complexity.py measures parameter count, operation estimates, serialized state size, and timing on the hardware and device options supplied to that run. scripts/evaluate_onnx_export_compatibility.py checks export and numerical agreement with ONNX Runtime for selected checkpoints. These scripts support measured resource and format-compatibility statements only; they do not establish performance on an edge device.

Run the scripts only with data and checkpoints that you are licensed and authorized to use. Each evaluation script writes its requested scientific summary to its output directory.
