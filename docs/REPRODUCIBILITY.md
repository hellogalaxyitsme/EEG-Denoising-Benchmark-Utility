# Reproducibility guide

This release contains code and aggregate result tables for the capacity-controlled EEG denoising evaluation. It excludes subject recordings, trained checkpoints, raw run directories, private correspondence, and local-machine information.

## Analysis map

| Question | Evaluation/statistics scripts | Published aggregate data |
|---|---|---|
| Controlled capacity and matched-width increments | `analyze_a5_capacity_diminishing_returns.py`, `analyze_a5_a7_subject_capacity_sensitivity.py` | `fig02_*.csv`, `supp_s1_*.csv` |
| Source reuse | `analyze_a11_mixed1m_source_reuse.py` | `supp_s3_*.csv` |
| BCI IV-2a hierarchical effects | `evaluate_bci2a_all_subjects_downstream_csp_lda.py`, `evaluate_a3_classifier_seed_uncertainty.py`, `analyze_hierarchical_downstream_stats.py` | `fig03_*.csv` |
| Same-condition metric--utility analysis | `analyze_a4_same_bci_metric_utility.py` | `fig04_*.csv`, `supp_s4_metric_utility_matrix.csv` |
| BCI transfer and controls | `evaluate_a7_bci_zero_shot_all_subjects.py`, `evaluate_a8_bci2b_downstream_replication.py`, `evaluate_a9_bci_domain_shift_single_channel.py`, `evaluate_a10_spatial_eog_projection_csp_lda.py`, `evaluate_a12_real_bci2a_eog_stratified_downstream.py` | `supp_s5_*.csv`, `supp_s7_*.csv`, `supp_s8_*.csv` |
| Sleep-EDF | `evaluate_sleep_edf_formal.py` | `fig05_*.csv`, `supp_s9_*.csv` |
| Export compatibility | `evaluate_b2_onnx_export_compatibility.py` | reported supplementary table |

## Released-data plotting

The released plotting path uses only the aggregate CSVs and does not train models or recompute scientific inference:

```bash
python scripts/audit_manuscript_figure_data.py
python scripts/plot_released_figures.py --output-dir results/released_plots
```

This writes inspection plots for Figures 2--5 from `results/manuscript_figure_data/`. The `make_fig*` and `make_supp*` scripts plot study run summaries for authorized users who have the licensed recordings, checkpoints, and run-summary inputs; they are not needed for the released-data plotting path. Complete numeric inputs for manuscript tables are in `results/manuscript_tables/`.

## Statistical guardrails

- Human subjects are the independent units for downstream analyses.
- Confidence intervals use subject-only bootstrap resampling where reported.
- The primary BCI hierarchical family has 27 effects; the metric--utility family has 75 tests. Exploratory width analyses are separate.
- Exact Wilcoxon analyses are sensitivity tests for the BCI hierarchical effects. Sleep-EDF uses directional Wilcoxon tests with BH-FDR correction.
- Capacity summaries with five training seeds are descriptive. Formal adjacent-width tests use the matched seeds 42, 43, and 44 and predefined practical margins.
- Reconstruction targets in semi-synthetic and low-artifact-reference settings are operational references, not direct neural ground truth.

Run scripts only with data that you are licensed and authorized to use. Do not infer biological sample size from the number of generated mixtures or technical repetitions.
