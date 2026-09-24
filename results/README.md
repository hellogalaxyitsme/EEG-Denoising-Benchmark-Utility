# Results

Analysis commands write user-generated CSV/JSON summaries to this directory or another selected output location. No recordings, checkpoints, or run archives are included in this release.

For reproducible aggregation, retain the generated input CSVs alongside summaries. Capacity outputs record their matched seed intersection; only the EOG base8-to-base16 CC and EMG base6-to-base8 CC comparisons with seeds `42`--`46` are primary. Other matched three-seed (`42`--`44`) metric comparisons are secondary.

Downstream summaries must retain the subject-level inferential unit and their correction-family columns. The BCI IV-2a 27-effect primary family, the 75-test same-condition metric--utility family, and the Sleep-EDF model-condition family are separate analyses. Timing and ONNX summaries are measurements from the reported run environment, not edge-device benchmarks.
