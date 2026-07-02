# Data Layout

Dataset files are not included in this repository.

The synthetic loaders expect each split directory to contain NumPy arrays for the contaminated signal, clean target, artifact target, and SNR/normalization metadata. The canonical split names are `train`, `val`, and `test`.

The training and reconstruction scripts use paths supplied in the config under `data`. The downstream BCI scripts accept explicit paths to BCI Competition IV-2a/IV-2b files and checkpoint directories through command-line arguments.

Recommended local layout:

```text
data/
  eegdenoisenet/
    eog/
      train/
      val/
      test/
    emg/
      train/
      val/
      test/
  mixed1m/
    train/
    val/
    test/
  bci_iv_2a/
  bci_iv_2b/
```

Keep raw datasets outside git. The `.gitignore` blocks common EEG and NumPy data formats.

## Mixed-1M Corpus Generation

`scripts/generate_mixed1m_corpus.py` reconstructs the Mixed-1M corpus deterministically from the public EEGDenoiseNet source pools. The expected source pools are clean EEG segments, EOG artifact segments, and EMG artifact segments from EEGDenoiseNet. The source arrays may be `.npy`, `.npz`, or `.mat`; if more than one numeric 2D array exists, pass `--clean-key`, `--eog-key`, or `--emg-key` explicitly.

The generation command used by default is:

```bash
python scripts/generate_mixed1m_corpus.py \
  --clean-eeg /path/to/eegdenoisenet/clean_eeg.npy \
  --eog /path/to/eegdenoisenet/eog.npy \
  --emg /path/to/eegdenoisenet/emg.npy \
  --output-dir data/mixed1m \
  --seed 42
```

The canonical source pools are:

```text
clean EEG: 4514 segments x 512 samples, 256 Hz
EOG:       3400 segments x 512 samples, 256 Hz
EMG:       5598 segments x 1024 samples, 512 Hz, downsampled to 256 Hz
```

The script performs disjoint source-level splits before mixing so clean EEG, EOG, and EMG source indices do not cross train/validation/test boundaries. It writes:

```text
data/mixed1m/
  meta.json
  train/chunk_0000.npz ... chunk_0079.npz
  val/chunk_0000.npz   ... chunk_0009.npz
  test/chunk_0000.npz  ... chunk_0009.npz
```

Each chunk contains `Y`, `X`, `A`, `sigma_y`, `snr_db`, `lambda`, `recipe_id`, `eog_used`, `emg_used`, `line_used`, `ecg_used`, `elec_used`, and source-index metadata. `Y`, `X`, and `A` are normalized by `std(Y_raw)` after artifact scaling.

The recipe probabilities are:

```text
EOG                    0.25
EMG                    0.25
EOG+EMG                0.20
EMG+LINE               0.10
EOG+LINE               0.10
EOG+EMG+LINE           0.05
EOG+EMG+LINE+ECG       0.05
```

The target SNR distribution is a two-part mixture: 70% uniform over [-7, 2] dB and 30% uniform over [-12, -7] dB. EOG polarity is flipped with probability 0.5, EOG is pre-scaled by U[0.7, 1.3], EMG is pre-scaled by U[0.6, 1.5], electrode noise is added with probability 0.35, line noise is 50 Hz with probability 0.85 and 60 Hz otherwise, and ECG is generated synthetically as a QRS-dominated pulse train with optional T-wave.
