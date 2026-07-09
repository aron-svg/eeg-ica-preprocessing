# EEG Signal Quality — Preprocessing Pipeline

An [MNE-Python](https://mne.tools/) pipeline that cleans a raw EEG/EOG/ECG recording (unit correction, bad-channel handling, referencing, filtering, ICA-based artifact removal) and saves a preprocessed `.fif` file. This project **only handles preprocessing** — model training/evaluation lives in a separate ML pipeline.

## What the pipeline does

Running `src/main.py` executes the following steps, in order, on the recording in `data/HZO004/recording.fif`:

1. **Load the recording** (`mne.io.read_raw_fif`).
2. **Fix a unit mismatch**: the EEG/EOG/ECG channels in the source `.fif` are declared in volts but actually contain microvolt-scale values, so they're rescaled by `1e-6`. (`misc` channels — temp/resp/gsr/timestamp — are left untouched.)
3. **Detect and interpolate bad EEG channels** (`mne.preprocessing.annotate_amplitude`): a channel is marked bad if it's flat/disconnected or exceeds the artifact amplitude threshold for more than `BAD_CHANNEL_PERCENT` of the recording, then interpolated from its neighbours. This must happen *before* referencing, otherwise a single bad channel would bias the average reference for every other channel.
4. **Re-reference to the common average** (`raw.set_eeg_reference("average")`) across EEG channels.
5. **Visual check**: a blocking scrollable plot of the ECG/EOG channels, so you can eyeball heartbeats/blinks before continuing.
6. **Filter** a copy of the data: high-pass at `LOW_FREQUENCY` (removes slow drift), low-pass at `HIGH_FREQUENCY` (removes high-frequency noise), notch at `NOTCH_FREQUENCY` (removes powerline interference).
7. **Fit ICA** (`N_COMPONENTS` components) on the filtered copy, logging the variance explained by each component.
8. **Inspect components**: plots ICA sources/topographies; in manual mode, also plots per-component properties and a before/after overlay for the components you flag.
9. **Exclude artifact components**, either:
   - **Manual mode** (`MANUAL_MODE = True`): you type in the indices of the components to exclude, based on the plots from step 8.
   - **Automatic mode** (`MANUAL_MODE = False`): `ica.find_bads_ecg` / `ica.find_bads_eog` pick the components correlated with the ECG/EOG channels.
10. **Apply the ICA cleaning** to the filtered signal and show a before/after comparison: an `ica.plot_overlay` (RMS/GFP on the same axes — the clearest way to see the effect) plus two scrollable EEG windows titled *"Raw (before ICA)"* / *"Preprocessed (after ICA)"*.
11. **Save** the final cleaned, filtered, re-referenced signal to `output/preprocessed_data.fif`.

The returned/saved data is a continuous `Raw` object (not epoched) — segmenting into fixed-length windows and labeling is left to the downstream ML pipeline.

## Configuration

All tunable parameters live in `src/config.py`:

| Constant | Default | Meaning |
|---|---|---|
| `DATA_PATH` | `data/HZO004/` | Folder containing `recording.fif` |
| `OUTPUT_PATH` | `output/` | Where the preprocessed `.fif` is saved |
| `MANUAL_MODE` | `True` | `True` = you pick which ICA components to exclude; `False` = automatic ECG/EOG correlation |
| `N_COMPONENTS` | `30` | Number of ICA components to compute |
| `LOW_FREQUENCY` | `1.0` Hz | High-pass cutoff (drift removal) |
| `HIGH_FREQUENCY` | `45.0` Hz | Low-pass cutoff (noise removal) |
| `NOTCH_FREQUENCY` | `50.0` Hz | Powerline notch filter |
| `ARTEFACT_MAX` | `200e-6` V | Peak-to-peak threshold used both for ICA-fit epoch rejection and bad-channel "noisy" detection |
| `FLAT_THRESHOLD` | `1e-6` V | Peak-to-peak threshold below which a channel is considered flat/disconnected |
| `BAD_CHANNEL_PERCENT` | `5` | % of the recording a channel may violate the thresholds above before being marked bad |

## Expected input data

`recording.fif` should contain, at minimum:
- **EEG** channels (32 in the reference recording, 10-20 style names like `Fp1`, `Cz`, ...) with valid electrode positions (needed for interpolation/topomaps).
- **ECG** channels matching `ECG0[1-3]`.
- **EOG** channels matching `EOG0[1-5]`.
- Optionally `misc` channels (e.g. `temp`, `resp`, `gsr`, `timestamp`) — left untouched by the unit fix and filtering.

## Project structure

```
eeg-signal-quality/
├── src/
│   ├── main.py              # Entry point: runs the pipeline, saves the output .fif
│   ├── ICA.py                # The preprocessing pipeline itself (see steps above)
│   ├── config.py             # All tunable parameters and paths
│   ├── logger_init.py        # Logger setup
│   ├── logging_config.py     # Colored console formatter
│   └── logger_config.yaml    # Logging handlers (console + rotating file)
├── data/HZO004/               # Input recording (gitignored, not versioned)
├── output/                    # Preprocessed .fif output (gitignored)
├── logs/                      # Rotating log files (gitignored)
├── pyproject.toml             # Dependencies (mne, numpy, pandas, scikit-learn, ...)
└── .vscode/                   # Editor/debug configuration
```

## Setup

```bash
uv sync
```

This creates a virtual environment and installs everything listed in `pyproject.toml` (notably `mne`, `numpy`, `scikit-learn`).

## Running the pipeline

```bash
python src/main.py
```

Notes:
- The pipeline is **interactive**: several plot windows are blocking (`block=True`) and, in manual mode, it prompts on stdin for the ICA components to exclude. It's meant to be run and watched, not used headlessly/in a batch over many recordings as-is.
- Logs go to both the console (colored) and `logs/eeg_signal_preprocessing.log` (rotating, 20 MB × 10 backups).
- The final output is written to `output/preprocessed_data.fif`, overwriting any previous run.

## Known limitations / possible next steps

- Processes a single recording per run; no batch/multi-subject support yet.
- Output is a continuous signal, not epoched — fine since epoching belongs to the separate ML pipeline, but worth knowing if you consume `output/preprocessed_data.fif` directly.
- Manual mode requires a display and a human in the loop; there's no way to run it unattended.
- Only ECG/EOG-correlated ICA components are targeted automatically; muscle artifacts or electrode pops aren't specifically detected beyond the bad-channel/reject-threshold checks.
