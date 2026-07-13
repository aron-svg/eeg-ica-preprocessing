# EEG ICA PREPROCESSING — Preprocessing Pipeline

An [MNE-Python](https://mne.tools/) pipeline that cleans a raw EEG/EOG/ECG recording (unit correction, filtering, bad-channel/segment detection, referencing, ICA-based artifact removal) and saves a preprocessed `.fif` file. This project **only handles preprocessing** — model training/evaluation lives in a separate ML pipeline.

## What the pipeline does

Running `src/main.py` executes the following steps, in order, on the recording in `DATA_PATH/recording.fif`. The order follows the standard EEG preprocessing sequence (filter → flag noisy segments/channels → reference → ICA → interpolate bad channels last), rather than interpolating bad channels immediately on detection:

1. **Load the recording** (`mne.io.read_raw_fif`).
2. **Fix a unit mismatch**: the EEG/EOG/ECG channels in the source `.fif` are declared in volts but actually contain microvolt-scale values, so they're rescaled by `1e-6`. (`misc` channels — temp/resp/gsr/timestamp — are left untouched.)
3. **Filter**: high-pass at `LOW_FREQUENCY` (removes slow drift), low-pass at `HIGH_FREQUENCY` (removes high-frequency noise), notch at `NOTCH_FREQUENCY` (removes powerline interference). This becomes the signal used for every step from here on.
4. **Detect (but don't yet interpolate) bad EEG channels**, using two complementary checks:
   - `mne.preprocessing.annotate_amplitude`: flags a channel "bad" if it's flat/disconnected or has large consecutive-sample jumps for more than `BAD_CHANNEL_PERCENT` of the recording.
   - A variance-based check: flags a channel whose overall amplitude is a statistical outlier (modified z-score above `VARIANCE_Z_THRESHOLD`) relative to the other channels. This catches broadband contamination (e.g. muscle tension) that `annotate_amplitude`'s consecutive-sample-jump criterion can't see, since that kind of contamination raises a channel's overall variance without producing large jumps between individual samples.

   Both checks run on an **unfiltered copy** of the signal, kept aside just for this step: `FLAT_THRESHOLD`/`ARTEFACT_MAX` are calibrated for the natural sample-to-sample jitter of the raw, full-bandwidth signal, and the band-pass filter from step 3 removes exactly that jitter — running the same checks on the filtered signal instead makes a perfectly healthy channel look "flat" almost everywhere. Detected channels are marked in `raw.info["bads"]` but **not interpolated yet** — interpolating now would either bias the average reference (step 5) or, if excluded from it, still leave the ICA fit (step 6) without that channel's information. Deferring interpolation to the very end (step 10) means bad channels are excluded from both the reference and the ICA fit, and get reconstructed from already ICA-cleaned neighbours instead of artifact-contaminated ones.
5. **Re-reference to the common average** (`raw.set_eeg_reference("average")`). MNE automatically excludes channels marked in `info["bads"]` from this average.
6. **Fit ICA** (`N_COMPONENTS` components). MNE automatically excludes `info["bads"]` channels from the fit too, so a still-uninterpolated bad channel can't corrupt the decomposition. Variance explained per component is logged (restricted to the channels actually used in the fit — see note below).
7. **Inspect components**: plots ICA sources/topographies; in manual mode, also plots per-component properties for the components you flag.
8. **Exclude artifact components**, either:
   - **Manual mode** (`MANUAL_MODE = True`): you type in the indices of the components to exclude, based on the plots from step 7.
   - **Automatic mode** (`MANUAL_MODE = False`): `ica.find_bads_ecg` / `ica.find_bads_eog` (cardiac/ocular correlation) plus `ica.find_bads_muscle` (a distinct spectral/spatial check for muscle-related components, which correlation with ECG/EOG can't catch). All three use `measure="correlation"` — thresholding directly on the correlation coefficient rather than the default z-score across components, which becomes statistically unreliable when `N_COMPONENTS` is small (a genuinely strong correlation can fail to ever cross the z-score threshold). `find_bads_muscle` has a known intermittent internal MNE failure (see Known limitations) and is wrapped in a `try/except` so it doesn't take down an otherwise-expensive ICA fit.
9. **Apply the ICA cleaning** and show a before/after comparison: an `ica.plot_overlay` (RMS/GFP on the same axes) plus two scrollable EEG windows titled *"Raw (before ICA)"* / *"Preprocessed (after ICA)"*.
10. **Interpolate the bad channels** marked in step 4, now that ICA has cleaned the other channels — so they're reconstructed from clean neighbours instead of the original artifact-contaminated ones.
11. **Save** the final cleaned, filtered, re-referenced, fully-interpolated signal to `OUTPUT_PATH/preprocessed_data.fif`.

The returned/saved data is a continuous `Raw` object (not epoched) — segmenting into fixed-length windows and labeling is left to the downstream ML pipeline.

## Configuration

All tunable parameters live in `src/config.py`:

| Constant | Default | Meaning | Why this value |
|---|---|---|---|
| `DATA_PATH` | `data/HZO031/` | Folder containing `recording.fif` | — |
| `OUTPUT_PATH` | `output/HZO031_test/` | Where the preprocessed `.fif` is saved | — |
| `MANUAL_MODE` | `True` | `True` = you pick bad channels and ICA components yourself; `False` = automatic detection for both | Manual inspection can catch artifacts (e.g. a channel dominated by muscle tension) that the automatic checks may miss or only partially clean; automatic mode is reproducible and doesn't need a human in the loop, at the cost of being blind to anything its specific detectors weren't built for |
| `N_COMPONENTS` | `10` | Number of ICA components to compute | Kept low relative to the channel count for faster iteration; see Known limitations for the trade-offs of a low count (weaker automatic z-score-based detection, more variance left unmodeled) |
| `LOW_FREQUENCY` | `1.0` Hz | High-pass cutoff (drift removal) | Standard ICA preprocessing recommendation (e.g. Makoto Miyakoshi/EEGLAB pipelines) — slow drift dominates variance and degrades the ICA decomposition if left in |
| `HIGH_FREQUENCY` | `45.0` Hz | Low-pass cutoff (noise removal) | Keeps the conventional EEG band while staying below the 50 Hz notch |
| `NOTCH_FREQUENCY` | `50.0` Hz | Powerline notch filter | Regional mains frequency (50 Hz) |
| `ARTEFACT_MAX` | `200e-6` V | Peak-to-peak/consecutive-sample-jump threshold, used both for ICA-fit epoch rejection and the `annotate_amplitude` "noisy" criterion | Empirically calibrated on the reference recordings: large enough that normal EEG/blink activity doesn't trigger it constantly, small enough to catch genuine large-amplitude artifacts (e.g. a single spike observed at ~200000 µV in one recording, versus a typical channel std of a few tens of µV) |
| `FLAT_THRESHOLD` | `1e-6` V | Consecutive-sample-difference threshold below which a channel is considered flat/disconnected | Must be evaluated on the **unfiltered** signal (see step 4 above) — on filtered data, a healthy channel's natural sample-to-sample difference already drops below this, which would flag almost every channel as flat |
| `BAD_CHANNEL_PERCENT` | `5` | % of the recording a channel may violate the thresholds above before being marked bad | Standard `annotate_amplitude` convention; validated on a reference recording where the noisiest genuinely-healthy channel sat at ~1.2%, leaving headroom before 5% without being lax |
| `VARIANCE_Z_THRESHOLD` | `3.5` | Modified z-score cutoff for the variance-based bad-channel check | Iglewicz & Hoaglin (1993) convention for the MAD-based modified z-score (3.5, vs. 3 for a classic mean/std z-score); validated empirically — on a reference recording, genuinely contaminated channels scored z > 4.9 while the next-highest (healthy) channel scored z < 2.5 |

## Expected input data

`recording.fif` should contain, at minimum:
- **EEG** channels (32 in the reference recordings, 10-20 style names like `Fp1`, `Cz`, ...) with valid electrode positions (needed for interpolation/topomaps).
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
├── data/<SUBJECT>/            # Input recording (gitignored, not versioned)
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
- The pipeline is **interactive**: several plot windows are blocking (`block=True`) and, in manual mode, it prompts on stdin for the bad channels and ICA components to exclude. It's meant to be run and watched, not used headlessly/in a batch over many recordings as-is.
- `OUTPUT_PATH` is created automatically if missing.
- Logs go to both the console (colored) and `logs/eeg_signal_preprocessing.log` (rotating, 20 MB × 10 backups).
- The final output is written to `OUTPUT_PATH/preprocessed_data.fif`, overwriting any previous run.

## Known limitations / possible next steps

- Processes a single recording per run; no batch/multi-subject support yet.
- Output is a continuous signal, not epoched — fine since epoching belongs to the separate ML pipeline, but worth knowing if you consume `preprocessed_data.fif` directly.
- Manual mode requires a display and a human in the loop; there's no way to run it unattended.
- `ica.find_bads_muscle` has an intermittent internal MNE failure (`ValueError` from an inhomogeneous-shape `np.array` call inside MNE itself) on some component/PSD combinations; it's caught so the run continues with only the ECG/EOG exclusions, but muscle-related components won't always be caught automatically. On a reference recording, manual inspection removed a muscle-contaminated channel's variance far more effectively than the automatic path.
- `N_COMPONENTS` set low (10) makes the automatic ECG/EOG/muscle detectors' correlation-based approach the main lever for quality; a low count also means more of the signal's variance is left unmodeled by any single component.
- The bad-channel/segment detection thresholds (`ARTEFACT_MAX`, `FLAT_THRESHOLD`, `BAD_CHANNEL_PERCENT`, `VARIANCE_Z_THRESHOLD`) were validated on a small number of reference recordings; a recording with different noise characteristics may need different values.
