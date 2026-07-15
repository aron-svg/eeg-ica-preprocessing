import os

import mne
import numpy as np
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs
from logger_init import logger
from config import (
    MANUAL_MODE,
    DATA_PATH,
    N_COMPONENTS,
    LOW_FREQUENCY,
    HIGH_FREQUENCY,
    NOTCH_FREQUENCY,
    ARTEFACT_MAX,
    FLAT_THRESHOLD,
    BAD_CHANNEL_PERCENT,
    VARIANCE_Z_THRESHOLD
)



def execute_analysis():
    """
    This function executes the analysis of the EEG data using the ICA method. 
    It first loads the data, then applies the ICA method to preprocess the data, 
    and finally returns the preprocessed data.
    return:
        -preprocessed mne data
    """
    ###################################################################################################
    # 1/ We first load the data using the mne tutorial
    ###################################################################################################

    raw_data = load_data()
    mne_raw = preprocess_for_ica_to_mne_format(raw_data)

    ###################################################################################################
    # 2/ We apply the ICA
    ###################################################################################################

    return _ICA_method(mne_raw)
  
def load_data():
    raw_file = os.path.join(DATA_PATH, "recording.fif")
    raw = mne.io.read_raw_fif(raw_file)
    return raw.load_data()

def preprocess_for_ica_to_mne_format(raw_data) -> mne:
    """
    This function loads the EEG recording located in DATA_PATH and plots
    the raw data before the preprocessing.
    standard preprocessing order: filter -> flag noisy segments/channels -> reference -> ICA -> interpolate bad channels last
    (see_plot_ica_components). Bad channels are only marked here, not yet
    interpolated, so they don't distort the ICA fit or the average
    reference, and so they get reconstructed from already ICA-cleaned
    neighbours instead of artifact-contaminated ones.

    return:
        -the raw data, filtered, with bad channels marked (not yet
         interpolated) and re-referenced
    """


    #  we rescale them to get physically plausible values
    volt_picks = mne.pick_types(raw_data.info, eeg=True, eog=True, ecg=True)
    raw_data.apply_function(lambda x: x * 1e-6, picks=volt_picks, channel_wise=True)

    unfiltered = raw_data.copy()

    # filter -> flag noisy segments/channels -> reference
    raw_data = _filter_raw(raw_data)
    raw_data = _detect_bad_channels(raw_data, unfiltered)
    raw_data = _set_reference(raw_data)

    return raw_data


###################################################################################################
###################################################################################################

#  -- This part correspond to the filtering before ICA (bad channel + variance detection + filtering  + reference)  --

###################################################################################################
###################################################################################################

def _detect_high_variance_channels(raw: mne, eeg_picks) -> list:
    """
    Flag EEG channels whose overall amplitude variance is a statistical
    outlier relative to the other channels (modified z-score based on the
    median absolute deviation).

    Variance is computed on a high-pass-filtered copy (slow drift removed)
    so a shared low-frequency trend doesn't dominate every channel's
    variance and mask the comparison.

    Args:
        -raw: the raw data in .fif (mne component)
        -eeg_picks: indices of the EEG channels to check
    Return:
        -list of channel names flagged as high-variance outliers
    """
    hp = raw.copy().filter(l_freq=LOW_FREQUENCY, h_freq=None, picks=eeg_picks, verbose="ERROR")
    stds = hp.get_data(picks=eeg_picks).std(axis=1)

    median = np.median(stds)
    mad = np.median(np.abs(stds - median))
    if mad == 0:
        return []
    modified_z = 0.6745 * (stds - median) / mad

    ch_names = [raw.ch_names[i] for i in eeg_picks]
    return [ch_names[i] for i in range(len(ch_names)) if modified_z[i] > VARIANCE_Z_THRESHOLD]


def _detect_bad_channels(raw: mne, unfiltered: mne) -> mne:
    """
    Detect and annotate noisy segments, and mark (but do not interpolate)
    flat/disconnected or excessively noisy EEG channels. Interpolation is
    deferred until after the ICA cleaning (see _plot_ica_components), so a
    bad channel doesn't distort the ICA fit or the average reference, and so
    it gets reconstructed from already ICA-cleaned neighbours instead of
    artifact-contaminated ones.

    Args:
        -raw: the filtered raw data; the detected bads/annotations are
         applied to this one, which is what the rest of the pipeline uses
        -unfiltered: an unfiltered copy used only to run the detection
         itself, since FLAT_THRESHOLD/ARTEFACT_MAX/the variance check are
         calibrated for the raw signal's natural sample-to-sample jitter
    Return:
        -raw: the same raw data, with bad EEG channels marked in info["bads"]
         and noisy segments annotated
    """
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    raw.plot(order=eeg_picks, n_channels=len(eeg_picks), block=True, show_scrollbars=True, title="Raw (before bad channel detection)")

    # annotate_amplitude also returns brief high-amplitude/flat segments in
    # addition to the channel-level bads; not applying them as annotations
    # for now (segment_annotations is unused), just using the channel-level
    # bads below
    _, amplitude_bads = mne.preprocessing.annotate_amplitude(
        unfiltered, flat=FLAT_THRESHOLD, peak=ARTEFACT_MAX, bad_percent=BAD_CHANNEL_PERCENT, picks=eeg_picks
    )

    if not(MANUAL_MODE):
        # annotate_amplitude only flags large consecutive-sample jumps or flat
        # segments; broadband contamination (e.g. muscle tension) instead
        # shows up as an overall elevated variance, which needs its own check
        variance_bads = _detect_high_variance_channels(raw, eeg_picks)
        bads = sorted(set(amplitude_bads) | set(variance_bads))

    else:
        bads = input("Manual mode is enabled. Please enter the names of the bad EEG channels to exclude (comma-separated): ")
        bads = [x.strip() for x in bads.split(",") if x.strip()]

    if bads:
        logger.info(f"Marking {len(bads)} bad EEG channel(s) (interpolation deferred until after ICA): {bads}")
        raw.info["bads"] = bads

    else:
        logger.info("No bad EEG channels detected")


    raw.plot(order=eeg_picks, block=True, n_channels=len(eeg_picks), show_scrollbars=True, title="Raw (bad channels marked, not yet interpolated)")

    return raw


def _set_reference(raw: mne) -> mne:
    """
    Apply the common average reference to the EEG channels. Must be called
    after bad channel detection (see _detect_bad_channels): bad channels are
    marked in info["bads"] by then, so MNE automatically excludes them from
    the average instead of letting them distort it.

    Args:
        -raw: the raw data in .fif (mne component)
    Return:
        -raw: the re-referenced raw data
    """
    raw.set_eeg_reference(ref_channels="average")
    return raw


def _filter_raw(raw: mne) -> mne:
    """
    Apply the standard frequency filtering to a copy of the raw data before
    running the ICA: high-pass at LOW_FREQUENCY (removes slow drift), low-pass
    at HIGH_FREQUENCY (removes high-frequency noise) and a notch filter at
    NOTCH_FREQUENCY (removes powerline interference).

    Args:
        -raw: the raw data in .fif (mne component)
    Return:
        -filt_raw: the filtered copy of the raw data
    """
    filt_raw = raw.copy().filter(l_freq=LOW_FREQUENCY, h_freq=HIGH_FREQUENCY)
    filt_raw.notch_filter(freqs=NOTCH_FREQUENCY)
    return filt_raw


###################################################################################################
###################################################################################################

#  -- This part correspond to the ICA method to preprocess the raw data --

###################################################################################################
###################################################################################################


def _ICA_method(raw: mne) :
    """
    This function use the ICA method to preprocess the raw data

    Args:
        -raw: the filtered, referenced raw data with bad channels marked
         (not yet interpolated), in .fif (mne component)
    Return:
        -preprocessed mne data
    """
    ica = ICA(n_components=N_COMPONENTS, max_iter="auto", random_state=97)
    ica.fit(raw, reject=dict(eeg=ARTEFACT_MAX))  # avoid a couple of big artifacts

   
    good_only = raw.copy().pick(picks=ica.ch_names)
    explained_var_ratio = ica.get_explained_variance_ratio(good_only)

    for channel_type, ratio in explained_var_ratio.items():
        logger.info(f"Fraction of {channel_type} variance explained by all components: {ratio}")

    for i in range(N_COMPONENTS):
        explained_var_ratio = ica.get_explained_variance_ratio(good_only, components=[i], ch_type="eeg")
        # This time, logger.info as percentage.
        ratio_percent = round(100 * explained_var_ratio["eeg"])
        logger.info(
            f"Fraction of variance in EEG signal explained by {i} component: {ratio_percent}%"
        )
    _component_visualization(ica, raw)
    # the user can choose to manually select the ICA components to exclude or let the algorithm do it automatically
    if MANUAL_MODE:
        return _manual_exclusion(ica, raw)
    else:
        return _automatic_exclusion(ica, raw)



###################################################################################################
###################################################################################################

#  -- This part correspond to the exclusion of the ICA components that correspond to the artifacts. --

###################################################################################################
###################################################################################################


   
def _manual_exclusion(ica,raw):
    """
    This function allows the user to manually select the ICA components to exclude from the analysis.

    Args:
        -ica: the ICA object that contains the components extracted from the raw data
        -raw: the raw data in .fif (mne component)
    Return:
        -preprocessed mne data
    """
    # we fist ask the user to input the indices of the ICA components to exclude
    excluded_components = input("Enter the indices of the ICA components to exclude (comma-separated): ")
    excluded_components = [int(x.strip()) for x in excluded_components.split(",") if x.strip().isdigit()]
    logger.info(f"Excluding ICA components: {excluded_components}")
    ica.exclude = excluded_components

    return _plot_ica_components(ica, raw, raw.copy())


def _automatic_exclusion(ica,raw):
    """
    This function automatically selects the ICA components to exclude from the analysis based
    on their correlation with the ECG and EOG signals.

    Args:
        -ica: the ICA object that contains the components extracted from the raw data
        -raw: the raw data in .fif (mne component)
    Return:
        -preprocessed mne data
    """

    # Automatically find which ICs match the ECG pattern
    ecg_inds, scores = ica.find_bads_ecg(raw, method="correlation", measure="correlation", threshold=0.5)
    ica.exclude.extend(ecg_inds)
    logger.info(f"Excluding {len(ecg_inds)} ICA components: {ecg_inds}")

    # Automatically find which ICs match the EOG pattern
    eog_inds, scores = ica.find_bads_eog(raw, measure="correlation", threshold=0.5)
    ica.exclude.extend(eog_inds)
    logger.info(f"Excluding {len(eog_inds)} ICA components: {eog_inds}")

    try:
        muscle_inds, scores = ica.find_bads_muscle(raw)
        ica.exclude.extend(muscle_inds)
        logger.info(f"Excluding {len(muscle_inds)} ICA components: {muscle_inds}")
    except ValueError:
        logger.warning("find_bads_muscle failed, skipping muscle-component exclusion", exc_info=True)

    return _plot_ica_components(ica, raw, raw.copy())


###################################################################################################
###################################################################################################

#  -- This part correspond to the visualization of all the eeg signals and the different components extracted by the ICA method. --

###################################################################################################
###################################################################################################




def _component_visualization(ica, raw):
    """
    This function plot the different components extracted by the ICA method and the sources of the raw data
    """

    ica.plot_sources(raw, show_scrollbars=True)
    ica.plot_components()


    if MANUAL_MODE:
        inspected_components= input("Manual mode is enabled. Please inspect the ICA components that you find to be artifacts")
        inspected_components = [int(x.strip()) for x in inspected_components.split(",") if x.strip().isdigit()]
        if not inspected_components:
            logger.warning("No components were selected for inspection. Proceeding without visualization.")
        else:
            logger.info(f"Inspected components: {inspected_components}")
           
            ica.plot_properties(raw, picks=inspected_components, reject=None)

            # # blinks
            # ica.plot_overlay(raw, exclude=inspected_components, picks="eeg")
            # # heartbeats
            # ica.plot_overlay(raw, exclude=inspected_components, picks="ecg")


def _plot_ica_components(ica, raw, reconst_raw):
    """
    This function plot the different components extracted by the ICA method and the sources of the raw data
    
    Args:
        -ica: the ICA object that contains the components extracted from the raw data
        -raw: the raw data in .fif (mne component)
    Return:
        -preprocessed mne data
    """

    ica.plot_overlay(raw, exclude=ica.exclude, picks="eeg", title="Effect of ICA cleaning (EEG)")

    ica.apply(reconst_raw)
    reconst_raw.interpolate_bads(reset_bads=True)
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])

    # Plot the original and reconstructed signals for comparison
    raw.plot(order=eeg_picks, n_channels=len(eeg_picks), show_scrollbars=True, title="Raw (before ICA)")
    reconst_raw.plot(
        order=eeg_picks, n_channels=len(eeg_picks), show_scrollbars=True, block=True, title="Preprocessed (after ICA)"
    )
    return reconst_raw

