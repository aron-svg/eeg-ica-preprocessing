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

    raw, _ = _create_mne_file()

    ###################################################################################################
    # 2 / Show the eog and ecg signals which correspond the ocular and cardiovascular movements
    ###################################################################################################

    #_eog_ecg_plot(raw)


    ###################################################################################################
    # 3/ We apply the ICA
    ###################################################################################################

    return _ICA_method(raw)
 


def _create_mne_file() -> mne:
    """
    This function loads the EEG recording located in DATA_PATH and plots
    the raw data before the preprocessing.

    return:
        -the raw data, filtered, with bad channels marked (not yet
         interpolated) and re-referenced
        -artifact_picks: the indices of the channels that correspond to the artifacts (eog and ecg)
    """

    raw_file = os.path.join(DATA_PATH, "recording.fif")
    raw = mne.io.read_raw_fif(raw_file)
    raw.load_data()


    # the eeg/eog/ecg channels are stored in microvolts but labeled as volts in the
    # fif header, so we rescale them to get physically plausible values (misc
    # channels like temp/resp/gsr/timestamp are not affected by this issue)
    volt_picks = mne.pick_types(raw.info, eeg=True, eog=True, ecg=True)
    raw.apply_function(lambda x: x * 1e-6, picks=volt_picks, channel_wise=True)

    # standard preprocessing order: filter -> flag noisy segments/channels ->
    # reference -> ICA -> interpolate bad channels last (see
    # _plot_ica_components). Bad channels are only marked here, not yet
    # interpolated, so they don't distort the ICA fit or the average
    # reference, and so they get reconstructed from already ICA-cleaned
    # neighbours instead of artifact-contaminated ones.

    # FLAT_THRESHOLD/ARTEFACT_MAX are calibrated for the natural
    # sample-to-sample jitter of the raw, full-bandwidth signal; band-pass
    # filtering removes exactly that high-frequency jitter, so running the
    # same detection on the filtered signal instead flags nearly every
    # channel as "flat". Keep this unfiltered copy just for detection, and
    # carry the result over onto the filtered signal used everywhere else.
    unfiltered = raw.copy()

    raw = _filter_raw(raw)
    raw = _detect_bad_channels(raw, unfiltered)
    raw = _set_reference(raw)

    # pick the EOG/ECG channels that clearly show heartbeats and blinks
    regexp = r"(ECG0[1-3]|EOG0[1-5])"
    artifact_picks = mne.pick_channels_regexp(raw.ch_names, regexp=regexp)
    #raw.plot(order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=True, block=True)


    return raw, artifact_picks


###################################################################################################
###################################################################################################

#  -- This part correspond to the filtering before ICA  --

###################################################################################################
###################################################################################################

def _detect_high_variance_channels(raw: mne, eeg_picks) -> list:
    """
    Flag EEG channels whose overall amplitude variance is a statistical
    outlier relative to the other channels (modified z-score based on the
    median absolute deviation). This catches broadband contamination (e.g.
    muscle tension) that annotate_amplitude's per-sample-jump criterion
    cannot see, since such contamination raises the channel's overall
    variance without producing the large consecutive-sample jumps
    annotate_amplitude looks for.

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

    # get_explained_variance_ratio() reads channels by type ("eeg"), which
    # includes bad channels even though ica.fit() excluded them: those
    # channels are never touched by ica.apply(), so they'd show a trivial
    # zero reconstruction error regardless of the component tested, inflating
    # every ratio. Restrict to the channels the ICA was actually fitted on.
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
    # measure="correlation" thresholds directly on the correlation coefficient
    # instead of a z-score computed across components; the z-score default is
    # unreliable with few components (N_COMPONENTS), since its achievable
    # range shrinks with the sample size and can make a genuine, strongly
    # correlated component (e.g. r=0.99) fail to ever cross the threshold

    # Automatically find which ICs match the ECG pattern
    ecg_inds, scores = ica.find_bads_ecg(raw, method="correlation", measure="correlation", threshold=0.5)
    ica.exclude.extend(ecg_inds)
    logger.info(f"Excluding {len(ecg_inds)} ICA components: {ecg_inds}")

    # Automatically find which ICs match the EOG pattern
    eog_inds, scores = ica.find_bads_eog(raw, measure="correlation", threshold=0.5)
    ica.exclude.extend(eog_inds)
    logger.info(f"Excluding {len(eog_inds)} ICA components: {eog_inds}")

    # ECG/EOG correlation only catches cardiac/ocular components; muscle (EMG)
    # contamination shows up as a distinct spectral/spatial signature instead,
    # so it needs its own dedicated detector. This has an intermittent internal
    # MNE failure on some component/PSD combinations (ValueError from an
    # inhomogeneous-shape np.array call); since this check is a bonus on top of
    # the ECG/EOG exclusion above and a fit is expensive to redo, don't let it
    # take down the whole run.
    try:
        muscle_inds, scores = ica.find_bads_muscle(raw)
        ica.exclude.extend(muscle_inds)
        logger.info(f"Excluding {len(muscle_inds)} ICA components: {muscle_inds}")
    except ValueError:
        logger.warning("find_bads_muscle failed, skipping muscle-component exclusion", exc_info=True)

    return _plot_ica_components(ica, raw, raw.copy())





def _plot_ica_components(ica, raw, reconst_raw):
    """
    This function plot the different components extracted by the ICA method and the sources of the raw data
    
    Args:
        -ica: the ICA object that contains the components extracted from the raw data
        -raw: the raw data in .fif (mne component)
    Return:
        -preprocessed mne data
    """

    # overlay before/after on the same axes (RMS/GFP) so the ICA's effect is
    # unambiguous, unlike comparing two independently-scrollable raw.plot() windows
    ica.plot_overlay(raw, exclude=ica.exclude, picks="eeg", title="Effect of ICA cleaning (EEG)")

    ica.apply(reconst_raw)

    # bad channels were only marked in _detect_bad_channels (not
    # interpolated), so they wouldn't distort the ICA fit or the average
    # reference; now that ICA has cleaned the other channels, interpolate
    # them from those already-cleaned neighbours instead of the original
    # artifact-contaminated ones
    reconst_raw.interpolate_bads(reset_bads=True)

    # ica.apply() only reconstructs the channels the ICA was fitted on (the EEG
    # channels), the ECG/EOG channels are raw sensor recordings and are never
    # touched, so we compare the EEG channels here to actually see the ICA's effect.
    # exclude=[] so the still-marked-bad channels on raw are still plotted for
    # comparison, alongside their now-interpolated counterpart in reconst_raw
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])

    # Plot the original and reconstructed signals for comparison
    raw.plot(order=eeg_picks, n_channels=len(eeg_picks), show_scrollbars=True, title="Raw (before ICA)")
    reconst_raw.plot(
        order=eeg_picks, n_channels=len(eeg_picks), show_scrollbars=True, block=True, title="Preprocessed (after ICA)"
    )
    return reconst_raw







###################################################################################################
###################################################################################################

#  -- This part correspond to the visualization of all the eeg signals and the different components extracted by the ICA method. --

###################################################################################################
###################################################################################################




def _eog_ecg_plot(raw : mne):
    """
    This function plot the average spike for the eog and the ecg captors using topography and 
    spectrophotometry

    Args: 
        raw data in .fif (mne component)
    """
    eog_evoked = create_eog_epochs(raw).average()
    eog_evoked.apply_baseline(baseline=(None, -0.2))
    eog_evoked.plot_joint()
    ecg_evoked = create_ecg_epochs(raw).average()
    ecg_evoked.apply_baseline(baseline=(None, -0.2))
    ecg_evoked.plot_joint()





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
            # reject=None works around an MNE bug (IndexError in
            # _fast_plot_ica_properties) that happens when reject='auto' (the
            # default) tries to reinsert dropped-epoch stats using indices from
            # the original, un-dropped epoch numbering, which go out of bounds
            # as soon as enough segments get rejected across the recording
            ica.plot_properties(raw, picks=inspected_components, reject=None)
            # # blinks
            # ica.plot_overlay(raw, exclude=inspected_components, picks="eeg")
            # # heartbeats
            # ica.plot_overlay(raw, exclude=inspected_components, picks="ecg")