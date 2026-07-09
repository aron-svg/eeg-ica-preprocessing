import os

import mne
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs
from logger_init import logger
from config import MANUAL_MODE, DATA_PATH, N_COMPONENTS, LOW_FREQUENCY, HIGH_FREQUENCY, NOTCH_FREQUENCY, ARTEFACT_MAX

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
    
    #We filter the raw data before running the ICA
    filt_raw = _filter_raw(raw)

    return _ICA_method(filt_raw, raw)
 


def _create_mne_file() -> mne:
    """
    This function loads the EEG recording located in DATA_PATH and plots
    the raw data before the preprocessing.

    return: 
        -the raw data
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

    # pick the EOG/ECG channels that clearly show heartbeats and blinks
    regexp = r"(ECG0[1-3]|EOG0[1-5])"
    artifact_picks = mne.pick_channels_regexp(raw.ch_names, regexp=regexp)
    raw.plot(order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=True, block=True)
    return raw,artifact_picks


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


def _ICA_method(filt_raw: mne, raw: mne) :
    """
    This function use the ICA method to preprocess the raw data 

    Args: 
        -data already filtered to remove slow drift .fif (mne component)
        -raw data in .fif (mne component)
    Return:
        -preprocessed mne data
    """
    ica = ICA(n_components=N_COMPONENTS, max_iter="auto", random_state=97)
    ica.fit(filt_raw, reject=dict(eeg=ARTEFACT_MAX))  # avoid a couple of big artifacts

    explained_var_ratio = ica.get_explained_variance_ratio(filt_raw)
    for channel_type, ratio in explained_var_ratio.items():
        logger.info(f"Fraction of {channel_type} variance explained by all components: {ratio}")
    
    for i in range(N_COMPONENTS):
        explained_var_ratio = ica.get_explained_variance_ratio(filt_raw, components=[i], ch_type="eeg")
        # This time, logger.info as percentage.
        ratio_percent = round(100 * explained_var_ratio["eeg"])
        logger.info(
            f"Fraction of variance in EEG signal explained by {i} component: {ratio_percent}%"
        )
    raw.load_data()
    _component_visualization(ica, raw, filt_raw)
    # the user can choose to manually select the ICA components to exclude or let the algorithm do it automatically
    if MANUAL_MODE:
        return _manual_exclusion(ica, raw, filt_raw)
    else:
        return _automatic_exclusion(ica, raw, filt_raw)



###################################################################################################
###################################################################################################

#  -- This part correspond to the exclusion of the ICA components that correspond to the artifacts. --

###################################################################################################
###################################################################################################


   
def _manual_exclusion(ica,raw,filt_raw):
    """
    This function allows the user to manually select the ICA components to exclude from the analysis.

    Args:
        -ica: the ICA object that contains the components extracted from the raw data
        -raw: the raw data in .fif (mne component)
        -filt_raw: the filtered (band-pass + notch) copy of the raw data
    Return:
        -preprocessed mne data
    """
    # we fist ask the user to input the indices of the ICA components to exclude
    excluded_components = input("Enter the indices of the ICA components to exclude (comma-separated): ")
    excluded_components = [int(x.strip()) for x in excluded_components.split(",") if x.strip().isdigit()]
    logger.info(f"Excluding ICA components: {excluded_components}")
    ica.exclude = excluded_components

    return _plot_ica_components(ica, raw, filt_raw.copy())


def _automatic_exclusion(ica,raw,filt_raw):
    """
    This function automatically selects the ICA components to exclude from the analysis based
    on their correlation with the ECG and EOG signals.

    Args:
        -ica: the ICA object that contains the components extracted from the raw data
        -raw: the raw data in .fif (mne component)
        -filt_raw: the filtered (band-pass + notch) copy of the raw data
    Return:
        -preprocessed mne data
    """
    # Automatically find which ICs match the ECG pattern
    ecg_inds, scores = ica.find_bads_ecg(raw, method="correlation", threshold="auto")
    ica.exclude.extend(ecg_inds)
    logger.info(f"Excluding {len(ecg_inds)} ICA components: {ecg_inds}")

    # Automatically find which ICs match the EOG pattern
    eog_inds, scores = ica.find_bads_eog(raw, threshold=3.0)
    ica.exclude.extend(eog_inds)
    logger.info(f"Excluding {len(eog_inds)} ICA components: {eog_inds}")

    return _plot_ica_components(ica, raw, filt_raw.copy())





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

    # ica.apply() only reconstructs the channels the ICA was fitted on (the EEG
    # channels), the ECG/EOG channels are raw sensor recordings and are never
    # touched, so we compare the EEG channels here to actually see the ICA's effect
    eeg_picks = mne.pick_types(raw.info, eeg=True)

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





def _component_visualization(ica, raw,filt_raw):
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
            ica.plot_properties(filt_raw, picks=inspected_components)
            # blinks
            ica.plot_overlay(raw, exclude=inspected_components, picks="eeg")
            # heartbeats
            ica.plot_overlay(raw, exclude=inspected_components, picks="ecg")