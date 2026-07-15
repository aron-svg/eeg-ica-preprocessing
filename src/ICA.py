import os

import mne
import numpy as np
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs
from assets.filter_ica import preprocess_for_ica_to_mne_format
from assets.viz_ica import component_visualization, plot_ica_components
from logger_init import logger
from config import (
    MANUAL_MODE,
    DATA_PATH,
    N_COMPONENTS,
    ARTEFACT_MAX,
)


def run_app():
    """
    This function executes the analysis of the EEG data using the ICA method. 
    It first loads the data, then applies the ICA method to preprocess the data, 
    and finally returns the preprocessed data.
    return:
        -preprocessed mne data
    """
    raw_data = load_data()
    mne_raw = preprocess_for_ica_to_mne_format(raw_data)
    return _ICA_method(mne_raw)
  
def load_data(): 
    # Load the data using the mne tutorial
    raw_file = os.path.join(DATA_PATH, "recording.fif")
    raw = mne.io.read_raw_fif(raw_file)
    return raw.load_data()


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
    component_visualization(ica, raw)
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

    return plot_ica_components(ica, raw, raw.copy())


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

    return plot_ica_components(ica, raw, raw.copy())

