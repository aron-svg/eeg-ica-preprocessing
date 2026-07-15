import mne
import numpy as np
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs
from logger_init import logger
from config import (
    MANUAL_MODE,
)


def component_visualization(ica, raw):
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


def plot_ica_components(ica, raw, reconst_raw):
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

