import os

import mne
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs
from logger_init import logger


def load_eeg_files():

    ###################################################################################################
    # 1/ We first load the data using the mne tutorial 
    ###################################################################################################

    raw = _create_mne_file()

    ###################################################################################################
    # 2 / Show the eog and ecg signals which correspond the ocular and cardiovascular movements
    ###################################################################################################

    #_eog_ecg_plot(raw) 
    
    
    ###################################################################################################
    # 3/ We apply the ICA 
    ###################################################################################################
    
    #We filter the raw data to remove the slow drift 
    filt_raw = raw.copy().filter(l_freq=1.0, h_freq=None)

    _ICA_method(filt_raw)


    


def _create_mne_file() -> mne:
    """This function upload the already existing file created in the MNE library and plot using matplotlib 
    the raw data before the preprocessing in a time duration of 10 seconds. 

    return the raw data 
    """

    sample_data_folder = mne.datasets.sample.data_path()
    sample_data_raw_file = os.path.join(
    sample_data_folder, "MEG", "sample", "sample_audvis_filt-0-40_raw.fif"
    )
    raw = mne.io.read_raw_fif(sample_data_raw_file)

    # Here we'll crop to 60 seconds and drop gradiometer channels for speed
    raw.crop(tmax=60.0).pick(picks=["mag", "eeg", "stim", "eog"])
    raw.load_data()

    # pick some channels that clearly show heartbeats and blinks
    regexp = r"(MEG [12][45][123]1|EEG 00.)"
    artifact_picks = mne.pick_channels_regexp(raw.ch_names, regexp=regexp)
    #raw.plot(order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=False, block=True)
    return raw

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


def _ICA_method(filt_raw: mne) :
    """
    This function use the ICA method to preprocess the raw data 

    Args: 
        data already filtered to remove slow drift .fif (mne component)
    """
    ica = ICA(n_components=15, max_iter="auto", random_state=97)
    ica.fit(filt_raw, reject=dict(eeg=200e-6))  # avoid a couple of big artifacts
    ica
    explained_var_ratio = ica.get_explained_variance_ratio(filt_raw)
    for channel_type, ratio in explained_var_ratio.items():
        logger.info(f"Fraction of {channel_type} variance explained by all components: {ratio}")
    explained_var_ratio = ica.get_explained_variance_ratio(filt_raw, components=[0], ch_type="eeg")
    # This time, logger.info as percentage.
    ratio_percent = round(100 * explained_var_ratio["eeg"])
    logger.info(
        f"Fraction of variance in EEG signal explained by first component: {ratio_percent}%"
    )