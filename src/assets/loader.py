import os

import mne
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs
from logger_init import logger

MANUAL_MODE = False  # Set to True for manual ICA component selection, False for automatic

def load_eeg_files():

    ###################################################################################################
    # 1/ We first load the data using the mne tutorial 
    ###################################################################################################

    raw,artifact_picks = _create_mne_file()

    ###################################################################################################
    # 2 / Show the eog and ecg signals which correspond the ocular and cardiovascular movements
    ###################################################################################################

    #_eog_ecg_plot(raw) 
    
    
    ###################################################################################################
    # 3/ We apply the ICA 
    ###################################################################################################
    
    #We filter the raw data to remove the slow drift 
    filt_raw = raw.copy().filter(l_freq=1.0, h_freq=None)

    _ICA_method(filt_raw, raw, artifact_picks)
 


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
    raw.plot(order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=False, block=True)
    return raw,artifact_picks


def _ICA_method(filt_raw: mne, raw: mne,artifact_picks) :
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
    
    for i in range(15):
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
        _manual_exclusion(ica, raw, artifact_picks)
    else: 
        _automatic_exclusion(ica, raw, artifact_picks)

    
   

   
def _manual_exclusion(ica,raw,artifact_picks):
    
    # we fist ask the user to input the indices of the ICA components to exclude
    excluded_components = input("Enter the indices of the ICA components to exclude (comma-separated): ")
    excluded_components = [int(x.strip()) for x in excluded_components.split(",") if x.strip().isdigit()]
    logger.info(f"Excluding ICA components: {excluded_components}")
    ica.exclude = excluded_components
 
    _plot_ica_components(ica, raw, raw.copy(), artifact_picks)
    

def _automatic_exclusion(ica,raw,artifact_picks):

    # Automatically find which ICs match the ECG pattern
    ecg_inds, scores = ica.find_bads_ecg(raw, method="correlation", threshold="auto")
    ica.exclude.extend(ecg_inds)
    logger.info(f"Excluding {len(ecg_inds)} ICA components: {ecg_inds}")

    # Automatically find which ICs match the EOG pattern
    eog_inds, scores = ica.find_bads_eog(raw, threshold=3.0)
    ica.exclude.extend(eog_inds)
    logger.info(f"Excluding {len(eog_inds)} ICA components: {eog_inds}")

    _plot_ica_components(ica, raw, raw.copy(), artifact_picks)





def _plot_ica_components(ica, raw, reconst_raw, artifact_picks):
    """
    This function plot the different components extracted by the ICA method and the sources of the raw data
    """

    ica.apply(reconst_raw)


    # Plot the original and reconstructed signals for comparison
    raw.plot(order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=False)
    reconst_raw.plot(
        order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=False,block=True
    )








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

    ica.plot_sources(raw, show_scrollbars=False)
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
            ica.plot_overlay(raw, exclude=inspected_components, picks="mag")