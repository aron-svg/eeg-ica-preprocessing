import os

import mne
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs



def load_eeg_files():
    sample_data_folder = mne.datasets.sample.data_path()
    sample_data_raw_file = os.path.join(
    sample_data_folder, "MEG", "sample", "sample_audvis_filt-0-40_raw.fif"
    )
    raw = mne.io.read_raw_fif(sample_data_raw_file)

    # Here we'll crop to 60 seconds and drop gradiometer channels for speed
    raw.crop(tmax=60.0).pick(picks=["mag", "eeg", "stim", "eog"])
    raw.load_data()