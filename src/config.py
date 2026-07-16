from pathlib import Path

class eeg_ica_Exception(Exception):
    """Base class for exceptions in this module."""
    pass


###############################################################################
# Paths
###############################################################################

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = str(BASE_DIR / "data" / "HZO040") + "/"
OUTPUT_PATH = str(BASE_DIR / "output"/ "HZO040_auto") + "/"





###############################################################################
# General Parameters
############################################################################### 

MANUAL_MODE = False  # Set to True for manual ICA component selection, False for automatic
N_COMPONENTS = 10  # Number of ICA components to compute
LOW_FREQUENCY = 1.0  # High-pass cutoff for filtering, removes slow drift (in Hz)
HIGH_FREQUENCY = 45.0  # Low-pass cutoff for filtering, removes high-frequency noise (in Hz)
NOTCH_FREQUENCY = 50.0  # Notch filter frequency, removes powerline interference (in Hz)
ARTEFACT_MAX = 200e-6  # Maximum amplitude for artefact rejection (in volts)
FLAT_THRESHOLD = 1e-6  # Peak-to-peak amplitude below which a channel is considered flat/disconnected (in volts)
BAD_CHANNEL_PERCENT = 3.5  # % of the recording a channel may be flat/too noisy before being marked bad
VARIANCE_Z_THRESHOLD = 6 # modified z-score cutoff (Iglewicz & Hoaglin convention)