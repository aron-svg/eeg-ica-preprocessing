from logger_init import logger
import sys
from assets.loader import load_eeg_files

def execute_analysis(DATA_PATH : str = None, OUTPUT_PATH : str = None):
    if DATA_PATH == None or OUTPUT_PATH == None: 
        logger.warning("the data/output path does not exist")
        sys.exit(1)
    
    ########################################################
    # first step: load the data
    ########################################################
    logger.info("loading data...")

    eeg_data = load_eeg_files()
    logger.info("data loaded successfully.")

    
    return eeg_data