from logger_init import logger
from config import DATA_PATH, OUTPUT_PATH
import sys
import os
from ICA import execute_analysis

if __name__ == "__main__":
    
    logger.info("Starting the main process")
    if DATA_PATH == None or OUTPUT_PATH == None: 
        logger.warning("the data/output path does not exist")
        sys.exit(1)

    ####################################################################
    # 1/ We execute the analysis and load the data using the ICA method
    ####################################################################

    logger.info("entering inside the analysis")
    eeg_data = execute_analysis()
    logger.info("analysis finished successfully")
    
    ####################################################################
    # 2/ Save the preprocessed data to the output path
    ####################################################################

    output_file = os.path.join(OUTPUT_PATH, "preprocessed_data.fif")
    eeg_data.save(output_file, overwrite=True)
    logger.info(f"Preprocessed data saved to {output_file}")