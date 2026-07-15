from logger_init import logger
from config import DATA_PATH, OUTPUT_PATH, eeg_ica_Exception
import sys
import os
from ICA import run_app

def check_input_paths():
    if os.path.exists(DATA_PATH) :
        logger.info(f"Data path exists: {DATA_PATH}")
    else:
        logger.error(f"Data path does not exist: {DATA_PATH}")
        raise eeg_ica_Exception(f"Data path does not exist: {DATA_PATH}")
    
    if os.path.exists(OUTPUT_PATH) :
        logger.info(f"Output path exists: {OUTPUT_PATH}")
    else:
        logger.warning(f"Output path does not exist: {OUTPUT_PATH}. It will be created.")
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        logger.info(f"Output path created: {OUTPUT_PATH}")


if __name__ == "__main__":
    try: 
        logger.info("Starting the main process")
        check_input_paths()
    
        ####################################################################
        # 1/ load the data using the ICA method and execute the analysis
        ####################################################################

        logger.info("entering inside the analysis")
        eeg_data = run_app()
        logger.info("analysis finished successfully")
        
        ####################################################################
        # 2/ Save the preprocessed data to the output path
        ####################################################################

        output_file = os.path.join(OUTPUT_PATH, "preprocessed_data.fif")
        eeg_data.save(output_file, overwrite=True)
        logger.info(f"Preprocessed data saved to {output_file}")
    except eeg_ica_Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)