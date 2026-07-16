from logger_init import logger
from config import DATA_PATH, OUTPUT_PATH, eeg_ica_Exception
import sys
import os
from ICA import run_app
from ica_tools.csv_ica import csv_creator

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

def output_path(OUTPUT_PATH, eeg_data):
    """
    This function saves the preprocessed data to the output path in .fif format.
    the versioning starts with v1 in the format preprocessed_data_v1.fif, and increments 
    depending on the number of existing preprocessed files in the output path.
    Args:
        -OUTPUT_PATH: the path where the preprocessed data will be saved
        -eeg_data: the preprocessed EEG data to be saved
    """
    while True:
        existing_files = [f for f in os.listdir(OUTPUT_PATH) if f.startswith("preprocessed_data_v") and f.endswith(".fif")]
        version_numbers = [int(f.split("_v")[1].split(".fif")[0]) for f in existing_files]
        next_version = max(version_numbers, default=0) + 1
        output_file = os.path.join(OUTPUT_PATH, f"preprocessed_data_v{next_version}.fif")
        csv_file = os.path.join(OUTPUT_PATH, f"preprocessed_data_v{next_version}.csv")
        if not os.path.exists(output_file):
            break
        else:
            logger.warning(f"Output file {output_file} already exists. Incrementing version number.")
    eeg_data.save(output_file, overwrite=True)
    return output_file, csv_file


if __name__ == "__main__":
    try: 
        logger.info("Starting the main process")
        check_input_paths()
    
        ####################################################################
        # 1/ load the data using the ICA method and execute the analysis
        ####################################################################

        logger.info("entering inside the analysis")
        eeg_data, csv_data = run_app()
        logger.info("analysis finished successfully")
        
        ####################################################################
        # 2/ Save the preprocessed data to the output path
        ####################################################################


        output_file, csv_file = output_path(OUTPUT_PATH, eeg_data)
        csv_creator(csv_data, csv_file)
        logger.info(f"Preprocessed data saved to {output_file}")
        logger.info(f"CSV file saved to {csv_file}")
    except eeg_ica_Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)