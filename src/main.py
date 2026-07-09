from logger_init import logger
from run import execute_analysis


DATA_PATH = "../data"
OUTPUT_PATH = "../output"


if __name__ == "__main__":
    logger.info("Starting the main process")
    execute_analysis(DATA_PATH,OUTPUT_PATH)
    
    
