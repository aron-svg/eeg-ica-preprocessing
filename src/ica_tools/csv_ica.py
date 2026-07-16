from config import (MANUAL_MODE,
N_COMPONENTS,
LOW_FREQUENCY,
HIGH_FREQUENCY,
NOTCH_FREQUENCY,
ARTEFACT_MAX,
FLAT_THRESHOLD,
BAD_CHANNEL_PERCENT,
VARIANCE_Z_THRESHOLD)

def csv_creator(data, output_path):
    """
    Creates a CSV file from the provided data. The csv file is gonna contain all the main configuration find in the
    config.py file, and the data is gonna be saved in the output_path provided.
    The CSV will also contains the choices made by the user in the manual mode, if the manual mode is set to True.
    If the automatic mode is set to True, the CSV will contain the automatic choices made by the algorithm.

    Parameters:
    - data: The data to be written to the CSV file. It should be in a format that can be converted to a DataFrame.
    - output_path: The path where the CSV file will be saved.

    Returns:
    - None
    """
    import pandas as pd

    config_values = {
        "MANUAL_MODE": MANUAL_MODE,
        "N_COMPONENTS": N_COMPONENTS,
        "LOW_FREQUENCY": LOW_FREQUENCY,
        "HIGH_FREQUENCY": HIGH_FREQUENCY,
        "NOTCH_FREQUENCY": NOTCH_FREQUENCY,
        "ARTEFACT_MAX": ARTEFACT_MAX,
        "FLAT_THRESHOLD": FLAT_THRESHOLD,
        "BAD_CHANNEL_PERCENT": BAD_CHANNEL_PERCENT,
        "VARIANCE_Z_THRESHOLD": VARIANCE_Z_THRESHOLD,
    }

    # Convert the data (manual/automatic choices) to a DataFrame
    df = pd.DataFrame(data)

    # Attach the pipeline configuration used for this run to every row
    for key, value in config_values.items():
        df[key] = value

    # Save the DataFrame to a CSV file
    df.to_csv(output_path, index=False)
