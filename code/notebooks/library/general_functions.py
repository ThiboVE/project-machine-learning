import pandas as pd
import numpy as np
import json
import os

def json_folder_to_dataframe(folder_path: str, single_dict=True) -> pd.DataFrame:
    """
    Reads all JSON files in a folder and returns a pandas DataFrame.

    Parameters
    ----------
    folder_path : str
        Path to the directory containing JSON files.
    
    single_dict: bool
        True if each json file contains a single dictionary, False if each json file contains a list of dictionaries

    Returns
    -------
    pd.DataFrame
        A DataFrame where each row corresponds to one JSON file.
    """
    records = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)

            try:
                with open(file_path, "r") as file:
                    data = json.load(file)
                    if single_dict:
                        records.append(data)
                    else:
                        for dict in data:
                            records.append(dict)
            except Exception as e:
                print(f"Failed to load {filename}: {e}")

    # Convert list of dicts into a DataFrame
    df = pd.DataFrame(records)

    return df