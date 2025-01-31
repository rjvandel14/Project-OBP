import pandas as pd

# Variables
depot_lat = 52.16521
depot_lon = 5.17215

def load_data(file_path=None, data=None):
    """
    Load data either from a file path or directly from a provided DataFrame.
    """
    if data is not None:
        # Use the provided DataFrame
        return data
    elif file_path is not None:
        # Load data from the given file path
        data = pd.read_csv(file_path, skiprows=1, names=["name", "lat", "lon"])
        return data
    else:
        raise ValueError("Either 'file_path' or 'data' must be provided.")

