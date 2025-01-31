import pandas as pd

# Variables for the central depot location (latitude and longitude)
depot_lat = 52.16521
depot_lon = 5.17215

def load_data(file_path=None, data=None):
    """
    Load location data for customers either from a file or directly from a provided DataFrame.

    Parameters:
    - file_path (str, optional): Path to the CSV file containing customer locations.
    - data (pd.DataFrame, optional): Preloaded DataFrame with customer names and coordinates.

    Returns:
    - pd.DataFrame: A DataFrame containing customer names and coordinates.

    Raises:
    - ValueError: If neither 'file_path' nor 'data' is provided.
    """
    if data is not None:
        # If a DataFrame is provided, return it directly
        return data
    elif file_path is not None:
        # Load customer data from the specified CSV file
        data = pd.read_csv(file_path, skiprows=1, names=["name", "lat", "lon"])
        return data
    else:
        # Raise an error if neither input is given
        raise ValueError("Either 'file_path' or 'data' must be provided.")
