# Decision Support System Logic
 
# Backend logic for ranking partnerships, calculating cost savings, and solving the VRP.
# Interfaces with data inputs and provides outputs for the GUI.

# Role: The central computational engine.
# Interactions:
# With gui.py: Receives user inputs (e.g., truck capacity, costs) and returns results like rankings, costs, and savings for display.
# With ranking.py: Calls functions to score and rank potential partnerships.
# With vehicle_routing_problem.py: Uses this module to solve VRPs for specific collaborations and validate rankings.

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


# def load_data(file_path):
#     # Load data and return it
#     data = pd.read_csv(file_path, skiprows=1, names=["name", "latitude", "longitude"])
#     return data

#df = load_data('../Data/mini.csv')
#df = load_data('../Data/medium.csv')

## USE YOUR OWN FILE PATH 
#df = load_data('C:/Users/malou/OneDrive/Documenten/VU/Business Analytics/YEAR 1 - 2024-2025 (Mc)/Project Optimization of Business Processes/Project-OBP/Data/mini.csv')


