# Decision Support System Logic
 
# Backend logic for ranking partnerships, calculating cost savings, and solving the VRP.
# Interfaces with data inputs and provides outputs for the GUI.

# Role: The central computational engine.
# Interactions:
# With gui.py: Receives user inputs (e.g., truck capacity, costs) and returns results like rankings, costs, and savings for display.
# With ranking.py: Calls functions to score and rank potential partnerships.
# With vehicle_routing_problem.py: Uses this module to solve VRPs for specific collaborations and validate rankings.

import pandas as pd

def load_data(file_path):
    # Load data and return it
    data = pd.read_csv(file_path, skiprows=1, names=["name", "latitude", "longitude"])
    return data