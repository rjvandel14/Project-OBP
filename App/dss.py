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

def calculate_collaboration(vehicle_capacity, cost_per_km, fixed_cost, company_a, company_b):
    """
    Simulates a cost analysis for collaboration between two companies.
    """
    # Example: Simulated individual and collaboration costs
    cost_a = vehicle_capacity * cost_per_km + fixed_cost  # Cost for Company A
    cost_b = vehicle_capacity * cost_per_km * 1.2 + fixed_cost  # Cost for Company B
    collaboration_cost = (vehicle_capacity * cost_per_km * 1.1 + fixed_cost) * 1.5  # Collaboration cost

    # Calculate savings
    savings = (cost_a + cost_b) - collaboration_cost

    results = {
        "cost_a": round(cost_a, 2),
        "cost_b": round(cost_b, 2),
        "collaboration_cost": round(collaboration_cost, 2),
        "savings": round(savings, 2),
    }
    return results

