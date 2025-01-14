# Vehicle routing and distance calculations

# Solves VRPs for individual and collaborative scenarios.
# Outputs route costs, distances, and truck requirements.

# Role: Solves VRPs for cost and route optimization.
# Interactions:
# With dss.py:
# Provides VRP solutions (costs, routes, truck usage) for individual companies or collaborations.
# Supports cost/savings calculations for DSS analysis.
# With ranking.py:
# Validates rankings by solving VRPs for selected partnerships.
# Provides ground truth for heuristic ranking evaluations.

def mock_cvrp(vehicle_capacity, cost_per_km, fixed_cost_per_truck):
    """
    Returns a mock cvrp.
    """
    mock_data = {
    "Scenario": ["Company A", "Company B", "Collaboration"],
    "Cost (€)": [500.0, 600.0, 900.0],
    "Trucks Required": [2, 1, 2],
    "Total Distance (km)": [120.0, 80.0, 150.0],
    "Routes": [
        [[0, 2, 3, 0], [0, 4, 1, 0]],  # Routes for Company A
        [[0, 5, 6, 0]],                # Routes for Company B
        [[0, 2, 3, 5, 6, 0], [0, 4, 1, 0]]  # Routes for collaboration
    ]}
    return mock_data