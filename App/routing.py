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

import pandas as pd
import math
from vrpy import VehicleRoutingProblem
import networkx as nx
import folium

from dss import depot_lat
from dss import depot_lon

    
# Function to solve VRP for a given dataset
def solve_vrp(data, vehicle_capacity, cost_per_km, fixed_cost_per_truck, distance_matrix):
    # Create a directed graph
    G = nx.DiGraph()

    # Add "Source" and "Sink" nodes for the depot
    depot = {'lat': depot_lat, 'lon': depot_lon}
    G.add_node("Source", demand=0)
    G.add_node("Sink", demand=0)

    # Add customer nodes
    for idx, row in data.iterrows():
        customer_index = row.name + 1
        G.add_node(idx, demand=1)  # Assuming demand of 1 for each customer

    # Add edges with costs (distances)
    for i, from_row in data.iterrows():
        for j, to_row in data.iterrows():          
            if i != j:  # No self-loops
                distance = distance_matrix.iloc[i + 1, j + 1]  # +1 to account for the depot row/column
                G.add_edge(i, j, cost=distance * cost_per_km)

        # Connect "Source" to all customer nodes
        distance_from_depot = distance_matrix.iloc[0, i + 1]  # +1 for depot offset
        G.add_edge("Source", i, cost=distance_from_depot * cost_per_km)

        # Connect all customer nodes to "Sink"
        distance_to_depot = distance_matrix.iloc[i + 1, 0]  # +1 for customer index and 0 for depot row
        G.add_edge(i, "Sink", cost=distance_to_depot * cost_per_km)

    # Solve VRP
    vrp = VehicleRoutingProblem(G)
    vrp.load_capacity = vehicle_capacity
    vrp.fixed_cost = fixed_cost_per_truck
    vrp.solve()

    return vrp.best_value, vrp.best_routes

def all_cvrp(vehicle_capacity, cost_per_km, fixed_cost_per_truck, company_a, company_b, data, dmatrix):
     # Define companies to collaborate
    collaborating_companies = (company_a, company_b)

    company1_data = data.loc[data['name'] == company_a].copy()
    company2_data = data.loc[data['name'] == company_b].copy()

    # Combine data for collaborating companies
    if collaborating_companies:
        collaboration_data = data[data['name'].isin(collaborating_companies)].copy()
        collaboration_data['name'] = "Collaboration"  # Label as one entity
        data = pd.concat([data[~data['name'].isin(collaborating_companies)], collaboration_data])

    # Solve VRP for individual companies
    cost_a, route_a = solve_vrp(company1_data, vehicle_capacity, cost_per_km, fixed_cost_per_truck, dmatrix )
    cost_b, route_b = solve_vrp(company2_data, vehicle_capacity, cost_per_km, fixed_cost_per_truck, dmatrix)

    # Solve VRP for combined companies
    combined_cost, route_combined = solve_vrp(collaboration_data, vehicle_capacity, cost_per_km, fixed_cost_per_truck, dmatrix)

    result = {
    "Scenario": [company_a, company_b, "Collaboration"],
    "Cost (€)": [round(cost_a,2), round(cost_b,2), round(combined_cost,2)],
    "Routes": [route_a, route_b, route_combined
    ]}

    return result

def mock_cvrp(vehicle_capacity, cost_per_km, fixed_cost_per_truck):
    """
    Returns a mock cvrp.
    """
    mock_data = {
    "Scenario": ["Company A", "Company B", "Collaboration"],
    "Cost (€)": [500.0, 600.0, 900.0],
    "Routes": [
        [[0, 2, 3, 0], [0, 4, 1, 0]],  # Routes for Company A
        [[0, 5, 6, 0]],                # Routes for Company B
        [[0, 2, 3, 5, 6, 0], [0, 4, 1, 0]]  # Routes for collaboration
    ]}
    return mock_data