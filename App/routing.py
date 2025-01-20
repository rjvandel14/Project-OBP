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
import folium
import networkx as nx
import streamlit as st
import pandas as pd
import folium
from vrpy import VehicleRoutingProblem
from dss import depot_lat
from dss import depot_lon

    
# Function to solve VRP for a given dataset
def solve_vrp(data, vehicle_capacity, cost_per_km, fixed_cost_per_truck, distance_matrix, timelimit):
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
    vrp.solve(cspy=True, time_limit= timelimit)

    return vrp.best_value, vrp.best_routes

def all_cvrp(vehicle_capacity, cost_per_km, fixed_cost_per_truck, company_a, company_b, data, dmatrix, timelimit = False):
     # Define companies to collaborate
    collaborating_companies = (company_a, company_b)

    company1_data = data.loc[data['name'] == company_a].copy()
    company2_data = data.loc[data['name'] == company_b].copy()

    # Combine data for collaborating companies
    if collaborating_companies:
        collaboration_data = data[data['name'].isin(collaborating_companies)].copy()
        collaboration_data['name'] = "Collaboration"  # Label as one entity
        data = pd.concat([data[~data['name'].isin(collaborating_companies)], collaboration_data])
    
    if timelimit:
        timelimit = 10 #+ len(collaboration_data['name'])

    # Solve VRP for individual companies
    cost_a, route_a = solve_vrp(company1_data, vehicle_capacity, cost_per_km, fixed_cost_per_truck, dmatrix, timelimit)
    cost_b, route_b = solve_vrp(company2_data, vehicle_capacity, cost_per_km, fixed_cost_per_truck, dmatrix, timelimit)

    # Solve VRP for combined companies
    combined_cost, route_combined = solve_vrp(collaboration_data, vehicle_capacity, cost_per_km, fixed_cost_per_truck, dmatrix, timelimit)

    result = {
    "Scenario": [company_a, company_b, "Collaboration"],
    "Total Cost": [round(cost_a,2), round(cost_b,2), round(combined_cost,2)],
    "Truck Cost": [len(route_a) * fixed_cost_per_truck, len(route_b) * fixed_cost_per_truck, len(route_combined) * fixed_cost_per_truck],
    "Driving Cost": [round(cost_a,2) - len(route_a) * fixed_cost_per_truck, round(cost_b,2) - len(route_b) * fixed_cost_per_truck, round(combined_cost,2) - len(route_combined) * fixed_cost_per_truck],
    "Routes": [route_a, route_b, route_combined]
    }

    return result

# Plots a map with the CVRP routes
def plot_routes_map(df, depot_lat, depot_lon, company_a, company_b, routes = None, output_file='map.html'):
    # Create a Folium map centered at the depot
    m = folium.Map(location=[depot_lat, depot_lon], zoom_start=12)

    # Add the depot marker
    folium.Marker(
        location=[depot_lat, depot_lon],
        popup="Depot",
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)

    # Filter the dataframe for the two selected companies
    filtered_df = df[df['name'].isin([company_a, company_b])]

    # Assign a unique color for each company in the selected companies
    colors = ['blue', 'green', 'purple', 'orange', 'darkred', 'darkblue', 'cadetblue', 'lightgreen']  # Add more if needed
    color_map = {company_a: colors[0], company_b: colors[1]}  # Assign colors to the two companies

    # Add customer markers for the selected companies
    for _, row in filtered_df.iterrows():
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=f"Customer of {row['name']}",  # Display company name in the popup
            icon=folium.Icon(color=color_map[row['name']])
        ).add_to(m)

    if routes:
        for route_id, route in routes.items():
            route_coords = []
            # Loop through the route and get coordinates for each customer (except 'Source' and 'Sink')
            for customer_index in route[1:-1]:
                # Find the customer name and coordinates by index
                customer_row = df.iloc[customer_index]
                route_coords.append((customer_row['lat'], customer_row['lon']))

            # Add polyline for this route
            folium.PolyLine(route_coords, color="blue", weight=2.5, opacity=1).add_to(m)

             # Add line from the first customer location to the depot
            first_customer_index = route[1]
            first_customer_row = df.iloc[first_customer_index]
            folium.PolyLine(
                locations=[(first_customer_row['lat'], first_customer_row['lon']),
                           (depot_lat, depot_lon)],
                color="blue", weight=2.5, opacity=1
            ).add_to(m)

            # Add line from the last customer location to the depot
            last_customer_index = route[-2]
            last_customer_row = df.iloc[last_customer_index]
            folium.PolyLine(
                locations=[(last_customer_row['lat'], last_customer_row['lon']),
                           (depot_lat, depot_lon)],
                color="blue", weight=2.5, opacity=1
            ).add_to(m)

    # Save the map to an HTML file
    m.save(output_file)

    # Streamlit output
    st.title("Partnership Map")
    st.write("Interactive map showing company customers and depot.")
    st.components.v1.html(m._repr_html_(), height=600)

    return m

def mock_cvrp(vehicle_capacity, cost_per_km, fixed_cost_per_truck):
    """
    Returns a mock cvrp.
    """
    mock_data = {
    "Scenario": ["Company A", "Company B", "Collaboration"],
    "Cost": [500.0, 600.0, 900.0],
    "Routes": [
        [[0, 2, 3, 0], [0, 4, 1, 0]],  # Routes for Company A
        [[0, 5, 6, 0]],                # Routes for Company B
        [[0, 2, 3, 5, 6, 0], [0, 4, 1, 0]]  # Routes for collaboration
    ]}
    return mock_data