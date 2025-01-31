import folium
import networkx as nx
import pandas as pd
from vrpy import VehicleRoutingProblem
import csv

# ----------------- VRP Solver Functions -----------------

def solve_vrp(data, vehicle_capacity, cost_per_km, fixed_cost_per_truck, distance_matrix, timelimit):
    """
    Solve the Vehicle Routing Problem (VRP) for the given dataset.

    Parameters:
    - data (pd.DataFrame): DataFrame containing customer locations.
    - vehicle_capacity (int): Capacity of the vehicle in units.
    - cost_per_km (float): Cost per kilometer for travel.
    - fixed_cost_per_truck (float): Fixed cost per truck.
    - distance_matrix (pd.DataFrame): Distance matrix between customers and the depot.
    - timelimit (int): Maximum time limit for solving the VRP (in seconds).

    Returns:
    - (float, list): The cost of the optimal solution and the best routes.
    """
    # Create a directed graph representing the problem
    G = nx.DiGraph()

    # Add nodes for the depot (Source and Sink)
    G.add_node("Source", demand=0)
    G.add_node("Sink", demand=0)

    # Add customer nodes with a demand of 1 unit per customer
    for idx, row in data.iterrows():
        G.add_node(idx, demand=1)

    # Add edges representing distances and costs between customers
    for i, from_row in data.iterrows():
        for j, to_row in data.iterrows():
            if i != j:  # Avoid self-loops
                distance = distance_matrix.iloc[i + 1, j + 1]  # Adjust for depot offset
                G.add_edge(i, j, cost=distance * cost_per_km)

        # Connect the depot to customer nodes and vice versa
        distance_from_depot = distance_matrix.iloc[0, i + 1]
        G.add_edge("Source", i, cost=distance_from_depot * cost_per_km)

        distance_to_depot = distance_matrix.iloc[i + 1, 0]
        G.add_edge(i, "Sink", cost=distance_to_depot * cost_per_km)

    # Set up and solve the VRP using the VRPy solver
    vrp = VehicleRoutingProblem(G)
    vrp.load_capacity = vehicle_capacity
    vrp.fixed_cost = fixed_cost_per_truck
    vrp.solve(cspy=True, time_limit=timelimit)

    return vrp.best_value, vrp.best_routes


def all_cvrp(vehicle_capacity, cost_per_km, fixed_cost_per_truck, company_a, company_b, data, dmatrix, timelimit):
    """
    Solve the VRP for individual companies and their collaboration.

    Parameters:
    - vehicle_capacity (int): Capacity of the vehicle.
    - cost_per_km (float): Cost per kilometer.
    - fixed_cost_per_truck (float): Fixed cost per truck.
    - company_a, company_b (str): Names of the companies being analyzed.
    - data (pd.DataFrame): Customer data for both companies.
    - dmatrix (pd.DataFrame): Distance matrix.
    - timelimit (int): Time limit for the VRP solver.

    Returns:
    - dict: Dictionary containing costs, routes, and comparison results.
    """
    # Select data for the individual companies
    company1_data = data.loc[data['name'] == company_a].copy()
    company2_data = data.loc[data['name'] == company_b].copy()

    # Combine data for collaboration scenario
    collaboration_data = data[data['name'].isin([company_a, company_b])].copy()
    collaboration_data['name'] = "Collaboration"

    # Solve VRP for individual companies
    cost_a, route_a = solve_vrp(company1_data, vehicle_capacity, cost_per_km, fixed_cost_per_truck, dmatrix, timelimit)
    cost_b, route_b = solve_vrp(company2_data, vehicle_capacity, cost_per_km, fixed_cost_per_truck, dmatrix, timelimit)

    # Solve VRP for the combined companies
    combined_cost, route_combined = solve_vrp(collaboration_data, vehicle_capacity, cost_per_km, fixed_cost_per_truck, dmatrix, timelimit)

    # Create result dictionary
    result = {
        "Scenario": [company_a, company_b, "Collaboration"],
        "Total Cost": [round(cost_a, 2), round(cost_b, 2), round(combined_cost, 2)],
        "Truck Cost": [len(route_a) * fixed_cost_per_truck, len(route_b) * fixed_cost_per_truck, len(route_combined) * fixed_cost_per_truck],
        "Driving Cost": [round(cost_a, 2) - len(route_a) * fixed_cost_per_truck, round(cost_b, 2) - len(route_b) * fixed_cost_per_truck, round(combined_cost, 2) - len(route_combined) * fixed_cost_per_truck],
        "Routes": [route_a, route_b, route_combined]
    }

    return result

# ----------------- Route Visualization -----------------

def plot_routes_map(df, depot_lat, depot_lon, company_a, company_b, routes=None, output_file='map.html', csv_file='routes.csv'):
    """
    Plot the CVRP routes on an interactive map and export route details to a CSV file.

    Parameters:
    - df (pd.DataFrame): DataFrame containing customer locations.
    - depot_lat, depot_lon (float): Latitude and longitude of the depot.
    - company_a, company_b (str): Names of the companies being visualized.
    - routes (dict, optional): Dictionary of routes from the VRP solution.
    - output_file (str): Name of the HTML file to save the map.
    - csv_file (str): Name of the CSV file to export route details.

    Returns:
    - folium.Map: The interactive map with plotted routes.
    - str: Name of the CSV file with route details.
    """
    # Initialize the map centered at the depot
    m = folium.Map(location=[depot_lat, depot_lon], zoom_start=7)

    # Add the depot marker
    folium.Marker(
        location=[depot_lat, depot_lon],
        popup="Depot",
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)

    # Filter customer data for the selected companies
    filtered_df = df[df['name'].isin([company_a, company_b])]

    # Assign unique colors to the companies
    colors = ['blue', 'green']
    color_map = {company_a: colors[0], company_b: colors[1]}

    # Plot customer markers
    for idx, row in filtered_df.iterrows():
        customer_number = list(filtered_df[filtered_df['name'] == row['name']].index).index(idx) + 1
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=f"{row['name']} {customer_number}",
            icon=folium.Icon(color=color_map[row['name']])
        ).add_to(m)

    # Plot routes and export details to a CSV file
    csv_data = []
    if routes:
        for route_id, route in routes.items():
            route_coords = [(df.iloc[customer_index]['lat'], df.iloc[customer_index]['lon']) for customer_index in route[1:-1]]
            folium.PolyLine(route_coords, color="blue", weight=2.5, opacity=1).add_to(m)

            # Prepare CSV data
            for customer_index in route[1:-1]:
                customer_row = df.iloc[customer_index]
                csv_data.append({"route_id": route_id, "company": customer_row['name'], "customer_number": customer_index + 1})

    # Save map and CSV
    m.save(output_file)
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["route_id", "company", "customer_number"])
        writer.writeheader()
        writer.writerows(csv_data)

    return m, csv_file
