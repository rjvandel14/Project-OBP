import folium
import networkx as nx
import pandas as pd
import folium
from vrpy import VehicleRoutingProblem
import csv

def solve_vrp(data, vehicle_capacity, cost_per_km, fixed_cost_per_truck, distance_matrix, timelimit):
    """
    Solve the Vehicle Routing Problem (VRP) for a given dataset of customer locations.

    Parameters:
    - data (pd.DataFrame): DataFrame containing customer details.
    - vehicle_capacity (int): Capacity of each vehicle (in units).
    - cost_per_km (float): Cost per kilometer for travel.
    - fixed_cost_per_truck (float): Fixed cost per truck.
    - distance_matrix (pd.DataFrame): Matrix with distances between customers and the depot.
    - timelimit (int): Time limit for solving the VRP (in seconds).

    Returns:
    - (float, list): The cost of the optimal solution and the corresponding routes.
    """
    
    # Create a directed graph
    G = nx.DiGraph()

    # Add "Source" and "Sink" nodes for the depot
    G.add_node("Source", demand=0)
    G.add_node("Sink", demand=0)

    # Add customer nodes
    for idx, row in data.iterrows():
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

def all_cvrp(vehicle_capacity, cost_per_km, fixed_cost_per_truck, company_a, company_b, data, dmatrix, timelimit):
    """
    Solve the VRP for individual companies and their collaboration scenario.

    Parameters:
    - vehicle_capacity (int): Vehicle capacity.
    - cost_per_km (float): Cost per kilometer of travel.
    - fixed_cost_per_truck (float): Fixed cost associated with each truck.
    - company_a, company_b (str): Names of the companies being analyzed.
    - data (pd.DataFrame): DataFrame containing customer locations and their respective companies.
    - dmatrix (pd.DataFrame): Distance matrix with distances between locations.
    - timelimit (int): Time limit for solving the VRP (in seconds).

    Returns:
    - dict: Results with costs, truck requirements, driving costs, and routes.
    """
    
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


def plot_routes_map(df, depot_lat, depot_lon, company_a, company_b, routes=None, output_file='map.html', csv_file='routes.csv'):
    """
    Plot the VRP routes on an interactive map and export route details to a CSV.

    Parameters:
    - df (pd.DataFrame): DataFrame containing customer locations.
    - depot_lat, depot_lon (float): Latitude and longitude of the depot.
    - company_a, company_b (str): Companies whose routes are being visualized.
    - routes (dict, optional): Routes from the VRP solution.
    - output_file (str): Name of the HTML file to save the map.
    - csv_file (str): Name of the CSV file to save route data.

    Returns:
    - (folium.Map, str): The generated map and path to the CSV file.
    """
    
    # Create a Folium map centered at the depot
    m = folium.Map(location=[depot_lat, depot_lon], zoom_start=7)

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
    for idx, row in filtered_df.iterrows():
        # Determine customer number
        company_name = row['name']
        customer_number = list(filtered_df[filtered_df['name'] == company_name].index).index(idx) + 1

        # Add marker for customers
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=f"{company_name} {customer_number}",  # Shows company name and number in popup
            icon=folium.Icon(color=color_map[company_name])
        ).add_to(m)

    # Prepare CSV data
    csv_data = []

    if routes:
        for route_id, route in routes.items():
            # Loop through the route and get details for each customer (except 'Source' and 'Sink')
            for customer_index in route[1:-1]:
                customer_row = df.iloc[customer_index]
                company_name = customer_row['name']

                # Generate label as "Company N"
                customer_number = list(filtered_df[filtered_df['name'] == company_name].index).index(customer_index) + 1

                # Append the data to the CSV output
                csv_data.append({
                    "route_id": route_id,
                    "company": company_name,
                    "customer_number": customer_number,
                })

            # Add lines to the map for visualization
            route_coords = [
                (df.iloc[customer_index]['lat'], df.iloc[customer_index]['lon']) 
                for customer_index in route[1:-1]
            ]
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

    # Export routes to a CSV file
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["route_id", "company", "customer_number"])
        writer.writeheader()
        writer.writerows(csv_data)

    return m, csv_file
