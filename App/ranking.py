import folium
import pandas as pd
import streamlit as st
from routing import all_cvrp
from scipy.stats import spearmanr

# ----------------- Visualization Function -----------------

def create_partnership_map(df, depot_lat, depot_lon, output_file='map.html'):
    """
    Create an interactive map displaying customer locations and the depot for visualizing partnerships.

    Parameters:
    - df (pd.DataFrame): DataFrame containing columns ['latitude', 'longitude', 'name'] representing customers.
    - depot_lat (float): Latitude of the depot.
    - depot_lon (float): Longitude of the depot.
    - output_file (str): Name of the HTML file to save the map.

    Returns:
    - folium.Map: Interactive map object showing customers and depot locations.
    """
    # Initialize a Folium map centered at the depot
    m = folium.Map(location=[depot_lat, depot_lon], zoom_start=12)

    # Add a marker for the depot location
    folium.Marker(
        location=[depot_lat, depot_lon],
        popup="Depot",
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)

    # Assign a unique color to each company for differentiation
    company_names = df['name'].unique()
    colors = ['blue', 'green', 'purple', 'orange', 'darkred', 'darkblue', 'cadetblue', 'lightgreen']  # Expand as needed
    color_map = {name: colors[i % len(colors)] for i, name in enumerate(company_names)}

    # Add markers for each customer's location
    for _, row in df.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"Customer of {row['name']}",  # Display company name in the marker popup
            icon=folium.Icon(color=color_map[row['name']])
        ).add_to(m)

    # Save the map to an HTML file and display it on Streamlit
    m.save(output_file)

    # Streamlit output
    st.title("Partnership Map")
    st.write("Interactive map showing company customers and depot.")
    st.components.v1.html(m._repr_html_(), height=600)

    return m

# ----------------- Validation Function -----------------

def get_validation(vehicle_capacity, cost_per_km, fixed_cost_per_truck, data, dmatrix, ranking):
    """
    Validate the heuristic ranking by evaluating the VRP for the top 3 and bottom 3 pairs.

    Parameters:
    - vehicle_capacity (int): Capacity of the vehicle in units.
    - cost_per_km (float): Cost per kilometer for travel.
    - fixed_cost_per_truck (float): Fixed cost per truck.
    - data (pd.DataFrame): DataFrame containing company and customer data.
    - dmatrix (pd.DataFrame): Distance matrix for the customers.
    - ranking (pd.DataFrame): DataFrame containing ranked pairs of companies.

    Returns:
    - None: Prints evaluation results and Spearman Rank Correlation.
    """
    # Select the top 3 and bottom 3 company pairs from the ranking
    selected_pairs = pd.concat([ranking.head(3), ranking.tail(3)])

    evaluation_results = []  # List to store evaluation details

    # Evaluate each selected pair using VRP
    for _, row in selected_pairs.iterrows():
        company_a = row["Company A"]
        company_b = row["Company B"]

        # Solve the VRP for the selected pair of companies
        vrp_result = all_cvrp(
            vehicle_capacity,
            cost_per_km,
            fixed_cost_per_truck,
            company_a,
            company_b,
            data,
            dmatrix
        )

        # Calculate collaboration cost savings and store the results
        evaluation_results.append({
            "Company A": company_a,
            "Company B": company_b,
            "Heuristic Rank": row["Rank"],
            "Score": row["Score"],
            "VRP Collaboration Saving Cost": vrp_result["Total Cost"][0] + vrp_result["Total Cost"][1] - vrp_result["Total Cost"][2]
        })

    # Create a DataFrame to store evaluation results
    evaluation_df = pd.DataFrame(evaluation_results)

    # Calculate Spearman Rank Correlation between heuristic scores and VRP results
    heuristic_scores = evaluation_df["Score"]
    vrp_scores = evaluation_df["VRP Collaboration Saving Cost"]
    spearman_corr_dbscan, p_value_dbscan = spearmanr(heuristic_scores, vrp_scores)

    # Display evaluation results
    print("Evaluation Results (Top 3 and Bottom 3) for Ranking:")
    print(evaluation_df)
    print(f"\nSpearman Rank Correlation for Ranking: {spearman_corr_dbscan:.2f}")
    print(f"P-Value: {p_value_dbscan:.2e}")
