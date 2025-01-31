import folium
import pandas as pd
import streamlit as st
from routing import all_cvrp
from scipy.stats import spearmanr

# Visualize the customer locations given a company
def create_partnership_map(df, depot_lat, depot_lon, output_file='map.html'):
    """
    Create an interactive map showing company customer locations and depot.

    Parameters:
    - df (pd.DataFrame): DataFrame containing customer data with columns ['latitude', 'longitude', 'name'].
    - depot_lat (float): Latitude of the depot.
    - depot_lon (float): Longitude of the depot.
    - output_file (str): Name of the HTML file to save the map.

    Returns:
    - folium.Map: The interactive Folium map object.
    """
    # Create a Folium map centered at the depot
    m = folium.Map(location=[depot_lat, depot_lon], zoom_start=12)

    # Add the depot marker
    folium.Marker(
        location=[depot_lat, depot_lon],
        popup="Depot",
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)

    # Assign a unique color for each company
    company_names = df['name'].unique()  # Use the 'name' column to identify companies
    colors = ['blue', 'green', 'purple', 'orange', 'darkred', 'darkblue', 'cadetblue', 'lightgreen']  # Add more if needed
    color_map = {name: colors[i % len(colors)] for i, name in enumerate(company_names)}

    # Add customer markers for each company
    for _, row in df.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"Customer of {row['name']}",  # Display company name in the popup
            icon=folium.Icon(color=color_map[row['name']])
        ).add_to(m)

    # Save the map to an HTML file
    m.save(output_file)

    # Streamlit output
    st.title("Partnership Map")
    st.write("Interactive map showing company customers and depot.")
    st.components.v1.html(m._repr_html_(), height=600)

    return m

def get_validation(vehicle_capacity, cost_per_km, fixed_cost_per_truck, data, dmatrix, ranking):
    selected_pairs = pd.concat([ranking.head(3), ranking.tail(3)])
    # Evaluate VRP for selected pairs
    evaluation_results = []
    for _, row in selected_pairs.iterrows():
        company_a = row["Company A"]
        company_b = row["Company B"]
        
        # Solve VRP for the selected companies using all_cvrp
        vrp_result = all_cvrp(
            vehicle_capacity,
            cost_per_km,
            fixed_cost_per_truck,
            company_a,
            company_b,
            data,
            dmatrix 
        )

        # Append results
        evaluation_results.append({
            "Company A": company_a,
            "Company B": company_b,
            "Heuristic Rank": row["Rank"],
            "Score": row["Score"],
            "VRP Collaboration Saving Cost": vrp_result["Total Cost"][0] + vrp_result["Total Cost"][1] - vrp_result["Total Cost"][2]  # Cost savings
        })

    # Create DataFrame with results
    evaluation_df = pd.DataFrame(evaluation_results)
    
    # Compute Spearman Rank Correlation for K-Means rankings
    heuristic_scores = evaluation_df["Score"]
    vrp_scores = evaluation_df["VRP Collaboration Saving Cost"]

    spearman_corr_dbscan, p_value_dbscan = spearmanr(heuristic_scores, vrp_scores)

    # Display results
    print("Evaluation Results (Top 3 and Bottom 3) for Ranking:")
    print(evaluation_df)
    print(f"\nSpearman Rank Correlation for Ranking: {spearman_corr_dbscan:.2f}")
    print(f"P-Value: {p_value_dbscan:.2e}")
