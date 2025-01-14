# Partnership ranking methods

# Implements ranking logic and scoring features like overlap and cost savings.
# Validates rankings using heuristics and exact VRP results.

# Role: Implements ranking logic.
# Interactions:
# With dss.py:
# Provides ranking algorithms and outputs scores based on features like overlap and savings potential.
# Ensures rankings are consistent with DSS results by validating with small VRP solutions.
# With routing.py:
# Compares rankings to exact VRP solutions for validation (top and bottom-ranked partnerships).
import folium
import pandas as pd
import streamlit as st
from dss import load_data

df = load_data('../Data/mini.csv')

def get_mock_ranking():
    """
    Returns a mock ranking table for collaborations.
    """
    mock_data = pd.DataFrame({
    "Rank": [1, 2, 3],
    "Company A": ["Company 1", "Company 2", "Company 3"],
    "Company B": ["Company 4", "Company 5", "Company 6"],
    })

    return mock_data

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

# Depot coordinates
depot_lat = 52.16521
depot_lon = 5.17215
create_partnership_map(df, depot_lat, depot_lon, output_file='partnership_map.html')

def get_min_max_ranking(distance_matrix, df, company_names):
    """
    Computes a ranking table for collaborations using the min-max method.
    
    Parameters:
    - distance_matrix (pd.DataFrame): A distance matrix containing pairwise distances between depot and customers.
    - df (pd.DataFrame): The original DataFrame with company and customer data.
    - company_names (list): A list of unique company names.
    
    Returns:
    - pd.DataFrame: A ranking table with columns ['Rank', 'Company A', 'Company B', 'Min_Max_Score'].
    """
    partnership_scores = []

    # Iterate through all pairs of companies
    for company1 in company_names:
        for company2 in company_names:
            if company1 != company2:  # Avoid self-comparison
                # Get the customers for each company
                customers1 = df[df['name'] == company1].index.tolist()
                customers2 = df[df['name'] == company2].index.tolist()

                # Get the maximum inter-customer distance
                max_inter_customer = distance_matrix.iloc[customers1, customers2].max().max()

                # Get the maximum distance to the depot for both companies
                max_depot_distance = max(
                    distance_matrix.iloc[customers1, 0].max(),  # Depot column is assumed to be the first column
                    distance_matrix.iloc[customers2, 0].max()
                )

                # Calculate the min-max score
                min_max_score = max(max_inter_customer, max_depot_distance)

                # Append the result
                partnership_scores.append((company1, company2, min_max_score))

    # Convert the results to a DataFrame for easier handling
    partnership_df = pd.DataFrame(partnership_scores, columns=['Company A', 'Company B', 'Min_Max_Score'])

    # Sort the partnerships by score in ascending order and add ranking
    partnership_df = partnership_df.sort_values('Min_Max_Score', ascending=True)
    partnership_df['Rank'] = range(1, len(partnership_df) + 1)

    # Reorder columns for output
    partnership_df = partnership_df[['Rank', 'Company A', 'Company B', 'Min_Max_Score']]

    return partnership_df

