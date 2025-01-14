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
from distancematrix import distance_matrix

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

# # Depot coordinates
# depot_lat = 52.16521
# depot_lon = 5.17215
# create_partnership_map(df, depot_lat, depot_lon, output_file='partnership_map.html')

def get_min_max_ranking(dmatrix, df):
    """
    Computes a ranking table for collaborations using the min-max method.
    
    Parameters:
    - dmatrix (pd.DataFrame): The distance matrix.
    - df (pd.DataFrame): DataFrame with company and customer data.

    Returns:
    - pd.DataFrame: A ranking table with columns ['Rank', 'Company A', 'Company B', 'Min_Max_Score'].
    """
    partnership_scores = []
    company_names = df['name'].unique()  # Extract unique company names

    # Iterate through all unique pairs of companies
    for i, company1 in enumerate(company_names):
        for j, company2 in enumerate(company_names):
            if i < j:  # Ensure each pair is only processed once
                # Log the partnership being evaluated
                print(f"Evaluating partnership: {company1} and {company2}")

                # Get customer indices for both companies
                customers1 = df[df['name'] == company1].index.tolist()
                customers2 = df[df['name'] == company2].index.tolist()

                # Max inter-customer distance
                max_inter_customer = dmatrix.iloc[customers1, customers2].max().max()

                # Max depot distance
                max_depot_distance = max(
                    dmatrix.iloc[customers1, 0].max(),  # Depot assumed to be the first row/column
                    dmatrix.iloc[customers2, 0].max()
                )

                # Calculate min-max score
                min_max_score = max(max_inter_customer, max_depot_distance)

                # Append the result, including company names
                partnership_scores.append({
                    'Company A': company1,
                    'Company B': company2,
                    'Min_Max_Score': min_max_score
                })

    # Create the DataFrame
    partnership_df = pd.DataFrame(partnership_scores)

    # Sort by score
    partnership_df = partnership_df.sort_values('Min_Max_Score', ascending=True).reset_index(drop=True)

    # Ensure all rows are included and ranked, even with duplicate scores
    partnership_df['Rank'] = partnership_df.index + 1

    # Reorder columns for clarity
    return partnership_df[['Rank', 'Company A', 'Company B']]


dmatrix = distance_matrix()
ranking = get_min_max_ranking(dmatrix, df)
# Set pandas to display all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(ranking)