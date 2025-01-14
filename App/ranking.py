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

def bla():
    print("bla")

def get_mock_ranking():
    """
    Returns a mock ranking table for collaborations.
    """
    mock_data = pd.DataFrame({
    "Rank": [1, 2, 3],
    "Company A": ["Company 1", "Company 2", "Company 3"],
    "Company B": ["Company 4", "Company 5", "Company 6"],
    "Savings Company A (€)": [100.00, 80.50, 70.00],  # Example savings for Company A
    "Savings Company B (€)": [150.00, 120.00, 80.75],  # Example savings for Company B
    "Savings (€)": [250.00, 200.50, 150.75]  # Total savings
    })

    return mock_data

# # Example: Replace this with your actual file path
# file_path = r'C:\Users\daydo\Downloads\mini.csv'

# # Load the CSV file
# df = pd.read_csv(file_path)

# Coordinates of the depot
# !! Are in the distance matrix, so for later, needs change
depot_lat = 52.16521
depot_lon = 5.17215

# Create a Folium map centered at the depot
m = folium.Map(location=[depot_lat, depot_lon], zoom_start=12)

# Add the depot marker
folium.Marker(
    location=[depot_lat, depot_lon],
    popup="Depot",
    icon=folium.Icon(color="red", icon="info-sign")
).add_to(m)

# Assign a unique color for each company
names = df['name'].unique()  # Use the 'name' column instead of 'company'
colors = ['blue', 'green', 'purple', 'orange', 'darkred', 'darkblue']  # Extend as needed
color_map = {name: colors[i % len(colors)] for i, name in enumerate(names)}

# Add customer markers for each company
for _, row in df.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=f"Customer of {row['name']}",  # Updated to use 'name'
        icon=folium.Icon(color=color_map[row['name']])
    ).add_to(m)
# Save the map to an HTML file
m.save('map.html')
st.write("Map saved as `map.html`. Open it in a browser to view the map.")

st.title("Partnership Map")
st.write("Interactive map showing company customers and depot.")
st.components.v1.html(m._repr_html_(), height=600)