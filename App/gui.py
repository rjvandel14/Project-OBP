# GUI/dashboard code

# Handles user inputs (e.g., truck capacity, costs) and displays rankings, routes, and savings.
# Fetches analysis results from dss.py.

# Role: The user-facing component.
# Interactions:
# With dss.py:
# Sends user inputs (e.g., selected companies, truck parameters) to dss.py.
# Displays results (rankings, costs, and savings) fetched from dss.py.
# With Data Files:
# Allows users to import/export customer datasets or ranked results in CSV format.

import streamlit as st
from gui_components.gui_sidebar import render_sidebar
from gui_components.gui_ranking import render_ranking
from gui_components.gui_analysis import render_analysis
from distancematrix import compute_distance_matrix

# Cache the computation of the distance matrix
@st.cache_data
def get_distance_matrix(data):
    # Compute the distance matrix here
    return compute_distance_matrix(data)

# Title
st.title("Logistics Collaboration Dashboard")

# Sidebar inputs and dataset
vehicle_capacity, cost_per_km, fixed_cost_per_truck, data = render_sidebar()

# Display rankings
if data is not None:

    dmatrix = get_distance_matrix(data)
    render_ranking(dmatrix, data,vehicle_capacity, cost_per_km, fixed_cost_per_truck)

    # Analyze collaboration
    render_analysis(vehicle_capacity, cost_per_km, fixed_cost_per_truck, data, dmatrix)
