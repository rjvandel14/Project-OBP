import streamlit as st
from gui_components.gui_sidebar import render_sidebar
from gui_components.gui_ranking import render_ranking
from gui_components.gui_analysis import render_analysis
from osrm_dmatrix import compute_distance_matrix

# Cache the computation of the distance matrix
@st.cache_data(show_spinner=False)
def get_distance_matrix(data):
    # Compute the distance matrix here
    return compute_distance_matrix(data)

st.image("1777.jpg", use_container_width=True)

# Title
st.title(":violet[Cost-Saving Parnership Evaluator]")

# Sidebar inputs and dataset
vehicle_capacity, cost_per_km, fixed_cost_per_truck, data = render_sidebar()

# Display rankings
if data is not None:

    dmatrix = get_distance_matrix(data)
    render_ranking(dmatrix, data,vehicle_capacity, cost_per_km, fixed_cost_per_truck)

    # Analyze collaboration
    render_analysis(vehicle_capacity, cost_per_km, fixed_cost_per_truck, data, dmatrix)
