import streamlit as st
import pandas as pd
from dss import load_data

data_files = {
    "Mini Dataset": "../Data/mini.csv",
    "Medium Dataset": "../Data/medium.csv",
    "Large Dataset": "../Data/many.csv",
    "Largest Dataset": "../Data/manyLarge.csv"
}

def render_sidebar():
    """Renders the sidebar and returns user inputs and loaded data."""
    # Sidebar inputs
    vehicle_capacity = st.sidebar.number_input("Vehicle Capacity", min_value=1, value=10)
    cost_per_km = st.sidebar.number_input("Costs per KM (€)", min_value=0.0, value=2.5, format="%.2f")
    fixed_cost_per_truck = st.sidebar.number_input("Fixed Costs per Truck (€)", min_value=0.0, value=50.0, format="%.2f")

    # File uploader
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

    # Initialize session state
    if "selected_file" not in st.session_state:
        st.session_state.selected_file = "Mini Dataset"
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None

    # Handle file upload or dataset selection
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        data = pd.read_csv(uploaded_file)
    else:
        st.session_state.uploaded_file = None
        st.session_state.selected_file = st.sidebar.selectbox(
            "Select a predefined dataset",
            options=list(data_files.keys()),
            index=list(data_files.keys()).index(st.session_state.selected_file)
        )
        data = pd.read_csv(data_files[st.session_state.selected_file])

    # Load the data
    data = load_data(data=data)

    return vehicle_capacity, cost_per_km, fixed_cost_per_truck, data
