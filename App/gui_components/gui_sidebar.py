import pandas as pd
import streamlit as st
from dss import load_data

# Predefined datasets for selection from the sidebar
data_files = {
    "Mini Dataset": "../Data/mini.csv",
    "Medium Dataset": "../Data/medium.csv",
    "Large Dataset": "../Data/many.csv",
    "Largest Dataset": "../Data/manyLarge.csv"
}

def render_sidebar():
    """
    Render the sidebar with user inputs and dataset selection/upload functionality.

    Returns:
    - vehicle_capacity (int): The capacity of each vehicle.
    - cost_per_km (float): Cost per kilometer of travel.
    - fixed_cost_per_truck (float): Fixed cost per truck.
    - data (pd.DataFrame): The loaded customer and company data.
    """
    st.sidebar.title(":violet[Choose your values:]")  # Sidebar title for user inputs

    # User inputs for vehicle parameters
    vehicle_capacity = st.sidebar.number_input("**Vehicle Capacity**", min_value=1, value=10)
    cost_per_km = st.sidebar.number_input("**Costs per KM (€)**", min_value=0.0, value=2.5, format="%.2f")
    fixed_cost_per_truck = st.sidebar.number_input("**Fixed Costs per Truck (€)**", min_value=0.0, value=50.0, format="%.2f")

    # File uploader to allow users to upload custom datasets
    uploaded_file = st.sidebar.file_uploader("**Upload your CSV file**", type=["csv"])

    # Initialize session state variables for selected files
    if "selected_file" not in st.session_state:
        st.session_state.selected_file = "Mini Dataset"  # Default to the mini dataset
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None

    # Handle dataset selection based on user input
    if uploaded_file is not None:
        # If a file is uploaded, use it as the selected dataset
        st.session_state.uploaded_file = uploaded_file
        data = pd.read_csv(uploaded_file)
    else:
        # Use a predefined dataset if no file is uploaded
        st.session_state.uploaded_file = None
        st.session_state.selected_file = st.sidebar.selectbox(
            "**Select a predefined dataset**",
            options=list(data_files.keys()),
            index=list(data_files.keys()).index(st.session_state.selected_file)  # Maintain previous selection
        )
        data = pd.read_csv(data_files[st.session_state.selected_file])  # Load the selected predefined dataset

    # Load and process the data using the provided function
    data = load_data(data=data)

    # Return the user input values and the loaded dataset
    return vehicle_capacity, cost_per_km, fixed_cost_per_truck, data
