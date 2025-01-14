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
import pandas as pd
from dss import load_data
from dash import Dash, html
from ranking import get_min_max_ranking
from routing import mock_cvrp
from distancematrix import distance_matrix

# Title
st.title("Logistics Collaboration Dashboard")

# Sidebar inputs
vehicle_capacity = st.sidebar.number_input("Vehicle Capacity", min_value=1, value=10)
cost_per_km = st.sidebar.number_input("Cost per KM (€)", min_value=0.0, value=2.5, format="%.2f")
fixed_cost_per_truck = st.sidebar.number_input("Fixed Cost per Truck (€)", min_value=0.0, value=50.0, format="%.2f")


# List of predefined data files
data_files = {
    "Mini Dataset": "../Data/mini.csv",
    "Medium Dataset": "../Data/medium.csv",
    "Large Dataset": "../Data/many.csv",
    "Largest Dataset": "../Data/manyLarge.csv"
}

# Add to the sidebar
st.sidebar.title("Dataset Selection")

# Dropdown to select a predefined dataset (default: Mini Dataset)
selected_file = st.sidebar.selectbox(
    "Select a predefined dataset",
    options=list(data_files.keys()),
    index=0  # Default to the first dataset (Mini Dataset)
)

# File uploader for custom input
uploaded_file = st.sidebar.file_uploader("Or upload your own CSV file", type=["csv"])

# Reset button in the sidebar
if st.sidebar.button("Reset"):
    st.experimental_rerun()

# Load the selected or uploaded file
data = None  # Initialize the data variable

required_columns = ["name", "latitude", "longitude"]
if uploaded_file is not None:
    # Load the uploaded file
    try:
        data = pd.read_csv(uploaded_file)
        # Validate the columns
        if not all(col in data.columns for col in required_columns):
            st.sidebar.error(f"Uploaded file must contain the following columns: {', '.join(required_columns)}")
            data = None  # Clear data if validation fails
        else:
            st.sidebar.success("Custom file loaded and validated successfully!")
    except Exception as e:
        st.error(f"Error loading file: {e}")
elif selected_file:
    # Load the selected predefined file
    try:
        data = pd.read_csv(data_files[selected_file])
        st.sidebar.success(f"Predefined file '{selected_file}' loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading predefined file: {e}")
else:
    st.sidebar.error("Please select or upload a file to proceed.")

# # Display the data if it has been successfully loaded
# if data is not None:
#     st.write("### Preview of the Loaded Dataset")
#     st.dataframe(data.head())

data = load_data('../Data/mini.csv')
unique_companies = data['name'].unique()

# Fetch ranking data
dmatrix = distance_matrix()
ranking_data = get_min_max_ranking(dmatrix, data)

# Display the top  10 ranked list -->
st.subheader("Full Ranked List of Collaborations")
st.dataframe(ranking_data.head(10), hide_index=True)

# Dropdowns for company selection
# Add placeholders to the company list
placeholder_companies = ["Select a company", *unique_companies]

st.subheader("Select Companies for Detailed Analysis")
company_a = st.selectbox("Select Company A", placeholder_companies,index=0)
company_b = st.selectbox("Select Company B", placeholder_companies,index=0)

# Button to trigger analysis
if st.button("Analyze Collaboration"):
    if company_a == "Select a company" or company_b == "Select a company":
        st.error("Please select valid companies for both dropdowns.")
    elif company_a == company_b:
        st.error("Please select two different companies.")
    else:
        # Call the backend function
        results = mock_cvrp(vehicle_capacity, cost_per_km, fixed_cost_per_truck)
        cost_a = results["Cost (€)"][0]
        cost_b = results["Cost (€)"][1]
        cost_collab = results["Cost (€)"][2]

        # Display the results
        st.subheader("Analysis Results")
        st.write(f"Cost for {company_a}: {cost_a}")
        st.write(f"Cost for {company_b}: {cost_b}")
        st.write(f"Cost for collaboration: {cost_collab}")
        st.write(f"Total savings: {cost_a + cost_b - cost_collab}")



