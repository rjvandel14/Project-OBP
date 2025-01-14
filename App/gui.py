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
from dss import calculate_collaboration, load_data
from dash import Dash, html
from ranking import get_mock_ranking

# Title
st.title("Logistics Collaboration Dashboard")

# Sidebar inputs
vehicle_capacity = st.sidebar.number_input("Vehicle Capacity", min_value=1, value=10)
cost_per_km = st.sidebar.number_input("Cost per KM (€)", min_value=0.0, value=2.5, format="%.2f")
fixed_cost_per_truck = st.sidebar.number_input("Fixed Cost per Truck (€)", min_value=0.0, value=50.0, format="%.2f")

data = load_data('../Data/mini.csv')
unique_companies = data['name'].unique()

# Fetch ranking data
ranking_data = get_mock_ranking()

# Display the full ranked list
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
        results = calculate_collaboration(vehicle_capacity, cost_per_km, fixed_cost_per_truck, company_a, company_b)

        # Display the results
        st.subheader("Analysis Results")
        st.write(f"Cost for {company_a}: {results['cost_a']}")
        st.write(f"Cost for {company_b}: {results['cost_b']}")
        st.write(f"Cost for collaboration: {results['collaboration_cost']}")
        st.write(f"Total savings: {results['savings']}")

