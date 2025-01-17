import streamlit as st
import pandas as pd
from routing import all_cvrp
from routing import plot_routes_map
from dss import depot_lat
from dss import depot_lon

def render_analysis(vehicle_capacity, cost_per_km, fixed_cost_per_truck, data, dmatrix):
    """Handles company selection and performs collaboration analysis."""
    # Dropdowns for company selection
    unique_companies = data["name"].unique()
    placeholder_companies = ["Select a company", *unique_companies]

    st.subheader("Select Companies for Detailed Analysis")
    company_a = st.selectbox("Select Company A", placeholder_companies, index=0)
    company_b = st.selectbox("Select Company B", placeholder_companies, index=0)

    # Analyze collaboration
    if st.button("Analyze Collaboration"):
        if company_a == "Select a company" or company_b == "Select a company":
            st.error("Please select valid companies for both dropdowns.")
        elif company_a == company_b:
            st.error("Please select two different companies.")
        else:
            # Mock collaboration analysis
            results = all_cvrp(vehicle_capacity, cost_per_km, fixed_cost_per_truck, company_a, company_b, data, dmatrix)
            cost_a = results["Cost (€)"][0]
            cost_b = results["Cost (€)"][1]
            cost_collab = results["Cost (€)"][2]

            # Display the results
            st.subheader("Analysis Results")
            st.write(f"Cost for {company_a}: {cost_a}")
            st.write(f"Cost for {company_b}: {cost_b}")
            st.write(f"Cost for collaboration: {cost_collab}")
            st.write(f"Total savings: {cost_a + cost_b - cost_collab}")
            
            plot_routes_map(data, depot_lat, depot_lon, company_a, company_b, results["Routes"][2], output_file='routes_map.html')