import streamlit as st
import pandas as pd
from routing import all_cvrp
from routing import plot_routes_map
from dss import depot_lat
from dss import depot_lon

def render_analysis(vehicle_capacity, cost_per_km, fixed_cost_per_truck, data, dmatrix):
    """Handles company selection and performs collaboration analysis."""
    # Dropdowns for company selection
    unique_companies = sorted(data["name"].unique())
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
            total_cost_a = results["Total Cost (€)"][0]
            total_cost_b = results["Total Cost (€)"][1]
            total_cost_collab = results["Total Cost (€)"][2]
            truck_costs_a = results["Truck Cost (€)"][0]
            truck_costs_b = results["Truck Cost (€)"][1]
            truck_costs_collab = results["Truck Cost (€)"][2]
            km_costs_a = results["Driving Cost (€)"][0]
            km_costs_b = results["Driving Cost (€)"][1]
            km_costs_collab = results["Driving Cost (€)"][2]

            # Display the results
            st.subheader("Analysis Results")
            st.write(f"{company_a}: Total costs {total_cost_a}, Fixed truck costs {truck_costs_a}, Kilometer costs {km_costs_a}")
            st.write(f"{company_b}: Total costs {total_cost_b}, Fixed truck costs {truck_costs_b}, Kilometer costs {km_costs_b}")
            st.write(f"Collab: Total costs {total_cost_collab}, Fixed truck costs {truck_costs_collab}, Kilometer costs {km_costs_collab}")
            st.write(f"Total savings: {total_cost_a + total_cost_b - total_cost_collab}")
            
            plot_routes_map(data, depot_lat, depot_lon, company_a, company_b, results["Routes"][2], output_file='routes_map.html')