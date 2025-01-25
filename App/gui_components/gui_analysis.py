import streamlit as st
import pandas as pd
from routing import all_cvrp
from routing import plot_routes_map
from dss import depot_lat
from dss import depot_lon
import json


def render_analysis(vehicle_capacity, cost_per_km, fixed_cost_per_truck, data, dmatrix):
    """Handles company selection and performs collaboration analysis."""
    # Dropdowns for company selection
    unique_companies = sorted(data["name"].unique())
    placeholder_companies = ["Select a company", *unique_companies]

    st.subheader("Select Companies for Detailed Analysis")
    company_a = st.selectbox("Select Company A", placeholder_companies, index=0, key="company_a")
    company_b = st.selectbox("Select Company B", placeholder_companies, index=0, key="company_b")

    # Analyze collaboration
    if st.button("Analyze Collaboration"):
        if company_a == "Select a company" or company_b == "Select a company":
            st.error("Please select valid companies for both dropdowns.")
        elif company_a == company_b:
            st.error("Please select two different companies.")
        else:
            results = all_cvrp(vehicle_capacity, cost_per_km, fixed_cost_per_truck, company_a, company_b, data, dmatrix)

            # Display the results
            st.subheader("Analysis Results")
            total_cost_a = results["Total Cost"][0]
            total_cost_b = results["Total Cost"][1]
            total_cost_collab = results["Total Cost"][2]

            st.write(f'{company_a}: Total costs {total_cost_a}, Fixed truck costs {results["Truck Cost"][0]}, Kilometer costs {results["Driving Cost"][0]}')
            st.write(f'{company_b}: Total costs {total_cost_b}, Fixed truck costs {results["Truck Cost"][1]}, Kilometer costs {results["Driving Cost"][1]}')
            st.write(f'Collaboration: Total costs {total_cost_collab}, Fixed truck costs {results["Truck Cost"][2]}, Kilometer costs {results["Driving Cost"][2]}')
            st.write(f"Total savings: {total_cost_a + total_cost_b - total_cost_collab}")

            # Generate the map and retrieve JSON data
            map, routes_json_data = plot_routes_map(data, depot_lat, depot_lon, company_a, company_b, results["Routes"][2], output_file='routes_map.html')

            # Store map and JSON data in session state
            st.session_state["map_html"] = map._repr_html_()
            st.session_state["routes_json"] = json.dumps(routes_json_data, indent=4)

    # Display map and download button if available in session state
    if "map_html" in st.session_state:
        st.title("Partnership Map")
        st.write("Interactive map showing company customers and depot.")
        st.components.v1.html(st.session_state["map_html"], height=450)

    if "routes_json" in st.session_state:
        st.subheader("Download Routes as JSON")
        st.download_button(
            label="Download Routes",
            data=st.session_state["routes_json"],
            file_name="routes.json",
            mime="application/json"
        )
