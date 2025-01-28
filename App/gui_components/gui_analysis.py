import streamlit as st
from routing import all_cvrp
from routing import plot_routes_map
from dss import depot_lat
from dss import depot_lon


def render_analysis(vehicle_capacity, cost_per_km, fixed_cost_per_truck, data, dmatrix):
    """Handles company selection and performs collaboration analysis."""
    # Dropdowns for company selection
    unique_companies = sorted(data["name"].unique())
    placeholder_companies = ["Select a company", *unique_companies]

    st.subheader("Detailed Analysis")
    company_a = st.selectbox("Select first company", placeholder_companies, index=0, key="company_a")
    company_b = st.selectbox("Select second company", placeholder_companies, index=0, key="company_b")

    # Initialize session state for storing results
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = {}

    # Analyze collaboration
    if st.button("Analyze Collaboration"):
        if company_a == "Select a company" or company_b == "Select a company":
            st.error("Please select valid companies for both dropdowns.")
        elif company_a == company_b:
            st.error("Please select two different companies.")
        else:
            # Generate a unique key for the selected companies
            analysis_key = f"{company_a}_{company_b}"
            if analysis_key not in st.session_state.analysis_results:
                results = all_cvrp(vehicle_capacity, cost_per_km, fixed_cost_per_truck, company_a, company_b, data, dmatrix)
                st.session_state.analysis_results[analysis_key] = results
            else:
                results = st.session_state.analysis_results[analysis_key]

            # Generate the map and CSV file
            map, csv_file_path = plot_routes_map(data, depot_lat, depot_lon, company_a, company_b, results["Routes"][2], output_file='routes_map.html')

            # Store map and CSV path in session state
            st.session_state["map_html"] = map._repr_html_()
            st.session_state["csv_file_path"] = csv_file_path

    # Retrieve and display stored results if available
    analysis_key = f"{company_a}_{company_b}"
    if analysis_key in st.session_state.analysis_results:
        results = st.session_state.analysis_results[analysis_key]
        st.subheader("Analysis Results")
        total_cost_a = results["Total Cost"][0]
        total_cost_b = results["Total Cost"][1]
        total_cost_collab = results["Total Cost"][2]

        st.write(f'**{company_a}**: Total costs €{total_cost_a:.2f}, Fixed truck costs €{results["Truck Cost"][0]:.2f}, Kilometer costs €{results["Driving Cost"][0]:.2f}')
        st.write(f'**{company_b}**: Total costs €{total_cost_b:.2f}, Fixed truck costs €{results["Truck Cost"][1]:.2f}, Kilometer costs €{results["Driving Cost"][1]:.2f}')
        st.write(f'**Collaboration**: Total costs €{total_cost_collab:.2f}, Fixed truck costs €{results["Truck Cost"][2]:.2f}, Kilometer costs €{results["Driving Cost"][2]:.2f}')
        st.write(f"Total savings: €{total_cost_a + total_cost_b - total_cost_collab:.2f}")

    # Display map and download button if available in session state
    if "map_html" in st.session_state:
        st.title("Partnership Map")
        st.write("Interactive map showing company customers and depot.")
        st.components.v1.html(st.session_state["map_html"], height=450)

    if "csv_file_path" in st.session_state:
        with open(st.session_state["csv_file_path"], 'r') as csv_file:
            csv_data = csv_file.read()
        st.download_button(
            label="Download Routes",
            data=csv_data,
            file_name="routes.csv",
            mime="text/csv"
        )
