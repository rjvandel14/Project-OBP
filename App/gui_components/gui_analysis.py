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
    placeholder_companies = ["Select or type", *unique_companies]

    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Detailed Analysis")

    with col2:
        timelimit = st.number_input("**Stop Calculation after (min)**", min_value=1, value=10)*60


    company_a = st.selectbox("**Select first company**", placeholder_companies, index=0, key="company_a")
    company_b = st.selectbox("**Select second company**", placeholder_companies, index=0, key="company_b")

    # Initialize session state for storing results
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = {}

    # Analyze collaboration
    if st.button(":violet[Analyze Collaboration]"):
        if company_a == "Select or type" or company_b == "Select or type":
            st.error("Please select valid companies for both dropdowns.")
        elif company_a == company_b:
            st.error("Please select two different companies.")
        else:
            # Generate a unique key for the selected companies
            analysis_key = f"{company_a}_{company_b}"
            if analysis_key not in st.session_state.analysis_results:
                results = all_cvrp(vehicle_capacity, cost_per_km, fixed_cost_per_truck, company_a, company_b, data, dmatrix, timelimit)
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

        # Prepare data for the table
        data = {
            "Category": [company_a, company_b, "Collaboration"],
            "Total Costs (€)": [results["Total Cost"][0], results["Total Cost"][1], results["Total Cost"][2]],
            "Fixed Truck Costs (€)": [results["Truck Cost"][0], results["Truck Cost"][1], results["Truck Cost"][2]],
            "Kilometer Costs (€)": [results["Driving Cost"][0], results["Driving Cost"][1], results["Driving Cost"][2]]
        }

        # Convert to DataFrame
        df = pd.DataFrame(data)
         # Set the index to "Category" and reset it to remove default numeric index
        df.set_index("Category", inplace=True)  # Use category names as the index
        df = df.style.format({"Total Costs (€)": "{:.2f}", 
                          "Fixed Truck Costs (€)": "{:.2f}", 
                          "Kilometer Costs (€)": "{:.2f}"})

        # Display the table in Streamlit
        st.table(df)
        
        # Calculate and display total savings separately
        total_savings = results["Total Cost"][0] + results["Total Cost"][1] - results["Total Cost"][2]
        st.markdown(f"**:violet[Total savings: €{total_savings:.2f}]**")

    # Display map and download button if available in session state
    if "map_html" in st.session_state:
        st.subheader("Partnership Map")
        st.write("Interactive map showing company customers and depot.")
        st.components.v1.html(st.session_state["map_html"], height=450)

    if "csv_file_path" in st.session_state:
        with open(st.session_state["csv_file_path"], 'r') as csv_file:
            csv_data = csv_file.read()
        st.download_button(
            label=":violet[Download Routes]",
            data=csv_data,
            file_name="routes.csv",
            mime="text/csv"
        )
