import pandas as pd
import streamlit as st
from dss import depot_lat, depot_lon
from routing import all_cvrp, plot_routes_map

def render_analysis(vehicle_capacity, cost_per_km, fixed_cost_per_truck, data, dmatrix):
    """
    Render the detailed analysis section for company collaboration using Streamlit.

    Parameters:
    - vehicle_capacity (int): Capacity of the vehicle in units.
    - cost_per_km (float): Cost per kilometer for travel.
    - fixed_cost_per_truck (float): Fixed cost per truck.
    - data (pd.DataFrame): DataFrame containing customer and company data.
    - dmatrix (pd.DataFrame): Distance matrix between customers and depot.
    """

    # Dropdown menus for selecting companies
    unique_companies = sorted(data["name"].unique())
    placeholder_companies = ["Select or type", *unique_companies]

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Detailed Analysis")  # Section title

    with col2:
        # Input for time limit in minutes, converted to seconds
        timelimit = st.number_input("**Stop Calculation after (min)**", min_value=1, value=10) * 60

    # Company selection using dropdowns
    company_a = st.selectbox("**Select first company**", placeholder_companies, index=0, key="company_a")
    company_b = st.selectbox("**Select second company**", placeholder_companies, index=0, key="company_b")

    # Initialize session state to store analysis results
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = {}

    # Analyze collaboration when the button is pressed
    if st.button(":violet[Analyze Collaboration]"):
        # Validate selected companies
        if company_a == "Select or type" or company_b == "Select or type":
            st.error("Please select valid companies for both dropdowns.")
        elif company_a == company_b:
            st.error("Please select two different companies.")
        else:
            # Unique key to store results for selected companies
            analysis_key = f"{company_a}_{company_b}"
            if analysis_key not in st.session_state.analysis_results:
                # Run CVRP analysis and store the results
                results = all_cvrp(vehicle_capacity, cost_per_km, fixed_cost_per_truck, company_a, company_b, data, dmatrix, timelimit)
                st.session_state.analysis_results[analysis_key] = results
            else:
                # Retrieve cached results
                results = st.session_state.analysis_results[analysis_key]

            # Generate an interactive map and CSV file with route information
            map, csv_file_path = plot_routes_map(data, depot_lat, depot_lon, company_a, company_b, results["Routes"][2], output_file='routes_map.html')

            # Store map HTML and CSV path in session state
            st.session_state["map_html"] = map._repr_html_()
            st.session_state["csv_file_path"] = csv_file_path

    # Display stored results if available
    analysis_key = f"{company_a}_{company_b}"
    if analysis_key in st.session_state.analysis_results:
        results = st.session_state.analysis_results[analysis_key]
        st.subheader("Analysis Results")

        # Prepare data for displaying cost breakdown
        data = {
            "Category": [company_a, company_b, "Collaboration"],
            "Total Costs (€)": [results["Total Cost"][0], results["Total Cost"][1], results["Total Cost"][2]],
            "Fixed Truck Costs (€)": [results["Truck Cost"][0], results["Truck Cost"][1], results["Truck Cost"][2]],
            "Kilometer Costs (€)": [results["Driving Cost"][0], results["Driving Cost"][1], results["Driving Cost"][2]]
        }

        # Create and style the DataFrame
        df = pd.DataFrame(data)
        df.set_index("Category", inplace=True)
        df = df.style.format({
            "Total Costs (€)": "{:.2f}",
            "Fixed Truck Costs (€)": "{:.2f}",
            "Kilometer Costs (€)": "{:.2f}"
        })

        # Display the cost breakdown table
        st.table(df)

        # Calculate and display total cost savings
        total_savings = results["Total Cost"][0] + results["Total Cost"][1] - results["Total Cost"][2]
        st.markdown(f"**:violet[Total savings: €{total_savings:.2f}]**")

    # Display the interactive map if available in session state
    if "map_html" in st.session_state:
        st.subheader("Partnership Map")
        st.write("Interactive map showing company customers and depot.")
        st.components.v1.html(st.session_state["map_html"], height=450)

    # Display a download button for the routes CSV file
    if "csv_file_path" in st.session_state:
        with open(st.session_state["csv_file_path"], 'r') as csv_file:
            csv_data = csv_file.read()
        st.download_button(
            label=":violet[Download Routes]",
            data=csv_data,
            file_name="routes.csv",
            mime="text/csv"
        )
