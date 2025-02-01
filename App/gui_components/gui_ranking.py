import pandas as pd
import streamlit as st
from routing import all_cvrp
from ranking_functions.ranking_dbscan import get_dbscan_ranking

def render_ranking(dmatrix, data, vehicle_capacity, cost_per_km, fixed_cost_per_truck):
    """
    Render the ranking of potential company collaborations based on DBSCAN clustering and provide analysis options.

    Parameters:
    - dmatrix (pd.DataFrame): Distance matrix between customers and depot.
    - data (pd.DataFrame): DataFrame containing company and customer data.
    - vehicle_capacity (int): Capacity of the vehicle.
    - cost_per_km (float): Cost per kilometer for the vehicles.
    - fixed_cost_per_truck (float): Fixed cost associated with each truck.
    """
    # Get initial ranking using DBSCAN clustering
    ranking_data = get_dbscan_ranking(data, dmatrix)

    # Layout for title and filtering options
    col1, col2 = st.columns([2, 1]) 

    with col1:
        st.subheader("Ranked Collaborations")  # Section title

    with col2:
        # Filter option to display collaborations related to a specific company
        companies = sorted(data["name"].unique())
        selected_company = st.selectbox("Select a company to filter", ["All"] + companies)
        if selected_company != "All":
            ranking_data = ranking_data[
                (ranking_data['Company A'] == selected_company) | 
                (ranking_data['Company B'] == selected_company)
            ]

    # Initialize session state variables for interaction tracking
    if "rows_to_display" not in st.session_state:
        st.session_state.rows_to_display = 5  # Start by displaying 5 rows
    if "click_count" not in st.session_state:
        st.session_state.click_count = 0
    if "toggle_states" not in st.session_state:
        st.session_state.toggle_states = {}
    if "results" not in st.session_state:
        st.session_state.results = {}

    # Generate a hash of the current dataset to track changes
    current_data_hash = hash(pd.util.hash_pandas_object(ranking_data).sum())

    # Reset state if the dataset has changed
    if (
        "current_data_hash" not in st.session_state
        or st.session_state.current_data_hash != current_data_hash
    ):
        st.session_state.current_data_hash = current_data_hash
        st.session_state.rows_to_display = 5
        st.session_state.click_count = 0
        st.session_state.toggle_states = {index: False for index in ranking_data.index}
        st.session_state.results = {}

    # Select the top rows to display
    rows_to_display = ranking_data.head(st.session_state.rows_to_display)

    # Headers for the ranking table
    col1, col2, col3, col4 = st.columns([1, 2, 2, 1.5]) 

    with col1:
        st.markdown("**Rank**")

    with col2:
        st.markdown("**First company**")

    with col3:
        st.markdown("**Second company**")

    with col4:
        # Tooltip with analysis instructions
        st.markdown(
            """
            <style>
            .tooltip {
                position: relative;
                display: inline-block;
                cursor: pointer;
                font-weight: 600;
                text-align: center;
            }
            .tooltip .tooltiptext {
                visibility: hidden;
                width: 250px;
                background-color: rgba(50, 50, 50, 0.9);
                color: #fff;
                text-align: left;
                border-radius: 5px;
                padding: 10px;
                position: absolute;
                z-index: 1;
                top: 130%;
                left: 50%;
                transform: translateX(-50%);
                opacity: 0;
                transition: opacity 0.3s ease;
                font-size: 12px;
                line-height: 1.6;
                box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
                white-space: normal;
            }
            .tooltip:hover .tooltiptext {
                visibility: visible;
                opacity: 1;
            }
            </style>
            <div class="tooltip">
                Short Analysis
                <span class="tooltiptext">
                    The analysis performed here is shorter, focusing on speed rather than precision. This allows faster results, but the calculations may be less accurate due to reduced computation time.
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Horizontal separator
    st.markdown("<hr style='border: 1px solid #ccc;'>", unsafe_allow_html=True)

    # Display ranked collaborations
    for index, row in rows_to_display.iterrows():
        col1, col2, col3, col4 = st.columns([1, 2, 2, 1.5])  # Adjust column widths

        with col1:
            st.write(f"{row['Rank']}")  # Rank of the collaboration

        with col2:
            st.write(f"{row['Company A']}")  # First company

        with col3:
            st.write(f"{row['Company B']}")  # Second company

        with col4:
            # Button to initiate analysis for a specific pair
            if st.button(f"Analyze {index + 1}", key=f"analyze_{index}"):
                if index not in st.session_state.results:
                    st.session_state.toggle_states[index] = True

        # Show analysis results when the user clicks the button
        if st.session_state.toggle_states.get(index, False):
            with st.expander(f"Analysis for {row['Company A']} ↔ {row['Company B']}", expanded=True):
                if index not in st.session_state.results:
                    with st.spinner(f"Analyzing collaboration between {row['Company A']} and {row['Company B']}..."):
                        timelimit = 10 + 0.5 * len(data[data['name'].isin([row['Company A'], row['Company B']])].copy())
                        results = all_cvrp(
                            vehicle_capacity,
                            cost_per_km,
                            fixed_cost_per_truck,
                            row["Company A"],
                            row["Company B"],
                            data,
                            dmatrix,
                            timelimit,
                        )
                        st.session_state.results[index] = results

                # Retrieve and display analysis results
                results = st.session_state.results.get(index)
                if results:
                    st.subheader(f"Analysis Results for {row['Company A']} ↔ {row['Company B']}")

                    # Prepare and display cost breakdown table
                    analysis_data = {
                        "Category": [row["Company A"], row["Company B"], "Collaboration"],
                        "Total Costs (€)": [results["Total Cost"][0], results["Total Cost"][1], results["Total Cost"][2]],
                        "Fixed Truck Costs (€)": [results["Truck Cost"][0], results["Truck Cost"][1], results["Truck Cost"][2]],
                        "Kilometer Costs (€)": [results["Driving Cost"][0], results["Driving Cost"][1], results["Driving Cost"][2]]
                    }

                    # Convert to DataFrame and format
                    df = pd.DataFrame(analysis_data)
                    df.set_index("Category", inplace=True)  # Remove default numeric index
                    df = df.style.format({
                        "Total Costs (€)": "{:.2f}", 
                        "Fixed Truck Costs (€)": "{:.2f}", 
                        "Kilometer Costs (€)": "{:.2f}"
                    })

                    # Display the table
                    st.table(df)
                    # Display total savings
                    total_savings = results["Total Cost"][0] + results["Total Cost"][1] - results["Total Cost"][2]
                    st.markdown(f"**:violet[Total savings: €{total_savings:.2f}]**")

        st.markdown("<hr style='border: 1px solid #ccc;'>", unsafe_allow_html=True)

    # Handle "Show More" button to display additional rows
    def show_more_callback():
        st.session_state.click_count += 1
        if st.session_state.click_count == 1:
            st.session_state.rows_to_display += 5
        elif st.session_state.click_count == 2:
            st.session_state.rows_to_display += 40
        else:
            st.session_state.rows_to_display += 50

    # Layout for "Show More" and "Download" buttons
    col1, col2 = st.columns([4, 2])

    with col1:
        if len(ranking_data) > st.session_state.rows_to_display:
            st.button(":violet[Show More]", on_click=show_more_callback)

    with col2:
        # Button to download the complete ranking as a CSV
        csv_data = ranking_data.drop(columns=["Score"]).to_csv(index=False)
        st.download_button(
            label=":violet[Download Complete Ranking]",
            data=csv_data,
            file_name='ranking_data.csv',
            mime='text/csv',
        )

    return ranking_data
