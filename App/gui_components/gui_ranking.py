import streamlit as st
import pandas as pd
from ranking_functions.ranking_minmax import get_min_max_ranking
from ranking_functions.ranking_clustering import get_cluster_kmeans
from ranking_functions.ranking_dbscan import get_dbscan_ranking, recommend_minPts, find_optimal_epsilon
from routing import all_cvrp

def render_ranking(dmatrix, data, vehicle_capacity, cost_per_km, fixed_cost_per_truck):
    """Generates and displays the ranking data."""

    #dmatrix_without_depot = dmatrix.drop(index='Depot', columns='Depot')

    #ranking_data = get_min_max_ranking(dmatrix, data)
    ranking_data = get_cluster_kmeans(data, dmatrix)
    # min_samples = recommend_minPts(len(data) -1)
    # eps = find_optimal_epsilon(dmatrix_without_depot, min_samples)
    # print("eps", eps)
    # ranking_data = get_dbscan_ranking(data,dmatrix_without_depot,eps,min_samples)

    # Create two columns: one for the title and the other for the filter options
    col1, col2 = st.columns([2, 1]) 
    
    with col1:
        st.subheader("Ranked Collaborations")

    # Display the filter options in the second column
    with col2:
        companies = sorted(sorted(data["name"].unique()))
        selected_company = st.selectbox("Select a company to filter", ["All"] + companies)
        if selected_company != "All":
            ranking_data = ranking_data[
                (ranking_data['Company A'] == selected_company) | 
                (ranking_data['Company B'] == selected_company)
            ]

    # Initialize session state variables
    if "rows_to_display" not in st.session_state:
        st.session_state.rows_to_display = 5  # Start with the top 5 rows
    if "click_count" not in st.session_state:
        st.session_state.click_count = 0  # Initialiseer de klik-teller
    if "toggle_states" not in st.session_state:
        st.session_state.toggle_states = {}
    if "results" not in st.session_state:
        st.session_state.results = {}

    # Generate a hash for the current dataset
    current_data_hash = hash(pd.util.hash_pandas_object(ranking_data).sum())

    # Reset states if the dataset changes
    if (
        "current_data_hash" not in st.session_state
        or st.session_state.current_data_hash != current_data_hash
    ):
        st.session_state.current_data_hash = current_data_hash
        st.session_state.rows_to_display = 5
        st.session_state.click_count = 0
        st.session_state.toggle_states = {index: False for index in ranking_data.index}
        st.session_state.results = {}

    # Decide how many rows to display
    rows_to_display = ranking_data.head(st.session_state.rows_to_display)

    #Show headers
    col1, col2, col3, col4 = st.columns([1, 2, 2, 1.5])  # Adjust column widths

    with col1:
        st.markdown("**Rank**")

    with col2:
        st.markdown("**Company A**")

    with col3:
        st.markdown("**Company B**")

    with col4:
        st.markdown(
            """
            <style>
            .tooltip {
                position: relative;
                display: inline-block;
                cursor: pointer;
                font-weight: 600; /* Match Streamlit header font weight */
                text-align: center;
            }
            .tooltip .tooltiptext {
                visibility: hidden;
                width: 250px; /* Define the box width */
                background-color: rgba(50, 50, 50, 0.9); /* Dark background for contrast */
                color: #fff; /* White text for readability */
                text-align: left; /* Align text to the left inside the box */
                border-radius: 5px; /* Rounded corners for a modern look */
                padding: 10px; /* Add padding for spacing */
                position: absolute;
                z-index: 1;
                top: 130%; /* Position below the header */
                left: 50%;
                transform: translateX(-50%);
                opacity: 0;
                transition: opacity 0.3s ease;
                font-size: 12px; /* Smaller font size for clarity */
                line-height: 1.6; /* Add spacing between lines */
                box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
                white-space: normal; /* Allow text wrapping for multi-line */
            }
            .tooltip:hover .tooltiptext {
                visibility: visible;
                opacity: 1;
            }
            </style>
            <div class="tooltip">
                Short Analysis
                <span class="tooltiptext">
                    This option allows you to set a time limit for calculations. 
                    By enabling it, the system will use faster methods to provide approximate results, 
                    which can save time during analysis but may reduce precision.
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<hr style='border: 1px solid #ccc; margin-top: 0px; margin-bottom: 10px;'>", unsafe_allow_html=True)

    #Loop trough rows to show data
    for index, row in rows_to_display.iterrows():
        col1, col2, col3, col4 = st.columns([1, 2, 2, 1.5])  # Adjust column widths

        with col1:
            st.write(f"{row['Rank']}")  # Display the rank

        with col2:
            st.write(f"{row['Company A']}")  # Display Company A

        with col3:
            st.write(f"{row['Company B']}")  # Display Company B

        with col4:
            # Toggle the state when the button is clicked
            if st.button(f"Analyze {index + 1}", key=f"analyze_{index}"):
                # Reset all toggle states to False
                for key in st.session_state.toggle_states.keys():
                    st.session_state.toggle_states[key] = False
                # Set the clicked row's state to True
                st.session_state.toggle_states[index] = True

        # Show or hide analysis based on the toggle state
        if st.session_state.toggle_states.get(index, False):
            # Expand the analysis section
            with st.expander(f"Analysis for {row['Company A']} ↔ {row['Company B']}", expanded=True):
                st.write(f"**Analyzing collaboration between {row['Company A']} and {row['Company B']}**")

                # Perform recalculation only if no results exist for this index
                if index not in st.session_state.results:
                    results = all_cvrp(
                        vehicle_capacity,
                        cost_per_km,
                        fixed_cost_per_truck,
                        row["Company A"],
                        row["Company B"],
                        data,
                        dmatrix,
                        True,
                    )
                    st.session_state.results[index] = results  # Save results in session state

        # Retrieve results from session state
        results = st.session_state.results.get(index)
        if results:
            # Always display results if they exist
            with st.expander(f"Analysis Results for {row['Company A']} ↔ {row['Company B']}, result may be suboptimal because of time limit ", expanded=True):
                total_cost_a = results["Total Cost"][0]
                total_cost_b = results["Total Cost"][1]
                total_cost_collab = results["Total Cost"][2]

                st.write(f'{row["Company A"]}: Total costs {total_cost_a}, Fixed truck costs {results["Truck Cost"][0]}, Kilometer costs {results["Driving Cost"][0]}')
                st.write(f'{row["Company B"]}: Total costs {total_cost_b}, Fixed truck costs {results["Truck Cost"][1]}, Kilometer costs {results["Driving Cost"][1]}')
                st.write(f'Collaboration: Total costs {total_cost_collab}, Fixed truck costs {results["Truck Cost"][2]}, Kilometer costs {results["Driving Cost"][2]}')
                st.write(f"Total savings: {total_cost_a + total_cost_b - total_cost_collab}")


        st.markdown("<hr style='border: 1px solid #ccc; margin-top: -10px; margin-bottom: 10px;'>", unsafe_allow_html=True)

    # Callback to handle "Show More" button
    def show_more_callback():
        st.session_state.click_count += 1  # add to click count

        if st.session_state.click_count == 1:
            st.session_state.rows_to_display += 5  # add 5 rows after first click
        elif st.session_state.click_count == 2:
            st.session_state.rows_to_display += 40  # add 40 rows adter second click
        else:
            st.session_state.rows_to_display += 50  # add 50 rows after

    # Place the "Show More" and download csv button below the table
    if len(ranking_data) > st.session_state.rows_to_display:
        col1, col2, col3, col4 = st.columns([0.2, 5, 2.5, 1])  # Center-align the button
        with col2:
            st.button("Show More", key="show_more_button", on_click=show_more_callback)
        with col3:
            csv_data = ranking_data.drop(columns=["Score"]).to_csv(index=False)
            st.download_button(
                label="Download Complete Ranking",
                data=csv_data,
                file_name='ranking_data.csv',
                mime='text/csv',
        )

    return ranking_data
