import streamlit as st
import pandas as pd
from ranking_functions.ranking_minmax import get_min_max_ranking
from routing import all_cvrp

def render_ranking(dmatrix, data, vehicle_capacity, cost_per_km, fixed_cost_per_truck):
    """Generates and displays the ranking data."""

    ranking_data = get_min_max_ranking(dmatrix, data)

    # Display the ranked collaborations
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
        st.session_state.rows_to_display = 10  # Start with the top 10 rows
    if "first_show_more" not in st.session_state:
        st.session_state.first_show_more = True  # Tracks whether it's the first click
    if "toggle_states" not in st.session_state:
        st.session_state.toggle_states = {}
    if "checkbox_states" not in st.session_state:
        st.session_state.checkbox_states = {}
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
        st.session_state.rows_to_display = 10
        st.session_state.first_show_more = True  # Reset first click flag
        st.session_state.toggle_states = {index: False for index in ranking_data.index}
        st.session_state.checkbox_states = {index: False for index in ranking_data.index}
        st.session_state.results = {}

    # Decide how many rows to display
    rows_to_display = ranking_data.head(st.session_state.rows_to_display)

    #Show headers
    col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 1.5])  # Adjust column widths

    with col1:
        st.markdown("**Rank**")

    with col2:
        st.markdown("**Company A**")

    with col3:
        st.markdown("**Company B**")

    with col4:
        st.markdown("**Analysis**")

    with col5:
        st.markdown("**Shorter Calculations**")

    st.markdown("<hr style='border: 1px solid #ccc; margin-top: -10px; margin-bottom: 10px;'>", unsafe_allow_html=True)

    #Loop trough rows to show data
    for index, row in rows_to_display.iterrows():
        col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 1.5])  # Adjust column widths

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

        with col5:
            # Manage checkbox state in session state
            if f"checkbox_{index}" not in st.session_state.checkbox_states:
                st.session_state.checkbox_states[f"checkbox_{index}"] = False

            # Use Streamlit's checkbox widget
            st.session_state.checkbox_states[f"checkbox_{index}"] = st.checkbox(
                f"",
                key=f"checkbox_{index}",
                value=st.session_state.checkbox_states[f"checkbox_{index}"]
            )

        st.markdown("<hr style='border: 1px solid #ccc; margin-top: -10px; margin-bottom: 10px;'>", unsafe_allow_html=True)

        # Show or hide analysis based on the toggle state
        timelimit = st.session_state.checkbox_states.get(f"checkbox_{index}", False)
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
                        timelimit,
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


    # Callback to handle "Show More" button
    def show_more_callback():
        if st.session_state.first_show_more:
            st.session_state.rows_to_display += 40  # First click adds 40 rows
            st.session_state.first_show_more = False  # After first click, switch to normal behavior
        else:
            st.session_state.rows_to_display += 50  # Subsequent clicks add 50 rows

    # Place the "Show More" button below the table
    if len(ranking_data) > st.session_state.rows_to_display:
        col1, col2, col3 = st.columns([1, 8, 1])  # Center-align the button
        with col2:
            st.button("Show More", key="show_more_button", on_click=show_more_callback)

    return ranking_data
