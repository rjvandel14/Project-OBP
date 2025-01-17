import streamlit as st
import pandas as pd
from ranking_functions.ranking_minmax import get_min_max_ranking
from routing import all_cvrp

def render_ranking(dmatrix, data, vehicle_capacity, cost_per_km, fixed_cost_per_truck):
    """Generates and displays the ranking data."""

    ranking_data = get_min_max_ranking(dmatrix, data)

    # Display the ranked collaborations
    st.subheader("Ranked Collaborations")

    # Initialize session state variables
    if "rows_to_display" not in st.session_state:
        st.session_state.rows_to_display = 10  # Start with the top 10 rows
    if "first_show_more" not in st.session_state:
        st.session_state.first_show_more = True  # Tracks whether it's the first click

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

    # Decide how many rows to display
    rows_to_display = ranking_data.head(st.session_state.rows_to_display)

    # Iterate through the rows and display them with toggle buttons
    for index, row in rows_to_display.iterrows():
        col1, col2, col3, col4 = st.columns([1, 3, 3, 2])  # Adjust column widths

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
            with st.expander(f"Analysis for {row['Company A']} ↔ {row['Company B']}", expanded=True):
                st.write(f"**Analyzing collaboration between {row['Company A']} and {row['Company B']}**")

                # Call the routing function to get analysis results
                results = all_cvrp(
                    vehicle_capacity,
                    cost_per_km,
                    fixed_cost_per_truck,
                    row["Company A"],
                    row["Company B"],
                    data,
                    dmatrix,
                )

                # Display the results
                cost_a = results["Cost (€)"][0]
                cost_b = results["Cost (€)"][1]
                cost_collab = results["Cost (€)"][2]
                st.write(f"**Results:**")
                st.write(f"Cost for {row['Company A']}: {cost_a}")
                st.write(f"Cost for {row['Company B']}: {cost_b}")
                st.write(f"Cost for collaboration: {cost_collab}")
                st.write(f"Total savings: {cost_a + cost_b - cost_collab}")

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
