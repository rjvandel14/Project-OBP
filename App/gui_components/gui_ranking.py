import streamlit as st
import pandas as pd
from ranking_functions.ranking_minmax import get_min_max_ranking
from routing import all_cvrp

def render_ranking(dmatrix, data, vehicle_capacity, cost_per_km, fixed_cost_per_truck):
    """Generates and displays the ranking data."""

    ranking_data = get_min_max_ranking(dmatrix, data)

    # Display the ranked collaborations
    st.subheader("Ranked Collaborations")

    # Initialize the "Show More" toggle state in session
    if "show_full_ranking" not in st.session_state:
        st.session_state.show_full_ranking = False

    # Generate a hash for the current dataset
    current_data_hash = hash(pd.util.hash_pandas_object(ranking_data).sum())

    # Reset states only if the dataset changes
    if (
        "current_data_hash" not in st.session_state
        or st.session_state.current_data_hash != current_data_hash
    ):
        st.session_state.current_data_hash = current_data_hash
        st.session_state.show_full_ranking = False
        st.session_state.toggle_states = {index: False for index in ranking_data.index}

    # Decide how many rows to display based on toggle state
    rows_to_display = ranking_data if st.session_state.show_full_ranking else ranking_data.head(10)

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

    # Place the "Show More" button below the table if there are more than 10 rows
    if len(ranking_data) > 10:
        show_more_button = st.button(
            "Show More" if not st.session_state.show_full_ranking else "Show Top 10",
            key="show_more_button",
        )
        if show_more_button:
            st.session_state.show_full_ranking = not st.session_state.show_full_ranking

    return ranking_data
