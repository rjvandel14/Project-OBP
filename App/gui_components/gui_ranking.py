import streamlit as st
from ranking import get_min_max_ranking
from distancematrix import distance_matrix

def render_ranking(data):
    """Generates and displays the ranking data."""
    # Generate the distance matrix
    dmatrix = distance_matrix(data)
    ranking_data = get_min_max_ranking(dmatrix, data)

    # Display the top 10 ranked collaborations
    st.subheader("Top 10 Ranked Collaborations")
    st.dataframe(ranking_data.head(10), hide_index=True)

    return ranking_data
