import streamlit as st
import pandas as pd
from ranking_functions.ranking_minmax import get_min_max_ranking

def render_ranking(dmatrix, data):
    """Generates and displays the ranking data."""

    ranking_data = get_min_max_ranking(dmatrix, data)

    # Display the top 10 ranked collaborations
    st.subheader("Top 10 Ranked Collaborations")
    st.dataframe(ranking_data.head(10), hide_index=True)

    return ranking_data
