from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
import sys
import os

# Add the App directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from distancematrix import dmatrix

def find_optimal_clusters(dmatrix, max_clusters=10):
    """
    Use the Elbow Method to find the optimal number of clusters.
    """
    wcss = []
    distance_array = dmatrix.to_numpy()

    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(distance_array)
        wcss.append(kmeans.inertia_)  # Inertia is the WCSS value

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(1, max_clusters + 1), wcss, marker='o')
    ax.set_title('Elbow Method to Find Optimal Number of Clusters')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Within-Cluster Sum of Squares (WCSS)')
    ax.grid(True)

    return fig

# Streamlit app
st.title("Elbow Method for Optimal Clusters")
st.write("This application calculates and visualizes the optimal number of clusters using the Elbow Method.")

# Slider to select maximum number of clusters
max_clusters = st.slider("Select the maximum number of clusters:", 2, 15, 10)

# Generate the plot
fig = find_optimal_clusters(dmatrix, max_clusters)

# Display the plot in Streamlit
st.pyplot(fig)
