from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
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
    Automatically determine the optimal number of clusters using:
    1. Silhouette Score
    2. Elbow Method (KneeLocator)
    """
    # Convert the distance matrix to a NumPy array if not already
    if hasattr(dmatrix, "to_numpy"):
        distance_array = dmatrix.to_numpy()
    else:
        distance_array = dmatrix

    # Store WCSS and Silhouette Scores
    wcss = []
    silhouette_scores = []

    for k in range(2, max_clusters + 1):  # Start from k=2
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(distance_array)

        # Calculate WCSS (inertia)
        wcss.append(kmeans.inertia_)

        # Silhouette Score using precomputed distances
        silhouette_scores.append(
            silhouette_score(distance_array, cluster_labels, metric="precomputed")
        )

    # Find the optimal k using the Elbow Method
    kn = KneeLocator(range(2, max_clusters + 1), wcss, curve="convex", direction="decreasing")
    elbow_k = kn.knee

    # Find the optimal k using Silhouette Score
    silhouette_k = np.argmax(silhouette_scores) + 2  # +2 because k starts from 2

    # Return the best k (resolve conflicts if elbow_k and silhouette_k differ)
    optimal_k = elbow_k if elbow_k == silhouette_k else silhouette_k

    return optimal_k, wcss, silhouette_scores

# Streamlit app
st.title("Optimal Number of Clusters")
st.write("Automatically determine the optimal number of clusters using Elbow Method and Silhouette Score.")

# Set the maximum number of clusters
max_clusters = st.slider("Select the maximum number of clusters to evaluate:", 2, 15, 10)

# Find the optimal number of clusters
optimal_k, wcss, silhouette_scores = find_optimal_clusters(dmatrix, max_clusters)

# Display the results
st.write(f"The optimal number of clusters is: {optimal_k}")
st.write("WCSS values for each k:")
st.write(wcss)
st.write("Silhouette Scores for each k:")
st.write(silhouette_scores)

# Optional: Visualize WCSS and Silhouette Scores
st.write("WCSS Plot:")
st.line_chart(wcss)

st.write("Silhouette Scores Plot:")
st.line_chart(silhouette_scores)