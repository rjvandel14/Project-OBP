import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import folium
from folium.plugins import MarkerCluster
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from distancematrix import distance_matrix
from dss import load_data

def find_optimal_clusters(data, max_clusters=10):
    """
    Automatically determine the optimal number of clusters using:
    1. Silhouette Score
    2. Elbow Method (KneeLocator)
    """
    wcss = []
    silhouette_scores = []

    for k in range(2, max_clusters + 1):  # Start from k=2
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(data)

        # Calculate WCSS
        wcss.append(kmeans.inertia_)

        # Calculate Silhouette Score
        silhouette_scores.append(silhouette_score(data, cluster_labels))

    # Find the optimal k using the Elbow Method
    kn = KneeLocator(range(2, max_clusters + 1), wcss, curve="convex", direction="decreasing")
    elbow_k = kn.knee

    # Fallback if no clear elbow is found
    if elbow_k is None:
        elbow_k = np.argmax(silhouette_scores) + 2  # +2 because k starts from 2

    return elbow_k, wcss, silhouette_scores

def plot_elbow_method(wcss, optimal_k):
    """
    Plot the elbow method for WCSS with the optimal number of clusters highlighted.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    x = range(2, len(wcss) + 2)

    ax.plot(x, wcss, marker="o", label="WCSS")
    if optimal_k:
        ax.axvline(optimal_k, color="red", linestyle="--", label=f"Optimal k = {optimal_k}")
        ax.scatter(optimal_k, wcss[optimal_k - 2], color="red", zorder=5)
    ax.set_title("Elbow Method for Optimal Clusters")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("WCSS")
    ax.grid()
    ax.legend()
    return fig

def plot_silhouette_scores(silhouette_scores, optimal_k):
    """
    Plot the silhouette scores for different numbers of clusters with the optimal k highlighted.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    x = range(2, len(silhouette_scores) + 2)

    ax.plot(x, silhouette_scores, marker="o", label="Silhouette Score")
    if optimal_k:
        ax.axvline(optimal_k, color="red", linestyle="--", label=f"Optimal k = {optimal_k}")
        ax.scatter(optimal_k, silhouette_scores[optimal_k - 2], color="red", zorder=5)
    ax.set_title("Silhouette Scores for Different Numbers of Clusters")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.grid()
    ax.legend()
    return fig

def create_cluster_map(df, cluster_assignments, depot_lat, depot_lon):
    """
    Create an interactive map showing customers and clusters.
    """
    # Merge cluster assignments with customer data
    df = df.copy()  # Avoid modifying the original dataframe
    df["Cluster"] = cluster_assignments

    # Create a Folium map centered at the depot
    m = folium.Map(location=[depot_lat, depot_lon], zoom_start=8)

    # Assign a unique color for each company
    company_names = df["name"].unique()
    company_colors = ['blue', 'green', 'purple', 'orange', 'darkred', 'darkblue', 'cadetblue', 'lightgreen']
    color_map = {name: company_colors[i % len(company_colors)] for i, name in enumerate(company_names)}

    # Add customer markers colored by company
    for _, row in df.iterrows():
        folium.Marker(
            location=[row["lat"], row["lon"]],  # Use correct column names
            popup=f"Customer of {row['name']} (Cluster {row['Cluster']})",
            icon=folium.Icon(color=color_map[row["name"]])
        ).add_to(m)

    # Draw circles around each cluster
    for cluster_id in df["Cluster"].unique():
        cluster_data = df[df["Cluster"] == cluster_id]
        cluster_center = cluster_data[["lat", "lon"]].mean().values
        folium.Circle(
            location=cluster_center,
            radius=15000,  # Adjust radius as needed
            color="black",
            fill=True,
            fill_opacity=0.2,
            popup=f"Cluster {cluster_id}"
        ).add_to(m)

    # Add depot marker
    folium.Marker(
        location=[depot_lat, depot_lon],
        popup="Depot",
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)

    # Streamlit output
    st.subheader("Cluster Map")
    st.write("Interactive map showing customer clusters.")
    st.components.v1.html(m._repr_html_(), height=600)
    

# Streamlit app for personal clustering exploration
st.title("Clustering Results and Visualizations")

# Load the data
# df = load_data('../Data/mini.csv')
df = load_data('C:/Users/daydo/Documents/GitHub/Project-OBP/Data/mini.csv')
# dmatrix = distance_matrix(df)

# Extract relevant data for clustering
customer_data = df[["lat", "lon"]]

# Dynamically calculate max_clusters
max_clusters = st.slider("Select the maximum number of clusters to evaluate:", 2, len(df["name"].unique()), 10)

# Find optimal clusters
optimal_k, wcss, silhouette_scores = find_optimal_clusters(customer_data, max_clusters)
st.write(f"Optimal number of clusters: {optimal_k}")

# Display WCSS and Silhouette plot
st.subheader("Elbow Method for Optimal Clusters")
elbow_fig = plot_elbow_method(wcss, optimal_k)
st.pyplot(elbow_fig)

st.subheader("Silhouette Scores Plot:")
silhouette_fig = plot_silhouette_scores(silhouette_scores, optimal_k)
st.pyplot(silhouette_fig)

# Perform clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans.fit_predict(customer_data)

# Create and display the cluster map
depot_lat = 52.16521
depot_lon = 5.17215
create_cluster_map(df, cluster_labels, depot_lat, depot_lon)