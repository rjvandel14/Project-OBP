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
    # Store WCSS and Silhouette Scores
    wcss = []
    silhouette_scores = []

    for k in range(2, max_clusters + 1):  # Start from k=2
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(data)

        # Calculate WCSS (inertia)
        wcss.append(kmeans.inertia_)

        # Silhouette Score
        silhouette_scores.append(silhouette_score(data, cluster_labels))

    # Find the optimal k using the Elbow Method
    kn = KneeLocator(range(2, max_clusters + 1), wcss, curve="convex", direction="decreasing")
    elbow_k = kn.knee

    return elbow_k, wcss, silhouette_scores

def get_clustering_ranking(dmatrix, df, max_clusters=10):
    """
    Computes a ranking table for collaborations using clustering.
    """
    # Filter out the depot
    dmatrix_filtered = dmatrix.iloc[1:, 1:]  # Remove the first row and column (depot)
    company_names = dmatrix.index[1:]  # Exclude the depot from the company list

    # Find optimal clusters
    optimal_k, _, _, = find_optimal_clusters(dmatrix, max_clusters)

    # Perform clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    cluster_labels = kmeans.fit_predict(dmatrix.iloc[1:, 1:])  # Use filtered distance matrix without depot

    # Assign clusters to companies
    cluster_assignments = pd.DataFrame({
        "Company": company_names,
        "Cluster": cluster_labels
    })

    # Find all unique partnerships within each cluster
    partnerships = []
    visited = set()  # Track unique partnerships to avoid duplicates
    for cluster in cluster_assignments["Cluster"].unique():
        cluster_members = cluster_assignments[cluster_assignments["Cluster"] == cluster]
        for i, company_a in enumerate(cluster_members["Company"]):
            for company_b in cluster_members["Company"].iloc[i + 1:]:
                # Ensure uniqueness of partnerships and avoid self-pairing
                if company_a != company_b:
                    pair = tuple(sorted([company_a, company_b]))  # Sort to avoid duplication
                    if pair not in visited:
                        partnerships.append({
                            "Company A": company_a,
                            "Company B": company_b,
                            "Cluster": cluster
                        })
                        visited.add(pair)

    partnerships_df = pd.DataFrame(partnerships)

    # Assign ranks based on clusters
    partnerships_df["Rank"] = partnerships_df["Cluster"].rank(method="dense").astype(int)

    return partnerships_df[["Rank", "Company A", "Company B", "Cluster"]], cluster_assignments

# plots elbow and silhouette
def plot_elbow_method(wcss, optimal_k):
    """
    Plot the elbow method for WCSS with the optimal number of clusters highlighted.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    x = range(2, len(wcss) + 2)
    
    # Plot the WCSS values
    ax.plot(x, wcss, marker="o", label="WCSS")
    
    # Highlight the optimal number of clusters
    ax.axvline(optimal_k, color="red", linestyle="--", label=f"Optimal k = {optimal_k}")
    ax.scatter(optimal_k, wcss[optimal_k - 2], color="red", zorder=5)  # Highlight the point
    
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
    
    # Plot the Silhouette scores
    ax.plot(x, silhouette_scores, marker="o", label="Silhouette Score")
    
    # Highlight the optimal number of clusters
    ax.axvline(optimal_k, color="red", linestyle="--", label=f"Optimal k = {optimal_k}")
    ax.scatter(optimal_k, silhouette_scores[optimal_k - 2], color="red", zorder=5)  # Highlight the point
    
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
    # Merge cluster assignments with original data to include lat and lon
    cluster_assignments = cluster_assignments.merge(df, left_on="Company", right_on="name")

    # Create a Folium map centered at the depot
    m = folium.Map(location=[depot_lat, depot_lon], zoom_start=8)

    # Assign unique colors for clusters
    unique_clusters = cluster_assignments["Cluster"].unique()
    colors = ['blue', 'green', 'purple', 'orange', 'darkred', 'darkblue', 'cadetblue', 'lightgreen']
    cluster_colors = {cluster: colors[i % len(colors)] for i, cluster in enumerate(unique_clusters)}

    # Add customer markers colored by cluster
    for _, row in cluster_assignments.iterrows():
        folium.Marker(
            location=[row["lat"], row["lon"]],
            popup=f"{row['name']} (Cluster {row['Cluster']})",
            icon=folium.Icon(color=cluster_colors[row["Cluster"]])
        ).add_to(m)

    # Draw circles around cluster centers (optional)
    for cluster_id, cluster_data in cluster_assignments.groupby("Cluster"):
        cluster_center = cluster_data[["lat", "lon"]].mean().values
        folium.Circle(
            location=cluster_center,
            radius=10000,  # Adjust radius as needed
            color=cluster_colors[cluster_id],
            fill=True,
            fill_opacity=0.2,
            popup=f"Cluster {cluster_id}"
        ).add_to(m)

    # Add the depot marker
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
df = load_data('../Data/mini.csv')
dmatrix = distance_matrix(df)

# Dynamically calculate max_clusters
max_clusters = st.slider("Select the maximum number of clusters to evaluate:", 2, len(df) - 1, 10)

# Generate the distance matrix
dmatrix = distance_matrix(df)

# Find optimal clusters
optimal_k, wcss, silhouette_scores = find_optimal_clusters(dmatrix, max_clusters)
st.write(f"Optimal number of clusters: {optimal_k}")

# Display WCSS and Silhouette plot
st.subheader("Elbow Method for Optimal Clusters")
elbow_fig = plot_elbow_method(wcss, optimal_k)
st.pyplot(elbow_fig)

st.subheader("Silhouette Scores Plot:")
silhouette_fig = plot_silhouette_scores(silhouette_scores, optimal_k)
st.pyplot(silhouette_fig)

# Perform clustering and get rankings
ranking, cluster_assignments = get_clustering_ranking(dmatrix, df, max_clusters)
st.subheader("Clustering-Based Rankings")
st.dataframe(ranking)

# Depot coordinates (provided separately)
depot_lat = 52.16521
depot_lon = 5.17215

# Create and display the cluster map
create_cluster_map(df, cluster_assignments, depot_lat, depot_lon)
