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

# Load your custom functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dss import load_data

def find_optimal_clusters(data, max_clusters=10):
    """
    Automatically determine the optimal number of clusters using:
    1. Silhouette Score
    2. Elbow Method (KneeLocator)
    """
    wcss = []
    silhouette_scores = []

    for k in range(2, min(max_clusters + 1, len(data))):  # Start from k=2
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(data)

        # Calculate WCSS
        wcss.append(kmeans.inertia_)

        # Calculate Silhouette Score if possible
        if len(data) > k:
            silhouette_scores.append(silhouette_score(data, cluster_labels))
        else:
            silhouette_scores.append(None)

    # Find the optimal k using the Elbow Method
    kn = KneeLocator(range(2, len(wcss) + 2), wcss, curve="convex", direction="decreasing")
    elbow_k = kn.knee

    # Fallback if no clear elbow is found
    if elbow_k is None:
        elbow_k = np.argmax(silhouette_scores) + 2  # +2 because k starts from 2

    return elbow_k, wcss, silhouette_scores

def create_customer_and_cluster_map(df, cluster_assignments, depot_lat, depot_lon):
    """
    Create an interactive map showing customers and cluster areas.
    """
    df["Cluster"] = cluster_assignments

    # Create a Folium map centered at the depot
    m = folium.Map(location=[depot_lat, depot_lon], zoom_start=8)

    # Assign unique colors for companies
    company_colors = ['blue', 'green', 'purple', 'orange', 'darkred', 'darkblue']
    company_color_map = {company: company_colors[i % len(company_colors)] for i, company in enumerate(df["name"].unique())}

    # Add all customers to the map
    for _, row in df.iterrows():
        folium.Marker(
            location=[row["lat"], row["lon"]],
            popup=f"Customer of {row['name']} (Cluster {row['Cluster']})",
            icon=folium.Icon(color=company_color_map[row["name"]])
        ).add_to(m)

    # Draw cluster clouds
    for cluster_id in df["Cluster"].unique():
        cluster_data = df[df["Cluster"] == cluster_id]
        cluster_center = cluster_data[["lat", "lon"]].mean().values
        distances = np.linalg.norm(cluster_data[["lat", "lon"]].values - cluster_center, axis=1) * 111000  # Approximate meters
        cluster_radius = max(distances.max(), 20000)  # Maximum distance or minimum radius
        folium.Circle(
            location=cluster_center,
            radius=cluster_radius,
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

    st.subheader("Customer and Cluster Map")
    st.write("Interactive map showing all customers and clusters with cluster clouds.")
    st.components.v1.html(m._repr_html_(), height=600)


def rank_partnerships(df):
    """
    Rank company partnerships based on cluster overlap.
    """
    partnerships = []
    for cluster_id in df["Cluster"].unique():
        companies_in_cluster = df[df["Cluster"] == cluster_id]["name"].unique()
        for i, company_a in enumerate(companies_in_cluster):
            for company_b in companies_in_cluster[i + 1:]:
                partnerships.append({
                    "Company A": company_a,
                    "Company B": company_b,
                    "Cluster": cluster_id
                })

    partnership_df = pd.DataFrame(partnerships)
    partnership_df["Rank"] = partnership_df["Cluster"].rank(method="dense").astype(int)

    return partnership_df

# Streamlit app
st.title("Company Partnership Optimization")

# Load data
df = load_data('C:/Users/daydo/Documents/GitHub/Project-OBP/Data/mini.csv')

# Dynamically calculate max_clusters
max_clusters = st.slider("Select the maximum number of clusters to evaluate:", 2, len(df), 4)

# Find the optimal clusters for customer locations
optimal_k, wcss, silhouette_scores = find_optimal_clusters(df[["lat", "lon"]], max_clusters)
st.write(f"Optimal number of clusters: {optimal_k}")

# Display WCSS and Silhouette Score plots
st.subheader("Elbow Method for Optimal Clusters")
fig1 = plt.figure()
plt.plot(range(2, len(wcss) + 2), wcss, marker="o")
plt.axvline(optimal_k, color="red", linestyle="--")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
st.pyplot(fig1)

st.subheader("Silhouette Scores for Clustering")
fig2 = plt.figure()
plt.plot(range(2, len(silhouette_scores) + 2), silhouette_scores, marker="o")
plt.axvline(optimal_k, color="red", linestyle="--")
plt.title("Silhouette Scores")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
st.pyplot(fig2)

# Perform clustering on customer locations
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
customer_clusters = kmeans.fit_predict(df[["lat", "lon"]])

# Create and display the customer and cluster map
depot_lat = 52.16521
depot_lon = 5.17215
create_customer_and_cluster_map(df, customer_clusters, depot_lat, depot_lon)

# Rank partnerships
st.subheader("Partnership Rankings")
partnerships = rank_partnerships(df)
st.dataframe(partnerships)
