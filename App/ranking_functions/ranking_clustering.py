import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import folium
import sys
import os

# Load your custom functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dss import load_data
df = load_data('C:/Users/daydo/Documents/GitHub/Project-OBP/Data/mini.csv')

# Determine optimal number of clusters
def optimal_clusters(data, max_clusters=10):
    scores = []
    cluster_range = range(2, max_clusters + 1)
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        score = silhouette_score(data, cluster_labels)
        scores.append(score)
    return cluster_range, scores

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

    return m

# Sidebar options with default depot values
st.sidebar.title("Clustering Parameters")
max_clusters = st.sidebar.slider("Max Clusters", 2, 15, 10)
depot_lat = st.sidebar.number_input("Depot Latitude", value=52.16521, step=0.01)  # Default depot latitude
depot_lon = st.sidebar.number_input("Depot Longitude", value=5.17215, step=0.01)  # Default depot longitude

# Main app
st.title("Clustering for Partnerships")
st.write("### Step 2: Determine Optimal Clusters and Visualize Results")

# Clustering based on lat/lon
st.write("#### Determining Optimal Number of Clusters")
cluster_data = df[["lat", "lon"]].values
cluster_range, scores = optimal_clusters(cluster_data, max_clusters=max_clusters)

# Show Silhouette Scores
st.write("#### Silhouette Scores")
st.line_chart(pd.DataFrame({"Clusters": cluster_range, "Silhouette Score": scores}).set_index("Clusters"))

# Optimal number of clusters
optimal_n = cluster_range[np.argmax(scores)]
st.write(f"The optimal number of clusters is **{optimal_n}** based on the highest silhouette score.")

# Run KMeans with optimal clusters
kmeans = KMeans(n_clusters=optimal_n, random_state=42)
df["Cluster"] = kmeans.fit_predict(cluster_data)

# Rank partnerships based on shared clusters
st.write("#### Partnership Rankings")
partnerships = {}
for name1 in df["name"].unique():
    for name2 in df["name"].unique():
        if name1 < name2:
            shared_clusters = len(set(df[df["name"] == name1]["Cluster"]).intersection(set(df[df["name"] == name2]["Cluster"])))
            partnerships[frozenset([name1, name2])] = shared_clusters

partnerships = sorted(partnerships.items(), key=lambda x: x[1], reverse=True)
partnerships_df = pd.DataFrame([(list(p)[0], list(p)[1], score) for p, score in partnerships], columns=["Company A", "Company B", "Shared Clusters"])
st.write(partnerships_df)

# Show cluster map
st.write("#### Customer and Cluster Map")
map = create_customer_and_cluster_map(df, kmeans.labels_, depot_lat, depot_lon)  # Pass depot coordinates
st.components.v1.html(map._repr_html_(), height=600)