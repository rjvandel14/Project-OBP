import streamlit as st
import sys
import os
import math
import folium
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load custom functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dss import load_data
from osrm_dmatrix import compute_distance_matrix

# Load data and distance matrix
df = load_data('../Data/mini.csv')
dmatrix = compute_distance_matrix(df)

# Function to calculate partnership rankings
def calculate_partnership_rankings(df, cluster_column):
    partnerships = {}
    for name1 in df["name"].unique():
        for name2 in df["name"].unique():
            if name1 < name2:  # Avoid duplicate pairs
                shared_clusters = len(
                    set(df[df["name"] == name1][cluster_column]).intersection(
                        set(df[df["name"] == name2][cluster_column])
                    )
                )
                partnerships[frozenset([name1, name2])] = shared_clusters
    partnerships = sorted(partnerships.items(), key=lambda x: x[1], reverse=True)
    partnership_df = pd.DataFrame(
        [(list(p)[0], list(p)[1], score) for p, score in partnerships],
        columns=["Company A", "Company B", f"Shared Clusters ({cluster_column})"]
    )
    return partnership_df

# Visualization functions
def create_silhouette_plot(cluster_range, silhouette_scores):
    plt.figure(figsize=(10, 5))
    plt.plot(cluster_range, silhouette_scores, marker='o', linestyle='-', color='b')
    plt.title("Silhouette Scores for K-Means Clustering")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    st.pyplot(plt)

def create_k_distance_plot(k_distances):
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(k_distances)), k_distances, marker='o', linestyle='-', color='r')
    plt.title("K-Distance Graph for DBSCAN")
    plt.xlabel("Data Points (sorted)")
    plt.ylabel("Distance to k-th Nearest Neighbor")
    st.pyplot(plt)

def create_customer_and_cluster_map_with_clouds(df, cluster_assignments, depot_lat, depot_lon, method):
    df["Cluster"] = cluster_assignments
    m = folium.Map(location=[depot_lat, depot_lon], zoom_start=8)
    company_colors = ['blue', 'green', 'purple', 'orange', 'darkred', 'darkblue', 'cadetblue', 'lightgreen']
    company_color_map = {company: company_colors[i % len(company_colors)] for i, company in enumerate(df["name"].unique())}

    # Add customer markers
    for _, row in df.iterrows():
        folium.Marker(
            location=[row["lat"], row["lon"]],
            popup=f"Customer of {row['name']} (Cluster {row['Cluster']}) [{method}]",
            icon=folium.Icon(color=company_color_map[row["name"]])
        ).add_to(m)

    # Draw cluster clouds
    for cluster_id in df["Cluster"].unique():
        if cluster_id == -1:  # Skip noise
            continue
        cluster_data = df[df["Cluster"] == cluster_id]
        cluster_center = cluster_data[["lat", "lon"]].mean().values
        distances = np.linalg.norm(cluster_data[["lat", "lon"]].values - cluster_center, axis=1) * 111000
        cluster_radius = max(distances.max(), 20000)
        folium.Circle(
            location=cluster_center,
            radius=cluster_radius,
            color="black",
            fill=True,
            fill_opacity=0.2,
            popup=f"Cluster {cluster_id} ({method})"
        ).add_to(m)

    folium.Marker(
        location=[depot_lat, depot_lon],
        popup="Depot",
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)

    return m

# Streamlit app
st.sidebar.title("Clustering Parameters")
depot_lat = st.sidebar.number_input("Depot Latitude", value=52.16521, step=0.01)
depot_lon = st.sidebar.number_input("Depot Longitude", value=5.17215, step=0.01)

max_clusters = st.sidebar.slider("Max Clusters (K-Means)", 2, 20, 10)
eps = st.sidebar.slider("Epsilon (DBSCAN)", 0.01, 1.0, 0.1, step=0.01)
min_samples = st.sidebar.slider("Min Samples (DBSCAN)", 1, 10, 5)

# K-Means clustering
st.write("## K-Means Clustering")
kmeans = KMeans(n_clusters=max_clusters, random_state=42)
kmeans_labels = kmeans.fit_predict(df[["lat", "lon"]])
df["KMeans Cluster"] = kmeans_labels

# Silhouette scores for K-Means
cluster_range = range(2, max_clusters + 1)
silhouette_scores = [silhouette_score(df[["lat", "lon"]], KMeans(n_clusters=k).fit_predict(df[["lat", "lon"]])) for k in cluster_range]

st.write("### Silhouette Scores")
create_silhouette_plot(cluster_range, silhouette_scores)

# K-Means cluster map
st.write("### K-Means Cluster Map")
kmeans_map = create_customer_and_cluster_map_with_clouds(df, kmeans_labels, depot_lat, depot_lon, method="K-Means")
st.components.v1.html(kmeans_map._repr_html_(), height=600)

# DBSCAN clustering
st.write("## DBSCAN Clustering")
dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")

# Remove depot from distance matrix
dmatrix_clustering = dmatrix.iloc[1:, 1:]

# Ensure index and columns match df
dmatrix_clustering.index = df.index
dmatrix_clustering.columns = df.index

# DBSCAN clustering
dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
dbscan_labels = dbscan.fit_predict(dmatrix_clustering)

# Assign DBSCAN cluster labels to df
df["DBSCAN Cluster"] = dbscan_labels


# K-Distance graph for DBSCAN
k = min_samples - 1
k_distances = np.sort(np.partition(dmatrix, k, axis=1)[:, k])

st.write("### K-Distance Graph for DBSCAN")
create_k_distance_plot(k_distances)

# DBSCAN cluster map
st.write("### DBSCAN Cluster Map")
dbscan_map = create_customer_and_cluster_map_with_clouds(df, dbscan_labels, depot_lat, depot_lon, method="DBSCAN")
st.components.v1.html(dbscan_map._repr_html_(), height=600)

# Partnership rankings
st.write("### Partnership Rankings (K-Means)")
kmeans_partnerships = calculate_partnership_rankings(df, "KMeans Cluster")
st.write(kmeans_partnerships)

st.write("### Partnership Rankings (DBSCAN)")
dbscan_partnerships = calculate_partnership_rankings(df, "DBSCAN Cluster")
st.write(dbscan_partnerships)

#streamlit run "C:\Users\Lenovo\OneDrive\Bureaublad\Master BA\Project Optimization of Business Processes P3 J1\Project-OBP\App\ranking_functions\ranking_plots.py"