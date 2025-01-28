import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import folium
import sys
import os
import math

# Load your custom functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dss import load_data
from distancematrix import compute_distance_matrix

# Load data and distance matrix
# df = load_data('C:/Users/daydo/Documents/GitHub/Project-OBP/Data/medium.csv')
# df = load_data('C:/Users/malou/OneDrive/Documenten/VU/Business Analytics/YEAR 1 - 2024-2025 (Mc)/Project Optimization of Business Processes/Project-OBP/Data/mini.csv')
# df = load_data('../Data/mini.csv')
# dmatrix = compute_distance_matrix(df)  # Precomputed distance matrix

# # Exclude depot row/column from the distance matrix
# dmatrix_without_depot = dmatrix.iloc[1:, 1:]  # Assuming the depot is the first row/column

# Function for DBSCAN clustering using a distance matrix
# def run_dbscan_with_matrix(dmatrix, eps=0.01, min_samples=5):
#     """
#     Perform DBSCAN clustering using a precomputed distance matrix.
#     """
#     dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
#     cluster_labels = dbscan.fit_predict(dmatrix)
#     return cluster_labels

# Function for visualizing clustering results with cluster clouds
def create_customer_and_cluster_map_with_clouds(df, cluster_assignments, depot_lat, depot_lon, method):
    """
    Create an interactive map showing customers and cluster areas for a given clustering method.
    """
    df["Cluster"] = cluster_assignments

    # Create a Folium map centered at the depot
    m = folium.Map(location=[depot_lat, depot_lon], zoom_start=8)

    # Assign unique colors for companies
    company_colors = ['blue', 'green', 'purple', 'orange', 'darkred', 'darkblue', 'cadetblue', 'lightgreen']
    company_color_map = {company: company_colors[i % len(company_colors)] for i, company in enumerate(df["name"].unique())}

    # Add all customers to the map
    for _, row in df.iterrows():
        folium.Marker(
            location=[row["lat"], row["lon"]],
            popup=f"Customer of {row['name']} (Cluster {row['Cluster']}) [{method}]",
            icon=folium.Icon(color=company_color_map[row["name"]])
        ).add_to(m)

    # Draw cluster clouds for valid clusters
    for cluster_id in df["Cluster"].unique():
        if cluster_id == -1:  # Skip noise in DBSCAN
            continue
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
            popup=f"Cluster {cluster_id} ({method})"
        ).add_to(m)

    # Add depot marker
    folium.Marker(
        location=[depot_lat, depot_lon],
        popup="Depot",
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)

    return m

# Function to calculate partnership rankings
def calculate_partnership_rankings(df, cluster_column):
    """
    Calculate rankings for partnerships based on shared clusters.
    """
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

# # Main app
# st.sidebar.title("Clustering Parameters")

# # Calculate silhouette scores
# cluster_range, silhouette_scores = calculate_silhouette_scores(df[["lat", "lon"]].values, max_clusters=15)
# optimal_n_clusters = cluster_range[np.argmax(silhouette_scores)]

# # Sidebar slider with default set to optimal clusters
# max_clusters = st.sidebar.slider("Max Clusters (K-Means)", 2, 15, optimal_n_clusters)
# # Adjust eps slider based on distance matrix max value
# max_distance = float(dmatrix_without_depot.max().max())
# eps = st.sidebar.slider("Epsilon (DBSCAN)", 0.1, max_distance, 0.05, step=0.1)
# st.write(f"Max Distance in Matrix: {max_distance}")

# min_samples = st.sidebar.slider("Min Samples (DBSCAN)", 1, 10, 5)

# depot_lat = st.sidebar.number_input("Depot Latitude", value=52.16521, step=0.01)
# depot_lon = st.sidebar.number_input("Depot Longitude", value=5.17215, step=0.01)

# st.title("Clustering for Partnerships")
# st.write("### K-Means and DBSCAN Clustering Results")

# # Show silhouette scores
# st.write("#### Silhouette Scores for K-Means")
# silhouette_df = pd.DataFrame({"Clusters": cluster_range, "Silhouette Score": silhouette_scores}).set_index("Clusters")
# st.line_chart(silhouette_df)

# # Perform K-Means clustering
# st.write("#### K-Means Clustering")
# kmeans = KMeans(n_clusters=max_clusters, random_state=42)
# kmeans_labels = kmeans.fit_predict(df[["lat", "lon"]].values)
# df["KMeans Cluster"] = kmeans_labels

# # Perform DBSCAN clustering
# st.write("#### DBSCAN Clustering (Using Precomputed Distance Matrix)")
# dbscan_labels = run_dbscan_with_matrix(dmatrix_without_depot, eps=eps, min_samples=min_samples)

# # Ensure alignment
# if len(dbscan_labels) != len(df):
#     raise ValueError(f"DBSCAN labels length ({len(dbscan_labels)}) does not match DataFrame length ({len(df)}).")

# df["DBSCAN Cluster"] = dbscan_labels

# # Visualize clustering results
# kmeans_map = create_customer_and_cluster_map_with_clouds(df, kmeans_labels, depot_lat, depot_lon, method="K-Means")
# dbscan_map = create_customer_and_cluster_map_with_clouds(df, dbscan_labels, depot_lat, depot_lon, method="DBSCAN")

# st.write("#### K-Means Cluster Map")
# st.components.v1.html(kmeans_map._repr_html_(), height=600)

# st.write("#### DBSCAN Cluster Map")
# st.components.v1.html(dbscan_map._repr_html_(), height=600)

# # Calculate partnership rankings
# kmeans_partnerships = calculate_partnership_rankings(df, "KMeans Cluster")
# dbscan_partnerships = calculate_partnership_rankings(df, "DBSCAN Cluster")

# # Display partnership rankings
# st.write("#### Partnership Rankings (K-Means)")
# st.write(kmeans_partnerships)

# st.write("#### Partnership Rankings (DBSCAN)")
# st.write(dbscan_partnerships)

# # Analyze the distance matrix
# st.write("### Distance Matrix Analysis")
# st.write(f"Minimum Distance: {dmatrix_without_depot.min().min()}")
# st.write(f"Maximum Distance: {dmatrix_without_depot.max().max()}")

# # DBSCAN diagnostics
# st.write("### DBSCAN Diagnostics")
# unique_labels, counts = np.unique(dbscan_labels, return_counts=True)
# st.write(f"Unique Clusters (Including Noise): {len(unique_labels)}")
# st.write(f"Cluster Labels: {unique_labels}")
# st.write(f"Cluster Sizes: {dict(zip(unique_labels, counts))}")

# if len(unique_labels) <= 1 or (len(unique_labels) == 2 and -1 in unique_labels):
#     st.warning("DBSCAN found no clusters or only noise. Try adjusting `eps` or `min_samples`.")

# st.write(f"Max Distance: {dmatrix_without_depot.max().max()}")
# st.write(f"Mean Distance: {dmatrix_without_depot.mean().mean()}")
# unique_labels = set(dbscan_labels)
# cluster_sizes = {label: (dbscan_labels == label).sum() for label in unique_labels}
# st.write(f"Unique Clusters: {len(unique_labels)}")
# st.write(f"Cluster Sizes: {cluster_sizes}")


# # Calculate silhouette score for DBSCAN if valid
# st.write("#### Silhouette Score for DBSCAN")
# unique_labels_dbscan = set(dbscan_labels)

# if len(unique_labels_dbscan) > 1:  # At least 2 clusters required
#     silhouette_dbscan = silhouette_score(dmatrix_without_depot, dbscan_labels, metric="precomputed")
#     st.write(f"Silhouette Score (DBSCAN): {silhouette_dbscan}")
# else:
#     st.warning("Silhouette Score for DBSCAN cannot be computed. DBSCAN found only one cluster or noise.")

# # Calculate silhouette score for K-Means
# st.write("#### Silhouette Scores for K-Means")
# max_silhouette_score = max(silhouette_scores)
# optimal_clusters = cluster_range[silhouette_scores.index(max_silhouette_score)]
# st.write(f"Maximum Silhouette Score for K-Means: {max_silhouette_score:.4f} (Optimal Clusters: {optimal_clusters})")


# st.write(f"K-Means Clusters: {len(set(kmeans_labels))}")
# st.write(f"DBSCAN Clusters (excluding noise): {len(set(dbscan_labels)) - (-1 in dbscan_labels)}")

# # Plot comparison of K-Means and DBSCAN cluster assignments
# st.write("### Clustering Comparison")
# comparison_df = df[["lat", "lon", "KMeans Cluster", "DBSCAN Cluster"]]
# st.map(comparison_df)

def calculate_optimal_clusters(data, dmatrix):
    """
    Calculate the optimal number of clusters using the silhouette score.

    Parameters:
    - data (pd.DataFrame): DataFrame with customer locations (latitude and longitude).
    - max_clusters (int): Maximum number of clusters to evaluate.

    Returns:
    - int: Optimal number of clusters based on the highest silhouette score.
    """
    k_min = max(2, int(np.floor(len(data) / 20)))

    if len(data) < 100:
        k_max = 10  
    elif len(data) < 500:
        k_max = min(30, int(len(data) / 10)) 
    else:
        k_max = min(50, int(len(data) / 20))  
    
    if k_max <= k_min:
        k_max = k_min + 20

    scores = []
    cluster_range = range(k_min, k_max)
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(data[["lat", "lon"]])
        score = silhouette_score(dmatrix.drop(index="Depot",columns = "Depot", errors = "ignore" ), cluster_labels, metric = 'precomputed')
        scores.append((n_clusters, score))
    
    # Find the number of clusters with the maximum silhouette score
    optimal_n_clusters = max(scores, key=lambda x: x[1])[0]
    return optimal_n_clusters

def get_cluster_kmeans(df, dmatrix):
    """
    Computes a ranking table for collaborations using K-Means clustering
    with an optimal number of clusters determined by OSRM-based silhouette score.

    Parameters:
    - df (pd.DataFrame): DataFrame with customer and company data.
    - dmatrix (pd.DataFrame): Precomputed OSRM-based distance matrix.

    Returns:
    - pd.DataFrame: A ranking table with ['Rank', 'Company A', 'Company B', 'Shared_Clusters'].
    """
    full_dmatrix = dmatrix.copy()
    cluster_data = df.copy()
    # Step 1: Ensure unique customer IDs for companies
    cluster_data["customer_id"] = cluster_data.groupby("name").cumcount()
    cluster_data["name"] = cluster_data["name"] + "_" + cluster_data["customer_id"].astype(str)
    cluster_data.drop(columns=["customer_id"], inplace=True)  # Remove temp column

    dmatrix_clustering = dmatrix.drop(index="Depot", columns="Depot", errors="ignore")

    dmatrix_clustering.index = cluster_data["name"]
    dmatrix_clustering.columns = cluster_data["name"]

    # ✅ Calculate the optimal number of clusters
    optimal_n_clusters = calculate_optimal_clusters(cluster_data, dmatrix_clustering)

    # ✅ Apply K-Means clustering
    kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42)
    cluster_data["KMeans Cluster"] = kmeans.fit_predict(cluster_data[["lat", "lon"]])

    # ✅ Reassign customers to closest clusters based on *OSRM distances*
    for idx, row in cluster_data.iterrows():
        min_dist = float("inf")
        best_cluster = None
        for cluster in range(optimal_n_clusters):
            centroid_idx = cluster_data[cluster_data["KMeans Cluster"] == cluster].index[0]  # Representative point
            
            # ✅ Use dmatrix_clustering (no depot) to get customer distances
            osrm_dist = dmatrix_clustering.loc[row["name"], cluster_data.loc[centroid_idx, "name"]]
            print(osrm_dist)
            if osrm_dist < min_dist:
                min_dist = osrm_dist
                best_cluster = cluster

        cluster_data.at[idx, "KMeans Cluster"] = best_cluster
    print(cluster_data)
    dmatrix = full_dmatrix
    cluster_data["name"] = df["name"].copy()
    #cluster_data.columns = df["name"]
    print(cluster_data)
    # ✅ Use full_dmatrix (with depot) for VRPy to avoid "Sink and Source Not Connected" error
    # Calculate shared clusters between companies
    partnership_scores = []
    company_names = cluster_data['name'].unique()  # Extract unique company names

    for i, company1 in enumerate(company_names):
        for j, company2 in enumerate(company_names):
            if i < j:  # Ensure each pair is only processed once
                # Get clusters for both companies
                clusters1 = set(cluster_data[cluster_data['name'] == company1]["KMeans Cluster"])
                clusters2 = set(cluster_data[cluster_data['name'] == company2]["KMeans Cluster"])

                # Count shared clusters
                shared_clusters = len(clusters1.intersection(clusters2))

                # Append the result
                partnership_scores.append({
                    'Company A': company1,
                    'Company B': company2,
                    'Score': shared_clusters
                })

    # Create the DataFrame
    partnership_df = pd.DataFrame(partnership_scores)

    # Sort by shared clusters in descending order
    partnership_df = partnership_df.sort_values('Score', ascending=False).reset_index(drop=True)

    # Add a ranking column
    partnership_df['Rank'] = partnership_df.index + 1

    # Reorder columns for clarity
    return partnership_df[['Rank', 'Company A', 'Company B', 'Score']] #, optimal_n_clusters 