import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ----------------- Partnership Rankings -----------------

def calculate_partnership_rankings(df, cluster_column):
    """
    Calculate rankings based on the number of shared clusters between companies.

    Parameters:
    - df (pd.DataFrame): DataFrame containing company and cluster assignment data.
    - cluster_column (str): Column name containing cluster assignments.

    Returns:
    - pd.DataFrame: DataFrame containing ranked partnerships based on shared clusters.
    """
    partnerships = {}
    
    # Iterate through all unique pairs of companies
    for name1 in df["name"].unique():
        for name2 in df["name"].unique():
            if name1 < name2:  # Avoid duplicate pairs
                # Count shared clusters between the two companies
                shared_clusters = len(
                    set(df[df["name"] == name1][cluster_column]).intersection(
                        set(df[df["name"] == name2][cluster_column])
                    )
                )
                partnerships[frozenset([name1, name2])] = shared_clusters

    # Sort partnerships by the number of shared clusters in descending order
    partnerships = sorted(partnerships.items(), key=lambda x: x[1], reverse=True)

    # Create a DataFrame to store partnership rankings
    partnership_df = pd.DataFrame(
        [(list(p)[0], list(p)[1], score) for p, score in partnerships],
        columns=["Company A", "Company B", f"Shared Clusters ({cluster_column})"]
    )
    return partnership_df

# ----------------- Optimal Cluster Calculation -----------------

def calculate_optimal_clusters(data, dmatrix):
    """
    Determine the optimal number of clusters using the silhouette score.

    Parameters:
    - data (pd.DataFrame): DataFrame containing location data (latitude, longitude).
    - dmatrix (pd.DataFrame): Precomputed distance matrix between customers.

    Returns:
    - int: Optimal number of clusters.
    """
    # Set minimum number of clusters
    k_min = max(2, int(np.floor(len(data) / 20)))

    # Set maximum number of clusters based on dataset size
    if len(data) < 100:
        k_max = 10  
    elif len(data) < 500:
        k_max = min(30, int(len(data) / 10))
    else:
        k_max = min(50, int(len(data) / 20))

    if k_max <= k_min:
        k_max = k_min + 20  # Ensure a minimum range of clusters

    scores = []
    cluster_range = range(k_min, k_max)

    # Evaluate the silhouette score for different cluster counts
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(data[["lat", "lon"]])
        score = silhouette_score(dmatrix.drop(index="Depot", columns="Depot", errors="ignore"), cluster_labels, metric='precomputed')
        scores.append((n_clusters, score))

    # Select the number of clusters with the maximum silhouette score
    optimal_n_clusters = max(scores, key=lambda x: x[1])[0]
    return optimal_n_clusters

# ----------------- KMeans Clustering and Partnership Scoring -----------------

def get_cluster_kmeans(df, dmatrix):
    """
    Apply K-Means clustering to customer data and compute partnership rankings.

    Parameters:
    - df (pd.DataFrame): DataFrame containing customer data with company names and locations.
    - dmatrix (pd.DataFrame): Precomputed distance matrix including customers and depot.

    Returns:
    - pd.DataFrame: Ranked partnerships based on shared clusters.
    """
    full_dmatrix = dmatrix.copy()
    cluster_data = df.copy()

    # Ensure unique identifiers for each customer
    cluster_data["customer_id"] = cluster_data.groupby("name").cumcount()
    cluster_data["name"] = cluster_data["name"] + "_" + cluster_data["customer_id"].astype(str)
    cluster_data.drop(columns=["customer_id"], inplace=True)

    # Remove depot from distance matrix for clustering
    dmatrix_clustering = full_dmatrix.drop(index="Depot", columns="Depot", errors="ignore")

    # Set correct names for clustering
    dmatrix_clustering.index = cluster_data["name"]
    dmatrix_clustering.columns = cluster_data["name"]

    # Calculate the optimal number of clusters
    optimal_n_clusters = calculate_optimal_clusters(cluster_data, dmatrix_clustering)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42)
    cluster_data["KMeans Cluster"] = kmeans.fit_predict(cluster_data[["lat", "lon"]])

    # Reassign customers to clusters based on nearest cluster center (using OSRM distances)
    for idx, row in cluster_data.iterrows():
        min_dist = float("inf")
        best_cluster = None

        # Find the closest cluster for the current customer
        for cluster in range(optimal_n_clusters):
            centroid_idx = cluster_data[cluster_data["KMeans Cluster"] == cluster].index[0]
            osrm_dist = dmatrix_clustering.loc[row["name"], cluster_data.loc[centroid_idx, "name"]]
            
            if osrm_dist < min_dist:
                min_dist = osrm_dist
                best_cluster = cluster

        cluster_data.at[idx, "KMeans Cluster"] = best_cluster

    # Restore original company names for evaluation
    cluster_data["name"] = df["name"].copy()

    # Calculate shared clusters between pairs of companies
    partnership_scores = []
    company_names = cluster_data['name'].unique()

    for i, company1 in enumerate(company_names):
        for j, company2 in enumerate(company_names):
            if i < j:  # Process each pair only once
                # Find clusters for both companies
                clusters1 = set(cluster_data[cluster_data['name'] == company1]["KMeans Cluster"])
                clusters2 = set(cluster_data[cluster_data['name'] == company2]["KMeans Cluster"])

                # Count shared clusters
                shared_clusters = len(clusters1.intersection(clusters2))

                # Store partnership information
                partnership_scores.append({
                    'Company A': company1,
                    'Company B': company2,
                    'Score': shared_clusters
                })

    # Create DataFrame with partnership rankings
    partnership_df = pd.DataFrame(partnership_scores)

    # Sort partnerships by score and assign ranks
    partnership_df = partnership_df.sort_values('Score', ascending=False).reset_index(drop=True)
    partnership_df['Rank'] = partnership_df.index + 1

    return partnership_df[['Rank', 'Company A', 'Company B', 'Score']]
