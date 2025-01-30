import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


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

def calculate_optimal_clusters(data, dmatrix):
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
    full_dmatrix = dmatrix.copy()
    cluster_data = df.copy()

    # Ensure unique customer IDs for companies
    cluster_data["customer_id"] = cluster_data.groupby("name").cumcount()
    cluster_data["name"] = cluster_data["name"] + "_" + cluster_data["customer_id"].astype(str)
    cluster_data.drop(columns=["customer_id"], inplace=True)  # Remove temp column

    # Remove depot from dmatrix
    dmatrix_clustering = full_dmatrix.drop(index="Depot", columns="Depot", errors="ignore")

    # Ensure matching name column
    dmatrix_clustering.index = cluster_data["name"]
    dmatrix_clustering.columns = cluster_data["name"]

    # Calculate the optimal number of clusters
    optimal_n_clusters = calculate_optimal_clusters(cluster_data, dmatrix_clustering)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42)
    cluster_data["KMeans Cluster"] = kmeans.fit_predict(cluster_data[["lat", "lon"]])

    # Reassign customers to closest clusters based on OSRM distances
    for idx, row in cluster_data.iterrows():
        min_dist = float("inf")
        best_cluster = None
        for cluster in range(optimal_n_clusters):
            centroid_idx = cluster_data[cluster_data["KMeans Cluster"] == cluster].index[0]  # Representative point
            
            # Use dmatrix_clustering (no depot) to get customer distances
            osrm_dist = dmatrix_clustering.loc[row["name"], cluster_data.loc[centroid_idx, "name"]]
            if osrm_dist < min_dist:
                min_dist = osrm_dist
                best_cluster = cluster

        cluster_data.at[idx, "KMeans Cluster"] = best_cluster
    cluster_data["name"] = df["name"].copy()

    # Calculate shared clusters between companies
    partnership_scores = []
    company_names = cluster_data['name'].unique()  # Get unique company names

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

    partnership_df = pd.DataFrame(partnership_scores)

    # Sort by shared clusters in descending order
    partnership_df = partnership_df.sort_values('Score', ascending=False).reset_index(drop=True)

    # Add a ranking column
    partnership_df['Rank'] = partnership_df.index + 1

    return partnership_df[['Rank', 'Company A', 'Company B', 'Score']] 