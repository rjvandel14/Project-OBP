from sklearn.cluster import DBSCAN
import pandas as pd

def get_dbscan_ranking(df, dmatrix, eps, min_samples):
    """
    Computes a ranking table for collaborations using DBSCAN clustering
    with shared clusters as the ranking criterion.

    Parameters:
    - df (pd.DataFrame): DataFrame with company and customer data. Must include 'name', 'lat', and 'lon'.
    - dmatrix (pd.DataFrame): Precomputed distance matrix.
    - eps (float): Epsilon parameter for DBSCAN (maximum distance for points to be in the same cluster).
    - min_samples (int): Minimum number of samples in a neighborhood to form a cluster.

    Returns:
    - pd.DataFrame: A ranking table with columns ['Rank', 'Company A', 'Company B', 'Shared_Clusters'].
    """

    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    cluster_labels = dbscan.fit_predict(dmatrix)

    # Add cluster labels to the DataFrame
    df["DBSCAN Cluster"] = cluster_labels

    # Calculate shared clusters between companies
    partnership_scores = []
    company_names = df['name'].unique()  # Extract unique company names

    for i, company1 in enumerate(company_names):
        for j, company2 in enumerate(company_names):
            if i < j:  # Ensure each pair is only processed once
                # Get clusters for both companies
                clusters1 = set(df[df['name'] == company1]["DBSCAN Cluster"])
                clusters2 = set(df[df['name'] == company2]["DBSCAN Cluster"])

                # Count shared clusters, ignoring noise (-1)
                shared_clusters = len(clusters1.intersection(clusters2) - {-1})

                # Append the result
                partnership_scores.append({
                    'Company A': company1,
                    'Company B': company2,
                    'Shared_Clusters': shared_clusters
                })

    # Create the DataFrame
    partnership_df = pd.DataFrame(partnership_scores)

    # Sort by shared clusters in descending order
    partnership_df = partnership_df.sort_values('Shared_Clusters', ascending=False).reset_index(drop=True)

    # Add a ranking column
    partnership_df['Rank'] = partnership_df.index + 1

    # Reorder columns for clarity
    return partnership_df[['Rank', 'Company A', 'Company B', 'Shared_Clusters']]

def get_optimal_eps():
    return 0

def get_min_samples():
    return 0