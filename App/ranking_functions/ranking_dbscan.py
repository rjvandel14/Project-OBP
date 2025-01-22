from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np

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

def recommend_minPts(dataset_size, dimensions=2):
    """
    Recommends an appropriate minPts value for DBSCAN based on dataset size and dimensions.
    
    Parameters:
    - dataset_size (int): Number of points in the dataset.
    - dimensions (int): Number of dimensions of the data (default is 2 for 2D data).
    
    Returns:
    - int: Recommended minPts value.
    """
    if dataset_size < 100:
        return max(4, dimensions + 1)  # Small datasets: Use the rule of thumb
    elif 100 <= dataset_size <= 1000:
        return max(5, dimensions + 1)  # Medium datasets: Slightly larger minPts
    else:
        return max(10, dimensions + 1)  # Large datasets: Higher minPts for noise filtering

def find_optimal_epsilon(dist_matrix, minPts):
    # Step 1: Compute k-distances
    k = minPts - 1
    k_distances = np.sort(np.partition(dist_matrix, k, axis=1)[:, k])

    # Step 2: Find the point of maximum curvature
    # Line from the first point to the last point
    start_point = np.array([0, k_distances[0]])
    end_point = np.array([len(k_distances) - 1, k_distances[-1]])

    # Vector from start to end
    line_vector = end_point - start_point
    line_vector_normalized = line_vector / np.linalg.norm(line_vector)

    # Distances from points to the line
    distances_to_line = []
    for i, distance in enumerate(k_distances):
        point = np.array([i, distance])
        vec_from_start = point - start_point
        projection = np.dot(vec_from_start, line_vector_normalized) * line_vector_normalized
        perpendicular_vec = vec_from_start - projection
        distances_to_line.append(np.linalg.norm(perpendicular_vec))

    # Find the index of the maximum distance
    optimal_index = np.argmax(distances_to_line)
    optimal_epsilon = k_distances[optimal_index]

    return optimal_epsilon
