import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

def get_dbscan_ranking(df, dmatrix):
    """
    Perform DBSCAN clustering and calculate partnership rankings based on shared clusters.

    Parameters:
    - df (pd.DataFrame): DataFrame containing company and customer data.
    - dmatrix (pd.DataFrame): Precomputed distance matrix including customers and depot.

    Returns:
    - pd.DataFrame: Ranked partnerships based on the number of shared clusters.
    """
    # Remove depot from the distance matrix to focus on customers
    dmatrix = dmatrix.drop(index="Depot", columns="Depot", errors="ignore")

    # Determine optimal minPts and epsilon for DBSCAN
    min_samples = recommend_minPts(len(df))
    eps = find_optimal_epsilon(dmatrix, min_samples)

    # Perform DBSCAN clustering using precomputed distances
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    cluster_labels = dbscan.fit_predict(dmatrix)

    # Add cluster labels to the DataFrame
    df["DBSCAN Cluster"] = cluster_labels

    # Calculate the number of shared clusters between pairs of companies
    partnership_scores = []
    company_names = df['name'].unique()  # Get unique company names

    for i, company1 in enumerate(company_names):
        for j, company2 in enumerate(company_names):
            if i < j:  # Ensure each pair is only processed once
                # Get clusters assigned to both companies
                clusters1 = set(df[df['name'] == company1]["DBSCAN Cluster"])
                clusters2 = set(df[df['name'] == company2]["DBSCAN Cluster"])

                # Count shared clusters, excluding noise (-1)
                shared_clusters = len(clusters1.intersection(clusters2) - {-1})

                # Store partnership score
                partnership_scores.append({
                    'Company A': company1,
                    'Company B': company2,
                    'Score': shared_clusters
                })

    # Create a DataFrame with partnership scores
    partnership_df = pd.DataFrame(partnership_scores)

    # Sort partnerships by shared clusters in descending order and assign ranks
    partnership_df = partnership_df.sort_values('Score', ascending=False).reset_index(drop=True)
    partnership_df['Rank'] = partnership_df.index + 1  # Assign sequential ranks

    # Reorder columns for clarity
    return partnership_df[['Rank', 'Company A', 'Company B', 'Score']]

def recommend_minPts(dataset_size, dimensions=2):
    """
    Recommend an appropriate minPts value for DBSCAN based on dataset size and dimensions.

    Parameters:
    - dataset_size (int): Number of data points in the dataset.
    - dimensions (int): Number of dimensions (default is 2 for latitude and longitude).

    Returns:
    - int: Recommended minPts value.
    """
    if dataset_size < 100:
        return max(4, dimensions + 1)  # Small datasets require smaller minPts
    elif dataset_size < 1000:
        return max(5, dimensions + 1)  # Medium datasets need slightly larger minPts
    else:
        return max(10, dimensions + 1)  # Large datasets require higher minPts to filter noise

def find_optimal_epsilon(dist_matrix, minPts):
    """
    Determine the optimal epsilon value for DBSCAN using the k-distance method.

    Parameters:
    - dist_matrix (pd.DataFrame or np.array): Precomputed distance matrix.
    - minPts (int): Minimum number of points for a dense region in DBSCAN.

    Returns:
    - float: Optimal epsilon value for clustering.
    """
    # Calculate the k-distances for each point
    k = minPts - 1
    k_distances = np.sort(np.partition(dist_matrix, k, axis=1)[:, k])

    # Define the start and end points for the elbow method
    start_point = np.array([0, k_distances[0]])
    end_point = np.array([len(k_distances) - 1, k_distances[-1]])

    # Calculate the vector from start to end
    line_vector = end_point - start_point
    line_vector_normalized = line_vector / np.linalg.norm(line_vector)

    # Compute perpendicular distances from each point to the line
    distances_to_line = []
    for i, distance in enumerate(k_distances):
        point = np.array([i, distance])
        vec_from_start = point - start_point
        projection = np.dot(vec_from_start, line_vector_normalized) * line_vector_normalized
        perpendicular_vec = vec_from_start - projection
        distances_to_line.append(np.linalg.norm(perpendicular_vec))

    # Identify the elbow point as the optimal epsilon
    optimal_index = np.argmax(distances_to_line)
    optimal_epsilon = k_distances[optimal_index]

    return optimal_epsilon
