# Partnership ranking methods

# Implements ranking logic and scoring features like overlap and cost savings.
# Validates rankings using heuristics and exact VRP results.

# Role: Implements ranking logic.
# Interactions:
# With dss.py:
# Provides ranking algorithms and outputs scores based on features like overlap and savings potential.
# Ensures rankings are consistent with DSS results by validating with small VRP solutions.
# With routing.py:
# Compares rankings to exact VRP solutions for validation (top and bottom-ranked partnerships).
import folium
import pandas as pd
import streamlit as st
from dss import load_data
from distancematrix import compute_distance_matrix
from routing import all_cvrp
from scipy.stats import spearmanr
from ranking_functions.ranking_minmax import get_min_max_ranking
from ranking_functions.ranking_clustering import get_cluster_kmeans
from ranking_functions.ranking_dbscan import get_dbscan_ranking

def get_mock_ranking():
    """
    Returns a mock ranking table for collaborations.
    """
    mock_data = pd.DataFrame({
    "Rank": [1, 2, 3],
    "Company A": ["Company 1", "Company 2", "Company 3"],
    "Company B": ["Company 4", "Company 5", "Company 6"],
    })

    return mock_data

# Visualize the customer locations given a company
def create_partnership_map(df, depot_lat, depot_lon, output_file='map.html'):
    """
    Create an interactive map showing company customer locations and depot.

    Parameters:
    - df (pd.DataFrame): DataFrame containing customer data with columns ['latitude', 'longitude', 'name'].
    - depot_lat (float): Latitude of the depot.
    - depot_lon (float): Longitude of the depot.
    - output_file (str): Name of the HTML file to save the map.

    Returns:
    - folium.Map: The interactive Folium map object.
    """
    # Create a Folium map centered at the depot
    m = folium.Map(location=[depot_lat, depot_lon], zoom_start=12)

    # Add the depot marker
    folium.Marker(
        location=[depot_lat, depot_lon],
        popup="Depot",
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)

    # Assign a unique color for each company
    company_names = df['name'].unique()  # Use the 'name' column to identify companies
    colors = ['blue', 'green', 'purple', 'orange', 'darkred', 'darkblue', 'cadetblue', 'lightgreen']  # Add more if needed
    color_map = {name: colors[i % len(colors)] for i, name in enumerate(company_names)}

    # Add customer markers for each company
    for _, row in df.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"Customer of {row['name']}",  # Display company name in the popup
            icon=folium.Icon(color=color_map[row['name']])
        ).add_to(m)

    # Save the map to an HTML file
    m.save(output_file)

    # Streamlit output
    st.title("Partnership Map")
    st.write("Interactive map showing company customers and depot.")
    st.components.v1.html(m._repr_html_(), height=600)

    return m

# # Depot coordinates
# depot_lat = 52.16521
# depot_lon = 5.17215
# # create_partnership_map(df, depot_lat, depot_lon, output_file='partnership_map.html')

# df = load_data('C:/Users/malou/OneDrive/Documenten/VU/Business Analytics/YEAR 1 - 2024-2025 (Mc)/Project Optimization of Business Processes/Project-OBP/Data/many.csv')

def get_validation(vehicle_capacity, cost_per_km, fixed_cost_per_truck, data, dmatrix, ranking):
    selected_pairs = pd.concat([ranking.head(3), ranking.tail(3)])
    # Evaluate VRP for selected pairs
    evaluation_results = []
    for _, row in selected_pairs.iterrows():
        company_a = row["Company A"]
        company_b = row["Company B"]
        
        # Solve VRP for the selected companies using all_cvrp
        vrp_result = all_cvrp(
            vehicle_capacity,
            cost_per_km,
            fixed_cost_per_truck,
            company_a,
            company_b,
            data,
            dmatrix 
        )

        # Append results
        evaluation_results.append({
            "Company A": company_a,
            "Company B": company_b,
            "Heuristic Rank": row["Rank"],
            "Score": row["Score"],
            "VRP Collaboration Saving Cost": vrp_result["Total Cost"][0] + vrp_result["Total Cost"][1] - vrp_result["Total Cost"][2]  # Cost savings
        })

    # Create DataFrame with results
    evaluation_df = pd.DataFrame(evaluation_results)
    
    # Compute Spearman Rank Correlation for K-Means rankings
    heuristic_scores = evaluation_df["Score"]
    vrp_scores = evaluation_df["VRP Collaboration Saving Cost"]

    spearman_corr_dbscan, p_value_dbscan = spearmanr(heuristic_scores, vrp_scores)

    # Display results
    print("Evaluation Results (Top 3 and Bottom 3) for Ranking:")
    print(evaluation_df)
    print(f"\nSpearman Rank Correlation for Ranking: {spearman_corr_dbscan:.2f}")
    print(f"P-Value: {p_value_dbscan:.2e}")


df = load_data('../Data/many.csv')
dmatrix = compute_distance_matrix(df)
# rankingdbscan = get_dbscan_ranking(df, dmatrix.drop(index='Depot', columns='Depot'), 30, 6) 
# rankingclusterkmeans = get_cluster_kmeans(df, max_clusters=10)
rankingdbscan = get_dbscan_ranking(df, dmatrix.drop(index='Depot', columns='Depot'), 38, 2) 
rankingclusterkmeans = get_cluster_kmeans(df) # second argument in get_cluster_kmeans is for vehicle capacity
rankingminmax = get_min_max_ranking(dmatrix, df)
get_validation(10, 2.5, 50, df, dmatrix, rankingclusterkmeans)

# # Show all rows and columns
# pd.set_option('display.max_rows', None)  # None means no limit on rows
# pd.set_option('display.max_columns', None)  # None means no limit on columns

# # # Print the ranking
# # print("Partnership Ranking (Min-Max Method):")
# # print(ranking)


## EVALUATION OF MINMAX RANKING
# Select top 3 and bottom 3 from the ranking
# selected_pairs = pd.concat([rankingminmax.head(3), rankingminmax.tail(3)])

# # Define parameters for VRP
# vehicle_capacity = 20  # Example vehicle capacity
# cost_per_km = 1        # Cost per kilometer
# fixed_cost_per_truck = 1  # Fixed cost for each truck

# # Evaluate VRP for selected pairs
# evaluation_results = []
# for _, row in selected_pairs.iterrows():
#     company_a = row["Company A"]
#     company_b = row["Company B"]
    
#     # Solve VRP for the selected companies using all_cvrp
#     vrp_result = all_cvrp(
#         vehicle_capacity=vehicle_capacity,
#         cost_per_km=cost_per_km,
#         fixed_cost_per_truck=fixed_cost_per_truck,
#         company_a=company_a,
#         company_b=company_b,
#         data=df,
#         dmatrix=dmatrix
#     )
    
#     # Append results
#     evaluation_results.append({
#         "Company A": company_a,
#         "Company B": company_b,
#         "Heuristic Rank": row["Rank"],
#         "Min-Max Score": row["Min_Max_Score"],
#         "VRP Collaboration Saving Cost": vrp_result["Total Cost"][0] + vrp_result["Total Cost"][1] - vrp_result["Total Cost"][2]  # Cost for collaboration
#     })

# # Create DataFrame with results
# evaluation_df = pd.DataFrame(evaluation_results)

# # Compute Spearman Rank Correlation
# heuristic_scores = evaluation_df["Min-Max Score"]
# vrp_scores = evaluation_df["VRP Collaboration Saving Cost"]

# spearman_corr, p_value = spearmanr(heuristic_scores, vrp_scores)

# # Display results
# print("Evaluation Results (Top 3 and Bottom 3):")
# print(evaluation_df)
# print(f"\nSpearman Rank Correlation: {spearman_corr:.2f}")
# print(f"P-Value: {p_value:.2e}")

# # GENERATE THE K-MEAN RANKING
# rankingclusterkmeans = get_cluster_kmeans(df, max_clusters=10)

# # Select top 3 and bottom 3 from the K-Means ranking
# selected_pairs_kmeans = pd.concat([rankingclusterkmeans.head(3), rankingclusterkmeans.tail(3)])

# # Define parameters for VRP
# vehicle_capacity = 20  # Example vehicle capacity
# cost_per_km = 1        # Cost per kilometer
# fixed_cost_per_truck = 1  # Fixed cost for each truck

# # Evaluate VRP for selected pairs
# evaluation_results_kmeans = []
# for _, row in selected_pairs_kmeans.iterrows():
#     company_a = row["Company A"]
#     company_b = row["Company B"]
    
#     # Solve VRP for the selected companies using all_cvrp
#     vrp_result = all_cvrp(
#         vehicle_capacity=vehicle_capacity,
#         cost_per_km=cost_per_km,
#         fixed_cost_per_truck=fixed_cost_per_truck,
#         company_a=company_a,
#         company_b=company_b,
#         data=df,
#         dmatrix=dmatrix
#     )
    
#     # Append results
#     evaluation_results_kmeans.append({
#         "Company A": company_a,
#         "Company B": company_b,
#         "Heuristic Rank": row["Rank"],
#         "Shared Clusters": row["Shared_Clusters"],
#         "VRP Collaboration Saving Cost": vrp_result["Total Cost"][0] + vrp_result["Total Cost"][1] - vrp_result["Total Cost"][2]  # Cost for collaboration
#     })

# # Create DataFrame with results
# evaluation_df_kmeans = pd.DataFrame(evaluation_results_kmeans)

# # Compute Spearman Rank Correlation for K-Means rankings
# heuristic_scores_kmeans = evaluation_df_kmeans["Shared Clusters"]
# vrp_scores_kmeans = evaluation_df_kmeans["VRP Collaboration Saving Cost"]

# spearman_corr_kmeans, p_value_kmeans = spearmanr(heuristic_scores_kmeans, vrp_scores_kmeans)

# # Display results
# print("Evaluation Results (Top 3 and Bottom 3) for K-Means Ranking:")
# print(evaluation_df_kmeans)
# print(f"\nSpearman Rank Correlation for K-Means Ranking: {spearman_corr_kmeans:.2f}")
# print(f"P-Value: {p_value_kmeans:.2e}")


# # GENERATE THE DBSCAN RANKING
# dmatrix_without_depot = dmatrix.drop(index='Depot', columns='Depot')
# eps = 38
# min_samples = 2
# rankingclusterdbscan = get_dbscan_ranking(df, dmatrix_without_depot, eps, min_samples)

# # Select top 3 and bottom 3 from the K-Means ranking
# selected_pairs_dbcscan = pd.concat([rankingclusterdbscan.head(3), rankingclusterdbscan.tail(3)])

# # Define parameters for VRP
# vehicle_capacity = 20 # Example vehicle capacity
# cost_per_km = 1        # Cost per kilometer
# fixed_cost_per_truck = 1  # Fixed cost for each truck

# # Evaluate VRP for selected pairs
# evaluation_results_dbscan = []
# for _, row in selected_pairs_dbcscan.iterrows():
#     company_a = row["Company A"]
#     company_b = row["Company B"]
    
#     # Solve VRP for the selected companies using all_cvrp
#     vrp_result = all_cvrp(
#         vehicle_capacity=vehicle_capacity,
#         cost_per_km=cost_per_km,
#         fixed_cost_per_truck=fixed_cost_per_truck,
#         company_a=company_a,
#         company_b=company_b,
#         data=df,
#         dmatrix=dmatrix 
#     )
    
#     # Append results
#     evaluation_results_dbscan.append({
#         "Company A": company_a,
#         "Company B": company_b,
#         "Heuristic Rank": row["Rank"],
#         "Shared Clusters": row["Shared_Clusters"],
#         "VRP Collaboration Saving Cost": vrp_result["Total Cost"][0] + vrp_result["Total Cost"][1] - vrp_result["Total Cost"][2]  # Cost for collaboration
#     })

# # Create DataFrame with results
# evaluation_df_dbscan = pd.DataFrame(evaluation_results_dbscan)

# # Compute Spearman Rank Correlation for K-Means rankings
# heuristic_scores_dbscan = evaluation_df_dbscan["Shared Clusters"]
# vrp_scores_dbscan = evaluation_df_dbscan["VRP Collaboration Saving Cost"]

# spearman_corr_dbscan, p_value_dbscan = spearmanr(heuristic_scores_dbscan, vrp_scores_dbscan)

# # Display results
# print("Evaluation Results (Top 3 and Bottom 3) for DBSCAN Ranking:")
# print(evaluation_df_dbscan)
# print(f"\nSpearman Rank Correlation for DBSCAN Ranking: {spearman_corr_dbscan:.2f}")
# print(f"P-Value: {p_value_dbscan:.2e}")
