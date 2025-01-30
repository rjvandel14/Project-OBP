import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time

from dss import depot_lat
from dss import depot_lon
from dss import load_data

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in kilometers
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c  # Distance in kilometers

def haversine_matrix(df):
    depot_row = pd.DataFrame({'name': ['Depot'], 'lat': [depot_lat], 'lon': [depot_lon]})
    df = pd.concat([depot_row, df], ignore_index=True)

    coordinates = df[['lat', 'lon']].values
    distances = pd.DataFrame(
        [[haversine(*coordinates[i], *coordinates[j]) for j in range(len(coordinates))] 
         for i in range(len(coordinates))],
        index=df['name'],
        columns=df['name']
    )

    return distances.round(1)

def OSRM(data_input):
    depot_row = pd.DataFrame({'name': ['Depot'], 'lat': [depot_lat], 'lon': [depot_lon]})
    df = pd.concat([depot_row, data_input], ignore_index=True)

    #osrm_url = 'http://router.project-osrm.org/table/v1/driving/'
    osrm_url = 'http://localhost:5000/table/v1/driving/'
    coordinates = ';'.join(df.apply(lambda row: f"{row['lon']},{row['lat']}", axis=1))

    response = requests.get(f"{osrm_url}{coordinates}?annotations=distance")

    if response.status_code == 200:
        try:
            data = response.json()
            if 'distances' in data:
                distance_matrix = pd.DataFrame(
                    data['distances'], index=df['name'], columns=df['name']
                )
                distance_matrix = distance_matrix / 1000
                return distance_matrix.round(1)
            else:
                print("Error: 'distances' field missing in OSRM response.")
        except ValueError:
            print("Error: Invalid JSON received from OSRM server.")
    else:
        print(f"Error: OSRM server responded with status code {response.status_code}")
        print(response.text)

    return None

def compute_distance_matrix(df):
    print("Attempting to compute distance matrix using OSRM...")
    dmatrix = OSRM(df)

    if dmatrix is not None:
        print("Successfully computed distance matrix using OSRM.")
    else:
        print("OSRM failed. Falling back to Haversine distance matrix...")
        dmatrix = haversine_matrix(df)
        print("Successfully computed distance matrix using Haversine.")

    return dmatrix

def plot_heat_dist(matrix):
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, annot=True, cmap='YlGnBu', fmt='g', annot_kws={'size': 8})
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45, va='top')
    plt.title('Distance matrix of depot and customers')
    plt.tight_layout()
    plt.show()

def fetch_osrm_distances(batch, ref_batch):
    """Fetch distance matrix for a batch vs. a reference batch."""
    batch_coords = ';'.join(batch.apply(lambda row: f"{row['lon']},{row['lat']}", axis=1))
    ref_coords = ';'.join(ref_batch.apply(lambda row: f"{row['lon']},{row['lat']}", axis=1))
    
    osrm_url = f'http://localhost:5000/table/v1/driving/{batch_coords};{ref_coords}?annotations=distance'
    response = requests.get(osrm_url)
    
    if response.status_code == 200:
        data = response.json()
        if 'distances' in data:
            return data['distances']
        else:
            raise ValueError("Missing 'distances' field in OSRM response.")
    else:
        raise ConnectionError(f"OSRM server error: {response.status_code}, {response.text}")

def OSRM_full_matrix_parallel(data_input, batch_size=50, max_workers=4):
    """Calculate a full NxN distance matrix using parallel requests."""
    depot_row = pd.DataFrame({'name': ['Depot'], 'lat': [depot_lat], 'lon': [depot_lon]})
    data_input = pd.concat([depot_row, data_input], ignore_index=True)
    # Ensure indices are reset for consistency
    data_input = data_input.reset_index(drop=True)

    # Create empty NxN matrix
    n = len(data_input)
    full_matrix = np.zeros((n, n))

    # Create batches
    batches = [data_input.iloc[i:i + batch_size] for i in range(0, n, batch_size)]

    # Process batches in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        
        for i, batch in enumerate(batches):
            for j, ref_batch in enumerate(batches):
                # Submit the task for parallel execution
                future = executor.submit(fetch_osrm_distances, batch, ref_batch)
                futures[(i, j)] = (future, batch.index, ref_batch.index)
        
        # Collect results
        for (i, j), (future, batch_indices, ref_indices) in futures.items():
            try:
                distances = future.result()  # Get result from the future
                for k, row_index in enumerate(batch_indices):
                    for l, col_index in enumerate(ref_indices):
                        full_matrix[row_index, col_index] = distances[k][l]
            except Exception as e:
                print(f"Error processing batch ({i}, {j}): {e}")
                continue

    # Convert to kilometers and return as a DataFrame
    full_matrix = full_matrix / 1000  # Convert meters to kilometers
    return pd.DataFrame(full_matrix, index=data_input['name'], columns=data_input['name'])


 # Load data and compute distance matrix
# df = load_data('../Data/medium.csv')
# #dmatrix = compute_distance_matrix(df)
# dmatrix = OSRM(df)
# dmatrixParallel = OSRM_full_matrix_parallel(df)

# print(dmatrix)
# print(dmatrixParallel)
# print(dmatrixParallel - dmatrix)

# if dmatrix is not None and not dmatrix.empty:
#     print(dmatrix)
#     plot_heat_dist(dmatrix)
# else:
#     print("Failed to compute a valid distance matrix.")



