import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time

from dss import depot_lat, depot_lon, load_data

# ----------------- Helper Functions -----------------

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the haversine distance in kilometers between two coordinates."""
    R = 6371  # Earth radius in kilometers
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon1 - lon2)

    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c  # Distance in kilometers

def haversine_matrix(df):
    """Calculate a distance matrix using the Haversine formula."""
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

# ----------------- OSRM Functions -----------------

def fetch_osrm_distances(batch, ref_batch, osrm_url):
    """Fetch distance matrix for a batch of coordinates using OSRM."""
    try:
        batch_coords = ';'.join(batch.apply(lambda row: f"{row['lon']},{row['lat']}", axis=1))
        ref_coords = ';'.join(ref_batch.apply(lambda row: f"{row['lon']},{row['lat']}", axis=1))
        url = f"{osrm_url}/table/v1/driving/{batch_coords};{ref_coords}?annotations=distance"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get('distances', None)
    except:
        return None

def OSRM_full_matrix_parallel(data_input, batch_size=50, max_workers=4, osrm_url='http://localhost:5000'):
    """Compute a full NxN distance matrix using OSRM with parallel requests."""
    depot_row = pd.DataFrame({'name': ['Depot'], 'lat': [depot_lat], 'lon': [depot_lon]})
    data_input = pd.concat([depot_row, data_input], ignore_index=True)
    data_input = data_input.reset_index(drop=True)

    n = len(data_input) 
    full_matrix = np.zeros((n, n))
    batches = [data_input.iloc[i:i + batch_size] for i in range(0, n, batch_size)]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            (i, j): executor.submit(fetch_osrm_distances, batch, ref_batch, osrm_url)
            for i, batch in enumerate(batches) for j, ref_batch in enumerate(batches)
        }
        for (i, j), future in futures.items():
            distances = future.result()
            if distances:
                for k, row_index in enumerate(batches[i].index):
                    for l, col_index in enumerate(batches[j].index):
                        full_matrix[row_index, col_index] = distances[k][l]

    return pd.DataFrame(full_matrix / 1000, index=data_input['name'], columns=data_input['name']).round(1)

# ----------------- Fallback Mechanism -----------------

def is_osrm_url_reachable(osrm_url, is_local=True):
    """Check if the OSRM URL is reachable."""
    try:
        # Use a lightweight /table request to check both local and public OSRM
        test_url = f"{osrm_url}/table/v1/driving/13.388860,52.517037;13.397634,52.529407?annotations=distance"
        
        response = requests.get(test_url, timeout=10)
        return response.status_code == 200
    except Exception:
        return False

def compute_distance_matrix(df):
    """Compute the distance matrix using OSRM with fallback to Haversine."""
    osrm_urls = [
        {"url": "http://localhost:5000", "is_local": True},  # Local OSRM via Docker
        {"url": "http://router.project-osrm.org", "is_local": False}  # Public OSRM API
    ]

    for osrm in osrm_urls:
        osrm_url, is_local = osrm["url"], osrm["is_local"]
        print(f"Checking if OSRM at {osrm_url} is reachable...")
        if is_osrm_url_reachable(osrm_url, is_local=is_local):
            print(f"Attempting OSRM at {osrm_url}...")
            try:
                start_time = time.time()  # Start the timer
                dmatrix = OSRM_full_matrix_parallel(df, batch_size=50, max_workers=4, osrm_url=osrm_url)
                end_time = time.time()  # Stop the timer
                if not dmatrix.empty:
                    elapsed_time = end_time - start_time  # Calculate elapsed time
                    print(f"Successfully computed distance matrix using OSRM at {osrm_url}.")
                    print(f"Time taken: {elapsed_time:.2f} seconds.")
                    return dmatrix
            except Exception as e:
                print(f"Failed to compute distance matrix using OSRM at {osrm_url}: {e}")
        else:
            print(f"OSRM at {osrm_url} is not reachable. Trying the next option...")

    print("Falling back to Haversine distance matrix...")
    return haversine_matrix(df)


# ----------------- Plotting Function -----------------

def plot_heat_dist(matrix):
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, annot=True, cmap='YlGnBu', fmt='g', annot_kws={'size': 8})
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45, va='top')
    plt.title('Distance matrix of depot and customers')
    plt.tight_layout()
    plt.show()

# ----------------- Main Execution -----------------

if __name__ == "__main__":
    df = load_data('../Data/medium.csv')  # Load your dataset
    dmatrix = compute_distance_matrix(df)

    if dmatrix is not None and not dmatrix.empty:
        print("Distance matrix successfully calculated!")
        print(dmatrix)
        #plot_heat_dist(dmatrix)
    else:
        print("Failed to compute the distance matrix.")
