import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time

from dss import depot_lat, depot_lon, load_data
from distancematrix import OSRM

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

def fetch_osrm_distances(batch, ref_batch, osrm_url, max_retries=3):
    """Fetch distance matrix for a batch of coordinates using OSRM, with retries and a fallback to Haversine."""
    batch_coords = ';'.join(batch.apply(lambda row: f"{row['lon']},{row['lat']}", axis=1))
    ref_coords = ';'.join(ref_batch.apply(lambda row: f"{row['lon']},{row['lat']}", axis=1))
    # url = f"{osrm_url}/table/v1/driving/{batch_coords};{ref_coords}?annotations=distance"

    sources = ';'.join(str(i) for i in range(len(batch)))
    destinations = ';'.join(str(i + len(batch)) for i in range(len(ref_batch)))

    url = (
        f"{osrm_url}/table/v1/driving/{batch_coords};{ref_coords}?"
        f"annotations=distance&sources={sources}&destinations={destinations}"
    )
   # Print the URL for debugging
    print(f"Requesting OSRM distances for batches:\nBatch: {batch['name'].values}\nRef Batch: {ref_batch['name'].values}")
    print(f"Constructed URL: {url}\n")

    for attempt in range(max_retries):
        try:
            start_time = time.time()
            response = requests.get(url, timeout=30)

            # If successful, return immediately
            if response.status_code == 200:
                elapsed_time = time.time() - start_time
                #print(f"Batch processed in {elapsed_time:.2f} seconds, Response Code: {response.status_code}")
                
                return response.json().get('distances', None)

            # If rate limit (429), wait and retry
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 2))  # Retry-After header or default to 2s
                print(f"Rate limit hit. Retrying in {retry_after} seconds...")
                time.sleep(retry_after)

        except requests.exceptions.Timeout:
            print(f"Timeout on attempt {attempt + 1} for URL: {url}")
            time.sleep(2 ** attempt)  # Exponential backoff: 2s, 4s, 8s

        except requests.exceptions.RequestException as e:
            print(f"Request failed on attempt {attempt + 1}: {e}")
            time.sleep(2 ** attempt)  # Retry with backoff

    # Fallback to Haversine if all retries fail
    print(f"Batch failed after {max_retries} attempts. Falling back to Haversine for this batch.")
    return haversine_matrix(batch, ref_batch)

def OSRM_full_matrix_parallel(data_input, batch_size, max_workers=4, osrm_url='http://localhost:5000'):
    """Compute a full NxN distance matrix using OSRM with parallel requests."""

    # Ensure unique names to avoid indexing conflicts
    data_input['name'] = data_input['name'] + '_' + data_input.groupby('name').cumcount().astype(str)

    depot_row = pd.DataFrame({'name': ['Depot'], 'lat': [depot_lat], 'lon': [depot_lon]})
    data_input = pd.concat([depot_row, data_input], ignore_index=True)
    data_input = data_input.reset_index(drop=True)

    n = len(data_input) # = 21

    full_matrix = pd.DataFrame(
    np.zeros((n, n)), 
    index=data_input['name'], 
    columns=data_input['name']
    )

    batches = [data_input.iloc[i:i + batch_size] for i in range(0, n, batch_size)]
    # We have 3 batches for mini dataset

    # Track mapping of batches to global indices
    row_indices = [batch.index.tolist() for batch in batches]
    #print(row_indices)
    col_indices = row_indices  # Since it’s a square distance matrix, we reuse the same indices

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            (i, j): executor.submit(fetch_osrm_distances, batch, ref_batch, osrm_url)
            for i, batch in enumerate(batches) for j, ref_batch in enumerate(batches)
        }

    for (i, j), future in futures.items():
        distances = future.result()

        if (i, j) == (0, 0):
            print("Batch (0, 0) - Full JSON Response:")
            #print(future.result())  # Print full JSON response
            print("distances[0] for (0, 0):", distances[0])

        if (i, j) == (0, 1):
            print("Batch (0, 1) - Full JSON Response:")
            #print(future.result())  # Print full JSON response
            print("distances[0] for (0, 1):", distances[0])

    # for i, batch in enumerate(batches):
    #     for j, ref_batch in enumerate(batches):
    #         print(f"Processing batch ({i}, {j})")
    #         print(f"Batch {i}: {batch['name'].values}")
    #         print(f"Batch {j}: {ref_batch['name'].values}")
    
    # Example logic where each batch maps correctly to the matrix
    for (i, j), future in futures.items():
        print(i,j)
        distances = future.result()
        if (i,j) == (0,0):
            print(distances[0])
        if (i,j) == (0,1):
            print(distances[0])
        if distances:
            for k in range(len(batches[i])):
                for l in range(len(batches[j])):
                    # Correctly identify labels
                    
                    row_label = batches[i].iloc[k]['name']
                    col_label = batches[j].iloc[l]['name']
                    if (i,j) == (0.0):
                        print(row_label)
                        print(col_label)
                    
                    # Avoid incorrect overwriting by checking batch offsets
                    if pd.isna(full_matrix.loc[row_label, col_label]) or full_matrix.loc[row_label, col_label] == 0:
                        full_matrix.loc[row_label, col_label] = distances[k][l]
                    else:
                        print(f"Overwriting at [{row_label}, {col_label}]! Existing: {full_matrix.loc[row_label, col_label]}, New: {distances[k][l]}")


    # Assuming full_matrix is a DataFrame and we want the first row (index 0)
    print("First row, columns 0 to 9:")
    print(full_matrix.iloc[0, 0:10])

    print("\nFirst row, columns 10 to 20:")
    print(full_matrix.iloc[0, 11:21])


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
                dmatrix = OSRM_full_matrix_parallel(df, batch_size=10, max_workers=4, osrm_url=osrm_url)
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
    df = load_data('../Data/mini.csv')  # Load your dataset
    dmatrix = compute_distance_matrix(df)

    if dmatrix is not None and not dmatrix.empty:
        print("Distance matrix successfully calculated!")
        print(dmatrix)

        #plot_heat_dist(dmatrix)
    else:
        print("Failed to compute the distance matrix.")

    # Assuming you have the matrices as DataFrames or arrays
    parallel_matrix = dmatrix # Your parallel computation result
    correct_matrix = dmatrixPOLD   # The correct reference matrix

    # **Step 1: Element-wise difference**
    difference_matrix = parallel_matrix - correct_matrix

    # **Step 2: Identify locations with mismatches**
    mismatches = np.where(difference_matrix != 0)

    # Print mismatch summary
    if len(mismatches[0]) == 0:
        print("No mismatches found! The parallel computation is correct.")
    else:
        print(f"Found {len(mismatches[0])} mismatches.")
        for i in range(len(mismatches[0])):
            row, col = mismatches[0][i], mismatches[1][i]
            
            # Get labels corresponding to row and col indices
            row_label = parallel_matrix.index[row]
            col_label = parallel_matrix.columns[col]
            
            # print(f"Mismatch at (row={row_label}, col={col_label}): "
            #     f"Parallel={parallel_matrix.loc[row_label, col_label]}, "
            #         f"Correct={correct_matrix.loc[row_label, col_label]}")
            
    # # **Optional Step 3: Batch-level comparison**
    # def check_batch(parallel_matrix, correct_matrix, batch_size):
    #     total_rows = parallel_matrix.shape[0]
        
    #     for i in range(0, total_rows, batch_size):
    #         # Get slices using DataFrame slicing (not NumPy slicing)
    #         batch_parallel = parallel_matrix.iloc[i:i+batch_size, :]
    #         batch_correct = correct_matrix.iloc[i:i+batch_size, :]

    #         # Compare batches
    #         if not np.allclose(batch_parallel.values, batch_correct.values):
    #             print(f"Mismatch detected in batch starting at row {i}.")
                
    #             # Identify mismatching elements within this batch
    #             batch_mismatches = np.where(batch_parallel != batch_correct)
    #             for j in range(len(batch_mismatches[0])):
    #                 row_offset, col = batch_mismatches[0][j], batch_mismatches[1][j]
                    
    #                 row_label = batch_parallel.index[row_offset]
    #                 col_label = batch_parallel.columns[col]

    #                 print(f"    Within batch at row={row_label}, col={col_label}: "
    #                     f"Parallel={batch_parallel.loc[row_label, col_label]}, "
    #                     f"Correct={batch_correct.loc[row_label, col_label]}")

    # # **Step 4: Run batch-level check**
    # batch_size = 10  # Adjust based on how batching is done
    # check_batch(parallel_matrix, correct_matrix, batch_size)