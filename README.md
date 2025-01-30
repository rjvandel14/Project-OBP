# Logistics Collaboration Dashboard

## Overview

This project is a logistics collaboration dashboard designed to evaluate potential partnerships between logistics companies. It leverages data-driven approaches to compute distance matrices, analyze collaboration opportunities, and provide visualizations for cost and resource savings.

The application supports:

- Computing distance matrices using the public OSRM API with fallback to Haversine calculations.
- Ranking potential partnerships based on cost savings and other metrics.
- Detailed analysis of selected partnerships.

---

## Prerequisites

- **Python 3.8 to 3.12**
- **Docker Desktop installed on your system.**

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/rjvandel14/Project-OBP.git
cd <repository-directory>
```

### 2. Install Python Dependencies

Use the `requirements.txt` file to install required Python packages:

```bash
pip install -r requirements.txt
```

### 3. OSRM Setup

#### Using Preprocessed Files 

1. Visit the [Geofabrik website]([https://download.geofabrik.de/]) https://download.geofabrik.de/europe/netherlands.html and download the `.osm.pbf` file.

2. Preprocess the map using OSRM tools:

   ```bash
   osrm-extract -p /path/to/profiles/car.lua /path/to/map.osm.pbf
   osrm-partition /path/to/map.osrm
   osrm-customize /path/to/map.osrm
   ```

3. Place all these files in the data/ directory inside the app folder.

---

## Running the Application

### 1. Start the OSRM Server

Use Docker Compose to build and run the OSRM server:

```bash
docker-compose up --build
```

This will start the OSRM server with the preprocessed files in `osrm_data/`.

### 2. Run the Streamlit Dashboard

Navigate to the app directory and start the dashboard:

```bash
cd app
streamlit run gui.py
```

---

## File Structure

- `docker-compose.yml`: Configures and runs the OSRM server container.
- `gui.py`: Main Streamlit dashboard interface.
- `osrm_dmatrix.py`: Handles OSRM and Haversine distance calculations.
- `ranking.py`: Implements ranking and scoring logic.
- `routing.py`: Solves VRPs and provides collaborative route analysis.
- `gui_sidebar.py`: Manages user inputs in the sidebar.
- `gui_ranking.py`: Displays ranked partnerships.
- `gui_analysis.py`: Analyzes selected collaborations.

---

## Notes for OSRM

- If the OSRM server fails or is unreachable, the system falls back to Haversine calculations to ensure the distance matrix is always computed.

---

## Troubleshooting

### Common Issues

1. **OSRM Fails to Start**:

   - Ensure the `.osrm` files are correctly placed in `osrm_data/`.
   - Check that Docker is running.

2. **Rate Limits with Public OSRM**:

   - Use the local OSRM instance provided via Docker Compose.

3. **Dataset Errors**:

   - Ensure uploaded datasets contain the required columns: `name`, `lat`, `lon`.

---

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Submit a pull request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

