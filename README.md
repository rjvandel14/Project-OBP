# Logistics Collaboration Dashboard

## Overview

This project is a logistics collaboration dashboard designed to evaluate potential partnerships between logistics companies. It leverages data-driven approaches to compute distance matrices, analyze collaboration opportunities, and provide visualizations for cost and resource savings.

---

## Prerequisites

- **Python 3.8 to 3.12**
- **Docker Desktop installed on your laptop.**

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

#### Using Preprocessed Files (Recommended)

1. Download the preprocessed `osrm_files.zip` provided via email.
2. Unzip the file into the `data/` directory inside the project folder:

   ```bash
   unzip osrm_files.zip -d data/
   ```

---

## Running the Application

### 1. Start the OSRM Server

1. Open Docker Desktop
2. Use Docker Compose to build and run the OSRM server:

```bash
docker-compose up --build
```

This will start the OSRM server with the preprocessed files in `data/`.

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

---

## Notes for OSRM

- If the OSRM server fails or is unreachable, the system falls back to Haversine calculations to ensure the distance matrix is always computed.

---

## Troubleshooting

### Common Issues

1. **OSRM Fails to Start**:

   - Ensure the `.osrm` files are correctly placed in `data/`.
   - Check that Docker is running.

2. **Rate Limits with Public OSRM**:

   - Use the local OSRM instance provided via Docker Compose.

3. **Dataset Errors**:

   - Ensure uploaded datasets contain the required columns: `name`, `lat`, `lon`.

---
