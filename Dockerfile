# Gebruik een Python-image
FROM python:3.10-slim

# Installeer OSRM-backend en dependencies
RUN apt-get update && apt-get install -y \
    osrm-backend \
    osrm-tools \
    && rm -rf /var/lib/apt/lists/*

# Stel de werkdirectory in
WORKDIR /app

# Kopieer je bestanden
COPY ./App /app/App
COPY ./data /app/data
COPY requirements.txt /app/

# Installeer Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Verwerk het OSM-bestand
RUN osrm-extract -p /opt/car.lua /app/data/netherlands-latest.osm.pbf
RUN osrm-partition /app/data/netherlands-latest.osrm
RUN osrm-customize /app/data/netherlands-latest.osrm

# Start de Python-applicatie
CMD ["python", "/app/App/distance_matrix.py"]
