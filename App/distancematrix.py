import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt

# cd app
# Lees het Excel-bestand in
from dss import load_data
df = load_data('../Data/mini.csv')

print(df.info())  # Inspect for NaN values
print(df.head())  # Check the first few rows

def haversine(lat1, lon1, lat2, lon2):
    print(lat1)
    R = 6371  # Straal van de aarde in kilometers
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c  # Afstand in kilometers

# Coördinaten van het depot
depot_lat = 52.16521
depot_lon = 5.17215

# Voeg het depot als eerste rij en kolom toe aan de dataframe
depot_row = pd.DataFrame({'name': ['Depot'], 'lat': [depot_lat], 'lon': [depot_lon]})
df = pd.concat([depot_row, df], ignore_index=True)

# Maak een lege lijst voor de volledige afstandsmatrix (inclusief depot)
full_distance_matrix = []

# Bereken de afstanden tussen het depot en alle klanten (inclusief depot zelf)
coordinates = df[['latitude', 'longitude']].values

# Bereken de afstand van het depot naar elke klant en tussen klanten onderling
for i in range(len(coordinates)):
    row = []
    for j in range(len(coordinates)):
        if i == j:
            row.append(0)  # Afstand van klant naar zichzelf is 0
        else:
            distance = haversine(coordinates[i][0], coordinates[i][1], coordinates[j][0], coordinates[j][1])
            row.append(distance)
    full_distance_matrix.append(row)

# Converteer de afstandsmatrix naar een Pandas DataFrame voor eenvoudiger manipulatie
customer_distance_df = pd.DataFrame(full_distance_matrix, columns=df['name'], index=df['name'])

# Afronden van de waarden in de DataFrame naar het dichtstbijzijnde gehele getal
customer_distance_df = customer_distance_df.round(1)

# Resultaten tonen
print(customer_distance_df)

# Plot de heatmap van de afstandsmatrix
plt.figure(figsize=(12, 10))  # Vergroot de figuur voor betere leesbaarheid
sns.heatmap(customer_distance_df, annot=True, cmap='YlGnBu', fmt='g', annot_kws={'size': 8})  # Verklein de annotatiegrootte

# Draai de x- en y-as labels voor betere leesbaarheid
plt.xticks(rotation=45, ha='right')  # Draai de x-as labels
plt.yticks(rotation=45, va='top')    # Draai de y-as labels

plt.title('Distance matrix of depot and customers')
plt.tight_layout()  # Zorgt ervoor dat alles netjes past
plt.show()