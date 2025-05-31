import os
import requests
import numpy as np
import pandas as pd
import time 


# API key (The actual key has been removed due to security reasons)
api_key = 'API_KEY'

# Load the CSV file
df = pd.read_csv("rawdata.csv")  

# Extract the coordinates
coordinates = list(zip(df["Latitude"], df["Longitude"]))
coordinates.append((-33.915200,151.267898))


n = len(coordinates)
distance_matrix = np.zeros((n, n))

# Convert lat/long to strings
def format_coords(coords):
    return "|".join([f"{lat},{lon}" for lat, lon in coords])

# Loop over each origin
for i in range(n):
    origin = [coordinates[i]]
    for j in range(0, n, 10):
        dest_chunk = coordinates[j:j + 10]

        url = (
            "https://maps.googleapis.com/maps/api/distancematrix/json"
            f"?origins={format_coords(origin)}"
            f"&destinations={format_coords(dest_chunk)}"
            f"&key={api_key}"
        )

        response = requests.get(url).json()

        if response["status"] != "OK":
            print(f"API error: {response.get('error_message', 'Unknown error')}")
            continue

        elements = response["rows"][0]["elements"]
        for k, element in enumerate(elements):
            idx = j + k
            if element["status"] == "OK":
                distance_matrix[i][idx] = element["distance"]["value"]
            else:
                print(f"Failed: {coordinates[i]} to {coordinates[idx]} - {element['status']}")
                distance_matrix[i][idx] = np.nan

        time.sleep(1.5) 

# Save to .npy file
np.save("distance_matrix.npy", distance_matrix)


