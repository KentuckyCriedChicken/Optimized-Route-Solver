import os
import requests
import numpy as np
import pandas as pd
import time 


#API key THIS WONT BE INCLUDED IN OUR REPORT BTW
api_key = 'AIzaSyDKLLonkQTeWDLzOIJFOxbOwh6qpE-is_w'

#Load and sort the data
df = pd.read_csv('rawdata.csv')
drivers = df[df['role'] == 'Yes']
riders = df[df['role'] == 'No']

drivers_add = drivers[['address']]
riders_add = riders[['address']]
destination = 'Clovelly Beach, Sydney'

addresses = [
    "Glebe Point Rd & Parramatta Rd, Sydney",
    "Glebe Public School",
    "Glebe Public School",
    "Elizabeth St & Alt St, Ashfield",
    "The Nags Head Pub, Glebe",
    "Glebe Public School",
    "Glebe Point Road, Glebe",
    "249 Homer Lane, Earlwood",
    "7-Eleven, Croydon Park Avenue, Croydon Park",
    "Summer Hill Station, NSW",
    "Ritz Cinema, Randwick",
    "Barneys Broadway, Sydney",
    "Federal Park, Annandale",
    "The Footbridge, University of Sydney",
    "Glebe Library",
    "1 King St, Newtown",
    "35 Arundel St, Glebe",
    "20 Brighton Ave, Croydon Park",
    "Annandale Street, Annandale",
    "Glebe Point Rd, Glebe",
    "Clovelly Beach, Sydney"
]



# Create empty distance matrix
n = len(addresses)
distance_matrix = np.zeros((n, n))

# Helper: format list for URL
def format_addresses(addresses):
    return "|".join([address.replace(" ", "+") for address in addresses])

# Make requests in chunks
for i in range(n):
    origins = format_addresses([addresses[i]])
    for j in range(0, n, 10):  # Google allows up to 100 elements => ~10 destinations per request
        destinations = format_addresses(addresses[j:j+10])
        url = (
            f"https://maps.googleapis.com/maps/api/distancematrix/json?"
            f"origins={origins}&destinations={destinations}&key={api_key}"
        )
        response = requests.get(url).json()

        if response["status"] != "OK":
            print(f"Error: {response['error_message']}")
            continue

        for k, element in enumerate(response["rows"][0]["elements"]):
            if element["status"] == "OK":
                distance_matrix[i][j + k] = element["distance"]["value"]  # in meters
            else:
                distance_matrix[i][j + k] = np.nan  # or a large number

        time.sleep(1)  # Respect API usage limits

# Save to CSV if needed
import pandas as pd
df = pd.DataFrame(distance_matrix, columns=addresses, index=addresses)
df.to_csv("distance_matrix.csv")