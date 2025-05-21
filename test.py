import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#load the data
df = pd.read_csv('testdata.csv')

destination = np.array([-33.894, 151.212])


#sorting the data
drivers = df[df['role'] == 'driver']
riders = df[df['role'] == 'driver']

driver_coord = drivers[['x', 'y']].to_numpy
rider_coord = riders[['x', 'y']].to_numpy





plt.scatter(df['x'], df['y'])
plt.scatter(destination[0], destination[1], c = 'red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('map_v1')
plt.show()

