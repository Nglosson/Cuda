# Cuda
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from time import time
url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data")
abalone = pd.read_csv(url, header=None)
abalone.head()
abalone.columns = ["Sex","Length","Diameter","Height","Whole weight","Shucked weight","Viscera weight","Shell weight","Rings",]
abalone = abalone.drop("Sex", axis=1)
abalone["Rings"].hist(bins=15)
plt.show()
correlation_matrix = abalone.corr()
print(correlation_matrix["Rings"])
X = abalone.drop("Rings", axis=1)
X = X.values
y = abalone["Rings"]
y = y.values
new_data_point =
np.array([0.569552,0.446407,0.154437,1.016849,0.439051,0.222526,0.291208,])
t1 = time()
#This next line should be parallelized in CUDA
distances = np.linalg.norm(X - new_data_point, axis=1)
k = 3
nearest_neighbor_ids = distances.argsort()[:k]
print(nearest_neighbor_ids)
nearest_neighbor_rings = y[nearest_neighbor_ids]
print(nearest_neighbor_rings)
prediction = nearest_neighbor_rings.mean()
print(prediction)
t2 = time()-t1
print(t2)
