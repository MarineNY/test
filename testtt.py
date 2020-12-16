from sklearn.neighbors import NearestNeighbors
import numpy as np


points = [[1, 5],[30, 56],[40, 54],[2, 6],[47,78],[6,3],[10,34]]

nbrs = NearestNeighbors(n_neighbors=2, metric='minkowski', p=2).fit(points)

distances, indices = nbrs.kneighbors(points)

result = distances[:, 1]

print(result)
print(indices)