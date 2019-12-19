import numpy as np
import pandas as pd
from PSO_main import *


def centroid_dist(centroid, data):
    distance = 0
    size = len(centroid)
    for i in range(size):
        distance += (centroid[i] - data[i])**2
    return distance


data = pd.read_csv('breastcancer.txt', sep=',', header=None)
data = data.sample(frac=1).reset_index(drop=True)

cut = int(len(data) * .7)

y_data = data[0]
x_data = data.drop([0], axis=1)
x_data = x_data.values
x_data = normalize(x_data)

x_training = x_data[:cut]
x_test = x_data[cut:]

y_test = y_data[cut:]
y_test = y_test.values

clusters = 2

pso = PSO(n_cluster=clusters, n_particles=50, data=x_training, hybrid=True, max_iter=250, print_debug=50)
pso.run()

pso_kmeans = KMeans(n_cluster=clusters, init_pp=False)
pso_kmeans.centroid = pso.gbest_centroids.copy()

centroid_cls = [[0, 0] for i in range(clusters)]
for x, y in zip(x_test, y_test):
    least = 999999
    for i in range(clusters):
        ddd = centroid_dist(x, pso_kmeans.centroid[i])
        if ddd < least:
            index = i
            least = ddd
    if y == 'M':
        centroid_cls[index][0] += 1
    else:
        centroid_cls[index][1] += 1

# print(centroid_cls)


if centroid_cls[0][0] > centroid_cls[1][0]:
    TP, FP, FN, TN = centroid_cls[0][0], centroid_cls[0][1], centroid_cls[1][0], centroid_cls[1][1]
else:
    TP, FP, FN, TN = centroid_cls[1][0], centroid_cls[1][1], centroid_cls[0][0], centroid_cls[0][1]


size = TP + FP + FN + TN
accuracy = (TP + TN) / size
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F_measure = (2 * precision * recall) / (precision + recall)
print('Accuracy =', accuracy)
print('Precision =', precision)
print('Recall =', recall)
print('F_measure =', F_measure)
