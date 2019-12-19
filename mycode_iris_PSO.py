import numpy as np
import pandas as pd
from PSO_main import *


def centroid_dist(centroid, data):
    distance = 0
    size = len(centroid)
    for i in range(size):
        distance += (centroid[i] - data[i])**2
    return distance


data = pd.read_csv('iris.txt', sep=',', header=None)
data = data.sample(frac=1).reset_index(drop=True)

cut = int(len(data) * .65)

y_data = data[4]
x_data = data.drop([4], axis=1)
x_data = x_data.values
x_data = normalize(x_data)

x_training = x_data[:cut]
x_test = x_data[cut:]

y_test = y_data[cut:]
y_test = y_test.values

clusters = 3

pso = PSO(n_cluster=clusters, n_particles=50, data=x_training, hybrid=True, max_iter=250, print_debug=50)
pso.run()

pso_kmeans = KMeans(n_cluster=clusters, init_pp=False)
pso_kmeans.centroid = pso.gbest_centroids.copy()

centroid_cls = [[0, 0, 0] for i in range(clusters)]
for x, y in zip(x_test, y_test):
    least = 999999
    for i in range(clusters):
        ddd = centroid_dist(x, pso_kmeans.centroid[i])
        if ddd < least:
            index = i
            least = ddd
    if y == 'Iris-setosa':
        centroid_cls[index][0] += 1
    elif y == 'Iris-versicolor':
        centroid_cls[index][1] += 1
    elif y == 'Iris-virginica':
        centroid_cls[index][2] += 1


if centroid_cls[0][0] < centroid_cls[1][0]:
    centroid_cls[0], centroid_cls[1] = centroid_cls[1], centroid_cls[0]
if centroid_cls[0][0] < centroid_cls[2][0]:
    centroid_cls[0], centroid_cls[2] = centroid_cls[2], centroid_cls[0]
if centroid_cls[0][0] < centroid_cls[1][0]:
    centroid_cls[0], centroid_cls[1] = centroid_cls[1], centroid_cls[0]
if centroid_cls[1][0] < centroid_cls[2][0]:
    centroid_cls[1], centroid_cls[2] = centroid_cls[2], centroid_cls[1]

acc = 0
total = 0
f1 = []
for i in range(3):
    acc += centroid_cls[i][i]
    total += sum(centroid_cls[i])
    TP = centroid_cls[i][i]
    precision = TP / (sum(centroid_cls[i]))
    recall = TP / (sum([centroid_cls[j][i] for j in range(3)]))
    f1.append((2 * precision * recall) / (precision + recall))

print('Accuracy =', acc / total)
print('F_measure =', sum(f1) / 3)
