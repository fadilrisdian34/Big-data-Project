import numpy as np


def calc_sse(centroids: np.ndarray, labels: np.ndarray, data: np.ndarray):
    distances = 0
    for i, c in enumerate(centroids):
        idx = np.where(labels == i)
        dist = np.sum((data[idx] - c)**2)
        distances += dist
    return distances


class KMeans:
    """K-Means clustering algorithm

        Attributes
        ----------
        n_cluster : int
            Num of cluster applied to data
        init_pp : bool
            Initialization method whether to use K-Means++ or not
            (the default is True, which use K-Means++)
        max_iter : int
            Max iteration to update centroid (the default is 300)
        tolerance : float
            Minimum centroid update difference value to stop iteration (the default is 1e-4)
        seed : int
            Seed number to use in random generator (the default is None)
        centroid : list
            List of centroid values
        SSE : float
            Sum squared error score
    """

    def __init__(self, n_cluster, init_pp=True, max_iter=100, tolerance=1e-4, seed: int = None):

        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.init_pp = init_pp
        self.seed = seed
        self.centroid = None
        self.SSE = None

    def fit(self, data: np.ndarray):
        """Fit K-Means algorithm to given data

        Parameters
        ----------
        data : np.ndarray
            Data matrix to be fitted

        """
        self.centroid = self._init_centroid(data)
        for _ in range(self.max_iter):
            distance = self._calc_distance(data)
            cluster = self._assign_cluster(distance)
            new_centroid = self._update_centroid(data, cluster)
            diff = np.abs(self.centroid - new_centroid).mean()
            self.centroid = new_centroid

            if diff <= self.tolerance:
                break

        self.SSE = calc_sse(self.centroid, cluster, data)

    def predict(self, data: np.ndarray):
        """Predict new data's cluster using minimum distance to centroid

        Parameters
        ----------
        data : np.ndarray
            New data to be predicted

        """
        distance = self._calc_distance(data)
        # print(distance.shape)
        cluster = self._assign_cluster(distance)
        # print(cluster.shape)
        return cluster

    def _init_centroid(self, data: np.ndarray):
        """Initialize centroid using random method or KMeans++

        Parameters
        ----------
        data : np.ndarray
            Data matrix to sample from

        """
        if self.init_pp:
            np.random.seed(self.seed)
            centroid = [int(np.random.uniform() * len(data))]
            for _ in range(1, self.n_cluster):
                dist = []
                dist = [min([np.inner(data[c] - x, data[c] - x) for c in centroid])
                        for i, x in enumerate(data)]
                dist = np.array(dist)
                dist = dist / dist.sum()
                cumdist = np.cumsum(dist)

                prob = np.random.rand()
                for i, c in enumerate(cumdist):
                    if prob > c and i not in centroid:
                        centroid.append(i)
                        break
            centroid = np.array([data[c] for c in centroid])
        else:
            np.random.seed(self.seed)
            idx = np.random.choice(range(len(data)), size=(self.n_cluster))
            centroid = data[idx]
        # print(centroid)
        return centroid

    def _calc_distance(self, data: np.ndarray):
        """Calculate distance between data and centroids

        Parameters
        ----------
        data : np.ndarray
            Data which distance to be calculated

        """
        distances = []
        for c in self.centroid:
            distance = np.sum((data - c) * (data - c), axis=1)
            distances.append(distance)

        distances = np.array(distances)
        distances = distances.T
        return distances

    def _assign_cluster(self, distance: np.ndarray):
        """Assign cluster to data based on minimum distance to centroids

        Parameters
        ----------
        distance : np.ndarray
            Distance from each data to each centroid

        """
        cluster = np.argmin(distance, axis=1)
        return cluster

    def _update_centroid(self, data: np.ndarray, cluster: np.ndarray):
        """Update centroid from means of each cluster's data

        Parameters
        ----------
        data : np.ndarray
            Data matrix to get mean from
        cluster : np.ndarray
            Cluster label for each data

        """
        centroids = []
        for i in range(self.n_cluster):
            idx = np.where(cluster == i)
            centroid = np.mean(data[idx], axis=0)
            centroids.append(centroid)
        centroids = np.array(centroids)
        return centroids


if __name__ == "__main__":
    pass
