import numpy as np
from sklearn.cluster import KMeans
from Gaussian import Gaussian


class GMM:
    def __init__(self, K):
        self.K = K
        self.gaussians = [Gaussian() for _ in range(self.K)]
        self.weights = np.array([1.0 / K] * K)

    def initialize_gmm(self, X):
        clusterer = KMeans(n_clusters=self.K, max_iter=10, random_state=None)
        clusters = clusterer.fit_predict(X)

        num_pixels = float(X.shape[0])

        for i, gaussian in enumerate(self.gaussians):
            gaussian.update_parameters(X[clusters == i])
            self.weights[i] = np.sum(clusters == i) / num_pixels

        return clusters

    def get_component(self, x):
        components = np.zeros((x.shape[0], len(self.gaussians)))

        for i, g in enumerate(self.gaussians):
            components[:, i] = self.weights[i] * g.compute_probability(x)

        return np.argmax(components, axis=1)

    def compute_probability(self, x):
        return np.dot(self.weights, [g.compute_probability(x) for g in self.gaussians])

    def update_components(self, X, assignments):
        num_pixels = float(np.sum(assignments != -1))

        for i, distribution in enumerate(self.gaussians):
            if X[assignments == i].shape[0] != 0:
                distribution.update_parameters(X[assignments == i])
                self.weights[i] = (np.sum(assignments == i) / num_pixels)
            else:
                distribution.mean = [-1e9, -1e9, -1e9]
                self.weights[i] = 0

    def update_gmm(self, X, means, sigmas, weight):
        num_pixels = X.shape[0]

        for i, gaussian in enumerate(self.gaussians):
            gaussian.set_parameter(means[i], sigmas[i])

#         temp1 = np.zeros((num_pixels,5))
#         temp2 = np.zeros((num_pixels))
#         temp_weight = np.zeros(5)
#         for i in range(5):
#             temp1[:,i] = FG_GMM.gaussians[i].compute_probability(X)
#         for i in range(num_pixels):
#             temp2[i] = np.argmax(temp1[i])
#         for i in range(5):
#             temp_weight[i] = np.sum(temp2==i) / num_pixels

        self.weights = weight
