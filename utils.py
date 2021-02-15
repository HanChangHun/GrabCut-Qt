import cv2
import numpy as np
import maxflow
from sklearn.cluster import KMeans
from scipy.spatial.distance import mahalanobis


class Gaussian:
    def __init__(self, mean=np.zeros((3, 1)), sigma=np.eye(3)):
        self.mean = np.array(mean)
        self.sigma = np.array(sigma) + np.eye(3) * 1e-7
        self.inv_sig = np.linalg.inv(self.sigma)
        self.factor = (2.0 * np.pi)**(3 / 2.0) * (np.fabs(np.linalg.det(self.sigma)))**(0.5)

    def compute_probability(self, x):
        dx = x - self.mean
        return np.exp(-0.5 * np.dot(np.dot(dx, self.inv_sig), dx)) / self.factor

    def update_parameters(self, data):
        self.mean = np.mean(data, axis=0)
        self.sigma = np.cov(data, rowvar=False) + np.eye(3) * 1e-7

    def set_parameter(self, mean, sigma):
        self.mean = np.array(mean)
        self.sigma = np.array(sigma) + np.eye(3) * 1e-7


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
        components = np.zeros((x.shape[0], self.K))

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

        self.weights = weight

def data_term(k, gmm, pixels):
    pi_base = gmm.weights
    pi = pi_base[k].reshape(pixels.shape[0])

    dets_base = np.array([np.linalg.det(gmm.gaussians[i].sigma) for i in range(5)])
    dets = dets_base[k].reshape(pixels.shape[0])

    means_base = np.array([gmm.gaussians[i].mean for i in range(5)])
    means = means_base[k]

    inv_cov_base = np.array([np.linalg.inv(gmm.gaussians[i].sigma) for i in range(5)])
    inv_cov = inv_cov_base[k]

    distances = []
    residual = pixels - means
    for i in range(residual.shape[0]):
        distance = mahalanobis(residual[i], [0, 0, 0], inv_cov[i])
        distances.append(distance)

    return -np.log(pi + 1e-7) + 0.5 * np.log(dets + 1e-7) + 0.5 * np.array(distances)


def compute_beta(img):
    beta = 0
    img = np.array(img, dtype=np.float32)

    e_diff = img - np.roll(img, 1, axis=0)
    temp = np.sum(np.multiply(e_diff, e_diff), axis=2)
    beta = np.sum(temp[1:, :])

    s_diff = img - np.roll(img, 1, axis=1)
    temp = np.sum(np.multiply(s_diff, s_diff), axis=2)
    beta += np.sum(temp[:, 1:])

    se_diff = img - np.roll(np.roll(img, 1, axis=0), 1, axis=1)
    temp = np.sum(np.multiply(se_diff, se_diff), axis=2)
    beta += np.sum(temp[1:, 1:])

    sw_diff = img - np.roll(np.roll(img, 1, axis=0), -1, axis=1)
    temp = np.sum(np.multiply(sw_diff, sw_diff), axis=2)
    beta += np.sum(temp[1:, :img.shape[-1] - 1])

    num_pixel = img.shape[0] * img.shape[1] - 3 * (img.shape[0] + img.shape[1])

    beta = 1.0 / (2 * (beta / num_pixel))

    return beta


def smoothness_term(img):
    beta = compute_beta(img)
    energies = []

    n = img - np.roll(img, 1, axis=0)
    s = img - np.roll(img, -1, axis=0)
    e = img - np.roll(img, 1, axis=1)
    w = img - np.roll(img, -1, axis=1)
    nw = img - np.roll(np.roll(img, 1, axis=0), 1, axis=1)
    ne = img - np.roll(np.roll(img, 1, axis=0), -1, axis=1)
    se = img - np.roll(np.roll(img, -1, axis=0), -1, axis=1)
    sw = img - np.roll(np.roll(img, -1, axis=0), 1, axis=1)

    energies.append(np.exp(-1 * beta * np.sum(np.multiply(n, n), axis=2)))
    energies.append(np.exp(-1 * beta * np.sum(np.multiply(s, s), axis=2)))
    energies.append(np.exp(-1 * beta * np.sum(np.multiply(e, e), axis=2)))
    energies.append(np.exp(-1 * beta * np.sum(np.multiply(w, w), axis=2)))
    energies.append(np.exp(-1 * beta * np.sum(np.multiply(nw, nw), axis=2)))
    energies.append(np.exp(-1 * beta * np.sum(np.multiply(ne, ne), axis=2)))
    energies.append(np.exp(-1 * beta * np.sum(np.multiply(se, se), axis=2)))
    energies.append(np.exp(-1 * beta * np.sum(np.multiply(sw, sw), axis=2)))

    return energies


def create_graph(img):
    num_neighbors = 8

    num_nodes = img.shape[0] * img.shape[1] + 2
    num_edges = img.shape[0] * img.shape[1] * num_neighbors

    g = maxflow.Graph[int](num_nodes, num_edges)

    nodes = g.add_nodes(num_nodes - 2)

    return g, nodes


def show(img):
    cv2.imshow("title", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_mask(img):
    show(img.astype(np.float32))