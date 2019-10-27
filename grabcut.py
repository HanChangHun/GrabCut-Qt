import numpy as np
import matplotlib.pyplot as plt
import cv2
import maxflow
import os, sys
from scipy.spatial.distance import mahalanobis, euclidean
from GMM import GMM

gamma = 50


def show(img):
	cv2.imshow("title", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


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


def grabcut(img, p1=[0, 0], p2=[0, 0], mask=np.array([])):
	pixels = img.reshape((img.shape[0] * img.shape[1], img.shape[2]))
	pixels = pixels.astype(int)
	height, width, _ = img.shape

	if not mask.any():
		xmin, xmax = min(p1[0], p2[0]), max(p1[0], p2[0])
		ymin, ymax = min(p1[1], p2[1]), max(p1[1], p2[1])

		print(xmin, xmax, ymin, ymax)

		alpha = np.zeros((height, width), dtype=np.int8)

		for h in range(height):
			for w in range(width):
				if (w >= xmin) and (w <= xmax) and (h >= ymin) and (h <= ymax):
					alpha[h, w] = 1

		mask_alpha = np.zeros((alpha.shape[0], alpha.shape[1], 3), dtype=np.uint8)
		mask_alpha[:, :, 0] = alpha
		mask_alpha[:, :, 1] = alpha
		mask_alpha[:, :, 2] = alpha

	else:
		alpha = np.zeros((height, width), dtype=np.int8)
		for h in range(height):
			for w in range(width):
				if mask[h, w, 0] != 0:
					alpha[h, w] = 1

	fore_ground = img[alpha == 1]
	back_ground = img[alpha == 0]

	FG_GMM = GMM(5)
	BG_GMM = GMM(5)

	FG_GMM.initialize_gmm(fore_ground)
	BG_GMM.initialize_gmm(back_ground)

	FG_comp = FG_GMM.get_component(pixels).reshape((img.shape[0], img.shape[1]))
	BG_comp = BG_GMM.get_component(pixels).reshape((img.shape[0], img.shape[1]))

	k = np.ones((img.shape[0], img.shape[1]), dtype=np.float32) * -1

	k[alpha == 1] = FG_comp[alpha == 1]
	k[alpha == 0] = BG_comp[alpha == 0]

	FG_assignments = -1 * np.ones(k.shape)
	FG_assignments[alpha == 1] = k[alpha == 1]

	BG_assignments = -1 * np.ones(k.shape)
	BG_assignments[alpha == 0] = k[alpha == 0]

	FG_GMM.update_components(img, FG_assignments)
	BG_GMM.update_components(img, BG_assignments)

	FG_component = FG_GMM.get_component(pixels)
	BG_component = BG_GMM.get_component(pixels)

	FG_data_term = data_term(FG_component, FG_GMM, pixels)
	BG_data_term = data_term(BG_component, BG_GMM, pixels)

	graph, nodes = create_graph(img)

	if not mask.any():
		for h in range(img.shape[0]):
			for w in range(img.shape[1]):
				index = h * img.shape[1] + w
				if w < xmin or w > xmax or h < ymin or h > ymax:
						w1 = 1e9
						w2 = 0
				else:
						w1 = FG_data_term[index]
						w2 = BG_data_term[index]

				graph.add_tedge(index, w1, w2)

	else:
		for h in range(img.shape[0]):
			for w in range(img.shape[1]):
				index = h * img.shape[1] + w
				if mask[h, w, 0] == 0:
					w1 = 1e9
					w2 = 0
				else:
					w1 = FG_data_term[index]
					w2 = BG_data_term[index]

				graph.add_tedge(index, w1, w2)

	smoothness_energies = smoothness_term(img)

	NEIGHBORHOOD = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, 1), (1, -1)]
	src_h = np.tile(np.arange(img.shape[0]).reshape(img.shape[0], 1), img.shape[1])
	src_w = np.tile(np.arange(img.shape[1]), (img.shape[0], 1))
	src_h = src_h.astype(np.int32)
	src_w = src_w.astype(np.int32)

	for i, energy in enumerate(smoothness_energies):
		if i in [1, 3, 6, 7]:
			continue
		height_offset, width_offset = NEIGHBORHOOD[i]

		dst_h = src_h + height_offset
		dst_w = src_w + width_offset

		idx = np.logical_and(np.logical_and(dst_h >= 0, dst_h < img.shape[0]),
							np.logical_and(dst_w >= 0, dst_w < img.shape[1]))

		src_idx = src_h * img.shape[1] + src_w
		dst_idx = dst_h * img.shape[1] + dst_w

		src_idx = src_idx[idx].flatten()
		dst_idx = dst_idx[idx].flatten()
		weights = energy.astype(np.float32)[idx].flatten()
		weights = gamma * weights

		for i in range(len(src_idx)):
			graph.add_edge(src_idx[i], dst_idx[i], weights[i], weights[i])

	graph.maxflow()
	partition = graph.get_grid_segments(nodes)
	alpha = partition.reshape(alpha.shape)

	mask_alpha = np.zeros((alpha.shape[0], alpha.shape[1], 3), dtype=np.uint8)
	mask_alpha[:, :, 0] = alpha
	mask_alpha[:, :, 1] = alpha
	mask_alpha[:, :, 2] = alpha

	return (img * mask_alpha), mask_alpha
