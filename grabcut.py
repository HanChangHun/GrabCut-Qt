import cv2
import numpy as np
from utils import *

gamma = 50
NEIGHBORHOOD = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, 1), (1, -1)]


def grabcut(img, p1=(0, 0), p2=(0, 0), mask=None):
    pixels = img.reshape((img.shape[0] * img.shape[1], img.shape[2])).astype(np.uint8)  # flatten image
    height, width, _ = img.shape  # get height and width
    alpha = np.zeros((height, width), dtype=np.bool)  # make matrix for transparency information

    if not mask.any():  # first iteration
        # rectangle points
        x_min, x_max = min(p1[0], p2[0]), max(p1[0], p2[0])
        y_min, y_max = min(p1[1], p2[1]), max(p1[1], p2[1])

        # initialize alpha
        for h in range(height):
            for w in range(width):
                if (w >= x_min) and (w <= x_max) and (h >= y_min) and (h <= y_max):
                    alpha[h, w] = 1

    else:  # if more than one iteration, use passed mask
        alpha = mask

    # initialize GMM
    FG_GMM = GMM(5)
    BG_GMM = GMM(5)
    FG_GMM.initialize_gmm(img[alpha == 1])
    BG_GMM.initialize_gmm(img[alpha == 0])

    # assign GMM components to pixels
    FG_comp = FG_GMM.get_component(pixels).reshape((height, width))
    BG_comp = BG_GMM.get_component(pixels).reshape((height, width))

    # learn GMM parameters from data z
    FG_assignments = -1 * np.ones((height, width))
    FG_assignments[alpha == 1] = FG_comp[alpha == 1]
    BG_assignments = -1 * np.ones((height, width))
    BG_assignments[alpha == 0] = BG_comp[alpha == 0]
    FG_GMM.update_components(img, FG_assignments)
    BG_GMM.update_components(img, BG_assignments)

    FG_comp = FG_GMM.get_component(pixels)
    BG_comp = BG_GMM.get_component(pixels)

    FG_data_term = data_term(FG_comp, FG_GMM, pixels)
    BG_data_term = data_term(BG_comp, BG_GMM, pixels)

    graph, nodes = create_graph(img)

    if not mask.any():
        for h in range(img.shape[0]):
            for w in range(img.shape[1]):
                index = h * img.shape[1] + w
                if w < x_min or w > x_max or h < y_min or h > y_max:
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
