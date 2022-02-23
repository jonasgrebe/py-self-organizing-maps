from som.topologies import Topology
import numpy as np
import matplotlib.pyplot as plt
from typing import List

from som.metrics import manhattan_distance, euclidian_distance


class GridTopology(Topology):

    def __init__(self, height: int = 1, width: int = 1, depth: int = 1, d: int = 2, periodicities: List[bool] = [False, False, False]):
        assert d in [1, 2, 3], "Only 1D, 2D and 3D grid topologies are supported! Please choose d accordingly."
        assert d <= len(periodicities), f"Either decrease the dimensionality d = {d} or specify enough periodicity flags (one for each of the {d} dimensions)"
        self.d = d

        self.periodicities = periodicities

        if d == 1:
            self.grid_shape = (height, 1)
        elif d == 2:
            self.grid_shape = (height, width)
        elif d == 3:
            self.grid_shape = (height, width, depth)


    def get_node(self, node_idx):
        return np.unravel_index(node_idx, self.grid_shape)


    def get_neighbors_of_node_by_radius(self, node_idx, radius):
        all_nodes = np.array([self.get_node(i) for i in range(len(self))])
        node = all_nodes[node_idx]
        distances = self.metric(all_nodes, node)
        neighbor_idxs = np.where(distances <= radius)
        return neighbor_idxs, distances[neighbor_idxs]


    def get_direct_neighbors_of_node(self, node_idx):
        return self.get_neighbors_of_node_by_radius(node_idx, radius=1)


    def metric(self, x, y):
        if not any(self.periodicities[:self.d]):
            return manhattan_distance(x, y)
        else:
            s = np.zeros_like(x-y)
            for i in range(self.d):
                L = self.grid_shape[i]
                d = np.abs((x-y)[:,i])
                if self.periodicities[i]:
                    d[d > L / 2] = L - d[d > L / 2]
                s[:,i] += d
            return s.sum(axis=-1)


    def get_number_of_nodes(self):
        return np.prod(self.grid_shape)


    def plot_map(self, weights, axis=None, title="", image_shape=None, nearest_neighbor_set=None):
        # replace weights with nearest neighbors if necessary
        if nearest_neighbor_set is not None:
            weights = weights.copy()
            for i in range(weights.shape[0]):
                d = euclidian_distance(nearest_neighbor_set, weights[i])
                min_idx = np.argmin(d)
                weights[i] = nearest_neighbor_set[min_idx]

        # create an appropriate axis if not already provided
        if axis is None:
            axis = plt.figure().add_subplot(projection='3d') if self.d == 3 else plt.figure().add_subplot()

        def spiralize(x):
            x = x.copy()
            root = int(np.sqrt(len(x)))
            for cols in range(root, 0, -1):
                if len(x) % cols == 0:
                    target_shape = (cols, len(weights) // cols, *x.shape[1:])
                    x = x.reshape(*target_shape)

                    def spiral_idxs(ncols, nrows):
                        A = np.arange(ncols * nrows).reshape(ncols, nrows)
                        out = []
                        while(A.size):
                            out.append(A[0][::-1])    # first row reversed
                            A = A[1:][::-1].T         # cut off first row and rotate clockwise
                        return np.concatenate(out)[::-1]

                    spiral = np.empty_like(x)
                    idxs = spiral_idxs(*spiral.shape[:2])
                    spiral = spiral.reshape(-1, spiral.shape[-1])
                    spiral[idxs] = x.reshape(-1, spiral.shape[-1])
                    return spiral.reshape(*target_shape)

            raise ValueError("No Spiralization possible")

        # case 1: weights themselves can be plotted
        if weights.shape[-1] in [1, 3]:

            weights = weights.reshape(*self.grid_shape[:self.d], weights.shape[-1])

            if self.d in [1, 2]:
                if self.d == 1:
                    axis.imshow(spiralize(weights))
                else:
                    axis.imshow(weights)
            elif self.d == 3:
                voxels = np.ones(self.grid_shape, dtype=bool)
                if weights.shape[-1] == 1:
                    weights = np.concatenate([weights]*3, axis=-1)
                axis.voxels(voxels, facecolors=weights, edgecolor='k', alpha=0.8)
            else:
                raise NotImplementedError()

        # case 2: treat weights as images if image_shape is provided
        elif image_shape is not None:

            if self.d == 1:
                imgs = spiralize(weights)
                imgs = imgs.reshape(*imgs.shape[:2], *image_shape, -1)
            else:
                imgs = weights.reshape(*self.grid_shape, *image_shape, -1)

            if imgs.shape[-1] == 1:
                imgs = np.concatenate([imgs]*3, axis=-1)

            imgs = np.hstack(np.hstack(imgs))

            axis.imshow(imgs)

        # case 3: ...
        else:
            raise NotImplementedError()

        axis.set_title(title)
        axis.axis('off')

        return axis


    def plot_nodes(self, weights, axis=None, title=""):
        assert weights.shape[-1] in [1, 3]

        if weights.shape[-1] == 1:
            weights = np.concatenate([weights]*3, axis=-1)

        if axis is None:
            axis = plt.figure().add_subplot(projection='3d')

        for node_idx, node_weight in enumerate(weights):
            neighbor_idxs, _ = self.get_direct_neighbors_of_node(node_idx)
            for idx in neighbor_idxs[0]:
                neighbor_weight = weights[idx]

                axis.plot([node_weight[0], neighbor_weight[0]], [node_weight[1], neighbor_weight[1]], [node_weight[2], neighbor_weight[2]], c='black')

        axis.scatter(weights.T[0], weights.T[1], weights.T[2], c=weights)
        axis.set_title(title)
        axis.axis('off')

        return axis
