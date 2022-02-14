from som.topologies import Topology
import numpy as np
import matplotlib.pyplot as plt

from som.metrics import manhattan_distance


class GridTopology(Topology):

    def __init__(self, height: int = 0, width: int = 0, depth: int = 0, d: int = 2, periodic=False, periodic_x=False, periodic_y=False, periodic_z=False):
        assert d in [1, 2, 3], "Only 1D, 2D and 3D grid topologies are supported!"
        self.d = d

        self.periodicities = [periodic_x, periodic_y, periodic_z] if not periodic else [True]*3

        if d == 1:
            self.grid_shape = (height, 1)
        elif d == 2:
            self.grid_shape = (height, width)
        elif d == 3:
            self.grid_shape = (height, width, depth)

    def get_node(self, node_idx):
        return np.unravel_index(node_idx, self.grid_shape)


    def get_neighbors_of_node(self, node_idx, radius):
        all_nodes = np.array([self.get_node(i) for i in range(len(self))])
        node = all_nodes[node_idx]
        distances = self.metric(all_nodes, node)
        neighbor_idxs = np.where(distances <= radius)
        return neighbor_idxs, distances[neighbor_idxs]


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


    def plot_map(self, weights, axis=None, title=""):
        assert weights.shape[-1] in [1, 3, 4]

        if axis is None:
            axis = plt.figure().add_subplot(projection='3d') if self.d == 3 else plt.figure().add_subplot()

        weights = weights.reshape(*self.grid_shape, weights.shape[-1])

        if self.d in [1, 2]:
            axis.imshow(weights)
        else:
            voxels = np.ones(self.grid_shape, dtype=bool)
            if weights.shape[-1] == 1:
                weights = np.concatenate([weights]*3, axis=-1)
            axis.voxels(voxels, facecolors=weights, edgecolor='k', alpha=0.8)
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
            neighbor_idxs, _ = self.get_neighbors_of_node(node_idx, radius=1)
            for idx in neighbor_idxs[0]:
                neighbor_weight = weights[idx]

                axis.plot([node_weight[0], neighbor_weight[0]], [node_weight[1], neighbor_weight[1]], [node_weight[2], neighbor_weight[2]], c='black')

        axis.scatter(weights.T[0], weights.T[1], weights.T[2], c=weights)
        axis.set_title(title)
        axis.axis('off')

        return axis
