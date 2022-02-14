import numpy as np
import matplotlib.pyplot as plt

from som.topologies import Topology
from som.metrics import get_metric

import os

class SelfOrganizingMap:

    def __init__(self, topology: Topology, metric: str = 'l2', initialization: str = 'random_uniform'):
        self.topology = topology
        self.initialization = initialization.lower()

        self.metric = get_metric(metric)
        self.X = None


    def get_initial_node_weights(self, X):
        N = len(self.topology)
        dim = X.shape[-1]

        # build the target shape for the weight matrix
        target_shape = (N, dim)

        # Boostrap Sampling: Randomly pick datasamples as initial weights (with replacement)
        if self.initialization == 'bootstrap':
            idxs = np.random.choice(len(X), self.height * self.width, replace=True)
            return X[idxs].reshape(target_shape).astype(float)

        # Random Uniform Weights in min-max-range of dataset
        if self.initialization == 'random_uniform':
            min = X.min()
            max = X.max()
            return min + np.random.rand(*target_shape) * (max - min)


    def get_best_matching_node_idx(self, x):
        # Find the minimal distance node
        d = self.metric(self.node_weights, x)
        min_idx = np.argmin(d)
        return min_idx


    def update_node_weights(self, node_idxs, learning_rate, distance_factors, x):
        self.node_weights[node_idxs] += learning_rate * distance_factors * (x - self.node_weights[node_idxs])


    def fit(self, X, iters: int = None, initial_lr: float = 1.0, lr_decay: float = 0.005, initial_radius: int = 12, radius_decay: float = 0.01):
        self.X = X.copy()
        self.node_weights = self.get_initial_node_weights(X)

        if iters is  None:
            iters = 10 * len(X)

        # for each epoch
        for t in range(iters):
            # update the learning rate and radius
            learning_rate = initial_lr * (0.01 / initial_lr) ** (t / iters)
            radius = initial_radius * (1 / initial_radius) ** (t / iters)

            x = X[np.random.choice(len(X))]

            best_node_idx = self.get_best_matching_node_idx(x)
            neighbor_node_idxs, topology_distances = self.topology.get_neighbors_of_node(best_node_idx, radius)

            distance_factors = np.exp(- topology_distances ** 2 / (2 * radius ** 2)).reshape(-1, 1)
            self.update_node_weights(neighbor_node_idxs, learning_rate, distance_factors, x)

            print(f"\rStep {t+1} - Learning Rate {learning_rate} - Radius {radius} - Neighborhood_Size {len(neighbor_node_idxs[0])}", end="")


    def plot_map(self, axis=None, title="Learned Map", savefig=True, filename="map"):
        axis = self.topology.plot_map(self.node_weights, axis, title)
        axis.figure.tight_layout()

        if savefig:
            axis.figure.savefig(filename + ".png", dpi=400, bbox_inches='tight')
            if self.topology.d == 3:
                angles = np.linspace(0, 360, 21)[:-1]
                self.create_rotation_animation(axis, filename + ".gif")

    def plot_differences_map(self, axis=None, title="Differences Map", savefig=True, filename="map"):
        diffs = np.zeros_like(self.node_weights)

        for node_idx in range(len(self.topology)):
            neighbor_idxs, _ = self.topology.get_neighbors_of_node(node_idx, radius=1)
            diffs[node_idx] = np.abs(self.node_weights[neighbor_idxs] - self.node_weights[node_idx]).mean()

        diffs /= diffs.max()

        axis = self.topology.plot_map(diffs, axis, title)
        axis.figure.tight_layout()

        if savefig:
            axis.figure.savefig(filename + ".png", dpi=400, bbox_inches='tight')
            if self.topology.d == 3:
                angles = np.linspace(0, 360, 21)[:-1]
                self.create_rotation_animation(axis, filename + ".gif")


    def plot_nodes(self, axis=None, title="Learned Manifold", plot_data=True, savefig=True, filename="nodes"):
        axis = self.topology.plot_nodes(self.node_weights, axis, title)

        if plot_data:
            if self.X.shape[-1] == 1:
                colors = np.concatenate([self.X]*3, axis=-1)
            else:
                colors = self.X
            axis.scatter(*self.X.T, c=colors, alpha=0.1)

        axis.figure.tight_layout()

        if savefig:
            axis.figure.savefig(filename + ".png", dpi=400, bbox_inches='tight')
            self.create_rotation_animation(axis, filename + ".gif")

    def create_rotation_animation(self, axis, filename, width=5, height=5, delay=0.5, repeat=True, magick_dir="imagemagick"):
        assert filename.endswith('.gif')
        prefix = "tmp"

        angles = np.linspace(0, 360, 31)[:-1]
        files = []

        axis.figure.set_size_inches(width, height)
        for i, angle in enumerate(angles):
            axis.view_init(elev=None, azim=angle)
            fname = '%s%03d.jpeg' % (prefix, i)
            axis.figure.savefig(os.path.join(magick_dir, fname), dpi=400, bbox_inches='tight')
            files.append(fname)

        loop = -1 if repeat else 0
        os.system('cd %s & magick convert -delay %d -loop %d %s %s'
                  % (magick_dir, delay, loop, " ".join(files), "../" + filename))

        for fname in files:
            os.remove(os.path.join(magick_dir, fname))
