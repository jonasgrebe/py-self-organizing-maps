import numpy as np
import matplotlib.pyplot as plt

from som.topologies import Topology
from som.metrics import get_metric
from som.neighborhood_functions import get_neighborhood_function

import os

class SelfOrganizingMap:

    def __init__(self, topology: Topology, metric: str = 'l2', initialization: str = 'random_uniform', neighborhood_function='gaussian'):
        self.topology = topology
        self.initialization = initialization.lower()

        self.metric = get_metric(metric)
        self.neighborhood_function = get_neighborhood_function(neighborhood_function)
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


    def fit(self, X, steps: int = None, initial_lr: float = 1.0, lr_decay: float = 0.005, initial_radius: int = 12, radius_decay: float = 0.01, eval_steps: int = 0, callback = None):
        self.X = X.copy()
        self.node_weights = self.get_initial_node_weights(X)

        if steps is None:
            steps = 10 * len(X)

        # for each epochs
        for t in range(steps):

            if callback is not None:
                callback(t, steps)

            # update the learning rate and radius
            learning_rate = initial_lr * (0.01 / initial_lr) ** (t / steps)
            radius = initial_radius * (1 / initial_radius) ** (t / steps)

            x = X[np.random.choice(len(X))]

            best_node_idx = self.get_best_matching_node_idx(x)
            neighbor_node_idxs, topology_distances = self.topology.get_neighbors_of_node_by_radius(best_node_idx, radius)

            distance_factors = self.neighborhood_function(topology_distances, radius)
            self.update_node_weights(neighbor_node_idxs, learning_rate, distance_factors, x)

            if t == 0:
                quantization_error, topographic_error = np.inf, np.inf
            elif t == steps - 1 or (eval_steps > 0 and t % steps % eval_steps == 0):
                quantization_error = self.compute_quantization_error()
                topographic_error = self.compute_topographic_error()

            print(f"\rStep {t+1} - Learning Rate: {learning_rate: .4f} - Radius: {radius: .2f} - Neighborhood_Size: {len(neighbor_node_idxs[0])} | Quantization Error: {quantization_error: .4f} - Topographic Error: {topographic_error: .4f}", end="")


    def fit_and_animate(self, X, frames=50, plot_data=True, filename="fit_and_animate", width=8, height=8, rotate=False, delay=0.1, repeat=True, magick_dir="imagemagick", **kwargs):
        files = []

        assert X.shape[-1] in [1, 3]

        def create_view_callback(t, steps):
            callback_every_steps = steps // frames

            if t % callback_every_steps == 0:
                tmp_filename = f"{t}.png"
                files.append(tmp_filename)
                f = plt.figure()
                ax1 = f.add_subplot(131, projection='3d') if self.topology.d == 3 else f.add_subplot(131)
                ax2 = f.add_subplot(132, projection='3d')

                ax1.figure.set_size_inches(width, height)
                ax2.figure.set_size_inches(width, height)

                if rotate:
                    angle = ((t // callback_every_steps) * (360 // frames)) % 360
                    ax2.view_init(elev=None, azim=angle)

                self.topology.plot_map(self.node_weights, axis=ax1, title=None)

                if plot_data:
                    if self.X.shape[-1] == 1:
                        colors = np.concatenate([self.X]*3, axis=-1)
                    else:
                        colors = self.X
                    ax2.scatter(*self.X.T, c=colors, alpha=0.1)

                self.topology.plot_nodes(self.node_weights, axis=ax2, title=None)

                ax2.dist = 8

                f.tight_layout()
                f.savefig(os.path.join(magick_dir, tmp_filename), dpi=400, bbox_inches='tight')
                plt.close()

        self.fit(X, callback=create_view_callback, **kwargs)

        loop = -1 if repeat else 0
        os.system('cd %s & magick convert -delay %d -loop %d %s %s'
                  % (magick_dir, delay, loop, " ".join(files), "../" + filename + ".gif"))

        for fname in files:
            os.remove(os.path.join(magick_dir, fname))



    def compute_quantization_error(self, X=None):
        if X is None:
            X = self.X
        error = []
        for i in range(len(X)):
            d = self.metric(self.node_weights, X[i])
            error.append(np.min(d))
        return np.mean(error)


    def compute_topographic_error(self, X=None):
        if X is None:
            X = self.X
        error = 0
        for i in range(len(X)):
            d = self.metric(self.node_weights, X[i])
            idx0, idx1 = np.argsort(d)[:2]
            neighbor_node_idxs, _ = self.topology.get_direct_neighbors_of_node(idx0)
            if not np.isin(idx1, neighbor_node_idxs):
                error += 1
        return error / len(X)


    def plot_map(self, axis=None, title="Learned Map", image_shape=None, nearest_neighbor_set=None, savefig=True, animate=False, filename="map"):
        axis = self.topology.plot_map(self.node_weights, axis, title, image_shape, nearest_neighbor_set)
        axis.figure.tight_layout()

        if savefig:
            axis.figure.savefig(filename + ".png", dpi=400, bbox_inches='tight')

            if animate and self.topology.d == 3:
                self.create_rotation_animation(axis, filename + ".gif")


    def plot_class_representation_map(self, y, colors, mode="discrete", axis=None, title="Class Representation Map", savefig=True, animate=False, filename="class_representation_map"):
        crm = np.empty(shape=(len(self.topology), 3))

        closest_data_points = {}
        for i in range(len(self.X)):
            d = self.metric(self.node_weights, self.X[i])
            closest_node = np.argmin(d)
            if closest_node not in closest_data_points:
                closest_data_points[closest_node] = []
            closest_data_points[closest_node].append(i)

        for node_idx in range(len(self.topology)):
            if node_idx not in closest_data_points:
                c = 0
            else:
                closest_point_idxs = closest_data_points[node_idx]
                closest_classes = y[closest_point_idxs]
                if mode == "continuous":
                    c = colors[closest_classes].mean(axis=0)
                elif mode == "discrete":
                    c = colors[np.bincount(closest_classes).argmax()]
                else:
                    raise NotImplementedError("Given mode ({mode}) is not supported. Use 'continuous' or 'discrete' instead.")

            crm[node_idx] = c

        axis = self.topology.plot_map(crm, axis, title)
        axis.figure.tight_layout()

        if savefig:
            axis.figure.savefig(filename + ".png", dpi=400, bbox_inches='tight')

            if animate and self.topology.d == 3:
                self.create_rotation_animation(axis, filename + ".gif")


    def plot_unified_distance_map(self, axis=None, title="Unified Distance Map", savefig=True, animate=False, filename="umap"):
        diffs = np.zeros(shape=(len(self.topology), 1))

        for node_idx in range(len(self.topology)):
            neighbor_idxs, _ = self.topology.get_direct_neighbors_of_node(node_idx)
            diffs[node_idx] = self.metric(self.node_weights[neighbor_idxs], self.node_weights[node_idx]).mean()

        diffs /= diffs.max()

        axis = self.topology.plot_map(diffs, axis, title)
        axis.figure.tight_layout()

        if savefig:
            axis.figure.savefig(filename + ".png", dpi=400, bbox_inches='tight')
            if animate and self.topology.d == 3:
                self.create_rotation_animation(axis, filename + ".gif")


    def plot_nodes(self, axis=None, title="Learned Manifold", plot_data=True, savefig=True, animate=False, filename="nodes"):
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
            if animate:
                self.create_rotation_animation(axis, filename + ".gif")


    def create_rotation_animation(self, axis, filename, frames=64, width=5, height=5, delay=0.5, repeat=True, magick_dir="imagemagick"):
        assert filename.endswith('.gif')

        angles = np.linspace(0, 360, frames+1)[:-1]
        files = []

        axis.figure.set_size_inches(width, height)
        for i, angle in enumerate(angles):
            axis.view_init(elev=None, azim=angle)
            fname = f"{i}.png"
            axis.figure.savefig(os.path.join(magick_dir, fname), dpi=400, bbox_inches='tight')
            files.append(fname)

        loop = -1 if repeat else 0
        os.system('cd %s & magick convert -delay %d -loop %d %s %s'
                  % (magick_dir, delay, loop, " ".join(files), "../" + filename))

        for fname in files:
            os.remove(os.path.join(magick_dir, fname))
