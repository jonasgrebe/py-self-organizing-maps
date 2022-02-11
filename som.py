import numpy as np
import matplotlib.pyplot as plt
from abc import ABC as Abstract
from abc import abstractmethod

class Topology(Abstract):

    @abstractmethod
    def get_node(self):
        pass

    @abstractmethod
    def get_neighbors_of_node(self, node, radius):
        pass

    @abstractmethod
    def metric(self, x):
        pass

    @abstractmethod
    def get_number_of_nodes(self):
        pass

    def __len__(self):
        return self.get_number_of_nodes()

    @abstractmethod
    def plot_map(self, weights, axis=None, title=""):
        pass

    @abstractmethod
    def plot_nodes(self, weights, axis=None, title=""):
        pass


class GridTopology(Topology):

    def __init__(self, height: int = 0, width: int = 0, depth: int = 0, d: int = 2):

        assert d in [1, 2, 3], "Only 1D, 2D and 3D grid topologies are supported!"
        self.d = d

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
        distances = self.metric(all_nodes - node).sum(axis=-1)
        neighbor_idxs = np.where(distances <= radius)
        return neighbor_idxs, distances[neighbor_idxs]

    def metric(self, x):
        return np.abs(x)

    def get_number_of_nodes(self):
        return np.prod(self.grid_shape)

    def plot_map(self, weights, axis=None, title=""):
        assert weights.shape[-1] in [1, 3, 4]

        if axis is None:
            axis = plt.figure().add_subplot(projection='3d') if self.d == 3 else plt.figure().add_subplot()

        if self.d == 1:
            weights = weights.reshape(*self.grid_shape, weights.shape[-1])
            axis.imshow(weights)
        elif self.d == 2:
            weights = weights.reshape(*self.grid_shape, weights.shape[-1])
            axis.imshow(weights)
        else:
            voxels = np.ones(self.grid_shape, dtype=bool)
            colors = weights.reshape(*self.grid_shape, weights.shape[-1])
            if colors.shape[-1] == 1:
                colors = np.concatenate([colors]*3, axis=-1)
            axis.voxels(voxels, facecolors=colors, edgecolor='k', alpha=0.8)
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


class SelfOrganizingMap:

    def __init__(self, topology, metric='l2', initialization='random_uniform'):
        self.topology = topology
        self.metric = metric.lower()
        self.initialization = initialization.lower()

        self.metric_fct = self.get_metric_function(metric)
        self.node_weights = self.get_initial_node_weights(X)


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


    def get_metric_function(self, metric):
        # Euclidian Distance
        if metric in ['l2', 'euclidian']:
            metric_fct = lambda W, x: np.sqrt(np.sum(np.square(W - x), axis=-1))

        # Manhattan Distance
        elif metric in ['l1', 'manhattan']:
            metric_fct = lambda W, x: np.abs(np.sum(W - x, axis=-1))

        return metric_fct


    def get_best_matching_node_idx(self, x):
        # Find the minimal distance node
        d = self.metric_fct(self.node_weights, x)
        min_idx = np.argmin(d)
        return min_idx


    def update_node_weights(self, node_idxs, learning_rate, distance_factors, x):
        self.node_weights[node_idxs] += learning_rate * distance_factors * (x - self.node_weights[node_idxs])


    def fit(self, X, iters: int = None, initial_lr: float = 1.0, lr_decay: float = 0.005, initial_radius: int = 12, radius_decay: float = 0.01):
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

            print(f"\rStep {t+1} - Learning Rate {learning_rate} - Radius {radius} - Neighborhood_Size {len(neighbor_node_idxs[0])}", end="")
            self.update_node_weights(neighbor_node_idxs, learning_rate, distance_factors, x)


    def plot_map(self, axis=None, title="Learned Map", filename="map"):
        axis = self.topology.plot_map(self.node_weights, axis, title)
        plt.tight_layout()

        #if self.topology.d == 3:
            #angles = np.linspace(0, 360, 21)[:-1]
            #rotanimate(axis, angles, filename + ".gif",delay=0.5, width=10, height=10)

        plt.savefig(filename + ".png", dpi=400)

    def plot_differences_map(self, axis=None, title="Differences Map", filename="map"):
        diffs = np.zeros_like(self.node_weights)

        for node_idx in range(len(self.topology)):
            neighbor_idxs, _ = self.topology.get_neighbors_of_node(node_idx, radius=2)
            diffs[node_idx] = np.abs(self.node_weights[neighbor_idxs] - self.node_weights[node_idx]).mean()

        diffs /= diffs.max()

        axis = self.topology.plot_map(diffs, axis, title)
        plt.tight_layout()

        #if self.topology.d == 3:
            #angles = np.linspace(0, 360, 21)[:-1]
            #rotanimate(axis, angles, filename + ".gif",delay=0.5, width=10, height=10)

        plt.savefig(filename + ".png", dpi=400)

    def plot_nodes(self, axis=None, title="Learned Manifold", filename="nodes"):
        axis = self.topology.plot_nodes(self.node_weights, axis, title)
        plt.tight_layout()

        #angles = np.linspace(0, 360, 21)[:-1]
        #rotanimate(axis, angles, filename + ".gif", delay=0.5, width=5, height=5)
        plt.savefig(filename + ".png", dpi=400)




if __name__ == '__main__':
    np.random.seed(0)

    N = 1000
    X = np.random.rand(N, 3)

    height = 8
    width = 8
    depth = 8

    for d in [1, 2, 3]:
        topo = GridTopology(height, width, depth, d=d)
        som = SelfOrganizingMap(topo)

        som.plot_nodes(filename = f"imgs/nodes_{d}_random", title=None)
        som.plot_differences_map(filename = f"imgs/differences_{d}_random", title=None)
        som.plot_map(filename = f"imgs/map_{d}_random", title=None)

        som.fit(X)
        som.plot_nodes(filename = f"imgs/nodes_{d}_trained", title=None)
        som.plot_differences_map(filename = f"imgs/differences_{d}_trained", title=None)
        som.plot_map(filename = f"imgs/map_{d}_trained", title=None)
