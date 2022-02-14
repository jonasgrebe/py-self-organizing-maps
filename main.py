from som.topologies import GridTopology
from som import SelfOrganizingMap
import numpy as np

seed = 0

for dataset in ['blobs']:
    N = 256

    if dataset == 'uniform':
        X = np.random.rand(N, 3)

    elif dataset == 'blobs':
        from sklearn.datasets import make_blobs
        X, _ = make_blobs(n_samples=N, centers=0.7*np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]]), cluster_std=10, n_features=3)
        min = X.min()
        max = X.max()
        X = (X - min) / (max - min)

    for d in [1, 2, 3]:

        if d == 1:
            height, width, depth = 64, 1, 1
        if d == 2:
            height, width, depth = 8, 8, 1
        if d == 3:
            height, width, depth = 4, 4, 4

        for periodic in [False, True]:

            print("\n>", dataset, d, periodic)

            np.random.seed(seed)
            topo = GridTopology(height, width, depth, d=d, periodic=periodic)
            som = SelfOrganizingMap(topo, metric='l2')

            som.node_weights = som.get_initial_node_weights(X)
            som.X = X.copy()

            if not periodic:
                som.plot_nodes(filename = f"imgs/{dataset}/nodes_{d}_random", title=None)
                som.plot_map(filename = f"imgs/{dataset}/map_{d}_random", title=None)
                #som.plot_differences_map(filename = f"imgs/{dataset}/differences_{d}_{'periodic' if periodic else 'normal'}_random", title=None)
                #plt.show()

            np.random.seed(seed)
            som.fit(X)

            som.plot_nodes(filename = f"imgs/{dataset}/nodes_{d}_{'periodic' if periodic else 'normal'}_trained", title=None)
            som.plot_map(filename = f"imgs/{dataset}/map_{d}_{'periodic' if periodic else 'normal'}_trained", title=None)
            #som.plot_differences_map(filename = f"imgs/{dataset}/differences_{d}_{'periodic' if periodic else 'normal'}_trained", title=None)
            #plt.show()
