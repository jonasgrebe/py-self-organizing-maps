from som.topologies import GridTopology
from som import SelfOrganizingMap

import numpy as np
import os
from collections import namedtuple

SEED = 0

Setup = namedtuple('Setup', 'dataset d grid_shape periodicities image_shape' )

setup_1 = Setup(dataset='uniform_colors', d=1, grid_shape=(64, 1, 1), periodicities=[True, False, False], image_shape=None)
setup_2 = Setup(dataset='uniform_colors', d=2, grid_shape=(8, 8, 1), periodicities=[False, False, False], image_shape=None)
setup_3 = Setup(dataset='uniform_colors', d=2, grid_shape=(8, 8, 1), periodicities=[True, False, False], image_shape=None)
setup_4 = Setup(dataset='uniform_colors', d=3, grid_shape=(4, 4, 4), periodicities=[False, False, False], image_shape=None)
setup_5 = Setup(dataset='color_blobs', d=1, grid_shape=(64, 1, 1), periodicities=[True, False, False], image_shape=None)
setup_6 = Setup(dataset='olivetti_faces', d=2, grid_shape=(20, 20, 1), periodicities=[False, False, False], image_shape=(64, 64))
setup_7 = Setup(dataset='cifar10', d=2, grid_shape=(16, 16, 1), periodicities=[False, False, False], image_shape=(32, 32))
setup_8 = Setup(dataset='digits', d=2, grid_shape=(8, 8, 1), periodicities=[False, False, False], image_shape=(8, 8))

setups = [setup_1, setup_2, setup_3, setup_4, setup_5, setup_6, setup_7, setup_8]

def load_dataset(dataset):
    DATASETS = ['uniform_colors', 'color_blobs', 'olivetti_faces', 'cifar10', 'digits']
    assert dataset in DATASETS

    if dataset == 'uniform_colors':
        N = 250

        X = np.random.rand(N, 3)
        return X, np.zeros(len(X), dtype=int)

    if dataset == 'color_blobs':
        from sklearn.datasets import make_blobs
        N = 250

        X, y = make_blobs(n_samples=N, centers=0.7*np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]]), cluster_std=10, n_features=3)
        min = X.min()
        max = X.max()
        X = (X - min) / (max - min)
        return X, y

    if dataset == 'olivetti_faces':
        from sklearn.datasets import fetch_olivetti_faces
        N = None

        X, y = fetch_olivetti_faces(return_X_y=True)
        X, y = X[:N], y[:N]
        return X, y

    if dataset == 'cifar10':
        import os
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
        import torchvision
        import torchvision.transforms as transforms
        N = 5000

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        X = dataset.data.reshape(len(dataset.data), -1) / 255
        y = np.array(dataset.targets)

        X, y = X[:N], y[:N]
        # classes = dataset.classes
        return X, y

    if dataset == 'digits':
        from sklearn.datasets import load_digits
        N = None

        X, y = load_digits(return_X_y=True)
        X /= X.max()

        X, y = X[:N], y[:N]
        return X, y



if __name__ == '__main__':
    from matplotlib import cm

    for s in setups:

        np.random.seed(SEED)
        X, y = load_dataset(dataset=s.dataset)

        identifier = f"{s.dataset}_{s.d}d_{'x'.join(list(map(str, s.grid_shape[:s.d])))}_p{'x'.join([str(int(b)) for b in s.periodicities[:s.d]])}"
        print("-"*128)
        print("Experiment:\t", identifier)
        print("Dataset:\t X:", X.shape, "y:", y.shape)
        print("-"*128)

        if not os.path.isdir(os.path.join("imgs", identifier)):
            os.makedirs(os.path.join("imgs", identifier))
        else:
            continue

        NUM_COLORS = len(np.unique(y))
        colors = np.array([cm.gist_rainbow(1.*i / NUM_COLORS) for i in range(NUM_COLORS)])[:,:3]

        topo = GridTopology(*s.grid_shape, d=s.d, periodicities=s.periodicities)
        som = SelfOrganizingMap(topo, metric='euclidian')

        try:
            som.fit_and_animate(X, filename=f'{identifier}_fit_animation', rotate=True)
        except AssertionError:
            som.fit(X)

        som.plot_map(image_shape=s.image_shape, title=None, filename=f'imgs/{identifier}/map', animate=True)

        try:
            som.plot_nodes(title=None, filename=f'imgs/{identifier}/nodes', animate=True)
        except:
            pass

        som.plot_unified_distance_map(title=None, filename=f'imgs/{identifier}/umap', animate=False)
        som.plot_class_representation_map(y, colors, mode="discrete", title=None,  filename=f'imgs/{identifier}/crm', animate=False)
