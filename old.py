import numpy as np
import matplotlib.pyplot as plt
from som import SelfOrganizingMap

def test_on_colors(epochs):
    N = 1000

    np.random.seed(7)
    X = np.random.rand(N, 3)
    #X = np.asarray([[1, 0, 0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[0.2,0.2,0.5]])  *255

    f = plt.figure()
    ax1 = f.add_subplot(131)
    ax2 = f.add_subplot(132)
    ax3 = f.add_subplot(133)

    s = SelfOrganizingMap(height=32, width=32, initialization='random_uniform', distance_metric='l2')
    s.W = s.initialize_weights(X)

    s.plot_som(ax1)
    ax1.set_title("Random")
    s.fit(X.copy(), epochs=epochs, lr_decay=0.1, radius_decay=0.1, initial_radius=4)

    s.plot_som(ax2)
    s.plot_node_difference_map(ax3)
    plt.tight_layout()
    plt.savefig(f"imgs/colors_epoch{epochs}.png", dpi=300, bbox_inches='tight')

def test_on_colors_3d(epochs):
    from mpl_toolkits import mplot3d

    N = 1000
    np.random.seed(7)
    X = np.random.rand(N, 3)


    s = SelfOrganizingMap(height=32, width=32, initialization='random_uniform', distance_metric='l2')
    s.fit(X.copy(), epochs=epochs, lr_decay=0.05, radius_decay=0.05)

    ax = plt.axes(projection = '3d')

    x = []
    y = []
    z = []
    c = []

    all_nodes = s.get_all_nodes()
    for node in all_nodes:
        w = s.W[node[0], node[1]]

        x.append(w[0])
        y.append(w[1])
        z.append(w[2])
        c.append(w)

        ny, nx = s.get_neighbors(node, radius=1, return_distances=False)

        for nn in zip(ny, nx):
            n_w = s.W[nn[0], nn[1]]

            ax.plot([w[0], n_w[0]], [w[1], n_w[1]], [w[2], n_w[2]], c='black')

    ax.set_title("Manifold Approximation")
    ax.scatter(x, y, z, c=c)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"imgs/colors3d_epoch{epochs}.png", dpi=500, bbox_inches='tight')



def test_on_mnist(epochs):
    from sklearn.datasets import load_digits
    X, _ = load_digits(return_X_y=True)

    np.random.seed(7)

    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    from matplotlib.cbook import get_sample_data

    f = plt.figure()
    ax1 = f.add_subplot(131)
    ax2 = f.add_subplot(132)
    ax3 = f.add_subplot(133)

    s = SelfOrganizingMap(height=16, width=16, topology='rectangular', initialization='random_uniform', distance_metric='l2')
    s.W = s.initialize_weights(X.copy())

    def imscatter(x, y, i, ax, zoom=0.8):
        ax.set_facecolor((0, 0, 0))
        x, y = np.atleast_1d(x, y)
        artists = []
        for x0, y0, idx in zip(x, y, i):
            im = OffsetImage(X[idx].reshape(8, 8), zoom=zoom, cmap=plt.get_cmap('gray'))
            ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
            artists.append(ax.add_artist(ab))
        ax.update_datalim(np.column_stack([x, y]))
        ax.autoscale()
        return artists

    subset = X.copy()
    np.random.shuffle(subset)

    points = s.predict(subset)
    idxs = np.arange(len(subset))
    imscatter(points.T[0], points.T[1], idxs, ax1)

    ax1.set_title("Random")
    s.fit(X.copy(), epochs=epochs, lr_decay=0.02, radius_decay=0.02, initial_radius=10)

    points = s.predict(subset)
    imscatter(points.T[0], points.T[1], idxs, ax2)

    s.plot_node_difference_map(ax3)
    plt.tight_layout()

    ax2.set_title("Learned Mapping")
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')

    for ax in [ax1, ax2, ax3]:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f"imgs/mnist_epoch{epochs}.png", dpi=400, bbox_inches='tight')

if __name__ == '__main__':

    for e in [15]:
        test_on_colors(e)
        test_on_colors_3d(e)
        #test_on_mnist(10)
