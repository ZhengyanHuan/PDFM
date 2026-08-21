import matplotlib.pyplot as plt
import os
from scipy.stats import wasserstein_distance

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from scipy.stats import gaussian_kde
import configs
import numpy as np
import ot
from ot.sliced import sliced_wasserstein_distance as SWD


def only_info(xy, uniform_samples, distance=True):

    num_out, prob = check_prob_l2ball(xy)

    if distance == True:
        distance = compute_SWD(uniform_samples, xy.transpose())

    return num_out, prob, distance


def plot_scatter_with_info(xy, uniform_samples, info, numItermax=200000, distance=True):
    xy2d = xy[:2,:]
    z = gaussian_kde(xy2d)(xy2d)
    # z = uniform_kde(xy, xy, 0.1)
    # heatmap, xedges, yedges = np.histogram2d(x_samples, y_samples, bins=30, range=[[0, 5], [0, 5]])
    # Create the plot

    fig, ax = plt.subplots()
    scatter = ax.scatter(xy2d.transpose()[:, 0], xy2d.transpose()[:, 1], c=z, s=3, cmap='viridis')

#     square = plt.Polygon(
#         [[- configs.bound + configs.uniform_center, - configs.bound + configs.uniform_center],
#          [configs.bound + configs.uniform_center, - configs.bound + configs.uniform_center],
#          [configs.bound + configs.uniform_center, configs.bound + configs.uniform_center],
#          [- configs.bound + configs.uniform_center, configs.bound + configs.uniform_center]],
#         closed=True, edgecolor='green', fill=False, linewidth=2
#     )
    # ax.add_patch(square)
    ax.set_aspect('equal')

    plt.colorbar(scatter, label='Density')

    # Set axis limits to clearly show the 5x5 area
    # if fix_bound == True:
    #     plt.xlim(-configs.bound * 1.5 + configs.uniform_center, configs.bound * 1.5 + configs.uniform_center)
    #     plt.ylim(-configs.bound * 1.5 + configs.uniform_center, configs.bound * 1.5 + configs.uniform_center)

    plt.xlabel('x')
    plt.ylabel('y')
    # plt.title('Density-Based Scatter Plot with Hexagonal Boundary')

    num_out, prob = check_prob_l2ball(xy)

    #     print(num_out)
    #     print(prob)

    if distance == True:
#         distance = compute_dist(uniform_samples, xy.transpose(), numItermax=numItermax)
        distance = compute_SWD(uniform_samples, xy.transpose())
        title = 'Total number:' + str(xy.shape[1]) + '  Outside number:' + str(
            num_out) + r'  Probabilty:%.4f' % prob + r'  Distance:%.4f' % distance
    else:
        title = 'Total number:' + str(xy.shape[1]) + '  Outside number:' + str(num_out) + r'  Probabilty:%.4f' % prob
    plt.title(title)
    plt.savefig('./fig/' + info + '.png', bbox_inches='tight')
    plt.show()


def check_prob_l2ball(samples_cpu):
    outliner_mat = np.linalg.norm(samples_cpu, axis=0) > 1
    # print(np.linalg.norm(samples_cpu, axis=0)[outliner_mat])
    num_out = np.sum(outliner_mat)
    prob = num_out / samples_cpu.shape[1]
    return num_out, prob


def compute_dist(uniform_samples, generated_samples, numItermax=100000):
    n_samples = int(uniform_samples.shape[0])
    M = ot.dist(uniform_samples, generated_samples)

    uniform_weights = np.ones(n_samples) / n_samples
    point_weights = np.ones(n_samples) / n_samples

    # Compute the 2D Wasserstein distance (optimal transport cost)
    emd_2d = ot.emd2(uniform_weights, point_weights, M, numItermax=numItermax)
    return emd_2d


def compute_SWD(ref, pred, sample_size=None):
    sample_size = min(pred.shape[0], ref.shape[0]) if sample_size is None else sample_size
    pred = shuffle(pred, sample_size=sample_size)
    ref = shuffle(ref, sample_size=sample_size)

    return SWD(pred, ref)

def shuffle(x, sample_size):
    """
        x: (B, D)
        ===
        return: (sample_size, D)
    """
    idx = np.random.choice(x.shape[0], sample_size, replace=False)
    return x[idx]


