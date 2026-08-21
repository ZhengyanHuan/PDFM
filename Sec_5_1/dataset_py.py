import numpy as np
import os
import configs
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from ot.sliced import sliced_wasserstein_distance as SWD



def sample_uniform_box(n, box_bound = configs.box_bound, save = False) :

    samples = np.random.uniform(
        low=-box_bound,
        high=box_bound,
        size=(n, 2)
    )

    # Save if requested
    if save:
        os.makedirs("./dataset", exist_ok=True)
        np.save("./dataset/samples.npy", samples)

    return samples

def plot_scatter(xy, fix_bound=1.2, bound = configs.box_bound, constraint_bound = configs.constraint_bound,
                 ref_samples = None, display_numout = True, constraint_type = configs.constraint_type, r_lower = configs.ring_radius_lower,
                 r_upper = configs.ring_radius_upper, dataset_type = configs.dataset_type):

    xy = xy.T
    z = gaussian_kde(xy)(xy)

    fig, ax = plt.subplots()
    scatter = ax.scatter(xy.transpose()[:, 0], xy.transpose()[:, 1], c=z, s=3, cmap='viridis')

    if dataset_type == "uniform":
        square = plt.Polygon(
            [[- bound , - bound ],
             [bound , - bound ],
             [bound , bound ],
             [- bound , bound ]],
            closed=True, edgecolor='green', fill=False, linewidth=2
        )
        ax.add_patch(square)

        if constraint_type == "box" or constraint_type == "boxandline":
            square_constraint = plt.Polygon(
                [[- constraint_bound, - constraint_bound],
                 [constraint_bound, - constraint_bound],
                 [constraint_bound, constraint_bound],
                 [- constraint_bound, constraint_bound]],
                closed=True, edgecolor='green', fill=False, linewidth=2
            )
            ax.add_patch(square_constraint)
        elif constraint_type == "ring":
            circle_lower = plt.Circle(
                (0, 0),  # center (x, y)
                r_lower,  # radius
                edgecolor='green',
                fill=False,
                linewidth=2
            )
            circle_upper = plt.Circle(
                (0, 0),  # center (x, y)
                r_upper,  # radius
                edgecolor='green',
                fill=False,
                linewidth=2
            )
            ax.add_patch(circle_lower)
            ax.add_patch(circle_upper)
        else:
            raise NotImplementedError("constraint_type not implemented")
    elif dataset_type == "uniform2":
        square1 = plt.Polygon(
            [[- configs.box2_bound - configs.box2_center, - configs.box2_bound - configs.box2_center],
             [configs.box2_bound - configs.box2_center, - configs.box2_bound - configs.box2_center],
             [configs.box2_bound - configs.box2_center, configs.box2_bound - configs.box2_center],
             [- configs.box2_bound - configs.box2_center, configs.box2_bound - configs.box2_center]],
            closed=True, edgecolor='green', fill=False, linewidth=2
        )
        square2 = plt.Polygon(
            [[- configs.box2_bound + configs.box2_center, - configs.box2_bound + configs.box2_center],
             [configs.box2_bound + configs.box2_center, - configs.box2_bound + configs.box2_center],
             [configs.box2_bound + configs.box2_center, configs.box2_bound + configs.box2_center],
             [- configs.box2_bound + configs.box2_center, configs.box2_bound + configs.box2_center]],
            closed=True, edgecolor='green', fill=False, linewidth=2
        )
        ax.add_patch(square1)
        ax.add_patch(square2)
        if constraint_type == 'boxandline':
            ax.plot([0, configs.xysum], [configs.xysum, 0], color='red', linewidth=2)
            ax.plot([0, -configs.xysum], [-configs.xysum, 0], color='red', linewidth=2)
    else:
        raise NotImplementedError("dataset_type not implemented")

    num_out, prob = prob_out(xy.T)

    plt.colorbar(scatter, label='Density')

    # Set axis limits to clearly show the 5x5 area
    if fix_bound is not None:
        if dataset_type == "uniform":
            plt.xlim(-bound * fix_bound , bound * fix_bound )
            plt.ylim(-bound * fix_bound , bound * fix_bound )
        elif dataset_type == "uniform2":
            plt.xlim((-configs.box2_bound - configs.box2_center)* fix_bound, (configs.box2_bound + configs.box2_center)* fix_bound )
            plt.ylim((-configs.box2_bound - configs.box2_center) * fix_bound,
                     (configs.box2_bound + configs.box2_center) * fix_bound)

    plt.xlabel('x')
    plt.ylabel('y')
    # plt.title('Density-Based Scatter Plot with Hexagonal Boundary')
    title = ''
    if display_numout:
        title += 'Total number:' + str(xy.shape[1]) + '  Outside number:' + str(
            num_out) + r'  Probabilty:%.4f' % prob
    if ref_samples is not None:
        distance = compute_SWD(ref_samples, xy.T)
        title  +=  r'  SWD:%.4f' % distance

    plt.title(title)
    plt.show()



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


if configs.dataset_type == "uniform":
    if configs.constraint_type == 'box':
        def prob_out(samples_cpu): #for uniform2
            outliner_mat = np.logical_or(np.abs(np.abs(samples_cpu[:, 0]) ) >= configs.constraint_bound,
                                         np.abs(np.abs(samples_cpu[:, 1]) ) >= configs.constraint_bound)
            num_out = np.sum(outliner_mat)
            prob = num_out / samples_cpu.shape[0]
            return num_out, prob
    elif configs.constraint_type == 'ring':
        def prob_out(samples_cpu):
            inner_mat_ring = ~np.logical_and(np.linalg.norm(samples_cpu, axis=-1)>= configs.ring_radius_lower,
                                       np.linalg.norm(samples_cpu, axis=-1)<= configs.ring_radius_upper)

            inner_mat_box = np.logical_and(np.abs(np.abs(samples_cpu[:, 0]) ) <= configs.box_bound,
                                         np.abs(np.abs(samples_cpu[:, 1]) ) <= configs.box_bound)

            inner_mat = np.logical_and(inner_mat_ring, inner_mat_box)
            num_in = np.sum(inner_mat)
            num_out = samples_cpu.shape[0] - num_in
            prob = num_out / samples_cpu.shape[0]
            return num_in, prob
    else:
        raise NotImplementedError
elif configs.dataset_type == "uniform2":
    if configs.constraint_type == 'boxandline':
        def prob_out(samples_cpu):  # for uniform2
            outliner_mat1 = np.logical_or(np.abs(np.abs(samples_cpu[:, 0]) - configs.box2_center) >= configs.box2_bound,
                                         np.abs(np.abs(samples_cpu[:, 1]) - configs.box2_center) >= configs.box2_bound)
            outliner_mat2 = np.abs(samples_cpu[:, 0] + samples_cpu[:, 1]) >= configs.xysum
            outliner_mat = np.logical_or(outliner_mat1, outliner_mat2)
            num_out = np.sum(outliner_mat)
            prob = num_out / samples_cpu.shape[0]
            return num_out, prob
    elif configs.constraint_type == 'box':
        def prob_out(samples_cpu):  # for uniform2
            outliner_mat = np.logical_or(np.abs(np.abs(samples_cpu[:, 0]) - configs.box2_center) >= configs.box2_bound,
                                         np.abs(np.abs(samples_cpu[:, 1]) - configs.box2_center) >= configs.box2_bound)
            # outliner_mat2 = np.abs(samples_cpu[:, 0] + samples_cpu[:, 1]) >= configs.xysum
            # outliner_mat = np.logical_or(outliner_mat1, outliner_mat2)
            num_out = np.sum(outliner_mat)
            prob = num_out / samples_cpu.shape[0]
            return num_out, prob
    else:
        raise NotImplementedError

else:
    raise NotImplementedError("dataset_type not implemented")


def only_info(xy, uniform_samples, distance=True):

    num_out, prob = prob_out(xy)

    if distance == True:
        distance = compute_SWD(uniform_samples, xy)

    return num_out, prob, distance

def add_Gaussian_noise(data, sigma = 0.5):
    return data + np.random.randn(*data.shape) * sigma


def plot_scatter_hist(xy, fix_bound=1.2, bound = configs.box_bound, constraint_bound = configs.constraint_bound,
                 ref_samples = None, display_numout = True, constraint_type = configs.constraint_type, r_lower = configs.ring_radius_lower,
                 r_upper = configs.ring_radius_upper, dataset_type = configs.dataset_type, save_name = None, plotfig = True, denote_red = True):

    xy = xy.T
    distance = 0
    if plotfig:

        fig, ax = plt.subplots()
        # scatter = ax.scatter(xy.transpose()[:, 0], xy.transpose()[:, 1], c=z, s=3, cmap='viridis')
        H, xedges, yedges = np.histogram2d(xy.transpose()[:, 0], xy.transpose()[:, 1], bins=100)
    
        H_transformed = H**0.2  # sqrt intensifies visual difference (log-like)
        H_masked = np.ma.masked_where(H == 0, H_transformed)
        c = ax.pcolormesh(xedges, yedges, H_masked.T, cmap='Blues')
    
        if dataset_type == "uniform":
            square = plt.Polygon(
                [[- bound , - bound ],
                 [bound , - bound ],
                 [bound , bound ],
                 [- bound , bound ]],
                closed=True, edgecolor='green', fill=False, linewidth=2
            )
            ax.add_patch(square)
    
            if constraint_type == "box":
                square_constraint = plt.Polygon(
                    [[- constraint_bound, - constraint_bound],
                     [constraint_bound, - constraint_bound],
                     [constraint_bound, constraint_bound],
                     [- constraint_bound, constraint_bound]],
                    closed=True, edgecolor='black', fill=False, linewidth=1.2
                )
                ax.add_patch(square_constraint)
            elif constraint_type == "ring":
                circle_lower = plt.Circle(
                    (0, 0),  # center (x, y)
                    r_lower,  # radius
                    edgecolor='green',
                    fill=False,
                    linewidth=2
                )
                circle_upper = plt.Circle(
                    (0, 0),  # center (x, y)
                    r_upper,  # radius
                    edgecolor='green',
                    fill=False,
                    linewidth=2
                )
                ax.add_patch(circle_lower)
                ax.add_patch(circle_upper)
            else:
                raise NotImplementedError("constraint_type not implemented")
        elif dataset_type == "uniform2":
            square1 = plt.Polygon(
                [[- configs.box2_bound - configs.box2_center, - configs.box2_bound - configs.box2_center],
                 [configs.box2_bound - configs.box2_center, - configs.box2_bound - configs.box2_center],
                 [configs.box2_bound - configs.box2_center, configs.box2_bound - configs.box2_center],
                 [- configs.box2_bound - configs.box2_center, configs.box2_bound - configs.box2_center]],
                closed=True, edgecolor='black', fill=False, linewidth=1.2
            )
            square2 = plt.Polygon(
                [[- configs.box2_bound + configs.box2_center, - configs.box2_bound + configs.box2_center],
                 [configs.box2_bound + configs.box2_center, - configs.box2_bound + configs.box2_center],
                 [configs.box2_bound + configs.box2_center, configs.box2_bound + configs.box2_center],
                 [- configs.box2_bound + configs.box2_center, configs.box2_bound + configs.box2_center]],
                closed=True, edgecolor='black', fill=False, linewidth=1.2
            )
            ax.add_patch(square1)
            ax.add_patch(square2)
            if constraint_type == 'boxandline':
                ax.plot([0, configs.xysum], [configs.xysum, 0], color='red', linewidth=2)
                ax.plot([0, -configs.xysum], [-configs.xysum, 0], color='red', linewidth=2)
        else:
            raise NotImplementedError("dataset_type not implemented")

    num_out, prob = prob_out(xy.T)

    # plt.colorbar(scatter, label='Density')

    # Set axis limits to clearly show the 5x5 area
    if plotfig:
        if fix_bound is not None:
            if dataset_type == "uniform":
                plt.xlim(-bound * fix_bound , bound * fix_bound )
                plt.ylim(-bound * fix_bound , bound * fix_bound )
            elif dataset_type == "uniform2":
                plt.xlim((-configs.box2_bound - configs.box2_center)* fix_bound, (configs.box2_bound + configs.box2_center)* fix_bound )
                plt.ylim((-configs.box2_bound - configs.box2_center) * fix_bound,
                         (configs.box2_bound + configs.box2_center) * fix_bound)
                outliner_mat = np.logical_or(np.abs(np.abs(xy.transpose()[:, 0]) - configs.box2_center) >= configs.box2_bound,
                             np.abs(np.abs(xy.transpose()[:, 1]) - configs.box2_center) >= configs.box2_bound)
                red_samples = xy.transpose()[outliner_mat]
                plt.scatter(red_samples[:, 0], red_samples[:, 1], color='red', s=5)
    
    # plt.xlabel('x')
    # plt.ylabel('y')
    
    # plt.title('Density-Based Scatter Plot with Hexagonal Boundary')
    title = ''
    if display_numout:
        title += 'Total number:' + str(xy.shape[1]) + '  Outside number:' + str(
            num_out) + r'  Probabilty:%.4f' % prob
    if ref_samples is not None:
        distance = compute_SWD(ref_samples, xy.T)
        title  +=  r'  SWD:%.4f' % distance
    if plotfig:     
        ax.set_aspect('equal')
        ax.tick_params(labelsize=30)
        ax.set_xticks([-5, 0, 5])
        ax.set_yticks([-5, 0, 5])
        if save_name is not None:
            plt.savefig('./fig/' + configs.dataset_type + "_"+save_name+'.png', bbox_inches='tight', dpi=300)
        plt.title(title)
        plt.show()
    return distance

    