import numpy as np
import tensorflow as tf
from sklearn.mixture import GaussianMixture
from source.mesh_sampling import get_vert_connectivity, _get_poly_vertices


def _generate_meshes(n_samples, training_variance, min_time, max_time, data_handler):
    mean_mesh = data_handler._get_mean_polydata(time=min_time)
    adjacency = get_vert_connectivity(mean_mesh).todense()

    vertices = []
    for i in range(n_samples):
        # generates the vector for modifying the PCs of the heart mesh (in length of modes)
        x = np.random.normal(0, training_variance, len(data_handler.modes))

        # clips very small values to 0
        x[np.abs(x) < 0.001] = 0

        # does not allow too great outliers
        x[x > 2 * training_variance] = 2 * training_variance
        x[x < -2 * training_variance] = -2 * training_variance

        # randomly sets some indices to 0
        x[np.random.choice(len(x), np.random.choice(len(x)), replace=False)] = 0

        # randomly selects a time point for the sample
        time = np.random.uniform(min_time, max_time)

        vertices.append(_get_poly_vertices(data_handler._generate_sample(modes=data_handler.modes, time=time, std_num=x)))

    return adjacency, vertices


def _compute_L(adjacency):
    np.fill_diagonal(adjacency, 0)
    L = adjacency / adjacency.sum(axis=1)
    np.fill_diagonal(L, -1)
    L = L.transpose()
    return L


def _get_scaling_from_scales(scales):
    x_min, x_max = scales['x_min'], scales['x_max']
    z_min, z_max = scales['z_min'], scales['z_max']
    y_min, y_max = scales['y_min'], scales['y_max']

    output_scaling = {
        'dataset_min': np.min([x_min, y_min, z_min]).astype(np.float32),
        'dataset_max': np.max([x_max, y_max, z_max]).astype(np.float32)
    }
    output_scaling['dataset_scale'] = output_scaling['dataset_max'] - output_scaling['dataset_min']
    return output_scaling


def _get_GMM_params(L, vertices, scaling, n_components):

    def delta_func(v):
        v = v - scaling['dataset_min']
        v = v / scaling['dataset_scale']
        return L @ v

    deltas = np.array([delta_func(v) for v in vertices])

    means = []
    covs = []
    for dim in tf.range(deltas.shape[2]):
        dim_means = []
        dim_covs = []
        gmm = GaussianMixture(n_components=n_components, covariance_type="diag")
        gmm.fit(deltas[:,:,dim])
        for k in tf.range(n_components):
            dim_means.append(gmm.means_[k])
            dim_covs.append(gmm.covariances_[k])
        means.append(dim_means)
        covs.append(dim_covs)

    means = tf.constant(np.array(means))
    covs = tf.constant(np.array(covs))

    return means, covs


def get_trained_GMM(training_variance, n_samples, n_components, data_handler):
    if data_handler.dynamic_model:
        phases = np.array(data_handler.phases) / 10.0
    else:
        phases = np.array(data_handler.fix_time_step)

    min_time = np.min(phases)
    max_time = np.max(phases)

    adjacency, vertices = _generate_meshes(n_samples, training_variance, min_time, max_time, data_handler)

    L = tf.constant(_compute_L(adjacency), dtype=tf.dtypes.float32)

    means, covs = _get_GMM_params(L, vertices, _get_scaling_from_scales(data_handler.scales), n_components)

    return L, means, covs