import scipy.stats
import numpy as np

from sklearn.datasets import make_spd_matrix


def sample_mu(min_feature_values: np.ndarray,
              max_feature_values: np.ndarray) -> np.ndarray:
    feature_bounds = zip(min_feature_values, max_feature_values)
    return np.array([
        np.random.uniform(low=min_val, high=max_val)
        for min_val, max_val in feature_bounds
    ])


def sample_cov(n_features: int) -> np.ndarray:
    # Make sure they are invertible
    sigma = make_spd_matrix(n_features)
    # Make it symmetric
    return np.tril(sigma) + np.triu(sigma.T, 1) 


def sample_coefs(n_clusters: int) -> np.ndarray:
    # Sample mixing coefficients randomly
    coefs = np.random.random(size=n_clusters)
    # Make them sum to one (i.e. softmax)
    return np.exp(coefs)/sum(np.exp(coefs))


def sample_mixture(n_clusters: int = 4,
                   min_feature_values: np.ndarray = np.array([-10, -10]),
                   max_feature_values: np.ndarray = np.array([10, 10]),
                   seed: int = 500):
    np.random.seed(seed)
    n_features = len(min_feature_values)
    
    mus = np.stack([
        sample_mu(min_feature_values, max_feature_values)
        for _ in range(n_clusters)
    ])
    sigmas = np.stack([sample_cov(n_features) for _ in range(n_clusters)])
    coefs = sample_coefs(n_clusters)
    return mus, sigmas, coefs


def sample_data(mus: np.ndarray,
                sigmas: np.ndarray,
                coefs: np.ndarray,
                seed: int = 500,
                n_samples: int = 1000):

    np.random.seed(seed)
    
    n_clusters, n_features = mus.shape

    gaussians = [
        scipy.stats.multivariate_normal(mean=mus[i],
                                        cov=sigmas[i])
        for i in range(n_clusters)
    ]

    # Sample cluster labels according to weights
    cluster_idxs = np.random.choice(n_clusters,
                                    size=n_samples,
                                    replace=True,
                                    p=coefs)

    return np.array([gaussians[idx].rvs() for idx in cluster_idxs]), \
           cluster_idxs


def log_likelihood(X: np.ndarray,
                   mus: np.ndarray,
                   sigmas: np.ndarray,
                   coefs: np.ndarray) -> float:
    n_clusters = mus.shape[0]
    X_pdfs = np.vstack([
        scipy.stats.multivariate_normal(mean=mus[i],
                                        cov=sigmas[i]).pdf(X)
        for i in range(n_clusters)
    ])
    return np.log(np.dot(X_pdfs.T, coefs)).sum()
