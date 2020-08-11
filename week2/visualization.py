from typing import Optional
import numpy as np

import gif

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.colors as mcolors


# Code inspired from
# https://scipython.com/book/chapter-7-matplotlib/examples/bmi-data-with-confidence-ellipses/
# More information can be read in Bishop book, page 81


def get_cov_ellipse(mu: np.ndarray,
                    sigma: np.ndarray,
                    nstd: int,
                    **kwargs) -> None:
    """
    Return a matplotlib Ellipse patch representing the covariance matrix
    cov centred at centre and scaled by the factor nstd.
    """
    # Find and sort eigenvalues and eigenvectors into descending order
    eigvals, eigvecs = np.linalg.eigh(sigma)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # The anti-clockwise angle to rotate our ellipse by 
    # We compute the angle between the components of biggest eigenvector
    # from biggest eigenvalue
    vx, vy = eigvecs[:,0][0], eigvecs[:,0][1]
    theta = np.arctan2(vy, vx)

    # Width and height of ellipse to draw
    width, height = 2 * nstd * np.sqrt(eigvals)
    return Ellipse(xy=mu,
                   width=width,
                   height=height,
                   fill=False,
                   angle=np.degrees(theta), **kwargs)


def plot_gaussian(mu: np.ndarray,
                  sigma: np.ndarray,
                  ax,
                  n_stds: int = 3,
                  **kwargs) -> None:

    for i in range(n_stds):
        ellipse = get_cov_ellipse(mu=mu,
                                  sigma=sigma,
                                  nstd=i,
                                  **kwargs)
        ax.add_artist(ellipse)



def plot_data(X: np.ndarray,
              y: np.ndarray,
              mus: np.ndarray,
              sigmas: np.ndarray,
              ax) -> None:

    n_clusters = mus.shape[0]
    cluster_colors = list(mcolors.TABLEAU_COLORS.values())
    
    # Plot data points
    for i in range(n_clusters):
        X_clusters = X[y == i]
        ax.scatter(X_clusters[:, 0],
                   X_clusters[:, 1],
                   c=cluster_colors[i],
                   alpha=0.30)

    # Plot Gaussian components
    for i in range(n_clusters):
        plot_gaussian(mu=mus[i],
                      sigma=sigmas[i],
                      ax=ax,
                      color=cluster_colors[i],
                      linewidth=3)
    
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2');


@gif.frame
def data_vs_estimation_frame(X: np.ndarray,
                             y: np.ndarray,
                             true_mus: np.ndarray,
                             true_sigmas: np.ndarray,
                             mus: np.ndarray,
                             sigmas: np.ndarray,
                             epoch: Optional[int] = None,
                             **plot_args):
    
    _, axs =  plt.subplots(1, 2, **plot_args)

    plot_data(X=X,
              y=y,
              mus=true_mus,
              sigmas=true_sigmas,
              ax=axs[0])

    axs[0].set_title('Original data');

    plot_data(X=X,
              y=y,
              mus=mus,
              sigmas=sigmas,
              ax=axs[1])
    
    epoch_info = f'. Epoch/Iteration: {epoch}' if epoch is not None else ''
    axs[1].set_title(f'Estimated GMM{epoch_info}');
