import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import Parallel, delayed
from scipy.stats import norm, bernoulli, invgamma, multivariate_normal

from src.utils.probability.density import truncated_beta, trunc_norm_mixture


def sample_loading_parallel(j, k, a, b, c):
    mu_pos = (b[j, k] - c[j, k]) / (2 * a[j, k])
    mu_neg = (b[j, k] + c[j, k]) / (2 * a[j, k])
    sigma = np.sqrt(1 / (2 * a[j, k]))

    # Sample from the truncated normal mixture
    return trunc_norm_mixture(mu_pos=mu_pos, mu_neg=mu_neg, sigma=sigma).rvs()


class SpSlNormalBayesianFactorGibbs:
    def __init__(
        self,
        Y: np.array,
        B: np.array,
        Sigma: np.array,
        Gamma: np.array,
        Theta: np.array,
        alpha: float,
        eta: float,
        epsilon: float,
        lambda0: float,
        lambda1: float,
        burn_in: int = 50,
        fast: bool = True,
    ):
        """Initialize the Bayesian Factor Gibbs Sampler."""
        # Data
        self.Y = Y

        # Shapes
        self.num_var, self.num_obs = Y.shape
        self.num_factor = B.shape[1]

        # Parameters
        self.B = B
        self.Omega = np.zeros((self.num_factor, self.num_obs))
        self.Sigma = Sigma
        self.Gamma = Gamma
        self.Theta = Theta

        # Hyperparameters
        self.alpha = alpha
        self.eta = eta
        self.epsilon = epsilon
        self.lambda0 = lambda0
        self.lambda1 = lambda1

        # Gibbs settings
        self.burn_in = burn_in
        self.num_iters = burn_in
        self.fast = fast

        # Trajectories for tracking parameters
        self.paths = {
            "B": [self.B],
            "Omega": [self.Omega],
            "Sigma": [self.Sigma],
            "Gamma": [self.Gamma],
            "Theta": [self.Theta],
        }

    def perform_gibbs_sampling(
        self, iterations: int = None, store: bool = True, plot: bool = True
    ):
        """Perform Gibbs sampling for the specified number of iterations."""
        if iterations:
            self.num_iters = iterations

        for i in range(self.num_iters):
            print(f"Iteration: {i + 1}")
            self.sample_factors(store=store)
            self.sample_features_allocation(store=store)
            self.sample_features_sparsity(store=store)
            self.sample_loadings_fast(store=store)
            self.sample_diag_covariance(store=store)

        if plot:
            self.plot_heatmaps()

    def sample_loadings_fast(self, store: bool = True):
        """Vectorized sampling of the loading matrix `B`."""
        a = np.sum(self.Omega**2, axis=1) / (2 * self.Sigma[:, np.newaxis])
        BOmega = self.B @ self.Omega
        adjusted_Y = (
            self.Y[:, :, np.newaxis]
            - BOmega[:, :, np.newaxis]
            + self.B[:, :, np.newaxis] * self.Omega
        )
        b = np.einsum("gk,gio->gk", self.Omega, adjusted_Y) / self.Sigma[:, np.newaxis]
        c = self.lambda1 * self.Gamma + self.lambda0 * (1 - self.Gamma)

        results = Parallel(n_jobs=-1)(  # Use all available CPU cores
            delayed(sample_loading_parallel)(j, k, a, b, c)
            for j, k in itertools.product(range(self.num_var), range(self.num_factor))
        )

        self.B = np.array(results).reshape(self.num_var, self.num_factor)

        if store:
            self.paths["B"].append(self.B)

    def sample_factors(self, store: bool = True):
        """Sample the latent factor matrix `Omega`."""
        precision = (
            np.eye(self.num_factor) + self.B.T @ np.diag(1 / self.Sigma) @ self.B
        )
        A = np.linalg.inv(precision)
        Z = A @ self.B.T @ np.diag(1 / self.Sigma)
        mean = Z @ self.Y
        self.Omega = np.array(
            [
                multivariate_normal.rvs(mean=mean[:, i], cov=A)
                for i in range(self.num_obs)
            ]
        ).T

        if store:
            self.paths["Omega"].append(self.Omega)

    def sample_features_allocation(self, store: bool = True):
        """Sample the feature allocation matrix `Gamma`."""
        p = (self.lambda1 * np.exp(-self.lambda1 * np.abs(self.B)) * self.Theta) / (
            self.lambda0 * np.exp(-self.lambda0 * np.abs(self.B)) * (1 - self.Theta)
            + self.lambda1 * np.exp(-self.lambda1 * np.abs(self.B)) * self.Theta
            + self.epsilon
        )
        self.Gamma = bernoulli(p).rvs()

        if store:
            self.paths["Gamma"].append(self.Gamma)

    def sample_features_sparsity(self, store: bool = True):
        """Sample the feature sparsity vector `Theta`."""
        for k in range(self.num_factor):
            alpha = (
                np.sum(self.Gamma[:, k])
                + self.alpha * (k == self.num_factor - 1)
                + self.epsilon
            )
            beta = np.sum(1 - self.Gamma[:, k]) + 1
            a, b = (0, self.Theta[k - 1]) if k > 0 else (self.Theta[k + 1], 1)
            self.Theta[k] = truncated_beta()._rvs(alpha=alpha, beta=beta, a=a, b=b)

        if store:
            self.paths["Theta"].append(self.Theta)

    def sample_diag_covariance(self, store: bool = True):
        """Sample the diagonal covariance matrix `Sigma`."""
        shape = (self.eta + self.num_obs) / 2
        squared_errors = np.sum((self.Y - self.B @ self.Omega) ** 2, axis=1)
        scale = (self.eta * self.epsilon + squared_errors) / 2
        self.Sigma = invgamma.rvs(a=shape, scale=scale)

        if store:
            self.paths["Sigma"].append(self.Sigma)

    def plot_heatmaps(
        self, str_param: str = "B", abs_value: bool = True, cmap: str = "viridis"
    ):
        """Plot heatmaps of parameter trajectories."""
        if str_param not in self.paths:
            raise KeyError(f"{str_param} not found in parameter paths.")

        iter_indices = np.logspace(
            0, np.log10(self.num_iters), num=min(10, self.num_iters), dtype=int
        )
        n_cols = 5
        n_rows = -(-len(iter_indices) // n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.flatten()

        for idx, ax in zip(iter_indices, axes):
            matrix = (
                np.abs(self.paths[str_param][idx])
                if abs_value
                else self.paths[str_param][idx]
            )
            sns.heatmap(matrix, cmap=cmap, cbar=False, ax=ax)
            ax.set_title(f"Iter {idx}")

        for ax in axes[len(iter_indices) :]:
            ax.axis("off")

        plt.tight_layout()
        plt.show()
