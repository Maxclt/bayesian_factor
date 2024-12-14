import numpy as np
from scipy.stats import norm, bernoulli, invgamma, multivariate_normal
import matplotlib.pyplot as plt
import itertools
import seaborn as sns

from src.utils.probability.density import truncated_beta, trunc_norm_mixture


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
    ):
        """_summary_

        Args:
            Y (np.array): Observed Data (G x n)
            B (np.array): Loadings Matrix (G x K)
            Omega (np.array): Factors Matrix (K x n)
            Sigma (np.array): Diagonal Covariance Matrix, represented as G-dimensionnal array
            Gamma (np.array): Feature Allocation Matrix (G x K)
            Theta (np.array): Feature Sparsity Vector, K-dimensionnal array
            alpha (float): _description_
            eta (float): _description_
            epsilon (float): _description_
            lambda0 (float): _description_
            lambda1 (float): _description_
            burn_in (int, optional): _description_. Defaults to 50.
        """

        # Data
        self.Y = Y

        # Shapes
        self.num_var, self.num_obs = Y.shape
        self.num_factor = B.shape[1]

        # Parameters
        self.B = B
        self.Omega = np.zeros((self.num_factor, self.num_obs))
        self.Sigma = Sigma
        self.Gamma = Gamma  # TODO add method sampling gamma from theta only or just pass it as np.zeros(B.shape) in init()
        self.Theta = Theta

        # Hyperparameters
        self.alpha = alpha
        self.eta = eta
        self.epsilon = epsilon
        self.lamba0 = lambda0
        self.lamba1 = lambda1

        # Gibbs Settings
        self.burn_in = burn_in
        self.num_iters = burn_in

        # Trajectories
        self.paths = {
            "B": [B],
            "Omega": [],
            "Sigma": [Sigma],
            "Gamma": [],
            "Theta": [Theta],
        }

    def perform_gibbs_sampling(
        self, iterations: int = False, store: bool = True, plot: bool = True
    ):
        """_summary_

        Args:
            iterations (int, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        if iterations:
            self.num_iters = iterations

        # Plot the initial set up
        # self.plot_points("Initial Parameters")

        # TODO Add a possibility to initialize Omega given only B, Sigma
        self.sample_factors(store=store)

        # Run for the given number of iterations
        for i in range(self.num_iters):
            self.sample_loadings(store=store)
            self.sample_factors(store=store)
            self.sample_features_allocation(store=store)
            self.sample_features_sparsity(store=store)
            self.sample_diag_covariance(store=store)

        # Plot the final Parameters (Heatmap for B)
        if plot:
            self.plot_heatmaps()

        # Final Plot of the log |B_{1,1}|
        # self.plot_prob()

        return self.B, self.Omega, self.Sigma, self.Gamma, self.Theta

    def sample_loading(self, j: int, k: int, product: float) -> float:

        # Compute ajk (∑_{i} Omega[k, i] ** 2 / (2 * Sigma[j]))
        ajk = np.sum(self.Omega[k, :] ** 2) / (2 * self.Sigma[j])

        # Compute bjk (∑_{i} Omega[k, i] * (Y[j, i] - ∑_{l ≠ k} B[j, l] * Omega[l, i]))
        adjusted_y = self.Y[j, :] - product + self.B[j, k] * self.Omega[k, :]
        bjk = np.dot(self.Omega[k, :], adjusted_y)

        # Compute cjk (lambda1 * Gamma[j,k] + lambda0 * (1 - Gamma[j,k]))
        cjk = self.lamba1 * self.Gamma[j, k] + self.lamba0 * (1 - self.Gamma[j, k])

        # Compute truncated normal mixture parameters
        mu_pos = (bjk - cjk) / (2 * ajk)
        mu_neg = (bjk + cjk) / (2 * ajk)
        sigma = np.sqrt(1 / (2 * ajk))

        # Sample from truncated normal mixture
        return trunc_norm_mixture(mu_pos=mu_pos, mu_neg=mu_neg, sigma=sigma).rvs()[0]

    def sample_loadings(self, store, get: bool = False):
        for j in range(self.num_var):
            product = np.dot(self.B[j, :], self.Omega)
            for k in range(self.num_factor):
                self.B[j, k] = self.sample_loading(j, k, product)

        if store:
            self.paths["B"].append(self.B)

        if get:
            return self.B

    def sample_factors(self, store, get: bool = False):

        # Compute (I_K + B^T Σ^-1 B)^-1
        Z = self.B.T @ np.diag(1 / self.Sigma)
        A = np.linalg.inv(np.eye(self.num_factor) + Z @ self.B)

        for i in range(self.num_obs):
            self.Omega[:, i] = multivariate_normal.rvs(mean=A @ Z @ self.Y[:, i], cov=A)

        if store:
            self.paths["Omega"].append(self.Omega)

        if get:
            return self.Omega

    def sample_features_allocation(
        self, store, get: bool = False, epsilon: float = 1e-10
    ):
        for j, k in itertools.product(range(self.num_var), range(self.num_factor)):
            p = (
                self.lamba1
                * np.exp(-self.lamba1 * np.abs(self.B[j, k]))
                * self.Theta[k]
            ) / (
                (
                    self.lamba0
                    * np.exp(-self.lamba0 * np.abs(self.B[j, k]))
                    * (1 - self.Theta[k])
                    + self.lamba1
                    * np.exp(-self.lamba1 * np.abs(self.B[j, k]))
                    * self.Theta[k]
                    + epsilon
                )
            )
            self.Gamma[j, k] = bernoulli(p).rvs()

        if store:
            self.paths["Gamma"].append(self.Gamma)

        if get:
            return self.Gamma

    def sample_features_sparsity(
        self,
        store,
        get: bool = False,
    ):
        for k in range(self.num_factor):
            alpha = np.sum(self.Gamma[:, k]) + self.alpha * (k == (self.num_factor - 1))
            beta = np.sum(self.Gamma[:, k] == 0) + 1

            if k == 0:
                self.Theta[k] = truncated_beta()._rvs(
                    alpha=alpha, beta=beta, a=self.Theta[k + 1], b=1
                )
            elif k == (self.num_factor - 1):
                self.Theta[k] = truncated_beta()._rvs(
                    alpha=alpha, beta=beta, a=0, b=self.Theta[k - 1]
                )
            else:
                self.Theta[k] = truncated_beta()._rvs(
                    alpha=alpha, beta=beta, a=self.Theta[k + 1], b=self.Theta[k - 1]
                )

        if store:
            self.paths["Theta"].append(self.Theta)

        if get:
            return self.Theta

    def sample_diag_covariance(self, store, get: bool = False):
        shape = (self.eta + self.num_obs) / 2
        for j in range(self.num_var):
            scale = (
                self.eta * self.epsilon
                + np.sum((self.Y[j, :] - self.B[j, :] @ self.Omega) ** 2)
            ) / 2
            self.Sigma[j] = invgamma.rvs(a=shape, scale=scale)

        if store:
            self.paths["Sigma"].append(self.Sigma)

        if get:
            return self.Sigma

    def plot_heatmaps(
        self,
        str_param: str = "B",
        iters: np.array = False,
        abs_value: bool = True,
        cmap: str = "viridis",
    ):

        if str_param not in self.paths:
            raise KeyError(
                f"{str_param} key not found in parameters paths keys: {self.paths.keys()}."
            )

        if not iters:
            iter_indices = np.logspace(
                np.log10(1),
                np.log10(self.num_iters),
                num=min(10, self.num_iters),
                dtype=int,
            )

        # Fixed columns per row
        n_plots = len(iter_indices)
        n_cols = 5  # Number of columns per row
        n_rows = int(np.ceil(n_plots / n_cols))

        # Create the figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.flatten()  # Flatten for easy indexing
        cbar = False

        for ax_idx, idx in enumerate(iter_indices):
            matrix = self.paths[str_param][idx]
            print(matrix.shape)
            # Apply absolute value if required
            if abs_value:
                matrix = np.abs(matrix)

            if (ax_idx + 1) % 5 == 0:
                cbar = True
            # Plot heatmap on the current axis
            sns.heatmap(matrix, cmap=cmap, annot=False, cbar=cbar, ax=axes[ax_idx])
            axes[ax_idx].set_title(f"Itr {idx}")
            cbar = False

        # Hide any unused axes
        for ax_idx in range(len(iter_indices), len(axes)):
            axes[ax_idx].axis("off")

        # Adjust layout
        plt.tight_layout()
        plt.show()

    # TODO add plot estimated against true covariance matrix of the normal bayesian factor model (BB^T + Sigma)
