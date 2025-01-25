import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.stats import bernoulli, invgamma, norm
from abc import ABC, abstractmethod

from src.utils.probability.density import truncated_beta, trunc_norm_mixture
from src.sampling.metropolis_hastings import metropolis_hastings


def simulate_loading(args):
    mu_pos, mu_neg, sigma = args
    return trunc_norm_mixture(mu_pos=mu_pos, mu_neg=mu_neg, sigma=sigma).rvs()


def parallelized_loading(mu_pos, mu_neg, sigma, num_var, num_factor, dtype):
    indices = itertools.product(range(num_var), range(num_factor))
    args = [(mu_pos[j, k], mu_neg[j, k], sigma[j, k]) for j, k in indices]
    # Use Joblib for parallel execution
    samples = Parallel(n_jobs=-1)(delayed(simulate_loading)(arg) for arg in args)

    # Reshape the output
    B = np.array(samples, dtype=dtype).reshape(num_var, num_factor)  # work on reshape
    return B


class SpSlFactorGibbs(ABC):

    def __init__(
        self,
        Y: np.ndarray,
        B: np.ndarray,
        Sigma: np.ndarray,
        Gamma: np.ndarray,
        Theta: np.ndarray,
        alpha: float,
        eta: float,
        epsilon: float,
        lambda0: float,
        lambda1: float,
        burn_in: int = 50,
        dtype=np.float64,
    ):
        """Initialize the Bayesian Factor Gibbs Sampler.

        Args:
            Y (np.ndarray): Observed Data Matrix (G x n).
            B (np.ndarray): Factor Loadings Matrix (G x K).
            Sigma (np.ndarray): Covariance Matrix (G x G).
            Gamma (np.ndarray): Feature Sparsity Matrix (K x 1).
            Theta (np.ndarray): Feature Allocation Matrix (G x K).
            alpha (float): _description_
            eta (float): _description_
            epsilon (float): _description_
            lambda0 (float): _description_
            lambda1 (float): _description_
            burn_in (int, optional): _description_. Defaults to 50.
            dtype (): Defaults to np.float64.
        """
        # Dtype
        self.dtype = dtype

        # Data
        self.Y = Y.astype(self.dtype)

        # Shapes
        self.num_var, self.num_obs = Y.shape
        self.num_factor = B.shape[1]

        # Parameters
        self.B = B.astype(self.dtype)
        self.Omega = self.initialize_factors()
        self.Sigma = Sigma.astype(self.dtype)
        self.Gamma = Gamma.astype(np.int32)
        self.Theta = Theta.astype(self.dtype)

        # Hyperparameters
        self.alpha = alpha
        self.eta = eta
        self.epsilon = epsilon
        self.lambda0 = lambda0
        self.lambda1 = lambda1

        # Gibbs settings
        self.burn_in = burn_in
        self.num_iters = burn_in

        # Trajectories for tracking parameters
        self.paths = {}
        self.paths["init"] = {
            "B": self.B,
            "Omega": self.Omega,
            "Sigma": self.Sigma,
            "Gamma": self.Gamma,
            "Theta": self.Theta,  # TODO modify plot accordingly
        }

        # Simulation Without Updating
        self.simulated_paths = {
            "B": [],
            "Omega": [],
            "Sigma": [],
            "Gamma": [],
            "Theta": [],
        }

    @abstractmethod
    def initialize_factors(self, epsilon: float = 1e-10):
        pass

    @abstractmethod
    def sample_factors(self, update=True):
        pass

    def perform_gibbs(
        self,
        iterations: int = None,
        scale: bool = False,
        store: bool = True,
        plot: bool = True,
        file_path: str = None,
    ):
        """Perform Gibbs sampling for the specified number of iterations.

        Args:
            iterations (int, optional): Number of iterations of the gibbs sampler. Defaults to None.
            store (bool, optional): Boolean to store or not the parameters. Defaults to True.
            plot (bool, optional): Boolean to plot or not. Defaults to True.
            file_path (str, optional): File path for storaga. Defaults to None.
        """
        if iterations:
            self.num_iters = iterations

        with tqdm(total=self.num_iters, desc="Gibbs Sampling", unit="iter") as pbar:
            for i in range(self.num_iters):
                self.sample_factors()
                self.sample_features_allocation()
                self.sample_features_sparsity()
                self.sample_loadings()
                self.sample_diag_covariance()

                if scale:
                    self.scale_group()

                if store:
                    self.paths[i] = {
                        "B": self.B,
                        "Omega": self.Omega,
                        "Sigma": self.Sigma,
                        "Gamma": self.Gamma,
                        "Theta": self.Theta,
                    }

                # Update the progress bar
                pbar.update(1)

                if i % (self.num_iters // 10) == 0 or i < 10:
                    print(f"Check Theta: {self.Theta}")

        if plot:
            self.plot_heatmaps()

        if file_path is not None:
            with open(file_path, "w") as json_file:
                json.dump(self.paths, json_file)

    def sample_loadings(self, update=True):
        """Vectorized sampling of the loading matrix `B`."""
        # Compute a
        a = np.einsum("ki,j->jk", self.Omega**2, 1 / (2 * self.Sigma), dtype=self.dtype)

        # Compute b
        mask = 1 - np.eye(self.num_factor, dtype=self.dtype)
        excluded_sum = np.einsum(
            "jl,li,lk->jik", self.B, self.Omega, mask, dtype=self.dtype
        )  # Shape (n,G)
        Y_expanded = np.expand_dims(self.Y, axis=2)  # Shape: (j, i, 1)
        Y_repeated = np.tile(Y_expanded, (1, 1, self.num_factor))
        b = np.einsum(
            "ki, jik -> jk",
            self.Omega,
            Y_repeated - excluded_sum,
            dtype=self.dtype,
        ) / np.expand_dims(
            self.Sigma, axis=1
        )  # should be about 2a but too large, problem Check

        # Compute c
        c = self.lambda1 * self.Gamma + self.lambda0 * (1 - self.Gamma)

        # Vectorized sampling from truncated normal mixture
        mu_pos = (b - c) / (2 * a)
        mu_neg = (b + c) / (2 * a)
        sigma = np.sqrt(1 / (2 * a))

        if update:
            # Simulate samples using CPU for now (replace with GPU-adapted truncated sampler if available)
            self.B = parallelized_loading(
                mu_pos=mu_pos,
                mu_neg=mu_neg,
                sigma=sigma,
                num_var=self.num_var,
                num_factor=self.num_factor,
                dtype=self.dtype,
            )  # parallelize and assess what is slow

        else:
            self.simulated_paths["B"].append(
                parallelized_loading(
                    mu_pos=mu_pos,
                    mu_neg=mu_neg,
                    sigma=sigma,
                    num_var=self.num_var,
                    num_factor=self.num_factor,
                    dtype=self.dtype,
                )
            )

    def sample_features_allocation(self, epsilon: float = 1e-10, update=True):
        """Sample the feature allocation matrix `Gamma`."""
        denominator = (
            self.lambda0 * np.exp(-self.lambda0 * np.abs(self.B)) * (1 - self.Theta)
            + self.lambda1 * np.exp(-self.lambda1 * np.abs(self.B)) * self.Theta
        ) * 1e15
        denominator = np.maximum(denominator, epsilon)

        p = (
            self.lambda1 * np.exp(-self.lambda1 * np.abs(self.B)) * self.Theta * 1e15
        ) / (denominator)

        if update:
            self.Gamma = bernoulli(p).rvs()
            self.p = p

        else:
            self.simulated_paths["Gamma"].append(bernoulli(p).rvs())

    def sample_features_sparsity(
        self,
        epsilon: float = 1e-15,
        update=True,
    ):
        for k in range(self.num_factor - 1, -1, -1):
            alpha = max(
                np.sum(self.Gamma[:, k]) + self.alpha * (k == (self.num_factor - 1)),
                epsilon,  # Ensure alpha is at least epsilon
            )
            beta = np.sum(self.Gamma[:, k] == 0) + 1

            if k == 0:
                a, b = self.Theta[k + 1], 1.0
            elif k == (self.num_factor - 1):
                a, b = 0.0, self.Theta[k - 1]
            else:
                a, b = self.Theta[k + 1], self.Theta[k - 1]

            if update:
                self.Theta[k] = truncated_beta()._rvs(alpha=alpha, beta=beta, a=a, b=b)

            else:
                self.simulated_paths["Theta"].append(
                    truncated_beta()._rvs(alpha=alpha, beta=beta, a=a, b=b)
                )

    def sample_diag_covariance(self, update=True):
        """Sample the diagonal covariance matrix `Sigma`."""
        shape = (self.eta + self.num_obs) / 2
        squared_errors = np.sum((self.Y - self.B @ self.Omega) ** 2, axis=1)
        scale = (self.eta * self.epsilon + squared_errors) / 2

        if update:
            self.Sigma = np.array(
                invgamma.rvs(a=shape, scale=scale),
                dtype=self.dtype,
            )

        else:
            self.simulated_paths["Sigma"].append(
                np.array(
                    invgamma.rvs(a=shape, scale=scale),
                    dtype=self.dtype,
                )
            )

    def scale_group(self, epsilon: float = 1e-10):

        def process_k(k):
            scale = self.sample_scale_metropolis(k=k)
            B_k = scale * self.B[:, k]
            Omega_k = (1 / max(scale, epsilon)) * self.Omega[k, :]
            return k, B_k, Omega_k

        # Parallel computation using joblib
        results = Parallel(n_jobs=-1)(
            delayed(process_k)(k) for k in range(self.num_factor)
        )

        # Update B, Omega with results
        for k, B_k, Omega_k in results:
            self.B[:, k] = B_k
            self.Omega[k, :] = Omega_k

    def sample_scale_metropolis(self, k: int, rw_scale: float = 0.1):

        def target_pdf(scale: float) -> float:
            abs_term = np.abs(scale * self.B[:, k])
            laplace_term = np.prod(
                self.lambda0 * np.exp(-self.lambda0 * abs_term) * (1 - self.Gamma[:, k])
                + self.lambda1 * np.exp(-self.lambda1 * abs_term) * self.Gamma[:, k]
            )
            gaussian_term = np.prod(np.exp(-self.Omega[k, :] ** 2 / (2 * scale**2)))
            # fonction de B, scale parallelized over G * fonction de Omega, scale parallelized over n
            return (
                laplace_term
                * gaussian_term
                * scale ** (self.num_var - self.num_obs - 1)
            )

        def proposal_sampler(scale) -> float:
            return scale + rw_scale * norm().rvs()

        scale = 1

        return metropolis_hastings(
            target_pdf=target_pdf,
            proposal_sampler=proposal_sampler,
            initial_state=scale,
        )

    def plot_heatmaps(
        self, str_param: str = "B", abs_value: bool = True, cmap: str = "viridis"
    ):
        """Plot heatmaps of parameter trajectories."""
        if str_param not in self.paths["init"]:
            raise KeyError(f"{str_param} not found in parameter paths.")

        iter_indices = np.logspace(
            0,
            np.log10(np.array(self.num_iters, dtype=np.float64)),
            num=min(10, self.num_iters),
            dtype=np.int64,
        )
        iter_indices = list(dict.fromkeys(iter_indices))
        n_cols = 5
        n_rows = -(-len(iter_indices) // n_cols)

        first_matrix = (
            abs(self.paths["init"][str_param])
            if abs_value
            else self.paths["init"][str_param]
        )
        vmin, vmax = first_matrix.min(), first_matrix.max()

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.flatten()

        for idx, ax in zip(iter_indices, axes):
            matrix = (
                abs(self.paths[idx - 1][str_param])
                if abs_value
                else self.paths[idx - 1][str_param]
            )
            sns.heatmap(matrix, cmap=cmap, cbar=False, ax=ax, vmin=vmin, vmax=vmax)
            ax.set_title(f"Iter {idx}")

        for ax in axes[len(iter_indices) :]:
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    def get_path(
        self,
        param: str = "B",
        coeff: tuple = (0, 0),
        abs_value: bool = True,
    ) -> np.ndarray:
        if param not in self.paths["init"]:
            raise KeyError(f"{param} not found in parameter paths.")

        path = [
            (
                np.abs(self.paths[iter][param][coeff[0], coeff[1]])
                if abs_value
                else self.paths[iter][param][coeff[0], coeff[1]]
            )
            for iter in self.paths
        ]

        return np.array(path)

    def get_sim(
        self,
        param: str = "B",
        coeff: tuple = (0, 0),
        abs_value: bool = True,
    ):
        if param not in self.simulated_paths:
            raise KeyError(f"{param} not found in parameter paths.")

        sim = [
            np.abs(iter[coeff[0], coeff[1]]) if abs_value else iter[coeff[0], coeff[1]]
            for iter in self.simulated_paths[param]
        ]

        return np.array(sim)

    def simulate(self, num_iter: int):
        for _ in range(num_iter):
            self.sample_factors(update=False)
            self.sample_features_allocation(update=False)
            self.sample_features_sparsity(update=False)
            self.sample_loadings(update=False)
            self.sample_diag_covariance(update=False)
