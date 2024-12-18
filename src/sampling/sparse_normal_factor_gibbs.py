import torch
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np

from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.stats import bernoulli, invgamma

from src.utils.probability.density import truncated_beta, trunc_norm_mixture


def simulate_loading(args):
    mu_pos, mu_neg, sigma = args
    return trunc_norm_mixture(mu_pos=mu_pos, mu_neg=mu_neg, sigma=sigma).rvs()


def parallelized_loading(
    mu_pos, mu_neg, sigma, num_var, num_factor, device, float_storage
):
    indices = itertools.product(range(num_var), range(num_factor))
    args = [
        (mu_pos[j, k].item(), mu_neg[j, k].item(), sigma[j, k].item())
        for j, k in indices
    ]
    # Use Joblib for parallel execution
    samples = Parallel(n_jobs=-1)(delayed(simulate_loading)(arg) for arg in args)

    # Reshape the output
    B = torch.tensor(np.array(samples), dtype=float_storage, device=device).reshape(
        num_var, num_factor
    )  # work on reshape
    return B


class SpSlNormalBayesianFactorGibbs:
    def __init__(
        self,
        Y: torch.Tensor,
        B: torch.Tensor,
        Sigma: torch.Tensor,
        Gamma: torch.Tensor,
        Theta: torch.Tensor,
        alpha: float,
        eta: float,
        epsilon: float,
        lambda0: float,
        lambda1: float,
        burn_in: int = 50,
        device: str = "cuda",
    ):
        """Initialize the Bayesian Factor Gibbs Sampler.

        Args:
            Y (torch.Tensor): Observed Data (G x n)
            B (torch.Tensor): _description_
            Sigma (torch.Tensor): _description_
            Gamma (torch.Tensor): _description_
            Theta (torch.Tensor): _description_
            alpha (float): _description_
            eta (float): _description_
            epsilon (float): _description_
            lambda0 (float): _description_
            lambda1 (float): _description_
            burn_in (int, optional): _description_. Defaults to 50.
            fast (bool, optional): _description_. Defaults to True.
            device (str, optional): _description_. Defaults to "cuda".
        """
        # Move to GPU if available, "mps" for Mac, "cuda" for Windows
        if device == "mps":
            available = torch.backends.mps.is_available()
            self.float_storage = torch.float32
        elif device == "cuda":
            available = torch.cuda.is_available()
            self.float_storage = torch.float64
        elif device == "cpu":
            available = True
            self.float_storage = torch.float64
        else:
            return KeyError("device should be either 'mps' or 'cuda'")
        self.device = torch.device(device if available else "cpu")

        # Data
        self.Y = Y.to(self.device, dtype=self.float_storage)

        # Shapes
        self.num_var, self.num_obs = Y.shape
        self.num_factor = B.shape[1]

        # Parameters
        self.B = B.to(self.device, dtype=self.float_storage)
        self.Omega = torch.zeros(
            (self.num_factor, self.num_obs),
            device=self.device,
            dtype=self.float_storage,
        )
        self.Sigma = Sigma.to(self.device, dtype=self.float_storage)
        self.Gamma = Gamma.to(self.device, dtype=self.float_storage)
        self.Theta = Theta.to(self.device, dtype=self.float_storage)

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
        self.paths = {
            "B": [self.B.cpu().numpy()],
            "Omega": [self.Omega.cpu().numpy()],
            "Sigma": [self.Sigma.cpu().numpy()],
            "Gamma": [self.Gamma.cpu().numpy()],
            "Theta": [self.Theta.cpu().numpy()],
        }

    def perform_gibbs_sampling(
        self, iterations: int = None, store: bool = True, plot: bool = True
    ):
        """Perform Gibbs sampling for the specified number of iterations."""
        if iterations:
            self.num_iters = iterations

        with tqdm(total=self.num_iters, desc="Gibbs Sampling", unit="iter") as pbar:
            for i in range(self.num_iters):
                self.sample_factors(store=store)
                self.sample_features_allocation(store=store)
                self.sample_features_sparsity(store=store)
                self.sample_loadings(store=store)
                self.sample_diag_covariance(store=store)

                # Update the progress bar
                pbar.update(1)

                if i % (self.num_iters // 10) == 0 or i < 10:
                    print(f"Check Theta: {self.Theta}")

        if plot:
            self.plot_heatmaps()

    def sample_loadings(self, store: bool = True):
        """Vectorized sampling of the loading matrix `B`."""
        # Compute a
        a = torch.einsum("ki,j->jk", self.Omega**2, 1 / (2 * self.Sigma))

        # Compute b
        mask = 1 - torch.eye(
            self.num_factor, device=self.device, dtype=self.float_storage
        )
        excluded_sum = torch.einsum(
            "jl,li,lk->jik", self.B, self.Omega, mask
        )  # Shape (n,G)
        b = torch.einsum(
            "ki, jik -> jk",
            self.Omega,
            self.Y.unsqueeze(2).repeat(1, 1, self.num_factor) - excluded_sum,
        ) / (self.Sigma).unsqueeze(
            1
        )  # should be about 2a but too large, problem

        # Compute c
        c = self.lambda1 * self.Gamma + self.lambda0 * (1 - self.Gamma)

        # Vectorized sampling from truncated normal mixture
        mu_pos = ((b - c) / (2 * a)).cpu()
        mu_neg = ((b + c) / (2 * a)).cpu()
        sigma = torch.sqrt(1 / (2 * a)).cpu()

        # Simulate samples using CPU for now (replace with GPU-adapted truncated sampler if available)
        self.B = parallelized_loading(
            mu_pos=mu_pos,
            mu_neg=mu_neg,
            sigma=sigma,
            num_var=self.num_var,
            num_factor=self.num_factor,
            device=self.device,
            float_storage=self.float_storage,
        )  # parallelize and assess what is slow

        if store:
            self.paths["B"].append(self.B.cpu().numpy())

    def sample_factors(self, store: bool = True):
        """Sample the latent factor matrix `Omega`."""
        precision = (
            torch.eye(self.num_factor, device=self.device)
            + self.B.T @ torch.diag(1 / self.Sigma) @ self.B
        )
        cov = torch.linalg.inv(precision)
        cov = (cov + cov.T) / 2
        mean = (cov @ self.B.T @ torch.diag(1 / self.Sigma) @ self.Y).cpu()
        self.Omega = torch.stack(
            [
                torch.distributions.MultivariateNormal(
                    mean[:, i], covariance_matrix=cov.cpu()
                ).sample()
                for i in range(self.num_obs)
            ],
            dim=1,
        ).to(device=self.device, dtype=self.float_storage)

        if store:
            self.paths["Omega"].append(self.Omega.cpu().numpy())

    def sample_features_allocation(self, store: bool = True, epsilon: float = 1e-10):
        """Sample the feature allocation matrix `Gamma`."""
        p = (
            self.lambda1 * torch.exp(-self.lambda1 * torch.abs(self.B)) * self.Theta
        ) / (
            self.lambda0
            * torch.exp(-self.lambda0 * torch.abs(self.B))
            * (1 - self.Theta)
            + self.lambda1 * torch.exp(-self.lambda1 * torch.abs(self.B)) * self.Theta
            + epsilon
        )
        self.Gamma = torch.bernoulli(p)

        self.p = p

        if store:
            self.paths["Gamma"].append(self.Gamma.cpu().numpy())

    def sample_features_sparsity(
        self,
        store,
        get: bool = False,
        epsilon: float = 1e-10,
    ):
        for k in range(self.num_factor - 1, -1, -1):
            alpha = max(
                torch.sum(self.Gamma[:, k], dtype=self.float_storage).cpu().item()
                + self.alpha * (k == (self.num_factor - 1)),
                epsilon,  # Ensure alpha is at least epsilon
            )
            beta = (
                torch.sum(self.Gamma[:, k] == 0, dtype=self.float_storage).cpu().item()
                + 1
            )

            if k == 0:
                a, b = self.Theta[k + 1].cpu(), 1.0
            elif k == (self.num_factor - 1):
                a, b = 0.0, self.Theta[k - 1].cpu()
            else:
                a, b = self.Theta[k + 1].cpu(), self.Theta[k - 1].cpu()

            self.Theta[k] = torch.tensor(
                truncated_beta()._rvs(alpha=alpha, beta=beta, a=a, b=b),
                device=self.device,
                dtype=self.float_storage,
            )

        if store:
            self.paths["Theta"].append(self.Theta)

        if get:
            return self.Theta

    def sample_diag_covariance(self, store: bool = True):
        """Sample the diagonal covariance matrix `Sigma`."""
        shape = (self.eta + self.num_obs) / 2
        squared_errors = torch.sum((self.Y - self.B @ self.Omega) ** 2, dim=1)
        scale = (self.eta * self.epsilon + squared_errors) / 2
        self.Sigma = torch.tensor(
            invgamma.rvs(a=shape, scale=scale.cpu().numpy()),
            dtype=self.float_storage,
            device=self.device,
        )

        if store:
            self.paths["Sigma"].append(self.Sigma.cpu().numpy())

    def plot_heatmaps(
        self, str_param: str = "B", abs_value: bool = True, cmap: str = "viridis"
    ):
        """Plot heatmaps of parameter trajectories."""
        if str_param not in self.paths:
            raise KeyError(f"{str_param} not found in parameter paths.")

        iter_indices = torch.logspace(
            0,
            torch.log10(torch.tensor(self.num_iters, dtype=torch.float)),
            steps=min(10, self.num_iters),
            dtype=torch.int,
        )
        n_cols = 5
        n_rows = -(-len(iter_indices) // n_cols)

        first_matrix = (
            abs(self.paths[str_param][iter_indices[0]])
            if abs_value
            else self.paths[str_param][iter_indices[0]]
        )
        vmin, vmax = first_matrix.min(), first_matrix.max()

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.flatten()

        for idx, ax in zip(iter_indices, axes):
            matrix = (
                abs(self.paths[str_param][idx])
                if abs_value
                else self.paths[str_param][idx]
            )
            sns.heatmap(matrix, cmap=cmap, cbar=False, ax=ax, vmin=vmin, vmax=vmax)
            ax.set_title(f"Iter {idx}")

        for ax in axes[len(iter_indices) :]:
            ax.axis("off")

        plt.tight_layout()
        plt.show()
