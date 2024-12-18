import torch
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import bernoulli, invgamma
from src.utils.probability.density import truncated_beta, trunc_norm_mixture


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
        fast: bool = True,
        device: str = "cuda",
    ):
        """Initialize the Bayesian Factor Gibbs Sampler.

        Args:
            Y (torch.Tensor): _description_
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
        # Move to GPU if available
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

        # Data
        self.Y = Y.to(self.device)

        # Shapes
        self.num_var, self.num_obs = Y.shape
        self.num_factor = B.shape[1]

        # Parameters
        self.B = B.to(self.device)
        self.Omega = torch.zeros((self.num_factor, self.num_obs), device=self.device)
        self.Sigma = Sigma.to(self.device)
        self.Gamma = Gamma.to(self.device)
        self.Theta = Theta.to(self.device)

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

        for i in range(self.num_iters):
            print(f"Iteration: {i + 1}")
            self.sample_factors(store=store)
            self.sample_features_allocation(store=store)
            self.sample_features_sparsity(store=store)
            self.sample_loadings_fast(store=store)
            self.sample_diag_covariance(store=store)

        if plot:
            self.plot_heatmaps()

    def sample_loadings(self, store: bool = True):
        """Vectorized sampling of the loading matrix `B`."""
        # Compute a
        a = torch.einsum("ki,j->jk", self.Omega**2, 1 / (2 * self.Sigma**2))

        # Compute b
        mask = 1 - torch.eye(self.num_factor, device="cuda")
        excluded_sum = torch.einsum(
            "jl,li,lk->jk", self.B, self.Omega, mask
        )  # Shape (G, K)
        b = (self.Y.sum(dim=0, keepdim=True).T - excluded_sum) / (
            self.Sigma**2
        ).unsqueeze(1)

        # Compute c
        c = self.lambda1 * self.Gamma + self.lambda0 * (1 - self.Gamma)

        # Vectorized sampling from truncated normal mixture
        mu_pos = (b - c) / (2 * a)
        mu_neg = (b + c) / (2 * a)
        sigma = torch.sqrt(1 / (2 * a))

        # Simulate samples using CPU for now (replace with GPU-adapted truncated sampler if available)
        B_new = torch.empty_like(self.B)
        for j, k in itertools.product(range(self.num_var), range(self.num_factor)):
            B_new[j, k] = trunc_norm_mixture(
                mu_pos=mu_pos[j, k].item(),
                mu_neg=mu_neg[j, k].item(),
                sigma=sigma[j, k].item(),
            ).rvs()

        self.B = B_new.to(self.device)

        if store:
            self.paths["B"].append(self.B.cpu().numpy())

    def sample_factors(self, store: bool = True):
        """Sample the latent factor matrix `Omega`."""
        precision = (
            torch.eye(self.num_factor, device=self.device)
            + self.B.T @ torch.diag(1 / self.Sigma) @ self.B
        )
        A = torch.linalg.inv(precision)
        mean = A @ self.B.T @ torch.diag(1 / self.Sigma) @ self.Y
        self.Omega = torch.stack(
            [
                torch.distributions.MultivariateNormal(
                    mean[:, i], covariance_matrix=A
                ).sample()
                for i in range(self.num_obs)
            ],
            dim=1,
        )

        if store:
            self.paths["Omega"].append(self.Omega.cpu().numpy())

    def sample_features_allocation(self, store: bool = True):
        """Sample the feature allocation matrix `Gamma`."""
        p = (
            self.lambda1 * torch.exp(-self.lambda1 * torch.abs(self.B)) * self.Theta
        ) / (
            self.lambda0
            * torch.exp(-self.lambda0 * torch.abs(self.B))
            * (1 - self.Theta)
            + self.lambda1 * torch.exp(-self.lambda1 * torch.abs(self.B)) * self.Theta
            + self.epsilon
        )
        self.Gamma = torch.bernoulli(p)

        if store:
            self.paths["Gamma"].append(self.Gamma.cpu().numpy())

    def sample_diag_covariance(self, store: bool = True):
        """Sample the diagonal covariance matrix `Sigma`."""
        shape = (self.eta + self.num_obs) / 2
        squared_errors = torch.sum((self.Y - self.B @ self.Omega) ** 2, dim=1)
        scale = (self.eta * self.epsilon + squared_errors) / 2
        self.Sigma = torch.tensor(
            [invgamma.rvs(a=shape.item(), scale=s.item()) for s in scale],
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

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.flatten()

        for idx, ax in zip(iter_indices, axes):
            matrix = (
                abs(self.paths[str_param][idx])
                if abs_value
                else self.paths[str_param][idx]
            )
            sns.heatmap(matrix, cmap=cmap, cbar=False, ax=ax)
            ax.set_title(f"Iter {idx}")

        for ax in axes[len(iter_indices) :]:
            ax.axis("off")

        plt.tight_layout()
        plt.show()
