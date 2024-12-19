import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from src.sampling.sparse_normal_factor_gibbs import SpSlNormalBayesianFactorGibbs
from src.utils.setup.orthonormal_setup import compute_projection_norm


class OrthonormalFactorGibbs(SpSlNormalBayesianFactorGibbs):
    """
    Implementation of the paper orthonormal decomposition of omega to
    tackle the large s, small n magnitude issue.
    """

    def __init__(self, *args, **kwargs):
        """
        Args:
            - Inherits all arguments from SpSlNormalBayesianFactorGibbs
        """
        super().__init__(*args, **kwargs)

        # Initialize Omega
        self.initialize_orthonormal_factors()

    def initialize_orthonormal_factors(self):
        """
        Initialize an num_factors * num_obs matrix Omega on the sqrt(num_obs)-radius Stiefel manifold.
        """
        Q, _ = torch.linalg.qr(
            torch.randn(self.num_obs, self.num_obs, device=self.device)
        )
        self.Omega = Q[: self.num_factor, :] * torch.sqrt(
            torch.tensor(self.num_obs, device=self.device)
        )

    def sample_factors(self, store, get: bool = False, epsilon: float = 1e-10):
        new_Omega = torch.zeros_like(self.Omega)
        for k in range(self.num_factor):
            B_k = self.B[:, k]

            bar_sigma_k = 1 / (B_k.T @ torch.diag(1 / self.Sigma) @ B_k + epsilon)
            bar_Omega_k = bar_sigma_k * self.compute_expression(k)

            d = self.sample_d_metropolis(k, bar_Omega_k, bar_sigma_k)
            Omega_k = self.sample_from_sphere(d, bar_Omega_k, epsilon)

            new_Omega[k, :] = Omega_k

        self.Omega = new_Omega

        if store:
            self.paths["Omega"].append(self.Omega.clone())
        if get:
            return self.Omega

    def compute_expression(self, k):
        sum_B_Omega = torch.zeros(self.num_var, self.num_obs, device=self.device)
        for t in range(self.B.shape[1]):
            if t != k:
                sum_B_Omega += self.B[:, t].unsqueeze(1) @ self.Omega[t].unsqueeze(0)
        residual = self.Y - sum_B_Omega
        temp = torch.diag(1 / self.Sigma) @ residual
        result = temp.T @ self.B[:, k]
        return result

    def sample_d_metropolis(self, k, bar_Omega_k, bar_sigma_k):
        def log_prob(d, proj):
            return (self.num_obs - d**2) ** (
                (self.num_obs - self.num_factor - 2) / 2
            ) * torch.exp(torch.norm(proj) * d / bar_sigma_k**2)

        d = torch.empty(1, device=self.device).uniform_(
            -torch.sqrt(torch.tensor(self.num_obs, device=self.device)),
            torch.sqrt(torch.tensor(self.num_obs, device=self.device)),
        )
        proposal_scale = 0.1

        Omega_minus_k = torch.cat((self.Omega[:k], self.Omega[k + 1 :]), dim=0).T
        proj = compute_projection_norm(Omega_minus_k, bar_Omega_k)

        for _ in range(self.burn_in):
            d_proposal = d + torch.randn(1, device=self.device) * proposal_scale
            if abs(d_proposal) > torch.sqrt(
                torch.tensor(self.num_obs, device=self.device)
            ):
                continue
            acceptance_ratio = log_prob(d_proposal, proj) / log_prob(d, proj)
            if torch.rand(1, device=self.device) < acceptance_ratio:
                d = d_proposal

        return d.item()

    def sample_from_sphere(self, d, bar_Omega_k, epsilon: float = 1e-10):
        X = torch.randn(self.num_obs - 1, device=self.device)
        lambda_ = torch.norm(X)
        X_prime = (
            torch.sqrt(torch.tensor(self.num_obs, device=self.device)) * X / lambda_
        )
        X_hyperplane = torch.cat([X_prime, torch.tensor([0.0], device=self.device)]) + (
            d * bar_Omega_k / (torch.norm(bar_Omega_k) + epsilon)
        )
        return X_hyperplane
