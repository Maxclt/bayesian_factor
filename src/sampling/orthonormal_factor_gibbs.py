import numpy as np

from src.sampling.sparse_factor_gibbs import SpSlFactorGibbs
from src.sampling.metropolis_hastings import metropolis_hastings


def compute_projection_norm(Omega_minus_k: torch.Tensor, bar_Omega_k: torch.Tensor):

    if Omega_minus_k.numel() > 0:  # Check if Omega_minus_k has any columns
        # Compute the projection onto the span of Omega_minus_k
        # (Omega_minus_k^T @ Omega_minus_k)^-1 @ Omega_minus_k^T @ bar_Omega_k
        pinv = torch.linalg.pinv(Omega_minus_k.T @ Omega_minus_k)  # Pseudo-inverse
        projection = Omega_minus_k @ (pinv @ (Omega_minus_k.T @ bar_Omega_k))
    else:
        projection = torch.zeros_like(bar_Omega_k)

    # Compute the orthogonal component
    orthogonal_component = bar_Omega_k - projection

    # Return the norm of the orthogonal component
    return torch.norm(orthogonal_component).item()


class SpSlOrthonormalFactorGibbs(SpSlFactorGibbs):
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
        new_Omega = torch.zeros_like(self.Omega, device=self.device)
        for k in range(self.num_factor):
            B_k = self.B[:, k]

            bar_sigma_k = 1 / (B_k.T @ torch.diag(1 / self.Sigma) @ B_k + epsilon)
            bar_Omega_k = bar_sigma_k * self.compute_expression(k)
            Omega_minus_k = torch.cat((self.Omega[:k], self.Omega[k + 1 :]), dim=0).T
            proj = compute_projection_norm(Omega_minus_k, bar_Omega_k)

            def log_prob(d):
                return (self.num_obs - d**2) ** (
                    (self.num_obs - self.num_factor - 2) / 2
                ) * torch.exp(torch.norm(proj) * d / bar_sigma_k**2).item()

            d = metropolis_hastings(
                target_pdf=log_prob,
                proposal_sampler=lambda x: np.random.normal(x, 1.0),
                initial_state=1,
                burn_in=50,
            )
            Omega_k = self.sample_from_sphere(d, bar_Omega_k, epsilon)

            new_Omega[k, :] = Omega_k

        self.Omega = new_Omega

        if store:
            self.paths["Omega"].append(self.Omega.cpu().numpy())
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
