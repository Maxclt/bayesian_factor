import torch

from src.sampling.sparse_factor_gibbs import SpSlFactorGibbs


class SpSlNormalFactorGibbs(SpSlFactorGibbs):
    """
    Implementation of the paper normal sampling of omega.
    """

    def __init__(self, *args, **kwargs):
        """
        Args:
            - Inherits all arguments from SpSlNormalBayesianFactorGibbs
        """
        super().__init__(*args, **kwargs)

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
