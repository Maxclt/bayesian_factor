import numpy as np

from scipy.stats import multivariate_normal

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

    def sample_factors(self):
        """Sample the latent factor matrix `Omega`."""
        precision = (
            np.eye(self.num_factor, dtype=self.dtype)
            + self.B.T @ np.diag(1 / self.Sigma) @ self.B
        )
        cov = np.linalg.inv(precision)
        cov = (cov + cov.T) / 2
        mean = cov @ self.B.T @ np.diag(1 / self.Sigma) @ self.Y
        self.Omega = np.stack(
            [
                multivariate_normal(mean[:, i], covariance_matrix=cov).rvs()
                for i in range(self.num_obs)
            ],
            axis=1,
        ).astype(dtype=self.dtype)
