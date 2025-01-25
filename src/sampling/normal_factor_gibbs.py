import numpy as np

from scipy.stats import multivariate_normal

from src.sampling.sparse_factor_gibbs import SpSlFactorGibbs


class SpSlNormalFactorGibbs(SpSlFactorGibbs):
    """
    Implementation of the paper normal sampling of omega.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize_factors(self):
        self.Omega = np.random.normal(size=(self.num_factor, self.num_obs)).astype(
            self.dtype
        )

    def sample_factors(self, epsilon: float = 1e-10):
        """Sample the latent factor matrix `Omega` of shape ()."""
        precision = (
            np.eye(self.num_factor, dtype=self.dtype)
            + self.B.T @ np.diag(1 / self.Sigma) @ self.B
        )
        cov = np.linalg.inv(precision)
        cov = (cov + cov.T) / 2
        mean = cov @ self.B.T @ np.diag(1 / self.Sigma) @ self.Y
        self.Omega = np.stack(
            [
                multivariate_normal(mean=mean[:, i], cov=cov).rvs()
                for i in range(self.num_obs)
            ],
            axis=1,
        ).astype(dtype=self.dtype)
