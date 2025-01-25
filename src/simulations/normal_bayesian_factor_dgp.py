import numpy as np
from scipy.stats import multivariate_normal


class NormalBayesianFactorDGP:
    def __init__(self, B: np.ndarray, Sigma: np.ndarray, dtype=np.float64):
        """_summary_

        Args:
            B (np.array): True Loadings Matrix (G x K)
            Sigma (np.array): True Covariance Matrix (G x G)
            num_sim (int): number of simulations
            dtype ():
        """
        self.dtype = dtype
        self.B = B.astype(self.dtype)
        self.Sigma = Sigma.astype(self.dtype)

    def simulate(self, size):
        return np.array(
            multivariate_normal.rvs(cov=self.B @ self.B.T + self.Sigma, size=size).T,
            dtype=self.dtype,
        )
