import numpy as np
from scipy.stats import multivariate_normal


class NormalBayesianFactorDGP:
    def __init__(self, B: np.array, Sigma: np.array):
        """_summary_

        Args:
            B (np.array): True Loadings Matrix (G x K)
            Sigma (np.array): True Covariance Matrix (G x G)
            num_sim (int): number of simulations
        """
        self.B = B
        self.Sigma = Sigma

    def simulate(self, size):
        return multivariate_normal.rvs(cov=self.B @ self.B.T + self.Sigma, size=size).T
