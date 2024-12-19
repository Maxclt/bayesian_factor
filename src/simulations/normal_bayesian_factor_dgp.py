import torch
from scipy.stats import multivariate_normal


class NormalBayesianFactorDGP:
    def __init__(self, B: torch.Tensor, Sigma: torch.Tensor):
        """_summary_

        Args:
            B (np.array): True Loadings Matrix (G x K)
            Sigma (np.array): True Covariance Matrix (G x G)
            num_sim (int): number of simulations
        """
        self.B = B.numpy()
        self.Sigma = Sigma.numpy()

    def simulate(self, size):
        return torch.tensor(
            multivariate_normal.rvs(cov=self.B @ self.B.T + self.Sigma, size=size).T,
            dtype=torch.float32,
        )
