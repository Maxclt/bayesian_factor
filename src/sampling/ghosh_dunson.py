import numpy as np

from typing import Tuple
from scipy.stats import norm
from joblib import Parallel, delayed

from src.sampling.sparse_factor_gibbs import SpSlFactorGibbs, parallelized_loading
from src.sampling.metropolis_hastings import metropolis_hastings
from src.utils.setup.orthonormal_setup import compute_projection_norm
from scipy.stats import bernoulli
from tqdm import tqdm
import json
from scipy.stats import multivariate_normal

#Notes for myself and any futre readers:
# B: adds a column_wise magnitude paramater r_k to the old loading  -> update sample_loadings
# Omega is prescribed and ~N(0, I) -> update initialize_factors and sample_factors
# Gamma : Same 
# Theta : Same
# Sigma : Same


class GhoshDunsonGibbs(SpSlFactorGibbs):
    """
    Implementation of the paper orthonormal decomposition of omega to
    tackle the large s, small n magnitude issue.
    """

    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)
        #initialize Q as a copy of B
        self.Q = self.B.copy()
        self.paths = {}
        self.paths["init"] = {
            "B": self.B,
            "Omega": self.Omega,
            "Sigma": self.Sigma,
            "Gamma": self.Gamma,
            "Theta": self.Theta,  # TODO modify plot accordingly
            "Q": self.Q,
        }

    def initialize_factors(self):
        """
        w_i ~N(0, I) is presribed in this model. w_i are the columns of Omega (Kxn)
        """
        self.Omega= np.random.normal(size=(self.num_factor, self.num_obs)).astype(self.dtype)
        
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
        ).astype(dtype=self.dtype) #Equivalent à la partie de LEO
        
    def sample_loadings(self, epsilon: float = 1e-10,lambda_r : float = 0.001):
        """
        Sample loadings from a normal distribution
        """
        # Compute a
        a = np.einsum("ki,j->jk", self.Omega**2, 1 / (2 * self.Sigma), dtype=self.dtype)

        # Compute b
        mask = 1 - np.eye(self.num_factor, dtype=self.dtype)
        excluded_sum = np.einsum(
            "jl,li,lk->jik", self.B, self.Omega, mask, dtype=self.dtype
        )  # Shape (n,G)
        Y_expanded = np.expand_dims(self.Y, axis=2)  # Shape: (j, i, 1)
        Y_repeated = np.tile(Y_expanded, (1, 1, self.num_factor))
        b = np.einsum(
            "ki, jik -> jk",
            self.Omega,
            Y_repeated - excluded_sum,
            dtype=self.dtype,
        ) / np.expand_dims(
            self.Sigma, axis=1
        )  # should be about 2a but too large, problem Check

        # Compute c
        c = self.lambda1 * self.Gamma + self.lambda0 * (1 - self.Gamma)

        # Vectorized sampling from truncated normal mixture
        mu_pos = (b - c) / (2 * a)
        mu_neg = (b + c) / (2 * a)
        sigma = np.sqrt(1 / (2 * a))

        # Simulate samples using CPU for now (replace with GPU-adapted truncated sampler if available)
        self.Q = parallelized_loading(
            mu_pos=mu_pos,
            mu_neg=mu_neg,
            sigma=sigma,
            num_var=self.num_var,
            num_factor=self.num_factor,
            dtype=self.dtype,
        )  # In ghosh dunson we add r_k to the loading.
        
        # the prior for r_k is a laplace distribution of parameter self.lambda
        # sample r_k a column wise magnitude parameter
        r = np.random.laplace(loc=0, scale=1/lambda_r, size=(1, self.num_factor)).astype(self.dtype)
        # print(f"    r: {r}")
        self.B = self.Q * r # nouveau calcul de B ?
        
    def sample_features_allocation(self, epsilon: float = 1e-10):
        """Sample the feature allocation matrix `Gamma`."""
        denominator = (
            self.lambda0 * np.exp(-self.lambda0 * np.abs(self.Q)) * (1 - self.Theta) # self.B ou self.Q ?
            + self.lambda1 * np.exp(-self.lambda1 * np.abs(self.Q)) * self.Theta
        ) * 1e15
        denominator = np.maximum(denominator, epsilon)

        p = (
            self.lambda1 * np.exp(-self.lambda1 * np.abs(self.Q)) * self.Theta * 1e15
        ) / (denominator)

        self.Gamma = bernoulli(p).rvs()

        self.p = p
        
        
    def sample_features_allocation_B(self, epsilon: float = 1e-10):
        """Sample the feature allocation matrix `Gamma`."""
        denominator = (
            self.lambda0 * np.exp(-self.lambda0 * np.abs(self.B)) * (1 - self.Theta) # self.B ou self.Q ?
            + self.lambda1 * np.exp(-self.lambda1 * np.abs(self.B)) * self.Theta
        ) * 1e15
        denominator = np.maximum(denominator, epsilon)

        p = (
            self.lambda1 * np.exp(-self.lambda1 * np.abs(self.B)) * self.Theta * 1e15
        ) / (denominator)

        self.Gamma = bernoulli(p).rvs()

        self.p = p
        
    def perform_gibbs(
        self,
        iterations: int = None,
        scale: bool = False,
        store: bool = True,
        plot: bool = True,
        file_path: str = None,
        tp: str='B'
    ):
        """Perform Gibbs sampling for the specified number of iterations.

        Args:
            iterations (int, optional): Number of iterations of the gibbs sampler. Defaults to None.
            store (bool, optional): Boolean to store or not the parameters. Defaults to True.
            plot (bool, optional): Boolean to plot or not. Defaults to True.
            file_path (str, optional): File path for storaga. Defaults to None.
        """
        if iterations:
            self.num_iters = iterations

        with tqdm(total=self.num_iters, desc="Gibbs Sampling", unit="iter") as pbar:
            
            for i in range(self.num_iters):
                
                self.sample_factors() #Update Omega
                if tp=='B':
                    self.sample_features_allocation_B()
                else:
                    self.sample_features_allocation() #Update Gamma
                self.sample_features_sparsity() #Update Theta
                self.sample_loadings() #Update B
                # print(self.Q)
                self.sample_diag_covariance() #Update Sigma

                if scale:
                    self.scale_group()

                if store:
                    self.paths[i] = {
                        "B": self.B,
                        "Omega": self.Omega,
                        "Sigma": self.Sigma,
                        "Gamma": self.Gamma,
                        "Theta": self.Theta,
                        "Q": self.Q,
                    }

                # Update the progress bar
                pbar.update(1)

                if i % (self.num_iters // 10) == 0 or i < 10:
                    print(f"Check Theta: {self.Theta}")
                    
                    # print(f"Check Sigma: {self.Sigma}")
                   

        if plot:
            self.plot_heatmaps()
            self.plot_heatmaps(str_param="Q")

        if file_path is not None:
            with open(file_path, "w") as json_file:
                json.dump(self.paths, json_file)