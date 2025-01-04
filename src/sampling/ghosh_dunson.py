import numpy as np

from typing import Tuple
from scipy.stats import norm
from joblib import Parallel, delayed

from src.sampling.sparse_factor_gibbs import SpSlFactorGibbs, parallelized_loading
from src.sampling.metropolis_hastings import metropolis_hastings
from src.utils.setup.orthonormal_setup import compute_projection_norm


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


    def initialize_factors(self):
        """
        w_i ~N(0, I) is presribed in this model. w_i are the columns of Omega (Kxn)
        """
        self.Omega= np.random.normal(size=(self.num_factor, self.num_obs)).astype(self.dtype)
        
    def sample_factors(self, epsilon: float = 1e-10):
        """
        Sample Omega from a normal distribution
        """
        self.Omega = np.random.normal(size=(self.num_factor, self.num_obs)).astype(self.dtype)
        
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
        Q = parallelized_loading(
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
        self.B = Q * r
        
        