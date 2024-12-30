import numpy as np

from typing import Tuple
from scipy.stats import norm
from joblib import Parallel, delayed

from src.sampling.sparse_factor_gibbs import SpSlFactorGibbs
from src.sampling.metropolis_hastings import metropolis_hastings
from src.utils.setup.orthonormal_setup import compute_projection_norm


class OrthonormalFactorGibbs(SpSlFactorGibbs):
    """
    Implementation of the paper orthonormal decomposition of omega to
    tackle the large s, small n magnitude issue.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

    def initialize_factors(self):
        """
        Initialize an num_factors * num_obs matrix Omega on the sqrt(num_obs)-radius Stiefel manifold.
        """
        # Get Haar distributed num_obs * num_obs orthogonal matrix
        Q, _ = np.linalg.qr(
            np.random.normal(size=(self.num_obs, self.num_obs)).astype(self.dtype)
        )
        # Keep only num_factors first rows
        self.Omega = Q[: self.num_factor, :] * np.sqrt(self.num_obs)

    def sample_factors(self, epsilon: float = 1e-10):

        def process_k(k):
            mean, var = self.compute_Omega_moments(k)
            d = self.sample_d_metropolis(k, mean, var)
            Omega_k = self.sample_from_sphere(d, mean, epsilon)
            return k, Omega_k

        # Parallel computation using joblib
        results = Parallel(n_jobs=-1)(
            delayed(process_k)(k) for k in range(self.num_factor)
        )

        # Initialize new Omega
        new_Omega = np.zeros_like(self.Omega, dtype=self.dtype)

        # Update Omega with results
        for k, Omega_k in results:
            new_Omega[k, :] = Omega_k

        self.Omega = new_Omega

    def compute_Omega_moments(
        self, k: int, epsilon: float = 1e-10
    ) -> Tuple[np.ndarray, float]:
        """Compute mean and covariance of the k-th row of Omega.

        Args:
            k (int): _description_
            epsilon (float, optional): _description_. Defaults to 1e-10.

        Returns:
            Tuple[np.ndarray, float]: _description_
        """
        # Compute variance = (B_k^T Σ^-1 B_k)⁻1
        var = 1 / max(self.B[:, k].T @ np.diag(1 / self.Sigma) @ self.B[:, k], epsilon)

        # Compute mean = (B_k^T Σ^-1 B_k)⁻1 * (Y - sum_{t != k} B_t @ Omega_t.T)^T @ Sigma_inv @ B_k
        B_excluded = np.delete(self.B, k, axis=1)  # Shape (G, K-1)
        Omega_excluded = np.delete(self.Omega, k, axis=0)  # Shape (K-1, N)
        excluded_sum = np.einsum(
            "gt,tn->gn", B_excluded, Omega_excluded
        )  # Shape (G, N)
        residual = (self.Y - excluded_sum).T  # Shape (N, G)
        mean = var * residual @ np.diag(1 / self.Sigma) @ self.B[:, k]  # Shape: (N, G)

        return mean, var

    def sample_d_metropolis(self, k, mean, var, rw_scale: float = 0.1) -> float:

        Omega_excluded = np.delete(self.Omega, k, axis=0).T
        proj = compute_projection_norm(Omega_excluded, mean)

        def target_pdf(d: float) -> float:
            return (self.num_obs - d**2) ** (
                (self.num_obs - self.num_factor - 2) / 2
            ) * np.exp(np.linalg.norm(proj) * d / var**2)

        def proposal_sampler(d: float) -> float:
            return d + rw_scale * norm().rvs()

        # Initialize d, proposal scale
        d = np.random.uniform(-np.sqrt(self.num_obs), np.sqrt(self.num_obs))

        return metropolis_hastings(
            target_pdf=target_pdf, proposal_sampler=proposal_sampler, initial_state=d
        )

    def sample_from_sphere(self, d, mean, epsilon: float = 1e-10):
        """
        First sample from the sphere in the plane orthogonal to the last vector
        of the base. Then translate it into the plane orthogonal to
        mean
        """
        # Generate n-1 samples from N(0, 1)
        X = np.random.randn(self.num_obs - 1).astype(self.dtype)
        # Normalize to lie on the sphere of radius sqrt(num_obs)
        lambda_ = np.linalg.norm(X)
        X_prime = np.sqrt(self.num_obs) * X / lambda_
        # Translate to land in the orthogonal plane to mean
        # that is at a distance d of the center
        X_hyperplane = np.append(X_prime, 0) + (
            d * mean / max(np.linalg.norm(mean), epsilon)
        )
        return X_hyperplane
