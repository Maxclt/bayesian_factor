import numpy as np

from typing import Tuple
from scipy.stats import norm
from joblib import Parallel, delayed

from src.sampling.sparse_factor_gibbs import SpSlFactorGibbs
from src.sampling.metropolis_hastings import metropolis_hastings
from src.utils.setup.orthonormal_setup import compute_orthogonal_projection, compute_projection_norm


class OrthonormalFactorGibbs(SpSlFactorGibbs):
    """
    Implementation of the paper orthonormal decomposition of Omega to
    tackle the large s, small n magnitude issue.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the OrthonormalFactorGibbs class.

        Args:
            *args: Positional arguments passed to the parent class.
            **kwargs: Keyword arguments passed to the parent class.
        """
        super().__init__(*args, **kwargs)

    def initialize_factors(self):
        """
        Initialize the factor matrix Omega as a num_factors x num_obs matrix
        that lies on the sqrt(num_obs)-radius Stiefel manifold.

        Returns:
            np.ndarray: A num_factors x num_obs orthonormal matrix Omega scaled
                        by sqrt(num_obs).
        """
        # Get Haar distributed num_obs x num_obs orthogonal matrix
        Q, _ = np.linalg.qr(
            np.random.normal(size=(self.num_obs, self.num_obs)).astype(self.dtype)
        )
        # Keep only num_factors first rows
        Omega = Q[: self.num_factor, :] * np.sqrt(self.num_obs)

        return Omega

    def sample_factors(self, epsilon: float = 1e-10):
        """
        Update the factor matrix Omega using Gibbs sampling for each row.

        Args:
            epsilon (float, optional): Small value to avoid division by zero. Defaults to 1e-10.
        """

        def process_k(k):
            """Sample the k-th row of Omega."""
            mean, var = self.compute_Omega_moments(k)
            d = self.sample_d_metropolis(k, mean, var)
            Omega_k = self.sample_from_sphere(d, mean, k, epsilon)
            return k, Omega_k

        # Parallel computation using joblib for faster processing
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
        """
        Compute the conditional mean and variance of the k-th row of Omega.

        Args:
            k (int): Index of the row to compute moments for.
            epsilon (float, optional): Small value to avoid division by zero. Defaults to 1e-10.

        Returns:
            Tuple[np.ndarray, float]: The conditional mean (np.ndarray of shape (num_obs,))
                                      and variance (float) for the k-th row.
        """
        # Compute variance = (B_k^T Σ^-1 B_k)⁻1
        var = 1 / max(self.B[:, k].T @ np.diag(1 / self.Sigma) @ self.B[:, k], epsilon)

        # Compute mean = (B_k^T Σ^-1 B_k)⁻1 * (Y - sum_{t != k} B_t @ Omega_t.T)^T @ Σ^-1 @ B_k
        B_excluded = np.delete(self.B, k, axis=1)  # Shape (G, K-1)
        Omega_excluded = np.delete(self.Omega, k, axis=0)  # Shape (K-1, N)
        excluded_sum = np.einsum(
            "gt,tn->gn", B_excluded, Omega_excluded
        )  # Shape (G, N)
        residual = (self.Y - excluded_sum).T  # Shape (N, G)
        mean = var * residual @ np.diag(1 / self.Sigma) @ self.B[:, k]  # Shape: (N, G)

        return mean, var

    def sample_d_metropolis(self, k, mean, var, rw_scale: float = 0.1) -> float:
        """
        Sample the distance `d` from its marginal distribution using the Metropolis-Hastings algorithm.

        Args:
            k (int): Index of the current row being updated.
            mean (np.ndarray): The conditional mean of the k-th row.
            var (float): The conditional variance of the k-th row.
            rw_scale (float, optional): Proposal scale for random-walk Metropolis-Hastings. Defaults to 0.1.

        Returns:
            float: A sample of `d` from its marginal distribution.
        """
        Omega_excluded = np.delete(self.Omega, k, axis=0).T
        proj = compute_projection_norm(Omega_excluded, mean)

        def target_pdf(d: float) -> float:
            """Target distribution for Metropolis-Hastings."""
            exp_term = np.clip(np.linalg.norm(proj) * d / var, a_min=-100, a_max=100)
            return (self.num_obs - d**2) ** (
                (self.num_obs - self.num_factor - 2) / 2
            ) * np.exp(exp_term)

        def proposal_sampler(d: float) -> float:
            """Proposal distribution for Metropolis-Hastings."""
            return d + rw_scale * norm().rvs()

        # Initialize d within the valid range [-sqrt(num_obs), sqrt(num_obs)]
        d = np.random.uniform(-np.sqrt(self.num_obs), np.sqrt(self.num_obs))

        return metropolis_hastings(
            target_pdf=target_pdf, proposal_sampler=proposal_sampler, initial_state=d
        )

    def sample_from_sphere(self, d, mean, k, epsilon: float = 1e-10):
        """
        Sample a point from the intersection of a sphere and a hyperplane.

        The sphere lies in the orthogonal complement of the space spanned by
        np.delete(self.Omega, k, axis=0). The hyperplane is defined by its distance `d`
        from the origin and normal vector `mean`.

        Args:
            d (float): Distance of the hyperplane from the origin.
            mean (np.ndarray): Normal vector defining the hyperplane.
            k (int): Index of the row being sampled.
            epsilon (float, optional): Small value to avoid division by zero. Defaults to 1e-10.

        Returns:
            np.ndarray: A sampled point on the intersection of the sphere and hyperplane.
        """
        # Get subspace of all other factors
        Omega_minus_k = np.delete(self.Omega, k, axis=0).T  # Shape (N, K-1)
        X = np.random.randn(self.num_obs).astype(self.dtype)
        X_unit_sphere = X / np.linalg.norm(X)

        # Unit sphere in the space orthogonal to Omega_minus_k (N-K+1)
        X_orthogonal = compute_orthogonal_projection(Omega_minus_k, X_unit_sphere)
        
        # Compute the component of X_orthogonal orthogonal to mean
        X_hyperplane = compute_orthogonal_projection(mean, X_orthogonal)

        lambda_ = np.linalg.norm(X_hyperplane)
        X_hyperplane_sphere = np.sqrt(self.num_obs - d**2) * X_hyperplane / lambda_  # Normalize to sphere of radius sqrt(n-d^2)
        
        # Shift by d along mean
        X_hyperplane_sphere_shift = X_hyperplane_sphere + (
            d * mean / max(np.linalg.norm(mean), epsilon)
        )
        return X_hyperplane_sphere_shift

