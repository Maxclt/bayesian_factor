import numpy as np

from src.sampling.sparse_normal_bayesian_factor_gibbs import SpSlNormalBayesianFactorGibbs
from src.utils.setup.orthonormal_setup import compute_projection_norm

class OrthonormalFactorGibbs(SpSlNormalBayesianFactorGibbs):
    """
    Implementation of the paper orthonormal decomposition of omega to
    tackle the large s, small n magnitude issue.
    """
    def __init__(
        self,
        Y: np.array,
        B: np.array,
        Sigma: np.array,
        Gamma: np.array,
        Theta: np.array,
        alpha: float,
        eta: float,
        epsilon: float,
        lambda0: float,
        lambda1: float,
        burn_in: int = 50,
    ):
        super().__init__(
            Y, B, Sigma, Gamma, Theta, alpha, eta, epsilon, lambda0, lambda1, burn_in,
        )

        # Initialize Omega
        self.initialize_orthonormal_factors()

    def initialize_orthonormal_factors(self):
        """
        Initialize an num_factors * num_obs matrix Omega on the sqrt(num_obs)-radius Stiefel manifold.
        """
        # Get Haar distributed num_obs * num_obs orthogonal matrix
        Q, _ = np.linalg.qr(np.random.normal(size=(self.num_obs, self.num_obs)))
        # Keep only num_factors first rows
        self.Omega = Q[:self.num_factor, :] * np.sqrt(self.num_obs)

    def sample_factors(self, store, get: bool = False, epsilon: float = 1e-10):
        new_Omega = np.zeros_like(self.Omega) # Will update the rows at the end
        for k in range(self.num_factor):
            B_k = self.B[:, k]

            # Compute (B_k^T Σ^-1 B_k)⁻1
            bar_sigma_k = 1 / (B_k.T @  np.diag(1 / self.Sigma) @ B_k + epsilon)
                
            # Compute (B_k^T Σ^-1 B_k)⁻1 * (Y - sum_{t != k} B_t @ Omega_t.T)^T @ Sigma_inv @ B_k
            bar_Omega_k = bar_sigma_k * self.compute_expression(k)    

            # Sample d using Metropolis algorithm
            d = self.sample_d_metropolis(k, bar_Omega_k, bar_sigma_k)

            # Sample from sphere S_d
            Omega_k = self.sample_from_sphere(d, bar_Omega_k, epsilon)

            # Update Omega_k
            new_Omega[k, :] = Omega_k

        # Update Omega
        self.Omega = new_Omega

        # Store sampled Omega
        if store:
            self.paths["Omega"].append(self.Omega)
        if get:
            return self.Omega
        

    def compute_expression(self, k):
        """
        Compute (Y - sum_{t != k} B_t @ Omega_t.T)^T @ Sigma_inv @ B_k.
        """
        sum_B_Omega = np.zeros((self.num_var, self.num_obs))  # Initialize the sum to a zero matrix
        for t in range(self.B.shape[1]):
            if t != k:  # Exclude the k-th term
                sum_B_Omega += np.expand_dims(self.B[:, t], 1) @ np.expand_dims(self.Omega[t].T,0)
        # Step 2: Compute the residual matrix (Y - sum_t B_t @ Omega_t.T)
        residual = self.Y - sum_B_Omega  # Shape: (n, m)

        # Step 3: Pre-multiply by Sigma_inv
        temp = np.diag(1 / self.Sigma) @ residual  # Shape: (n, m)

        # Step 4: Multiply by B_k
        result = temp.T @ self.B[:, k]  # Shape: (m, r)

        return result
        
        

    def sample_d_metropolis(self, k, bar_Omega_k, bar_sigma_k):
        def log_prob(d, proj):
            return ((self.num_obs - d**2)**((self.num_obs - self.num_factor - 2) / 2) *
                    np.exp(np.linalg.norm(proj) * d / bar_sigma_k**2))

        # Initialize d, proposal scale
        d = np.random.uniform(-np.sqrt(self.num_obs), np.sqrt(self.num_obs))
        proposal_scale = 0.1  # Adjust as needed

        # Remaining rows as vector columns (thats how I understand it)
        Omega_minus_k = np.delete(self.Omega, k, axis=0).T
        proj = compute_projection_norm(Omega_minus_k, bar_Omega_k)

        for _ in range(self.burn_in):
            d_proposal = d + np.random.normal(scale=proposal_scale)
            if abs(d_proposal) > np.sqrt(self.num_obs):
                continue  # Skip invalid proposals
            acceptance_ratio = log_prob(d_proposal, proj) / log_prob(d, proj)
            if np.random.uniform() < acceptance_ratio:
                d = d_proposal

        return d
    
    def sample_from_sphere(self, d, bar_Omega_k, epsilon: float = 1e-10):
        """
        First sample from the sphere in the plane orthogonal to the last vector
        of the base. Then translate it into the plane orthogonal to
        bar_Omega_k
        """
        # Generate n-1 samples from N(0, 1)
        X = np.random.randn(self.num_obs - 1)
        # Normalize to lie on the sphere of radius sqrt(num_obs)
        lambda_ = np.linalg.norm(X)
        X_prime = np.sqrt(self.num_obs) * X / lambda_
        # Translate to land in the orthogonal plane to bar_Omega_k
        # that is at a distance d of the center
        X_hyperplane = np.append(X_prime, 0) + (
            d * bar_Omega_k / (np.linalg.norm(bar_Omega_k) + epsilon)
        )
        return X_hyperplane
    
    



