import numpy as np
from scipy.stats import norm, bernoulli, invgamma, multivariate_normal
import matplotlib as plt
import itertools

from src.utils.probability.density import truncated_beta, trunc_norm_mixture


class SpSlNormalBayesianFactorGibbs:
    def __init__(
        self,
        Y: np.array,
        B: np.array,
        Omega: np.array,
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
        """_summary_

        Args:
            Y (np.array): Observed Data (G x n)
            B (np.array): Loadings Matrix (G x K)
            Omega (np.array): Factors Matrix (K x n)
            Sigma (np.array): Diagonal Covariance Matrix, represented as G-dimensionnal array
            Gamma (np.array): Feature Allocation Matrix (G x K)
            Theta (np.array): Feature Sparsity Vector, K-dimensionnal array
            alpha (float): _description_
            eta (float): _description_
            epsilon (float): _description_
            lambda0 (float): _description_
            lambda1 (float): _description_
            burn_in (int, optional): _description_. Defaults to 50.
        """

        # Data
        self.Y = Y

        # Parameters
        self.B = B
        self.Omega = Omega
        self.Sigma = Sigma
        self.Gamma = Gamma
        self.Theta = Theta

        # Shapes
        self.num_var, self.num_obs = Y.shape
        self.num_factor = B.shape[1]

        # Hyperparameters
        self.alpha = alpha
        self.eta = eta
        self.epsilon = epsilon
        self.lamba0 = lambda0
        self.lamba1 = lambda1

        # Gibbs Settings
        self.burn_in = burn_in

        # Trajectories
        self.B_path = [B]
        self.Omega_path = [Omega]
        self.Sigma_path = [Sigma]
        self.Gamma_path = [Gamma]
        self.Theta_path = [Theta]

    def perform_gibbs_sampling(self, iterations: int = False):
        """_summary_

        Args:
            iterations (int, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        if not iterations:
            num_iters = self.burn_in
        else:
            num_iters = iterations

        # Plot the initial set up
        self.plot_points("Initial Random Assignments")

        # Run for the given number of iterations
        for i in range(num_iters):
            self.calc_dist_log_prob()
            self.sample_mixture_locations()
            self.sample_mixture_assignments()

        # Plot the final mixture assignments
        self.plot_points("Final Mixture Assignments")

        # Final Plot of the log probability as a function of iteration
        self.plot_prob()

        return self.u_locations, self.Y

    def sample_loading(self, j: int, k: int, product: float) -> float:

        # Compute ajk (∑_{i} Omega[k, i] ** 2 / (2 * Sigma[j]))
        ajk = np.sum(self.Omega[k, :] ** 2) / (2 * self.Sigma[j])

        # Compute bjk (∑_{i} Omega[k, i] * (Y[j, i] - ∑_{l ≠ k} B[j, l] * Omega[l, i]))
        adjusted_y = self.Y[j, :] - product + self.B[j, k] * self.Omega[k, :]
        bjk = np.dot(self.Omega[k, :], adjusted_y)

        # Compute cjk (lambda1 * Gamma[j,k] + lambda0 * (1 - Gamma[j,k]))
        cjk = self.lamba1 * self.Gamma[j, k] + self.lamba0 * (1 - self.Gamma[j, k])

        # Compute truncated normal mixture parameters
        mu_pos = (bjk - cjk) / (2 * ajk)
        mu_neg = (bjk + cjk) / (2 * ajk)
        sigma = np.sqrt(1 / (2 * ajk))

        # Sample from truncated normal mixture
        return trunc_norm_mixture(mu_pos=mu_pos, mu_neg=mu_neg, sigma=sigma).rvs()[0]

    def sample_loadings(self, get: bool = False):
        for j in range(self.num_var):
            product = np.dot(self.B[j, :], self.Omega)
            for k in range(self.num_factor):
                self.B[j, k] = self.sample_loading(j, k, product)

        if get:
            return self.B

    def sample_factors(self, get: bool = False):

        # Compute (I_K + B^T Σ^-1 B)^-1
        Z = self.B.T @ np.diag(1 / self.Sigma)
        A = np.linalg.inv(np.eye(self.num_factor) + Z @ self.B)

        for i in range(self.num_obs):
            self.Omega[:, i] = multivariate_normal.rvs(mean=A @ Z, cov=A)

        if get:
            return self.Omega

    def sample_features_allocation(self, get: bool = False):
        for j, k in itertools.product(range(self.num_var), range(self.num_factor)):
            p = (
                self.lamba1
                * np.exp(-self.lamba1 * np.abs(self.B[j, k]))
                * self.Theta[k]
            ) / (
                (
                    self.lamba0
                    * np.exp(-self.lamba0 * np.abs(self.B[j, k]))
                    * (1 - self.Theta[k])
                    + self.lamba1
                    * np.exp(-self.lamba1 * np.abs(self.B[j, k]))
                    * self.Theta[k]
                )
            )
            self.Gamma[j, k] = bernoulli(p).rvs()

        if get:
            return self.Gamma

    def sample_features_sparsity(self, get: bool = False):
        for k in range(self.num_factor):
            alphak = np.sum(self.Gamma[:, k]) + self.alpha * (
                k == (self.num_factor - 1)
            )
            betak = np.sum(self.Gamma[:, k] == 0) + 1

            if k == 0:
                self.Theta[k] = truncated_beta._rvs(alphak, betak, self.Theta[k + 1], 1)
            elif k == (self.num_factor - 1):
                self.Theta[k] = truncated_beta._rvs(alphak, betak, 0, self.Theta[k - 1])
            else:
                self.Theta[k] = truncated_beta._rvs(
                    alphak, betak, self.Theta[k + 1], self.Theta[k - 1]
                )
        if get:
            return self.Theta

    def sample_diag_covariance(self, get: bool=False):
        