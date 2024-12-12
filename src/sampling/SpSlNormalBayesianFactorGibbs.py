import numpy as np
from scipy.stats import norm, bernoulli, invgamma
import matplotlib as plt

from src.utils.probability.density import truncated_beta


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
            Y (np.array): _description_
            B (np.array): _description_
            Omega (np.array): _description_
            Sigma (np.array): _description_
            Gamma (np.array): _description_
            Theta (np.array): _description_
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
        self.num_obs, self.num_var = Y.shape
        self.num_factor = B.shape[0]

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

    def sample_loading(j: int, k: int) -> float:

        return False

    def sample_loadings():
        return False
