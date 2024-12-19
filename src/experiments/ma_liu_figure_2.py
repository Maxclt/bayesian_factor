import torch

from src.simulations.normal_bayesian_factor_dgp import NormalBayesianFactorDGP
from src.sampling.sparse_normal_factor_gibbs import (
    SpSlNormalBayesianFactorGibbs,
)
from src.utils.setup.create_true_loadings import create_true_loadings

# TODO modify this to have the possibility to set the parameter when running the file

# Force Random Seed
# np.random.seed(42)

# Normal Factor Bayesian Dimensions
num_sim_1 = 100
num_sim_2 = 1000
num_variables = 1956
num_factors = 8

# True Loadings Settings
block_size = 500
overlap = 136
random = False
mean = 1
std = 5

# Hyperparameters
alpha = 1 / num_variables
eta = 1
epsilon = 1
lambda0 = 20  # try with lambda greater
lambda1 = 0.1


# True Parameters
BTrue = create_true_loadings(
    num_factors=num_factors,
    num_variables=num_variables,
    block_size=block_size,
    overlap=overlap,
    random=random,
    mean=mean,
    std=std,
)

SigmaTrue = torch.ones(
    num_variables
)  # TODO define a function to create_true_covariance either random or not

# Initial Latent Parameters
Gamma0 = create_true_loadings(
    num_factors=num_factors,
    num_variables=num_variables,
    block_size=block_size,
    overlap=overlap,
)

Theta0 = torch.full((num_factors,), 0.5)

# Simulated Value for Y
DataGeneratingProcess = NormalBayesianFactorDGP(B=BTrue, Sigma=SigmaTrue)

Y_sim_1 = DataGeneratingProcess.simulate(size=num_sim_1)
Y_sim_2 = DataGeneratingProcess.simulate(size=num_sim_2)

# This block ensures multiprocessing works on Windows/macOS
if __name__ == "__main__":

    # Initiate Bayesian Normal Factor Gibbs Sampler
    SparseGibbsSampling1 = SpSlNormalBayesianFactorGibbs(
        Y=Y_sim_1,
        B=BTrue,
        Sigma=SigmaTrue,
        Gamma=Gamma0,
        Theta=Theta0,
        alpha=alpha,
        eta=eta,
        epsilon=epsilon,
        lambda0=lambda0,
        lambda1=lambda1,
        device="cpu",  # You can change it to "cuda" if running on GPU
    )

    # Perform Gibbs Sampler for posterior
    SparseGibbsSampling1.perform_gibbs_sampling(iterations=1000, plot=False)
    B11_path_sim_1 = SparseGibbsSampling1.get_trajectory(
        param="B", coeff=(1, 1), abs_value=True
    )

    SparseGibbsSampling2 = SpSlNormalBayesianFactorGibbs(
        Y=Y_sim_2,
        B=BTrue,
        Sigma=SigmaTrue,
        Gamma=Gamma0,
        Theta=Theta0,
        alpha=alpha,
        eta=eta,
        epsilon=epsilon,
        lambda0=lambda0,
        lambda1=lambda1,
        device="cpu",  # You can change it to "cuda" if running on GPU
    )

    SparseGibbsSampling2.perform_gibbs_sampling(iterations=1000, plot=False)
    B11_path_sim_2 = SparseGibbsSampling2.get_trajectory(
        param="B", coeff=(1, 1), abs_value=True
    )
