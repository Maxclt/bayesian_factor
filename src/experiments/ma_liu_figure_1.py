import numpy as np

from src.simulations.normal_bayesian_factor_dgp import NormalBayesianFactorDGP
from src.sampling.sparse_normal_bayesian_factor_gibbs import (
    SpSlNormalBayesianFactorGibbs,
)
from src.utils.setup.create_true_loadings import create_true_loadings

# TODO modify this to have the possibility to set the parameter when running the file

# Force Random Seed
# np.random.seed(42)

# Normal Factor Bayesian Dimensions
num_sim = 100
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

SigmaTrue = np.ones(
    num_variables
)  # TODO define a function to create_true_covariance either random or not

# Initial Latent Parameters
Gamma0 = create_true_loadings(
    num_factors=num_factors,
    num_variables=num_variables,
    block_size=block_size,
    overlap=overlap,
)

Theta0 = np.full(num_factors, 0.5)

# Simulated Value for Y
DataGeneratingProcess = NormalBayesianFactorDGP(B=BTrue, Sigma=SigmaTrue)

Y_sim = DataGeneratingProcess.simulate(size=num_sim)

# Initiate Bayesian Normal Factor Gibbs Sampler
SparseGibbsSampling = SpSlNormalBayesianFactorGibbs(
    Y=Y_sim,
    B=BTrue,
    Sigma=SigmaTrue,
    Gamma=Gamma0,
    Theta=Theta0,
    alpha=alpha,
    eta=eta,
    epsilon=epsilon,
    lambda0=lambda0,
    lambda1=lambda1,
    fast=False,
)

# Perform Gibbs Sampler for posterior
SparseGibbsSampling.perform_gibbs_sampling(iterations=100)
