import numpy as np

from src.simulations.normal_bayesian_factor_dgp import NormalBayesianFactorDGP
from src.sampling.sparse_normal_bayesian_factor_gibbs import (
    SpSlNormalBayesianFactorGibbs,
)
from src.utils.setup.create_true_loadings import create_true_loadings

# TODO modify this to have the possibility to set the parameter when running the file

# Force Random Seed
seed = 42

if not seed == False:
    np.random.seed(seed)

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

BTrue = create_true_loadings(
    num_factors=num_factors,
    num_variables=num_variables,
    block_size=block_size,
    overlap=overlap,
    random=random,
    mean=mean,
    std=std,
)

SigmaTrue = np.eye(
    num_variables
)  # TODO define a function to create_true_covariance either random or not

DataGeneratingProcess = NormalBayesianFactorDGP(B=BTrue, Sigma=SigmaTrue)

Y_sim = DataGeneratingProcess.simulate(size=num_sim)
