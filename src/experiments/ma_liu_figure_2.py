import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.simulations.normal_bayesian_factor_dgp import NormalBayesianFactorDGP
from src.sampling.normal_factor_gibbs import SpSlNormalFactorGibbs
from src.utils.setup.create_true_loadings import create_true_loadings

# TODO modify this to have the possibility to set the parameter when running the file

# Force Random Seed
# np.random.seed(42)

# Normal Factor Bayesian Dimensions
num_sim = 100  # small = 100; big = 1000
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
lambda0 = 20


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

Theta0 = np.full((num_factors,), 0.5)

# Simulated Value for Y
DataGeneratingProcess = NormalBayesianFactorDGP(B=BTrue, Sigma=SigmaTrue)

Y_sim = DataGeneratingProcess.simulate(size=num_sim)

# This block ensures multiprocessing works on Windows/macOS
if __name__ == "__main__":

    # Initiate Bayesian Normal Factor Gibbs Sampler
    SparseGibbsSampling1 = SpSlNormalFactorGibbs(
        Y=Y_sim,
        B=BTrue,
        Sigma=SigmaTrue,
        Gamma=Gamma0,
        Theta=Theta0,
        alpha=alpha,
        eta=eta,
        epsilon=epsilon,
        lambda0=lambda0,
        lambda1=0.001,
        dtype=np.float32,
    )

    # Perform Gibbs Sampler for posterior
    SparseGibbsSampling1.perform_gibbs(iterations=1000, plot=False)
    B11_path_small = SparseGibbsSampling1.get_path()

    SparseGibbsSampling2 = SpSlNormalFactorGibbs(
        Y=Y_sim,
        B=BTrue,
        Sigma=SigmaTrue,
        Gamma=Gamma0,
        Theta=Theta0,
        alpha=alpha,
        eta=eta,
        epsilon=epsilon,
        lambda0=lambda0,
        lambda1=0.1,
        dtype=np.float32,
    )

    SparseGibbsSampling2.perform_gibbs(iterations=1000, plot=False)
    B11_path_big = SparseGibbsSampling2.get_path()


# Prepare data for seaborn
data = pd.DataFrame(
    {
        "Index": range(len(B11_path_small)),  # X-axis values
        "small": np.log(B11_path_small),
        "big": np.log(B11_path_big),
    }
)

# Create the plot
sns.set_theme(style="whitegrid")  # Set the style
plt.figure(figsize=(8, 6))
sns.lineplot(data=data, x="Index", y="small", label=r"$\lambda_1=0.001$")
sns.lineplot(data=data, x="Index", y="big", label=r"$\lambda_1=0.1$")

# Customize the plot
plt.title(r"Magnitude Inflation with $n = 100$", fontsize=14)
plt.xlabel("Iter")
plt.ylabel(r"$\log(|\beta_{00}|)$")
plt.legend()
plt.show()
