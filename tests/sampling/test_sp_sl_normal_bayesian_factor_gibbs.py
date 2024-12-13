import numpy as np
import pytest

from src.sampling.SpSlNormalBayesianFactorGibbs import SpSlNormalBayesianFactorGibbs


def test_gibbs_sampler():
    # Sample data
    Y = np.random.randn(5, 10)  # 5 variables, 10 observations
    B = np.random.randn(5, 3)  # 5 variables, 3 factors
    Omega = np.random.randn(3, 10)  # 3 factors, 10 observations
    Sigma = np.ones(5)  # Diagonal covariance for 5 variables
    Gamma = np.random.randint(0, 2, size=(5, 3))  # Binary allocation matrix
    Theta = np.random.rand(3)  # Feature sparsity vector for 3 factors

    # Hyperparameters
    alpha = 1.0
    eta = 1.0
    epsilon = 1.0
    lambda0 = 20
    lambda1 = 0.01
    burn_in = 50

    # Instantiate the Gibbs sampler
    model = SpSlNormalBayesianFactorGibbs(
        Y, B, Omega, Sigma, Gamma, Theta, alpha, eta, epsilon, lambda0, lambda1, burn_in
    )

    # Perform Gibbs sampling
    B, Omega, Sigma, Gamma, Theta = model.perform_gibbs_sampling(iterations=10)

    # Check that parameters are updated after sampling
    assert B.shape == (5, 3), "B has incorrect shape"
    assert Omega.shape == (3, 10), "Omega has incorrect shape"
    assert Sigma.shape == (5,), "Sigma has incorrect shape"
    assert Gamma.shape == (5, 3), "Gamma has incorrect shape"
    assert Theta.shape == (3,), "Theta has incorrect shape"

    # Check that paths are being stored
    assert len(model.B_path) == 11, "B path length incorrect"
    assert len(model.Omega_path) == 11, "Omega path length incorrect"
    assert len(model.Sigma_path) == 11, "Sigma path length incorrect"
    assert len(model.Gamma_path) == 11, "Gamma path length incorrect"
    assert len(model.Theta_path) == 11, "Theta path length incorrect"


# Run the test
test_gibbs_sampler()
