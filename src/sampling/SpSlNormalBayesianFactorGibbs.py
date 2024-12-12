import numpy as np
from scipy.stats import norm, bernoulli, invgamma
import matplotlib as plt

from src.utils.probability.density import truncated_beta


class SpSlNormalBayesianFactorGibbs:
    def __init__(
        self, Y, num_factor, alpha, eta, epsilon, lambda_0, lambda_1, burn_in=50
    ):
        self.Y = Y
