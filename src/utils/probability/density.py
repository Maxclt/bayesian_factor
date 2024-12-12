import numpy as np
from scipy import stats


class truncated_beta(stats.rv_continuous):
    """Contains methods to sample from a truncated beta distribution

    Args:
        stats (_type_): scipy.stats class
    """

    def _pdf(self, x: float, alpha: float, beta: float, a: float, b: float) -> float:
        """Return the value of a truncated beta(alpha, beta) density function
        of support [a,b] at x

        Args:
            x (float): float between 0 and 1
            alpha (float): shape parameter
            beta (float): shape parameter
            a (float): lower bound between 0 and 1
            b (float): upper bound between 0 and 1

        Returns:
            float: value of a truncated beta density function at x
        """
        return stats.beta.pdf(x, alpha, beta) / (
            stats.beta.cdf(b, alpha, beta) - stats.beta.cdf(a, alpha, beta)
        )

    def _cdf(self, x, alpha, beta, a, b) -> float:
        """Return the value of a truncated beta(alpha, beta) cumulative
        distribution function of support [a,b] at x

        Args:
            x (float): float between 0 and 1
            alpha (float): shape parameter
            beta (float): shape parameter
            a (float): lower bound between 0 and 1
            b (float): upper bound between 0 and 1

        Returns:
            float: value of a truncated beta cumulative distribution function
            at x
        """
        return (stats.beta.cdf(x, alpha, beta) - stats.beta.cdf(a, alpha, beta)) / (
            stats.beta.cdf(b, alpha, beta) - stats.beta.cdf(a, alpha, beta)
        )
