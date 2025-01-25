import numpy as np
from scipy import stats

#TODO change all truncated to rejection sampling


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
        return (
            stats.beta.pdf(x, alpha, beta)
            * (x >= a)
            * (x <= b)
            / (stats.beta.cdf(b, alpha, beta) - stats.beta.cdf(a, alpha, beta))
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
        truncated_cdf = (
            stats.beta.cdf(x, alpha, beta) - stats.beta.cdf(a, alpha, beta)
        ) / (stats.beta.cdf(b, alpha, beta) - stats.beta.cdf(a, alpha, beta))
        return (x > a) * (x < b) * truncated_cdf + (x >= b).astype(float)

    def _rvs(
        self, alpha: float, beta: float, a: float, b: float, size: int = 1
    ) -> np.ndarray:
        """Generate random samples from the truncated beta distribution.

        Args:
            alpha (float): shape parameter
            beta (float): shape parameter
            a (float): lower bound between 0 and 1
            b (float): upper bound between 0 and 1
            size (int): number of samples to generate

        Returns:
            np.ndarray: random samples from the truncated beta distribution
        """
        # Generate uniform samples on the truncated range [F(a), F(b)]
        cdf_a = stats.beta.cdf(a, alpha, beta)
        cdf_b = stats.beta.cdf(b, alpha, beta)
        u = np.random.uniform(cdf_a, cdf_b, size)

        # Map the uniform samples to the beta distribution using the PPF (inverse CDF)
        return stats.beta.ppf(u, alpha, beta)


class trunc_norm_mixture(stats.rv_continuous):
    def __init__(
        self, mu_pos: float, mu_neg: float, sigma: float, epsilon: float = 1e-10
    ):
        """_summary_

        Args:
            mu_pos (float): _description_
            mu_neg (float): _description_
            sigma (float): _description_
            epsilon (float, optional): _description_. Defaults to 1e-10.
        """
        super().__init__()
        # Initialize the parameters
        self.mu_pos = mu_pos
        self.mu_neg = mu_neg
        self.sigma = sigma

        # Precompute the Z and w values
        self.Z_pos = 1 - stats.norm.cdf(0, loc=self.mu_pos, scale=self.sigma)
        self.Z_neg = stats.norm.cdf(0, loc=self.mu_neg, scale=self.sigma)
        self.Z = self.Z_pos + self.Z_neg + epsilon
        self.w_pos = (self.Z_pos + epsilon / 2) / (self.Z)
        self.w_neg = (self.Z_neg + epsilon / 2) / (self.Z)

    def _pdf(self, x: float) -> float:

        return (
            np.where(
                x > 0,
                stats.norm.pdf(x, loc=self.mu_pos, scale=self.sigma),
                stats.norm.pdf(x, loc=self.mu_neg, scale=self.sigma),
            )
            / self.Z
        )

    def _cdf(self, x: float) -> float:

        if x < 0:
            return (
                self.w_neg
                * stats.norm.cdf(x, loc=self.mu_neg, scale=self.sigma)
                / self.Z_neg
            )
        else:
            return (
                self.w_neg
                + self.w_pos
                * (
                    stats.norm.cdf(x, loc=self.mu_pos, scale=self.sigma)
                    - stats.norm.cdf(0, loc=self.mu_pos, scale=self.sigma)
                )
                / self.Z_pos
            )

    def rvs(self, size=1):

        component = np.random.choice([1, -1], size=size, p=[self.w_pos, self.w_neg])
        samples = []

        for c in component:
            if c > 0:
                sample = stats.truncnorm.rvs(
                    a=-self.mu_pos / self.sigma,
                    b=np.inf,
                    loc=self.mu_pos,
                    scale=self.sigma,
                    size=1,
                )
            else:
                sample = stats.truncnorm.rvs(
                    a=-np.inf,
                    b=-self.mu_neg / self.sigma,
                    loc=self.mu_neg,
                    scale=self.sigma,
                    size=1,
                )
            samples.append(sample[0])

        return np.array(samples)
