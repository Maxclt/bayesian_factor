import numpy as np
from scipy import stats


class truncated_beta(stats.rv_continuous):
    def _pdf(self, x, alpha, beta, a, b):
        return stats.beta.pdf(x, alpha, beta) / (
            stats.beta.cdf(b, alpha, beta) - stats.beta.cdf(a, alpha, beta)
        )

    def _cdf(self, x, alpha, beta, a, b):
        return (stats.beta.cdf(x, alpha, beta) - stats.beta.cdf(a, alpha, beta)) / (
            stats.beta.cdf(b, alpha, beta) - stats.beta.cdf(a, alpha, beta)
        )
