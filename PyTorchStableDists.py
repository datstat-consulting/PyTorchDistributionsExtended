"""Stable laws and compatibility constructors."""

from torch.distributions import Cauchy, Normal, Pareto, StudentT

from .native import Levy, Logistic
from .numerical import ScipyContinuous
from .PyTorchExponential import expon


class StableDistribution(ScipyContinuous):
    def __init__(self, alpha, beta, mu=0., sigma=1., validate_args=None):
        super().__init__("levy_stable", alpha, beta, loc=mu, scale=sigma,
                         validate_args=validate_args)
        self.alpha, self.beta = self._shape_parameters
        self.mu, self.sigma = self.loc, self.scale


class SymmetricStable(StableDistribution):
    def __init__(self, alpha, mu=0., sigma=1., validate_args=None):
        super().__init__(alpha, 0., mu, sigma, validate_args)


levy = Levy


def LevySkew(mu, sigma):
    return StableDistribution(0.5, 1., mu, sigma)


def norm(mu=0., sigma=1.):
    return Normal(mu, sigma)


def cauchy(mu=0., sigma=1.):
    return Cauchy(mu, sigma)


def pareto(alpha, xm):
    return Pareto(xm, alpha)


def logistic(mu=0., sigma=1.):
    return Logistic(mu, sigma)


def t(nu, mu=0., sigma=1.):
    return StudentT(nu, mu, sigma)
