"""Circular distribution compatibility constructors."""

from torch.distributions import Distribution, VonMises

from .native import WrappedCauchy

CircularDistribution = Distribution


def vonmises(mu, kappa):
    return VonMises(mu, kappa)


def wrapcauchy(mu, kappa):
    return WrappedCauchy(mu, kappa)
