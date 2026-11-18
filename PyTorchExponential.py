"""Exponential and asymmetric Laplace compatibility constructors."""

from torch.distributions import Exponential

from .native import AsymmetricLaplace

expon = Exponential


def laplace_asymmetric(lambda1, lambda2):
    return AsymmetricLaplace(lambda1, lambda2)
