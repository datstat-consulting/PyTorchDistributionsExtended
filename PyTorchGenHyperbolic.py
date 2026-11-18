"""Generalized hyperbolic laws using the SciPy numerical backend."""

import torch
from torch.distributions.utils import broadcast_all

from .numerical import ScipyContinuous


class GeneralizedHyperbolic(ScipyContinuous):
    """Use canonical alpha, beta, delta, mu and lambda parameters."""

    def __init__(self, alpha, beta, delta, mu, lam, validate_args=None):
        alpha, beta, delta, mu, lam = broadcast_all(alpha, beta, delta, mu, lam)
        if torch.any((delta <= 0) | (alpha <= beta.abs())):
            raise ValueError("Require delta > 0 and alpha > |beta|.")
        super().__init__("genhyperbolic", lam, alpha * delta, beta * delta, loc=mu,
                         scale=delta, validate_args=validate_args)
        self.alpha, self.beta, self.delta, self.mu, self.lam = alpha, beta, delta, mu, lam


class NormalInverseGaussian(ScipyContinuous):
    """Normal inverse Gaussian law in canonical parameters."""

    def __init__(self, alpha, beta, mu, delta, validate_args=None):
        alpha, beta, mu, delta = broadcast_all(alpha, beta, mu, delta)
        if torch.any((delta <= 0) | (alpha <= beta.abs())):
            raise ValueError("Require delta > 0 and alpha > |beta|.")
        super().__init__("norminvgauss", alpha * delta, beta * delta, loc=mu,
                         scale=delta, validate_args=validate_args)
        self.alpha, self.beta, self.mu, self.delta = alpha, beta, mu, delta


genhyperbolic = GeneralizedHyperbolic
norminvgauss = NormalInverseGaussian
