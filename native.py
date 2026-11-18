"""Additional torch-native continuous distributions."""

import math

import torch
from torch.distributions import Distribution, constraints, utils
from torch.nn.functional import softplus


class _ScalarDistribution(Distribution):
    """Broadcast parameters and preserve them when expanding a batch."""

    def _initialize(self, parameters, validate_args):
        self._parameter_names = tuple(parameters)
        values = utils.broadcast_all(*parameters.values())
        for name, value in zip(self._parameter_names, values):
            setattr(self, name, value)
        super().__init__(values[0].shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(type(self), _instance)
        new._parameter_names = self._parameter_names
        batch_shape = torch.Size(batch_shape)
        for name in self._parameter_names:
            setattr(new, name, getattr(self, name).expand(batch_shape))
        Distribution.__init__(new, batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def _tensor(self, value):
        reference = getattr(self, self._parameter_names[0])
        return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)

    def _uniform(self, sample_shape):
        reference = getattr(self, self._parameter_names[0])
        u = torch.rand(self._extended_shape(torch.Size(sample_shape)),
                       dtype=reference.dtype, device=reference.device)
        return u.clamp_min(torch.finfo(u.dtype).tiny)

    def pdf(self, value):
        return self.log_prob(value).exp()


class Logistic(_ScalarDistribution):
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc=0., scale=1., validate_args=None):
        self._initialize(dict(loc=loc, scale=scale), validate_args)

    def log_prob(self, value):
        z = (self._tensor(value) - self.loc) / self.scale
        return -self.scale.log() - softplus(z) - softplus(-z)

    def cdf(self, value):
        return ((self._tensor(value) - self.loc) / self.scale).sigmoid()

    def icdf(self, value):
        return self.loc + self.scale * self._tensor(value).logit()

    def rsample(self, sample_shape=torch.Size()):
        return self.icdf(self._uniform(sample_shape))

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return (math.pi * self.scale)**2 / 3


class Levy(_ScalarDistribution):
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    has_rsample = True

    def __init__(self, loc=0., scale=1., validate_args=None):
        self._initialize(dict(loc=loc, scale=scale), validate_args)

    @property
    def support(self):
        return constraints.greater_than(self.loc)

    def log_prob(self, value):
        difference = self._tensor(value) - self.loc
        safe = torch.where(difference > 0, difference, 1.)
        log_density = 0.5 * (self.scale.log() - math.log(2 * math.pi)
                             - 3 * safe.log() - self.scale / safe)
        return torch.where(difference > 0, log_density, -torch.inf)

    def cdf(self, value):
        difference = self._tensor(value) - self.loc
        safe = torch.where(difference > 0, difference, 1.)
        return torch.where(difference > 0, torch.erfc((self.scale / (2 * safe)).sqrt()), 0.)

    def icdf(self, value):
        z = -torch.special.ndtri(self._tensor(value) / 2)
        return self.loc + self.scale / z.square()

    def rsample(self, sample_shape=torch.Size()):
        z = torch.randn(self._extended_shape(torch.Size(sample_shape)),
                        dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + self.scale / z.square()

    @property
    def mean(self):
        return torch.full_like(self.loc, torch.inf)

    @property
    def variance(self):
        return torch.full_like(self.loc, torch.inf)


class AsymmetricLaplace(_ScalarDistribution):
    """Two-sided exponential density with separate positive and negative rates."""

    arg_constraints = {"right_rate": constraints.positive, "left_rate": constraints.positive,
                       "loc": constraints.real}
    support = constraints.real
    has_rsample = True

    def __init__(self, right_rate, left_rate, loc=0., validate_args=None):
        self._initialize(dict(right_rate=right_rate, left_rate=left_rate, loc=loc), validate_args)

    def log_prob(self, value):
        x = self._tensor(value) - self.loc
        log_normalizer = self.right_rate.log() + self.left_rate.log() - (
            self.right_rate + self.left_rate
        ).log()
        return log_normalizer - torch.where(x >= 0, x * self.right_rate, -x * self.left_rate)

    def cdf(self, value):
        x = self._tensor(value) - self.loc
        left_mass = self.right_rate / (self.left_rate + self.right_rate)
        left = left_mass * (self.left_rate * x.clamp_max(0)).exp()
        right = left_mass - (1 - left_mass) * torch.expm1(-self.right_rate * x.clamp_min(0))
        return torch.where(x < 0, left, right)

    def icdf(self, value):
        p = self._tensor(value)
        left_mass = self.right_rate / (self.left_rate + self.right_rate)
        left = (torch.minimum(p, left_mass).log() - left_mass.log()) / self.left_rate
        right = ((-left_mass).log1p() - (-torch.maximum(p, left_mass)).log1p()) / self.right_rate
        return self.loc + torch.where(p < left_mass, left, right)

    def rsample(self, sample_shape=torch.Size()):
        return self.icdf(self._uniform(sample_shape))

    @property
    def mean(self):
        return self.loc + self.right_rate.reciprocal() - self.left_rate.reciprocal()

    @property
    def variance(self):
        return self.right_rate.reciprocal().square() + self.left_rate.reciprocal().square()


class WrappedCauchy(_ScalarDistribution):
    """Circular density, with wrapped samples in [-pi, pi)."""

    arg_constraints = {"loc": constraints.real, "rho": constraints.half_open_interval(0., 1.)}
    support = constraints.interval(-math.pi, math.pi)
    has_rsample = True

    def __init__(self, loc=0., rho=0., validate_args=None):
        self._initialize(dict(loc=loc, rho=rho), validate_args)

    def log_prob(self, value):
        angle = self._tensor(value) - self.loc
        denominator = (1 - self.rho).square() + 4 * self.rho * (angle / 2).sin().square()
        return (-self.rho.square()).log1p() - math.log(2 * math.pi) - denominator.log()

    def rsample(self, sample_shape=torch.Size()):
        u = self._uniform(sample_shape)
        angle = 2 * (((1 - self.rho) / (1 + self.rho)) * (math.pi * (u - 0.5)).tan()).atan()
        return (self.loc + angle + math.pi).remainder(2 * math.pi) - math.pi


class SkewNormal(_ScalarDistribution):
    arg_constraints = {"shape": constraints.real, "loc": constraints.real,
                       "scale": constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(self, shape, loc=0., scale=1., validate_args=None):
        self._initialize(dict(shape=shape, loc=loc, scale=scale), validate_args)

    def log_prob(self, value):
        z = (self._tensor(value) - self.loc) / self.scale
        return (math.log(2) - math.log(2 * math.pi) / 2 - self.scale.log()
                - z.square() / 2 + torch.special.log_ndtr(self.shape * z))

    def rsample(self, sample_shape=torch.Size()):
        z = torch.randn((2,) + self._extended_shape(torch.Size(sample_shape)),
                        dtype=self.loc.dtype, device=self.loc.device)
        symmetric = torch.hypot(self.shape, torch.ones_like(self.shape)).reciprocal()
        return self.loc + self.scale * (symmetric * z[0] + (symmetric * self.shape) * z[1].abs())

    @property
    def mean(self):
        delta = self.shape / torch.hypot(self.shape, torch.ones_like(self.shape))
        return self.loc + self.scale * delta * math.sqrt(2 / math.pi)

    @property
    def variance(self):
        delta = self.shape / torch.hypot(self.shape, torch.ones_like(self.shape))
        return self.scale.square() * (1 - 2 * delta.square() / math.pi)
