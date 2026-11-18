"""Numerical SciPy distributions with torch batch and sample conventions."""

from copy import copy

import numpy as np
from scipy import stats
from scipy.stats import rv_continuous
import torch
from torch.distributions import Distribution, constraints, utils


class _Interval(constraints.Constraint):
    def __init__(self, lower, upper):
        self.lower_bound, self.upper_bound = lower, upper

    def check(self, value):
        return (value >= self.lower_bound) & (value <= self.upper_bound)


def _catalog():
    return tuple(sorted(name for name in dir(stats)
                        if isinstance(getattr(stats, name), rv_continuous)))


class ScipyContinuous(Distribution):
    """Numerical CPU evaluations with samples and results returned as tensors.

    Pass shape parameters in SciPy order. Sampling uses a seed from Torch's RNG.
    Numerical methods require inputs that do not track gradients.
    """

    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    has_rsample = False
    catalog = _catalog()

    def __init__(self, name, *shape, loc=0., scale=1., validate_args=None):
        if name not in self.catalog:
            raise ValueError(f"Unknown SciPy continuous distribution {name!r}.")
        self.name = name
        self._law = self._get_law(name)
        if len(shape) != self._law.numargs:
            raise ValueError(f"{name} requires {self._law.numargs} shape parameters: {self._law.shapes}")
        self._shape_parameters = tuple()
        *shape, self.loc, self.scale = utils.broadcast_all(*shape, loc, scale)
        self._shape_parameters = tuple(shape)
        super().__init__(self.loc.shape, validate_args=validate_args)
        if self._validate_args:
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                lower, upper = self._law.support(*self._shape_arrays(detach=True))
            if np.isnan(lower).any() or np.isnan(upper).any():
                raise ValueError(f"Invalid shape parameters for {name}.")

    @staticmethod
    def _get_law(name):
        law = getattr(stats, name)
        if name == "levy_stable":
            law = copy(law)
            law.parameterization = "S1"
        return law

    @staticmethod
    def _numpy(value, *, detach=False):
        if value.requires_grad and not detach:
            raise RuntimeError("SciPy numerical evaluation does not support autograd.")
        return value.detach().cpu().double().numpy()

    def _shape_arrays(self, *, detach=False):
        return tuple(self._numpy(p, detach=detach) for p in self._shape_parameters)

    def _result(self, result):
        return torch.as_tensor(result, dtype=self.loc.dtype, device=self.loc.device)

    @property
    def support(self):
        lower, upper = self._law.support(
            *self._shape_arrays(detach=True), loc=self._numpy(self.loc, detach=True),
            scale=self._numpy(self.scale, detach=True)
        )
        return _Interval(self._result(lower), self._result(upper))

    def _evaluate(self, method, value):
        value = torch.as_tensor(value, dtype=self.loc.dtype, device=self.loc.device)
        return self._result(getattr(self._law, method)(
            self._numpy(value), *self._shape_arrays(),
            loc=self._numpy(self.loc), scale=self._numpy(self.scale)
        ))

    def log_prob(self, value):
        return self._evaluate("logpdf", value)

    def pdf(self, value):
        return self.log_prob(value).exp()

    def cdf(self, value):
        return self._evaluate("cdf", value)

    def icdf(self, value):
        return self._evaluate("ppf", value)

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(torch.Size(sample_shape))
        if any(size == 0 for size in shape):
            return self.loc.new_empty(shape)
        seed = int(torch.randint(2**32, (), dtype=torch.int64, device=self.loc.device))
        draw = self._law.rvs(
            *self._shape_arrays(detach=True), loc=self._numpy(self.loc, detach=True),
            scale=self._numpy(self.scale, detach=True), size=tuple(shape),
            random_state=np.random.default_rng(seed)
        )
        return self._result(draw)

    def _moment(self, method):
        return self._result(getattr(self._law, method)(
            *self._shape_arrays(), loc=self._numpy(self.loc), scale=self._numpy(self.scale)
        ))

    @property
    def mean(self):
        return self._moment("mean")

    @property
    def variance(self):
        return self._moment("var")

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(type(self), _instance)
        shape = torch.Size(batch_shape)
        new.name, new._law = self.name, self._law
        new.loc, new.scale = self.loc.expand(shape), self.scale.expand(shape)
        new._shape_parameters = tuple(value.expand(shape) for value in self._shape_parameters)
        for field, value in vars(self).items():
            if field not in vars(new) and isinstance(value, torch.Tensor):
                setattr(new, field, value.expand(shape))
        Distribution.__init__(new, shape, validate_args=False)
        new._validate_args = self._validate_args
        return new
