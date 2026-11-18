import numpy as np
from scipy import stats
import torch

from PyTorchDistributionsExtended import (
    AsymmetricLaplace, Levy, Logistic, NormalInverseGaussian, ScipyContinuous,
    SkewNormal, StableDistribution, WrappedCauchy,
)


def test_native_laws_shape_quantiles_and_gradients():
    loc = torch.tensor([0., 1.], dtype=torch.float64, requires_grad=True)
    for law in (Levy(loc), Logistic(loc), SkewNormal(1.5, loc), WrappedCauchy(loc, 0.2),
                AsymmetricLaplace(1., 2., loc)):
        sample = law.rsample((3,))
        assert sample.shape == (3, 2) and sample.dtype == loc.dtype
        assert law.expand((4, 2)).batch_shape == (4, 2)
        value = law.log_prob(torch.tensor(2.)).sum()
        assert torch.isfinite(torch.autograd.grad(value, loc)[0]).all()
    p = torch.tensor([0.1, 0.4, 0.8], dtype=torch.float64)[:, None]
    law = AsymmetricLaplace(1., 2., loc)
    torch.testing.assert_close(law.cdf(law.icdf(p)), p.expand(3, 2))


def test_numerical_catalog_and_sampling():
    values = torch.tensor([-1., 0.2, 2.], dtype=torch.float64)[:, None]
    law = ScipyContinuous("gennorm", torch.tensor([1.5, 2.], dtype=torch.float64))
    np.testing.assert_allclose(law.log_prob(values), stats.gennorm.logpdf(values.numpy(), [1.5, 2.]))
    torch.testing.assert_close(law.icdf(law.cdf(values)), values.expand(3, 2))
    torch.manual_seed(4)
    first = law.sample((5,))
    torch.manual_seed(4)
    torch.testing.assert_close(first, law.sample((5,)))
    assert len(ScipyContinuous.catalog) >= 80
    stable = StableDistribution(2.0, 0.0).expand((2,))
    assert stable.sample((5,)).shape == (5, 2)
    inverse_gaussian = NormalInverseGaussian(4., 1., 0., 2.)
    np.testing.assert_allclose(inverse_gaussian.log_prob(torch.tensor(1.)),
                               stats.norminvgauss.logpdf(1., 8., 2., scale=2.), rtol=1e-6)
