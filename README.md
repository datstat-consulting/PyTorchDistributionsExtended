# PyTorchDistributionsExtended

Use extra scalar probability distributions through PyTorch's `Distribution` interface. The package grew out of work on stable innovations for [PyLevyProcess](https://github.com/datstat-consulting/PyLevyProcess). It includes a small set of differentiable Torch implementations and a numerical interface to SciPy's continuous distribution catalog.

## Installation

~~~bash
python -m pip install git+https://github.com/datstat-consulting/PyTorchDistributionsExtended.git
~~~

Run `python -m pip install .` from a checkout. Python 3.10 or later is required.

## Differentiable Torch distributions

`Levy`, `Logistic`, `AsymmetricLaplace`, `SkewNormal`, and `WrappedCauchy` support batched parameters, differentiable log densities, and reparameterized sampling. Tensor calculations follow the parameters' device and floating-point dtype.

~~~python
import torch
from PyTorchDistributionsExtended import AsymmetricLaplace

rate = torch.tensor([1.0, 2.0], requires_grad=True)
distribution = AsymmetricLaplace(right_rate=rate, left_rate=3.0)
draws = distribution.rsample((100,))  # Shape (100, 2).
loss = -distribution.log_prob(torch.tensor(0.5)).mean()
loss.backward()
~~~

Use a Torch-native distribution for gradient-based fitting or a Hamiltonian Monte Carlo target. The scalar event shape is empty. The sample shape precedes the broadcast shape of the parameters.

## Numerical SciPy distributions

`ScipyContinuous.catalog` lists the continuous distributions in the installed SciPy version. Pass shape parameters in SciPy order, followed by optional `loc` and `scale` keywords. Named constructors include `StableDistribution`, `SymmetricStable`, `GeneralizedHyperbolic`, and `NormalInverseGaussian`.

~~~python
import torch
from PyTorchDistributionsExtended import ScipyContinuous, StableDistribution

law = ScipyContinuous("gennorm", 1.5, loc=0.0, scale=2.0)
bounds = law.icdf(torch.tensor([0.05, 0.95]))
stable = StableDistribution(alpha=1.7, beta=0.2)
draws = stable.sample((256,))
log_density = stable.log_prob(draws)
~~~

The numerical backend evaluates on CPU and returns tensors on the parameter device. It provides sampling, log density, CDF, quantiles, and moments where SciPy supplies them. These evaluations detach their inputs from autograd and use ordinary sampling. Stable laws use SciPy's S1 parameterization.
