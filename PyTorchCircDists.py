import torch
import math
from torch.distributions import Distribution, constraints

class CircularDistribution(Distribution):
    def __init__(self, params):
        self.params = params

    def pdf(self, x):
        raise NotImplementedError("Subclasses should implement this method.")

    def log_prob(self, x):
        raise NotImplementedError("Subclasses should implement this method.")

    def sample(self, n):
        raise NotImplementedError("Subclasses should implement this method.")

    def rsample(self, n)
        raise NotImplementedError("Subclasses should implement this method.")

   def circular_mean(self, angles):
        x_mean = torch.mean(torch.cos(angles))
        y_mean = torch.mean(torch.sin(angles))
        return torch.atan2(y_mean, x_mean)

    def circular_variance(self, angles):
        x_mean = torch.mean(torch.cos(angles))
        y_mean = torch.mean(torch.sin(angles))
        return 1 - torch.sqrt(x_mean ** 2 + y_mean ** 2)
      
class vonmises(CircularDistribution):
  arg_constraints = {
        "mu": constraints.interval(-torch.pi, torch.pi),
        "kappa": constraints.positive,
    }
    support = constraints.interval(-torch.pi, torch.pi)
  def __init__(self, mu, kappa):
        super().__init__({"mu": mu, "kappa": kappa})
        self.mu = mu
        self.kappa = kappa

  def pdf(self, x):
    return torch.exp(self.log_prob(x))

  def log_prob(self, x):
    return self.kappa * torch.cos(x - self.mu) - torch.log(2 * torch.pi) - torch.log(torch.i0(self.kappa))
    
  def _rejection_sample(self, n):
    samples = []
    while len(samples) < n:
      u = torch.rand(1)
      v = torch.rand(1) * 2 - 1
      s = u ** 2 + v ** 2
      if s < 1:
        x = u
        y = v
        sample = self.mu + torch.atan2(self.kappa * y, self.kappa * x - s)
        samples.append(sample)
    return torch.stack(samples)
      
  def sample(self, sample_shape=torch.Size()):
    n = torch.prod(torch.tensor(sample_shape)).item()
    return self._rejection_sample(n).view(sample_shape)

  def rsample(self, sample_shape=torch.Size()):
    with torch.no_grad():
      return self.sample(sample_shape)
        
class wrapcauchy(CircularDistribution):
  arg_constraints = {
        "mu": constraints.interval(-torch.pi, torch.pi),
        "kappa": constraints.positive,
    }
  support = constraints.real

  def __init__(self, mu, kappa):
    self.mu = mu
    self.kappa = kappa
    super().__init__()

  def sample(self, sample_shape=torch.Size()):
    u = torch.rand(sample_shape) - 0.5
    angle = self.mu.unsqueeze(-1) + torch.atan(torch.tan(torch.pi * u) / self.kappa.unsqueeze(-1))
    return angle % (2 * torch.pi) - torch.pi

  def rsample(self, sample_shape=torch.Size()):
    return self.sample(sample_shape)

  def log_prob(self, value):
    log_prob = torch.log(self.kappa) - torch.log(2 * torch.pi * (1 + self.kappa ** 2 - 2 * self.kappa * torch.cos(value - self.mu)))
    return log_prob
