import torch
from torch import exp, log
from scipy.special import kv
from scipy.integrate import quad
from torch.distributions import Distribution, constraints
import numpy as np

class genhyperbolic(Distribution):
    def __init__(self, alpha, beta, delta, mu, lam):
        self.alpha = torch.as_tensor(alpha, dtype=torch.float32)
        self.beta = torch.as_tensor(beta, dtype=torch.float32)
        self.delta = torch.as_tensor(delta, dtype=torch.float32)
        self.mu = torch.as_tensor(mu, dtype=torch.float32)
        self.lam = torch.as_tensor(lam, dtype=torch.float32)
        super(GeneralizedHyperbolic, self).__init__()
        
    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        z = torch.randn(shape)
        d = self.delta * torch.sqrt(1 + (self.beta * z)**2 / (self.alpha**2 - self.lam))
        return self.mu + d * z
      
    def pdf(self, value):
        value = torch.as_tensor(value)
        d = self.delta * torch.sqrt(1 + (self.beta * (value - self.mu))**2 / (self.alpha**2 - self.lam))
        prefactor = (self.alpha**2 - self.lam)**(self.lam / 2) * exp(self.delta * self.gamma) / (2 * self.alpha**self.lam * torch.sqrt(math.pi) * torch.exp(log(gamma(self.lam)) - log(gamma(self.lam / 2))))
        numerator = exp(-(self.delta * torch.sqrt(self.alpha**2 - self.lam + (self.beta * (value - self.mu))**2)))
        denominator = d**(2 * self.lam - 1)
        return prefactor * (numerator / denominator)
      
    def cdf(self, value):
        value = torch.as_tensor(value)

        def integrand(x):
            x = torch.tensor(x)
            d = self.delta * torch.sqrt(1 + (self.beta * (x - self.mu))**2 / (self.alpha**2 - self.lam))
            prefactor = (self.alpha**2 - self.lam)**(self.lam / 2) * exp(self.delta * self.gamma) / (2 * self.alpha**self.lam * torch.sqrt(math.pi) * torch.exp(log(gamma(self.lam)) - log(gamma(self.lam / 2))))
            numerator = exp(-(self.delta * torch.sqrt(self.alpha**2 - self.lam + (self.beta * (x - self.mu))**2)))
            denominator = d**(2 * self.lam - 1)
            return prefactor * (numerator / denominator).numpy()
        
        # Need to implement numerical integration in PyTorch later on
        # For each value, we need to integrate the PDF from -inf to that value
        result = []
        for val in value:
            integral, _ = quad(integrand, -np.inf, val.item())
            result.append(integral)

        return torch.tensor(result)
      
    def cdf(self, value):
        t1 = (self.delta * self.alpha * torch.sign(value - self.mu)).exp()
        t2 = self.gamma * (-(self.beta * (value - self.mu)).abs()).exp()

        result = t1 * (t2 - self.gamma) / (self.delta * self.alpha * self.beta)
        return result

class norminvgauss(GeneralizedHyperbolic):
    def __init__(self, alpha, beta, mu, delta):
        lambda_param = -0.5
        chi = torch.sqrt(alpha**2 - beta**2)
        psi = chi
        super().__init__(lambda_param, alpha, beta, mu, delta, chi, psi)

    @property
    def mean(self):
        return self.mu + self.delta * self.beta / self.chi

    @property
    def variance(self):
        return self.delta * (1 - self.beta**2 / self.alpha**2) / self.chi
