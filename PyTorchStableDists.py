import torch
import math
from torch.distributions import Distribution, constraints

class StableDistribution(Distribution):
    arg_constraints = {
        "alpha": constraints.interval(0, 2),
        "beta": constraints.interval(-1, 1),
        "mu": constraints.real,
        "sigma": constraints.positive,
    }
    support = constraints.real

    def __init__(self, alpha, beta, mu, sigma):
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.sigma = sigma

class levy(StableDistribution):
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.real

    def __init__(self, loc, scale, validate_args=None):
        alpha = torch.tensor(0.5)
        beta = torch.tensor(1.0)
        mu = torch.as_tensor(loc)
        sigma = torch.as_tensor(scale)
        super().__init__(alpha, beta, mu, sigma)
        self.loc = self.mu
        self.scale = self.sigma
        
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        z = (self.scale / (value - self.loc))
        return - torch.log(z) - 1.5 * torch.log(2 * torch.tensor(math.pi)) - 0.5 * (z ** 2)

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        u = torch.rand(shape)
        v = torch.rand(shape)
        w = torch.randn(shape)
        c = 1.0 / u
        x = c * torch.exp(-torch.sqrt(2 * v) * w)
        return self.loc + self.scale * x
      
    def pdf(self, x):
        #return 1 / (self.sigma * torch.pi * (1 + ((x - self.mu) / self.sigma) ** 2))
        return torch.exp(self.log_prob(x))

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return torch.erfc(torch.sqrt(self.scale / (2 * (value - self.loc))))

    def icdf(self, value):
        return self.loc + self.scale / (2 * (torch.erfcinv(value)) ** 2)

class cauchy(StableDistribution):
    support = constraints.real
    
    def __init__(self, mu, sigma):
        super().__init__(alpha=1, beta=0, mu=mu, sigma=sigma)

    def pdf(self, x):
        #return 1 / (self.sigma * torch.pi * (1 + ((x - self.mu) / self.sigma) ** 2))
        return torch.exp(self.log_prob(x))

    def cdf(self, x):
        return torch.atan((x - self.mu) / self.sigma) / torch.pi + 0.5

    def icdf(self, q):
        return self.mu + self.sigma * torch.tan(torch.pi * (q - 0.5))
      
    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        u = torch.rand(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + self.scale * torch.tan(math.pi * (u - 0.5))

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape)

    def log_prob(self, value):
        return -torch.log(self.scale * (math.pi * (1 + ((value - self.loc) / self.scale) ** 2)))
      
class norm(StableDistribution):
    def __init__(self, mu, sigma):
        alpha = 2
        beta = 0
        super().__init__(alpha, beta, mu, sigma)
        
     def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = torch.randn(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + eps * self.scale

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape)
  
    def pdf(self, x):
        #return (1 / (torch.sqrt(2 * torch.tensor(math.pi) * self.sigma**2))) * torch.exp(-0.5 * ((x - self.mu) / self.sigma)**2)
        return torch.exp(self.log_prob(x))
          
    def log_prob(self, value):
        var = (self.scale ** 2)
        log_scale = torch.log(self.scale)
        return -((value - self.loc) ** 2) / (2 * var) - log_scale - 0.5 * torch.tensor(math.log(2 * math.pi))

    def cdf(self, value):
        normal_dist = Normal(self.mu, self.sigma)
        return normal_dist.cdf(value)

    def icdf(self, value):
        normal_dist = Normal(self.mu, self.sigma)
        return normal_dist.icdf(value)
      
class LevySkew(StableDistribution):
    def __init__(self, mu, sigma):
        alpha = 0.5
        beta = 1
        super().__init__(alpha, beta, mu, sigma)

    def log_prob(self, value):
        # No closed-form expression for the logpdf
        raise NotImplementedError("No closed-form expression for the Levy skew logpdf")

    def cdf(self, value):
        # No closed-form expression for the cdf
        raise NotImplementedError("No closed-form expression for the Levy skew cdf")

    def icdf(self, value):
        # No closed-form expression for the icdf
        raise NotImplementedError("No closed-form expression for the Levy skew icdf")

class SymmetricStable(StableDistribution):
    def __init__(self, alpha, mu, sigma):
        beta = 0
        super().__init__(alpha, beta, mu, sigma)

    def log_prob(self, value):
        # No closed-form expression for the logpdf
        raise NotImplementedError("No closed-form expression for the symmetric stable logpdf")

    def cdf(self, value):
        # No closed-form expression for the cdf
        raise NotImplementedError("No closed-form expression for the symmetric stable cdf")

    def icdf(self, value):
        # No closed-form expression for the icdf
        raise NotImplementedError("No closed-form expression for the symmetric stable icdf")

class pareto(StableDistribution):
    def __init__(self, alpha, xm):
        super().__init__(alpha, torch.tensor(0.0), xm, alpha * xm)

    def log_prob(self, value):
        return torch.log(self.alpha) + self.alpha * torch.log(self.mu) - (self.alpha + 1) * torch.log(value)

    def pdf(self, x):
        return torch.exp(self.log_prob(x))

    def cdf(self, value):
        return 1 - torch.pow(self.mu / value, self.alpha)

    def icdf(self, value):
        return self.mu / torch.pow(1 - value, 1 / self.alpha)

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        u = torch.rand(shape)
        return self.icdf(u)

class logistic(StableDistribution):
    def __init__(self, mu, sigma):
        super().__init__(torch.tensor(1.0), torch.tensor(0.0), mu, sigma)

    def log_prob(self, value):
        z = (value - self.mu) / self.sigma
        return -z - 2 * torch.log(1 + torch.exp(-z))

    def pdf(self, x):
        return torch.exp(self.log_prob(x))

    def cdf(self, value):
        z = (value - self.mu) / self.sigma
        return 1 / (1 + torch.exp(-z))

    def icdf(self, value):
        return self.mu - self.sigma * torch.log(1 / value - 1)

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        u = torch.rand(shape)
        return self.icdf(u)

class t(StableDistribution):
    arg_constraints = {
        "nu": constraints.positive,
        "mu": constraints.real,
        "sigma": constraints.positive,
    }
    support = constraints.real
    def __init__(self, nu, mu, sigma):
        self.nu = torch.as_tensor(nu)
        super().__init__(None, None, mu, sigma)

    def log_prob(self, value):
        z = (value - self.mu) / self.sigma
        return torch.lgamma((self.nu + 1) / 2) - torch.lgamma(self.nu / 2) - 0.5 * torch.log(self.nu * torch.tensor(math.pi)) - torch.log(self.sigma) - ((self.nu + 1) / 2) * torch.log(1 + z ** 2 / self.nu)

    def pdf(self, x):
        return torch.exp(self.log_prob(x))

    def cdf(self, value):
        z = (value - self.mu) / self.sigma
        x = torch.sqrt(self.nu) * z / torch.sqrt(1 + z ** 2)
        p = 0.5 * (1 + torch.erf(x / math.sqrt(2)))
        return p

    def icdf(self, value):
        x = torch.erfinv(2 * value - 1) * math.sqrt(2)
        z = torch.sqrt(self.nu) * x / torch.sqrt(self.nu - 2 + x ** 2)
        return self.mu + self.sigma * z

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
