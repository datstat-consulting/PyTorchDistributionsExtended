import torch
from torch.distributions import Distribution

class expon(Distribution):
    def __init__(self, rate):
        self.rate = rate

    def log_prob(self, value):
        return torch.log(self.rate) - self.rate * value

    def pdf(self, x):
        return torch.exp(self.log_prob(x))

    def cdf(self, value):
        return 1 - torch.exp(-self.rate * value)

    def icdf(self, value):
        return -torch.log(1 - value) / self.rate

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        u = torch.rand(shape)
        return self.icdf(u)

    def _extended_shape(self, sample_shape):
        return sample_shape + self.rate.shape

class laplace_asymmetric(Distribution):
    def __init__(self, lambda1, lambda2):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.exp_pos = expon(self.lambda1)
        self.exp_neg = expon(self.lambda2)

    def log_prob(self, value):
        pos_part = torch.where(value >= 0, self.lambda1 * value, torch.tensor(0.0))
        neg_part = torch.where(value < 0, self.lambda2 * (-value), torch.tensor(0.0))
        return torch.log(self.lambda1 * self.lambda2 / (self.lambda1 + self.lambda2)) - (pos_part + neg_part)

    def pdf(self, value):
        pos_part = torch.where(value >= 0, self.lambda1 * torch.exp(-self.lambda1 * value), torch.tensor(0.0))
        neg_part = torch.where(value < 0, self.lambda2 * torch.exp(-self.lambda2 * (-value)), torch.tensor(0.0))
        return (pos_part + neg_part) / (self.lambda1 + self.lambda2)

    def cdf(self, value):
        pos_part = torch.where(value >= 0, 1 - torch.exp(-self.lambda1 * value), torch.tensor(0.0))
        neg_part = torch.where(value < 0, torch.exp(self.lambda2 * value), torch.tensor(0.0))
        return (pos_part + neg_part) / (self.lambda1 + self.lambda2)

    def icdf(self, value):
        pos_boundary = self.lambda1 / (self.lambda1 + self.lambda2)
        pos_part = torch.where(value >= pos_boundary, -torch.log(1 - (self.lambda1 + self.lambda2) * value / self.lambda1), torch.tensor(0.0))
        neg_part = torch.where(value < pos_boundary, torch.log((self.lambda1 + self.lambda2) * value / self.lambda2), torch.tensor(0.0))
        return pos_part - neg_part

    def sample(self, sample_shape=torch.Size()):
        pos_samples = self.exp_pos.sample(sample_shape)
        neg_samples = self.exp_neg.sample(sample_shape)
        return pos_samples - neg_samples

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        pos_samples = self.exp_pos.rsample(sample_shape)
        neg_samples = self.exp_neg.rsample(sample_shape)
        u = torch.rand(shape)
        return torch.where(u < self.lambda1 / (self.lambda1 + self.lambda2), pos_samples, -neg_samples)

    def _extended_shape(self, sample_shape):
        return sample_shape + self.lambda1.shape
