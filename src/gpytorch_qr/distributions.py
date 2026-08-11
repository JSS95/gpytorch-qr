"""Asymmetric Laplace distributions for Bayesian quantile regression."""

import torch
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all
from torch.types import _Number

__all__ = [
    "AsymmetricLaplace",
]


class AsymmetricLaplace(Distribution):
    """Asymmetric Laplace distribution.

    Parameters
    ----------
    loc : torch.Tensor
        Location parameter of the distribution.
    scale : torch.Tensor
        Scale of the distribution.
    asymmetry : torch.Tensor
        Asymmetry of the distribution.
    """

    arg_constraints = {
        "loc": torch.distributions.constraints.real,
        "scale": torch.distributions.constraints.positive,
        "asymmetry": torch.distributions.constraints.positive,
    }
    support = constraints.real
    has_rsample = True

    @property
    def mean(self):
        m, L, k = self.loc, self.scale, self.asymmetry
        return m + (1 - k**2) / (L * k)

    @property
    def mode(self):
        return self.loc

    @property
    def variance(self):
        L, k = self.scale, self.asymmetry
        return (1 + k**4) / (L**2 * k**2)

    @property
    def stddev(self):
        return self.variance.sqrt()

    def __init__(self, loc, scale, asymmetry, validate_args=None):
        self.loc, self.scale, self.asymmetry = broadcast_all(loc, scale, asymmetry)
        if isinstance(loc, _Number) and isinstance(scale, _Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        """Return a new distribution instance with batch dimensions expanded."""
        new = self._get_checked_instance(AsymmetricLaplace, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.asymmetry = self.asymmetry.expand(batch_shape)
        super(AsymmetricLaplace, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        value = torch.empty(
            shape, dtype=self.loc.dtype, device=self.loc.device
        ).uniform_(0.0, 1.0)
        return self.icdf(value)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        L, k = self.scale, self.asymmetry
        residual = value - self.loc
        rate = torch.where(residual < 0, L / k, L * k)
        return torch.log(L * k) - torch.log1p(k**2) - rate * residual.abs()

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)

        L, k = self.scale, self.asymmetry
        residual = value - self.loc
        left_mass = k**2 / (1 + k**2)
        return torch.where(
            residual < 0,
            left_mass * torch.exp(L * residual / k),
            left_mass - (1 - left_mass) * torch.expm1(-L * k * residual),
        )

    def icdf(self, value):
        L, k = self.scale, self.asymmetry
        left_mass = k**2 / (1 + k**2)
        return torch.where(
            value <= left_mass,
            self.loc + k / L * torch.log(value / left_mass),
            self.loc - 1 / (L * k) * (torch.log1p(-value) - torch.log1p(-left_mass)),
        )

    def entropy(self):
        L, k = self.scale, self.asymmetry
        return 1 + torch.log(1 + k**2) - torch.log(L * k)
