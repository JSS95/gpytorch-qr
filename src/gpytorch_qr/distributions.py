"""Asymmetric Laplace distributions for Bayesian quantile regression."""

import torch
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all
from torch.types import _Number

__all__ = [
    "AsymmetricLaplace",
    "ALD",
    "QuantileALD",
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


class ALD(torch.distributions.Distribution):
    """Asymmetric Laplace distribution.

    Parameters
    ----------
    m : torch.Tensor
        The location parameter of the distribution.
    lamda : torch.Tensor
        The scale parameter of the distribution.
    kappa : torch.Tensor
        The asymmetry parameter of the distribution.

    Attributes
    ----------
    m : torch.Tensor
    lamda : torch.Tensor
    kappa : torch.Tensor
    """

    arg_constraints = {
        "m": torch.distributions.constraints.real,
        "lamda": torch.distributions.constraints.positive,
        "kappa": torch.distributions.constraints.unit_interval,
    }
    support = torch.distributions.constraints.real
    has_rsample = False

    def __init__(self, m, lamda, kappa):
        self.m = m
        self.lamda = lamda
        self.kappa = kappa
        super().__init__(m.shape)

    def log_prob(self, value):
        """Log probability of the asymmetric Laplace distribution"""
        residual = value - self.m
        check = residual * (self.kappa - (residual < 0).to(residual))
        logp = (
            torch.log(self.kappa)
            + torch.log(1 - self.kappa)
            - torch.log(self.lamda)
            - check / self.lamda
        )
        return logp

    def icdf(self, value):
        """Inverse CDF of the asymmetric Laplace distribution."""
        return torch.where(
            value <= self.kappa,
            self.m + self.lamda / (1 - self.kappa) * torch.log(value / self.kappa),
            self.m
            - self.lamda / self.kappa * torch.log((1 - value) / (1 - self.kappa)),
        )


class QuantileALD(ALD):
    """Asymmetric Laplace distribution for multiple quantiles.

    Parameters
    ----------
    m : torch.Tensor with shape ``(S, *B, N, Q)``
        The location parameters of the distribution.
    lamda : torch.Tensor with shape ``(*B, 1, Q)``
        The scale parameters of the distribution for each quantile.
    kappa : torch.Tensor with shape ``(*B, 1, Q)``
        The quantile levels of the distribution.

    Attributes
    ----------
    m : torch.Tensor with shape ``(S, *B, N, Q)``
    lamda : torch.Tensor with shape ``(1, *B, 1, Q)``
    kappa : torch.Tensor with shape ``(1, *B, 1, Q)``

    Notes
    -----
    This class is designed for univariate Bayesian quantile regression
    with multiple quantiles, i.e., ``y`` is a vector of univariate response
    from which multiple quantiles are estimated.

    - ``S`` : the number of samples drawn from the posterior distribution.
    - ``Q`` : the number of quantiles.
    - ``B`` : additional batches.
    - ``N`` : the number of data points.

    Although the response is univariate, the distribution is multitask where
    each task corresponds to a different quantile.
    Multivariate regression with multiple quantiles can be achieved by
    defining a likelihood class that uses multiple instances of this distribution.
    """

    def __init__(self, m, lamda, kappa):
        super().__init__(m, lamda.unsqueeze(0), kappa.unsqueeze(0))

    def log_prob(self, value):
        """Log probability of the asymmetric Laplace distribution at the given value.

        Parameters
        ----------
        value : torch.Tensor with shape ``(*B, N, 1)`` or ``(*B, N, Q)``
            Observed response variables at which to evaluate the log probability.

        Returns
        -------
        logp : torch.Tensor with shape ``(S, *B, N, Q)``
            The log probability at the given values for each task and sample.
        """
        value = value.reshape(1, *value.shape)  # (1, *B, N, Q)
        return super().log_prob(value)
