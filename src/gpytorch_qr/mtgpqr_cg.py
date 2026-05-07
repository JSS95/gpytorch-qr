"""
Multitask GPQR (Center-gap representation)
------------------------------------------

1D regression dataset with heteroskedastic noise:

.. plot::
   :context:

    import torch
    from torch.distributions import Normal

    def mean(x):
        return torch.cos(x * 2 * 3.14)

    def std(x):
        return x + 0.1

    x_range = torch.linspace(0, 1, 100).reshape(-1, 1)
    x = x_range.repeat(5, 1)
    y = (mean(x) + torch.randn(x.shape).mul(std(x))).squeeze()
    q = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
    true_quantiles = mean(x_range) + std(x_range) * Normal(0, 1).icdf(q)
    import matplotlib.pyplot as plt
    plt.scatter(x, y, c='gray', marker='.', alpha=0.1)
    plt.plot(x_range, true_quantiles, '--', c='k')
"""

import gpytorch
import torch
import torch.nn.functional as F

__all__ = [
    "centergap_to_quantiles",
    "CenterGapQuantileGP",
    "ALD",
    "CenterGapALDLikelihood",
]


def centergap_to_quantiles(central, lower_gaps, upper_gaps):
    """Convert center-gap representation to quantiles.

    Parameters
    ----------
    central : torch.Tensor with shape (..., 1)
        The central quantile values.
    lower_gaps : torch.Tensor with shape (..., L)
        Pre-transformed lower gap values.
    upper_gaps : torch.Tensor with shape (..., U)
        Pre-transformed upper gap values.

    Returns
    -------
    quantiles : torch.Tensor with shape (..., Q)
        The quantile values. (Q = L + U + 1)
    """
    lower_gaps = F.softplus(lower_gaps)
    lower_quantiles = central - lower_gaps.flip(dims=[-1]).cumsum(dim=-1).flip(
        dims=[-1]
    )

    upper_gaps = F.softplus(upper_gaps)
    upper_quantiles = central + upper_gaps.cumsum(dim=-1)

    ret = torch.concat([lower_quantiles, central, upper_quantiles], dim=-1)
    return ret


class CenterGapQuantileGP(gpytorch.models.ApproximateGP):
    """Multitask approximate GP for multiple quantiles using center-gap representation.

    Parameters
    ----------
    variational_strategy : gpytorch.variational.VariationalStrategy
        The variational strategy for the Gaussian process.
    center_mean : gpytorch.means.Mean
        The mean module for the central quantile.
    gap_mean : gpytorch.means.Mean
        The mean module for the gaps between quantiles.
    covar_module : gpytorch.kernels.Kernel
        The covariance module for the Gaussian process.
    """

    def __init__(self, variational_strategy, center_mean, gap_mean, covar_module):
        super().__init__(variational_strategy)
        self.center_mean = center_mean
        self.gap_mean = gap_mean
        self.covar_module = covar_module

    def forward(self, x):
        center_mean = self.center_mean(x)
        gap_mean = self.gap_mean(x)
        mean = torch.concat([center_mean.unsqueeze(0), gap_mean], dim=0)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class ALD(torch.distributions.Distribution):
    """Asymmetric Laplace distribution for multitask quantile regression.

    Parameters
    ----------
    m : torch.Tensor with shape (S, N, Q)
        The location parameters of the distribution.
    lamda : torch.Tensor with shape (Q,)
        The scale parameters of the distribution for each quantile.
    kappa : torch.Tensor with shape (Q,)
        The quantile levels of the distribution.

    Notes
    -----
    In the context of multitask quantile regression, the location parameter *m*
    corresponds to sample points drawn from posterior distributions of latent GPs.
    For *Q* quantiles, *S* samples are drawn for *N* data points.

    The value passed to :meth:`log_prob` is the observed *y* values.
    """

    arg_constraints = {
        "m": torch.distributions.constraints.real,
        "lamda": torch.distributions.constraints.positive,
        "kappa": torch.distributions.constraints.unit_interval,
    }
    support = torch.distributions.constraints.real
    has_rsample = False

    def __init__(self, m, lamda, kappa):
        # Reshape lamda and kappa as (1, 1, Q)
        self.m = m
        self.lamda = lamda.view(1, 1, -1)
        self.kappa = kappa.view(1, 1, -1)
        super().__init__(m.size())

    def log_prob(self, value):
        """Log probability of the asymmetric Laplace distribution at the given value.

        Parameters
        ----------
        value : torch.Tensor with shape (N,)
            The values at which to evaluate the log probability.

        Returns
        -------
        logp : torch.Tensor with shape (S, N, Q)
            The log probability at the given values for each quantile.
        """
        # value: (N,), m: (S, N, Q), lamda & kappa: (1, 1, Q)
        diff = value.unsqueeze(0) - self.m  # (S, N, Q)
        rho = diff * (self.kappa - (diff < 0).float())  # (S, N, Q)
        logp = (
            torch.log(self.kappa * (1 - self.kappa) / self.lamda) - rho / self.lamda
        )  # (S, N, Q)
        return logp


class CenterGapALDLikelihood(torch.distributions.Distribution):
    """ALD likelihood for multitask quantile regression with center-gap representation.

    Parameters
    ----------
    q : torch.Tensor with shape (Q,)
        The quantile levels.
    """

    def __init__(self, q):
        super().__init__()
        self.register_buffer("q", q.float())
        self.register_parameter(
            "raw_scales",
            torch.nn.Parameter(torch.zeros(len(q))),
        )
        self.register_constraint("raw_scales", gpytorch.constraints.Positive())
        central_quantile = self.q[torch.argmin(torch.abs(self.q - 0.5))]
        self.lower_count = (self.q < central_quantile).count_nonzero()

    @property
    def scales(self):
        return self.raw_scales_constraint.transform(self.raw_scales)

    def forward(self, function_samples):
        """Return the ALD distribution for the given function samples.

        Parameters
        ----------
        function_samples : torch.Tensor with shape (S, N, 1 + L + U)
            The function samples drawn from the posterior distributions of quantile
            functions. *S* is the number of samples, *N* is the number of data points,
            *L* is the number of lower quantiles, and *U* is the number of upper
            quantiles.
            The first channel corresponds to the central quantile,
            the next *L* channels correspond to the lower gaps,
            and the last *U* channels correspond to the upper gaps.
        """
        center = function_samples[:, :, :1]
        lower_gaps = function_samples[:, :, 1 : 1 + self.lower_count]
        upper_gaps = function_samples[:, :, 1 + self.lower_count :]
        quantiles = centergap_to_quantiles(center, lower_gaps, upper_gaps)
        return ALD(
            locs=quantiles,  # (S, N, Q)
            scales=self.scales,  # (Q,)
            taus=self.taus,  # (Q,)
        )
