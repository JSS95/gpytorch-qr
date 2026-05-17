"""Center-gap representation."""

import gpytorch
import torch
import torch.nn.functional as F
from gpytorch.means import Mean

__all__ = [
    "centergap_to_quantiles",
    "CenterGapToQuantileTransform",
    "transform_centergap_posterior",
    "CenterGapMean",
]


def centergap_to_quantiles(central, lower_gaps, upper_gaps, quantile_dim=-1):
    """Convert center-gap representation samples to quantiles.

    Parameters
    ----------
    central : torch.Tensor with shape (..., 1, ...)
        The central quantile values.
    lower_gaps : torch.Tensor with shape (..., L, ...)
        Pre-transformed lower gap values.
    upper_gaps : torch.Tensor with shape (..., U, ...)
        Pre-transformed upper gap values.
    quantile_dim : int, default=-1
        The dimension along which the quantiles are represented.

    Returns
    -------
    quantiles : torch.Tensor with shape (..., Q, ...)
        Quantile values. (Q = L + U + 1 at *quantile_dim*)
        The quantiles are ordered from lowest to highest along the quantile dimension.
    """
    lower_gaps = F.softplus(lower_gaps)
    lower_quantiles = central - lower_gaps.flip(dims=[quantile_dim]).cumsum(
        dim=quantile_dim
    ).flip(dims=[quantile_dim])

    upper_gaps = F.softplus(upper_gaps)
    upper_quantiles = central + upper_gaps.cumsum(dim=quantile_dim)

    ret = torch.concat([lower_quantiles, central, upper_quantiles], dim=quantile_dim)
    return ret


def _softplus_inverse(y):
    return y + torch.log(-torch.expm1(-y))


class CenterGapToQuantileTransform(torch.distributions.transforms.Transform):
    """Bijective transform from center-gap distribution to quantile distribution.

    Parameters
    ----------
    L : int
        Number of lower quantile gaps in the center-gap representation.
    quantile_dim : {-1, -2}
        The dimension along which the quantiles are represented.

    Notes
    -----
    If *quantile_dim* is -1, shape of input tensor is either
    ``(N, Q)`` or ``(S, N, Q)``.
    If *quantile_dim* is -2, shape of input tensor is either
    ``(Q, N)`` or ``(S, Q, N)``.
    Here, *Q* is the number of quantiles, *N* is the number of data points,
    and *S* is the number of samples.

    The center-gap components along the quantile dimension is ordered as
    a central quantile, *L* lower pre-gaps, and *U* upper pre-gaps
    (``Q = 1 + L + U``).
    """

    domain = torch.distributions.constraints.real_vector
    codomain = torch.distributions.constraints.real_vector
    bijective = True

    def __init__(self, L, quantile_dim=-2):
        super().__init__()
        self.L = L
        self.quantile_dim = quantile_dim

    def _call(self, x):
        qdim = self.quantile_dim
        C = torch.narrow(x, qdim, 0, 1)
        L = torch.narrow(x, qdim, 1, self.L)
        U = torch.narrow(x, qdim, 1 + self.L, x.shape[qdim] - 1 - self.L)
        Q = centergap_to_quantiles(C, L, U, quantile_dim=qdim)
        return Q

    def _inverse(self, y):
        L = self.L
        qdim = self.quantile_dim
        central = torch.narrow(y, qdim, L, 1)
        lower_gaps_linear = torch.narrow(y, qdim, 0, L + 1).diff(dim=qdim)
        upper_gaps_linear = torch.narrow(y, qdim, L, y.shape[qdim] - L).diff(dim=qdim)
        return torch.cat(
            [
                central,
                _softplus_inverse(lower_gaps_linear),
                _softplus_inverse(upper_gaps_linear),
            ],
            dim=qdim,
        )

    def log_abs_det_jacobian(self, x, y):
        qdim = self.quantile_dim
        gaps = torch.narrow(x, qdim, 1, x.shape[qdim] - 1)
        return F.logsigmoid(gaps).sum(dim=(-2, -1))


def transform_centergap_posterior(posterior, L):
    """Convert the center-gap posterior to quantile posterior.

    Parameters
    ----------
    posterior : gpytorch.distributions.MultivariateNormal
        The center-gap posterior distribution.
    L : int
        The number of lower quantiles in center-gap representation.

    Returns
    -------
    quantile_posterior : torch.distributions.TransformedDistribution
        Posterior over quantiles, obtained by applying
        :class:`CenterGapToQuantileTransform` to a batched
        :class:`gpytorch.distributions.MultivariateNormal`.

    Notes
    -----
    The quantile dimension consists of the central quantile,
    followed by *L* lower gaps and *U* upper gaps, where *U = Q - L - 1*.
    """
    if isinstance(posterior, gpytorch.distributions.MultitaskMultivariateNormal):
        quantile_dim = -1
    elif isinstance(posterior, gpytorch.distributions.MultivariateNormal):
        quantile_dim = -2
    else:
        raise ValueError("Posterior is not a multivariate normal.")
    transform = CenterGapToQuantileTransform(L, quantile_dim=quantile_dim)
    return torch.distributions.TransformedDistribution(posterior, transform)


class CenterGapMean(Mean):
    """Mean module for center-gap.

    Parameters
    ----------
    center_mean : gpytorch.means.Mean
        Mean module for the central quantile.
        If GPQR treats quantiles as batches, this module should have batch shape
        ``(1, *B)`` where *B* is additional batch shape.
        If GPQR treats quantiles as task, this module should have batch shape
        ``(*B, 1)``.
    gap_mean : gpytorch.means.Mean
        Mean module for the quantile gaps.
        If GPQR treats quantiles as batches, this module should have batch shape
        ``(Q-1, *B)`` where *Q* is the number of quantiles and
        *B* is additional batch shape.
        If GPQR treats quantiles as task, this module should have batch shape
        ``(*B, L-1)`` where *L* is the number of latent GPs.
    latent_dim : {0, -1}
        The dimension along which the latent GPs are represented in module batch shape.
        ``0`` if quantiles are batches, ``-1`` if quantiles are tasks.

    Notes
    -----
    If GPQR treats quantiles as batches, input predictors are expected to have shape
    ``(1, *B, N, D)``.
    If GPQR treats quantiles as tasks, input predictors are expected to have shape
    ``(*B, 1, N, D)``.
    *N* is the number of data points and *D* is the number of input dimensions.
    """

    def __init__(self, center_mean, gap_mean, latent_dim):
        super().__init__()
        self.center_mean = center_mean
        self.gap_mean = gap_mean
        if latent_dim == 0:
            self.concat_dim = 0
        elif latent_dim == -1:
            self.concat_dim = -2
        else:
            raise ValueError("latent_dim should be either 0 or -1.")

    def forward(self, x):
        """Compute the mean of center-gap representation.

        Parameters
        ----------
        x : torch.Tensor in shape ``(1, *B, N, D)`` or ``(*B, 1, N, D)``
        """
        center_mean = self.center_mean(x)  # (1, *B, N) or (*B, 1, N)
        gap_mean = self.gap_mean(x)
        return torch.concat([center_mean, gap_mean], dim=self.concat_dim)
