"""Center-gap representation."""

import torch
import torch.nn.functional as F

__all__ = [
    "centergap_to_quantiles",
    "CenterGapToQuantileTransform",
    "transform_centergap_posterior",
]


def centergap_to_quantiles(central, lower_gaps, upper_gaps):
    """Convert center-gap representation samples to quantiles.

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


def _softplus_inverse(y):
    return y + torch.log(-torch.expm1(-y))


class CenterGapToQuantileTransform(torch.distributions.transforms.Transform):
    """Bijective transform from center-gap distribution to quantile distribution.

    Parameters
    ----------
    L : int
        Number of lower quantile gaps in the center-gap representation.

    Notes
    -----
    Input is the tensor of shape ``(..., Q)`` in center-gap representation.
    The first component is the central quantile, followed by *L* lower pre-gaps and
    *U* upper pre-gaps, i.e., ``Q = 1 + L + U``.

    The pre-gaps are transformed by softplus to get the actual gaps, which are then
    cumulatively summed to the central quantile to get the quantiles.
    """

    domain = torch.distributions.constraints.real_vector
    codomain = torch.distributions.constraints.real_vector
    bijective = True

    def __init__(self, L):
        super().__init__()
        self.L = L

    def _call(self, x):
        L = self.L
        central = x[..., :1]
        lower_gaps = x[..., 1 : 1 + L]
        upper_gaps = x[..., 1 + L :]
        return centergap_to_quantiles(central, lower_gaps, upper_gaps)

    def _inverse(self, y):
        L = self.L
        central = y[..., L : L + 1]

        lower_with_central = y[..., : L + 1]
        lower_gaps_linear = lower_with_central[..., 1:] - lower_with_central[..., :-1]

        upper_with_central = y[..., L:]
        upper_gaps_linear = upper_with_central[..., 1:] - upper_with_central[..., :-1]

        lower_gaps_pre = _softplus_inverse(lower_gaps_linear)
        upper_gaps_pre = _softplus_inverse(upper_gaps_linear)

        return torch.cat([central, lower_gaps_pre, upper_gaps_pre], dim=-1)

    def log_abs_det_jacobian(self, x, y):
        return F.logsigmoid(x[..., 1:]).sum(dim=-1)


def transform_centergap_posterior(loc, covar, L):
    """Convert the center-gap posterior to quantile posterior.

    Parameters
    ----------
    loc : torch.Tensor with shape (N, Q)
        The mean of the posterior distribution in center-gap representation.
    covar : torch.Tensor with shape (N, N, Q)
        The covariance matrix of the posterior distribution in
        center-gap representation.
    L : int
        The number of lower quantiles in center-gap representation.

    Returns
    -------
    quantile_posterior : torch.distributions.TransformedDistribution
        Posterior over quantiles, obtained by applying
        :class:`CenterGapToQuantileTransform` to a batched
        :class:`torch.distributions.MultivariateNormal`.

    Notes
    -----
    The quantile dimension consists of the central quantile,
    followed by *L* lower gaps and *U* upper gaps, where *U = Q - L - 1*.
    """
    base_dist = torch.distributions.MultivariateNormal(loc, covariance_matrix=covar)
    transform = CenterGapToQuantileTransform(L)
    return torch.distributions.TransformedDistribution(base_dist, transform)
