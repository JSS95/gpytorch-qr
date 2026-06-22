"""Utility functions."""

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
        Quantile values. (Q = L + U + 1)
        The quantiles are ordered in increasing order along the quantile dimension.
    """
    quantile_dim = -1
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
    """Transformation from center-gap distribution to quantile distribution.

    Parameters
    ----------
    Qs : list of int
        The number of quantiles for each task, i.e., ``[Q_1, Q_2, ..., Q_k]``.
    Ls : list of int
        The number of lower quantiles in center-gap representation for each task,
        i.e., ``[L_1, L_2, ..., L_k]``.

    Notes
    -----
    The input distribution's quantile dimension should be laid out as:

    .. code-block:: text

        [c_1, c_2, ..., c_k,  *L_1, *U_1,  *L_2, *U_2,  ...,  *L_k, *U_k]

    where:

    - ``c_i`` is the central quantile for *i*-th output dimension,
    - ``L_i`` contains pre-softplus-transformed lower gaps for *i*-th output dimension,
    - ``U_i`` contains pre-softplus-transformed upper gaps for *i*-th output dimension.

    The output distribution's quantile dimension is laid out as:

    .. code-block:: text

        [*Q_1, *Q_2, ..., *Q_k].
    """

    domain = torch.distributions.constraints.real_vector
    codomain = torch.distributions.constraints.real_vector
    bijective = True

    def __init__(self, Qs, Ls):
        super().__init__()
        if len(Qs) != len(Ls):
            raise ValueError("Qs and Ls must have the same length.")

        self.Qs = [int(q) for q in Qs]
        self.Ls = [int(L) for L in Ls]
        for q, l in zip(self.Qs, self.Ls):
            if q < 1:
                raise ValueError("Each Q must be >= 1.")
            if l < 0 or l >= q:
                raise ValueError("Each L must satisfy 0 <= L < Q.")

        offsets = [0]
        for q in self.Qs:
            offsets.append(offsets[-1] + q)
        self._offsets = offsets

        k = len(self.Qs)
        gap_offsets = [k]
        for q in self.Qs:
            gap_offsets.append(gap_offsets[-1] + q - 1)
        self._gap_offsets = gap_offsets

        self.quantile_dim = -1

    def _call(self, x):
        qdim = self.quantile_dim
        if x.shape[qdim] != self._offsets[-1]:
            raise ValueError(
                f"Expected input size {self._offsets[-1]} at dim {qdim}, "
                f"got {x.shape[qdim]}."
            )

        out = []
        for i, (q, l) in enumerate(zip(self.Qs, self.Ls)):
            c = torch.narrow(x, qdim, i, 1)
            gap_start = self._gap_offsets[i]
            lower = torch.narrow(x, qdim, gap_start, l)
            upper = torch.narrow(x, qdim, gap_start + l, q - 1 - l)
            out.append(centergap_to_quantiles(c, lower, upper))
        return torch.cat(out, dim=qdim)

    def _inverse(self, y):
        qdim = self.quantile_dim
        if y.shape[qdim] != self._offsets[-1]:
            raise ValueError(
                f"Expected input size {self._offsets[-1]} at dim {qdim}, "
                f"got {y.shape[qdim]}."
            )

        centrals = []
        gap_parts = []
        for start, q, l in zip(self._offsets[:-1], self.Qs, self.Ls):
            yi = torch.narrow(y, qdim, start, q)
            central = torch.narrow(yi, qdim, l, 1)
            lower_gaps_linear = torch.narrow(yi, qdim, 0, l + 1).diff(dim=qdim)
            upper_gaps_linear = torch.narrow(yi, qdim, l, q - l).diff(dim=qdim)
            centrals.append(central)
            gap_parts.append(_softplus_inverse(lower_gaps_linear))
            gap_parts.append(_softplus_inverse(upper_gaps_linear))
        return torch.cat(centrals + gap_parts, dim=qdim)

    def log_abs_det_jacobian(self, x, y):
        qdim = self.quantile_dim
        if x.shape[qdim] != self._offsets[-1]:
            raise ValueError(
                f"Expected input size {self._offsets[-1]} at dim {qdim}, "
                f"got {x.shape[qdim]}."
            )

        gap_blocks = []
        for i, q in enumerate(self.Qs):
            gap_blocks.append(torch.narrow(x, qdim, self._gap_offsets[i], q - 1))
        gaps = torch.cat(gap_blocks, dim=qdim)
        return F.logsigmoid(gaps).sum(dim=(-2, -1))


def transform_centergap_posterior(posterior, Qs, Ls):
    """Convert the center-gap posterior to quantile posterior.

    Parameters
    ----------
    posterior : gpytorch.distributions.MultitaskMultivariateNormal
        The center-gap posterior distribution.
        Event shape must be ``(N, Q_1 + Q_2 + ..., Q_k)``,
        where ``Q_i`` is the number of quantiles for *i*-th output dimension.
    Qs : list of int
        The number of quantiles for each task, i.e., ``[Q_1, Q_2, ..., Q_k]``.
    Ls : list of int
        The number of lower quantiles in center-gap representation for each task,
        i.e., ``[L_1, L_2, ..., L_k]``.

    Returns
    -------
    quantile_posterior : torch.distributions.TransformedDistribution
        Posterior over quantiles, obtained by applying
        :class:`CenterGapToQuantileTransform` to a batched
        :class:`gpytorch.distributions.MultitaskMultivariateNormal`.

    Notes
    -----
    Input and output distribution has specific structure in the quantile dimension.
    See :class:`CenterGapToQuantileTransform` for details.
    """
    transform = CenterGapToQuantileTransform(Qs, Ls)
    return torch.distributions.TransformedDistribution(posterior, transform)
