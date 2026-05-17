"""Mean modules."""

import torch
from gpytorch.means import Mean

__all__ = [
    "CenterGapMean",
]


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
