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
        Should have batch shape ``(*B, 1)``.
    gap_mean : gpytorch.means.Mean
        Mean module for the quantile gaps.
        Should have batch shape ``(*B, L-1)`` where *L* is the number of latent GPs.

    Notes
    -----
    Input predictors are expected to have shape ``(*B, 1, N, D)``.
    *N* is the number of data points and *D* is the number of input dimensions.
    """

    def __init__(self, center_mean, gap_mean):
        super().__init__()
        self.center_mean = center_mean
        self.gap_mean = gap_mean

    def forward(self, x):
        """Compute the mean of center-gap representation.

        Parameters
        ----------
        x : torch.Tensor in shape ``(*B, 1, N, D)``
            *N* is the number of data points and *D* is the number of input dimensions.
        """
        center_mean = self.center_mean(x)  # (*B, 1, N)
        gap_mean = self.gap_mean(x)
        return torch.concat([center_mean, gap_mean], dim=-2)
