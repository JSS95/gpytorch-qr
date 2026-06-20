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
    center_mean : gpytorch.means.Mean or torch.nn.ModuleList of gpytorch.means.Mean
        Mean module for the central quantile.
        If a ``torch.nn.ModuleList`` is provided, each of the contained mean modules
        applies to a different output dimension.
        Each mean should have batch shape ``(*B, 1)``.
    gap_mean : gpytorch.means.Mean
        Mean module for the quantile gaps.
        Should have batch shape ``(*B, L-k)`` where *L* is the number of latent GPs
        and *k* is the number of output dimensions.

    Notes
    -----
    Input predictors are expected to have shape ``(*B, 1, N, D)``.
    *N* is the number of data points and *D* is the number of input dimensions.
    """

    def __init__(self, center_mean, gap_mean):
        super().__init__()
        if not isinstance(center_mean, torch.nn.ModuleList):
            center_mean = torch.nn.ModuleList([center_mean])
        self.center_mean = center_mean
        self.gap_mean = gap_mean

    def forward(self, x):
        """Compute the mean of center-gap representation.

        Parameters
        ----------
        x : torch.Tensor in shape ``(*B, 1, N, D)``
            *N* is the number of data points and *D* is the number of input dimensions.

        Returns
        -------
        torch.Tensor in shape ``(*B, L, N)``
        """
        center_mean = torch.concat(
            [m(x) for m in self.center_mean], dim=-2
        )  # (*B, k, N)
        gap_mean = self.gap_mean(x)  # (*B, L-k, N)
        return torch.concat([center_mean, gap_mean], dim=-2)
