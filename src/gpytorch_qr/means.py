"""Mean modules."""

import torch
from gpytorch.means import Mean

__all__ = [
    "CenterGapMean",
]


class CenterGapMean(Mean):
    r"""Mean module for the center-gap representation.

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

    See Also
    --------
    gpytorch_qr.variational.CenterGapLMCVariationalStrategy :
        LMC variational strategy that needs this mean module.

    Notes
    -----
    Input predictors are expected to have shape ``(*B, 1, N, D)``.
    *N* is the number of data points and *D* is the number of input dimensions.

    .. hint::

        If this mean is used with LMC variational strategy, the prior means will be
        placed on latent GPs instead of the output GPs.
        This may lead to unexpected behavior by the linear combination of the
        latent GPs.

        Using :class:`gpytorch_qr.variational.CenterGapLMCVariationalStrategy` is a
        quick way to separate the prior means for the central quantiles and the gap
        functions.

        For more fundamental solution, consider modifying the input observations by
        :math:`y \leftarrow y - \mu(x)` and use a standard LMC variational strategy to
        model the residuals.
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
