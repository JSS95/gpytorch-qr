"""GPQR where quantiles are represented by center-gap and treated as batches.

>>> import torch
>>> from torch.distributions import Normal
>>> torch.manual_seed(42)  # doctest: +IGNORE_OUTPUT
>>> def mean(x):
...     return torch.cos(x * 2 * 3.14)
>>> def std(x):
...     return x + 0.1
>>> x_range = torch.linspace(0, 1, 10).reshape(-1, 1)
>>> x = x_range.repeat(2, 1)
>>> y = (mean(x) + torch.randn(x.shape).mul(std(x))).squeeze()
>>> q = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
>>> true_quantiles = mean(x_range) + std(x_range) * Normal(0, 1).icdf(q)
>>> from gpytorch.variational import CholeskyVariationalDistribution
>>> from gpytorch.variational import VariationalStrategy
>>> from gpytorch.means import ConstantMean
>>> from gpytorch.kernels import RBFKernel, ScaleKernel
>>> from gpytorch_qr.centergap import CenterGapMean
>>> from gpytorch_qr.gpqr_cg import (
...     BatchCenterGapQuantileGP,
...     BatchCenterGapQuantileGPLikelihood,
... )
>>> class MyGP(BatchCenterGapQuantileGP):
...     def __init__(self, inducing_points, num_quantiles, num_lower_quantiles):
...         N, D = inducing_points.size()
...         variational_distribution = CholeskyVariationalDistribution(
...             N,
...             batch_shape=torch.Size([num_quantiles]),
...         )
...         variational_strategy = VariationalStrategy(
...             self,
...             inducing_points,
...             variational_distribution,
...             learn_inducing_locations=True,
...         )
...         mean = CenterGapMean(
...             ConstantMean(batch_shape=torch.Size([1])),
...             ConstantMean(batch_shape=torch.Size([num_quantiles - 1])),
...         )
...         covar = ScaleKernel(
...             RBFKernel(ard_num_dims=D, batch_shape=torch.Size([num_quantiles])),
...             batch_shape=torch.Size([num_quantiles]),
...         )
...         super().__init__(variational_strategy, mean, covar, num_lower_quantiles)
>>> inducing_points = torch.linspace(0, 1, 10).reshape(-1, 1)
>>> central_q_index = (q - 0.5).abs().argmin().item()
>>> gp = MyGP(inducing_points, len(q), central_q_index)
>>> likelihood = BatchCenterGapQuantileGPLikelihood(q, central_q_index)
>>> from gpytorch.mlls import VariationalELBO
>>> gp.train()  # doctest: +IGNORE_OUTPUT
>>> likelihood.train()  # doctest: +IGNORE_OUTPUT
>>> mll = VariationalELBO(likelihood, gp, num_data=y.numel())
>>> optimizer = torch.optim.Adam(
...     list(gp.parameters()) + list(likelihood.parameters()),
...     lr=0.001,
... )
>>> N = 1  # Set to 1 for faster training; increase for better performance
>>> for _ in range(N):
...     output = gp(x)
...     loss = -mll(output, y).sum()
...     loss.backward()
...     optimizer.step()
...     optimizer.zero_grad()
>>> gp.eval()  # doctest: +IGNORE_OUTPUT
>>> x_pred = torch.linspace(0, 2, 100).reshape(-1, 1)
>>> with torch.no_grad():
...     quantiles = gp.mean_quantiles_mc(x_pred)
>>> import matplotlib.pyplot as plt
>>> plt.scatter(x, y, c='gray', marker='.', alpha=0.1)  # doctest: +IGNORE_OUTPUT
>>> plt.plot(x_range, true_quantiles, '--', c='k')  # doctest: +IGNORE_OUTPUT
>>> plt.plot(x_pred, quantiles.T)  # doctest: +IGNORE_OUTPUT
"""

from .ald import BatchQuantileALDLikelihood
from .centergap import centergap_to_quantiles
from .gp import CenterGapGPQR

__all__ = [
    "BatchCenterGapQuantileGP",
    "BatchCenterGapQuantileGPLikelihood",
]


class BatchCenterGapQuantileGP(CenterGapGPQR):
    """Approximate GP with center-gap representation and batch quantiles.

    Parameters
    ----------
    variational_strategy
        The variational strategy.
        Must wrap a variational distribution with batch shape ``(Q, *B)``,
        where *Q* is the number of quantiles and *B* is additional batch shape.
    mean_module : gpytorch_qr.centergap.CenterGapMean
        Mean module for center-gap representation with batch shape ``(Q, *B)``.
    covar_module
        Covariance module with batch shape ``(Q, *B)``.
    num_lower_quantiles : int
        The number of lower quantiles in center-gap representation.

    Notes
    -----
    Posterior distribution is :class:`gpytorch.distributions.MultivariateNormal`
    with batch shape ``(Q, *B)`` and event shape ``(N,)``
    for input of shape ``(*B, N, D)``.

    MLL loss is a tensor of shape ``(Q, *B)``.
    """


class BatchCenterGapQuantileGPLikelihood(BatchQuantileALDLikelihood):
    """Likelihood for :class:`BatchQuantileALD` with center-gap representation.

    Parameters
    ----------
    q
    central_quantile_index : int
        The index of the central quantile in the quantile levels.
    raw_scales
    learn_scales
    """

    def __init__(self, q, central_quantile_index, raw_scales=0.0, learn_scales=True):
        super().__init__(q, raw_scales, learn_scales)
        central_quantile = self.q[central_quantile_index]
        self.lower_count = (self.q < central_quantile).count_nonzero()

    def latent_to_quantiles(self, function_samples):
        center = function_samples[:, :1, ...]
        lower_gaps = function_samples[:, 1 : 1 + self.lower_count, ...]
        upper_gaps = function_samples[:, 1 + self.lower_count :, ...]
        quantiles = centergap_to_quantiles(
            center, lower_gaps, upper_gaps, quantile_dim=1
        )
        return quantiles
