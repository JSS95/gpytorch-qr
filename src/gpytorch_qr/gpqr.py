"""GPQR where quantiles are directly represented and treated as batches.

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
>>> from gpytorch_qr.gpqr import BatchQuantileGP, BatchQuantileGPLikelihood
>>> class MyGP(BatchQuantileGP):
...     def __init__(self, inducing_points, num_quantiles):
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
...         mean = ConstantMean(batch_shape=torch.Size([num_quantiles]))
...         covar = ScaleKernel(
...             RBFKernel(ard_num_dims=D, batch_shape=torch.Size([num_quantiles])),
...             batch_shape=torch.Size([num_quantiles]),
...         )
...         super().__init__(variational_strategy, mean, covar)
>>> inducing_points = torch.linspace(0, 1, 10).reshape(-1, 1)
>>> gp = MyGP(inducing_points, len(q))
>>> likelihood = BatchQuantileGPLikelihood(q)
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
...     mean_q = gp.mean_quantiles(x_pred)
>>> import matplotlib.pyplot as plt
>>> plt.scatter(x, y, c='gray', marker='.', alpha=0.1)  # doctest: +IGNORE_OUTPUT
>>> plt.plot(x_range, true_quantiles, '--', c='k')  # doctest: +IGNORE_OUTPUT
>>> plt.plot(x_pred, mean_q.T)  # doctest: +IGNORE_OUTPUT
"""

from .ald import BatchQuantileALDLikelihood
from .gp import DirectGPQR

__all__ = [
    "BatchQuantileGP",
    "BatchQuantileGPLikelihood",
]


class BatchQuantileGP(DirectGPQR):
    """Batch approximate GP for *Q* quantiles.

    Parameters
    ----------
    variational_strategy : gpytorch.variational.VariationalStrategy
        The variational strategy.
        Must wrap a variational distribution with batch shape ``(Q, *B)``,
        where *Q* is the number of quantiles.
    mean_module : gpytorch.means.Mean
        Mean module with batch shape ``(Q, *B)``.
    covar_module : gpytorch.kernels.Kernel
        Covariance module with batch shape ``(Q, *B)``.
    """


class BatchQuantileGPLikelihood(BatchQuantileALDLikelihood):
    """Likelihood for :class:`BatchQuantileALD` with direct representation."""

    def latent_to_quantiles(self, function_samples):
        return function_samples
