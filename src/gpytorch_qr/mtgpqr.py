"""GPQR where quantiles are directly represented and treated as tasks.

Latent GPs directly construct quantiles.

It is recommended to use fewer latent GPs than the number of tasks(=quantiles)
to model the correlation structure.

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
>>> from gpytorch.variational import VariationalStrategy, LMCVariationalStrategy
>>> from gpytorch.means import ConstantMean
>>> from gpytorch.kernels import RBFKernel, ScaleKernel
>>> from gpytorch_qr.models import DirectQuantileGP
>>> from gpytorch_qr.mtgpqr import MultitaskQuantileGPLikelihood
>>> class MyGP(DirectQuantileGP):
...     def __init__(self, inducing_points, num_latents, num_quantiles):
...         N, D = inducing_points.size()
...         variational_distribution = CholeskyVariationalDistribution(
...             N,
...             batch_shape=torch.Size([num_latents]),
...         )
...         variational_strategy = LMCVariationalStrategy(
...             VariationalStrategy(
...                 self,
...                 inducing_points,
...                 variational_distribution,
...                 learn_inducing_locations=True,
...             ),
...             num_tasks=num_quantiles,
...             num_latents=num_latents,
...         )
...         mean_module = ConstantMean(batch_shape=torch.Size([num_latents]))
...         covar_module = ScaleKernel(
...             RBFKernel(ard_num_dims=D, batch_shape=torch.Size([num_latents])),
...             batch_shape=torch.Size([num_latents]),
...         )
...         super().__init__(variational_strategy, mean_module, covar_module, -1)
>>> inducing_points = torch.linspace(0, 1, 10).reshape(-1, 1)
>>> num_latents = len(q) - 2  # recommended to be smaller than q
>>> gp = MyGP(inducing_points, num_latents, len(q))
>>> likelihood = MultitaskQuantileGPLikelihood(q)
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
...     loss = -mll(output, y)
...     loss.backward()
...     optimizer.step()
...     optimizer.zero_grad()
>>> gp.eval()  # doctest: +IGNORE_OUTPUT
>>> x_pred = torch.linspace(0, 2, 100).reshape(-1, 1)
>>> with torch.no_grad():
...     quantiles = gp.mean_quantiles(x_pred)
>>> import matplotlib.pyplot as plt
>>> plt.scatter(x, y, c='gray', marker='.', alpha=0.1)  # doctest: +IGNORE_OUTPUT
>>> plt.plot(x_range, true_quantiles, '--', c='k')  # doctest: +IGNORE_OUTPUT
>>> plt.plot(x_pred, quantiles)  # doctest: +IGNORE_OUTPUT
"""

from .likelihoods import MultitaskQuantileALDLikelihood

__all__ = [
    "MultitaskQuantileGPLikelihood",
]


class MultitaskQuantileGPLikelihood(MultitaskQuantileALDLikelihood):
    """Likelihood for :class:`MultitaskQuantileALD` with direct representation."""

    def latent_to_quantiles(self, function_samples):
        return function_samples
