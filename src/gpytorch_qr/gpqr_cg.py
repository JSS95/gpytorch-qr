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
>>> from gpytorch_qr.gpqr_cg import (
...     BatchCenterGapQuantileGP,
...     BatchCenterGapALDLikelihood,
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
...         center_mean = ConstantMean()
...         gap_mean = ConstantMean(
...             batch_shape=torch.Size([num_quantiles - 1])
...         )
...         covar = ScaleKernel(
...             RBFKernel(ard_num_dims=D, batch_shape=torch.Size([num_quantiles])),
...             batch_shape=torch.Size([num_quantiles]),
...         )
...         super().__init__(
...             variational_strategy, center_mean, gap_mean, covar, num_lower_quantiles
...         )
>>> inducing_points = torch.linspace(0, 1, 10).reshape(-1, 1)
>>> central_q_index = (q - 0.5).abs().argmin().item()
>>> gp = MyGP(inducing_points, len(q), central_q_index)
>>> likelihood = BatchCenterGapALDLikelihood(q, central_q_index)
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

import gpytorch
import torch

from .ald import ALDLikelihood, BatchALD
from .base import BayesianQRMixin
from .centergap import centergap_to_quantiles, transform_centergap_posterior

__all__ = [
    "BatchCenterGapQuantileGP",
    "BatchCenterGapALDLikelihood",
]


class BatchCenterGapQuantileGP(gpytorch.models.ApproximateGP, BayesianQRMixin):
    """Batch approximate GP for multiple quantiles using center-gap representation.

    Parameters
    ----------
    variational_strategy : gpytorch.variational.VariationalStrategy
        The variational strategy for the Gaussian process.
    center_mean : gpytorch.means.Mean
        The mean module for the central quantile.
    gap_mean : gpytorch.means.Mean
        The mean module for the gaps between quantiles.
    covar_module : gpytorch.kernels.Kernel
        The covariance module for the Gaussian process.
    num_lower_quantiles : int
        The number of lower quantiles in center-gap representation.
    """

    def __init__(
        self,
        variational_strategy,
        center_mean,
        gap_mean,
        covar_module,
        num_lower_quantiles,
    ):
        super().__init__(variational_strategy)
        self.center_mean = center_mean
        self.gap_mean = gap_mean
        self.covar_module = covar_module
        self.num_lower_quantiles = num_lower_quantiles

    def forward(self, x):
        center_mean = self.center_mean(x)
        gap_mean = self.gap_mean(x)
        mean = torch.concat([center_mean.unsqueeze(0), gap_mean], dim=0)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def joint_quantile_posterior(self, x):
        return transform_centergap_posterior(self(x), self.num_lower_quantiles)

    def mean_quantiles_mc(self, x, num_samples=10):
        dist = self.joint_quantile_posterior(x)
        samples = dist.rsample(torch.Size([num_samples]))
        return samples.mean(dim=0)

    def quantile_quantiles_mc(self, x, q, num_samples=10):
        dist = self.joint_quantile_posterior(x)
        samples = dist.rsample(torch.Size([num_samples]))
        return samples.quantile(q, dim=0)


class BatchCenterGapALDLikelihood(ALDLikelihood):
    """Likelihood for :class:`BatchALD` with center-gap representation.

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

    def forward(self, function_samples):
        """Return the ALD distribution for the given function samples.

        Parameters
        ----------
        function_samples : torch.Tensor with shape (S, 1 + L + U, [batch_shape], N)
            The function samples drawn from the posterior distributions of quantile
            functions. *S* is the number of samples, *L* is the number of lower
            quantiles, *U* is the number of upper quantiles, and *N* is the number of
            data points.
        """
        center = function_samples[:, :1, ...]
        lower_gaps = function_samples[:, 1 : 1 + self.lower_count, ...]
        upper_gaps = function_samples[:, 1 + self.lower_count :, ...]
        quantiles = centergap_to_quantiles(
            center, lower_gaps, upper_gaps, quantile_dim=1
        )
        return BatchALD(
            m=quantiles,
            lamda=self.scales,
            kappa=self.q,
        )

    def expected_log_prob(self, observations, function_dist, *args, **kwargs):
        lp = super().expected_log_prob(observations, function_dist, *args, **kwargs)
        return lp.sum(dim=0)
