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
>>> from gpytorch_qr.gpqr import BatchQuantileGP, BatchALDLikelihood
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
>>> likelihood = BatchALDLikelihood(q)
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

import gpytorch
import torch

from .ald import BatchALD
from .base import ALDLikelihoodMixin, BayesianQRMixin

__all__ = [
    "BatchQuantileGP",
    "BatchALDLikelihood",
]


class BatchQuantileGP(gpytorch.models.ApproximateGP, BayesianQRMixin):
    """Batch approximate GP for *Q* quantiles.

    Parameters
    ----------
    variational_strategy : gpytorch.variational.VariationalStrategy
        The variational strategy.
    mean_module : gpytorch.means.Mean
        Mean module with batch shape ``(Q, [batch_shape])``.
    covar_module : gpytorch.kernels.Kernel
        Covariance module with batch shape ``(Q, [batch_shape])``.
    """

    def __init__(self, variational_strategy, mean_module, covar_module):
        super().__init__(variational_strategy)
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def joint_quantile_posterior(self, x):
        return self(x)

    def marginal_quantile_posterior(self, x):
        dist = self(x)
        return torch.distributions.Normal(dist.mean, dist.variance.sqrt())

    def mean_quantiles(self, x):
        return self(x).mean

    def mean_quantiles_mc(self, x, num_samples=10):
        dist = self(x)
        samples = dist.rsample(torch.Size([num_samples]))
        return samples.mean(dim=0)

    def quantile_quantiles(self, x, q):
        dist = self.marginal_quantile_posterior(x)
        shape = [-1] + [1 for _ in range(len(dist.batch_shape))]
        return dist.icdf(q.reshape(*shape))

    def quantile_quantiles_mc(self, x, q, num_samples=10):
        dist = self(x)
        samples = dist.rsample(torch.Size([num_samples]))
        return samples.quantile(q, dim=0)


class BatchALDLikelihood(gpytorch.likelihoods.Likelihood, ALDLikelihoodMixin):
    """Likelihood for :class:`BatchALD` with direct quantile representation.

    Parameters
    ----------
    q : torch.Tensor with shape (Q, [batch_shape])
        The quantile levels.
    raw_scales : torch.Tensor with shape (Q, [batch_shape]) or scalar, default=0
        The initial untransformed scales of the asymmetric Laplace distribution.
        The actual scales are obtained by applying the positive transformation.
        Scalar value is broadcasted to the shape of *q*.
    learn_scales : bool, default=True
        Whether to update scales by gradients.

    Notes
    -----
    The ``batch_shape`` can either be:

    - A broadcastable shape, e.g., ``(1,)``.
    - A fixed shape, e.g., ``(B,)``.

    Different batch shape representations are allowed for *q* and *raw_scales*.
    For example, *q* can be in  ``(Q, 1)`` and *raw_scales* can be in ``(Q, B)``.
    But different choices can lead to vastly different model behavior (see below).

    Usually, if there is no batch dimension, you would want to use:

    - *q* in shape ``(Q,)``.
    - *raw_scales* in scalar (usually 0).
    - ``learn_scales=True``.

    If there is batch dimension, you would usually want to use:

    - *q* in shape ``(Q, 1)``.
    - *raw_scales* in shape ``(Q, B)``.
    - ``learn_scales=True``.

    First of all, if ``learn_scales`` is False, there is no learnable parameter and
    broadcasting does not matter.
    The rest of this section focuses on the case when ``learn_scales`` is True.

    .. rubric:: q

    If batch shape of ``(1,)`` is used, same levels of quantiles are used for
    all batches.
    Should different quantile levels be desired for different batches,
    use batch shape of ``(B,)``.
    Note that it is impossible to vary the number of quantiles *Q* across batches.

    .. rubric:: raw_scales

    If batch shape of ``(1,)`` is used, same scales are used for all batches.
    This makes the scales to be updated by gradients averaged across batches.
    If different batches should be independent, use batch shape of ``(B,)``.

    The quantile dimension of *raw_scales* can be broadcasted as well, which
    makes the scales to be updated by gradients averaged across quantiles.
    Avert this if you want different quantiles to be completely independent.
    """

    def __init__(self, q, raw_scales=0.0, learn_scales=True):
        super().__init__()
        self.register_buffer("q", q.float())

        raw_scales = torch.as_tensor(raw_scales, dtype=torch.float32)
        if raw_scales.ndim == 0:
            raw_scales = torch.full_like(q, raw_scales)
        if learn_scales:
            self.register_parameter("raw_scales", torch.nn.Parameter(raw_scales))
        else:
            self.register_buffer("raw_scales", raw_scales)
        self.register_constraint("raw_scales", gpytorch.constraints.Positive())

    @property
    def scales(self):
        return self.raw_scales_constraint.transform(self.raw_scales)

    def forward(self, function_samples):
        return BatchALD(
            m=function_samples,
            lamda=self.scales,
            kappa=self.q,
        )

    def expected_log_prob(self, observations, function_dist, *args, **kwargs):
        lp = super().expected_log_prob(observations, function_dist, *args, **kwargs)
        return lp.sum(dim=0)
