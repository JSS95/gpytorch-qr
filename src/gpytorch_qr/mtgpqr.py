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
>>> from gpytorch_qr.mtgpqr import MultitaskQuantileGP, MultitaskALDLikelihood
>>> class MyGP(MultitaskQuantileGP):
...     def __init__(self, inducing_points, num_quantiles, num_latents):
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
...         super().__init__(variational_strategy, mean_module, covar_module)
>>> inducing_points = torch.linspace(0, 1, 10).reshape(-1, 1)
>>> num_latents = len(q) - 2  # recommended to be smaller than q
>>> gp = MyGP(inducing_points, len(q), num_latents=num_latents)
>>> likelihood = MultitaskALDLikelihood(q)
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

import gpytorch
import torch

from .ald import ALDLikelihood, MultitaskALD
from .base import BayesianQRMixin

__all__ = [
    "MultitaskQuantileGP",
    "MultitaskALDLikelihood",
]


class MultitaskQuantileGP(gpytorch.models.ApproximateGP, BayesianQRMixin):
    """Multitask approximate GP for multiple quantiles.

    Parameters
    ----------
    variational_strategy : gpytorch.variational.VariationalStrategy
        The variational strategy.
    mean_module : gpytorch.means.Mean
        Mean module with batch shape equal to the number of latent GPs.
    covar_module : gpytorch.kernels.Kernel
        Covariance module with batch shape equal to the number of latent GPs.
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
        """Joint posterior over quantiles at input locations.

        Parameters
        ----------
        x : torch.Tensor with shape (N, D)
            The input locations.

        Returns
        -------
        distribution : gpytorch.distributions.MultitaskMultivariateNormal
            Joint posterior over quantiles at input locations.
            ``mean`` has shape (N, Q) and ``covariance_matrix`` has shape (N*Q, N*Q),
            where *Q* is the number of quantiles and *N* is the number of data points.
        """
        return self(x)

    def marginal_quantile_posterior(self, x):
        """Marginal posterior over quantiles.

        Parameters
        ----------
        x : torch.Tensor with shape (N, D)
            The input locations.

        Returns
        -------
        distribution : torch.distributions.Normal
            Marginal posterior over quantiles at input locations.
            ``loc`` has shape (N, Q) and ``scale`` has shape (N, Q),
            where *Q* is the number of quantiles and *N* is the number of data points.
        """
        dist = self(x)
        return torch.distributions.Normal(dist.mean, dist.variance.sqrt())

    def mean_quantiles(self, x):
        """Predict quantiles by analytical posterior mean.

        Parameters
        ----------
        x : torch.Tensor with shape (N, D)
            The input locations.

        Returns
        -------
        quantiles : torch.Tensor with shape (N, Q)
            The predicted quantiles at the input locations.
        """
        return self(x).mean

    def mean_quantiles_mc(self, x, num_samples=10):
        """Predict quantiles by Monte Carlo mean of the quantile posterior.

        Parameters
        ----------
        x : torch.Tensor with shape (N, D)
            The input locations.
        num_samples : int, default=10
            Number of MC samples used to estimate the mean.

        Returns
        -------
        quantiles : torch.Tensor with shape (N, Q)
            The predicted quantiles at the input locations.
            *Q* is the number of quantiles and *N* is the number of data points.
        """
        dist = self(x)
        samples = dist.rsample(torch.Size([num_samples]))  # (num_samples, N, Q)
        return samples.mean(dim=0)  # (N, Q)

    def quantile_quantiles(self, x, q):
        """Analytic quantile of quantile posterior.

        Parameters
        ----------
        x : torch.Tensor with shape (N, D)
            The input locations.
        q : torch.Tensor with shape (q,)
            The quantile levels.

        Returns
        -------
        quantiles : torch.Tensor with shape (q, N, Q)
            The predicted quantiles at the input locations.
            *Q* is the number of quantiles and *N* is the number of data points.
        """
        dist = self.marginal_quantile_posterior(x)
        return dist.icdf(q.reshape(-1, 1, 1))  # (q, N, Q)

    def quantile_quantiles_mc(self, x, q, num_samples=10):
        """Monte Carlo quantile of quantile posterior.

        Parameters
        ----------
        x : torch.Tensor with shape (N, D)
            The input locations.
        q : torch.Tensor with shape (q,)
            The quantile levels.
        num_samples : int, default=10
            Number of MC samples used to estimate the quantiles.

        Returns
        -------
        quantiles : torch.Tensor with shape (q, N, Q)
            The predicted quantiles at the input locations.
            *Q* is the number of quantiles and *N* is the number of data points.
        """
        dist = self(x)
        samples = dist.rsample(torch.Size([num_samples]))  # (num_samples, N, Q)
        return samples.quantile(q, dim=0)  # (q, N, Q)


class MultitaskALDLikelihood(ALDLikelihood):
    """Likelihood for :class:`MultitaskALD` with direct quantile representation."""

    @property
    def scales(self):
        return self.raw_scales_constraint.transform(self.raw_scales)

    def forward(self, function_samples):
        """Return the ALD distribution for the given function samples.

        Parameters
        ----------
        function_samples : torch.Tensor with shape (S, N, Q)
            The function samples drawn from the posterior distributions of quantile
            functions. *S* is the number of samples, *N* is the number of data points,
            and *Q* is the number of quantiles.
        """
        return MultitaskALD(
            m=function_samples,  # (S, N, Q)
            lamda=self.scales,  # (Q,)
            kappa=self.q,  # (Q,)
        )

    def expected_log_prob(self, observations, function_dist, *args, **kwargs):
        lp = super().expected_log_prob(
            observations, function_dist, *args, **kwargs
        )  # (N, Q)
        return lp.sum(dim=1)  # (N,)
