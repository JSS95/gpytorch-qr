"""Batch independent GPQR with center-gap representation.

.. code-block:: python
   :caption: Example
   :linenos:

    import torch
    from torch.distributions import Normal

    torch.manual_seed(42)

    def mean(x):
        return torch.cos(x * 2 * 3.14)

    def std(x):
        return x + 0.1

    x_range = torch.linspace(0, 1, 100).reshape(-1, 1)
    x = x_range.repeat(5, 1)
    y = (mean(x) + torch.randn(x.shape).mul(std(x))).squeeze()
    q = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
    true_quantiles = mean(x_range) + std(x_range) * Normal(0, 1).icdf(q)

    from gpytorch.variational import CholeskyVariationalDistribution
    from gpytorch.variational import VariationalStrategy
    from gpytorch.means import ConstantMean
    from gpytorch.kernels import RBFKernel, ScaleKernel
    from gpytorch_qr.gpqr_cg import (
        BatchCenterGapQuantileGP,
        BatchCenterGapALDLikelihood,
    )

    class MyGP(BatchCenterGapQuantileGP):
        def __init__(self, inducing_points, num_quantiles):
            N, D = inducing_points.size()
            variational_distribution = CholeskyVariationalDistribution(
                N,
                batch_shape=torch.Size([num_quantiles]),
            )
            variational_strategy = VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=True,
            )

            center_mean = ConstantMean()
            gap_mean = ConstantMean(
                batch_shape=torch.Size([num_quantiles - 1])
            )
            covar = ScaleKernel(
                RBFKernel(ard_num_dims=D, batch_shape=torch.Size([num_quantiles])),
                batch_shape=torch.Size([num_quantiles]),
            )
            super().__init__(variational_strategy, center_mean, gap_mean, covar)

    inducing_points = torch.linspace(0, 1, 10).reshape(-1, 1)
    central_q_index = 2
    gp = MyGP(inducing_points, len(q))
    likelihood = BatchCenterGapALDLikelihood(q, central_q_index)

    from gpytorch.mlls import VariationalELBO

    gp.train()
    likelihood.train()
    mll = VariationalELBO(likelihood, gp, num_data=y.numel())
    optimizer = torch.optim.Adam(
        list(gp.parameters()) + list(likelihood.parameters()),
        lr=0.001,
    )

    for _ in range(1000):
        output = gp(x)
        loss = -mll(output, y).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    gp.eval()
    x_pred = torch.linspace(0, 2, 100).reshape(-1, 1)
    with torch.no_grad():
        quantiles = gp.mean_quantiles(x_pred, central_q_index).detach()

    import matplotlib.pyplot as plt
    plt.scatter(x, y, c='gray', marker='.', alpha=0.1)
    plt.plot(x_range, true_quantiles, '--', c='k')
    plt.plot(x_pred, quantiles.T)
"""

import gpytorch
import torch

from .ald import BatchALD
from .centergap import centergap_to_quantiles

__all__ = [
    "BatchCenterGapQuantileGP",
    "BatchCenterGapALDLikelihood",
]


class BatchCenterGapQuantileGP(gpytorch.models.ApproximateGP):
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
    """

    def __init__(self, variational_strategy, center_mean, gap_mean, covar_module):
        super().__init__(variational_strategy)
        self.center_mean = center_mean
        self.gap_mean = gap_mean
        self.covar_module = covar_module

    def forward(self, x):
        center_mean = self.center_mean(x)
        gap_mean = self.gap_mean(x)
        mean = torch.concat([center_mean.unsqueeze(0), gap_mean], dim=0)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def mean_quantiles(self, x, num_lower_quantiles):
        """Predict quantiles by posterior mean.

        Parameters
        ----------
        x : torch.Tensor with shape (N, D)
            The input locations.
        num_lower_quantiles : int
            The number of lower quantiles in center-gap representation.

        Returns
        -------
        quantiles : torch.Tensor with shape (Q, N)
            The predicted quantiles at the input locations.
        """
        function_means = self(x).mean.T  # (N, Q)
        median = function_means[..., :1]
        lower_gaps = function_means[..., 1 : 1 + num_lower_quantiles]
        upper_gaps = function_means[..., 1 + num_lower_quantiles :]
        return centergap_to_quantiles(median, lower_gaps, upper_gaps).T


class BatchCenterGapALDLikelihood(gpytorch.likelihoods.Likelihood):
    """ALD likelihood for batch quantile regression with center-gap representation.

    Parameters
    ----------
    q : torch.Tensor with shape (Q,)
        The quantile levels.
    central_quantile_index : int
        The index of the central quantile in the quantile levels.
    """

    def __init__(self, q, central_quantile_index):
        super().__init__()
        self.register_buffer("q", q.float())
        self.register_parameter(
            "raw_scales",
            torch.nn.Parameter(torch.zeros(len(q))),
        )
        self.register_constraint("raw_scales", gpytorch.constraints.Positive())
        central_quantile = self.q[central_quantile_index]
        self.lower_count = (self.q < central_quantile).count_nonzero()

    @property
    def scales(self):
        return self.raw_scales_constraint.transform(self.raw_scales)

    def forward(self, function_samples):
        """Return the ALD distribution for the given function samples.

        Parameters
        ----------
        function_samples : torch.Tensor with shape (S, 1 + L + U, N)
            The function samples drawn from the posterior distributions of quantile
            functions. *S* is the number of samples, *L* is the number of lower
            quantiles, *U* is the number of upper quantiles, and *N* is the number of
            data points,
            The first dimension in the second axis corresponds to the central quantile,
            followed by lower quantiles and then upper quantiles.
        """
        function_samples = function_samples.permute(0, 2, 1)  # (S, N, Q)
        center = function_samples[:, :, :1]
        lower_gaps = function_samples[:, :, 1 : 1 + self.lower_count]
        upper_gaps = function_samples[:, :, 1 + self.lower_count :]
        quantiles = centergap_to_quantiles(center, lower_gaps, upper_gaps)
        quantiles = quantiles.permute(0, 2, 1)  # (S, Q, N)
        return BatchALD(
            m=quantiles,  # (S, Q, N)
            lamda=self.scales,  # (Q,)
            kappa=self.q,  # (Q,)
        )

    def expected_log_prob(self, observations, function_dist, *args, **kwargs):
        # lp: (Q, N)
        lp = super().expected_log_prob(observations, function_dist, *args, **kwargs)
        return lp.sum(dim=0)  # (N,)
