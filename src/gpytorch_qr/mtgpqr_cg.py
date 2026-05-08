"""Multitask GPQR with center-gap representation.

Latent GPs model the central quantile and the gaps between quantiles separately.

.. plot::
   :context: reset
   :include-source: False

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
    from gpytorch_qr.mtgpqr_cg import (
        MultitaskCenterGapQuantileGP,
        CenterGapLmcVariationalStrategy,
        MultitaskCenterGapALDLikelihood,
    )

    class MyGP(MultitaskCenterGapQuantileGP):
        def __init__(
            self,
            inducing_points,
            num_quantiles,
            num_lower_quantiles,
            num_latents,
            num_lower_latents,
        ):
            N, D = inducing_points.size()
            variational_distribution = CholeskyVariationalDistribution(
                N,
                batch_shape=torch.Size([num_latents]),
            )
            variational_strategy = CenterGapLmcVariationalStrategy(
                VariationalStrategy(
                    self,
                    inducing_points,
                    variational_distribution,
                    learn_inducing_locations=True,
                ),
                num_quantiles=num_quantiles,
                num_latents=num_latents,
                num_lower_quantiles=num_lower_quantiles,
                num_lower_latents=num_lower_latents,
            )

            center_mean = ConstantMean()
            gap_mean = ConstantMean(
                batch_shape=torch.Size([num_latents - 1])
            )
            covar_module = ScaleKernel(
                RBFKernel(ard_num_dims=D, batch_shape=torch.Size([num_latents])),
                batch_shape=torch.Size([num_latents]),
            )
            super().__init__(variational_strategy, center_mean, gap_mean, covar_module)

    inducing_points = torch.linspace(0, 1, 10).reshape(-1, 1)
    central_q_index = 2
    num_latents = len(q)
    gp = MyGP(inducing_points, len(q), central_q_index, num_latents, num_latents // 2)
    likelihood = MultitaskCenterGapALDLikelihood(q, central_q_index)

    from gpytorch.mlls import VariationalELBO

    gp.train()
    likelihood.train()
    mll = VariationalELBO(likelihood, gp, num_data=y.numel())
    optimizer = torch.optim.Adam(
        list(gp.parameters()) + list(likelihood.parameters()),
        lr=0.01,
    )

    for _ in range(100):
        output = gp(x)
        loss = -mll(output, y)
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
    plt.plot(x_pred, quantiles)
"""

import gpytorch
import torch

from .ald import MultitaskALD
from .centergap import centergap_to_quantiles

__all__ = [
    "MultitaskCenterGapQuantileGP",
    "MultitaskCenterGapALDLikelihood",
    "CenterGapLmcVariationalStrategy",
]


class MultitaskCenterGapQuantileGP(gpytorch.models.ApproximateGP):
    """Multitask approximate GP for multiple quantiles using center-gap representation.

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
        quantiles : torch.Tensor with shape (N, Q)
            The predicted quantiles at the input locations.
        """
        function_means = self(x).mean  # (N, Q)
        median = function_means[..., :1]
        lower_gaps = function_means[..., 1 : 1 + num_lower_quantiles]
        upper_gaps = function_means[..., 1 + num_lower_quantiles :]
        return centergap_to_quantiles(median, lower_gaps, upper_gaps)


class MultitaskCenterGapALDLikelihood(gpytorch.likelihoods.Likelihood):
    """ALD likelihood for multitask quantile regression with center-gap representation.

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
        function_samples : torch.Tensor with shape (S, N, 1 + L + U)
            The function samples drawn from the posterior distributions of quantile
            functions. *S* is the number of samples, *N* is the number of data points,
            *L* is the number of lower quantiles, and *U* is the number of upper
            quantiles.
            The first channel corresponds to the central quantile,
            the next *L* channels correspond to the lower gaps,
            and the last *U* channels correspond to the upper gaps.
        """
        center = function_samples[:, :, :1]
        lower_gaps = function_samples[:, :, 1 : 1 + self.lower_count]
        upper_gaps = function_samples[:, :, 1 + self.lower_count :]
        quantiles = centergap_to_quantiles(center, lower_gaps, upper_gaps)
        return MultitaskALD(
            m=quantiles,  # (S, N, Q)
            lamda=self.scales,  # (Q,)
            kappa=self.q,  # (Q,)
        )

    def expected_log_prob(self, observations, function_dist, *args, **kwargs):
        lp = super().expected_log_prob(
            observations, function_dist, *args, **kwargs
        )  # (N, Q)
        return lp.sum(dim=1)  # (N,)


class CenterGapLmcVariationalStrategy(gpytorch.variational.LMCVariationalStrategy):
    """LMC variational strategy for the center-gap quantile regression model.

    This class modifies the standard LMC coefficients to fit the center-gap
    representation.
    The first latent function directly represents the central quantile, and it
    does not form any linear combinations with the other latent functions.
    The remaining latent functions are linearly combined to model the gap
    functions between quantiles. Upper and lower gap functions are modeled
    separately.
    """

    def __init__(
        self,
        base_variational_strategy,
        num_quantiles,  # Q
        num_latents,  # T
        num_lower_quantiles,
        num_lower_latents,
        latent_dim=-1,
        jitter_val=None,
    ):
        super().__init__(
            base_variational_strategy,
            num_quantiles,
            num_latents,
            latent_dim,
            jitter_val,
        )
        lmc_coefficients = self.lmc_coefficients.detach().clone()  # (T, Q)
        del self.lmc_coefficients

        num_upper_quantiles = num_quantiles - num_lower_quantiles - 1
        num_upper_latents = num_latents - num_lower_latents - 1
        self.register_buffer("g0_coeff", torch.ones((1, 1)))
        self.register_parameter(
            "lower_lmc_coefficients",
            torch.nn.Parameter(
                lmc_coefficients[1 : 1 + num_lower_latents, 1 : 1 + num_lower_quantiles]
            ),
        )
        self.register_parameter(
            "upper_lmc_coefficients",
            torch.nn.Parameter(
                lmc_coefficients[-num_upper_latents:, -num_upper_quantiles:]
            ),
        )

    @property
    def lmc_coefficients(self):
        return torch.block_diag(
            self.g0_coeff,
            self.lower_lmc_coefficients,
            self.upper_lmc_coefficients,
        )
