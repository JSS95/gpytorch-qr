"""
1D regression dataset with heteroskedastic noise:

.. plot::
   :context:

    import torch
    from torch.distributions import Normal

    def mean(x):
        return torch.cos(x * 2 * 3.14)

    def std(x):
        return x + 0.1

    x_range = torch.linspace(0, 1, 100).reshape(-1, 1)
    x = x_range.repeat(5, 1)
    y = (mean(x) + torch.randn(x.shape).mul(std(x))).squeeze()
    taus = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])
    true_quantiles = mean(x_range) + std(x_range) * Normal(0, 1).icdf(taus)
    import matplotlib.pyplot as plt
    plt.scatter(x, y, c='gray', marker='.', alpha=0.1)
    plt.plot(x_range, true_quantiles, '--', c='k')

Prior mean function for the central quantile:

.. plot::
   :context: close-figs

    from gpytorch.means import Mean

    class PriorMean(Mean):
        def forward(self, x):
            return mean(x).squeeze()
    x_pred = torch.linspace(0, 1.5, 100).reshape(-1, 1)
    plt.scatter(x, y, c='k', marker='.')
    plt.plot(x_pred, PriorMean()(x_pred).detach(), c='r')

Define multi-task Gaussian process model:

.. plot::
   :context: close-figs

    from gpytorch.variational import CholeskyVariationalDistribution
    from gpytorch.variational import VariationalStrategy
    from gpytorch.means import ConstantMean
    from gpytorch.kernels import RBFKernel, ScaleKernel
    from gpytorch_qr import CenterGapLmcVariationalStrategy, CenterGapGP

    class MTGP(CenterGapGP):
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
                num_tasks=num_quantiles,
                num_latents=num_latents,
                latent_dim=-1,
                num_lower_quantiles=num_lower_quantiles,
                num_lower_latents=num_lower_latents,
            )

            center_mean = PriorMean()
            gap_mean = ConstantMean(
                batch_shape=torch.Size([num_latents - 1])
            )
            covar_module = ScaleKernel(
                RBFKernel(ard_num_dims=D, batch_shape=torch.Size([num_latents])),
                batch_shape=torch.Size([num_latents]),
            )
            super().__init__(variational_strategy, center_mean, gap_mean, covar_module)

Define multi-task Gaussian process quantile regression model:

.. plot::
   :context: close-figs

    from gpytorch_qr import MTGPQR

    class MyModel(MTGPQR):
        def __init__(self):
            inducing_points = torch.linspace(0, 1, 20).reshape(-1, 1)
            central_tau = taus[(taus - 0.5).abs().argmin()]
            num_lower_quantiles = len(taus[taus < central_tau])
            num_latents = 9
            num_lower_latents = (num_latents - 1) // 2
            gp = MTGP(
                inducing_points=inducing_points,
                num_quantiles=len(taus),
                num_lower_quantiles=num_lower_quantiles,
                num_latents=num_latents,
                num_lower_latents=num_lower_latents,
            )
            super().__init__(taus, gp)

    model = MyModel()

Train the model:

.. plot::
   :context: close-figs

    from gpytorch.mlls import VariationalELBO

    model.train()
    mll = VariationalELBO(model.likelihood, model.gp, num_data=y.numel())
    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.01)

    for _ in range(100):
        output = model.gp(x)
        loss = -mll(output, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

Evaluate:

.. plot::
   :context: close-figs

    model.eval()
    with torch.no_grad():
        quantiles = model(x_pred).detach()

    plt.scatter(x, y, c='gray', marker='.', alpha=0.1)
    plt.plot(x_range, true_quantiles, '--', c='k')
    plt.plot(x_pred, quantiles)
"""

import gpytorch
import torch
import torch.nn.functional as F

from .ald import ALD

__all__ = [
    "centergap_to_quantiles",
    "CenterGapGP",
    "CenterGapLikelihood",
    "CenterGapLmcVariationalStrategy",
    "MTGPQR",
]


def centergap_to_quantiles(central, lower_gaps, upper_gaps):
    """Convert center-gap representation to quantiles.

    Parameters
    ----------
    central : torch.Tensor with shape (..., 1)
        The central quantile values.
    lower_gaps : torch.Tensor with shape (..., L)
        Pre-transformed lower gap values.
    upper_gaps : torch.Tensor with shape (..., U)
        Pre-transformed upper gap values.

    Returns
    -------
    quantiles : torch.Tensor with shape (..., T)
        The quantile values.
    """
    lower_gaps = F.softplus(lower_gaps)
    lower_quantiles = central - lower_gaps.flip(dims=[-1]).cumsum(dim=-1).flip(
        dims=[-1]
    )

    upper_gaps = F.softplus(upper_gaps)
    upper_quantiles = central + upper_gaps.cumsum(dim=-1)

    ret = torch.concat([lower_quantiles, central, upper_quantiles], dim=-1)
    return ret


class CenterGapGP(gpytorch.models.ApproximateGP):
    """Gaussian process modeling the center-gap representation of quantiles.

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
        """Compute distribution of the center-gap Gaussian process."""
        center_mean = self.center_mean(x)
        gap_mean = self.gap_mean(x)
        mean = torch.concatenate([center_mean.unsqueeze(0), gap_mean], dim=0)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class CenterGapLikelihood(gpytorch.likelihoods.Likelihood):
    """Likelihood of the center-gap quantile regression model.

    Parameters
    ----------
    taus : array of shape (T,)
        The quantile levels of the distribution.
    """

    def __init__(self, taus=(0.5,)):
        super().__init__()
        self.register_buffer("taus", taus)
        self.register_parameter(
            name="raw_scales",
            parameter=torch.nn.Parameter(torch.zeros(len(taus))),
        )
        self.register_constraint(
            "raw_scales",
            gpytorch.constraints.Positive(),
        )
        central_quantile = self.taus[torch.argmin(torch.abs(self.taus - 0.5))]
        self.lower_count = (self.taus < central_quantile).count_nonzero()

    @property
    def scales(self):
        # (T,)
        return self.raw_scales_constraint.transform(self.raw_scales)

    def forward(self, function_samples):
        """Compute likelihood of center-gap model."""
        # function_samples: (S, N, T)
        median = function_samples[:, :, :1]
        lower_gaps = function_samples[:, :, 1 : 1 + self.lower_count]
        upper_gaps = function_samples[:, :, 1 + self.lower_count :]
        quantiles = centergap_to_quantiles(median, lower_gaps, upper_gaps)
        return ALD(
            locs=quantiles,  # (S, N, T)
            scales=self.scales,  # (T,)
            taus=self.taus,  # (T,)
        )

    def expected_log_prob(self, observations, function_dist, *args, **kwargs):
        lp = super().expected_log_prob(
            observations, function_dist, *args, **kwargs
        )  # (N, T)
        return lp.sum(dim=1)  # (N,)

    def log_marginal(self, observations, function_dist, *args, **kwargs):
        lp = super().log_marginal(
            observations, function_dist, *args, **kwargs
        )  # (N, T)
        return lp.sum(dim=1)  # (N,)


class CenterGapLmcVariationalStrategy(gpytorch.variational.LMCVariationalStrategy):
    """LMC variational strategy for the center-gap quantile regression model."""

    def __init__(
        self,
        base_variational_strategy,
        num_tasks,  # T
        num_latents=1,  # Q
        latent_dim=-1,
        jitter_val=None,
        num_lower_quantiles=0,
        num_lower_latents=0,
    ):
        super().__init__(
            base_variational_strategy, num_tasks, num_latents, latent_dim, jitter_val
        )
        lmc_coefficients = self.lmc_coefficients.detach().clone()  # (Q, T)
        del self.lmc_coefficients

        num_upper_quantiles = num_tasks - num_lower_quantiles - 1
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


class MTGPQR(torch.nn.Module):
    """Multi-task Gaussian process quantile regression model.

    Parameters
    ----------
    taus : tensor with shape (T,)
        The quantile levels of the distribution.
    gp : CenterGapGP
        The Gaussian process modeling the center-gap representation of quantiles.
    """

    def __init__(self, taus, gp):
        super().__init__()
        self.gp = gp
        self.likelihood = CenterGapLikelihood(taus=taus)

        central_tau = taus[(taus - 0.5).abs().argmin()]
        self.num_lower_quantiles = len(taus[taus < central_tau])

    def forward(self, x):
        """Compute quantile functions."""
        function_means = self.gp(x).mean
        median = function_means[..., :1]
        lower_gaps = function_means[..., 1 : 1 + self.num_lower_quantiles]
        upper_gaps = function_means[..., 1 + self.num_lower_quantiles :]
        return centergap_to_quantiles(median, lower_gaps, upper_gaps)
