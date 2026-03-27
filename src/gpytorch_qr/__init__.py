"""
1D regression dataset with heteroskedastic noise:

.. plot::
   :context:

    import torch

    x = torch.linspace(0, 1, 100).repeat(5).reshape(-1, 1)
    y = (torch.cos(x * 2 * 3.14) + torch.randn(x.shape).mul(x + 0.1)).squeeze()
    import matplotlib.pyplot as plt
    plt.scatter(x, y, c='k', marker='.')

Prior mean function for the central quantile:

.. plot::
   :context: close-figs

    from gpytorch.means import Mean

    class PriorMean(Mean):
        def forward(self, x):
            return torch.cos(x * 2 * 3.14).squeeze()
    x_pred = torch.linspace(0, 1.5, 100).reshape(-1, 1)
    plt.scatter(x, y, c='k', marker='.')
    plt.plot(x_pred, PriorMean()(x_pred).detach(), c='r')

Define multi-task Gaussian process model:

.. plot::
   :context: close-figs

    from gpytorch.models import ApproximateGP
    from gpytorch.variational import CholeskyVariationalDistribution
    from gpytorch.variational import VariationalStrategy
    from gpytorch.means import ConstantMean
    from gpytorch.kernels import RBFKernel, ScaleKernel
    from gpytorch_qr import QrLmcVariationalStrategy, CenterGapModel

    taus = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])
    central_tau = taus[(taus - 0.5).abs().argmin()]
    num_lower_quantiles = len(taus[taus < central_tau])

    class MTGP(CenterGapModel):
        def __init__(self, inducing_points):
            Q = 9
            N, D = inducing_points.size()
            variational_distribution = CholeskyVariationalDistribution(
                N,
                batch_shape=torch.Size([Q]),
            )
            variational_strategy = QrLmcVariationalStrategy(
                VariationalStrategy(
                    self,
                    inducing_points,
                    variational_distribution,
                    learn_inducing_locations=True,
                ),
                num_tasks=len(taus),
                num_latents=Q,
                latent_dim=-1,
                num_lower_quantiles=num_lower_quantiles,
                num_lower_latents=(Q - 1) // 2,
            )
            super().__init__(variational_strategy)
            self.register_buffer("taus", taus)

            self.center_mean = PriorMean()
            self.gap_mean = ConstantMean(
                batch_shape=torch.Size([Q - 1])
            )
            self.covar_module = ScaleKernel(
                RBFKernel(ard_num_dims=D, batch_shape=torch.Size([Q])),
                batch_shape=torch.Size([Q]),
            )

    inducing_points = torch.linspace(0, 1, 20).reshape(-1, 1)
    model = MTGP(inducing_points)

Define likelihood:

.. plot::
   :context: close-figs

    from gpytorch_qr import CenterGapLikelihood

    likelihood = CenterGapLikelihood(taus=model.taus)

Train the model:

.. plot::
   :context: close-figs

    from gpytorch.mlls import VariationalELBO

    model.train()
    likelihood.train()
    parameters = list(model.parameters()) + list(likelihood.parameters())
    mll = VariationalELBO(likelihood, model, num_data=y.numel())
    optimizer = torch.optim.Adam(parameters, lr=0.01)

    for _ in range(100):
        output = model(x)
        loss = -mll(output, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

Evaluate:

.. plot::
   :context: close-figs

    from gpytorch_qr import centergap_to_quantiles

    model.eval()
    likelihood.eval()
    with torch.no_grad():
        model_pred = model(x_pred).mean.detach()
    central = model_pred[..., 0:1]
    lower_gaps = model_pred[..., 1:1 + num_lower_quantiles]
    upper_gaps = model_pred[..., 1 + num_lower_quantiles:]
    quantiles = centergap_to_quantiles(central, lower_gaps, upper_gaps)

    plt.scatter(x, y, c='k', marker='.')
    plt.plot(x_pred, quantiles)
"""

import gpytorch
import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "centergap_to_quantiles",
    "CenterGapModel",
    "ALD",
    "CenterGapLikelihood",
    "QrLmcVariationalStrategy",
]


def centergap_to_quantiles(central, lower_gaps, upper_gaps):
    """Convert center-gap representation to quantiles.

    Parameters
    ----------
    central : torch.Tensor with shape (..., 1)
        The central quantile values.
    lower_gaps : torch.Tensor with shape (..., L)
        The lower gap values.
    upper_gaps : torch.Tensor with shape (..., U)
        The upper gap values.

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


class CenterGapModel(gpytorch.models.ApproximateGP):
    """Gaussian process modeling the center-gap representation of quantiles."""

    @property
    def center_mean(self):
        """Prior mean for the central quantile."""
        return self._center_mean

    @center_mean.setter
    def center_mean(self, model):
        self._center_mean = model

    @property
    def gap_mean(self):
        """Batched prior mean for the gaps."""
        return self._gap_mean

    @gap_mean.setter
    def gap_mean(self, model):
        self._gap_mean = model

    @property
    def covar_module(self):
        """Batched covariance module for the center and gaps."""
        return self._covar_module

    @covar_module.setter
    def covar_module(self, model):
        self._covar_module = model

    def forward(self, x):
        center_mean = self.center_mean(x)
        gap_mean = self.gap_mean(x)
        mean = torch.concatenate([center_mean.unsqueeze(0), gap_mean], dim=0)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class ALD(torch.distributions.Distribution):
    """Batched asymmetric Laplace distribution.

    Parameters
    ----------
    locs : torch.Tensor with shape (..., N, T)
        The location parameters of the distribution.
    scales : torch.Tensor with shape (T,)
        The scale parameters of the distribution for each quantile.
    taus : torch.Tensor with shape (T,)
        The quantile levels of the distribution.
    """

    arg_constraints = {
        "locs": torch.distributions.constraints.real,
        "scales": torch.distributions.constraints.positive,
        "taus": torch.distributions.constraints.unit_interval,
    }
    support = torch.distributions.constraints.real
    has_rsample = False

    def __init__(self, locs, scales, taus):
        # Reshape scales and taus as (1, 1, ..., 1, T)
        self.locs = locs
        self.scales = scales.view(*([1] * (locs.ndim - 1)), -1)
        self.taus = taus.view(*([1] * (locs.ndim - 1)), -1)
        super().__init__(locs.size())

    def log_prob(self, value):
        """Log probability of the asymmetric Laplace distribution at the given value.

        Parameters
        ----------
        value : torch.Tensor with shape (..., N)
            The values at which to evaluate the log probability.

        Returns
        -------
        logp : torch.Tensor with shape (..., N, T)
            The log probability at the given values for each quantile.
        """
        # value: (N,), locs: (..., N, T), scales & taus: (1, ..., 1, T)
        diff = value.unsqueeze(-1) - self.locs  # (..., N, T)
        rho = diff * (self.taus - (diff < 0).float())  # (..., N, T)
        logp = (
            torch.log(self.taus * (1 - self.taus) / self.scales) - rho / self.scales
        )  # (..., N, T)
        return logp


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
        central_quantile = self.taus[np.argmin(torch.abs(self.taus - 0.5))]
        self.lower_count = (self.taus < central_quantile).count_nonzero()

    @property
    def scales(self):
        # (T,)
        return self.raw_scales_constraint.transform(self.raw_scales)

    def forward(self, function_samples):
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


class QrLmcVariationalStrategy(gpytorch.variational.LMCVariationalStrategy):
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
