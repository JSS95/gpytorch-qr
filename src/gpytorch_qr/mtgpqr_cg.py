"""
------------------------------------------
Multitask GPQR (Center-gap representation)
------------------------------------------

.. plot::
   :context: reset
   :include-source: False

    import torch
    from torch.distributions import Normal

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
        CenterGapQuantileGP,
        CenterGapLmcVariationalStrategy,
        CenterGapALDLikelihood,
    )

    class MyGP(CenterGapQuantileGP):
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
    num_latents = 7
    gp = MyGP(inducing_points, len(q), central_q_index, num_latents,  num_latents // 2)
    likelihood = CenterGapALDLikelihood(q, central_q_index)

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
        quantiles = gp.mean_quantiles(x_pred, central_q_index)

    import matplotlib.pyplot as plt
    plt.scatter(x, y, c='gray', marker='.', alpha=0.1)
    plt.plot(x_range, true_quantiles, '--', c='k')
    plt.plot(x_pred, quantiles)
"""

import gpytorch
import torch
import torch.nn.functional as F

__all__ = [
    "centergap_to_quantiles",
    "CenterGapQuantileGP",
    "ALD",
    "CenterGapALDLikelihood",
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
    quantiles : torch.Tensor with shape (..., Q)
        The quantile values. (Q = L + U + 1)
    """
    lower_gaps = F.softplus(lower_gaps)
    lower_quantiles = central - lower_gaps.flip(dims=[-1]).cumsum(dim=-1).flip(
        dims=[-1]
    )

    upper_gaps = F.softplus(upper_gaps)
    upper_quantiles = central + upper_gaps.cumsum(dim=-1)

    ret = torch.concat([lower_quantiles, central, upper_quantiles], dim=-1)
    return ret


class CenterGapQuantileGP(gpytorch.models.ApproximateGP):
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
        function_means = self(x).mean
        median = function_means[..., :1]
        lower_gaps = function_means[..., 1 : 1 + num_lower_quantiles]
        upper_gaps = function_means[..., 1 + num_lower_quantiles :]
        return centergap_to_quantiles(median, lower_gaps, upper_gaps)


class ALD(torch.distributions.Distribution):
    """Asymmetric Laplace distribution for multitask quantile regression.

    Parameters
    ----------
    m : torch.Tensor with shape (S, N, Q)
        The location parameters of the distribution.
    lamda : torch.Tensor with shape (Q,)
        The scale parameters of the distribution for each quantile.
    kappa : torch.Tensor with shape (Q,)
        The quantile levels of the distribution.

    Notes
    -----
    In the context of multitask quantile regression, the location parameter *m*
    corresponds to sample points drawn from posterior distributions of latent GPs.
    For *Q* quantiles, *S* samples are drawn for *N* data points.

    The value passed to :meth:`log_prob` is the observed *y* values.
    """

    arg_constraints = {
        "m": torch.distributions.constraints.real,
        "lamda": torch.distributions.constraints.positive,
        "kappa": torch.distributions.constraints.unit_interval,
    }
    support = torch.distributions.constraints.real
    has_rsample = False

    def __init__(self, m, lamda, kappa):
        # Reshape lamda and kappa as (1, 1, Q)
        self.m = m
        self.lamda = lamda.view(1, 1, -1)
        self.kappa = kappa.view(1, 1, -1)
        super().__init__(m.size())

    def log_prob(self, value):
        """Log probability of the asymmetric Laplace distribution at the given value.

        Parameters
        ----------
        value : torch.Tensor with shape (N,)
            The values at which to evaluate the log probability.

        Returns
        -------
        logp : torch.Tensor with shape (S, N, Q)
            The log probability at the given values for each quantile.
        """
        # value: (N,), m: (S, N, Q), lamda & kappa: (1, 1, Q)
        diff = value.view(1, -1, 1) - self.m  # (S, N, Q)
        rho = diff * (self.kappa - (diff < 0).float())  # (S, N, Q)
        logp = (
            torch.log(self.kappa * (1 - self.kappa) / self.lamda) - rho / self.lamda
        )  # (S, N, Q)
        return logp


class CenterGapALDLikelihood(gpytorch.likelihoods.Likelihood):
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
        return ALD(
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
