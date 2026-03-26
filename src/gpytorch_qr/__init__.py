import gpytorch
import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "ALD",
    "CenterGapLikelihood",
    "QrLmcVariationalStrategy",
]


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
        taus = np.asarray(taus)
        central_quantile = taus[np.argmin(np.abs(taus - 0.5))]
        self.lower_count = (taus < central_quantile).count_nonzero()

    @staticmethod
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

    @property
    def scales(self):
        # (T,)
        return self.raw_scales_constraint.transform(self.raw_scales)

    def forward(self, function_samples):
        # function_samples: (S, N, T) <- from MedianGapGP
        median = function_samples[:, :, :1]
        lower_gaps = function_samples[:, :, 1 : 1 + self.lower_count]
        upper_gaps = function_samples[:, :, 1 + self.lower_count :]
        quantiles = self.centergap_to_quantiles(median, lower_gaps, upper_gaps)
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

        num_upper_quantiles = num_latents - num_lower_quantiles
        num_upper_latents = num_latents - num_lower_latents
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
