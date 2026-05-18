"""Variational strategies for GPQR."""

import gpytorch
import torch

__all__ = [
    "CGBlkdiagLmcVariationalStrategy",
]


class CGBlkdiagLmcVariationalStrategy(gpytorch.variational.LMCVariationalStrategy):
    """LMC variational strategy for the center-gap quantile regression model.

    This class modifies the standard LMC coefficients to fit the center-gap
    representation.
    The first latent function directly represents the central quantile, and it
    does not form any linear combinations with the other latent functions.
    The remaining latent functions are linearly combined to model the gap
    functions between quantiles. Upper and lower gap functions are modeled
    separately by block diagonal matrices.
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
        # lmc_coefficients: ([batch_shape], T, Q)
        lmc_coefficients = self.lmc_coefficients.detach().clone()
        del self.lmc_coefficients

        num_upper_quantiles = num_quantiles - num_lower_quantiles - 1
        num_upper_latents = num_latents - num_lower_latents - 1

        mask = torch.zeros_like(lmc_coefficients)
        mask[..., 1 : 1 + num_lower_latents, 1 : 1 + num_lower_quantiles] = 1
        mask[..., -num_upper_latents:, -num_upper_quantiles:] = 1
        self.register_buffer("lmc_mask", mask)

        g0_mask = torch.zeros_like(lmc_coefficients)
        g0_mask[..., 0, 0] = 1
        self.register_buffer("g0_mask", g0_mask)

        self.register_parameter(
            "_lmc_coefficients", torch.nn.Parameter(lmc_coefficients)
        )

    @property
    def lmc_coefficients(self):
        return self._lmc_coefficients * self.lmc_mask + self.g0_mask
