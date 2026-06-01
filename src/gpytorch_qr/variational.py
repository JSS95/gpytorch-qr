"""Variational strategies for GPQR."""

import gpytorch
import torch

__all__ = [
    "CGLmcVariationalStrategy",
    "CGBlkdiagLmcVariationalStrategy",
]


class CGLmcVariationalStrategy(gpytorch.variational.LMCVariationalStrategy):
    """LMC variational strategy for the center-gap quantile regression model.

    This class modifies the standard LMC coefficients to fit the center-gap
    representation.
    The first latent function directly represents the central quantile, and it
    does not form any linear combinations with the other latent functions.
    The remaining latent functions are linearly combined to model the gap
    functions between quantiles.

    Subclass can extend :meth:`construct_lmc_mask` to further restrict the
    linear combinations, e.g., to model upper and lower gap functions
    separately by block diagonal matrices.
    """

    def __init__(
        self,
        base_variational_strategy,
        num_quantiles,  # Q
        num_latents,  # T
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

        g0_mask = torch.zeros_like(lmc_coefficients)
        g0_mask[..., 0, 0] = 1
        self.register_buffer("g0_mask", g0_mask)

        lmc_mask = torch.zeros_like(lmc_coefficients)
        lmc_mask[..., 1:, 1:] = self.construct_lmc_mask(
            torch.Size(
                list(lmc_coefficients.shape[:-2])
                + [lmc_coefficients.shape[-2] - 1]
                + [lmc_coefficients.shape[-1] - 1]
            )
        )
        self.register_buffer("lmc_mask", lmc_mask)

        self.register_parameter(
            "_lmc_coefficients", torch.nn.Parameter(lmc_coefficients)
        )

    def construct_lmc_mask(self, shape):
        """Construct a mask to restrict the LMC structure.

        Parameters
        ----------
        shape : torch.Size
            The shape of the LMC coefficients.
            Must be ``([batch_shape], T - 1, Q - 1)``, where ``T`` is the
            number of latent functions and ``Q`` is the number of quantiles.

        Returns
        -------
        lmc_mask : torch.Tensor with shape ``shape``
            A binary mask of the same shape as the LMC coefficients, where 1
            indicates the positions of the LMC coefficients to be learned, and 0
            indicates the positions of the LMC coefficients to be fixed at 0.
        """
        return torch.ones(shape)

    @property
    def lmc_coefficients(self):
        return self._lmc_coefficients * self.lmc_mask + self.g0_mask


class CGBlkdiagLmcVariationalStrategy(CGLmcVariationalStrategy):
    """LMC variational strategy for the center-gap quantile regression model.

    Upper and lower gap functions are modeled separately by block diagonal matrices.
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
        num_upper_quantiles = num_quantiles - num_lower_quantiles - 1
        num_upper_latents = num_latents - num_lower_latents - 1

        self.num_lower_quantiles = num_lower_quantiles
        self.num_lower_latents = num_lower_latents
        self.num_upper_quantiles = num_upper_quantiles
        self.num_upper_latents = num_upper_latents

        super().__init__(
            base_variational_strategy,
            num_quantiles,
            num_latents,
            latent_dim,
            jitter_val,
        )

    def construct_lmc_mask(self, shape):
        mask = super().construct_lmc_mask(shape)
        mask[..., : self.num_lower_latents, -self.num_upper_quantiles :] = 0
        mask[..., -self.num_upper_latents :, : self.num_lower_quantiles] = 0
        return mask

    @property
    def lmc_coefficients(self):
        return self._lmc_coefficients * self.lmc_mask + self.g0_mask
