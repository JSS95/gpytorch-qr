"""Variational strategies for GPQR."""

import gpytorch
import torch

__all__ = [
    "CGLmcVariationalStrategy",
]


class CGLmcVariationalStrategy(gpytorch.variational.LMCVariationalStrategy):
    """LMC variational strategy for the center-gap quantile regression model.

    This class allows all gaps to be correlated.

    Parameters
    ----------
    base_variational_strategy : gpytorch.variational.VariationalStrategy
        The base variational strategy to wrap.
    num_quantiles : list of int
        The number of quantiles in each output dimension.
    num_lower_quantiles : list of int
        The number of lower quantiles in each output dimension
        for center-gap representation.
    num_latents : int
        The number of latent GP functions.
    latent_dim : int, optional
        The dimension along which the latent functions are defined. Default is -1.
    jitter_val : float, optional
        The jitter value to add to the covariance matrix for numerical stability.

    Notes
    -----
    This class modifies the standard LMC coefficients to fit the center-gap
    representation.
    The first latent functions directly represent the central quantiles of each
    output dimension, and it does not form any linear combinations with the other
    latent functions.
    The remaining latent functions are linearly combined to model the gap
    functions between quantiles.

    Subclass can extend :meth:`construct_lmc_mask` to further restrict the
    linear combinations.
    """

    def __init__(
        self,
        base_variational_strategy,
        num_quantiles,
        num_lower_quantiles,
        num_latents,
        latent_dim=-1,
        jitter_val=None,
    ):
        if num_latents < len(num_quantiles):
            raise ValueError(
                "num_latents must be at least the number of output dimensions."
            )
        if num_latents == len(num_quantiles) and any(Q > 1 for Q in num_quantiles):
            raise ValueError(
                "If any output dimension has more than one quantile, "
                "num_latents must be greater than the number of output dimensions."
            )
        super().__init__(
            base_variational_strategy,
            sum(num_quantiles),  # Q
            num_latents,  # T
            latent_dim,
            jitter_val,
        )
        self.list_num_quantiles = num_quantiles
        self.list_num_lower_quantiles = num_lower_quantiles

        num_outputs = len(num_quantiles)
        # lmc_coefficients: ([batch_shape], T, Q)
        lmc_coefficients = self.lmc_coefficients.detach().clone()
        del self.lmc_coefficients

        g0_mask = torch.zeros_like(lmc_coefficients)
        for i in range(num_outputs):
            g0_mask[..., i, i] = 1
        self.register_buffer("g0_mask", g0_mask)

        lmc_mask = torch.zeros_like(lmc_coefficients)
        lmc_mask[..., num_outputs:, num_outputs:] = self.construct_lmc_mask(
            torch.Size(
                list(lmc_coefficients.shape[:-2])
                + [lmc_coefficients.shape[-2] - num_outputs]
                + [lmc_coefficients.shape[-1] - num_outputs]
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
            Must be ``([batch_shape], T - k, Q - k)``, where ``T`` is the
            number of latent functions, ``Q`` is the number of quantiles,
            and ``k`` is the number of output dimensions.

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
