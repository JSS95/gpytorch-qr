"""Variational strategies for GPQR."""

import gpytorch
import torch

__all__ = [
    "CenterGapLMCVariationalStrategy",
]


class CenterGapLMCVariationalStrategy(gpytorch.variational.LMCVariationalStrategy):
    r"""A special LMC variational strategy for the center-gap representation.

    This class enforces independence between the central quantiles and the
    gap functions.

    Parameters
    ----------
    base_variational_strategy
    num_tasks
    num_latents
    latent_dim
    jitter_val
    num_central_latents : int, optional
        The number of latent functions that are linearly combined to model the
        central quantiles.
        If not passed, defaults to ``len(num_quantiles)``.
    num_quantiles : list of int, optional
        The number of quantiles in each output dimension.
        Its sum must equal *num_tasks*.
        If not passed, defaults to ``[num_tasks]``.

    Notes
    -----
    .. rubric:: Input GP

    The input ``T`` latent GPs are split to

    .. code-block:: text

        [c_1, c_2, ..., c_k,  g_1, g_2, ..., g_{T-k}]

    where:

    - ``c_i`` are linearly combined to form central quantiles for each output dimension.
    - ``g_j`` are linearly combined to form the gap functions between quantiles.
    - ``c_i`` and ``g_j`` are not combined with each other, i.e., central quantiles and
      gap functions are independent.

    The number of ``c_i``, i.e., ``k``, is *num_central_latents*.

    .. rubric:: Output GP

    The output multitask GPs are structured as

    .. code-block:: text

        [C_1, C_2, ..., C_q,  *L_1, *U_1,  *L_2, *U_2,  ...,  *L_q, *U_q]

    where:

    - ``C_i`` is the central quantile for *i*-th output dimension,
    - ``L_i`` contains pre-softplus-transformed lower gaps for *i*-th output dimension,
    - ``U_i`` contains pre-softplus-transformed upper gaps for *i*-th output dimension.

    The number of output dimensions, i.e., ``q``, is ``len(num_quantiles)``.
    The sum ``1 + len(L_i) + len(U_i)`` equals ``num_quantiles[i]`` for
    each output dimension *i*.
    """

    def __init__(
        self,
        base_variational_strategy,
        num_tasks,
        num_latents,
        latent_dim=-1,
        jitter_val=None,
        num_central_latents=None,
        num_quantiles=None,
    ):
        if num_quantiles is None:
            num_quantiles = [num_tasks]
        if num_central_latents is None:
            num_central_latents = len(num_quantiles)
        if not sum(num_quantiles) == num_tasks:
            raise ValueError("The sum of num_quantiles must equal num_tasks.")

        super().__init__(
            base_variational_strategy,
            num_tasks,  # Q
            num_latents,  # T
            latent_dim,
            jitter_val,
        )
        self.num_central_latents = num_central_latents
        self.num_quantiles = num_quantiles

        # lmc_coefficients: (*B, T, Q)
        lmc_coefficients = self.lmc_coefficients.detach().clone()
        del self.lmc_coefficients
        T, Q = lmc_coefficients.shape[-2:]
        self.register_buffer("lmc_mask", self.construct_lmc_mask(T, Q))

        self.register_parameter("_lmc_coeff", torch.nn.Parameter(lmc_coefficients))

    def construct_lmc_mask(self, T, Q):
        """Construct a mask to restrict the learnable LMC coefficients.

        Parameters
        ----------
        T : int
            The number of latent functions.
        Q : int
            The number of quantiles.

        Returns
        -------
        lmc_mask : torch.Tensor with shape ``(T, Q)``
            A binary mask of the same shape as the LMC coefficients.
            1 indicates learnable coefficients, and 0 indicates fixed coefficients.
        """
        mask = torch.zeros(T, Q)
        mask[: self.num_central_latents, : len(self.num_quantiles)] = 1
        mask[self.num_central_latents :, len(self.num_quantiles) :] = 1
        return mask

    @property
    def lmc_coefficients(self):
        return self._lmc_coeff * self.lmc_mask
