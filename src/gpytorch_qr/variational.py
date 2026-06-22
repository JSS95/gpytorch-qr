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
    base_variational_strategy
    num_tasks
    num_latents
    latent_dim
    jitter_val
    num_quantiles : list of int, optional
        The number of quantiles in each output dimension.
        Its sum must equal *num_tasks*.
        If not passed, defaults to ``[num_tasks]``, i.e.,
        output is assumed to be 1-dimensional.
    num_lower_quantiles : list of int, optional
        The number of lower quantiles in each output dimension
        for center-gap representation.
        If not passed, defaults to a balanced split of the quantiles.

    Notes
    -----
    This class modifies the standard LMC coefficients to fit the center-gap
    representation with ``k`` outputs.
    The first ``k`` latent functions directly represent the central quantiles of each
    output dimension, and they do not form any linear combinations with the other
    latent functions.
    The remaining latent functions are linearly combined to model the gap
    functions between quantiles.

    The input ``T`` latent GPs are structured as

    .. code-block:: text

        [c_1, c_2, ..., c_k,  g_1, g_2, ..., g_{T-k}]

    where:

    - ``c_i`` is the central quantile for *i*-th output dimension,
    - ``g_j`` is the *j*-th latent function for modeling the gaps between quantiles.

    The output multitask GPs are structured as

    .. code-block:: text

        [c_1, c_2, ..., c_k,  *L_1, *U_1,  *L_2, *U_2,  ...,  *L_k, *U_k]

    where:

    - ``c_i`` is the central quantile for *i*-th output dimension,
    - ``L_i`` contains pre-softplus-transformed lower gaps for *i*-th output dimension,
    - ``U_i`` contains pre-softplus-transformed upper gaps for *i*-th output dimension.

    Subclass can extend :meth:`construct_lmc_mask` to further restrict the
    linear combinations.
    """

    def __init__(
        self,
        base_variational_strategy,
        num_tasks,
        num_latents,
        latent_dim=-1,
        jitter_val=None,
        num_quantiles=None,
        num_lower_quantiles=None,
    ):
        if num_quantiles is None:
            num_quantiles = [num_tasks]
        if num_lower_quantiles is None:
            nlq = []
            for Q in num_quantiles:
                nlq.append((Q - 1) // 2)
            num_lower_quantiles = nlq
        if not sum(num_quantiles) == num_tasks:
            raise ValueError("The sum of num_quantiles must equal num_tasks.")

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
            num_tasks,  # Q
            num_latents,  # T
            latent_dim,
            jitter_val,
        )
        self.num_quantiles = num_quantiles
        self.num_lower_quantiles = num_lower_quantiles

        # lmc_coefficients: (*B, T, Q)
        lmc_coefficients = self.lmc_coefficients.detach().clone()
        del self.lmc_coefficients
        self.register_parameter(
            "_lmc_coefficients", torch.nn.Parameter(lmc_coefficients)
        )

        T, Q = lmc_coefficients.shape[-2:]
        k = len(num_quantiles)
        self.register_buffer("lmc_mask", self.construct_lmc_mask(T, Q, k))

    def construct_lmc_mask(self, T, Q, k):
        """Construct a mask to restrict the LMC structure.

        Parameters
        ----------
        T : int
            The number of latent functions.
        Q : int
            The number of quantiles.
        k : int
            The number of output dimensions.

        Returns
        -------
        lmc_mask : torch.Tensor with shape ``(T, Q)``
            A binary mask of the same shape as the LMC coefficients, where 1
            indicates the positions of the LMC coefficients to be learned, and 0
            indicates the positions of the LMC coefficients to be fixed at 0.
        """
        mask = torch.zeros(T, Q)
        for i in range(k):
            mask[i, i] = 1  # Central quantiles
        mask[k:, k:] = 1  # Gap functions
        return mask

    @property
    def lmc_coefficients(self):
        return self._lmc_coefficients * self.lmc_mask
