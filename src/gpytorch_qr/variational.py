"""Variational strategies for GPQR."""

import gpytorch
import torch

__all__ = [
    "CenterGapLMCVariationalStrategy",
]


class CenterGapLMCVariationalStrategy(gpytorch.variational.LMCVariationalStrategy):
    r"""Special LMC variational strategy for the center-gap representation.

    This class forces the following structure:

    1. Each central quantile of each output dimension is directly represented by
       a dedicated independent latent function.
    2. Gap functions for all output dimensions are represented by
       linear combinations of the remaining latent functions.
    3. Only the coefficients for the gap functions are learned.

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
    This class is introduced to facilitate implementing center-gap model
    where the gaps are correlated while the center has prior mean.

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

    .. hint::

        The limitation of this class is that

        1. It cannot correlate the central quantile and the gap functions.
        2. It cannot correlate the central quantiles of different output dimensions.

        Should such correlations be desired, one can modify the input observations by
        :math:`y \leftarrow y - \mu(x)` and use a standard LMC variational strategy to
        model the residuals.
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
        T, Q = lmc_coefficients.shape[-2:]
        k = len(num_quantiles)
        self.register_buffer("lmc_mask", self.construct_lmc_mask(T, Q, k))

        self.register_parameter("_lmc_coeff", torch.nn.Parameter(lmc_coefficients))
        coeff = torch.zeros(T, Q)
        for i in range(k):
            coeff[i, i] = 1.0
        self.register_buffer("_fixed_coeff", coeff)

    @classmethod
    def construct_lmc_mask(cls, T, Q, k):
        """Construct a mask to restrict the learnable LMC coefficients.

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
            A binary mask of the same shape as the LMC coefficients.
            1 indicates learnable coefficients, and 0 indicates fixed coefficients.
        """
        mask = torch.zeros(T, Q)
        mask[k:, k:] = 1  # Gap functions
        return mask

    @property
    def lmc_coefficients(self):
        return self._lmc_coeff * self.lmc_mask + self._fixed_coeff
