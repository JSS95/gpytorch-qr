import torch

__all__ = ["ALD"]


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

    Examples
    --------
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
