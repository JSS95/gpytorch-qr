import torch

__all__ = [
    "MultitaskALD",
]


class MultitaskALD(torch.distributions.Distribution):
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
