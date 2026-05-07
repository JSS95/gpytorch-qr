"""
Multitask GPQR (Center-gap representation)
------------------------------------------

1D regression dataset with heteroskedastic noise:

.. plot::
   :context:

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
    import matplotlib.pyplot as plt
    plt.scatter(x, y, c='gray', marker='.', alpha=0.1)
    plt.plot(x_range, true_quantiles, '--', c='k')
"""

import torch

__all__ = [
    "QuantileGP",
    "ALD",
    "ALDLikelihood",
]


class ALD(torch.distributions.Distribution):
    """Asymmetric Laplace distribution for multitask quantile regression.

    Parameters
    ----------
    m : torch.Tensor with shape (S, N, T)
        The location parameters of the distribution.
    lamda : torch.Tensor with shape (T,)
        The scale parameters of the distribution for each task.
    kappa : torch.Tensor with shape (T,)
        The quantile levels of the distribution.

    Notes
    -----
    In the context of multitask quantile regression, the location parameter *m*
    corresponds to sample points drawn from posterior distributions of latent GPs.
    For *T* tasks, *S* samples are drawn for *N* data points.

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
        # Reshape lamda and kappa as (1, 1, T)
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
        logp : torch.Tensor with shape (S, N, T)
            The log probability at the given values for each task.
        """
        # value: (N,), m: (S, N, T), lamda & kappa: (1, 1, T)
        diff = value.unsqueeze(0) - self.m  # (S, N, T)
        rho = diff * (self.kappa - (diff < 0).float())  # (S, N, T)
        logp = (
            torch.log(self.kappa * (1 - self.kappa) / self.lamda) - rho / self.lamda
        )  # (S, N, T)
        return logp
