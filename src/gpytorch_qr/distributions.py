"""Asymmetric Laplace distributions for quantile regression."""

import torch

__all__ = [
    "ALD",
    "BatchQuantileALD",
    "MultitaskQuantileALD",
]


class ALD(torch.distributions.Distribution):
    """Asymmetric Laplace distribution.

    Parameters
    ----------
    m : torch.Tensor
        The location parameter of the distribution.
    lamda : torch.Tensor
        The scale parameter of the distribution.
    kappa : torch.Tensor
        The quantile level of the distribution.

    Attributes
    ----------
    m : torch.Tensor
    lamda : torch.Tensor
    kappa : torch.Tensor
    """

    arg_constraints = {
        "m": torch.distributions.constraints.real,
        "lamda": torch.distributions.constraints.positive,
        "kappa": torch.distributions.constraints.unit_interval,
    }
    support = torch.distributions.constraints.real
    has_rsample = False

    def __init__(self, m, lamda, kappa):
        self.m = m
        self.lamda = lamda
        self.kappa = kappa
        super().__init__(m.shape)

    def log_prob(self, value):
        """Log probability of the asymmetric Laplace distribution"""
        residual = value - self.m
        check = residual * (self.kappa - (residual < 0).to(residual))
        logp = (
            torch.log(self.kappa)
            + torch.log(1 - self.kappa)
            - torch.log(self.lamda)
            - check / self.lamda
        )
        return logp

    def icdf(self, value):
        """Inverse CDF of the asymmetric Laplace distribution."""
        return torch.where(
            value <= self.kappa,
            self.m + self.lamda / (1 - self.kappa) * torch.log(value / self.kappa),
            self.m
            - self.lamda / self.kappa * torch.log((1 - value) / (1 - self.kappa)),
        )


class BatchQuantileALD(ALD):
    """Asymmetric Laplace distribution where quantiles are treated as batches.

    Parameters
    ----------
    m : torch.Tensor with shape ``(S, Q, *B, N)``
        The location parameters of the distribution.
    lamda : torch.Tensor with shape ``(Q, *B, 1)``
        The scale parameters of the distribution for each quantile.
    kappa : torch.Tensor with shape ``(Q, *B, 1)``
        The quantile levels of the distribution.

    Attributes
    ----------
    m : torch.Tensor with shape ``(S, Q, *B, N)``
    lamda : torch.Tensor with shape ``(1, Q, *B, 1)``
    kappa : torch.Tensor with shape ``(1, Q, *B, 1)``

    Notes
    -----
    - ``S`` : the number of samples drawn from the posterior distribution.
    - ``Q`` : the number of quantiles.
    - ``B`` : additional batches.
    - ``N`` : the number of data points.

    The posterior distribution have batch shape ``(Q, *B)``.
    """

    def __init__(self, m, lamda, kappa):
        super().__init__(m, lamda.unsqueeze(0), kappa.unsqueeze(0))

    def log_prob(self, value):
        """Log probability of the asymmetric Laplace distribution at the given value.

        Parameters
        ----------
        value : torch.Tensor with shape ``(*B, N)``
            Observed response variables at which to evaluate the log probability.

        Returns
        -------
        logp : torch.Tensor with shape ``(S, Q, *B, N)``
            The log probability at the given values for each quantile and sample.
        """
        return super().log_prob(value.reshape(1, 1, *value.shape))


class MultitaskQuantileALD(ALD):
    """Asymmetric Laplace distribution where quantiles are treated as tasks.

    Parameters
    ----------
    m : torch.Tensor with shape ``(S, *B, N, Q)``
        The location parameters of the distribution.
    lamda : torch.Tensor with shape ``(*B, 1, Q)``
        The scale parameters of the distribution for each quantile.
    kappa : torch.Tensor with shape ``(*B, 1, Q)``
        The quantile levels of the distribution.

    Attributes
    ----------
    m : torch.Tensor with shape ``(S, *B, N, Q)``
    lamda : torch.Tensor with shape ``(1, *B, 1, Q)``
    kappa : torch.Tensor with shape ``(1, *B, 1, Q)``

    Notes
    -----
    - ``S`` : the number of samples drawn from the posterior distribution.
    - ``Q`` : the number of quantiles.
    - ``B`` : additional batches.
    - ``N`` : the number of data points.

    The posterior distribution have batch shape ``(*B)``.
    """

    def __init__(self, m, lamda, kappa):
        super().__init__(m, lamda.unsqueeze(0), kappa.unsqueeze(0))

    def log_prob(self, value):
        """Log probability of the asymmetric Laplace distribution at the given value.

        Parameters
        ----------
        value : torch.Tensor with shape ``(*B, N)``
            Observed response variables at which to evaluate the log probability.

        Returns
        -------
        logp : torch.Tensor with shape ``(S, *B, N, Q)``
            The log probability at the given values for each quantile and sample.
        """
        return super().log_prob(value.reshape(1, *value.shape, 1))
