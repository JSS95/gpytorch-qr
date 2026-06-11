"""Gaussian process classes for quantile regression."""

import abc

import gpytorch
import torch

from .utils import centergap_to_quantiles, transform_centergap_posterior

__all__ = [
    "QuantileGP",
    "DirectQuantileGP",
    "CenterGapQuantileGP",
]


class QuantileGP(gpytorch.models.ApproximateGP, abc.ABC):
    """Base class for Gaussian process quantile regression.

    Parameters
    ----------
    variational_strategy : gpytorch.variational.VariationalStrategy
    mean_module : gpytorch.means.Mean
    covar_module : gpytorch.kernels.Kernel

    Notes
    -----
    Input predictors are expected to have shape ``(*B, N, D)``, where ``*B`` are
    optional batch shapes (e.g., for cross validation), *N* is the number of data points
    and *D* is the number of input dimensions.

    Quantiles are task dimension with shape *Q*, constructed by combination of
    *L* latent GPs.

    - ``variational_strategy`` must wrap a variational distribution with batch shape
      ``(*B, L)``.
    - ``mean_module`` and ``covar_module`` must have batch shape ``(*B, L)``.
    - Posterior is :class:`gpytorch.distributions.MultitaskMultivariateNormal`
      with batch shape ``(*B)`` and event shape ``(N, Q)``.
    - MLL loss is a tensor of shape ``(*B)``.
    """

    def __init__(self, variational_strategy, mean_module, covar_module):
        super().__init__(variational_strategy)
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x):
        # x : (*B, N, D) -> (*B, 1, N, D)
        x = x.unsqueeze(-3)
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    @abc.abstractmethod
    def joint_quantile_posterior(self, x):
        """Joint posterior over quantiles at input locations.

        Parameters
        ----------
        x : torch.Tensor with shape ``(*B, N, D)``
            The input locations.

        Returns
        -------
        torch.distributions.Distribution
        """
        pass

    def marginal_quantile_posterior(self, x):
        """Marginal posterior over quantiles at input locations.

        Parameters
        ----------
        x : torch.Tensor with shape ``(*B, N, D)``

            The input locations.

        Returns
        -------
        torch.distributions.Distribution
        """
        raise NotImplementedError

    def mean_quantiles(self, x):
        """Predict quantiles by analytical posterior mean.

        Parameters
        ----------
        x : torch.Tensor with shape ``(*B, N, D)``
            The input locations.

        Returns
        -------
        quantiles : torch.Tensor
            The predicted quantiles at the input locations.
        """
        raise NotImplementedError

    def mean_quantiles_mc(self, x, num_samples=10):
        """Posterior mean of quantiles by Monte Carlo approximation.

        Parameters
        ----------
        x : torch.Tensor with shape ``(*B, N, D)``
            The input locations.
        num_samples : int, default=10
            The number of Monte Carlo samples.

        Returns
        -------
        quantiles : torch.Tensor
            The predicted quantiles at the input locations.
        """
        dist = self.joint_quantile_posterior(x)
        samples = dist.rsample(torch.Size([num_samples]))
        return samples.mean(dim=0)

    def mean_quantiles_delta(self, x):
        """Posterior mean of quantiles by 0th-order delta method.

        Parameters
        ----------
        x : torch.Tensor with shape ``(*B, N, D)``
            The input locations.

        Returns
        -------
        quantiles : torch.Tensor
            The predicted quantiles at the input locations.
        """
        raise NotImplementedError

    def quantile_quantiles(self, x, q):
        """Analytic quantile of quantile posterior.

        Parameters
        ----------
        x : torch.Tensor with shape ``(*B, N, D)``
            The input locations.
        q : torch.Tensor with shape (q,)
            The quantile levels.

        Returns
        -------
        quantiles : torch.Tensor
            The predicted quantiles at the input locations.
        """
        raise NotImplementedError

    def quantile_quantiles_mc(self, x, q, num_samples=10):
        """Quantile of quantile posterior by Monte Carlo approximation.

        Parameters
        ----------
        x : torch.Tensor with shape ``(*B, N, D)``
            The input locations.
        q : torch.Tensor with shape (q,)
            The quantile levels.
        num_samples : int, default=10
            The number of Monte Carlo samples.

        Returns
        -------
        quantiles : torch.Tensor
            The predicted quantiles at the input locations.
        """
        dist = self.joint_quantile_posterior(x)
        samples = dist.rsample(torch.Size([num_samples]))
        return samples.quantile(q, dim=0)


class DirectQuantileGP(QuantileGP):
    """Gaussian process quantile regression with direct quantile representation."""

    def joint_quantile_posterior(self, x):
        return self(x)

    def marginal_quantile_posterior(self, x):
        dist = self(x)
        return torch.distributions.Normal(dist.mean, dist.variance.sqrt())

    def mean_quantiles(self, x):
        return self(x).mean

    def mean_quantiles_delta(self, x):
        return self(x).mean

    def quantile_quantiles(self, x, q):
        dist = self.marginal_quantile_posterior(x)
        shape = [-1] + [1 for _ in range(len(dist.batch_shape))]
        return dist.icdf(q.reshape(*shape))


class CenterGapQuantileGP(QuantileGP):
    """Gaussian process quantile regression with center-gap quantile representation.

    Parameters
    ----------
    variational_strategy
    mean_module : gpytorch_qr.centergap.CenterGapMean
        Mean module for center-gap representation.
    covar_module
    num_lower_quantiles : int
        The number of lower quantiles in center-gap representation.
    """

    def __init__(
        self,
        variational_strategy,
        mean_module,
        covar_module,
        num_lower_quantiles,
    ):
        super().__init__(variational_strategy, mean_module, covar_module)
        self.num_lower_quantiles = num_lower_quantiles

    def joint_quantile_posterior(self, x):
        return transform_centergap_posterior(self(x), self.num_lower_quantiles)

    def mean_quantiles_delta(self, x):
        latent_posterior = self(x)
        if isinstance(
            latent_posterior, gpytorch.distributions.MultitaskMultivariateNormal
        ):
            quantile_dim = -1
        elif isinstance(latent_posterior, gpytorch.distributions.MultivariateNormal):
            quantile_dim = -2
        else:
            raise ValueError("Posterior is not a multivariate normal.")
        latent_mean = latent_posterior.mean
        num_upper = latent_mean.shape[quantile_dim] - 1 - self.num_lower_quantiles
        center_mean = torch.narrow(latent_mean, quantile_dim, 0, 1)
        lower_gaps = torch.narrow(
            latent_mean, quantile_dim, 1, self.num_lower_quantiles
        )
        upper_gaps = torch.narrow(
            latent_mean, quantile_dim, 1 + self.num_lower_quantiles, num_upper
        )
        return centergap_to_quantiles(
            center_mean, lower_gaps, upper_gaps, quantile_dim=quantile_dim
        )
