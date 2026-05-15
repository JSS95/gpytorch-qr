"""Gaussian process classes for quantile regression."""

import abc

import gpytorch
import torch

from .centergap import transform_centergap_posterior

__all__ = [
    "BayesianQRMixin",
    "GPQR",
    "DirectGPQR",
    "CenterGapGPQR",
]


class BayesianQRMixin(abc.ABC):
    """Mixin class for Bayesian quantile regression.

    Notes
    -----
    Input tensor ``x`` should have ``(*B, N, D)`` shape, where ``*B`` are
    optional batch shapes (e.g., for cross validation),
    *N* is the number of data points and *D* is the number of input dimensions.
    """

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

    @abc.abstractmethod
    def mean_quantiles_mc(self, x, num_samples):
        """Predict quantiles by MC mean of the quantile posterior.

        Parameters
        ----------
        x : torch.Tensor with shape ``(*B, N, D)``
            The input locations.
        num_samples : int
            Number of MC samples used to estimate the mean.

        Returns
        -------
        torch.Tensor
            The predicted quantiles at the input locations.
        """
        pass

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

    @abc.abstractmethod
    def quantile_quantiles_mc(self, x, q, num_samples):
        """Monte Carlo quantile of quantile posterior.

        Parameters
        ----------
        x : torch.Tensor with shape ``(*B, N, D)``
            The input locations.
        q : torch.Tensor with shape (q,)
            The quantile levels.
        num_samples : int
            Number of MC samples used to estimate the quantiles.

        Returns
        -------
        quantiles : torch.Tensor
            The predicted quantiles at the input locations.
        """
        pass


class GPQR(gpytorch.models.ApproximateGP, BayesianQRMixin):
    """Base class for Gaussian process quantile regression.

    Parameters
    ----------
    variational_strategy : gpytorch.variational.VariationalStrategy
    mean_module : gpytorch.means.Mean
    covar_module : gpytorch.kernels.Kernel
    """

    def __init__(self, variational_strategy, mean_module, covar_module):
        super().__init__(variational_strategy)
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def mean_quantiles_mc(self, x, num_samples=10):
        dist = self.joint_quantile_posterior(x)
        samples = dist.rsample(torch.Size([num_samples]))
        return samples.mean(dim=0)

    def quantile_quantiles_mc(self, x, q, num_samples=10):
        dist = self.joint_quantile_posterior(x)
        samples = dist.rsample(torch.Size([num_samples]))
        return samples.quantile(q, dim=0)


class DirectGPQR(GPQR):
    """Gaussian process quantile regression with direct quantile representation."""

    def joint_quantile_posterior(self, x):
        return self(x)

    def marginal_quantile_posterior(self, x):
        dist = self(x)
        return torch.distributions.Normal(dist.mean, dist.variance.sqrt())

    def mean_quantiles(self, x):
        return self(x).mean

    def quantile_quantiles(self, x, q):
        dist = self.marginal_quantile_posterior(x)
        shape = [-1] + [1 for _ in range(len(dist.batch_shape))]
        return dist.icdf(q.reshape(*shape))


class CenterGapGPQR(GPQR):
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
