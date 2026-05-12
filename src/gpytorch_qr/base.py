import abc

import torch

__all__ = [
    "BayesianQRMixin",
    "ALDLikelihoodMixin",
]


class BayesianQRMixin(abc.ABC):
    """Mixin class for Bayesian quantile regression."""

    @abc.abstractmethod
    def joint_quantile_posterior(self, x):
        """Joint posterior over quantiles at input locations.

        Parameters
        ----------
        x : torch.Tensor with shape (N, D)
            The input locations.

        Returns
        -------
        gpytorch.distributions.MultivariateNormal
        """
        pass

    def marginal_quantile_posterior(self, x):
        """Marginal posterior over quantiles at input locations.

        Parameters
        ----------
        x : torch.Tensor with shape (N, D)
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
        x : torch.Tensor with shape (N, D)
            The input locations.

        Returns
        -------
        quantiles : torch.Tensor with shape (Q, N) or (N, Q)
            The predicted quantiles at the input locations.
            *Q* is the number of quantiles and *N* is the number of data points.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def mean_quantiles_mc(self, x, num_samples=10):
        """Predict quantiles by MC mean of the quantile posterior.

        Parameters
        ----------
        x : torch.Tensor with shape (N, D)
            The input locations.
        num_samples : int, default=10
            Number of MC samples used to estimate the mean.

        Returns
        -------
        torch.Tensor with shape (Q, N) or (N, Q)
            The predicted quantiles at the input locations.
            *Q* is the number of quantiles and *N* is the number of data points.
        """
        pass

    def quantile_quantiles(self, x, q):
        """Analytic quantile of quantile posterior.

        Parameters
        ----------
        x : torch.Tensor with shape (N, D)
            The input locations.
        q : torch.Tensor with shape (q,)
            The quantile levels.

        Returns
        -------
        quantiles : torch.Tensor with shape (q, Q, N) or (q, N, Q)
            The predicted quantiles at the input locations.
            *Q* is the number of quantiles and *N* is the number of data points.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def quantile_quantiles_mc(self, x, q, num_samples=10):
        """Monte Carlo quantile of quantile posterior.

        Parameters
        ----------
        x : torch.Tensor with shape (N, D)
            The input locations.
        q : torch.Tensor with shape (q,)
            The quantile levels.
        num_samples : int, default=10
            Number of MC samples used to estimate the quantiles.

        Returns
        -------
        quantiles : torch.Tensor with shape (q, Q, N) or (q, N, Q)
            The predicted quantiles at the input locations.
            *Q* is the number of quantiles and *N* is the number of data points.
        """
        pass


class ALDLikelihoodMixin:
    """Mixin class for asymmetric Laplace distribution likelihood."""

    def predictive_posterior(self, gp_posterior):
        """Predictive posterior distribution of function values.

        Parameters
        ----------
        gp_posterior : gpytorch.distributions.MultivariateNormal
            The joint posterior over latent GPs at input locations.

        Returns
        -------
        samples : torch.Tensor with shape (S, Q, N) or (S, N, Q)
            Samples drawn from the predictive posterior distribution of function values.

        Examples
        --------
        Get 95% predictive intervals:

        .. code-block:: python

            with torch.no_grad(), gpytorch.settings.num_likelihood_samples(1000):
                pp_dist = gp(x_pred)
            ci_lower = pp_dist.quantile(0.025, dim=0)
            ci_upper = pp_dist.quantile(0.975, dim=0)
        """
        ald = self(gp_posterior)
        u = torch.rand_like(ald.m)
        return ald.icdf(u)
