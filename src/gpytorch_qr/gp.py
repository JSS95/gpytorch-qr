"""Gaussian process classes for quantile regression."""

import abc

__all__ = [
    "BayesianQRMixin",
]


class BayesianQRMixin(abc.ABC):
    """Mixin class for Bayesian quantile regression.

    Notes
    -----
    Input tensor ``x`` should have ``([B, ...], N, D)`` shape, where ``[B, ...]`` are
    optional batch shapes (e.g., for cross validation),
    *N* is the number of data points and *D* is the number of input dimensions.

    If the model treats quantile dimension as batch output, the output distribution
    must have *Q* quantiles in its first batch dimension.

    If the model treats quantile dimension as multitask output, the output distribution
    must have *Q* quantiles in its last event dimension.
    """

    @abc.abstractmethod
    def joint_quantile_posterior(self, x):
        """Joint posterior over quantiles at input locations.

        Parameters
        ----------
        x : torch.Tensor with shape ([B, ...], N, D)
            The input locations.

        Returns
        -------
        torch.distributions.Distribution
            For batch quantile output, the distribution has batch shape
            ``(Q, [B, ...])`` and event shape ``(N,)``.
            For multitask quantile output, the distribution has batch shape ``[B, ...]``
            and event shape ``(N, Q)``.
        """
        pass

    def marginal_quantile_posterior(self, x):
        """Marginal posterior over quantiles at input locations.

        Parameters
        ----------
        x : torch.Tensor with shape ([B, ...], N, D)
            The input locations.

        Returns
        -------
        torch.distributions.Distribution
            For batch quantile output, the distribution has batch shape
            ``(Q, [B, ...])`` and event shape ``(N,)``.
            For multitask quantile output, the distribution has batch shape ``[B, ...]``
            and event shape ``(N, Q)``.
        """
        raise NotImplementedError

    def mean_quantiles(self, x):
        """Predict quantiles by analytical posterior mean.

        Parameters
        ----------
        x : torch.Tensor with shape ([B, ...], N, D)
            The input locations.

        Returns
        -------
        quantiles : torch.Tensor
            The predicted quantiles at the input locations.
            For batch quantile output, the shape is ``(Q, [B, ...], N)``.
            For multitask quantile output, the shape is ``([B, ...], N, Q)``.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def mean_quantiles_mc(self, x, num_samples=10):
        """Predict quantiles by MC mean of the quantile posterior.

        Parameters
        ----------
        x : torch.Tensor with shape ([B, ...], N, D)
            The input locations.
        num_samples : int, default=10
            Number of MC samples used to estimate the mean.

        Returns
        -------
        torch.Tensor
            The predicted quantiles at the input locations.
            For batch quantile output, the shape is ``(Q, [B, ...], N)``.
            For multitask quantile output, the shape is ``([B, ...], N, Q)``.
        """
        pass

    def quantile_quantiles(self, x, q):
        """Analytic quantile of quantile posterior.

        Parameters
        ----------
        x : torch.Tensor with shape ([B, ...], N, D)
            The input locations.
        q : torch.Tensor with shape (q,)
            The quantile levels.

        Returns
        -------
        quantiles : torch.Tensor
            The predicted quantiles at the input locations.
            For batch quantile output, the shape is ``(q, Q, [B, ...], N)``.
            For multitask quantile output, the shape is ``(q, [B, ...], N, Q)``.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def quantile_quantiles_mc(self, x, q, num_samples=10):
        """Monte Carlo quantile of quantile posterior.

        Parameters
        ----------
        x : torch.Tensor with shape ([B, ...], N, D)
            The input locations.
        q : torch.Tensor with shape (q,)
            The quantile levels.
        num_samples : int, default=10
            Number of MC samples used to estimate the quantiles.

        Returns
        -------
        quantiles : torch.Tensor
            The predicted quantiles at the input locations.
            For batch quantile output, the shape is ``(q, Q, [B, ...], N)``.
            For multitask quantile output, the shape is ``(q, [B, ...], N, Q)``.
        """
        pass
