import abc

import torch

__all__ = [
    "BayesianQRMixin",
    "ALDLikelihoodMixin",
]


class BayesianQRMixin(abc.ABC):
    """Mixin class for Bayesian quantile regression.

    Notes
    -----
    Input tensor can have ``([input_batch_shape], N, D)`` shape,
    where ``input_batch_shape`` are optional leading batch shapes
    (e.g., for cross validation),
    *N* is the number of data points and *D* is the number of input dimensions.

    Internal variational distribution and modules (e.g., prior mean and covariance)
    should have batch shape ``(Q, [module_batch_shape])``, where *Q* is the number
    of quantiles and ``module_batch_shape`` are shapes broadcastable to
    ``input_batch_shape``.

    Number of batch dimension can be zero or more, but you must fix how many dimensions
    will be supported when you define the concrete class for the model.
    In other words, ``len(input_batch_shape)`` and ``len(module_batch_shape)``
    should be equal and fixed for a concrete model class.

    Batch shapes can be either fixed or variable depending on your purpose.
    For example, suppose you want to define a model that allows one batch dimension.
    You can let ``module_batch_shape`` be either ``(1,)`` or ``(B,)``.
    When the shape is ``(1,)``, the model can accept any input batch but the model
    parameters will be shared across batches.
    When the shape is ``(B,)``, the model can only accept input batch with size *B*,
    but the model parameters will be different across batches.

    It is usually recommended to use ``(B,)`` instead of ``(1,)`` to prevent
    data leakage across batches, e.g., for cross validation with ``B`` folds.

    The resulting posterior distribution of latent GPs by ``self(x)`` have
    batch shape ``(Q, [input_batch_shape])`` and event shape ``(N,)``.
    """

    @abc.abstractmethod
    def joint_quantile_posterior(self, x):
        """Joint posterior over quantiles at input locations.

        Parameters
        ----------
        x : torch.Tensor with shape ([batch_shape], N, D)
            The input locations.

        Returns
        -------
        torch.distributions.Distribution
            Distribution with batch_shape ``(Q, [batch_shape])``
            and event shape ``(N,)``.
        """
        pass

    def marginal_quantile_posterior(self, x):
        """Marginal posterior over quantiles at input locations.

        Parameters
        ----------
        x : torch.Tensor with shape ([batch_shape], N, D)
            The input locations.

        Returns
        -------
        torch.distributions.Distribution
            Distribution with batch shape ``(Q, [batch_shape], N)``
            and event shape ``()``.
        """
        raise NotImplementedError

    def mean_quantiles(self, x):
        """Predict quantiles by analytical posterior mean.

        Parameters
        ----------
        x : torch.Tensor with shape ([batch_shape], N, D)
            The input locations.

        Returns
        -------
        quantiles : torch.Tensor
            The predicted quantiles at the input locations.
            The shape is ``(Q, [batch_shape], N)`` or ``([batch_shape], N, Q)``,
            where *Q* is the number of quantiles and *N* is the number of data points.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def mean_quantiles_mc(self, x, num_samples=10):
        """Predict quantiles by MC mean of the quantile posterior.

        Parameters
        ----------
        x : torch.Tensor with shape ([batch_shape], N, D)
            The input locations.
        num_samples : int, default=10
            Number of MC samples used to estimate the mean.

        Returns
        -------
        torch.Tensor with shape (Q, N) or (N, Q)
            The predicted quantiles at the input locations.
            The shape is ``(Q, [batch_shape], N)`` or ``([batch_shape], N, Q)``,
            where *Q* is the number of quantiles and *N* is the number of data points.
        """
        pass

    def quantile_quantiles(self, x, q):
        """Analytic quantile of quantile posterior.

        Parameters
        ----------
        x : torch.Tensor with shape ([batch_shape], N, D)
            The input locations.
        q : torch.Tensor with shape (q,)
            The quantile levels.

        Returns
        -------
        quantiles : torch.Tensor
            The predicted quantiles at the input locations.
            The shape is ``(q, Q, [batch_shape], N)`` or ``(q, [batch_shape], N, Q)``,
            where *Q* is the number of quantiles and *N* is the number of data points.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def quantile_quantiles_mc(self, x, q, num_samples=10):
        """Monte Carlo quantile of quantile posterior.

        Parameters
        ----------
        x : torch.Tensor with shape ([batch_shape], N, D)
            The input locations.
        q : torch.Tensor with shape (q,)
            The quantile levels.
        num_samples : int, default=10
            Number of MC samples used to estimate the quantiles.

        Returns
        -------
        quantiles : torch.Tensor
            The predicted quantiles at the input locations.
            The shape is ``(q, Q, [batch_shape], N)`` or ``(q, [batch_shape], N, Q)``,
            where *Q* is the number of quantiles and *N* is the number of data points.
        """
        pass


class ALDLikelihoodMixin(abc.ABC):
    """Mixin class for asymmetric Laplace distribution likelihood."""

    @abc.abstractmethod
    def forward(self, function_samples):
        """Return the ALD distribution for the given function samples.

        Parameters
        ----------
        function_samples : torch.Tensor
            The function samples drawn from the posterior of latent GP.
            Shape is ``(S, Q, [batch_shape], N)`` for batch GPQR
            or ``(S, [batch_shape], N, Q)`` for multitask GPQR,
            where *S* is the number of samples, *Q* is the number of quantiles,
            and *N* is the number of data points.

        Returns
        -------
        torch.distribution.Distribution
            Batch ALD or multitask ALD.
        """

    @abc.abstractmethod
    def expected_log_prob(self, observations, function_dist, *args, **kwargs):
        """Expected log probability of observations under the ALD likelihood.

        Parameters
        ----------
        observations : torch.Tensor in shape ``([batch_shape], N)``
            The observed target values.
        function_dist : torch.distributions.Distribution
            The distribution over function values
            with batch shape ``(Q, [batch_shape])`` and event shape ``(N,)``.

        Returns
        -------
        torch.Tensor with shape ``([batch_shape], N)``
            The expected log probability of observations under the ALD likelihood.
        """
        ...

    def predictive_posterior(self, gp_posterior):
        """Predictive posterior distribution of function values.

        Parameters
        ----------
        gp_posterior : gpytorch.distributions.MultivariateNormal
            The joint posterior over latent GPs at input locations.

        Returns
        -------
        samples : torch.Tensor
            Samples drawn from the predictive posterior distribution of function values.
            Shape is ``(S, Q, [batch_shape], N)`` for batch GPQR
            or ``(S, [batch_shape], N, Q)`` for multitask GPQR,
            where *S* is the number of samples, *Q* is the number of quantiles,
            and *N* is the number of data points.

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
