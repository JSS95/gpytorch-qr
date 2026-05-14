import abc

__all__ = [
    "BayesianQRMixin",
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
