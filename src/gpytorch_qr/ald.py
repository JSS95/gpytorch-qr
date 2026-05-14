"""Asymmetric Laplace distributions for quantile regression."""

import gpytorch
import torch

__all__ = [
    "BatchALD",
    "MultitaskALD",
    "ALDLikelihood",
]


class BatchALD(torch.distributions.Distribution):
    """Asymmetric Laplace distribution where quantiles are treated as batches.

    Parameters
    ----------
    m : torch.Tensor with shape (S, Q, [batch_shape], N)
        The location parameters of the distribution.
    lamda : torch.Tensor with shape (Q, [batch_shape])
        The scale parameters of the distribution for each quantile.
    kappa : torch.Tensor with shape (Q, [batch_shape])
        The quantile levels of the distribution.

    Notes
    -----
    ``batch_shape`` is for optional additional batches,
    e.g., cross validation folds.
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
        self.lamda = lamda.reshape(1, *lamda.shape, 1)
        self.kappa = kappa.reshape(1, *kappa.shape, 1)
        super().__init__(m.shape)

    def log_prob(self, value):
        """Log probability of the asymmetric Laplace distribution at the given value.

        Parameters
        ----------
        value : torch.Tensor with shape ([batch_shape], N)
            Observed response variables at which to evaluate the log probability.

        Returns
        -------
        logp : torch.Tensor with shape (S, Q, [batch_shape], N)
            The log probability at the given values for each quantile and sample.
        """
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
        """Inverse CDF of the asymmetric Laplace distribution.

        Parameters
        ----------
        value : torch.Tensor with shape (S, Q, [batch_shape], N)
            Probabilities at which to evaluate the inverse CDF. Must be in (0, 1).

        Returns
        -------
        torch.Tensor with shape (S, Q, [batch_shape], N)
            The corresponding quantiles of the distribution.
        """
        return torch.where(
            value <= self.kappa,
            self.m + self.lamda / (1 - self.kappa) * torch.log(value / self.kappa),
            self.m
            - self.lamda / self.kappa * torch.log((1 - value) / (1 - self.kappa)),
        )


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

    def icdf(self, value):
        """Inverse CDF of the asymmetric Laplace distribution.

        Parameters
        ----------
        value : torch.Tensor with shape (S, N, Q)
            Probabilities at which to evaluate the inverse CDF. Must be in (0, 1).

        Returns
        -------
        torch.Tensor with shape (S, N, Q)
            The corresponding quantiles of the distribution.
        """
        return torch.where(
            value <= self.kappa,
            self.m + self.lamda / (1 - self.kappa) * torch.log(value / self.kappa),
            self.m
            - self.lamda / self.kappa * torch.log((1 - value) / (1 - self.kappa)),
        )


class ALDLikelihood(gpytorch.likelihoods.Likelihood):
    """Asymmetric Laplace distribution likelihood.

    Parameters
    ----------
    q : torch.Tensor with shape (Q, [batch_shape])
        The quantile levels.
    raw_scales : torch.Tensor with shape (Q, [batch_shape]) or scalar, default=0
        The initial untransformed scales of the asymmetric Laplace distribution.
        The actual scales are obtained by applying the positive transformation.
        Scalar value is broadcasted to the shape of *q*.
    learn_scales : bool, default=True
        Whether to update scales by gradients.

    Notes
    -----
    The ``batch_shape`` can either be:

    - A broadcastable shape, e.g., ``(1,)``.
    - A fixed shape, e.g., ``(B,)``.

    Different batch shape representations are allowed for *q* and *raw_scales*.
    For example, *q* can be in  ``(Q, 1)`` while *raw_scales* is in ``(Q, B)``.
    However, different choices can lead to vastly different model behavior.

    Usually, if there is no batch dimension, you would want to use:

    - *q* in shape ``(Q,)``.
    - *raw_scales* in scalar (usually 0).
    - ``learn_scales=True``.

    If there is batch dimension, you would usually want to use:

    - *q* in shape ``(Q, 1)``.
    - *raw_scales* in shape ``(Q, B)``.
    - ``learn_scales=True``.

    .. rubric:: q

    If batch shape of ``(1,)`` is used, same levels of quantiles are used for
    all batches.
    Should different quantile levels be desired for different batches,
    use batch shape of ``(B,)``.

    Note that it is impossible to vary the number of quantiles *Q* across batches.

    .. rubric:: raw_scales

    If batch shape of ``(1,)`` is used, same scales are used for all batches.
    When ``learn_scales`` is True, this makes the scales to be updated by gradients
    averaged across batches.
    To allow different batches to have different scales, use batch shape of ``(B,)``.

    The quantile dimension of *raw_scales* can be broadcasted as well by using
    quantile shape of *1* instead of *Q*.
    When ``learn_scales`` is True, this makes the scales to be updated by gradients
    averaged across quantiles.

    .. rubric:: learn_scales

    If ``learn_scales`` is False, there is no learnable parameter and
    broadcasting does not matter.
    """

    def __init__(self, q, raw_scales=0.0, learn_scales=True):
        super().__init__()
        self.register_buffer("q", q.float())

        raw_scales = torch.as_tensor(raw_scales, dtype=torch.float32)
        if raw_scales.ndim == 0:
            raw_scales = torch.full_like(q, raw_scales)
        if learn_scales:
            self.register_parameter("raw_scales", torch.nn.Parameter(raw_scales))
        else:
            self.register_buffer("raw_scales", raw_scales)
        self.register_constraint("raw_scales", gpytorch.constraints.Positive())

    @property
    def scales(self):
        return self.raw_scales_constraint.transform(self.raw_scales)

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
