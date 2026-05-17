"""Asymmetric Laplace distributions likelihoods for quantile regression."""

import gpytorch
import torch

from .distributions import MultitaskQuantileALD

__all__ = [
    "ALDLikelihood",
    "MultitaskQuantileALDLikelihood",
]


class ALDLikelihood(gpytorch.likelihoods.Likelihood):
    """Asymmetric Laplace distribution likelihood.

    Parameters
    ----------
    q : torch.Tensor
        The quantile levels.
    raw_scales : torch.Tensor or scalar, default=0
        The initial untransformed scales of the asymmetric Laplace distribution.
        The actual scales are obtained by applying the positive transformation.
        If tensor, dimension should be same to *q* and shape should be broadcastable.
        Scalar value is repeated to the shape of *q*.
    learn_scales : bool, default=True
        Whether to update scales by gradients.

    Notes
    -----
    When ``learn_scales=True``, broadcasted parameters in ``raw_scales`` are
    updated from multiple channels.
    It is usually recommended to use independent scale parameters for all channels.

    If ``raw_scales`` is a scalar, e.g., ``Tensor(1)``, it is repeated to the shape
    of *q* instead of being broadcasted.
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
            The first dimension is the sampling dimension.

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


class MultitaskQuantileALDLikelihood(ALDLikelihood):
    """Likelihood for :class:`MultitaskQuantileALD`.

    Parameters
    ----------
    q
        The quantile levels.
        Shape is ``(*B, Q)``.
    raw_scales
        The initial untransformed scales of the asymmetric Laplace distribution.
        Shape is either ``()`` or ``(*B, Q)``.
    learn_scales

    Attributes
    ----------
    q : torch.Tensor with shape ``(*B, Q)``
    raw_scales : torch.Tensor with shape ``(*B, Q)``
    """

    def forward(self, function_samples):
        """Return the ALD distribution for the given function samples.

        Parameters
        ----------
        function_samples : torch.Tensor with shape ``(S, *B, N, Q)``
            The function samples drawn from the posterior distributions of quantile
            functions. *S* is the number of samples, *Q* is the number of quantiles,
            *B* is the batch shape, and *N* is the number of data points.

        Returns
        -------
        MultitaskQuantileALD
        """
        function_samples = self.latent_to_quantiles(function_samples)
        return MultitaskQuantileALD(
            m=function_samples,
            lamda=self.scales.unsqueeze(-2),  # (*B, 1, Q)
            kappa=self.q.unsqueeze(-2),  # (*B, 1, Q)
        )

    def expected_log_prob(self, observations, function_dist, *args, **kwargs):
        """Expected log probability of the observed data under the ALD likelihood.

        Parameters
        ----------
        observations : torch.Tensor with shape ``(*B, N)``
            The observed response variables.
        function_dist : torch.distributions.Distribution
            The distribution of the function values at the input locations.

        Returns
        -------
        torch.Tensor with shape ``(*B, Q)``
            The expected log probability of the observed data under the ALD likelihood.
        """
        # lp: (*B, N, Q)
        lp = super().expected_log_prob(observations, function_dist, *args, **kwargs)
        return lp.sum(dim=-2)
