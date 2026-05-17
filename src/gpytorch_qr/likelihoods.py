"""Asymmetric Laplace distributions likelihoods for quantile regression."""

import gpytorch
import torch

__all__ = [
    "ALDLikelihood",
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
