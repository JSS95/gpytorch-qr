"""Asymmetric Laplace distributions likelihoods for quantile regression."""

import gpytorch
import torch

from .distributions import BatchQuantileALD, MultitaskQuantileALD
from .utils import centergap_to_quantiles

__all__ = [
    "ALDLikelihood",
    "BatchQuantileGPLikelihood",
    "MultitaskQuantileGPLikelihood",
    "BatchCenterGapQuantileGPLikelihood",
    "MultitaskCenterGapQuantileGPLikelihood",
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


class BatchQuantileGPLikelihood(ALDLikelihood):
    """Likelihood for :class:`BatchQuantileALD` with direct representation.

    Parameters
    ----------
    q
        The quantile levels.
        Shape is ``(Q, *B)``.
    raw_scales
        The initial untransformed scales of the asymmetric Laplace distribution.
        Shape is either ``()`` or ``(Q, *B)``.
    learn_scales

    Attributes
    ----------
    q : torch.Tensor with shape ``(Q, *B)``
    raw_scales : torch.Tensor with shape ``(Q, *B)``

    Examples
    --------
    >>> import torch
    >>> from torch.distributions import Normal
    >>> torch.manual_seed(42)  # doctest: +IGNORE_OUTPUT
    >>> def mean(x):
    ...     return torch.cos(x * 2 * 3.14)
    >>> def std(x):
    ...     return x + 0.1
    >>> x_range = torch.linspace(0, 1, 10).reshape(-1, 1)
    >>> x = x_range.repeat(2, 1)
    >>> y = (mean(x) + torch.randn(x.shape).mul(std(x))).squeeze()
    >>> q = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
    >>> true_quantiles = mean(x_range) + std(x_range) * Normal(0, 1).icdf(q)
    >>> from gpytorch.variational import CholeskyVariationalDistribution
    >>> from gpytorch.variational import VariationalStrategy
    >>> from gpytorch.means import ConstantMean
    >>> from gpytorch.kernels import RBFKernel, ScaleKernel
    >>> from gpytorch_qr.models import DirectQuantileGP
    >>> from gpytorch_qr.likelihoods import BatchQuantileGPLikelihood
    >>> class MyGP(DirectQuantileGP):
    ...     def __init__(self, inducing_points, num_quantiles):
    ...         N, D = inducing_points.size()
    ...         variational_distribution = CholeskyVariationalDistribution(
    ...             N,
    ...             batch_shape=torch.Size([num_quantiles]),
    ...         )
    ...         variational_strategy = VariationalStrategy(
    ...             self,
    ...             inducing_points,
    ...             variational_distribution,
    ...             learn_inducing_locations=True,
    ...         )
    ...         mean = ConstantMean(batch_shape=torch.Size([num_quantiles]))
    ...         covar = ScaleKernel(
    ...             RBFKernel(ard_num_dims=D, batch_shape=torch.Size([num_quantiles])),
    ...             batch_shape=torch.Size([num_quantiles]),
    ...         )
    ...         super().__init__(variational_strategy, mean, covar, 0)
    >>> inducing_points = torch.linspace(0, 1, 10).reshape(-1, 1)
    >>> gp = MyGP(inducing_points, len(q))
    >>> likelihood = BatchQuantileGPLikelihood(q)
    >>> from gpytorch.mlls import VariationalELBO
    >>> gp.train()  # doctest: +IGNORE_OUTPUT
    >>> likelihood.train()  # doctest: +IGNORE_OUTPUT
    >>> mll = VariationalELBO(likelihood, gp, num_data=y.numel())
    >>> optimizer = torch.optim.Adam(
    ...     list(gp.parameters()) + list(likelihood.parameters()),
    ...     lr=0.001,
    ... )
    >>> N = 1  # Set to 1 for faster training; increase for better performance
    >>> for _ in range(N):
    ...     output = gp(x)
    ...     loss = -mll(output, y).sum()
    ...     loss.backward()
    ...     optimizer.step()
    ...     optimizer.zero_grad()
    >>> gp.eval()  # doctest: +IGNORE_OUTPUT
    >>> x_pred = torch.linspace(0, 2, 100).reshape(-1, 1)
    >>> with torch.no_grad():
    ...     mean_q = gp.mean_quantiles(x_pred)
    >>> import matplotlib.pyplot as plt
    >>> plt.scatter(x, y, c='gray', marker='.', alpha=0.1)  # doctest: +IGNORE_OUTPUT
    >>> plt.plot(x_range, true_quantiles, '--', c='k')  # doctest: +IGNORE_OUTPUT
    >>> plt.plot(x_pred, mean_q.T)  # doctest: +IGNORE_OUTPUT
    """

    def forward(self, function_samples):
        """Return the ALD distribution for the given function samples.

        Parameters
        ----------
        function_samples : torch.Tensor with shape ``(S, Q, *B, N)``
            The function samples drawn from the posterior distributions of quantile
            functions. *S* is the number of samples, *Q* is the number of quantiles,
            *B* is the batch shape, and *N* is the number of data points.

        Returns
        -------
        BatchQuantileALD
        """
        return BatchQuantileALD(
            m=function_samples,
            lamda=self.scales.unsqueeze(-1),  # (Q, *B, 1)
            kappa=self.q.unsqueeze(-1),  # (Q, *B, 1)
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
        torch.Tensor with shape ``(*B, N)``
            The expected log probability of the observed data under the ALD likelihood.
        """
        # lp: (Q, *B, N)
        lp = super().expected_log_prob(observations, function_dist, *args, **kwargs)
        return lp.sum(dim=0)


class MultitaskQuantileGPLikelihood(ALDLikelihood):
    """Likelihood for :class:`MultitaskQuantileALD` with direct representation.

    It is recommended to use fewer latent GPs than the number of tasks(=quantiles)
    to model the correlation structure.

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

    Example
    -------
    >>> import torch
    >>> from torch.distributions import Normal
    >>> torch.manual_seed(42)  # doctest: +IGNORE_OUTPUT
    >>> def mean(x):
    ...     return torch.cos(x * 2 * 3.14)
    >>> def std(x):
    ...     return x + 0.1
    >>> x_range = torch.linspace(0, 1, 10).reshape(-1, 1)
    >>> x = x_range.repeat(2, 1)
    >>> y = (mean(x) + torch.randn(x.shape).mul(std(x))).squeeze()
    >>> q = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
    >>> true_quantiles = mean(x_range) + std(x_range) * Normal(0, 1).icdf(q)
    >>> from gpytorch.variational import CholeskyVariationalDistribution
    >>> from gpytorch.variational import VariationalStrategy, LMCVariationalStrategy
    >>> from gpytorch.means import ConstantMean
    >>> from gpytorch.kernels import RBFKernel, ScaleKernel
    >>> from gpytorch_qr.models import DirectQuantileGP
    >>> from gpytorch_qr.likelihoods import MultitaskQuantileGPLikelihood
    >>> class MyGP(DirectQuantileGP):
    ...     def __init__(self, inducing_points, num_latents, num_quantiles):
    ...         N, D = inducing_points.size()
    ...         variational_distribution = CholeskyVariationalDistribution(
    ...             N,
    ...             batch_shape=torch.Size([num_latents]),
    ...         )
    ...         variational_strategy = LMCVariationalStrategy(
    ...             VariationalStrategy(
    ...                 self,
    ...                 inducing_points,
    ...                 variational_distribution,
    ...                 learn_inducing_locations=True,
    ...             ),
    ...             num_tasks=num_quantiles,
    ...             num_latents=num_latents,
    ...         )
    ...         mean_module = ConstantMean(batch_shape=torch.Size([num_latents]))
    ...         covar_module = ScaleKernel(
    ...             RBFKernel(ard_num_dims=D, batch_shape=torch.Size([num_latents])),
    ...             batch_shape=torch.Size([num_latents]),
    ...         )
    ...         super().__init__(variational_strategy, mean_module, covar_module, -1)
    >>> inducing_points = torch.linspace(0, 1, 10).reshape(-1, 1)
    >>> num_latents = len(q) - 2  # recommended to be smaller than q
    >>> gp = MyGP(inducing_points, num_latents, len(q))
    >>> likelihood = MultitaskQuantileGPLikelihood(q)
    >>> from gpytorch.mlls import VariationalELBO
    >>> gp.train()  # doctest: +IGNORE_OUTPUT
    >>> likelihood.train()  # doctest: +IGNORE_OUTPUT
    >>> mll = VariationalELBO(likelihood, gp, num_data=y.numel())
    >>> optimizer = torch.optim.Adam(
    ...     list(gp.parameters()) + list(likelihood.parameters()),
    ...     lr=0.001,
    ... )
    >>> N = 1  # Set to 1 for faster training; increase for better performance
    >>> for _ in range(N):
    ...     output = gp(x)
    ...     loss = -mll(output, y)
    ...     loss.backward()
    ...     optimizer.step()
    ...     optimizer.zero_grad()
    >>> gp.eval()  # doctest: +IGNORE_OUTPUT
    >>> x_pred = torch.linspace(0, 2, 100).reshape(-1, 1)
    >>> with torch.no_grad():
    ...     quantiles = gp.mean_quantiles(x_pred)
    >>> import matplotlib.pyplot as plt
    >>> plt.scatter(x, y, c='gray', marker='.', alpha=0.1)  # doctest: +IGNORE_OUTPUT
    >>> plt.plot(x_range, true_quantiles, '--', c='k')  # doctest: +IGNORE_OUTPUT
    >>> plt.plot(x_pred, quantiles)  # doctest: +IGNORE_OUTPUT
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


class BatchCenterGapQuantileGPLikelihood(ALDLikelihood):
    """Likelihood for :class:`BatchQuantileALD` with center-gap representation.

    Parameters
    ----------
    q
        The quantile levels.
        Shape is ``(Q, *B)``.
    central_quantile_index : int
        The index of the central quantile in the quantile levels.
    raw_scales
        The initial untransformed scales of the asymmetric Laplace distribution.
        Shape is either ``()`` or ``(Q, *B)``.
    learn_scales

    Attributes
    ----------
    q : torch.Tensor with shape ``(Q, *B)``
    raw_scales : torch.Tensor with shape ``(Q, *B)``

    Examples
    --------
    >>> import torch
    >>> from torch.distributions import Normal
    >>> torch.manual_seed(42)  # doctest: +IGNORE_OUTPUT
    >>> def mean(x):
    ...     return torch.cos(x * 2 * 3.14)
    >>> def std(x):
    ...     return x + 0.1
    >>> x_range = torch.linspace(0, 1, 10).reshape(-1, 1)
    >>> x = x_range.repeat(2, 1)
    >>> y = (mean(x) + torch.randn(x.shape).mul(std(x))).squeeze()
    >>> q = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
    >>> true_quantiles = mean(x_range) + std(x_range) * Normal(0, 1).icdf(q)
    >>> from gpytorch.variational import CholeskyVariationalDistribution
    >>> from gpytorch.variational import VariationalStrategy
    >>> from gpytorch.means import ConstantMean
    >>> from gpytorch.kernels import RBFKernel, ScaleKernel
    >>> from gpytorch_qr.means import CenterGapMean
    >>> from gpytorch_qr.models import CenterGapQuantileGP
    >>> from gpytorch_qr.likelihoods import BatchCenterGapQuantileGPLikelihood
    >>> class MyGP(CenterGapQuantileGP):
    ...     def __init__(self, inducing_points, num_quantiles, num_lower_q):
    ...         N, D = inducing_points.size()
    ...         variational_distribution = CholeskyVariationalDistribution(
    ...             N,
    ...             batch_shape=torch.Size([num_quantiles]),
    ...         )
    ...         variational_strategy = VariationalStrategy(
    ...             self,
    ...             inducing_points,
    ...             variational_distribution,
    ...             learn_inducing_locations=True,
    ...         )
    ...         mean = CenterGapMean(
    ...             ConstantMean(batch_shape=torch.Size([1])),
    ...             ConstantMean(batch_shape=torch.Size([num_quantiles - 1])),
    ...             latent_dim=0,
    ...         )
    ...         covar = ScaleKernel(
    ...             RBFKernel(ard_num_dims=D, batch_shape=torch.Size([num_quantiles])),
    ...             batch_shape=torch.Size([num_quantiles]),
    ...         )
    ...         super().__init__(variational_strategy, mean, covar, 0, num_lower_q)
    >>> inducing_points = torch.linspace(0, 1, 10).reshape(-1, 1)
    >>> central_q_index = (q - 0.5).abs().argmin().item()
    >>> gp = MyGP(inducing_points, len(q), central_q_index)
    >>> likelihood = BatchCenterGapQuantileGPLikelihood(q, central_q_index)
    >>> from gpytorch.mlls import VariationalELBO
    >>> gp.train()  # doctest: +IGNORE_OUTPUT
    >>> likelihood.train()  # doctest: +IGNORE_OUTPUT
    >>> mll = VariationalELBO(likelihood, gp, num_data=y.numel())
    >>> optimizer = torch.optim.Adam(
    ...     list(gp.parameters()) + list(likelihood.parameters()),
    ...     lr=0.001,
    ... )
    >>> N = 1  # Set to 1 for faster training; increase for better performance
    >>> for _ in range(N):
    ...     output = gp(x)
    ...     loss = -mll(output, y).sum()
    ...     loss.backward()
    ...     optimizer.step()
    ...     optimizer.zero_grad()
    >>> gp.eval()  # doctest: +IGNORE_OUTPUT
    >>> x_pred = torch.linspace(0, 2, 100).reshape(-1, 1)
    >>> with torch.no_grad():
    ...     quantiles = gp.mean_quantiles_mc(x_pred)
    >>> import matplotlib.pyplot as plt
    >>> plt.scatter(x, y, c='gray', marker='.', alpha=0.1)  # doctest: +IGNORE_OUTPUT
    >>> plt.plot(x_range, true_quantiles, '--', c='k')  # doctest: +IGNORE_OUTPUT
    >>> plt.plot(x_pred, quantiles.T)  # doctest: +IGNORE_OUTPUT
    """

    def __init__(self, q, central_quantile_index, raw_scales=0.0, learn_scales=True):
        super().__init__(q, raw_scales, learn_scales)
        central_quantile = self.q[central_quantile_index]
        self.lower_count = (self.q < central_quantile).count_nonzero()

    def forward(self, function_samples):
        """Return the ALD distribution for the given function samples.

        Parameters
        ----------
        function_samples : torch.Tensor with shape ``(S, Q, *B, N)``
            The function samples drawn from the posterior distributions of quantile
            functions. *S* is the number of samples, *Q* is the number of quantiles,
            *B* is the batch shape, and *N* is the number of data points.

        Returns
        -------
        BatchQuantileALD
        """
        center = function_samples[:, :1, ...]
        lower_gaps = function_samples[:, 1 : 1 + self.lower_count, ...]
        upper_gaps = function_samples[:, 1 + self.lower_count :, ...]
        quantiles = centergap_to_quantiles(
            center, lower_gaps, upper_gaps, quantile_dim=1
        )
        return BatchQuantileALD(
            m=quantiles,
            lamda=self.scales.unsqueeze(-1),  # (Q, *B, 1)
            kappa=self.q.unsqueeze(-1),  # (Q, *B, 1)
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
        torch.Tensor with shape ``(*B, N)``
            The expected log probability of the observed data under the ALD likelihood.
        """
        # lp: (Q, *B, N)
        lp = super().expected_log_prob(observations, function_dist, *args, **kwargs)
        return lp.sum(dim=0)


class MultitaskCenterGapQuantileGPLikelihood(ALDLikelihood):
    """Likelihood for :class:`MultitaskQuantileALD` with center-gap representation.

    Latent GPs model the central quantile and the gaps between quantiles separately.

    It is recommended to use fewer latent GPs than the number of tasks(=quantiles)
    to model the correlation structure.

    Parameters
    ----------
    q
        The quantile levels.
        Shape is ``(*B, Q)``.
    central_quantile_index : int
        The index of the central quantile in the quantile levels.
    raw_scales
        The initial untransformed scales of the asymmetric Laplace distribution.
        Shape is either ``()`` or ``(*B, Q)``.
    learn_scales

    Attributes
    ----------
    q : torch.Tensor with shape ``(*B, Q)``
    raw_scales : torch.Tensor with shape ``(*B, Q)``

    Examples
    --------
    >>> import torch
    >>> from torch.distributions import Normal
    >>> torch.manual_seed(42)  # doctest: +IGNORE_OUTPUT
    >>> def mean(x):
    ...     return torch.cos(x * 2 * 3.14)
    >>> def std(x):
    ...     return x + 0.1
    >>> x_range = torch.linspace(0, 1, 10).reshape(-1, 1)
    >>> x = x_range.repeat(2, 1)
    >>> y = (mean(x) + torch.randn(x.shape).mul(std(x))).squeeze()
    >>> q = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
    >>> true_quantiles = mean(x_range) + std(x_range) * Normal(0, 1).icdf(q)
    >>> from gpytorch.variational import CholeskyVariationalDistribution
    >>> from gpytorch.variational import VariationalStrategy
    >>> from gpytorch.means import ConstantMean
    >>> from gpytorch.kernels import RBFKernel, ScaleKernel
    >>> from gpytorch_qr.means import CenterGapMean
    >>> from gpytorch_qr.models import CenterGapQuantileGP
    >>> from gpytorch_qr.likelihoods import MultitaskCenterGapQuantileGPLikelihood
    >>> from gpytorch_qr.variational import CGBlkdiagLmcVariationalStrategy
    >>> class MyGP(CenterGapQuantileGP):
    ...     def __init__(
    ...         self,
    ...         inducing_points,
    ...         num_quantiles,
    ...         num_lower_q,
    ...         num_latents,
    ...         num_lower_latents,
    ...     ):
    ...         N, D = inducing_points.size()
    ...         variational_distribution = CholeskyVariationalDistribution(
    ...             N,
    ...             batch_shape=torch.Size([num_latents]),
    ...         )
    ...         variational_strategy = CGBlkdiagLmcVariationalStrategy(
    ...             VariationalStrategy(
    ...                 self,
    ...                 inducing_points,
    ...                 variational_distribution,
    ...                 learn_inducing_locations=True,
    ...             ),
    ...             num_quantiles=num_quantiles,
    ...             num_latents=num_latents,
    ...             num_lower_quantiles=num_lower_q,
    ...             num_lower_latents=num_lower_latents,
    ...         )
    ...         mean = CenterGapMean(
    ...             ConstantMean(batch_shape=torch.Size([1])),
    ...             ConstantMean(batch_shape=torch.Size([num_latents - 1])),
    ...             latent_dim=-1,
    ...         )
    ...         covar = ScaleKernel(
    ...             RBFKernel(ard_num_dims=D, batch_shape=torch.Size([num_latents])),
    ...             batch_shape=torch.Size([num_latents]),
    ...         )
    ...         super().__init__(variational_strategy, mean, covar, -1, num_lower_q)
    >>> inducing_pts = torch.linspace(0, 1, 10).reshape(-1, 1)
    >>> central_q_index = (q - 0.5).abs().argmin().item()
    >>> num_latents = len(q) - 2  # recommended to be smaller than q
    >>> gp = MyGP(inducing_pts, len(q), central_q_index, num_latents, num_latents // 2)
    >>> likelihood = MultitaskCenterGapQuantileGPLikelihood(q, central_q_index)
    >>> from gpytorch.mlls import VariationalELBO
    >>> gp.train()  # doctest: +IGNORE_OUTPUT
    >>> likelihood.train()  # doctest: +IGNORE_OUTPUT
    >>> mll = VariationalELBO(likelihood, gp, num_data=y.numel())
    >>> optimizer = torch.optim.Adam(
    ...     list(gp.parameters()) + list(likelihood.parameters()),
    ...     lr=0.001,
    ... )
    >>> N = 1  # Set to 1 for faster training; increase for better performance
    >>> for _ in range(N):
    ...     output = gp(x)
    ...     loss = -mll(output, y)
    ...     loss.backward()
    ...     optimizer.step()
    ...     optimizer.zero_grad()
    >>> gp.eval()  # doctest: +IGNORE_OUTPUT
    >>> x_pred = torch.linspace(0, 2, 100).reshape(-1, 1)
    >>> with torch.no_grad():
    ...     quantiles = gp.mean_quantiles_mc(x_pred)
    >>> import matplotlib.pyplot as plt
    >>> plt.scatter(x, y, c='gray', marker='.', alpha=0.1)  # doctest: +IGNORE_OUTPUT
    >>> plt.plot(x_range, true_quantiles, '--', c='k')  # doctest: +IGNORE_OUTPUT
    >>> plt.plot(x_pred, quantiles)  # doctest: +IGNORE_OUTPUT
    """

    def __init__(self, q, central_quantile_index, raw_scales=0.0, learn_scales=True):
        super().__init__(q, raw_scales, learn_scales)
        central_quantile = self.q[..., central_quantile_index]
        self.lower_count = (self.q < central_quantile).count_nonzero()

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
        center = function_samples[..., :1]
        lower_gaps = function_samples[..., 1 : 1 + self.lower_count]
        upper_gaps = function_samples[..., 1 + self.lower_count :]
        quantiles = centergap_to_quantiles(
            center, lower_gaps, upper_gaps, quantile_dim=-1
        )
        return MultitaskQuantileALD(
            m=quantiles,
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
