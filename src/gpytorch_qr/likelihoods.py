"""Asymmetric Laplace distributions likelihoods for quantile regression."""

import gpytorch
import torch

from .distributions import QuantileALD
from .utils import centergap_to_quantiles

__all__ = [
    "ALDLikelihood",
    "DirectQuantileLikelihood",
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
    Whether to broadcast ``raw_scales`` is important when ``learn_scales=True``.
    When the scale is broadcasted, the same scale parameter is shared across and updated
    along the broadcasted dimension, e.g., across different quantile levels or batches.

    Sharing scales across quantiles may be deliberately used to reduce the number of
    parameters and increase the stability of training.
    On the other hand, sharing scales across batches usually does not make sense and
    should be avoided.
    In general, it is recommended to use independent scale parameters for all channels.

    To encourage the use of independent scale parameters, scalar ``raw_scales`` is
    repeated to the shape of *q* instead of being broadcasted.
    For example, if *q* has shape ``(*B, T)`` and ``raw_scales`` is ``Tensor(1)``
    whose shape is ``()``, then it is converted to a tensor of shape ``(*B, T)``
    where all values are 1.
    On the other hand, if ``raw_scales`` is ``Tensor([[1]])`` whose shape is ``(1, 1)``,
    then it is broadcasted to shape ``(*B, T)`` and shared across tasks and batches.
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


class DirectQuantileLikelihood(ALDLikelihood):
    """Likelihood for :class:`QuantileALD` with direct representation.

    It is recommended to use fewer latent GPs than the number of tasks
    to model the correlation structure.

    Parameters
    ----------
    q
        The quantile levels.
        Shape is ``(*B, T)``.
    raw_scales
        The initial untransformed scales of the asymmetric Laplace distribution.
        Shape is either ``()`` or ``(*B, T)``.
    learn_scales

    Attributes
    ----------
    q : torch.Tensor with shape ``(*B, T)``
    raw_scales : torch.Tensor with shape ``(*B, T)``

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
    >>> from gpytorch_qr.likelihoods import DirectQuantileLikelihood
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
    ...         super().__init__(variational_strategy, mean_module, covar_module)
    >>> inducing_points = torch.linspace(0, 1, 10).reshape(-1, 1)
    >>> num_latents = len(q) - 2  # recommended to be smaller than q
    >>> gp = MyGP(inducing_points, num_latents, len(q))
    >>> likelihood = DirectQuantileLikelihood(q)
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
        function_samples : torch.Tensor with shape ``(S, *B, N, T)``
            The function samples drawn from the posterior distributions of quantile
            functions. *S* is the number of samples, *T* is the number of tasks,
            *B* is the batch shape, and *N* is the number of data points.

        Returns
        -------
        QuantileALD
        """
        return QuantileALD(
            m=function_samples,
            lamda=self.scales.unsqueeze(-2),  # (*B, 1, T)
            kappa=self.q.unsqueeze(-2),  # (*B, 1, T)
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
        torch.Tensor with shape ``(*B, T)``
            The expected log probability of the observed data under the ALD likelihood.
        """
        # lp: (*B, N, T)
        lp = super().expected_log_prob(observations, function_dist, *args, **kwargs)
        return lp.sum(dim=-2)


class MultitaskCenterGapQuantileGPLikelihood(ALDLikelihood):
    """Likelihood for :class:`QuantileALD` with center-gap representation.

    Latent GPs model the central quantile and the gaps between quantiles separately.

    It is recommended to use fewer latent GPs than the number of tasks
    to model the correlation structure.

    Parameters
    ----------
    q
        The quantile levels.
        Shape is ``(*B, T)``.
    central_quantile_index : int
        The index of the central quantile in the quantile levels.
    raw_scales
        The initial untransformed scales of the asymmetric Laplace distribution.
        Shape is either ``()`` or ``(*B, T)``.
    learn_scales

    Attributes
    ----------
    q : torch.Tensor with shape ``(*B, T)``
    raw_scales : torch.Tensor with shape ``(*B, T)``

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
    ...         )
    ...         covar = ScaleKernel(
    ...             RBFKernel(ard_num_dims=D, batch_shape=torch.Size([num_latents])),
    ...             batch_shape=torch.Size([num_latents]),
    ...         )
    ...         super().__init__(variational_strategy, mean, covar, num_lower_q)
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
        idx = torch.as_tensor(central_quantile_index).long()
        if idx.dim() == 0:
            idx_for_gather = idx.view(1).expand(
                list(self.q.shape[:-1]) + [1]
            )  # (*B, 1)
        else:
            idx_for_gather = idx.unsqueeze(-1)  # (*B, 1)
        idx_for_gather = idx_for_gather.to(self.q.device)
        central_quantile = self.q.gather(-1, idx_for_gather).squeeze(-1)  # (*B)
        self.lower_count = (self.q < central_quantile.unsqueeze(-1)).sum(dim=-1)  # (*B)

    def forward(self, function_samples):
        """Return the ALD distribution for the given function samples.

        Parameters
        ----------
        function_samples : torch.Tensor with shape ``(S, *B, N, T)``
            The function samples drawn from the posterior distributions of quantile
            functions. *S* is the number of samples, *T* is the number of tasks,
            *B* is the batch shape, and *N* is the number of data points.

        Returns
        -------
        QuantileALD
        """
        lc = self.lower_count
        if lc.dim() == 0:
            lc_int = int(lc)
            center = function_samples[..., :1]
            lower_gaps = function_samples[..., 1 : 1 + lc_int]
            upper_gaps = function_samples[..., 1 + lc_int :]
            quantiles = centergap_to_quantiles(
                center, lower_gaps, upper_gaps, quantile_dim=-1
            )
        else:
            # Derive actual batch shape from function_samples, not from lc,
            # because lc may have been computed from a broadcasted q.
            S = function_samples.shape[0]
            N = function_samples.shape[-2]
            T = function_samples.shape[-1]
            B_shape = function_samples.shape[1:-2]  # actual (*B)
            B_flat = 1
            for d in B_shape:
                B_flat *= d
            # Flatten *B: (S, B_flat, N, T)
            fs_flat = function_samples.reshape(S, B_flat, N, T)
            lc_flat = lc.reshape(-1).expand(B_flat)  # broadcast lc to (B_flat,)
            quantiles_flat = torch.empty_like(fs_flat)
            for unique_lc in lc_flat.unique():
                lc_val = int(unique_lc)
                mask = lc_flat == unique_lc
                fs_group = fs_flat[:, mask, :, :]  # (S, G, N, T)
                center = fs_group[..., :1]
                lower_gaps = fs_group[..., 1 : 1 + lc_val]
                upper_gaps = fs_group[..., 1 + lc_val :]
                quantiles_flat[:, mask, :, :] = centergap_to_quantiles(
                    center, lower_gaps, upper_gaps, quantile_dim=-1
                )
            quantiles = quantiles_flat.reshape(S, *B_shape, N, T)
        return QuantileALD(
            m=quantiles,
            lamda=self.scales.unsqueeze(-2),  # (*B, 1, T)
            kappa=self.q.unsqueeze(-2),  # (*B, 1, T)
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
        torch.Tensor with shape ``(*B, T)``
            The expected log probability of the observed data under the ALD likelihood.
        """
        # lp: (*B, N, T)
        lp = super().expected_log_prob(observations, function_dist, *args, **kwargs)
        return lp.sum(dim=-2)
