"""Asymmetric Laplace distributions likelihoods for quantile regression."""

import gpytorch
import torch
from gpytorch.likelihoods import Likelihood
from gpytorch.likelihoods.noise_models import HomoskedasticNoise

from .distributions import ALD, QuantileALD
from .utils import centergap_to_quantiles

__all__ = [
    "AsymmetricLaplaceLikelihood",
    "MultitaskAsymmetricLaplaceLikelihood",
    "DirectQuantileLikelihood",
    "CenterGapQuantileLikelihood",
    "MultiOutputDirectQuantileLikelihood",
    "MultiOutputCenterGapQuantileLikelihood",
]


class _ALDLikelihoodBase(Likelihood):
    """Base class for ALD likelihoods for quantile regression."""

    has_analytical_marginal = False

    def __init__(self, kappa, noise_covar):
        super().__init__()
        self.register_buffer(
            "kappa", torch.as_tensor(kappa, dtype=torch.get_default_dtype())
        )
        self.noise_covar = noise_covar

    @property
    def noise(self):
        r"""The ALD noise variance :math:`\sigma^2`."""
        return self.noise_covar.noise

    @noise.setter
    def noise(self, value):
        self.noise_covar.initialize(noise=value)

    @property
    def raw_noise(self):
        """The unconstrained parameter corresponding to :attr:`noise`."""
        return self.noise_covar.raw_noise

    @raw_noise.setter
    def raw_noise(self, value):
        self.noise_covar.initialize(raw_noise=value)

    def _shaped_noise_covar(self, base_shape, *params, **kwargs):
        return self.noise_covar(*params, shape=base_shape, **kwargs)

    def expected_log_prob(self, observations, function_dist, *args, **kwargs):
        """Expected log probability of the observed data under the ALD likelihood.

        Parameters
        ----------
        observations : torch.Tensor with shape ``(*B, N)``
            The observed response variables.
        function_dist : torch.distributions.Distribution
            Latent GP posterior at the observed locations.

        Returns
        -------
        torch.Tensor with shape ``(*B, N)``
            The expected log probability of the observed data under the ALD likelihood.
        """
        # res: (*B, N).
        # super().expected_log_prob internally uses self.forward() to convert
        # GP posterior to ALD, then computes the log probability of observations.
        # Thus, subclass can just implement forward().
        res = super().expected_log_prob(observations, function_dist, *args, **kwargs)

        num_event_dim = len(function_dist.event_shape)
        if num_event_dim > 1:
            res = res.sum(list(range(-1, -num_event_dim, -1)))
        return res


class AsymmetricLaplaceLikelihood(_ALDLikelihoodBase):
    r"""Likelihood with homoscedastic asymmetric Laplace distribution.

    Parameters
    ----------
    kappa : torch.Tensor
        The asymmetry parameters of the distribution.
    noise_prior
        Prior for noise parameter :math:`\sigma^2`.
    noise_constraint
        Constraint for noise parameter :math:`\sigma^2`.
    batch_shape: torch.Size, default=torch.Size()
        The batch shape of the learned noise parameter.
    """

    def __init__(
        self,
        kappa,
        noise_prior=None,
        noise_constraint=None,
        batch_shape=torch.Size(),
        **kwargs,
    ):
        noise_covar = HomoskedasticNoise(
            noise_prior=noise_prior,
            noise_constraint=noise_constraint,
            batch_shape=batch_shape,
        )
        super().__init__(kappa, noise_covar=noise_covar)

    def forward(self, function_samples, *params, **kwargs):
        r"""Return the ALD distribution conditional on latent function samples.

        ``noise`` follows :class:`gpytorch.likelihoods.GaussianLikelihood`'s
        convention and represents a variance.  The ALD scale parameter is
        therefore :math:`\lambda = \sqrt{\sigma^2}`.

        Parameters
        ----------
        function_samples : torch.Tensor
            Samples of the latent function, with shape ``(*B, N)``.

        Returns
        -------
        ALD
            An asymmetric Laplace distribution centred at ``function_samples``.
        """
        noise = self._shaped_noise_covar(
            function_samples.shape, *params, **kwargs
        ).diagonal(dim1=-1, dim2=-2)
        return ALD(
            m=function_samples,
            lamda=noise.sqrt(),
            kappa=self.kappa.unsqueeze(-1),
        )


# Old


class _QuantileLikelihoodMixin:
    """Asymmetric Laplace distribution likelihood for Bayesian quantile regression.

    Parameters
    ----------
    kappa : torch.Tensor with shape ``(*B)``
        The asymmetry parameters of the distribution.
    raw_scales : torch.Tensor with shape ``(*B)`` or scalar, default=0
        The initial untransformed scales of the asymmetric Laplace distribution.
        The actual scales are obtained by applying the positive transformation.
        If tensor, dimension should be same to *kappa* and shape should be
        broadcastable.
        Scalar value is repeated to the shape of *kappa*.
    learn_scales : bool, default=True
        Whether to update scales by gradients.

    Notes
    -----
    Whether to broadcast ``raw_scales`` is important when ``learn_scales=True``.
    When the scale is broadcasted, the same scale parameter is shared across and updated
    along the broadcasted dimension, e.g., across different asymmetry parameters or
    batches.

    Sharing scales across asymmetry parameters may be deliberately used to reduce the
    number of parameters and increase the stability of training.
    On the other hand, sharing scales across batches usually does not make sense and
    should be avoided.
    In general, it is recommended to use independent scale parameters for all channels.

    To encourage the use of independent scale parameters, scalar ``raw_scales`` is
    repeated to the shape of *kappa* instead of being broadcasted.
    For example, if *kappa* has shape ``(B1, B2)`` and ``raw_scales`` is ``Tensor(1)``
    whose shape is ``()``, then it is converted to a tensor of shape ``(B1, B2)``
    where all values are 1.
    On the other hand, if ``raw_scales`` is ``Tensor([[1]])`` whose shape is ``(1, 1)``,
    then it is broadcasted to shape ``(B1, B2)`` and shared across all batches.
    Likewise, you can pass tensor of shape either ``(B1, 1)`` or ``(1, B2)`` to share
    scales across a specific batch.
    """

    def __init__(self, kappa, raw_scales=0.0, learn_scales=True):
        super().__init__()
        self.register_buffer("kappa", kappa.float())

        raw_scales = torch.as_tensor(raw_scales, dtype=torch.float32)
        if raw_scales.ndim == 0:
            raw_scales = torch.full_like(kappa, raw_scales)
        if learn_scales:
            self.register_parameter("raw_scales", torch.nn.Parameter(raw_scales))
        else:
            self.register_buffer("raw_scales", raw_scales)
        self.register_constraint("raw_scales", gpytorch.constraints.Positive())

    @property
    def scales(self):
        return self.raw_scales_constraint.transform(self.raw_scales)

    def expected_log_prob(self, observations, function_dist, *args, **kwargs):
        """Expected log probability of the observed data under the ALD likelihood.

        Parameters
        ----------
        observations : torch.Tensor with shape ``(*B, N)``
            The observed response variables.
        function_dist : torch.distributions.Distribution
            Latent GP posterior at the observed locations.

        Returns
        -------
        torch.Tensor with shape ``(*B, N)``
            The expected log probability of the observed data under the ALD likelihood.
        """
        # res: (*B, N, *T).
        # super().expected_log_prob internally uses self.forward() to convert
        # GP posterior to ALD, then computes the log probability of observations.
        # Thus, subclass can just implement forward().
        observations = observations.unsqueeze(-1)  # (*B, N, 1)
        res = super().expected_log_prob(observations, function_dist, *args, **kwargs)

        num_event_dim = len(function_dist.event_shape)
        if num_event_dim > 1:
            res = res.sum(list(range(-1, -num_event_dim, -1)))
        return res

    def predictive_posterior(self, gp_posterior):
        """Sample from predictive posterior distribution of function values.

        Parameters
        ----------
        gp_posterior : gpytorch.distributions.MultivariateNormal
            Latent GP posterior at input locations.

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
                pp_sample = likelihood.predictive_posterior(gp(x_pred))
            ci_lower = pp_sample.quantile(0.025, dim=0)
            ci_upper = pp_sample.quantile(0.975, dim=0)
        """
        ald = self(gp_posterior)
        u = torch.rand_like(ald.m)
        return ald.icdf(u)


class DirectQuantileLikelihood(_QuantileLikelihoodMixin, Likelihood):
    """Likelihood for single-output multi-quantile GPQR with direct representation.

    Parameters
    ----------
    kappa : torch.Tensor with shape ``(*B, Q)``
        The quantile levels.
    raw_scales : torch.Tensor with shape ``(*B, Q)`` or scalar, default=0
        The initial untransformed scales of the asymmetric Laplace distribution.
    learn_scales

    Attributes
    ----------
    kappa : torch.Tensor with shape ``(*B, Q)``
    raw_scales : torch.Tensor with shape ``(*B, Q)``

    Notes
    -----
    The task dimension of the input GP posterior should be structured as

    .. code-block:: text

        [q_1, q_2, ..., q_Q]

    where ``q_i`` is the *i*-th quantile function.

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
    """

    def forward(self, function_samples):
        """Return the ALD distribution for the given function samples.

        Parameters
        ----------
        function_samples : torch.Tensor with shape ``(S, *B, N, Q)``
            The function samples drawn from the GP posterior distribution.
            *S* is the number of samples, *Q* is the number of tasks,
            *B* is the batch shape, and *N* is the number of data points.

        Returns
        -------
        QuantileALD
        """
        return QuantileALD(
            m=function_samples,
            lamda=self.scales.unsqueeze(-2),  # (*B, 1, Q)
            kappa=self.kappa.unsqueeze(-2),  # (*B, 1, Q)
        )


class CenterGapQuantileLikelihood(_QuantileLikelihoodMixin, Likelihood):
    """Likelihood for single-output multi-quantile GPQR with center-gap representation.

    Parameters
    ----------
    kappa : torch.Tensor with shape ``(*B, Q)``
        The quantile levels.
    central_quantile_index : int
        The index of the central quantile in the quantile levels.
    raw_scales : torch.Tensor with shape ``(*B, Q)`` or scalar, default=0
        The initial untransformed scales of the asymmetric Laplace distribution.
    learn_scales

    Attributes
    ----------
    kappa : torch.Tensor with shape ``(*B, Q)``
    raw_scales : torch.Tensor with shape ``(*B, Q)``

    Notes
    -----
    The task dimension of the input GP posterior should be structured as

    .. code-block:: text

        [c, *L, *U]

    where ``c`` is the central quantile,
    ``L`` contains the pre-softplus-transformed lower gaps, and
    ``U`` contains the pre-softplus-transformed upper gaps.

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
    >>> from gpytorch_qr.models import CenterGapQuantileGP
    >>> from gpytorch_qr.likelihoods import CenterGapQuantileLikelihood
    >>> from gpytorch_qr.variational import CenterGapLMCVariationalStrategy
    >>> class MyGP(CenterGapQuantileGP):
    ...     def __init__(
    ...         self,
    ...         inducing_points,
    ...         num_q,
    ...         num_lower_q,
    ...         num_latents,
    ...     ):
    ...         N, D = inducing_points.size()
    ...         variational_distribution = CholeskyVariationalDistribution(
    ...             N,
    ...             batch_shape=torch.Size([num_latents]),
    ...         )
    ...         var_strat = CenterGapLMCVariationalStrategy(
    ...             VariationalStrategy(
    ...                 self,
    ...                 inducing_points,
    ...                 variational_distribution,
    ...                 learn_inducing_locations=True,
    ...             ),
    ...             num_q,
    ...             num_latents,
    ...             num_quantiles=[num_q],
    ...             num_lower_quantiles=[num_lower_q],
    ...         )
    ...         mean = ConstantMean(batch_shape=torch.Size([num_latents]))
    ...         covar = ScaleKernel(
    ...             RBFKernel(ard_num_dims=D, batch_shape=torch.Size([num_latents])),
    ...             batch_shape=torch.Size([num_latents]),
    ...         )
    ...         super().__init__(var_strat, mean, covar, [num_q], [num_lower_q])
    >>> inducing_pts = torch.linspace(0, 1, 10).reshape(-1, 1)
    >>> central_q_index = (q - 0.5).abs().argmin().item()
    >>> num_latents = len(q) - 2  # recommended to be smaller than q
    >>> gp = MyGP(inducing_pts, len(q), central_q_index, num_latents)
    >>> likelihood = CenterGapQuantileLikelihood(q, central_q_index)
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
    """

    def __init__(
        self, kappa, central_quantile_index, raw_scales=0.0, learn_scales=True
    ):
        super().__init__(kappa, raw_scales, learn_scales)
        idx = torch.as_tensor(central_quantile_index).long()
        if idx.dim() == 0:
            idx_for_gather = idx.view(1).expand(
                list(self.kappa.shape[:-1]) + [1]
            )  # (*B, 1)
        else:
            idx_for_gather = idx.unsqueeze(-1)  # (*B, 1)
        idx_for_gather = idx_for_gather.to(self.kappa.device)
        central_quantile = self.kappa.gather(-1, idx_for_gather).squeeze(-1)  # (*B)
        self.lower_count = (self.kappa < central_quantile.unsqueeze(-1)).sum(
            dim=-1
        )  # (*B)

    def forward(self, function_samples):
        """Return the ALD distribution for the given function samples.

        Parameters
        ----------
        function_samples : torch.Tensor with shape ``(S, *B, N, Q)``
            The function samples drawn from the GP posterior distribution.
            *S* is the number of samples, *Q* is the number of tasks,
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
            quantiles = centergap_to_quantiles(center, lower_gaps, upper_gaps)
        else:
            # Derive actual batch shape from function_samples, not from lc,
            # because lc may have been computed from a broadcasted kappa.
            S = function_samples.shape[0]
            N = function_samples.shape[-2]
            Q = function_samples.shape[-1]
            B_shape = function_samples.shape[1:-2]  # actual (*B)
            B_flat = 1
            for d in B_shape:
                B_flat *= d
            # Flatten *B: (S, B_flat, N, Q)
            fs_flat = function_samples.reshape(S, B_flat, N, Q)
            lc_flat = lc.reshape(-1).expand(B_flat)  # broadcast lc to (B_flat,)
            quantiles_flat = torch.empty_like(fs_flat)
            for unique_lc in lc_flat.unique():
                lc_val = int(unique_lc)
                mask = lc_flat == unique_lc
                fs_group = fs_flat[:, mask, :, :]  # (S, G, N, Q)
                center = fs_group[..., :1]
                lower_gaps = fs_group[..., 1 : 1 + lc_val]
                upper_gaps = fs_group[..., 1 + lc_val :]
                quantiles_flat[:, mask, :, :] = centergap_to_quantiles(
                    center, lower_gaps, upper_gaps
                )
            quantiles = quantiles_flat.reshape(S, *B_shape, N, Q)
        return QuantileALD(
            m=quantiles,
            lamda=self.scales.unsqueeze(-2),  # (*B, 1, Q)
            kappa=self.kappa.unsqueeze(-2),  # (*B, 1, Q)
        )


class _MultiOutputQuantileLikelihoodMixin:
    """Likelihood for multi-output multi-quantile GPQR."""

    def __init__(self, *likelihoods):
        super().__init__()
        self.likelihoods = torch.nn.ModuleList(likelihoods)
        self.num_outputs = len(likelihoods)
        self.num_quantiles = [likelihood.kappa.shape[-1] for likelihood in likelihoods]

    def expected_log_prob(self, observations, function_dist, *args, **kwargs):
        """Expected log probability of the observed data under the ALD likelihood.

        Parameters
        ----------
        observations : torch.Tensor with shape ``(*B, N, *T)``
            The observed response variables.
        function_dist : torch.distributions.Distribution
            Latent GP posterior at the observed locations.

        Returns
        -------
        torch.Tensor with shape ``(*B, N)``
            The expected log probability of the observed data under the ALD likelihood.
        """
        rep_observations = []
        for i in range(self.num_outputs):
            num_q = self.num_quantiles[i]
            obs = observations[..., i : i + 1]
            rep_observations.append(
                obs.repeat(*([1 for _ in range(len(obs.shape) - 1)] + [num_q]))
            )
        observations = torch.cat(rep_observations, dim=-1)
        ret = super().expected_log_prob(observations, function_dist, *args, **kwargs)
        return ret.sum(dim=-1)


class MultiOutputDirectQuantileLikelihood(
    _MultiOutputQuantileLikelihoodMixin,
    Likelihood,
):
    """Likelihood for multi-output multi-quantile direct GPQR.

    Parameters
    ----------
    *likelihoods : list of DirectQuantileLikelihood

    Notes
    -----
    The task dimension of the input GP posterior should be structured as

    .. code-block:: text

        [*Q_1, *Q_2, ..., *Q_k]

    where ``Q_i`` contains quantiles for the i-th output dimension.
    """

    def forward(self, function_samples):
        """Return the ALD distribution for the given function samples.

        Parameters
        ----------
        function_samples : torch.Tensor with shape ``(S, *B, N, Q)``
            The function samples drawn from the GP posterior distribution.
            *S* is the number of samples, *Q* is the number of tasks,
            *B* is the batch shape, and *N* is the number of data points.

        Returns
        -------
        QuantileALD
        """
        alds = []
        idx = 0
        for i in range(self.num_outputs):
            likelihood = self.likelihoods[i]
            num_q = self.num_quantiles[i]
            fs = function_samples[..., idx : idx + num_q]
            alds.append(likelihood(fs))
            idx += num_q

        m = torch.cat([ald.m for ald in alds], dim=-1)
        lamda = torch.cat([ald.lamda.squeeze(0) for ald in alds], dim=-1)
        kappa = torch.cat([ald.kappa.squeeze(0) for ald in alds], dim=-1)
        return QuantileALD(m=m, lamda=lamda, kappa=kappa)


class MultiOutputCenterGapQuantileLikelihood(
    _MultiOutputQuantileLikelihoodMixin,
    Likelihood,
):
    """Likelihood for multi-output multi-quantile center-gap GPQR.

    Parameters
    ----------
    *likelihoods : list of CenterGapQuantileLikelihood

    Notes
    -----
    The task dimension of the input GP posterior should be structured as

    .. code-block:: text

        [c_1, c_2, ..., c_k,  *L_1, *U_1,  *L_2, *U_2,  ...,  *L_k, *U_k]

    where ``c_i`` is the central quantile for the i-th output dimension,
    ``L_i`` contains the pre-softplus-transformed lower gaps,
    and ``U_i`` contains the pre-softplus-transformed upper gaps.
    """

    def forward(self, function_samples):
        """Return the ALD distribution for the given function samples.

        Parameters
        ----------
        function_samples : torch.Tensor with shape ``(S, *B, N, Q)``
            The function samples drawn from the GP posterior distribution.
            *S* is the number of samples, *Q* is the number of tasks,
            *B* is the batch shape, and *N* is the number of data points.

        Returns
        -------
        QuantileALD
        """
        alds = []
        gap_idx = self.num_outputs
        for i in range(self.num_outputs):
            likelihood = self.likelihoods[i]
            num_q = self.num_quantiles[i]
            num_gaps = num_q - 1

            center = function_samples[..., i : i + 1]
            gaps = function_samples[..., gap_idx : gap_idx + num_gaps]
            fs = torch.cat([center, gaps], dim=-1)
            alds.append(likelihood(fs))
            gap_idx += num_gaps

        m = torch.cat([ald.m for ald in alds], dim=-1)
        lamda = torch.cat([ald.lamda.squeeze(0) for ald in alds], dim=-1)
        kappa = torch.cat([ald.kappa.squeeze(0) for ald in alds], dim=-1)
        return QuantileALD(m=m, lamda=lamda, kappa=kappa)
