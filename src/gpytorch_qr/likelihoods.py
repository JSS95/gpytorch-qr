"""Asymmetric Laplace distributions likelihoods for quantile regression."""

import torch
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import Likelihood
from gpytorch.likelihoods.noise_models import HomoskedasticNoise
from linear_operator.operators import (
    ConstantDiagLinearOperator,
    DiagLinearOperator,
    KroneckerProductDiagLinearOperator,
)
from torch.distributions import Independent

from .distributions import AsymmetricLaplace
from .settings import quantile_gap_lower_bound
from .utils import centergap_to_quantiles

__all__ = [
    "AsymmetricLaplaceLikelihood",
    "MultitaskAsymmetricLaplaceLikelihood",
    "DirectQuantilesLikelihood",
    "MultioutputDirectQuantilesLikelihood",
    "CenterGapQuantilesLikelihood",
    "MultioutputCenterGapQuantilesLikelihood",
]


# ALD likelihood compatible with GPyTorch's Gaussian likelihood interface


class _ALDLikelihoodBase(Likelihood):
    """Base class for ALD likelihoods for quantile regression."""

    has_analytic_marginal = False

    def __init__(self, kappa, noise_covar):
        super().__init__()
        self.register_buffer(
            "kappa", torch.as_tensor(kappa, dtype=torch.get_default_dtype())
        )
        self.noise_covar = noise_covar

    def _shaped_noise_covar(self, base_shape, *params, **kwargs):
        return self.noise_covar(*params, shape=base_shape, **kwargs)

    @staticmethod
    def _noise_to_rate(noise, asymmetry):
        r"""Convert squared quantile-loss scale to the ALD rate parameter.

        ``noise`` follows GPyTorch's variance-like convention and stores
        :math:`\lambda^2`, where :math:`\lambda` is the scale in the
        quantile-parameterized ALD. :class:`AsymmetricLaplace` instead uses
        the rate :math:`L`, so

        .. math::

            L = \frac{\kappa}{(1 + \kappa^2)\lambda}.
        """
        return asymmetry / ((1 + asymmetry.square()) * noise.sqrt())

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

    def forward(self, function_samples, *params, **kwargs):
        r"""Return the ALD distribution conditional on latent function samples.

        Parameters
        ----------
        function_samples : torch.Tensor
            Samples of the latent function, with shape ``(*B, N)``.

        Returns
        -------
        AsymmetricLaplace
            An asymmetric Laplace distribution centred at ``function_samples``.
        """
        noise = self._shaped_noise_covar(
            function_samples.shape, *params, **kwargs
        ).diagonal(dim1=-1, dim2=-2)
        asymmetry = self.kappa.unsqueeze(-1)
        return AsymmetricLaplace(
            loc=function_samples,
            scale=self._noise_to_rate(noise, asymmetry),
            asymmetry=asymmetry,
        )


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

    See Also
    --------
    gpytorch.likelihoods.GaussianLikelihood
        Gaussian distribution equivalent of this likelihood.
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

    @property
    def noise(self):
        r"""The squared ALD quantile-loss scale :math:`\lambda^2`."""
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


class _MultitaskALDLikelihoodBase(_ALDLikelihoodBase):
    r"""Base class for multitask asymmetric Laplace likelihoods.

    Parameters
    ----------
    kappa : torch.Tensor
        Asymmetry parameter of the asymmetric Laplace distribution.
    num_tasks : int
        Number of tasks.
    noise_covar : gpytorch.module.Module
        Model for the noise covariance.
    rank : int, default=0
        Rank of the task noise covariance. Only ``0`` is supported.
    task_correlation_prior
        Prior over the task noise correlation matrix.
        Only used when ``rank > 0``.
    batch_shape : torch.Size, default=torch.Size()
        Batch shape of the learned noise parameters.
    """

    @staticmethod
    def _prepare_kappa(kappa, num_tasks):
        if num_tasks < 1:
            raise ValueError("num_tasks must be positive")
        kappa = torch.as_tensor(kappa, dtype=torch.get_default_dtype())
        if kappa.ndim == 0:
            kappa = kappa.expand(num_tasks)
        elif kappa.shape[-1] not in (1, num_tasks):
            raise ValueError(
                "The trailing dimension of kappa must be 1 or num_tasks "
                f"({num_tasks}), got {kappa.shape[-1]}"
            )
        return kappa

    def __init__(
        self,
        kappa,
        num_tasks,
        noise_covar,
        rank=0,
        task_correlation_prior=None,
        batch_shape=torch.Size(),
    ):
        kappa = self._prepare_kappa(kappa, num_tasks)
        super().__init__(kappa=kappa, noise_covar=noise_covar)
        if rank != 0:
            raise NotImplementedError(
                "Correlated asymmetric Laplace task noise is not implemented; "
                "rank must be 0."
            )
        if task_correlation_prior is not None:
            raise ValueError(
                "task_correlation_prior is unsupported because rank must be 0"
            )

        self.num_tasks = num_tasks
        self.rank = rank

    def _shaped_noise_covar(
        self, shape, add_noise=True, interleaved=True, *params, **kwargs
    ):
        if not self.has_task_noise:
            return ConstantDiagLinearOperator(
                self.noise, diag_shape=shape[-2] * self.num_tasks
            )

        task_noises = self.task_noises
        task_variance = DiagLinearOperator(task_noises)
        dtype = task_noises.dtype
        device = task_noises.device

        identity = ConstantDiagLinearOperator(
            torch.ones(*shape[:-2], 1, dtype=dtype, device=device),
            diag_shape=shape[-2],
        )
        task_variance = task_variance.expand(*shape[:-2], *task_variance.matrix_shape)

        if add_noise and self.has_global_noise:
            global_noise = ConstantDiagLinearOperator(
                self.noise, diag_shape=task_variance.shape[-1]
            )
            task_variance = task_variance + global_noise

        if interleaved:
            return KroneckerProductDiagLinearOperator(identity, task_variance)
        return KroneckerProductDiagLinearOperator(task_variance, identity)

    def forward(self, function_samples, *params, **kwargs):
        r"""Return the conditional multitask ALD distribution.

        Parameters
        ----------
        function_samples : torch.Tensor with shape ``(*B, N, T)``
            Samples from the multitask latent function.

        Returns
        -------
        torch.distributions.Independent
            Independent ALD marginals with the final task dimension treated
            as one event.
        """
        if function_samples.shape[-1] != self.num_tasks:
            raise ValueError(
                "The trailing dimension of function_samples must equal "
                f"num_tasks ({self.num_tasks}), got {function_samples.shape[-1]}"
            )
        noise = self._shaped_noise_covar(
            function_samples.shape, *params, **kwargs
        ).diagonal(dim1=-1, dim2=-2)
        noise = noise.reshape(*noise.shape[:-1], *function_samples.shape[-2:])
        asymmetry = self.kappa.unsqueeze(-2)
        base_distribution = AsymmetricLaplace(
            loc=function_samples,
            scale=self._noise_to_rate(noise, asymmetry),
            asymmetry=asymmetry,  # (1, T)
        )  # batch_shape: (*B, N, T), event_shape: ()
        return Independent(base_distribution, 1)  # batch: (*B, N), event: (T,)

    def expected_log_prob(self, observations, function_dist, *args, **kwargs):
        # forward() already treats the task axis as an event, so GPyTorch's
        # generic implementation returns one log likelihood per data point.
        return Likelihood.expected_log_prob(
            self, observations, function_dist, *args, **kwargs
        )


class MultitaskAsymmetricLaplaceLikelihood(_MultitaskALDLikelihoodBase):
    r"""Homoscedastic asymmetric Laplace likelihood for multitask models.

    Parameters
    ----------
    kappa : torch.Tensor
        Asymmetry parameter of the asymmetric Laplace distribution.
    num_tasks : int
        Number of tasks.
    rank : int, default=0
        Rank of the task noise covariance. Only ``0`` is supported.
    batch_shape : torch.Size, default=torch.Size()
        Batch shape of the learned noise parameters.
    task_prior
        Prior over the task noise covariance when ``rank > 0``.
    noise_prior
        Prior over the global or task-specific squared ALD scales.
    noise_constraint
        Constraint for squared ALD scales. Defaults to ``GreaterThan(1e-4)``.
    has_global_noise : bool, default=True
        Whether to include a shared squared ALD scale.
    has_task_noise : bool, default=True
        Whether to include task-specific squared ALD scales.

    See Also
    --------
    gpytorch.likelihoods.MultitaskGaussianLikelihood
        Multitask Gaussian distribution equivalent of this likelihood.
    """

    def __init__(
        self,
        kappa,
        num_tasks,
        rank=0,
        batch_shape=torch.Size(),
        task_prior=None,
        noise_prior=None,
        noise_constraint=None,
        has_global_noise=True,
        has_task_noise=True,
    ):
        # Match MultitaskGaussianLikelihood: this concrete class owns its
        # homoskedastic noise parameters and bypasses the general base
        # constructor, which accepts an externally supplied noise model.
        super(Likelihood, self).__init__()
        kappa = self._prepare_kappa(kappa, num_tasks)
        self.register_buffer("kappa", kappa)

        if rank != 0:
            raise NotImplementedError(
                "Correlated asymmetric Laplace task noise is not implemented; "
                "rank must be 0."
            )

        if not has_task_noise and not has_global_noise:
            raise ValueError(
                "At least one of has_task_noise or has_global_noise must be "
                "specified. Attempting to specify a likelihood that has no "
                "noise terms."
            )
        if noise_constraint is None:
            noise_constraint = GreaterThan(1e-4)

        if has_task_noise:
            self.register_parameter(
                "raw_task_noises",
                torch.nn.Parameter(torch.zeros(*batch_shape, num_tasks)),
            )
            self.register_constraint("raw_task_noises", noise_constraint)
            if noise_prior is not None:
                self.register_prior(
                    "raw_task_noises_prior",
                    noise_prior,
                    lambda module: module.task_noises,
                )
        if task_prior is not None:
            raise ValueError("task_prior is unsupported because rank must be 0")

        self.num_tasks = num_tasks
        self.rank = rank

        if has_global_noise:
            self.register_parameter(
                "raw_noise", torch.nn.Parameter(torch.zeros(*batch_shape, 1))
            )
            self.register_constraint("raw_noise", noise_constraint)
            if noise_prior is not None:
                self.register_prior(
                    "raw_noise_prior",
                    noise_prior,
                    lambda module: module.noise,
                )

        self.has_global_noise = has_global_noise
        self.has_task_noise = has_task_noise

    @property
    def noise(self):
        """The shared squared ALD scale."""
        return self.raw_noise_constraint.transform(self.raw_noise)

    @noise.setter
    def noise(self, value):
        self._set_noise(value)

    @property
    def task_noises(self):
        """The independent task-specific squared ALD scales."""
        return self.raw_task_noises_constraint.transform(self.raw_task_noises)

    @task_noises.setter
    def task_noises(self, value):
        self._set_task_noises(value)

    def _set_noise(self, value):
        self.initialize(raw_noise=self.raw_noise_constraint.inverse_transform(value))

    def _set_task_noises(self, value):
        self.initialize(
            raw_task_noises=self.raw_task_noises_constraint.inverse_transform(value)
        )


# Special likelihoods for quantile regression


class _DirectQuantilesLikelihoodBase(MultitaskAsymmetricLaplaceLikelihood):
    """Likelihood for GPQR with direct quantile representation.

    Parameters
    ----------
    quantile_levels : list of torch.Tensor
        Tensors whose last dimension corresponds to quantile levels.
        Each tensor in the list corresponds to quantile levels in each output dimension.
    rank
    batch_shape
    task_prior
    noise_prior
    noise_constraint
    has_global_noise
    has_task_noise
    """

    def __init__(
        self,
        quantile_levels,
        rank=0,
        batch_shape=torch.Size(),
        task_prior=None,
        noise_prior=None,
        noise_constraint=None,
        has_global_noise=True,
        has_task_noise=True,
    ):
        quantile_levels_tensor = torch.cat(quantile_levels, dim=-1)
        kappa = (quantile_levels_tensor / (1 - quantile_levels_tensor)).sqrt()
        num_tasks = quantile_levels_tensor.shape[-1]
        super().__init__(
            kappa=kappa,
            num_tasks=num_tasks,
            rank=rank,
            batch_shape=batch_shape,
            task_prior=task_prior,
            noise_prior=noise_prior,
            noise_constraint=noise_constraint,
            has_global_noise=has_global_noise,
            has_task_noise=has_task_noise,
        )
        self.register_buffer(
            "num_quantiles",
            torch.tensor([q.shape[-1] for q in quantile_levels], dtype=torch.long),
        )


class DirectQuantilesLikelihood(_DirectQuantilesLikelihoodBase):
    """Likelihood for single-output GPQR with direct quantile representation.

    Parameters
    ----------
    quantile_levels : torch.Tensor with shape ``(*B, Q)``
        The quantile levels.
    rank
    batch_shape
    task_prior
    noise_prior
    noise_constraint
    has_global_noise
    has_task_noise
    """

    def __init__(
        self,
        quantile_levels,
        rank=0,
        batch_shape=torch.Size(),
        task_prior=None,
        noise_prior=None,
        noise_constraint=None,
        has_global_noise=True,
        has_task_noise=True,
    ):
        super().__init__(
            [quantile_levels],
            rank,
            batch_shape,
            task_prior,
            noise_prior,
            noise_constraint,
            has_global_noise,
            has_task_noise,
        )

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
        torch.Tensor with shape ``(*B, N, Q)``
            The expected log probability of the observed data under the ALD likelihood.
        """
        likelihood_samples = self._draw_likelihood_samples(
            function_dist, *args, **kwargs
        )  # batch shape: (*B, N), event_shape: (Q,)
        res = likelihood_samples.log_prob(
            observations.unsqueeze(-1), *args, **kwargs
        ).mean(dim=0)
        return res


class MultioutputDirectQuantilesLikelihood(_DirectQuantilesLikelihoodBase):
    """Likelihood for multi-output GPQR with direct quantile representation.

    Parameters
    ----------
    quantile_levels
    rank
    batch_shape
    task_prior
    noise_prior
    noise_constraint
    has_global_noise
    has_task_noise
    """

    def expected_log_prob(self, observations, function_dist, *args, **kwargs):
        """Expected log probability of the observed data under the ALD likelihood.

        Parameters
        ----------
        observations : torch.Tensor with shape ``(*B, N, D)``
            The observed response variables.
        function_dist : torch.distributions.Distribution
            Latent GP posterior at the observed locations.

        Returns
        -------
        torch.Tensor with shape ``(*B, N, Q)``
            The expected log probability of the observed data under the ALD likelihood.

        Notes
        -----
        The last dimension of *observations* is the output dimension, which is not the
        task dimension. The task dimension is the sum of all number of quantiles across
        all output dimensions, i.e., ``T = Q_1 + Q_2 + ... + Q_D``.
        """
        likelihood_samples = self._draw_likelihood_samples(
            function_dist, *args, **kwargs
        )  # batch shape: (*B, N), event_shape: (T,)
        each_observation = []
        for i, n in enumerate(self.num_quantiles):
            each_observation.append(
                observations[..., i]
                .unsqueeze(-1)
                .expand(*[-1 for _ in range(observations.ndim - 1)], n)
            )
        expanded_observations = torch.cat(each_observation, dim=-1)
        res = likelihood_samples.log_prob(expanded_observations, *args, **kwargs).mean(
            dim=0
        )
        return res


class _CenterGapQuantilesLikelihoodBase(MultitaskAsymmetricLaplaceLikelihood):
    """Likelihood for GPQR with center-gap quantile representation.

    Parameters
    ----------
    quantile_levels : list of torch.Tensor
        Tensors whose last dimension corresponds to quantile levels.
        Each tensor in the list corresponds to quantile levels in each output dimension.
    central_quantile_idxs : list of int or torch.Tensor
        The indices of the central quantiles for each quantile level tensor.
        Can have batch shape to apply different central quantile indices for
        different batches.
    rank
    batch_shape
    task_prior
    noise_prior
    noise_constraint
    has_global_noise
    has_task_noise
    """

    def __init__(
        self,
        quantile_levels,
        central_quantile_idxs,
        rank=0,
        batch_shape=torch.Size(),
        task_prior=None,
        noise_prior=None,
        noise_constraint=None,
        has_global_noise=True,
        has_task_noise=True,
    ):
        quantile_levels_tensor = torch.cat(quantile_levels, dim=-1)
        kappa = (quantile_levels_tensor / (1 - quantile_levels_tensor)).sqrt()
        num_tasks = quantile_levels_tensor.shape[-1]
        super().__init__(
            kappa=kappa,
            num_tasks=num_tasks,
            rank=rank,
            batch_shape=batch_shape,
            task_prior=task_prior,
            noise_prior=noise_prior,
            noise_constraint=noise_constraint,
            has_global_noise=has_global_noise,
            has_task_noise=has_task_noise,
        )
        self.register_buffer(
            "num_quantiles",
            torch.tensor([q.shape[-1] for q in quantile_levels], dtype=torch.long),
        )
        lower_counts = []
        quantile_level_offsets = []
        for q, idx in zip(quantile_levels, central_quantile_idxs):
            idx = torch.as_tensor(idx, dtype=torch.long)
            if idx.dim() == 0:
                idx_for_gather = idx.view(1).expand(list(q.shape[:-1]) + [1])  # (*B, 1)
            else:
                idx_for_gather = idx.unsqueeze(-1)  # (*B, 1)
            idx_for_gather = idx_for_gather.to(q.device)
            central_quantile = q.gather(-1, idx_for_gather).squeeze(-1)  # (*B)
            lower_count = (q < central_quantile.unsqueeze(-1)).sum(dim=-1)  # (*B)
            lower_counts.append(lower_count.unsqueeze(-1))
            if (q.diff(dim=-1) <= 0).any():
                raise ValueError("quantile_levels must be strictly increasing.")
            quantile_level_offsets.append(q - central_quantile.unsqueeze(-1))
        self.register_buffer("lower_counts", torch.cat(lower_counts, dim=-1))
        self.register_buffer(
            "quantile_level_offsets",
            torch.cat(quantile_level_offsets, dim=-1),
        )

    def forward(self, function_samples):
        """Return the ALD distribution for the given function samples.

        Parameters
        ----------
        function_samples : torch.Tensor with shape ``(*B, N, T)``
            Samples from the multitask latent function in center-gap representation.

        Notes
        -----
        The task dimension of the input *function_samples* should be structured as

        .. code-block:: text

            [c_1, c_2, ..., c_k,  *L_1, *U_1,  *L_2, *U_2,  ...,  *L_k, *U_k]

        where ``c_i`` is the central quantile for the i-th output dimension,
        ``L_i`` contains the pre-softplus-transformed lower gaps,
        and ``U_i`` contains the pre-softplus-transformed upper gaps.
        """
        # 1. Restructure multi-output *function_samples*.
        each_samples = []  # [c_1, *L_1, *U_1], [c_2, *L_2, *U_2], ...
        gap_idx = len(
            self.num_quantiles
        )  # index of the first gap in the task dimension
        for i in range(len(self.num_quantiles)):
            num_quantiles = self.num_quantiles[i]
            num_gaps = num_quantiles - 1

            center = function_samples[..., i].unsqueeze(-1)
            gaps = function_samples[..., gap_idx : gap_idx + num_gaps]
            each_samples.append(torch.cat([center, gaps], dim=-1))
            gap_idx += num_gaps

        # 2. Convert center-gap function_samples to quantiles
        quantiles = []
        quantile_idx = 0
        for i, samples in enumerate(each_samples):
            lc = self.lower_counts[..., i]
            q = self._convert_to_quantiles(samples, lc)
            num_quantiles = int(self.num_quantiles[i])
            offsets = self.quantile_level_offsets[
                ..., quantile_idx : quantile_idx + num_quantiles
            ].unsqueeze(-2)
            q = q + quantile_gap_lower_bound.value() * offsets
            quantiles.append(q)
            quantile_idx += num_quantiles
        quantile_function_samples = torch.cat(quantiles, dim=-1)  # (*B, N, T)
        return super().forward(quantile_function_samples)

    @staticmethod
    def _convert_to_quantiles(samples, lc):
        if lc.dim() == 0:
            lc_int = int(lc)
            center = samples[..., :1]
            lower_gaps = samples[..., 1 : 1 + lc_int]
            upper_gaps = samples[..., 1 + lc_int :]
            quantiles = centergap_to_quantiles(center, lower_gaps, upper_gaps)
        else:
            # Derive actual batch shape from samples, not from lc,
            # because lc may have been computed from a broadcasted kappa.
            S = samples.shape[0]
            N = samples.shape[-2]
            Q = samples.shape[-1]
            B_shape = samples.shape[1:-2]  # actual (*B)
            B_flat = 1
            for d in B_shape:
                B_flat *= d
            # Flatten *B: (S, B_flat, N, Q)
            fs_flat = samples.reshape(S, B_flat, N, Q)
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
        return quantiles


class CenterGapQuantilesLikelihood(_CenterGapQuantilesLikelihoodBase):
    """Likelihood for single-output GPQR with center-gap quantile representation.

    Parameters
    ----------
    quantile_levels : torch.Tensor with shape ``(*B, Q)``
        The quantile levels.
    central_quantile_idx : int or torch.Tensor with shape ``(*B)``
        The index of the central quantile in the quantile levels.
    rank
    batch_shape
    task_prior
    noise_prior
    noise_constraint
    has_global_noise
    has_task_noise
    """

    def __init__(
        self,
        quantile_levels,
        central_quantile_idx,
        rank=0,
        batch_shape=torch.Size(),
        task_prior=None,
        noise_prior=None,
        noise_constraint=None,
        has_global_noise=True,
        has_task_noise=True,
    ):
        super().__init__(
            [quantile_levels],
            [central_quantile_idx],
            rank,
            batch_shape,
            task_prior,
            noise_prior,
            noise_constraint,
            has_global_noise,
            has_task_noise,
        )

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
        torch.Tensor with shape ``(*B, N, Q)``
            The expected log probability of the observed data under the ALD likelihood.
        """
        likelihood_samples = self._draw_likelihood_samples(
            function_dist, *args, **kwargs
        )  # batch shape: (*B, N), event_shape: (Q,)
        res = likelihood_samples.log_prob(
            observations.unsqueeze(-1), *args, **kwargs
        ).mean(dim=0)
        return res


class MultioutputCenterGapQuantilesLikelihood(_CenterGapQuantilesLikelihoodBase):
    """Likelihood for multi-output GPQR with center-gap quantile representation.

    Parameters
    ----------
    quantile_levels
    central_quantile_idxs
    rank
    batch_shape
    task_prior
    noise_prior
    noise_constraint
    has_global_noise
    has_task_noise
    """

    def expected_log_prob(self, observations, function_dist, *args, **kwargs):
        """Expected log probability of the observed data under the ALD likelihood.

        Parameters
        ----------
        observations : torch.Tensor with shape ``(*B, N, D)``
            The observed response variables.
        function_dist : torch.distributions.Distribution
            Latent GP posterior at the observed locations.

        Returns
        -------
        torch.Tensor with shape ``(*B, N, Q)``
            The expected log probability of the observed data under the ALD likelihood.

        Notes
        -----
        The last dimension of *observations* is the output dimension, which is not the
        task dimension. The task dimension is the sum of all number of quantiles across
        all output dimensions, i.e., ``T = Q_1 + Q_2 + ... + Q_D``.
        """
        likelihood_samples = self._draw_likelihood_samples(
            function_dist, *args, **kwargs
        )  # batch shape: (*B, N), event_shape: (T,)
        each_observation = []
        for i, n in enumerate(self.num_quantiles):
            each_observation.append(
                observations[..., i]
                .unsqueeze(-1)
                .expand(*[-1 for _ in range(observations.ndim - 1)], n)
            )
        expanded_observations = torch.cat(each_observation, dim=-1)
        res = likelihood_samples.log_prob(expanded_observations, *args, **kwargs).mean(
            dim=0
        )
        return res
