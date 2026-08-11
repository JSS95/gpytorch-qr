"""Asymmetric Laplace distributions likelihoods for quantile regression."""

import gpytorch
import torch
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import Likelihood
from gpytorch.likelihoods.noise_models import HomoskedasticNoise
from linear_operator import to_linear_operator
from linear_operator.operators import (
    ConstantDiagLinearOperator,
    DiagLinearOperator,
    KroneckerProductDiagLinearOperator,
    KroneckerProductLinearOperator,
    RootLinearOperator,
)
from torch.distributions import Independent

from .distributions import AsymmetricLaplace, QuantileALD
from .utils import centergap_to_quantiles

__all__ = [
    "AsymmetricLaplaceLikelihood",
    "MultitaskAsymmetricLaplaceLikelihood",
    "DirectQuantilesLikelihood",
    "MultioutputDirectQuantilesLikelihood",
    "CenterGapQuantilesLikelihood",
    "CenterGapQuantileLikelihood",
    "MultiOutputCenterGapQuantileLikelihood",
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

        ``noise`` follows :class:`gpytorch.likelihoods.GaussianLikelihood`'s
        convention and represents a variance.  The ALD scale parameter is
        therefore :math:`\lambda = \sqrt{\sigma^2}`.

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
        return AsymmetricLaplace(
            loc=function_samples,
            scale=noise.sqrt(),
            asymmetry=self.kappa.unsqueeze(-1),
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
        Rank of the task noise covariance matrix.
    task_correlation_prior
        Prior over the task noise correlation matrix. Only used when
        ``rank > 0``.
    batch_shape : torch.Size, default=torch.Size()
        Batch shape of the task correlation parameters.
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
            if rank > num_tasks:
                raise ValueError(
                    f"Cannot have rank ({rank}) greater than num_tasks "
                    f"({num_tasks})"
                )
            tidcs = torch.tril_indices(num_tasks, rank, dtype=torch.long)
            # (1, 1) must be 1.0, so it does not need parameterization.
            self.tidcs = tidcs[:, 1:]
            task_noise_corr = torch.randn(*batch_shape, self.tidcs.size(-1))
            self.register_parameter(
                "task_noise_corr", torch.nn.Parameter(task_noise_corr)
            )
            if task_correlation_prior is not None:
                self.register_prior(
                    "MultitaskErrorCorrelationPrior",
                    task_correlation_prior,
                    lambda module: module._eval_corr_matrix(),
                )
        elif task_correlation_prior is not None:
            raise ValueError("Can only specify task_correlation_prior if rank>0")

        self.num_tasks = num_tasks
        self.rank = rank

    def _eval_corr_matrix(self):
        task_noise_corr = self.task_noise_corr
        factor_diag = torch.ones(
            *task_noise_corr.shape[:-1],
            self.num_tasks,
            device=task_noise_corr.device,
            dtype=task_noise_corr.dtype,
        )
        correlation_factor = torch.diag_embed(factor_diag)
        correlation_factor[..., self.tidcs[0], self.tidcs[1]] = task_noise_corr
        # Squared rows must sum to one for this to be a correlation matrix.
        correlation_factor = (
            correlation_factor
            / correlation_factor.pow(2).sum(dim=-1, keepdim=True).sqrt()
        )
        return correlation_factor @ correlation_factor.transpose(-1, -2)

    def _shaped_noise_covar(
        self, shape, add_noise=True, interleaved=True, *params, **kwargs
    ):
        if not self.has_task_noise:
            return ConstantDiagLinearOperator(
                self.noise, diag_shape=shape[-2] * self.num_tasks
            )

        if self.rank == 0:
            task_noises = self.task_noises
            task_variance = DiagLinearOperator(task_noises)
            dtype = task_noises.dtype
            device = task_noises.device
            kron_type = KroneckerProductDiagLinearOperator
        else:
            factor = self.task_noise_covar_factor
            task_variance = RootLinearOperator(factor)
            dtype = factor.dtype
            device = factor.device
            kron_type = KroneckerProductLinearOperator

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
            return kron_type(identity, task_variance)
        return kron_type(task_variance, identity)

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
        base_distribution = AsymmetricLaplace(
            loc=function_samples,
            scale=noise.sqrt(),
            asymmetry=self.kappa.unsqueeze(-2),  # (1, T)
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
        Rank of the task noise covariance.  ``0`` fits independent task
        variances.
    batch_shape : torch.Size, default=torch.Size()
        Batch shape of the learned noise parameters.
    task_prior
        Prior over the task noise covariance when ``rank > 0``.
    noise_prior
        Prior over the global or diagonal task noise variances.
    noise_constraint
        Constraint for noise variances.  Defaults to ``GreaterThan(1e-4)``.
    has_global_noise : bool, default=True
        Whether to include the shared :math:`\sigma^2` variance.
    has_task_noise : bool, default=True
        Whether to include task-specific noise variances.

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

        if not has_task_noise and not has_global_noise:
            raise ValueError(
                "At least one of has_task_noise or has_global_noise must be "
                "specified. Attempting to specify a likelihood that has no "
                "noise terms."
            )
        if noise_constraint is None:
            noise_constraint = GreaterThan(1e-4)

        if has_task_noise:
            if rank == 0:
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
                    raise RuntimeError("Cannot set a task_prior if rank=0")
            else:
                self.register_parameter(
                    "task_noise_covar_factor",
                    torch.nn.Parameter(torch.randn(*batch_shape, num_tasks, rank)),
                )
                if task_prior is not None:
                    self.register_prior(
                        "MultitaskErrorCovariancePrior",
                        task_prior,
                        lambda module: module._eval_covar_matrix(),
                    )

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
        """The global noise variance."""
        return self.raw_noise_constraint.transform(self.raw_noise)

    @noise.setter
    def noise(self, value):
        self._set_noise(value)

    @property
    def task_noises(self):
        """The diagonal task noise variances when ``rank=0``."""
        if self.rank == 0:
            return self.raw_task_noises_constraint.transform(self.raw_task_noises)
        raise AttributeError(
            "Cannot set diagonal task noises when covariance has ",
            self.rank,
            ">0",
        )

    @task_noises.setter
    def task_noises(self, value):
        if self.rank == 0:
            self._set_task_noises(value)
            return
        raise AttributeError(
            "Cannot set diagonal task noises when covariance has ",
            self.rank,
            ">0",
        )

    def _set_noise(self, value):
        self.initialize(raw_noise=self.raw_noise_constraint.inverse_transform(value))

    def _set_task_noises(self, value):
        self.initialize(
            raw_task_noises=self.raw_task_noises_constraint.inverse_transform(value)
        )

    @property
    def task_noise_covar(self):
        """The low-rank task noise covariance when ``rank>0``."""
        if self.rank > 0:
            factor = self.task_noise_covar_factor
            return factor.matmul(factor.transpose(-1, -2))
        raise AttributeError("Cannot retrieve task noises when covariance is diagonal.")

    @task_noise_covar.setter
    def task_noise_covar(self, value):
        if self.rank > 0:
            with torch.no_grad():
                factor = to_linear_operator(value).pivoted_cholesky(rank=self.rank)
                self.task_noise_covar_factor.copy_(factor)
            return
        raise AttributeError(
            "Cannot set non-diagonal task noises when covariance is diagonal."
        )

    def _eval_covar_matrix(self):
        covariance_factor = self.task_noise_covar_factor
        noise = self.noise
        identity = torch.eye(
            self.num_tasks,
            dtype=noise.dtype,
            device=noise.device,
        )
        return (
            covariance_factor.matmul(covariance_factor.transpose(-1, -2))
            + noise.unsqueeze(-1) * identity
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
        self.register_buffer("lower_counts", torch.cat(lower_counts, dim=-1))

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
        for samples, lc in zip(each_samples, self.lower_counts):
            q = self._convert_to_quantiles(samples, lc)
            quantiles.append(q)
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
