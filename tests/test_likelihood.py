import gpytorch
import pytest
import torch
from gpytorch.likelihoods.noise_models import HomoskedasticNoise
from torch.distributions import Independent

from gpytorch_qr.distributions import AsymmetricLaplace
from gpytorch_qr.likelihoods import (
    AsymmetricLaplaceLikelihood,
    CenterGapQuantilesLikelihood,
    MultioutputCenterGapQuantilesLikelihood,
    MultitaskAsymmetricLaplaceLikelihood,
    _MultitaskALDLikelihoodBase,
)
from gpytorch_qr.utils import centergap_to_quantiles

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
S = 4  # MC sample count
N = 10  # data points
B = 3  # batch size

Q = 5
Q_LEVELS = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
CENTRAL_IDX = 2  # 0.5 → lower_count = 2

Q1 = 3
Q1_LEVELS = torch.tensor([0.25, 0.5, 0.75])
CENTRAL_IDX1 = 1  # 0.5 → lower_count = 1

Q2 = 5
Q2_LEVELS = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
CENTRAL_IDX2 = 2  # 0.5 → lower_count = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mtmvn(N, T, batch_shape=torch.Size([])):
    """Create a MultitaskMultivariateNormal with T independent tasks."""
    mvns = [
        gpytorch.distributions.MultivariateNormal(
            torch.zeros(*batch_shape, N),
            torch.eye(N).expand(*batch_shape, N, N),
        )
        for _ in range(T)
    ]
    return gpytorch.distributions.MultitaskMultivariateNormal.from_independent_mvns(
        mvns
    )


# ===========================================================================
# AsymmetricLaplaceLikelihood
# ===========================================================================


class TestAsymmetricLaplaceLikelihood:
    def test_exposes_noise_parameters(self):
        likelihood = AsymmetricLaplaceLikelihood(1.0)
        likelihood.noise = 0.4

        assert torch.allclose(likelihood.noise, torch.tensor([0.4]))
        assert likelihood.raw_noise is likelihood.noise_covar.raw_noise

    def test_forward_converts_squared_scale_to_ald_rate(self):
        likelihood = AsymmetricLaplaceLikelihood(2.0)
        likelihood.noise = 0.25

        output = likelihood(torch.zeros(N))

        assert torch.allclose(output.scale, torch.full((N,), 0.8))


# ===========================================================================
# MultitaskAsymmetricLaplaceLikelihood
# ===========================================================================


class TestMultitaskAsymmetricLaplaceLikelihood:
    def test_inherits_multitask_ald_base(self):
        assert issubclass(
            MultitaskAsymmetricLaplaceLikelihood,
            _MultitaskALDLikelihoodBase,
        )

    def test_base_rejects_correlated_task_noise(self):
        with pytest.raises(NotImplementedError, match="rank must be 0"):
            _MultitaskALDLikelihoodBase(
                Q1_LEVELS,
                Q1,
                HomoskedasticNoise(),
                rank=2,
            )

    def test_base_rejects_task_correlation_prior(self):
        with pytest.raises(ValueError, match="task_correlation_prior is unsupported"):
            _MultitaskALDLikelihoodBase(
                Q1_LEVELS,
                Q1,
                HomoskedasticNoise(),
                task_correlation_prior=gpytorch.priors.NormalPrior(0, 1),
            )

    def test_forward_returns_independent_ald_with_task_event(self):
        likelihood = MultitaskAsymmetricLaplaceLikelihood(Q1_LEVELS, Q1)
        output = likelihood(torch.randn(S, N, Q1))

        assert isinstance(output, Independent)
        assert isinstance(output.base_dist, AsymmetricLaplace)
        assert output.batch_shape == torch.Size([S, N])
        assert output.event_shape == torch.Size([Q1])

    def test_forward_uses_global_and_task_noise(self):
        likelihood = MultitaskAsymmetricLaplaceLikelihood(Q1_LEVELS, Q1)
        likelihood.noise = 0.2
        likelihood.task_noises = torch.tensor([0.1, 0.3, 0.5])

        output = likelihood(torch.zeros(N, Q1))
        noise = torch.tensor([0.3, 0.5, 0.7])
        expected_scale = (Q1_LEVELS / ((1 + Q1_LEVELS.square()) * noise.sqrt())).expand(
            N, Q1
        )

        assert torch.allclose(output.base_dist.scale, expected_scale)
        assert torch.equal(output.base_dist.asymmetry[0], Q1_LEVELS)

    def test_expected_log_prob_sums_tasks_only(self):
        likelihood = MultitaskAsymmetricLaplaceLikelihood(Q1_LEVELS, Q1)
        function_dist = _make_mtmvn(N, Q1)

        with gpytorch.settings.num_likelihood_samples(S):
            result = likelihood.expected_log_prob(torch.randn(N, Q1), function_dist)

        assert result.shape == torch.Size([N])

    def test_batched_noise_parameters_broadcast_over_samples_and_data(self):
        likelihood = MultitaskAsymmetricLaplaceLikelihood(
            Q1_LEVELS,
            Q1,
            batch_shape=torch.Size([B]),
        )
        output = likelihood(torch.randn(S, B, N, Q1))

        assert output.base_dist.scale.shape == torch.Size([S, B, N, Q1])
        assert output.base_dist.asymmetry.shape == torch.Size([S, B, N, Q1])

    def test_concrete_rejects_correlated_task_noise(self):
        with pytest.raises(NotImplementedError, match="rank must be 0"):
            MultitaskAsymmetricLaplaceLikelihood(1.0, Q1, rank=Q1)

    def test_global_noise_only(self):
        likelihood = MultitaskAsymmetricLaplaceLikelihood(
            1.0,
            Q1,
            has_task_noise=False,
        )
        likelihood.noise = 0.4

        output = likelihood(torch.zeros(N, Q1))

        expected_rate = 1 / (2 * 0.4**0.5)
        assert torch.allclose(
            output.base_dist.scale,
            torch.full((N, Q1), expected_rate),
        )

    def test_rejects_invalid_configuration(self):
        with pytest.raises(ValueError, match="no noise terms"):
            MultitaskAsymmetricLaplaceLikelihood(
                1.0,
                Q1,
                has_global_noise=False,
                has_task_noise=False,
            )
        with pytest.raises(ValueError, match="trailing dimension of kappa"):
            MultitaskAsymmetricLaplaceLikelihood(torch.ones(Q1 + 1), Q1)


# ===========================================================================
# CenterGapQuantilesLikelihood
# ===========================================================================


class TestCenterGapQuantilesLikelihood:
    # --- lower_counts ---

    def test_lower_count_scalar_index(self):
        likelihood = CenterGapQuantilesLikelihood(Q_LEVELS, CENTRAL_IDX)

        assert likelihood.lower_counts.shape == torch.Size([1])
        assert likelihood.lower_counts.item() == 2

    def test_lower_count_tensor_index(self):
        likelihood = CenterGapQuantilesLikelihood(
            Q_LEVELS,
            torch.tensor(CENTRAL_IDX),
        )

        assert likelihood.lower_counts.shape == torch.Size([1])
        assert likelihood.lower_counts.item() == 2

    def test_lower_count_uniform_batch(self):
        q = Q_LEVELS.unsqueeze(0).expand(B, Q).contiguous()
        likelihood = CenterGapQuantilesLikelihood(q, CENTRAL_IDX)

        assert likelihood.lower_counts.shape == torch.Size([B, 1])
        assert (likelihood.lower_counts == 2).all()

    def test_lower_count_varying_batch(self):
        # batch 0: center at index 2 (0.5) → lc=2
        # batch 1: center at index 3 (0.6) → lc=3
        q = torch.stack(
            [
                torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]),
                torch.tensor([0.1, 0.2, 0.4, 0.6, 0.9]),
            ]
        )
        likelihood = CenterGapQuantilesLikelihood(q, torch.tensor([2, 3]))

        assert likelihood.lower_counts.tolist() == [[2], [3]]

    # --- forward ---

    def test_forward_no_batch_type_and_shape(self):
        likelihood = CenterGapQuantilesLikelihood(Q_LEVELS, CENTRAL_IDX)

        output = likelihood.forward(torch.randn(S, N, Q))

        assert isinstance(output, Independent)
        assert isinstance(output.base_dist, AsymmetricLaplace)
        assert output.batch_shape == torch.Size([S, N])
        assert output.event_shape == torch.Size([Q])

    def test_forward_with_batch_type_and_shape(self):
        q = Q_LEVELS.unsqueeze(0).expand(B, Q).contiguous()
        likelihood = CenterGapQuantilesLikelihood(q, CENTRAL_IDX)

        output = likelihood.forward(torch.randn(S, B, N, Q))

        assert isinstance(output, Independent)
        assert output.base_dist.loc.shape == torch.Size([S, B, N, Q])
        assert output.batch_shape == torch.Size([S, B, N])
        assert output.event_shape == torch.Size([Q])

    def test_forward_reconstruction_no_batch(self):
        lc = CENTRAL_IDX
        likelihood = CenterGapQuantilesLikelihood(Q_LEVELS, CENTRAL_IDX)
        torch.manual_seed(0)
        fs = torch.randn(S, N, Q)

        output = likelihood.forward(fs)

        expected = centergap_to_quantiles(
            fs[..., :1], fs[..., 1 : 1 + lc], fs[..., 1 + lc :]
        )
        assert torch.allclose(output.base_dist.loc, expected, atol=1e-5)

    def test_forward_reconstruction_uniform_batch(self):
        lc = CENTRAL_IDX
        q = Q_LEVELS.unsqueeze(0).expand(B, Q).contiguous()
        likelihood = CenterGapQuantilesLikelihood(q, CENTRAL_IDX)
        torch.manual_seed(0)
        fs = torch.randn(S, B, N, Q)

        output = likelihood.forward(fs)

        for b in range(B):
            expected_b = centergap_to_quantiles(
                fs[:, b : b + 1, :, :1],
                fs[:, b : b + 1, :, 1 : 1 + lc],
                fs[:, b : b + 1, :, 1 + lc :],
            ).squeeze(1)
            assert torch.allclose(
                output.base_dist.loc[:, b],
                expected_b,
                atol=1e-5,
            )

    def test_forward_reconstruction_varying_lower_count(self):
        q = torch.stack(
            [
                torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]),
                torch.tensor([0.1, 0.2, 0.4, 0.6, 0.9]),
            ]
        )
        likelihood = CenterGapQuantilesLikelihood(q, torch.tensor([2, 3]))
        torch.manual_seed(0)
        fs = torch.randn(S, 2, N, Q)

        output = likelihood.forward(fs)

        for b, lc in enumerate([2, 3]):
            expected = centergap_to_quantiles(
                fs[:, b : b + 1, :, :1],
                fs[:, b : b + 1, :, 1 : 1 + lc],
                fs[:, b : b + 1, :, 1 + lc :],
            ).squeeze(1)
            assert torch.allclose(
                output.base_dist.loc[:, b],
                expected,
                atol=1e-5,
            )

    def test_forward_broadcast_q_larger_batch(self):
        """q shape (1, Q) but function_samples has actual batch K > 1."""
        K = 5
        q = Q_LEVELS.unsqueeze(0)  # (1, Q)
        likelihood = CenterGapQuantilesLikelihood(q, CENTRAL_IDX)
        torch.manual_seed(0)
        fs = torch.randn(S, K, N, Q)

        output = likelihood.forward(fs)

        assert output.base_dist.loc.shape == torch.Size([S, K, N, Q])
        lc = CENTRAL_IDX
        expected = centergap_to_quantiles(
            fs[..., :1], fs[..., 1 : 1 + lc], fs[..., 1 + lc :]
        )
        assert torch.allclose(output.base_dist.loc, expected, atol=1e-5)

    def test_forward_asymmetry_matches_quantile_levels(self):
        likelihood = CenterGapQuantilesLikelihood(Q_LEVELS, CENTRAL_IDX)

        output = likelihood.forward(torch.randn(S, N, Q))

        expected = (Q_LEVELS / (1 - Q_LEVELS)).sqrt().expand(S, N, Q)
        assert torch.allclose(output.base_dist.asymmetry, expected)

    def test_forward_output_is_sorted(self):
        """Quantiles must be non-decreasing after the center-gap transform."""
        likelihood = CenterGapQuantilesLikelihood(Q_LEVELS, CENTRAL_IDX)
        torch.manual_seed(0)

        output = likelihood.forward(torch.randn(S, N, Q))

        diffs = output.base_dist.loc[..., 1:] - output.base_dist.loc[..., :-1]
        assert (diffs >= 0).all()

    # --- expected_log_prob ---

    def test_expected_log_prob_no_batch_shape(self):
        likelihood = CenterGapQuantilesLikelihood(Q_LEVELS, CENTRAL_IDX)
        obs = torch.randn(N)
        dist = _make_mtmvn(N, Q)

        with gpytorch.settings.num_likelihood_samples(3):
            result = likelihood.expected_log_prob(obs, dist)

        assert result.shape == torch.Size([N])


# ===========================================================================
# MultioutputCenterGapQuantilesLikelihood
# ===========================================================================


class TestMultioutputCenterGapQuantilesLikelihood:
    def _make_lik_symmetric(self):
        """Q1=Q2=3, central_idx=1 for both. Layout: [c1,c2,L1,U1,L2,U2]."""
        q = torch.tensor([0.25, 0.5, 0.75])
        likelihood = MultioutputCenterGapQuantilesLikelihood([q, q], [1, 1])
        return likelihood, q

    def _make_lik_asymmetric(self):
        """Q1=3 (lc=1), Q2=5 (lc=2). Layout: [c1,c2, L1,U1, L2,L2,U2,U2]."""
        return MultioutputCenterGapQuantilesLikelihood(
            [Q1_LEVELS, Q2_LEVELS],
            [CENTRAL_IDX1, CENTRAL_IDX2],
        )

    # --- forward ---

    def test_forward_output_shape_symmetric(self):
        likelihood, _ = self._make_lik_symmetric()

        output = likelihood.forward(torch.randn(S, N, 6))

        assert isinstance(output, Independent)
        assert isinstance(output.base_dist, AsymmetricLaplace)
        assert output.base_dist.loc.shape == torch.Size([S, N, 6])
        assert output.event_shape == torch.Size([6])

    def test_forward_output_shape_asymmetric(self):
        likelihood = self._make_lik_asymmetric()

        output = likelihood.forward(torch.randn(S, N, Q1 + Q2))

        assert isinstance(output, Independent)
        assert output.base_dist.loc.shape == torch.Size([S, N, Q1 + Q2])
        assert output.event_shape == torch.Size([Q1 + Q2])

    def test_forward_correct_layout_symmetric(self):
        """
        Task layout: [c1, c2, L1, U1, L2, U2]  (Q1=Q2=3, lc1=lc2=1)
        Output 1 uses indices [0, 2, 3]  (c1, L1, U1)
        Output 2 uses indices [1, 4, 5]  (c2, L2, U2)
        """
        likelihood, _ = self._make_lik_symmetric()
        torch.manual_seed(0)
        fs = torch.randn(S, N, 6)

        output = likelihood.forward(fs)

        expected_1 = centergap_to_quantiles(fs[..., 0:1], fs[..., 2:3], fs[..., 3:4])
        expected_2 = centergap_to_quantiles(fs[..., 1:2], fs[..., 4:5], fs[..., 5:6])
        expected = torch.cat([expected_1, expected_2], dim=-1)

        assert torch.allclose(output.base_dist.loc, expected, atol=1e-5)

    def test_forward_correct_layout_asymmetric(self):
        """
        Task layout: [c1, c2, L1, U1, L2a, L2b, U2a, U2b]  (Q1=3 lc1=1, Q2=5 lc2=2)
        Output 1 uses indices [0, 2, 3]       (c1, L1, U1)
        Output 2 uses indices [1, 4, 5, 6, 7] (c2, L2a, L2b, U2a, U2b)
        """
        likelihood = self._make_lik_asymmetric()
        torch.manual_seed(0)
        fs = torch.randn(S, N, Q1 + Q2)

        output = likelihood.forward(fs)

        expected_1 = centergap_to_quantiles(fs[..., 0:1], fs[..., 2:3], fs[..., 3:4])
        expected_2 = centergap_to_quantiles(fs[..., 1:2], fs[..., 4:6], fs[..., 6:8])
        expected = torch.cat([expected_1, expected_2], dim=-1)

        assert torch.allclose(output.base_dist.loc, expected, atol=1e-5)

    def test_forward_asymmetry_values(self):
        likelihood, q = self._make_lik_symmetric()

        output = likelihood.forward(torch.randn(S, N, 6))

        expected = (q / (1 - q)).sqrt().repeat(2).expand(S, N, 6)
        assert torch.allclose(output.base_dist.asymmetry, expected)

    def test_forward_output_is_sorted_per_task(self):
        """Quantiles within each output must be non-decreasing."""
        likelihood = self._make_lik_asymmetric()
        torch.manual_seed(0)

        output = likelihood.forward(torch.randn(S, N, Q1 + Q2))

        # output 1: columns 0..Q1-1
        diffs_1 = (
            output.base_dist.loc[..., 1:Q1] - output.base_dist.loc[..., 0 : Q1 - 1]
        )
        assert (diffs_1 >= 0).all()
        # output 2: columns Q1..Q1+Q2-1
        diffs_2 = (
            output.base_dist.loc[..., Q1 + 1 : Q1 + Q2]
            - output.base_dist.loc[..., Q1 : Q1 + Q2 - 1]
        )
        assert (diffs_2 >= 0).all()

    # --- expected_log_prob ---

    def test_expected_log_prob_shape(self):
        likelihood, _ = self._make_lik_symmetric()
        obs = torch.randn(N, 2)  # K=2 outputs
        dist = _make_mtmvn(N, 6)

        with gpytorch.settings.num_likelihood_samples(3):
            result = likelihood.expected_log_prob(obs, dist)

        assert result.shape == torch.Size([N])
