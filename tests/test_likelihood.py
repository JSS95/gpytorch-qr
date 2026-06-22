import gpytorch
import torch

from gpytorch_qr.distributions import QuantileALD
from gpytorch_qr.likelihoods import (
    CenterGapQuantileLikelihood,
    DirectQuantileLikelihood,
    MultiOutputCenterGapQuantileLikelihood,
    MultiOutputDirectQuantileLikelihood,
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
# DirectQuantileLikelihood
# ===========================================================================


class TestDirectQuantileLikelihood:
    # --- forward ---

    def test_forward_no_batch_returns_quantile_ald(self):
        lik = DirectQuantileLikelihood(Q_LEVELS)
        out = lik.forward(torch.randn(S, N, Q))
        assert isinstance(out, QuantileALD)

    def test_forward_no_batch_m_shape(self):
        lik = DirectQuantileLikelihood(Q_LEVELS)
        out = lik.forward(torch.randn(S, N, Q))
        assert out.m.shape == torch.Size([S, N, Q])

    def test_forward_with_batch_m_shape(self):
        q = Q_LEVELS.unsqueeze(0).expand(B, Q).contiguous()
        lik = DirectQuantileLikelihood(q)
        out = lik.forward(torch.randn(S, B, N, Q))
        assert out.m.shape == torch.Size([S, B, N, Q])

    def test_forward_m_is_function_samples(self):
        lik = DirectQuantileLikelihood(Q_LEVELS)
        fs = torch.randn(S, N, Q)
        assert torch.equal(lik.forward(fs).m, fs)

    def test_forward_kappa_lamda_shapes_no_batch(self):
        lik = DirectQuantileLikelihood(Q_LEVELS)
        out = lik.forward(torch.randn(S, N, Q))
        assert out.kappa.shape == torch.Size([1, 1, Q])
        assert out.lamda.shape == torch.Size([1, 1, Q])

    def test_forward_kappa_lamda_shapes_with_batch(self):
        q = Q_LEVELS.unsqueeze(0).expand(B, Q).contiguous()
        lik = DirectQuantileLikelihood(q)
        out = lik.forward(torch.randn(S, B, N, Q))
        assert out.kappa.shape == torch.Size([1, B, 1, Q])
        assert out.lamda.shape == torch.Size([1, B, 1, Q])

    def test_forward_kappa_values(self):
        lik = DirectQuantileLikelihood(Q_LEVELS)
        out = lik.forward(torch.randn(S, N, Q))
        assert torch.allclose(out.kappa.squeeze(0), Q_LEVELS)

    # --- expected_log_prob ---

    def test_expected_log_prob_no_batch_shape(self):
        lik = DirectQuantileLikelihood(Q_LEVELS)
        obs = torch.randn(N)
        dist = _make_mtmvn(N, Q)
        with gpytorch.settings.num_likelihood_samples(3):
            result = lik.expected_log_prob(obs, dist)
        assert result.shape == torch.Size([N])

    def test_expected_log_prob_with_batch_shape(self):
        q = Q_LEVELS.unsqueeze(0).expand(B, Q).contiguous()
        lik = DirectQuantileLikelihood(q)
        obs = torch.randn(B, N)
        dist = _make_mtmvn(N, Q, batch_shape=torch.Size([B]))
        with gpytorch.settings.num_likelihood_samples(3):
            result = lik.expected_log_prob(obs, dist)
        assert result.shape == torch.Size([B, N])


# ===========================================================================
# CenterGapQuantileLikelihood
# ===========================================================================


class TestCenterGapQuantileLikelihood:
    # --- lower_count ---

    def test_lower_count_scalar_index(self):
        lik = CenterGapQuantileLikelihood(Q_LEVELS, CENTRAL_IDX)
        assert lik.lower_count.dim() == 0
        assert lik.lower_count.item() == 2

    def test_lower_count_tensor_index(self):
        lik = CenterGapQuantileLikelihood(Q_LEVELS, torch.tensor(CENTRAL_IDX))
        assert lik.lower_count.dim() == 0
        assert lik.lower_count.item() == 2

    def test_lower_count_uniform_batch(self):
        q = Q_LEVELS.unsqueeze(0).expand(B, Q).contiguous()
        lik = CenterGapQuantileLikelihood(q, CENTRAL_IDX)
        assert lik.lower_count.shape == torch.Size([B])
        assert (lik.lower_count == 2).all()

    def test_lower_count_varying_batch(self):
        # batch 0: center at index 2 (0.5) → lc=2
        # batch 1: center at index 3 (0.6) → lc=3
        q = torch.stack(
            [
                torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]),
                torch.tensor([0.1, 0.2, 0.4, 0.6, 0.9]),
            ]
        )
        lik = CenterGapQuantileLikelihood(q, torch.tensor([2, 3]))
        assert lik.lower_count.tolist() == [2, 3]

    # --- forward ---

    def test_forward_no_batch_type_and_shape(self):
        lik = CenterGapQuantileLikelihood(Q_LEVELS, CENTRAL_IDX)
        out = lik.forward(torch.randn(S, N, Q))
        assert isinstance(out, QuantileALD)
        assert out.m.shape == torch.Size([S, N, Q])

    def test_forward_with_batch_type_and_shape(self):
        q = Q_LEVELS.unsqueeze(0).expand(B, Q).contiguous()
        lik = CenterGapQuantileLikelihood(q, CENTRAL_IDX)
        out = lik.forward(torch.randn(S, B, N, Q))
        assert isinstance(out, QuantileALD)
        assert out.m.shape == torch.Size([S, B, N, Q])

    def test_forward_reconstruction_no_batch(self):
        lc = CENTRAL_IDX
        lik = CenterGapQuantileLikelihood(Q_LEVELS, CENTRAL_IDX)
        torch.manual_seed(0)
        fs = torch.randn(S, N, Q)
        out = lik.forward(fs)
        expected = centergap_to_quantiles(
            fs[..., :1], fs[..., 1 : 1 + lc], fs[..., 1 + lc :]
        )
        assert torch.allclose(out.m, expected, atol=1e-5)

    def test_forward_reconstruction_uniform_batch(self):
        lc = CENTRAL_IDX
        q = Q_LEVELS.unsqueeze(0).expand(B, Q).contiguous()
        lik = CenterGapQuantileLikelihood(q, CENTRAL_IDX)
        torch.manual_seed(0)
        fs = torch.randn(S, B, N, Q)
        out = lik.forward(fs)
        for b in range(B):
            expected_b = centergap_to_quantiles(
                fs[:, b : b + 1, :, :1],
                fs[:, b : b + 1, :, 1 : 1 + lc],
                fs[:, b : b + 1, :, 1 + lc :],
            ).squeeze(1)
            assert torch.allclose(out.m[:, b], expected_b, atol=1e-5)

    def test_forward_reconstruction_varying_lower_count(self):
        q = torch.stack(
            [
                torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]),
                torch.tensor([0.1, 0.2, 0.4, 0.6, 0.9]),
            ]
        )
        lik = CenterGapQuantileLikelihood(q, torch.tensor([2, 3]))
        torch.manual_seed(0)
        fs = torch.randn(S, 2, N, Q)
        out = lik.forward(fs)
        for b, lc in enumerate([2, 3]):
            expected = centergap_to_quantiles(
                fs[:, b : b + 1, :, :1],
                fs[:, b : b + 1, :, 1 : 1 + lc],
                fs[:, b : b + 1, :, 1 + lc :],
            ).squeeze(1)
            assert torch.allclose(out.m[:, b], expected, atol=1e-5)

    def test_forward_broadcast_q_larger_batch(self):
        """q shape (1, Q) but function_samples has actual batch K > 1."""
        K = 5
        q = Q_LEVELS.unsqueeze(0)  # (1, Q)
        lik = CenterGapQuantileLikelihood(q, CENTRAL_IDX, raw_scales=torch.zeros(K, Q))
        torch.manual_seed(0)
        fs = torch.randn(S, K, N, Q)
        out = lik.forward(fs)
        assert out.m.shape == torch.Size([S, K, N, Q])
        lc = CENTRAL_IDX
        expected = centergap_to_quantiles(
            fs[..., :1], fs[..., 1 : 1 + lc], fs[..., 1 + lc :]
        )
        assert torch.allclose(out.m, expected, atol=1e-5)

    def test_forward_kappa_lamda_shapes_no_batch(self):
        lik = CenterGapQuantileLikelihood(Q_LEVELS, CENTRAL_IDX)
        out = lik.forward(torch.randn(S, N, Q))
        assert out.kappa.shape == torch.Size([1, 1, Q])
        assert out.lamda.shape == torch.Size([1, 1, Q])

    def test_forward_kappa_lamda_shapes_with_batch(self):
        q = Q_LEVELS.unsqueeze(0).expand(B, Q).contiguous()
        lik = CenterGapQuantileLikelihood(q, CENTRAL_IDX)
        out = lik.forward(torch.randn(S, B, N, Q))
        assert out.kappa.shape == torch.Size([1, B, 1, Q])
        assert out.lamda.shape == torch.Size([1, B, 1, Q])

    def test_forward_output_is_sorted(self):
        """Quantiles must be non-decreasing after the center-gap transform."""
        lik = CenterGapQuantileLikelihood(Q_LEVELS, CENTRAL_IDX)
        torch.manual_seed(0)
        out = lik.forward(torch.randn(S, N, Q))
        diffs = out.m[..., 1:] - out.m[..., :-1]
        assert (diffs >= 0).all()

    # --- expected_log_prob ---

    def test_expected_log_prob_no_batch_shape(self):
        lik = CenterGapQuantileLikelihood(Q_LEVELS, CENTRAL_IDX)
        obs = torch.randn(N)
        dist = _make_mtmvn(N, Q)
        with gpytorch.settings.num_likelihood_samples(3):
            result = lik.expected_log_prob(obs, dist)
        assert result.shape == torch.Size([N])


# ===========================================================================
# MultiOutputDirectQuantileLikelihood
# ===========================================================================


class TestMultiOutputDirectQuantileLikelihood:
    def _make_lik(self):
        return MultiOutputDirectQuantileLikelihood(
            DirectQuantileLikelihood(Q1_LEVELS),
            DirectQuantileLikelihood(Q2_LEVELS),
        )

    # --- forward ---

    def test_forward_output_shape(self):
        lik = self._make_lik()
        out = lik.forward(torch.randn(S, N, Q1 + Q2))
        assert isinstance(out, QuantileALD)
        assert out.m.shape == torch.Size([S, N, Q1 + Q2])

    def test_forward_asymmetric_output_shape(self):
        lik = MultiOutputDirectQuantileLikelihood(
            DirectQuantileLikelihood(torch.tensor([0.1, 0.9])),
            DirectQuantileLikelihood(torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])),
        )
        out = lik.forward(torch.randn(S, N, 7))
        assert out.m.shape == torch.Size([S, N, 7])

    def test_forward_m_is_function_samples(self):
        """DirectQuantileLikelihood passes m through unchanged."""
        lik = self._make_lik()
        fs = torch.randn(S, N, Q1 + Q2)
        assert torch.equal(lik.forward(fs).m, fs)

    def test_forward_first_output_m(self):
        """First Q1 columns of m should match forward of first likelihood alone."""
        lik = self._make_lik()
        fs = torch.randn(S, N, Q1 + Q2)
        out = lik.forward(fs)
        assert torch.equal(out.m[..., :Q1], fs[..., :Q1])

    def test_forward_second_output_m(self):
        """Last Q2 columns of m should match forward of second likelihood alone."""
        lik = self._make_lik()
        fs = torch.randn(S, N, Q1 + Q2)
        out = lik.forward(fs)
        assert torch.equal(out.m[..., Q1:], fs[..., Q1:])

    def test_forward_kappa_values(self):
        lik = self._make_lik()
        out = lik.forward(torch.randn(S, N, Q1 + Q2))
        expected_kappa = torch.cat([Q1_LEVELS, Q2_LEVELS])
        assert torch.allclose(out.kappa.flatten(), expected_kappa)

    # --- expected_log_prob ---

    def test_expected_log_prob_shape(self):
        lik = self._make_lik()
        obs = torch.randn(N, 2)  # K=2 outputs
        dist = _make_mtmvn(N, Q1 + Q2)
        with gpytorch.settings.num_likelihood_samples(3):
            result = lik.expected_log_prob(obs, dist)
        assert result.shape == torch.Size([N])


# ===========================================================================
# MultiOutputCenterGapQuantileLikelihood
# ===========================================================================


class TestMultiOutputCenterGapQuantileLikelihood:
    def _make_lik_symmetric(self):
        """Q1=Q2=3, central_idx=1 for both. Layout: [c1,c2,L1,U1,L2,U2]."""
        q = torch.tensor([0.25, 0.5, 0.75])
        lik = MultiOutputCenterGapQuantileLikelihood(
            CenterGapQuantileLikelihood(q, 1),
            CenterGapQuantileLikelihood(q, 1),
        )
        return lik, q

    def _make_lik_asymmetric(self):
        """Q1=3 (lc=1), Q2=5 (lc=2). Layout: [c1,c2, L1,U1, L2,L2,U2,U2]."""
        return MultiOutputCenterGapQuantileLikelihood(
            CenterGapQuantileLikelihood(Q1_LEVELS, CENTRAL_IDX1),
            CenterGapQuantileLikelihood(Q2_LEVELS, CENTRAL_IDX2),
        )

    # --- forward ---

    def test_forward_output_shape_symmetric(self):
        lik, _ = self._make_lik_symmetric()
        out = lik.forward(torch.randn(S, N, 6))
        assert isinstance(out, QuantileALD)
        assert out.m.shape == torch.Size([S, N, 6])

    def test_forward_output_shape_asymmetric(self):
        lik = self._make_lik_asymmetric()
        out = lik.forward(torch.randn(S, N, Q1 + Q2))
        assert isinstance(out, QuantileALD)
        assert out.m.shape == torch.Size([S, N, Q1 + Q2])

    def test_forward_correct_layout_symmetric(self):
        """
        Task layout: [c1, c2, L1, U1, L2, U2]  (Q1=Q2=3, lc1=lc2=1)
        Output 1 uses indices [0, 2, 3]  (c1, L1, U1)
        Output 2 uses indices [1, 4, 5]  (c2, L2, U2)
        """
        lik, _ = self._make_lik_symmetric()
        torch.manual_seed(0)
        fs = torch.randn(S, N, 6)
        out = lik.forward(fs)

        expected_1 = centergap_to_quantiles(fs[..., 0:1], fs[..., 2:3], fs[..., 3:4])
        expected_2 = centergap_to_quantiles(fs[..., 1:2], fs[..., 4:5], fs[..., 5:6])
        expected = torch.cat([expected_1, expected_2], dim=-1)

        assert torch.allclose(out.m, expected, atol=1e-5)

    def test_forward_correct_layout_asymmetric(self):
        """
        Task layout: [c1, c2, L1, U1, L2a, L2b, U2a, U2b]  (Q1=3 lc1=1, Q2=5 lc2=2)
        Output 1 uses indices [0, 2, 3]       (c1, L1, U1)
        Output 2 uses indices [1, 4, 5, 6, 7] (c2, L2a, L2b, U2a, U2b)
        """
        lik = self._make_lik_asymmetric()
        torch.manual_seed(0)
        fs = torch.randn(S, N, Q1 + Q2)
        out = lik.forward(fs)

        expected_1 = centergap_to_quantiles(fs[..., 0:1], fs[..., 2:3], fs[..., 3:4])
        expected_2 = centergap_to_quantiles(fs[..., 1:2], fs[..., 4:6], fs[..., 6:8])
        expected = torch.cat([expected_1, expected_2], dim=-1)

        assert torch.allclose(out.m, expected, atol=1e-5)

    def test_forward_kappa_values(self):
        lik, q = self._make_lik_symmetric()
        out = lik.forward(torch.randn(S, N, 6))
        expected_kappa = torch.cat([q, q])
        assert torch.allclose(out.kappa.flatten(), expected_kappa)

    def test_forward_output_is_sorted_per_task(self):
        """Quantiles within each output must be non-decreasing."""
        lik = self._make_lik_asymmetric()
        torch.manual_seed(0)
        out = lik.forward(torch.randn(S, N, Q1 + Q2))
        # output 1: columns 0..Q1-1
        diffs_1 = out.m[..., 1:Q1] - out.m[..., 0 : Q1 - 1]
        assert (diffs_1 >= 0).all()
        # output 2: columns Q1..Q1+Q2-1
        diffs_2 = out.m[..., Q1 + 1 : Q1 + Q2] - out.m[..., Q1 : Q1 + Q2 - 1]
        assert (diffs_2 >= 0).all()

    # --- expected_log_prob ---

    def test_expected_log_prob_shape(self):
        lik, _ = self._make_lik_symmetric()
        obs = torch.randn(N, 2)  # K=2 outputs
        dist = _make_mtmvn(N, 6)
        with gpytorch.settings.num_likelihood_samples(3):
            result = lik.expected_log_prob(obs, dist)
        assert result.shape == torch.Size([N])
