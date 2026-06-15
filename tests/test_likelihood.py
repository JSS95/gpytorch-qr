import torch

from gpytorch_qr.distributions import QuantileALD
from gpytorch_qr.likelihoods import CenterGapQuantileLikelihood
from gpytorch_qr.utils import centergap_to_quantiles

Q = 5
S = 4
N = 10
Q_LEVELS = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])


MT_Q = Q_LEVELS  # (Q,) — no batch


def test_mt_lower_count_scalar_index_no_batch():
    lik = CenterGapQuantileLikelihood(MT_Q, central_quantile_index=2)
    assert lik.lower_count.dim() == 0
    assert lik.lower_count.item() == 2


def test_mt_lower_count_tensor_index_no_batch():
    idx = torch.tensor(2)
    lik = CenterGapQuantileLikelihood(MT_Q, central_quantile_index=idx)
    assert lik.lower_count.dim() == 0
    assert lik.lower_count.item() == 2


def test_mt_lower_count_scalar_index_with_batch():
    q = Q_LEVELS.unsqueeze(0).expand(3, Q).contiguous()  # (3, Q)
    lik = CenterGapQuantileLikelihood(q, central_quantile_index=2)
    assert lik.lower_count.shape == torch.Size([3])
    assert (lik.lower_count == 2).all()


def test_mt_lower_count_tensor_index_varying():
    # batch 0: center at q=0.5 (index 2) -> 2 lower
    # batch 1: center at q=0.6 (index 3) -> 3 lower
    q = torch.stack(
        [
            torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]),
            torch.tensor([0.1, 0.2, 0.4, 0.6, 0.9]),
        ],
        dim=0,
    )  # (2, Q)
    idx = torch.tensor([2, 3])
    lik = CenterGapQuantileLikelihood(q, central_quantile_index=idx)
    assert lik.lower_count.tolist() == [2, 3]


def test_mt_forward_no_batch_output_type_and_shape():
    lik = CenterGapQuantileLikelihood(MT_Q, central_quantile_index=2)
    fs = torch.randn(S, N, Q)
    out = lik.forward(fs)
    assert isinstance(out, QuantileALD)
    assert out.m.shape == torch.Size([S, N, Q])


def test_mt_forward_with_batch_output_type_and_shape():
    B = 3
    q = Q_LEVELS.unsqueeze(0).expand(B, Q).contiguous()  # (B, Q)
    lik = CenterGapQuantileLikelihood(q, central_quantile_index=2)
    fs = torch.randn(S, B, N, Q)
    out = lik.forward(fs)
    assert isinstance(out, QuantileALD)
    assert out.m.shape == torch.Size([S, B, N, Q])


def test_mt_forward_reconstruction_no_batch():
    lc = 2
    lik = CenterGapQuantileLikelihood(MT_Q, central_quantile_index=2)
    torch.manual_seed(0)
    fs = torch.randn(S, N, Q)

    out = lik.forward(fs)

    expected = centergap_to_quantiles(
        fs[..., :1],
        fs[..., 1 : 1 + lc],
        fs[..., 1 + lc :],
        quantile_dim=-1,
    )
    assert torch.allclose(out.m, expected, atol=1e-5)


def test_mt_forward_reconstruction_uniform_batch():
    B, lc = 3, 2
    q = Q_LEVELS.unsqueeze(0).expand(B, Q).contiguous()  # (B, Q)
    lik = CenterGapQuantileLikelihood(q, central_quantile_index=2)
    torch.manual_seed(0)
    fs = torch.randn(S, B, N, Q)

    out = lik.forward(fs)

    for b in range(B):
        expected_b = centergap_to_quantiles(
            fs[:, b : b + 1, :, :1],
            fs[:, b : b + 1, :, 1 : 1 + lc],
            fs[:, b : b + 1, :, 1 + lc :],
            quantile_dim=-1,
        ).squeeze(1)
        assert torch.allclose(out.m[:, b, :, :], expected_b, atol=1e-5)


def test_mt_forward_reconstruction_varying_lower_count():
    q = torch.stack(
        [
            torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]),
            torch.tensor([0.1, 0.2, 0.4, 0.6, 0.9]),
        ],
        dim=0,
    )  # (2, Q): lower_counts [2, 3]
    idx = torch.tensor([2, 3])
    lik = CenterGapQuantileLikelihood(q, central_quantile_index=idx)

    torch.manual_seed(0)
    fs = torch.randn(S, 2, N, Q)

    out = lik.forward(fs)

    # batch 0: lc=2
    lc0 = 2
    expected0 = centergap_to_quantiles(
        fs[:, 0:1, :, :1],
        fs[:, 0:1, :, 1 : 1 + lc0],
        fs[:, 0:1, :, 1 + lc0 :],
        quantile_dim=-1,
    ).squeeze(1)
    assert torch.allclose(out.m[:, 0, :, :], expected0, atol=1e-5)

    # batch 1: lc=3
    lc1 = 3
    expected1 = centergap_to_quantiles(
        fs[:, 1:2, :, :1],
        fs[:, 1:2, :, 1 : 1 + lc1],
        fs[:, 1:2, :, 1 + lc1 :],
        quantile_dim=-1,
    ).squeeze(1)
    assert torch.allclose(out.m[:, 1, :, :], expected1, atol=1e-5)


def test_mt_forward_broadcast_q_with_larger_batch():
    """q shape (1, Q) but function_samples has actual batch K > 1."""
    K = 5
    q_broadcast = Q_LEVELS.unsqueeze(0)  # (1, Q)
    raw_scales = torch.zeros(K, Q)
    lik = CenterGapQuantileLikelihood(
        q_broadcast, central_quantile_index=2, raw_scales=raw_scales
    )
    assert lik.lower_count.shape == torch.Size([1])

    torch.manual_seed(0)
    fs = torch.randn(S, K, N, Q)

    out = lik.forward(fs)

    assert out.m.shape == torch.Size([S, K, N, Q])
    lc = 2
    expected = centergap_to_quantiles(
        fs[..., :1],
        fs[..., 1 : 1 + lc],
        fs[..., 1 + lc :],
        quantile_dim=-1,
    )
    assert torch.allclose(out.m, expected, atol=1e-5)


def test_mt_forward_ald_kappa_lamda_shapes_no_batch():
    lik = CenterGapQuantileLikelihood(MT_Q, central_quantile_index=2)
    fs = torch.randn(S, N, Q)
    out = lik.forward(fs)
    assert out.kappa.shape == torch.Size([1, 1, Q])
    assert out.lamda.shape == torch.Size([1, 1, Q])


def test_mt_forward_ald_kappa_lamda_shapes_with_batch():
    B = 3
    q = Q_LEVELS.unsqueeze(0).expand(B, Q).contiguous()
    lik = CenterGapQuantileLikelihood(q, central_quantile_index=2)
    fs = torch.randn(S, B, N, Q)
    out = lik.forward(fs)
    assert out.kappa.shape == torch.Size([1, B, 1, Q])
    assert out.lamda.shape == torch.Size([1, B, 1, Q])
