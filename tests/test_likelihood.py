import torch

from gpytorch_qr.distributions import BatchQuantileALD, MultitaskQuantileALD
from gpytorch_qr.likelihoods import (
    BatchCenterGapQuantileGPLikelihood,
    MultitaskCenterGapQuantileGPLikelihood,
)
from gpytorch_qr.utils import centergap_to_quantiles

Q = 5
S = 4
N = 10
Q_LEVELS = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])


def test_lower_count_scalar_index_no_batch():
    """Scalar index, no batch: lower_count should be a 0-dim tensor."""
    lik = BatchCenterGapQuantileGPLikelihood(Q_LEVELS, central_quantile_index=2)
    assert lik.lower_count.dim() == 0
    assert lik.lower_count.item() == 2


def test_lower_count_tensor_index_no_batch():
    """0-dim tensor index behaves the same as scalar."""
    idx = torch.tensor(2)
    lik = BatchCenterGapQuantileGPLikelihood(Q_LEVELS, central_quantile_index=idx)
    assert lik.lower_count.dim() == 0
    assert lik.lower_count.item() == 2


def test_lower_count_scalar_index_with_batch():
    """Scalar index with batched q: lower_count shape should match batch shape."""
    q = Q_LEVELS.unsqueeze(1).expand(Q, 3).contiguous()  # (5, 3)
    lik = BatchCenterGapQuantileGPLikelihood(q, central_quantile_index=2)
    assert lik.lower_count.shape == torch.Size([3])
    assert (lik.lower_count == 2).all()


def test_lower_count_tensor_index_uniform():
    """Tensor index with uniform value across batches."""
    q = Q_LEVELS.unsqueeze(1).expand(Q, 3).contiguous()  # (5, 3)
    idx = torch.tensor([2, 2, 2])
    lik = BatchCenterGapQuantileGPLikelihood(q, central_quantile_index=idx)
    assert lik.lower_count.shape == torch.Size([3])
    assert (lik.lower_count == 2).all()


def test_lower_count_tensor_index_varying():
    """Tensor index with different values per batch."""
    # batch 0: center at q=0.5 -> 2 lower quantiles
    # batch 1: center at q=0.6 -> 3 lower quantiles
    q = torch.stack(
        [
            torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]),
            torch.tensor([0.1, 0.2, 0.4, 0.6, 0.9]),
        ],
        dim=1,
    )  # (5, 2)
    idx = torch.tensor([2, 3])
    lik = BatchCenterGapQuantileGPLikelihood(q, central_quantile_index=idx)
    assert lik.lower_count.tolist() == [2, 3]


def test_forward_no_batch_output_type_and_shape():
    lik = BatchCenterGapQuantileGPLikelihood(Q_LEVELS, central_quantile_index=2)
    fs = torch.randn(S, Q, N)
    out = lik.forward(fs)
    assert isinstance(out, BatchQuantileALD)
    assert out.m.shape == torch.Size([S, Q, N])


def test_forward_with_batch_output_type_and_shape():
    B = 3
    q = Q_LEVELS.unsqueeze(1).expand(Q, B).contiguous()
    lik = BatchCenterGapQuantileGPLikelihood(q, central_quantile_index=2)
    fs = torch.randn(S, Q, B, N)
    out = lik.forward(fs)
    assert isinstance(out, BatchQuantileALD)
    assert out.m.shape == torch.Size([S, Q, B, N])


def test_forward_reconstruction_no_batch():
    """Reconstructed quantiles match manual centergap_to_quantiles call."""
    lc = 2  # center at index 2 -> 2 lower quantiles
    lik = BatchCenterGapQuantileGPLikelihood(Q_LEVELS, central_quantile_index=2)
    torch.manual_seed(0)
    fs = torch.randn(S, Q, N)

    out = lik.forward(fs)

    expected = centergap_to_quantiles(
        fs[:, :1, :],
        fs[:, 1 : 1 + lc, :],
        fs[:, 1 + lc :, :],
        quantile_dim=1,
    )
    assert torch.allclose(out.m, expected)


def test_forward_reconstruction_uniform_batch():
    """Batched case with uniform lower_count matches per-batch manual reconstruction."""
    B, lc = 3, 2
    q = Q_LEVELS.unsqueeze(1).expand(Q, B).contiguous()
    lik = BatchCenterGapQuantileGPLikelihood(q, central_quantile_index=2)
    torch.manual_seed(0)
    fs = torch.randn(S, Q, B, N)

    out = lik.forward(fs)

    for b in range(B):
        expected_b = centergap_to_quantiles(
            fs[:, :1, b : b + 1, :],
            fs[:, 1 : 1 + lc, b : b + 1, :],
            fs[:, 1 + lc :, b : b + 1, :],
            quantile_dim=1,
        ).squeeze(2)
        assert torch.allclose(out.m[:, :, b, :], expected_b)


def test_forward_reconstruction_varying_lower_count():
    """Each batch has a different lower_count; reconstruction is independent."""
    q = torch.stack(
        [
            torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]),
            torch.tensor([0.1, 0.2, 0.4, 0.6, 0.9]),
        ],
        dim=1,
    )  # (5, 2): lower_counts [2, 3]
    idx = torch.tensor([2, 3])
    lik = BatchCenterGapQuantileGPLikelihood(q, central_quantile_index=idx)

    torch.manual_seed(0)
    fs = torch.randn(S, Q, 2, N)

    out = lik.forward(fs)

    # batch 0: lc=2
    lc0 = 2
    expected0 = centergap_to_quantiles(
        fs[:, :1, 0:1, :],
        fs[:, 1 : 1 + lc0, 0:1, :],
        fs[:, 1 + lc0 :, 0:1, :],
        quantile_dim=1,
    ).squeeze(2)
    assert torch.allclose(out.m[:, :, 0, :], expected0)

    # batch 1: lc=3
    lc1 = 3
    expected1 = centergap_to_quantiles(
        fs[:, :1, 1:2, :],
        fs[:, 1 : 1 + lc1, 1:2, :],
        fs[:, 1 + lc1 :, 1:2, :],
        quantile_dim=1,
    ).squeeze(2)
    assert torch.allclose(out.m[:, :, 1, :], expected1)


def test_forward_ald_kappa_lamda_shapes_no_batch():
    lik = BatchCenterGapQuantileGPLikelihood(Q_LEVELS, central_quantile_index=2)
    fs = torch.randn(S, Q, N)
    out = lik.forward(fs)
    # kappa and lamda are stored with an extra leading sample dim in BatchQuantileALD
    assert out.kappa.shape == torch.Size([1, Q, 1])
    assert out.lamda.shape == torch.Size([1, Q, 1])


def test_forward_ald_kappa_lamda_shapes_with_batch():
    B = 3
    q = Q_LEVELS.unsqueeze(1).expand(Q, B).contiguous()
    lik = BatchCenterGapQuantileGPLikelihood(q, central_quantile_index=2)
    fs = torch.randn(S, Q, B, N)
    out = lik.forward(fs)
    assert out.kappa.shape == torch.Size([1, Q, B, 1])
    assert out.lamda.shape == torch.Size([1, Q, B, 1])


def test_forward_broadcast_q_with_larger_batch():
    """q shape (Q, 1) but function_samples has actual batch K > 1.

    This is the broadcast scenario from the cross-validation notebook:
        likelihood = BatchCenterGapQuantileGPLikelihood(q.unsqueeze(1), idx, scales)
    where scales has shape (Q, K), so function_samples has batch dim K.
    The bug was that lc.shape == (1,) while function_samples.shape[2] == K,
    causing reshape to fail. The fix derives B_shape from function_samples.
    """
    K = 5
    # q is broadcast: shape (Q, 1) — same quantiles for all batches
    q_broadcast = Q_LEVELS.unsqueeze(1)  # (Q, 1)
    raw_scales = torch.zeros(Q, K)
    lik = BatchCenterGapQuantileGPLikelihood(
        q_broadcast, central_quantile_index=2, raw_scales=raw_scales
    )
    # lc was computed from q of shape (Q, 1) -> lower_count shape (1,)
    assert lik.lower_count.shape == torch.Size([1])

    torch.manual_seed(0)
    fs = torch.randn(S, Q, K, N)  # actual batch K=5

    # Should not raise RuntimeError
    out = lik.forward(fs)

    assert out.m.shape == torch.Size([S, Q, K, N])
    # All batches use the same lc=2, so results must match manual reconstruction
    lc = 2
    expected = centergap_to_quantiles(
        fs[:, :1, :, :],
        fs[:, 1 : 1 + lc, :, :],
        fs[:, 1 + lc :, :, :],
        quantile_dim=1,
    )
    assert torch.allclose(out.m, expected)


# ---------------------------------------------------------------------------
# MultitaskCenterGapQuantileGPLikelihood tests
# q shape: (*B, Q)  — Q is the LAST dimension
# function_samples shape: (S, *B, N, Q)
# ---------------------------------------------------------------------------

MT_Q = Q_LEVELS  # (Q,) — no batch


def test_mt_lower_count_scalar_index_no_batch():
    lik = MultitaskCenterGapQuantileGPLikelihood(MT_Q, central_quantile_index=2)
    assert lik.lower_count.dim() == 0
    assert lik.lower_count.item() == 2


def test_mt_lower_count_tensor_index_no_batch():
    idx = torch.tensor(2)
    lik = MultitaskCenterGapQuantileGPLikelihood(MT_Q, central_quantile_index=idx)
    assert lik.lower_count.dim() == 0
    assert lik.lower_count.item() == 2


def test_mt_lower_count_scalar_index_with_batch():
    q = Q_LEVELS.unsqueeze(0).expand(3, Q).contiguous()  # (3, Q)
    lik = MultitaskCenterGapQuantileGPLikelihood(q, central_quantile_index=2)
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
    lik = MultitaskCenterGapQuantileGPLikelihood(q, central_quantile_index=idx)
    assert lik.lower_count.tolist() == [2, 3]


def test_mt_forward_no_batch_output_type_and_shape():
    lik = MultitaskCenterGapQuantileGPLikelihood(MT_Q, central_quantile_index=2)
    fs = torch.randn(S, N, Q)
    out = lik.forward(fs)
    assert isinstance(out, MultitaskQuantileALD)
    assert out.m.shape == torch.Size([S, N, Q])


def test_mt_forward_with_batch_output_type_and_shape():
    B = 3
    q = Q_LEVELS.unsqueeze(0).expand(B, Q).contiguous()  # (B, Q)
    lik = MultitaskCenterGapQuantileGPLikelihood(q, central_quantile_index=2)
    fs = torch.randn(S, B, N, Q)
    out = lik.forward(fs)
    assert isinstance(out, MultitaskQuantileALD)
    assert out.m.shape == torch.Size([S, B, N, Q])


def test_mt_forward_reconstruction_no_batch():
    lc = 2
    lik = MultitaskCenterGapQuantileGPLikelihood(MT_Q, central_quantile_index=2)
    torch.manual_seed(0)
    fs = torch.randn(S, N, Q)

    out = lik.forward(fs)

    expected = centergap_to_quantiles(
        fs[..., :1],
        fs[..., 1 : 1 + lc],
        fs[..., 1 + lc :],
        quantile_dim=-1,
    )
    assert torch.allclose(out.m, expected)


def test_mt_forward_reconstruction_uniform_batch():
    B, lc = 3, 2
    q = Q_LEVELS.unsqueeze(0).expand(B, Q).contiguous()  # (B, Q)
    lik = MultitaskCenterGapQuantileGPLikelihood(q, central_quantile_index=2)
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
        assert torch.allclose(out.m[:, b, :, :], expected_b)


def test_mt_forward_reconstruction_varying_lower_count():
    q = torch.stack(
        [
            torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]),
            torch.tensor([0.1, 0.2, 0.4, 0.6, 0.9]),
        ],
        dim=0,
    )  # (2, Q): lower_counts [2, 3]
    idx = torch.tensor([2, 3])
    lik = MultitaskCenterGapQuantileGPLikelihood(q, central_quantile_index=idx)

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
    assert torch.allclose(out.m[:, 0, :, :], expected0)

    # batch 1: lc=3
    lc1 = 3
    expected1 = centergap_to_quantiles(
        fs[:, 1:2, :, :1],
        fs[:, 1:2, :, 1 : 1 + lc1],
        fs[:, 1:2, :, 1 + lc1 :],
        quantile_dim=-1,
    ).squeeze(1)
    assert torch.allclose(out.m[:, 1, :, :], expected1)


def test_mt_forward_broadcast_q_with_larger_batch():
    """q shape (1, Q) but function_samples has actual batch K > 1."""
    K = 5
    q_broadcast = Q_LEVELS.unsqueeze(0)  # (1, Q)
    raw_scales = torch.zeros(K, Q)
    lik = MultitaskCenterGapQuantileGPLikelihood(
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
    assert torch.allclose(out.m, expected)


def test_mt_forward_ald_kappa_lamda_shapes_no_batch():
    lik = MultitaskCenterGapQuantileGPLikelihood(MT_Q, central_quantile_index=2)
    fs = torch.randn(S, N, Q)
    out = lik.forward(fs)
    assert out.kappa.shape == torch.Size([1, 1, Q])
    assert out.lamda.shape == torch.Size([1, 1, Q])


def test_mt_forward_ald_kappa_lamda_shapes_with_batch():
    B = 3
    q = Q_LEVELS.unsqueeze(0).expand(B, Q).contiguous()
    lik = MultitaskCenterGapQuantileGPLikelihood(q, central_quantile_index=2)
    fs = torch.randn(S, B, N, Q)
    out = lik.forward(fs)
    assert out.kappa.shape == torch.Size([1, B, 1, Q])
    assert out.lamda.shape == torch.Size([1, B, 1, Q])
