import pytest
import torch

from gpytorch_qr import settings
from gpytorch_qr.utils import CenterGapToQuantileTransform, centergap_to_quantiles


def test_quantile_reconstruction_dtype_prevents_float32_gap_collapse():
    central = torch.tensor([1000.0], dtype=torch.float32)
    lower_gaps = torch.empty(0, dtype=torch.float32)
    upper_gaps = torch.tensor([-100.0], dtype=torch.float32)
    quantile_level_offsets = torch.tensor([0.0, 0.1], dtype=torch.float32)

    with settings.quantile_gap_lower_bound(1e-4):
        float32_quantiles = centergap_to_quantiles(
            central,
            lower_gaps,
            upper_gaps,
            quantile_level_offsets,
        )
        with settings.quantile_reconstruction_dtype(torch.float64):
            float64_quantiles = centergap_to_quantiles(
                central,
                lower_gaps,
                upper_gaps,
                quantile_level_offsets,
            )

    assert float32_quantiles.dtype is torch.float32
    assert float32_quantiles[0] == float32_quantiles[1]
    assert float64_quantiles.dtype is torch.float64
    assert float64_quantiles[1] > float64_quantiles[0]
    assert torch.allclose(
        float64_quantiles.diff(),
        torch.tensor([1e-5], dtype=torch.float64),
        rtol=1e-5,
    )


def test_strict_quantile_order_uses_next_representable_values():
    transform = CenterGapToQuantileTransform([3], [1])
    center_and_raw_gaps = torch.tensor(
        [1000.0, -1000.0, -1000.0],
        dtype=torch.float32,
    )

    with settings.enforce_strict_quantile_order():
        quantiles = transform(center_and_raw_gaps)

    assert (quantiles.diff() > 0).all()
    assert quantiles[1] == torch.nextafter(quantiles[0], torch.tensor(float("inf")))
    assert quantiles[2] == torch.nextafter(quantiles[1], torch.tensor(float("inf")))

    with settings.enforce_strict_quantile_order():
        with pytest.raises(NotImplementedError, match="not invertible"):
            transform.inv(quantiles)


def test_strict_quantile_order_is_independent_per_output():
    transform = CenterGapToQuantileTransform([2, 2], [0, 0])
    center_and_raw_gaps = torch.tensor(
        [1000.0, -1000.0, -1000.0, -1000.0],
        dtype=torch.float32,
    )

    with settings.enforce_strict_quantile_order():
        quantiles = transform(center_and_raw_gaps)

    assert (quantiles[:2].diff() > 0).all()
    assert (quantiles[2:].diff() > 0).all()
    assert quantiles[2] == -1000.0
