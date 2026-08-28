import pytest
import torch

from gpytorch_qr import settings


def test_quantile_gap_lower_bound_default_and_nested_contexts():
    assert settings.quantile_gap_lower_bound.value() == 0.0

    with settings.quantile_gap_lower_bound(0.1):
        assert settings.quantile_gap_lower_bound.value() == 0.1
        with settings.quantile_gap_lower_bound(0.2):
            assert settings.quantile_gap_lower_bound.value() == 0.2
        assert settings.quantile_gap_lower_bound.value() == 0.1

    assert settings.quantile_gap_lower_bound.value() == 0.0


def test_quantile_gap_lower_bound_restores_value_after_exception():
    with pytest.raises(RuntimeError):
        with settings.quantile_gap_lower_bound(0.1):
            raise RuntimeError

    assert settings.quantile_gap_lower_bound.value() == 0.0


@pytest.mark.parametrize("value", [-1.0, float("inf"), float("nan")])
def test_quantile_gap_lower_bound_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="finite and non-negative"):
        settings.quantile_gap_lower_bound(value)


def test_quantile_reconstruction_dtype_default_and_context():
    assert settings.quantile_reconstruction_dtype.value() is None

    with settings.quantile_reconstruction_dtype(torch.float64):
        assert settings.quantile_reconstruction_dtype.value() is torch.float64

    assert settings.quantile_reconstruction_dtype.value() is None


@pytest.mark.parametrize("value", [torch.int64, "float64", 64])
def test_quantile_reconstruction_dtype_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="floating-point"):
        settings.quantile_reconstruction_dtype(value)


def test_enforce_strict_quantile_order_default_and_context():
    assert settings.enforce_strict_quantile_order.value() is False

    with settings.enforce_strict_quantile_order():
        assert settings.enforce_strict_quantile_order.value() is True

    assert settings.enforce_strict_quantile_order.value() is False


def test_enforce_strict_quantile_order_rejects_non_bool():
    with pytest.raises(ValueError, match="bool"):
        settings.enforce_strict_quantile_order(1)
