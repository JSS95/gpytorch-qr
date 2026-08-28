import pytest

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
