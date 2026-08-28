"""Runtime settings for GPyTorch-QR."""

import math

__all__ = ["quantile_gap_lower_bound"]


class _value_context:
    """Base class for settings that temporarily override a global value."""

    _global_value = None

    @classmethod
    def value(cls):
        """Return the currently active value."""
        return cls._global_value

    @classmethod
    def _set_value(cls, value):
        cls._global_value = value

    def __init__(self, value):
        self._original_value = self.__class__.value()
        self._instance_value = value

    def __enter__(self):
        self.__class__._set_value(self._instance_value)

    def __exit__(self, *args):
        self.__class__._set_value(self._original_value)
        return False


class quantile_gap_lower_bound(_value_context):
    r"""Set the minimum center-gap quantile slope.

    Within this context, every adjacent center-gap quantile difference is
    transformed as

    .. math::

        \operatorname{softplus}(\mathrm{raw\_gap})
        + \mathrm{lower\_bound} \, \Delta q.

    The setting is read when center-gap latent values are converted to
    quantiles, so the context must enclose likelihood evaluation and posterior
    prediction. Nested contexts restore the previously active value on exit.

    Parameters
    ----------
    value : float
        A finite, non-negative lower bound. The default outside a context is
        ``0.0``.

    Examples
    --------
    >>> import gpytorch_qr
    >>> with gpytorch_qr.settings.quantile_gap_lower_bound(1e-4):
    ...     gpytorch_qr.settings.quantile_gap_lower_bound.value()
    0.0001
    """

    _global_value = 0.0

    def __init__(self, value):
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                "quantile_gap_lower_bound must be finite and non-negative."
            )
        super().__init__(value)
