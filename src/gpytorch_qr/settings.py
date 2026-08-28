"""Runtime settings for GPyTorch-QR."""

import math

import torch

__all__ = [
    "enforce_strict_quantile_order",
    "quantile_gap_lower_bound",
    "quantile_reconstruction_dtype",
]


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


class quantile_reconstruction_dtype(_value_context):
    """Set the dtype used to reconstruct center-gap quantiles.

    The center, raw gaps, and quantile-level offsets are converted before the
    softplus, cumulative-sum, and addition operations. The reconstructed
    quantiles retain the selected dtype. This prevents a small gap from being
    rounded away when it is added to a large central quantile.

    Parameters
    ----------
    dtype : torch.dtype or None
        A real floating-point dtype. ``None`` preserves the dtype of the latent
        center and gaps and is the default.
    """

    _global_value = None

    def __init__(self, dtype):
        if dtype is not None and (
            not isinstance(dtype, torch.dtype) or not dtype.is_floating_point
        ):
            raise ValueError(
                "quantile_reconstruction_dtype must be a real floating-point "
                "torch.dtype or None."
            )
        super().__init__(dtype)


class enforce_strict_quantile_order(_value_context):
    """Enforce representably distinct center-gap quantile predictions.

    When enabled, each predicted quantile is at least the next representable
    floating-point value above the preceding quantile. The correction is
    applied independently to each output dimension and only to model posterior
    predictions, not to the training likelihood.

    This is an inference-oriented, dtype-dependent correction. It guarantees
    strict order for finite predictions but does not preserve a requested
    mathematical gap when that gap is smaller than one ULP.

    Parameters
    ----------
    state : bool, default=True
        Whether to enable strict quantile ordering.
    """

    _global_value = False

    def __init__(self, state=True):
        if not isinstance(state, bool):
            raise ValueError("enforce_strict_quantile_order must be a bool.")
        super().__init__(state)
