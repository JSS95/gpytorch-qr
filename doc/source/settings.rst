================
Runtime settings
================

Quantile gap lower bound
------------------------

.. autoclass:: gpytorch_qr.settings.quantile_gap_lower_bound
   :members: value
   :no-index:

For center-gap models, the setting changes each adjacent quantile gap from

.. math::

   \operatorname{softplus}(\mathrm{raw\_gap})

to

.. math::

   \operatorname{softplus}(\mathrm{raw\_gap})
   + \mathrm{lower\_bound}\,\Delta q,

where :math:`\Delta q` is the difference between the corresponding adjacent
quantile levels. The default is ``0.0``, which preserves the original
center-gap transformation.

The setting is read when latent center-gap values are converted to quantiles,
not when the model or likelihood is constructed. Enclose both likelihood
evaluation during training and posterior sampling or summary computation during
prediction:

.. code-block:: python

   import gpytorch_qr

   with gpytorch_qr.settings.quantile_gap_lower_bound(1e-4):
       output = model(train_x)
       loss = -mll(output, train_y)

       posterior = model.joint_quantile_posterior(test_x)
       samples = posterior.sample()
       mean = model.mean_quantiles_delta(test_x)

``CenterGapQuantileGP`` must receive ``quantile_levels`` when a nonzero bound is
used, because posterior conversion needs the adjacent level differences. The
center-gap likelihood already receives quantile levels in its constructor.

Nested contexts are supported and restore the outer value:

.. code-block:: python

   with gpytorch_qr.settings.quantile_gap_lower_bound(1e-4):
       # value is 1e-4
       with gpytorch_qr.settings.quantile_gap_lower_bound(1e-3):
           # value is 1e-3
           pass
       # value is 1e-4 again

Quantile reconstruction dtype
-----------------------------

.. autoclass:: gpytorch_qr.settings.quantile_reconstruction_dtype
   :members: value
   :no-index:

A lower bound can be smaller than the spacing between representable values at
the magnitude of a central quantile. For example, adding a gap of ``1e-5`` to
``1000.0`` in ``float32`` produces ``1000.0`` again. Use
``quantile_reconstruction_dtype`` to perform the center-gap reconstruction in a
higher-precision dtype:

.. code-block:: python

   import torch
   import gpytorch_qr

   with (
       gpytorch_qr.settings.quantile_gap_lower_bound(1e-4),
       gpytorch_qr.settings.quantile_reconstruction_dtype(torch.float64),
   ):
       posterior = model.joint_quantile_posterior(test_x)
       samples = posterior.sample()
       mean = model.mean_quantiles_delta(test_x)

The conversion happens before softplus, cumulative sums, and additions. The
returned quantiles retain the configured dtype; casting them back to a lower
precision can collapse the gaps again. The default value is ``None``, which
preserves the latent input dtype. This setting changes only center-gap
reconstruction precision, not GP covariance computation or posterior sampling
precision.

Strict quantile ordering
------------------------

.. autoclass:: gpytorch_qr.settings.enforce_strict_quantile_order
   :members: value
   :no-index:

``enforce_strict_quantile_order`` applies a final ``torch.nextafter``-based
correction to center-gap model posterior predictions. Every quantile is made at
least the next representable value above its predecessor:

.. code-block:: python

   with gpytorch_qr.settings.enforce_strict_quantile_order():
       quantiles = model.mean_quantiles_delta(test_x)

The correction is applied independently within each output dimension. It does
not affect center-gap likelihood evaluation during training. It guarantees
strict order for finite model predictions, but a corrected gap may be one ULP
rather than ``quantile_gap_lower_bound * delta_q``. Consequently, this setting
is intended for inference and export, not as a differentiable training
constraint. Because the correction is not invertible, transformed-posterior
``log_prob`` and inverse-transform operations are unavailable while it is
enabled.

Strict ordering is guaranteed at the point where GPyTorch-QR returns the model
prediction. Subsequent low-precision operations, such as adding a large
external mean or casting to ``float32``, can collapse gaps again. Apply any such
operations before the final ordering correction, or retain the higher
reconstruction dtype.
