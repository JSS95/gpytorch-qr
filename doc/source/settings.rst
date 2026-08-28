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
