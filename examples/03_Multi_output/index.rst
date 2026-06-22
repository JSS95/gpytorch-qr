Multi-output GPQR
=================

:math:`y = f(X)` where the dimension of :math:`y` is greater than 1.

Default quantile regression with 1-D :math:`y` is already multi-task, as it models multiple quantiles levels.
Multi-output quantile regression expands this, modeling multiple quantiles of multiple outputs at the same time.

Placing prior mean is difficult for direct GPQR because latents GPs are mixed to form different outputs.
On the other hand, center-gap GPQR more easily supports placing prior means.

.. toctree::
   :maxdepth: 1

   mtgpqr.ipynb
   mtgpqr_cg.ipynb
