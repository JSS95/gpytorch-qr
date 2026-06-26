Subtracting the mean
====================

Model can learn the quantiles of the residuals :math:`y - \mu(x)`.
After inference, the prior mean can be added back to the predicted quantiles.

This method is more fundamental and it is useful when complex correlation structure between latent GPs is used.
The disadvantage is that it requires additional preprocessing and postprocessing steps.

.. toctree::
   :maxdepth: 1

   mtgpqr.ipynb
   mtgpqr_cg.ipynb
