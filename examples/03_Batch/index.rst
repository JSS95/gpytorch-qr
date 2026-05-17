Batch GPQR
==========

Additional batches can be added to the GPQR model.

Note that at least one batch dimension always exists to model multiple quantiles.
For batch GPQR, quantiles are directly included in the batch dimension.
For multitask GPQR, latents are in the batch dimension, and quantiles are in the event dimension.

.. toctree::
   :maxdepth: 1

   cross_validation.ipynb
