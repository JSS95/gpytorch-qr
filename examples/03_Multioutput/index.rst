Multi-output quantile regression
================================

Multi-output quantile regression is supported.

Default quantile regression is already multi-output, as it models multiple quantiles.
Multi-output quantile regression means that multiple variables are modeled at the same time, along with multiple quantiles of each variable.

Multiple outputs can be implemented by either batch GPQR or multitask GPQR.

- Use batch GPQR when the outputs are independent, e.g., cross validation.
- Use multitask GPQR when the outputs are correlated, e.g., modeling vector-valued functions.

.. toctree::
   :maxdepth: 1

   cross_validation.ipynb
