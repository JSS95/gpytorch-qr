Placing the Prior Mean
======================

Prior mean function can be used to provide informative bias to the quantiles.

In this example, the data generating process is

1. Mean function is sinusoidal.
2. Noise is heteroscedastic, with the variance increasing as the input increases.

This process is modeled by prior mean functions where

1. Quantiles have sinusoidal means with different offsets.
2. All input points are used as inducing points, with unwhitened variational inference.

To show how informative bias by prior mean enhances the performance under data-deficient regimes, a relatively small number of sample data are used.
Compare the following four notebooks to see its effect on different architectures.
It can be observed that independent GPQR easily overfits and numerically susceptible, while multi-task correlation stabilizes the model.

.. toctree::
   :maxdepth: 1

   gpqr.ipynb
   gpqr_cg.ipynb
   mtgpqr.ipynb
   mtgpqr_cg.ipynb
