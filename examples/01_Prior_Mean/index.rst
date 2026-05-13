Placing the Prior Mean
======================

Prior mean function can be used to provide informative bias to the quantiles.

In this example, the data generating process is

1. Mean function is sinusoidal.
2. Noise is heteroscedastic, with the variance increasing as the input increases.

This process is modeled by prior mean functions where

1. Quantiles have sinusoidal means with different offsets.
2. All input points are used as inducing points, with unwhitened variational inference.

Prediction with informative prior mean is compared to the one with the default (constant) prior mean.

.. toctree::
   :maxdepth: 1

   gpqr.ipynb
   gpqr_cg.ipynb
   mtgpqr.ipynb
   mtgpqr_cg.ipynb
