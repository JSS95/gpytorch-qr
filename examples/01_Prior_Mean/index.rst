Placing the Prior Mean
======================

Prior mean function can be used to provide informative bias to the quantiles.

In this example, the data generating process is

1. Mean function is sinusoidal.
2. Noise is heteroscedastic, with the variance increasing as the input increases.
3. A relatively small number of sample data are generated, to simulate data-deficient scenarios.

The process is modeled by GPQR where

1. Prior means of quantiles are sinusoidal functions with different offsets.
2. All input points are used as inducing points, with unwhitened variational inference.

Compare the following four notebooks to see the performance of each architecture.
It can be observed that only the multitask correlated center-gap GPQR shows robust performance.

.. toctree::
   :maxdepth: 1
   :caption: Successful architecture

   mtgpqr_cg.ipynb

Other GPQRs are uncessful.
Independent GPQR is numerically unstable, and direct regression without center-gap representation leads to collapsing quantiles.

.. toctree::
   :maxdepth: 1
   :caption: Unsuccessful architecture

   gpqr.ipynb
   gpqr_cg.ipynb
   mtgpqr.ipynb
