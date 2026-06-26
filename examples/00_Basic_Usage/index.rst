.. _basic-usage:

Basic Usage
===========

There are two design choices for GPQR:

1. Quantile representation (direct vs. center-gap)
2. Correlation structure (independent vs. correlated)

.. rubric:: Quantile representation

Quantiles can be represented either directly or by a center-gap representation.

In direct representation, latent GPs directly model quantiles.
This method is more flexible but can suffer from quantile crossing.

In center-gap representation, latent GPs model the central quantile and the gaps between adjacent quantiles.
The quantiles are then constructed by cumulatively adding the positive gaps to the central quantile, preventing quantile crossing.

.. rubric:: Correlation structure

If quantiles are directly represented, their correlations can be modeled by LMC structure of the latent GPs.

If center-gap representation is used, quantiles are naturally correlated by the additive structure.
Allowing LMC structure between latent GPs additionally models the correlation between gaps.

.. toctree::
   :maxdepth: 1
   :caption: Independent GPs

   mtgpqr_independent.ipynb
   mtgpqr_cg_independent.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Correlated GPs

   mtgpqr.ipynb
   mtgpqr_cg.ipynb
