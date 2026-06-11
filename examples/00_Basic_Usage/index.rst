.. _basic-usage:

Basic Usage
===========

There are two design choices for GPQR:

1. Quantile representation (direct vs. center-gap)
2. Correlation between different latent GPs (independent vs. correlated)

.. rubric:: Quantile representation

Quantiles can be represented either directly or by a center-gap representation.

In direct representation, latent GPs directly model quantiles.

In center-gap representation, one latent GP models the central quantile and the rest of the latent GPs model the gaps between adjacent quantiles.
The quantiles are then constructed by cumulatively adding the positive gaps to the central quantile, preventing quantile crossing.

The direct representation is more flexible but can suffer from quantile crossing, while the center-gap representation has more inductive bias and prevents quantile crossing.

.. rubric:: Correlation between different quantiles

If quantiles are directly represented, their correlations can be modeled by LMC structure of the latent GPs.

If center-gap representation is used, quantiles are naturally correlated by the additive structure.
Allowing LMC structure between latent GPs models the correlation between gaps.
For this purpose, we provide special LMC classes.

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
