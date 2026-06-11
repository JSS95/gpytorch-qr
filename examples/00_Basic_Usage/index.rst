.. _basic-usage:

Basic Usage
===========

There are two design choices for GPQR:

1. Quantile representation (direct vs. center-gap)
2. Correlation between different latent GPs (independent vs. correlated)

.. rubric:: Representing the quantiles

Quantiles can be represented either directly or by a center-gap representation.

In direct representation, each latent GP models a quantile directly.

In center-gap representation, one latent GP models the central quantile and the rest of the latent GPs model the gaps between adjacent quantiles by softplus transformation.
The quantiles are then additively constructed by cumulatively adding the gaps to the central quantile.

The direct representation is more flexible but can suffer from quantile crossing, while the center-gap representation is more robust and prevents quantile crossing.

.. rubric:: Correlation between different quantiles

If quantiles are directly represented, their correlations can be modeled by LMC structure of the latent GPs.

If center-gap representation is used, quantiles are naturally correlated by the additive structure.
Allowing LMC structure between latent GPs can further model the correlation between gaps.
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
