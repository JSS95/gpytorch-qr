.. _basic-usage:

Basic Usage
===========

There are several design choices for GPQR:

1. Output structure (batch vs multitask)
2. Correlation between different quantiles (independent vs. correlated)
3. Representing the quantiles (direct vs. center-gap)

The recommended choice is:

- Use :ref:`mtgpqr_cg_example` when multiple quantiles should be modeled without quantile crossing.
- Use :ref:`gpqr_example` when only one quantile is modeled or quantile crossing is not a concern.

.. rubric:: Output structure

Output structure determines whether different levels of quantiles are modeled as batches or as multitasks.
This design choice affects both the output tensor shape and correlation structure between quantiles.

.. rubric:: Correlation between different quantiles

When the quantiles are modeled as batches, they are independent by design.
When the quantiles are modeled as multitasks, they can be either independent or correlated.

Multitask structure learns correlation between multiple outputs (=quantiles) by linear model of coregionalization (LMC), which models the outputs as a linear combination of latent GPs.
For center-gap representation, we use a special LMC structure which will be described in this page.

.. rubric:: Representing the quantiles

Quantiles can be represented either directly or by a center-gap representation.

In direct representation, each latent GP models a quantile directly.

In center-gap representation, one latent GP models the central quantile and the rest of the latent GPs model the gaps between adjacent quantiles by softplus transformation.
The quantiles are then additively constructed by cumulatively adding the gaps to the central quantile.

The direct representation is more flexible but can suffer from quantile crossing, while the center-gap representation is more robust and prevents quantile crossing by design.

.. toctree::
   :maxdepth: 1
   :caption: Independent quantiles

   gpqr.ipynb
   gpqr_cg.ipynb
   mtgpqr_independent.ipynb
   mtgpqr_cg_independent.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Correlated quantiles

   mtgpqr.ipynb
   mtgpqr_cg.ipynb
