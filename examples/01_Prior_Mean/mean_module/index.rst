Setting the mean module
=======================

Prior distribution of latent functions can be directly set to :math:`f(x) \sim \mathcal{N}(\mu(x), k(x, x'))`.

This method is more convenient, but it can be inappropriate when latent GP does not directly form output GPs.
The caveat is that :math:`\mu(x)` usually describes :math:`y`, while the prior mean is on the latent GP :math:`f(x)`.
When output GPs are independent, i.e., each latent GP directly forms output GPs, this is not a problem.
But when output GPs are correlated by LMC structure, special care is needed.
Refer to :ref:`basic-usage` for more details on correlation structure.

Direct representation
---------------------

.. toctree::
   :maxdepth: 1

   mtgpqr_independent.ipynb
   mtgpqr.ipynb

Center-gap representation
-------------------------

To support this method for center-gap representation with correlated gaps, we provide a special variational strategy :class:`CenterGapLMCVariationalStrategy<gpytorch_qr.variational.CenterGapLMCVariationalStrategy>`.

.. toctree::
   :maxdepth: 1

   mtgpqr_cg_independent.ipynb
   mtgpqr_cg.ipynb
