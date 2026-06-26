Setting the mean module
=======================

Prior distribution of latent functions can be directly set to :math:`f(x) \sim \mathcal{N}(\mu(x), k(x, x'))`.

This method is more convenient, but it can be inappropriate when latent GP does not directly form output GPs.
The caveat is that :math:`\mu(x)` usually describes :math:`y`, while the prior mean is on the latent GP.
When indirect representation or correlation structure is involved, special care is needed.
Refer to :ref:`basic-usage` for more details on representation and correlation structure.

Direct representation
---------------------

.. toctree::
   :maxdepth: 1

   mtgpqr_independent.ipynb
   mtgpqr.ipynb

Center-gap representation
-------------------------

Directly setting prior mean for each :math:`Q_{\tau_i}(x)` is impossible for center-gap representation.
However, setting informative prior mean only for the central quantile :math:`Q_{\tau_0}(x)` is often enough, as the information can be propagated to other quantiles through the additive structure.
A special mean module :class:`CenterGapMean<gpytorch_qr.means.CenterGapMean>` is provided to support this approach.

If :math:`Q_{\tau_0}(x)` and :math:`\Delta Q_{\tau_i}(x)` are correlated by LMC structure, segregating prior means for central quantile and gaps is generally impossible.
This problem can be circumvented by introducing a special LMC structure that assumes no correlation between :math:`Q_{\tau_0}(x)` and :math:`\Delta Q_{\tau_i}(x)`.
A special variational strategy :class:`CenterGapLMCVariationalStrategy<gpytorch_qr.variational.CenterGapLMCVariationalStrategy>` is provided to support this approach.

.. toctree::
   :maxdepth: 1

   mtgpqr_cg_independent.ipynb
   mtgpqr_cg.ipynb
