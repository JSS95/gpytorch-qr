.. _basic-usage:

Basic Usage
===========

There are two design choices for GPQR:

1. Quantile representation (direct vs. center-gap)
2. Correlation structure (independent vs. correlated)

.. rubric:: Quantile representation

Quantiles can be represented either directly or by a center-gap representation.

In direct representation, each :math:`i`-th quantile function :math:`Q_{\tau_i}(x)` with quantile level :math:`\tau_i` is directly modeled by GP output :math:`f_i(x)`:

.. math::

   Q_{\tau_i}(x) = f_i(x)

This method is more flexible but can suffer from quantile crossing.

In center-gap representation, GP outputs model the central quantile :math:`Q_{\tau_0}(x)` and the gaps between adjacent quantiles :math:`\Delta Q_i(x) > 0`:

.. math::

   Q_{\tau_0}(x) = f_0(x), \quad Q_{\tau_i}(x) = \begin{cases} Q_{\tau_0}(x) + \sum^i_{j=1} \Delta Q_j(x), \quad & i > 0 \\ Q_{\tau_0}(x) - \sum^i_{j=1} \Delta Q_{-j}(x), \quad & i < 0 \end{cases}

where :math:`\Delta Q_j(x) = \log \left(1 + \exp f_j(x) \right)`.
This structure prevents quantile crossing.

.. rubric:: Correlation structure

Correlation structure defines how GP outputs :math:`f_i(x)` are correlated with each other.

Correlation between :math:`f_i(x)` is determined by their relation to latent independent GPs :math:`g_j(x)`.
Independent :math:`f_i(x)` can be implemented by using :class:`IndependentMultitaskVariationalStrategy<gpytorch.variational.IndependentMultitaskVariationalStrategy>`, which is

.. math::
   
   f_i(x) = g_i(x).

Correlated :math:`f_i(x)` can be implemented by using :class:`LMCVariationalStrategy<gpytorch.variational.LMCVariationalStrategy>`, which is

.. math::

   f_i(x) = \sum_j a_{ij} g_j(x),

where :math:`a_{ij}` is a learnable coefficient matrix.

For direct representation, uncorrelated GP output means that :math:`Q_{\tau_i}(x)` are independent.

For center-gap representation, :math:`Q_{\tau_i}(x)` are always correlated by the additive structure.
Here, correlation between :math:`f_i(x)` dictates correlation between gaps :math:`\Delta Q_i(x)`.


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
