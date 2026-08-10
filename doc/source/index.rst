.. GPyTorch-QR documentation master file, created by
   sphinx-quickstart on Wed Mar 25 13:52:33 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

*************************
GPyTorch-QR documentation
*************************

GPyTorch-QR is a Python package for quantile regression using GPyTorch.

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

For direct representation, correlation structure dictates the correlation of :math:`Q_{\tau_i}(x)`.

For center-gap representation, :math:`Q_{\tau_i}(x)` are always correlated by the additive structure.
Here, correlation between :math:`f_i(x)` dictates correlation between gaps :math:`\Delta Q_i(x)`.

.. toctree::
   :maxdepth: 2

   examples/tutorial
   examples/index
   reference/index

==================
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
