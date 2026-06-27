Placing the Prior Mean
======================

Prior mean function can be used to provide informative bias to quantiles, which is useful when the data is scarse.

There are two ways to place the prior mean :math:`\mu(x)`.
The first is subtracting the prior mean from the input data :math:`y`.
The second is setting the mean module to the prior distribution itself.

.. toctree::
   :glob:
   :maxdepth: 2

   ./subtract_mean/index
   ./mean_module/index
