Center-gap GPQR
===============

Center-gap representation models the central quantile and gaps between adjacent
quantiles, then additively constructs the quantiles afterwards.
This representation structurally forbids quantile crossing.

Useful when:
- Quantile crossing is a concern.
- Full conditional distribution needs to be approximated.

Multitask
---------

.. automodule:: gpytorch_qr.mtgpqr_cg
    :members:

Batch Independent
-----------------

.. automodule:: gpytorch_qr.gpqr_cg
    :members:
