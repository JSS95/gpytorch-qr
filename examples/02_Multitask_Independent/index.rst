Multitask Independent GPQR
==========================

Multitask independent GPQR is useful when you want to model multiple quantiles independently, but your downstream task requires multitask GP structure.

Multitask GPQR (:mod:`gpytorch_qr.mtgpqr` and :mod:`gpytorch_qr.mtgpqr_cg`) models correlation between latent GP by linear model of corregionalization (LMC).
Should the latent GPs be independent, one can use :class:`gpytorch.variational.IndependentMultitaskVariationalStrategy`.
This leads to results equivalent to those from batch independent GPQR (:mod:`gpytorch_qr.gpqr` and :mod:`gpytorch_qr.gpqr_cg`), but with tensor shapes compatible to multitask GP.

This section contains two examples of multitask independent GPQR: one by direct quantiles and the other by center-gap representation.
Compare these examples with the corresponding batch independent GPQR examples in :ref:`basic-usage`.
