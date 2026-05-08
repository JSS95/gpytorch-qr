"""Batch independent GPQR with center-gap representation."""

import gpytorch

__all__ = [
    "BatchCenterGapQuantileGP",
    "BatchCenterGapALDLikelihood",
]


class BatchCenterGapQuantileGP(gpytorch.models.ApproximateGP):
    pass


class BatchCenterGapALDLikelihood(gpytorch.likelihoods.Likelihood):
    pass
