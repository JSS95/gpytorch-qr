# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.0] - 2026-08-17

### Added

New ALD class which is more compatible to PyTorch API is introduced.

- `AsymmetricLaplace` is introduced.

New likelihood classes which are more compatiblie to GPyTorch API are introduced.

- `AsymmetricLaplaceLikelihood` is introduced. This class is directly compatible to `GaussianLikelihood`.
- `MultitaskAsymmetricLaplaceLikelihood` is introduced. This class is directly compatible to `MultitaskGaussianLikelihood`.

- `DirectQuantilesLikelihood` is introduced.
- `MultioutputDirectQuantilesLikelihood` is introduced.
- `CenterGapQuantilesLikelihood` is introduced.
- `MultioutputCenterGapQuantilesLikelihood` is introduced.

### Changed

- Signature of `gpytorch_qr.variatonal.CenterGapLMCVariationalStrategy` is changed.
- Example notebooks are changed.

### Removed

- `ALD` and `QuantileALD` are removed. Use `AsymmetricLaplace` instead.

- `DirectQuantileLikelihood` is removed. Use `DirectQuantilesLikelihood` instead.
- `MultiOutputDirectQuantileLikelihood` is removed. Use `MultioutputDirectQuantilesLikelihood` instead.
- `CenterGapQuantileLikelihood` is removed. Use `CenterGapQuantilesLikelihood` instead.
- `MultiOutputCenterGapQuantileLikelihood` is removed. Use `MultioutputCenterGapQuantilesLikelihood` instead.
- `gpytorch_qr.means` module is removed.

### Fixed

Cross validation example now correctly passes `batch_shape` to the likelihood.
