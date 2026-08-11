# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.0rc0] - 2026-08-12

### Changed

Example notebooks are changed.

## [0.9.0a0] - 2026-08-11

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

### Removed

- `ALD` and `QuantileALD` are removed. Use `AsymmetricLaplace` instead.

- `DirectQuantileLikelihood` is removed. Use `DirectQuantilesLikelihood` instead.
- `MultiOutputDirectQuantileLikelihood` is removed. Use `MultioutputDirectQuantilesLikelihood` instead.
- `CenterGapQuantileLikelihood` is removed. Use `CenterGapQuantilesLikelihood` instead.
- `MultiOutputCenterGapQuantileLikelihood` is removed. Use `MultioutputCenterGapQuantilesLikelihood` instead.

## [0.9.0.dev4] - 2026-08-10

### Removed

- `gpytorch_qr.means` module is removed.
