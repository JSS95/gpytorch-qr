# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `DirectQuantilesLikelihood` is introduced. This class is more compatible to GPyTorch.
- `MultioutputDirectQuantilesLikelihood` is introduced. This class is more compatible to GPyTorch.

### Changed

- Signature of `gpytorch_qr.variatonal.CenterGapLMCVariationalStrategy` is changed.

### Removed

- `DirectQuantileLikelihood` is removed. Use `DirectQuantilesLikelihood` instead.
- `MultiOutputDirectQuantileLikelihood` is removed. Use `MultioutputDirectQuantilesLikelihood` instead.

## [0.9.0.dev4] - 2026-08-10

### Removed

- `gpytorch_qr.means` module is removed.
