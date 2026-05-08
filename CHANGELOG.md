# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## UNRELEASED

### Added

- `gpytorch_qr.mtgpqr` is added for multi-task GPQR.

### Changed

- `gpytorch_qr.gpqr.ALD()` is moved to `gpytorch_qr.ald.BatchALD()`.
- `gpytorch_qr.gpqr.ALDLikelihood()` is renamed to `gpytorch_qr.gpqr.BatchALDLikelihood()`.

## [0.2.0] - 2026-05-08

### Added

- `gpytorch_qr.gpqr` is added for batch independent GPQR.
- `gpytorch_qr.mtgpqr_cg` is added for multi-task GPQR with center-gap representation.

### Fixed

- `CenterGapLikelihood` now works when input `tau` is on GPU.

## [0.1.0] - 2026-03-28

### Added

- Basic functions and classes for center-gap representation.
