# Changelog

Every release of GpFlash and what it changed. Version numbers follow
[semantic versioning](https://semver.org); before 1.0, a minor release may change the interface.

## 0.2.0 (2026-10-17)

### Added

- **Python package**, `pip install gpflash`, with wheels for Linux (x86_64, aarch64), macOS
  (arm64, x86_64) and Windows (x64) on Python 3.11 to 3.14, every dependency included. It offers
  a NumPy-style `GaussianProcess` (`fit`, `predict`, `predict_gradient`, `save`, `load`) and the
  C++ interface under its C++ names. See [`python/README.md`](python/README.md).
- CMake option `GPFLASH_BUILD_PYTHON`, off by default, to build the Python module.
- Citation information for the GpFlash paper, [arXiv:2610.16861](https://arxiv.org/abs/2610.16861):
  `CITATION.cff`, which gives the repository a "Cite this repository" button, and a BibTeX
  entry in [`python/README.md`](python/README.md).
- Continuous integration: the C++ tests on Linux and macOS, and the Python wheels on every
  platform, after each push to `main`.

### Fixed

- A cache save that fails part way, as on a full disk, no longer leaves its file open in HDF5. HDF5 1.10
  cannot close it, and the process may crash as it exits.

### Changed

- The version is set in one place, `include/gpflash/Version.hpp`; CMake and the Python package
  read it from there.

## 0.1.0 (2026-10-12)

First release: a header-only C++14 library for exact Gaussian process regression, with the
marginal log likelihood and its analytic hyperparameter gradients, hyperparameter fitting
through NLopt, posterior mean and variance with their gradients, and an HDF5 posterior cache.
