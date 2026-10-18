# Changelog

Every release of GpFlash and what it changed. Version numbers follow
[semantic versioning](https://semver.org); before 1.0, a minor release may change the interface.

## 0.3.0 (2026-10-18)

### Changed

- **Python:** `GaussianProcess` now uses the C++ library's names, which breaks code written for 0.2.0:

  | 0.2.0 | Now |
  |---|---|
  | `GaussianProcess(kernel=..., mean=..., n_starts=..., seed=...)`, `from_options` | `GaussianProcess(optz_opts)`, an `OptzOptions` whose fields carry the C++ names |
  | `noise=...` in the constructor; `"estimate"` | `fit(..., sig2_f_val=...)`; a negative value estimates it |
  | `predict(X)` returns arrays; `predict_gradient(X)` | `predict(X)` returns a `PosteriorPrediction`; flags select the mean, variance and gradients, and unrequested fields are `None` |
  | `save` / `load`, `include_variance` | `save_precompute` / `load_posterior`, `include_variance_data` |
  | `summary()`, `fit_result_`, `posterior_` | `to_summary_string()`, `mll_optz_result()`, `posterior()` |
  | `hyperparameters_` (a dict), `log_marginal_likelihood_`, `kernel` | `hp_val_struct()` (an `HpVal`), `hp_beta_vec()`, `mll()`, `kernel_id()` |

- **Python:** `predict` returns the variance by default, as C++'s `PosteriorOptions` does.
- **Python:** points must be 2-D, `(n_points, n_dim)`; a 1-D array is no longer read as one point.
- **Python:** the wrapper's input checks raise `InvalidInputError`, as the C++ layer does.
- **Python:** the enums (`KernelId`, `OptzStopReason`, ...) are `enum.Enum` subclasses.

### Added

- **Python:** `GaussianProcess.eval_mll`, `make_posterior`, `load_posterior_verified` and
  `optz_opts()`. `HpVal`, `TrainingData`, `MllResult` and `PosteriorModelInfo` print their fields.
- Continuous integration: a weekly run of both workflows, and a ruff check of the Python files.

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
