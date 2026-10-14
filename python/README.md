# GpFlash for Python

Fast, exact Gaussian process regression: the marginal log likelihood (MLL) with analytic
hyperparameter gradients, hyperparameter fitting, and posterior mean and variance. Python bindings
to the header-only C++ library [GpFlash](https://github.com/marchildon/gpflash).

```
pip install gpflash
```

Wheels are available for Linux (x86_64, aarch64), macOS (arm64, x86_64) and Windows (x64), on
Python 3.11 to 3.14. They need nothing else installed: BLAS/LAPACK, HDF5, NLopt and the OpenMP
runtime are included.

## Quick start

```python
import numpy as np
import gpflash

rng = np.random.default_rng(0)
X = rng.random((50, 2))                      # (n_points, n_dim)
y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2

gp = gpflash.GaussianProcess("matern_5_2").fit(X, y)
mean, var = gp.predict(rng.random((100, 2)), return_var=True)

print(gp.summary())                          # MLL, optimality, evaluations, bounds hit
print(gp.hyperparameters_)                   # sig2_k, sig2_f, gamma, kern_extra, beta
```

## `GaussianProcess`

| Argument | Meaning |
|---|---|
| `kernel` | `"sq_exp"`, `"matern_5_2"` (default), `"matern_3_2"`, `"rat_quad"`, `"periodic"`, `"linear"`, `"quadratic"` |
| `noise` | `0.0` for noise-free data (default), a known variance, or `"estimate"` to fit it |
| `mean` | `"constant"` (default) or `"zero"`; the weights are solved in closed form |
| `n_starts` | Local optimizations; above 1, the extra starts are Latin-hypercube samples, and the best fit is kept |
| `seed` | Seed for those starts, so a fit is reproducible |

`GaussianProcess.from_options(optz_options, noise=...)` takes an `OptzOptions` instead, for full
control of the fit: bounds, tolerances, the algorithm, symmetry, mean model degrees.

| Method | Returns |
|---|---|
| `fit(X, y)` | `self`; `X` is `(n_points, n_dim)` |
| `predict(X, return_var=False)` | The posterior mean, and the variance when asked |
| `predict_gradient(X)` | The gradients of the mean and variance, each `(n_points, n_dim)` |
| `save(path)` / `GaussianProcess.load(path)` | An HDF5 file the C++ library also reads |
| `summary()` | A short report of the fit |

A noise-free fit adds a small nugget, sized from `ModelOptions.cond_max`, so the factorization is
always well conditioned; predictions at the training points then match the data closely rather
than exactly.

## The C++ interface

Everything in the C++ library is also available under its C++ names: `TrainingData`, `HpVal`,
`ModelOptions`, `OptzOptions`, `eval_mll`, `select_hp_with_max_mll`, `make_posterior`,
`GpPosterior`, `load_posterior` and the rest, so the
[C++ README](https://github.com/marchildon/gpflash#readme) documents this layer too. It keeps the
C++ shapes: points are columns, so `TrainingData.X_ndim_ntrain_mat` is `(n_dim, n_train)`.

```python
train_data = gpflash.TrainingData()
train_data.X_ndim_ntrain_mat = X.T           # (n_dim, n_train)
train_data.f_train_vec = y

hp_val = gpflash.HpVal()
hp_val.hp_gamma_vec = [1.0, 1.0]             # inverse length scales
mll_result = gpflash.eval_mll(train_data, hp_val, gpflash.ModelOptions(), return_mll_gradient=True)
print(mll_result.mll, mll_result.mll_hp_grad_struct.d_mll_d_hp_gamma_vec)
```

Array fields are copied in and out, so assign a whole array to change one:
`hp_val.hp_gamma_vec = new_values` works; `hp_val.hp_gamma_vec[0] = 2.0` changes only a copy.

Invalid inputs raise `gpflash.InvalidInputError`, a `ValueError`. Numerical failures raise
`gpflash.NumericalFailureError`, an `ArithmeticError`.

### Inspecting a fit and a saved file

```python
fit = gp.fit_result_                                   # an MllOptzResult
fit.build_best_local_optz_result().stop_reason         # why the winning solve stopped
fit.calc_hp_at_bound_struct().to_summary_string()      # which hyperparameters a bound holds
gpflash.print_multistart_report(fit)                   # every optimization, where each started and landed

info = gpflash.peek_info("model.h5")                   # the file's header, without loading it
gpflash.print_cache_info(info)                         # kernel, MLL, size, note, writing version
```

`save_multistart_report` writes the report to a file. `GpPosterior.post_model_info()` is the
record of the fit a posterior came from, and `calc_default_cache_path` names a file after the
kernel and the training set's shape. Because `None` is a Python keyword, `SymmetryKind`'s
no-symmetry value is `SymmetryKind.None_`.

The one C++ feature the Python package leaves out is MPI: a fit runs in one process, using its
cores through OpenMP.

## Threads

Fits and queries run in parallel with OpenMP and release the GIL, so other Python threads run
meanwhile. `OMP_NUM_THREADS` sets the thread count. Loading two different OpenMP runtimes into
one process, for example GpFlash's alongside another library's, can slow both down; if that
happens, set `OMP_NUM_THREADS` or import one library at a time.

## Building from source

`pip install .` from a clone of the repository builds the module with CMake. It needs a C++14
compiler and the C++ library's dependencies; the
[C++ README](https://github.com/marchildon/gpflash#building) lists them per platform. On macOS,
point CMake at Homebrew's OpenMP:

```
CMAKE_ARGS="-DOpenMP_ROOT=$(brew --prefix libomp)" pip install ".[test]"
pytest python/tests
```

After changing `python/src/bindings.cpp`, regenerate the type stubs with
`pybind11-stubgen gpflash._core -o python` (it writes `python/gpflash/_core.pyi`), then change the
`__version__` line it writes to `__version__: str`, so the stub carries no copy of the version.

## Citing

If you use GpFlash, please cite the paper,
[arXiv:2610.16861](https://arxiv.org/abs/2610.16861):

```bibtex
@misc{marchildon2026gpflash,
  title         = {GpFlash: Maximizing Speed and Minimizing Memory for Gaussian Process Training and Prediction},
  author        = {Marchildon, Andr{\'e}},
  year          = {2026},
  eprint        = {2610.16861},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2610.16861}
}
```

## License

Apache License 2.0. The wheels also contain compiled third-party libraries under their own
licenses; `LICENSES_bundled.txt`, installed with the package metadata, reproduces them.
