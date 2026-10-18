# GpFlash for Python

Fast, exact Gaussian process regression: the marginal log likelihood (MLL) with analytic
hyperparameter gradients, hyperparameter fitting, and posterior mean and variance. Python bindings
to the header-only C++ library [GpFlash](https://github.com/marchildon/gpflash).

```
pip install gpflash
```

Wheels for Linux, macOS and Windows need nothing else installed: BLAS/LAPACK, HDF5, NLopt and the
OpenMP runtime are included.

## Quick start

```python
import numpy as np
import gpflash

n_dim = 2
n_train = 50
n_post = 100

# Training data: n_train points, one per row, and the function's values there.
rng = np.random.default_rng(0)
X_train_mat = rng.random((n_train, n_dim))
f_train_vec = np.sin(3 * X_train_mat[:, 0]) + X_train_mat[:, 1] ** 2

gp = gpflash.GaussianProcess().fit(X_train_mat, f_train_vec)  # the Matern 5/2 kernel and a constant mean

# The posterior mean and variance at n_post new points.
X_post_mat = rng.random((n_post, n_dim))
posterior_prediction = gp.predict(X_post_mat)
mean_post_vec = posterior_prediction.mean_post_vec  # (n_post)
var_post_vec = posterior_prediction.var_post_vec  # (n_post)

print(gp.to_summary_string())  # MLL, optimality, evaluations, bounds hit
print(gp.hp_val_struct())  # hp_sig2_k, hp_sig2_f_val, hp_gamma_vec, hp_kern_extra_vec
```

## `GaussianProcess`

The options are C++'s `OptzOptions`, passed to the constructor; the data, with its noise
variance, go to `fit`. `GaussianProcess()` uses the defaults: the Matern 5/2 kernel, a constant
mean and one optimization. Any other choice is a field of `OptzOptions`, under its C++ name:

```python
optz_opts = gpflash.OptzOptions()
optz_opts.model_options.kernel_id = gpflash.KernelId.RatQuad  # or gpflash.parse_kernel_id("rat_quad")
optz_opts.model_options.mean_model_degree_vec = [1]  # a linear mean model
optz_opts.use_multistart = True  # 5 optimizations, the best kept
optz_opts.n_ind_optz = 5

gp = gpflash.GaussianProcess(optz_opts).fit(X_train_mat, f_train_vec, sig2_f_val=-1.0)  # estimate the noise
```

The [kernel table](https://github.com/marchildon/gpflash#kernels) lists the kernels.
[`OptzOptions.hpp`](https://github.com/marchildon/gpflash/blob/main/include/gpflash/optimization/search_space/OptzOptions.hpp)
and [`ModelOptions.hpp`](https://github.com/marchildon/gpflash/blob/main/include/gpflash/inputs/ModelOptions.hpp)
list every other option with its default: bounds, tolerances, the algorithm, the conditioning
limit, symmetry. The constructor copies the options, so they are set once;
`gp.optz_opts()` returns a copy. Array fields are copied in and out too, so assign a whole array:
`optz_opts.hp_gamma_ub_vec = [100.0]` works; `optz_opts.hp_gamma_ub_vec[0] = 100.0` changes only
a copy. Every points argument is `(n_points, n_dim)`, one point per row,
and must be 2-D: a 1-D problem's points are `(n_points, 1)`, a single point is `(1, n_dim)`.

| Method | Returns |
|---|---|
| `fit(X_train_mat, f_train_vec, sig2_f_val=0.0)` | `self`. `sig2_f_val` is the noise variance on the values: `0.0` for noise-free data, a known variance > 0, or a negative value to estimate it |
| `predict(X_post_mat, return_mean=True, return_var=True, return_mean_gradient=False, return_var_gradient=False)` | A `PosteriorPrediction` with `mean_post_vec`, `var_post_vec`, `d_mean_post_d_x_post_mat` and `d_var_post_d_x_post_mat`; the gradients are `(n_post, n_dim)`, and a field not asked for is `None`. The variance costs `O(n_train^2)` per point against `O(n_train)` for the mean |
| `eval_mll(X_train_mat, f_train_vec, hp_val_struct, return_mll_gradient=False, sig2_f_val=0.0)` | The MLL at given hyperparameters, an `MllResult` |
| `make_posterior(X_train_mat, f_train_vec, hp_val_struct, sig2_f_val=0.0)` | `self`, with a posterior at given hyperparameters, without fitting |
| `hp_val_struct()`, `hp_beta_vec()`, `mll()`, `kernel_id()`, `optz_opts()` | The hyperparameters, the mean model weights, the MLL, the kernel and the options |
| `posterior()`, `mll_optz_result()` | The `GpPosterior` that `predict` evaluates, and the fit's `MllOptzResult`, for the C++ interface below |
| `to_summary_string()` | A short report of the fit |
| `save_precompute(path)` / `GaussianProcess.load_posterior(path)` | An HDF5 file the C++ library also reads; a loaded model predicts but holds no fit |
| `load_posterior_verified(path, X_train_mat, f_train_vec, hp_val_struct, sig2_f_val=0.0)` | `self`, loaded from a file only if these data and hyperparameters produced it |

Invalid inputs raise `gpflash.InvalidInputError`, a
`ValueError`, from either layer.

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
# The quick start's training data, transposed to the C++ layout.
train_data = gpflash.TrainingData()
train_data.X_ndim_ntrain_mat = X_train_mat.T  # (n_dim, n_train)
train_data.f_train_vec = f_train_vec

hp_val_struct = gpflash.HpVal()
hp_val_struct.hp_gamma_vec = [1.0, 1.0]  # inverse length scales
mll_result = gpflash.eval_mll(train_data, hp_val_struct, gpflash.ModelOptions(), return_mll_gradient=True)
print(mll_result.mll, mll_result.mll_hp_grad_struct.d_mll_d_hp_gamma_vec)
```

Numerical failures raise `gpflash.NumericalFailureError`, an `ArithmeticError`. The enums
(`KernelId`, `OptzStopReason`, ...) are Python `enum.Enum` subclasses.

### Inspecting a fit and a saved file

```python
mll_optz_result = gp.mll_optz_result()
mll_optz_result.build_best_local_optz_result().stop_reason  # why the winning solve stopped
mll_optz_result.calc_hp_at_bound_struct().to_summary_string()  # which hyperparameters a bound holds
gpflash.print_multistart_report(mll_optz_result)  # every optimization, where each started and landed

gp.save_precompute("model.h5")  # the posterior, as an HDF5 file
cache_info = gpflash.peek_info("model.h5")  # the file's header, without loading it
gpflash.print_cache_info(cache_info)  # kernel, MLL, size, note, writing version
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

`ruff check .` and `ruff format .`, with the version `.github/workflows/ci.yml` pins, keep the Python
files and the READMEs' Python blocks in the style CI checks. After changing `python/src/bindings.cpp`, regenerate the type stubs with
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
