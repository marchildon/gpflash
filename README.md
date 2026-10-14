# GpFlash

Header-only C++ Gaussian process library: marginal log-likelihood (MLL) with analytic
hyperparameter gradients, posterior mean and variance, and hyperparameter fitting. Dense and
exact — see [Assumptions and limitations](#assumptions-and-limitations).

The algorithms are those of **"GpFlash: Maximizing Speed and Minimizing Memory for
Gaussian Process Training and Prediction"**, A. Marchildon
([arXiv:2610.16861](https://arxiv.org/abs/2610.16861)) — referred to throughout as
**the GpFlash paper**. It derives the model, the marginal log likelihood and every
gradient expression, which this README does not restate. If you use GpFlash, please cite
it; GitHub's "Cite this repository" button, from `CITATION.cff`, gives the reference.

## Requirements

C++14, Armadillo 9.870 or later, a BLAS/LAPACK with the standard 32-bit integer interface (Accelerate on macOS and
OpenBLAS elsewhere by default; not an ILP64 build such as openblas64), OpenMP, HDF5 for the
[posterior cache](#posterior-caching), and [NLopt](https://github.com/stevengj/nlopt), which only the optimizer calls
([Two build configurations](#two-build-configurations)). Building the tests needs CMake 3.22 or later.
[Building](#building) shows how a consumer's CMake project links the headers.

## Kernels

`ModelOptions::kernel_id` selects one. Every kernel fits `hp_gamma`, one per dimension: an inverse
length scale, or for `Periodic` an inverse period. `hp_kern_extra` holds whatever else it
declares. `ModelOptions::symmetry_spec` gives a kernel that supports it an even or odd mirror
symmetry about an axis-aligned plane.

| `KernelId` | kernel | type | `hp_kern_extra` | symmetry |
|---|---|---|---|---|
| `SqExp` | squared exponential | radial | none | yes |
| `Matern52` | Matérn 5/2, the default | radial | none | yes |
| `Matern32` | Matérn 3/2 | radial | none | yes |
| `RatQuad` | rational quadratic | radial | the shape `hp_rho` | yes |
| `Periodic` | periodic | stationary | one amplitude per dimension | yes |
| `Linear` | inhomogeneous linear | inner product | none | no |
| `Quadratic` | inhomogeneous quadratic | inner product | none | no |

Radial kernels are functions of the scaled distance alone. An inner-product kernel's diagonal
varies with the point. Adding a kernel means one header under `kernels/`, one entry in
`GPFLASH_KERNEL_LIST` (`inputs/KernelId.hpp`), and its include in `kernels/KernelDispatch.hpp`.

## Naming convention

- `n_` prefix for a count (`n_train`, `n_dim`, `n_hp`), `_vec` for a vector, `_mat` for a matrix.
- `hp_` abbreviates hyperparameter, and marks one the fit selects by maximizing the MLL.
- `_train` for training points and the values observed there: `X_ndim_ntrain_mat`, `f_train_vec`.
- `_post` for posterior evaluation points and outputs: `X_ndim_npost_mat`, `mean_post_vec`.
- A matrix's name carries its shape: `X_ndim_ntrain_mat` is `(n_dim, n_train)`, one point per column.
- Derivatives are `d_<numerator>_d_<denominator>`: `d_mll_d_hp_gamma_vec`.

## Model conventions

```
Sigma = hp_sig2_k (K + eta W) + sig2_f_val I,   W = diag(K), which is I when K has a unit diagonal
```

- The prior mean is `basis(x)^T hp_beta_vec`. `ModelOptions::has_mean_model_constant` and
  `mean_model_degree_vec` choose the mean model basis `basis(x)`, defined in `mean/MeanModel.hpp`;
  the default is the constant mean model. The mean model weights `hp_beta_vec` are solved in closed
  form.
- `hp_sig2_k` is the signal variance, `sig2_f_val` the noise variance on the function
  evaluations, and `hp_sig2_f_val` the fit's estimate of `sig2_f_val` for when it is unknown.

The sign of `sig2_f_val` decides which variances a fit searches; every gradient is analytic:

| `sig2_f_val` | noise variance | `hp_sig2_k` |
|---|---|---|
| `0` (default) | none | closed form |
| `> 0` | `sig2_f_val`, as given | searched |
| `< 0` | `hp_sig2_f_val`, searched | searched |

### Addressing the ill-conditioning problem

A nugget `eta` on the kernel diagonal guarantees `kappa(K + eta I) <= cond_max` (default `1e10`)
for a unit-diagonal `K`, and so `kappa(Sigma) <= cond_max`, rather than merely improving the
conditioning. `ModelOptions::kmat_eig_max_ub_method` picks the bound on `lambda_max` it is sized
from. The default, Gershgorin's, reads the entries of `K`, so `eta` depends on the
hyperparameters, and its derivative is in the gradient. The trace bound, `n_train`, is constant
and never tighter.

This is the method of
[Marchildon & Zingg, *Int. J. Numer. Methods Eng.* (2024), doi:10.1002/nme.7498](https://onlinelibrary.wiley.com/doi/10.1002/nme.7498)
([preprint](https://arxiv.org/abs/2307.05855)). A stationary `K` already has a unit diagonal;
for a kernel whose diagonal varies, `K` is first scaled to a unit diagonal, the nugget is added
there, and the factor is scaled back (`preconditioning/ConditionedCholesky.hpp`).

## Usage

Three inputs, separated by how often they change: the **data** (`TrainingData`) is fixed for a
fit, the **hyperparameters** (`HpVal`) change every objective evaluation, and the **options**
(`ModelOptions`) are set once. All three are in `inputs/`; a fit takes its `ModelOptions` inside
`OptzOptions` (`optimization/search_space/`).

```cpp
#include "gpflash/optimization/HpOptimizer.hpp"       // select_hp_with_max_mll, make_posterior_from_fit
#include "gpflash/training/EvalMllDispatch.hpp"       // eval_mll

const int n_dim = 2, n_train = 50;

gpflash::TrainingData train_data;
train_data.X_ndim_ntrain_mat = arma::randu<arma::mat>(n_dim, n_train);  // one training point per column
train_data.f_train_vec = arma::randu<arma::vec>(n_train);               // evaluations at those points
train_data.sig2_f_val = 0.0;                                            // noise variance; 0 = noise-free
```

### 1. Evaluate the MLL and its gradient

```cpp
gpflash::HpVal hp_val_struct;
hp_val_struct.hp_gamma_vec = arma::vec(n_dim, arma::fill::ones);  // inverse length scales

gpflash::ModelOptions model_opts;  // Matern52 and the constant mean model by default
gpflash::MllResult mll_result = gpflash::eval_mll(train_data, hp_val_struct, model_opts, /*return_mll_gradient=*/true);
// mll_result.mll, mll_result.mll_hp_grad_struct.d_mll_d_hp_gamma_vec, and, with sig2_f_val = 0, mll_result.hp_sig2_k

// Another kernel. RatQuad reads its shape from hp_kern_extra_vec.
model_opts.kernel_id = gpflash::kernel::KernelId::RatQuad;
hp_val_struct.hp_kern_extra_vec = {2.0};
mll_result = gpflash::eval_mll(train_data, hp_val_struct, model_opts);
```

### 2. Fit the hyperparameters

`select_hp_with_max_mll` maximizes the MLL with the analytic gradient, in log space, within
box bounds. It runs one optimization by default; `OptzOptions::use_multistart` adds
Latin-hypercube starts up to `n_ind_optz` and keeps the best.
[`optimization/README.md`](include/gpflash/optimization/README.md) covers the fit in detail.

```cpp
gpflash::OptzOptions optz_opts;
optz_opts.model_options = model_opts;
gpflash::MllOptzResult mll_optz_result = gpflash::select_hp_with_max_mll(train_data, optz_opts);
// mll_optz_result.hp_val_struct holds every fitted hyperparameter; to_summary_string(mll_optz_result) reports the fit
```

**Noise.** `TrainingData::sig2_f_val` is read by sign, as [Model conventions](#model-conventions)
tabulates. A negative value asks the fit to estimate the noise, returned in
`mll_optz_result.hp_val_struct.hp_sig2_f_val`.

### 3. Evaluate the posterior

```cpp
// The fit's own options, so the posterior's kernel and nugget match the fit's.
gpflash::GpPosterior posterior = gpflash::make_posterior_from_fit(train_data, mll_optz_result);

arma::mat X_ndim_npost_mat = arma::randu<arma::mat>(n_dim, 100);  // one query point per column
gpflash::PosteriorResult posterior_result = posterior.eval(X_ndim_npost_mat);
// posterior_result.mean_post_vec, posterior_result.var_post_vec; PosteriorOptions asks for gradients
```

`make_posterior(train_data, hp_val_struct, model_opts)` builds one from hyperparameters held
directly, and `fit_and_make_posterior(train_data, optz_opts)` fits and builds in one call.

## Posterior caching

Building a posterior costs an `O(n_train^3)` Cholesky factorization and a query `O(n_train^2)` or
less, so a posterior can be saved once and reloaded where only predictions are needed.

```cpp
#include "gpflash/posterior_eval/GpPosterior.hpp"  // load_posterior; the verified load is in posterior_build/

posterior.save_precompute("model.h5", "fitted from wt_run_42.h5");  // the note is optional free text

auto posterior_loaded = gpflash::load_posterior("model.h5");  // no training data needed
auto posterior_checked = gpflash::load_posterior_verified(  // the fit's own inputs, or it refuses the file
        "model.h5", train_data, mll_optz_result.hp_val_struct, mll_optz_result.model_options);
```

[`posterior_cache/README.md`](include/gpflash/posterior_cache/README.md) covers what a file
holds, what each load checks, and when a load raises.

## Assumptions and limitations

Deliberate restrictions, not defects, collected here because most are invisible from the
signatures.

**What a kernel must satisfy.**

- `K` positive semi-definite. Load-bearing: the conditioning guarantee rests on
  `lambda_min(K) >= 0`. Any genuine kernel satisfies it; a hand-built combination need not.
- A positive diagonal, since the preconditioner divides by it. A kernel declaring a unit
  diagonal (`is_kern_diag_unity`) skips the preconditioner unless a symmetry model is on;
  `test_gpflash_kernel_contract` checks every kernel's declaration against its assembled diagonal.
- A kernel with a varying diagonal and extra hyperparameters cannot be fitted: d(diag K)/d(extras)
  is implemented only for symmetry models, so its gradient raises `ErrorNotImplemented`. No current
  kernel has both.

**Model.**

- The posterior variance treats `hp_beta_vec` as known, so it understates the uncertainty, more so
  with more mean model basis functions.
- A mean model degree above `MEAN_MODEL_DEGREE_MAX` (`inputs/ModelOptions.hpp`), a mean model
  basis with more functions than training points, or one with a function that is zero at every
  training point, is rejected. So is a mean model basis column whose parity about the mirror plane
  differs from a symmetry model's.
- A mean model with more than one weight is solved with the mean model nugget, sized by
  `cond_max` from the bound `ModelOptions::mean_model_eig_max_ub_method` picks: the trace by
  default, or Gershgorin's. Where the basis is dependent or nearly dependent at the training
  points, as when a dimension's column takes one value beside the constant, the nugget splits the
  undetermined part of the weights between the dependent functions.
- Single scalar output.
- Repeated training points are accepted; the nugget keeps `K` factorizable.
- **Homoscedastic noise.** `sig2_f_val` is one scalar for every point, so genuinely varying
  uncertainty (mixed CFD and wind-tunnel data, differing sample counts) mis-weights the fit. A
  known per-point vector is a contained change; per-point *estimated* noise is not identifiable.

**Numerics.**

- Dense and exact: `O(n_train^2)` memory, `O(n_train^3)` factorization. No inducing points,
  low-rank approximation, or iterative solver.
- Under the Gershgorin bound, `eta` follows the row of `K` with the largest off-diagonal sum,
  so its derivative changes where that row changes. That derivative carries the factor
  `1 / (cond_max - 1)`, so at the default `cond_max` the MLL is smooth in practice.
- With more than one mean model weight, the gradient neglects one small term. It is nonzero only
  because of the mean model nugget, which keeps the weights' linear system well conditioned. It
  scales as `1 / (cond_max - 1)`, about `1e-10` of the gradient at the default `cond_max` of
  `1e10`, and is larger only where the mean model basis is nearly dependent at the training points.
- One length scale per dimension is the only per-coordinate treatment; inputs are not standardized. A stationary
  kernel centers them on the training points' per-dimension median, so an offset shared by all the data costs no
  accuracy.
- `cond_max` must lie in `(1, COND_MAX_LIMIT]` (`1e13`, `inputs/ModelOptions.hpp`); above that the nugget
  would change the model rather than regularize it.

## Two build configurations

Being header-only, what you compile is decided by what you include:

| configuration | for | headers | calls NLopt |
|---|---|---|---|
| full | fitting a model, then predicting from it | all of `include/gpflash/` | yes |
| prediction-only | evaluating an already-fitted model, e.g. on an embedded target | `Version.hpp`, `core/`, `inputs/`, `linear_algebra/`, `preconditioning/`, `kernels/`, `mean/`, `assembly/`, `posterior_cache/`, `posterior_eval/` | no |

None of those includes `posterior_build/`, `training/` or `optimization/`, which is what makes
the second configuration possible; the `check_prediction_only_includes` test enforces it. NLopt is
reached from one header only, so it drops out with the optimizer. A model fitted offline reaches a
prediction-only build through the [posterior cache](#posterior-caching).

## Building


With plain CMake, against the dependencies under [Requirements](#requirements). The configure step refuses a BLAS
with 64-bit integers, and warns on a 32-bit target, which has not been tested.

Linux (Debian/Ubuntu):

```
sudo apt install cmake g++ libarmadillo-dev libopenblas-dev libhdf5-dev libnlopt-cxx-dev
cmake -S . -B build && cmake --build build && ctest --test-dir build
```

macOS, with Homebrew:

```
brew install cmake armadillo libomp hdf5 nlopt
cmake -S . -B build -DOpenMP_ROOT=$(brew --prefix libomp)
cmake --build build && ctest --test-dir build
```

Apple clang has no OpenMP of its own, so `OpenMP_ROOT` points FindOpenMP at Homebrew's libomp. `-DBLA_VENDOR=<name>`
picks any other BLAS that CMake's FindBLAS knows.

Windows, with MSVC Build Tools and [vcpkg](https://vcpkg.io):

```
vcpkg install armadillo openblas hdf5 nlopt --triplet x64-windows
cmake -S . -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake
cmake --build build && ctest --test-dir build
```

`check_formatting` runs under ctest when clang-format 15, the version the sources are formatted with, is found; the
`clang-format-in-place-all` target reformats the sources with it. On macOS, `brew install llvm@15` provides it;
pass `-DGPFLASH_CLANG_FORMAT=$(brew --prefix llvm@15)/bin/clang-format` to the configure step.


### Using GpFlash from CMake
Install the package, then link the target `GpFlash::GpFlash`. It carries every define, compiler flag and library the
headers need, so a consumer sets none of them:
```
cmake --install build --prefix <dir>
```
```cmake
find_package(GpFlash CONFIG REQUIRED)  # with <dir> on CMAKE_PREFIX_PATH
target_link_libraries(app PRIVATE GpFlash::GpFlash)
```
A parent project can instead `add_subdirectory` the package, which builds no tests, and link the same target.
[Compiling the headers](#compiling-the-headers) is for a build that does not use CMake.

## Compiling the headers

Without the CMake target `GpFlash::GpFlash` ([Building](#building)), a build defines and links the following by hand.

Define, for every translation unit that includes a GpFlash header:

| Define | Why |
|---|---|
| `ARMA_DONT_USE_WRAPPER` | GpFlash declares the BLAS/LAPACK symbols itself and binds them to the linked library |
| `ARMA_USE_BLAS`, `ARMA_USE_LAPACK` | Armadillo's own operations call the same library |
| `ARMA_USE_HDF5` | selects the save path the posterior cache needs |
| `ARMA_64BIT_WORD` | an `n_train` by `n_train` buffer passes 2^31 elements at `n_train` ~ 46,000 |
| `ARMA_NO_DEBUG` | drops Armadillo's bounds checks; GpFlash's own checks guard what reaches its loops |
| `GPFLASH_BLAS_SKINNY_GEMV_IS_SLOW=1` | with Accelerate only; `core/Platform.hpp` lists every probe and its default |

Leave Armadillo's hidden-argument and `noexcept` options at their defaults. The headers reject a missing define from
the first three rows, or an Armadillo configured otherwise, at compile time (`linear_algebra/BlasLapack.hpp`,
`posterior_cache/Hdf5Store.hpp`).

Compile as C++14 or later with OpenMP enabled, and add:

- `-funroll-loops`: faster batch posterior evaluation.
- `-fno-math-errno`: with GCC on x86-64 Linux, lets the kernels' `exp`, `log`, `pow`, `sin` and `cos` vectorize. The
  vectorized functions can differ from the scalar ones in the last bits; define `GPFLASH_NO_VECTOR_MATH` to keep the
  scalar ones.

Do not use `-ffast-math` or `-Ofast`: they fold `std::isfinite` to true, which disables the input validation, and
`core/Platform.hpp` rejects them.

Link BLAS, LAPACK and HDF5's C library, and NLopt when `optimization/` is included.

## Python

`pip install gpflash` installs Python bindings with every dependency included. They offer a NumPy-style
`GaussianProcess` (`fit`, `predict`, `save`, `load`) and the C++ interface above under its C++ names and shapes.
[`python/README.md`](python/README.md) covers both, and building them from source.

## License

Copyright 2026 André Marchildon, under the Apache License 2.0; see `LICENSE` and `NOTICE`. `THIRD-PARTY-NOTICES`
lists the libraries GpFlash builds against and their licenses.
