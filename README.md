# GpFlash

GpFlash: Gaussian Process -- Fast Likelihood, Analytic, Storage-efficient, and High-dimensional

A fast, header-only C++ library for Gaussian process regression, built on
[Armadillo](https://arma.sourceforge.net/).

GpFlash computes the marginal log-likelihood (MLL) **and** its analytic gradient
with respect to the kernel hyperparameters for any stationary kernel, using a
single dense workspace and BLAS-level-3 kernel assembly. All hyperparameter
gradients are closed-form — no finite differences.

## Status

**Pre-release.** This repository currently holds the project scaffold; the
library source is being prepared and will land shortly.

## Planned features

- Marginal log-likelihood value and analytic gradient for stationary kernels
- Squared-exponential, Matérn 5/2, rational-quadratic, and periodic kernels
- Hyperparameter fitting by MLL maximization, with the prior mean and signal
  variance profiled out in closed form
- Posterior mean and variance evaluation with a reusable factorization cache
- Adaptive nugget bounding the kernel-matrix condition number

## Requirements

- A C++17 compiler with OpenMP support
- [Armadillo](https://arma.sourceforge.net/)
- A platform BLAS/LAPACK (Apple Accelerate, OpenBLAS, or AOCL)
- CMake ≥ 3.18

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
