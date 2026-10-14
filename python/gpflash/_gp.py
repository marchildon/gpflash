"""GaussianProcess: fit and predict with NumPy arrays of shape (n_points, n_dim).

A wrapper over the C++ interface in gpflash._core, which holds points as columns. A C-ordered
(n_points, n_dim) array's transpose is column-major, so the points reach the C++ side without a
reordering copy.
"""

from __future__ import annotations

import os
from typing import Optional, Union

import numpy as np

from . import _core

# The mean models this wrapper names. Others are set through ModelOptions.
_MEAN_MODELS = ("constant", "zero")


def _as_points(X, name: str) -> np.ndarray:
    """X as a float64 (n_points, n_dim) array; a 1-D array is one point."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[np.newaxis, :]
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n_points, n_dim); got {X.ndim} dimensions.")
    return np.ascontiguousarray(X)


def _to_kernel_id(kernel) -> _core.KernelId:
    if isinstance(kernel, _core.KernelId):
        return kernel
    if isinstance(kernel, str):
        return _core.parse_kernel_id(kernel)
    raise TypeError(f"kernel must be a name such as 'matern_5_2' or a KernelId; got {type(kernel).__name__}.")


class GaussianProcess:
    """An exact Gaussian process, fitted by maximizing the marginal log likelihood (MLL).

    Parameters
    ----------
    kernel : str or KernelId, default "matern_5_2"
        One of "sq_exp", "matern_5_2", "matern_3_2", "rat_quad", "periodic", "linear",
        "quadratic".
    noise : float or "estimate", default 0.0
        Noise variance on the observations: 0 for noise-free data, a positive value for a known
        variance held fixed, or "estimate" to fit it.
    mean : {"constant", "zero"}, default "constant"
        The prior mean. Its weights are solved in closed form.
    n_starts : int, default 1
        Local optimizations to run. Above 1, the extra starts are Latin-hypercube samples within
        the bounds, and the best fit is kept.
    seed : int, default 5
        Seed for the Latin-hypercube starts, so a fit is reproducible.

    ``GaussianProcess.from_options(optz_options)`` takes an ``OptzOptions`` instead, for full
    control of the fit: bounds, tolerances, the algorithm, symmetry, mean model degrees.

    Attributes, set by fit
    ----------------------
    fit_result_ : MllOptzResult
        The full fit; ``fit_result_.to_summary_string()`` reports it.
    posterior_ : GpPosterior
        The posterior the predictions use.
    """

    def __init__(
        self,
        kernel: Union[str, _core.KernelId] = "matern_5_2",
        *,
        noise: Union[float, str] = 0.0,
        mean: str = "constant",
        n_starts: int = 1,
        seed: int = 5,
    ) -> None:
        optz_options = _core.OptzOptions()
        optz_options.model_options.kernel_id = _to_kernel_id(kernel)
        if mean not in _MEAN_MODELS:
            raise ValueError(f"mean must be one of {_MEAN_MODELS}; got {mean!r}. Use from_options for others.")
        optz_options.model_options.has_mean_model_constant = mean == "constant"
        if int(n_starts) < 1:
            raise ValueError(f"n_starts must be >= 1; got {n_starts}.")
        optz_options.use_multistart = int(n_starts) > 1
        optz_options.n_ind_optz = int(n_starts)
        optz_options.hypercube_sampling_seed = int(seed)
        self._init(optz_options, noise)

    @classmethod
    def from_options(cls, optz_options: _core.OptzOptions, *, noise: Union[float, str] = 0.0) -> "GaussianProcess":
        """A model fitted with these OptzOptions, copied, whose model_options name the kernel and mean model."""
        gp = cls.__new__(cls)
        gp._init(_core.OptzOptions(optz_options), noise)
        return gp

    def _init(self, optz_options: _core.OptzOptions, noise: Union[float, str]) -> None:
        self._optz_options = optz_options
        self._sig2_f_val = self._parse_noise(noise)
        self.fit_result_: Optional[_core.MllOptzResult] = None
        self.posterior_: Optional[_core.GpPosterior] = None

    @staticmethod
    def _parse_noise(noise) -> float:
        if isinstance(noise, str):
            if noise != "estimate":
                raise ValueError(f"noise must be a variance >= 0 or 'estimate'; got {noise!r}.")
            return -1.0
        noise = float(noise)
        if not np.isfinite(noise) or noise < 0.0:
            raise ValueError(f"noise must be a finite variance >= 0, or 'estimate'; got {noise}.")
        return noise

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(self, X, y) -> "GaussianProcess":
        """Fit the hyperparameters to (n_points, n_dim) points X and their values y."""
        X = _as_points(X, "X")
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError(f"y must be 1-D with one value per row of X ({X.shape[0]}); got shape {y.shape}.")
        train_data = _core.TrainingData()
        train_data.X_ndim_ntrain_mat = X.T
        train_data.f_train_vec = y
        train_data.sig2_f_val = self._sig2_f_val
        self.fit_result_ = _core.select_hp_with_max_mll(train_data, self._optz_options)
        self.posterior_ = _core.make_posterior_from_fit(train_data, self.fit_result_)
        return self

    # ── Predict ──────────────────────────────────────────────────────────────

    def predict(self, X, return_var: bool = False):
        """Posterior mean at (n_points, n_dim) points X, and the variance when return_var is True."""
        post_opts = _core.PosteriorOptions()
        post_opts.return_var = bool(return_var)
        result = self._eval(X, post_opts)
        if return_var:
            return result.mean_post_vec, result.var_post_vec
        return result.mean_post_vec

    def predict_gradient(self, X):
        """Gradients of the posterior mean and variance at X, each (n_points, n_dim)."""
        post_opts = _core.PosteriorOptions()
        post_opts.return_mean = False
        post_opts.return_var = False
        post_opts.return_mean_gradient = True
        post_opts.return_var_gradient = True
        result = self._eval(X, post_opts)
        return result.d_mean_post_d_x_post_ndim_npost_mat.T, result.d_var_post_d_x_post_ndim_npost_mat.T

    def _eval(self, X, post_opts: _core.PosteriorOptions) -> _core.PosteriorResult:
        posterior = self._fitted_posterior()
        X = _as_points(X, "X")
        if X.shape[1] != posterior.n_dim():
            raise ValueError(f"X has {X.shape[1]} columns; the model was fitted in {posterior.n_dim()} dimensions.")
        return posterior.eval(X.T, post_opts)

    # ── Fitted values ────────────────────────────────────────────────────────

    @property
    def hyperparameters_(self) -> dict:
        """The fitted hyperparameters, and the mean model weights."""
        posterior = self._fitted_posterior()
        hp_sig2_f_val = None
        if self.fit_result_ is not None and self.fit_result_.hp_val_struct.hp_sig2_f_val > 0.0:
            hp_sig2_f_val = self.fit_result_.hp_val_struct.hp_sig2_f_val
        return {
            "sig2_k": posterior.hp_sig2_k(),
            "sig2_f": hp_sig2_f_val,
            "gamma": posterior.hp_gamma_vec(),
            "kern_extra": posterior.hp_kern_extra_vec(),
            "beta": posterior.hp_beta_vec(),
        }

    @property
    def log_marginal_likelihood_(self) -> float:
        """The MLL at the fitted hyperparameters."""
        return self._fitted_posterior().mll()

    @property
    def kernel(self) -> str:
        if self.posterior_ is not None:
            return _core.kernel_name(self.posterior_.kernel_id())
        return _core.kernel_name(self._optz_options.model_options.kernel_id)

    def summary(self) -> str:
        """A short report of the fit."""
        if self.fit_result_ is None:
            raise RuntimeError("summary() needs a fit; a loaded model carries none.")
        return self.fit_result_.to_summary_string()

    # ── Save and load ────────────────────────────────────────────────────────

    def save(self, path: Union[str, os.PathLike], note: str = "", include_variance: bool = True) -> None:
        """Save the posterior to an HDF5 file that GaussianProcess.load or the C++ load_posterior reads.

        include_variance=False stores only what the mean needs, a smaller file that cannot return variances.
        """
        self._fitted_posterior().save_precompute(os.fspath(path), note, include_variance)

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> "GaussianProcess":
        """Load a model saved with save(). It predicts; it holds no fit, so summary() is unavailable."""
        posterior = _core.load_posterior(os.fspath(path))
        gp = cls(posterior.kernel_id())
        gp.posterior_ = posterior
        return gp

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _fitted_posterior(self) -> _core.GpPosterior:
        if self.posterior_ is None:
            raise RuntimeError("This GaussianProcess is not fitted; call fit(X, y) first.")
        return self.posterior_

    def __repr__(self) -> str:
        state = "fitted" if self.posterior_ is not None else "not fitted"
        return f"GaussianProcess(kernel={self.kernel!r}, {state})"
