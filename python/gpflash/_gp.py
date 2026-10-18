"""GaussianProcess: fit and predict with NumPy arrays of shape (n_points, n_dim), one point per row.

A wrapper over the C++ interface in gpflash._core, which holds points as columns. A C-ordered
(n_points, n_dim) array's transpose is column-major, so the points reach the C++ side without a
reordering copy. Names follow the C++ library's.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

from . import _core  # pyright: ignore[reportPrivateUsage]  (the package's own compiled module)

# What the methods accept for points or values: a NumPy array, or nested lists of numbers.
ArrayLike = NDArray[Any] | Sequence[float] | Sequence[Sequence[float]]


def _as_points_mat(X_mat: ArrayLike, arg_name: str) -> NDArray[np.float64]:
    """X_mat as a C-ordered float64 (n_points, n_dim) array; 2-D only, as 1-D could mean one point or a 1-D problem."""
    X_points_mat: NDArray[np.float64] = np.asarray(X_mat, dtype=np.float64)
    if X_points_mat.ndim != 2:
        raise _core.InvalidInputError(
            f"{arg_name} must be 2-D (n_points, n_dim); got {X_points_mat.ndim} dimensions. "
            "Reshape to (n_points, 1) for a 1-D problem, or (1, n_dim) for a single point."
        )
    return np.ascontiguousarray(X_points_mat)


@dataclass(frozen=True)
class PosteriorPrediction:
    """What GaussianProcess.predict returns: C++'s PosteriorResult with one point per row.

    A field is None when its predict flag did not ask for it.
    """

    mean_post_vec: NDArray[np.float64] | None  # (n_post)
    var_post_vec: NDArray[np.float64] | None  # (n_post)
    d_mean_post_d_x_post_mat: NDArray[np.float64] | None  # (n_post, n_dim), one gradient per row
    d_var_post_d_x_post_mat: NDArray[np.float64] | None  # (n_post, n_dim), one gradient per row


class GaussianProcess:
    """An exact Gaussian process, fitted by maximizing the marginal log likelihood (MLL).

    The options are C++'s OptzOptions, set once, in the constructor; the data, with its noise
    variance, go to fit(). Every points argument is (n_points, n_dim), one point per row.

    Parameters
    ----------
    optz_opts : OptzOptions, optional
        How the hyperparameters are fitted, copied; its model_options name the kernel and the mean
        model. The default, OptzOptions(), is the Matern 5/2 kernel, a constant mean and one
        optimization.

    The state is read through methods, as in C++: posterior() and mll_optz_result() return what
    fit(), make_posterior() or loading last set.
    """

    def __init__(self, optz_opts: _core.OptzOptions | None = None) -> None:
        if optz_opts is None:
            optz_opts = _core.OptzOptions()
        elif not isinstance(optz_opts, _core.OptzOptions):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(
                f"GaussianProcess takes an OptzOptions; got {type(optz_opts).__name__}. To choose a kernel by name, "
                "set optz_opts.model_options.kernel_id = gpflash.parse_kernel_id('sq_exp')."
            )
        self._optz_opts = _core.OptzOptions(optz_opts)
        self._mll_optz_result: _core.MllOptzResult | None = None
        self._posterior: _core.GpPosterior | None = None
        self._train_sig2_f_val: float | None = None  # the posterior's data's sig2_f_val; None after a plain load

    def _set_posterior(
        self,
        posterior: _core.GpPosterior,
        mll_optz_result: _core.MllOptzResult | None,
        train_sig2_f_val: float | None,
    ) -> None:
        """Every change to the fitted state goes through here, so the three attributes always describe one posterior."""
        self._posterior = posterior
        self._mll_optz_result = mll_optz_result
        self._train_sig2_f_val = train_sig2_f_val

    def _make_train_data(self, X_train_mat: ArrayLike, f_train_vec: ArrayLike, sig2_f_val: float) -> _core.TrainingData:
        X_train_points_mat = _as_points_mat(X_train_mat, "X_train_mat")
        f_train_values_vec: NDArray[np.float64] = np.asarray(f_train_vec, dtype=np.float64)
        n_train = X_train_points_mat.shape[0]
        if f_train_values_vec.ndim != 1 or f_train_values_vec.shape[0] != n_train:
            raise _core.InvalidInputError(
                f"f_train_vec must be 1-D with one value per row of X_train_mat ({n_train}); "
                f"got shape {f_train_values_vec.shape}."
            )
        if not np.isfinite(sig2_f_val):
            raise _core.InvalidInputError(
                f"sig2_f_val must be finite: 0 for noise-free data, a known variance > 0, or < 0 to estimate it; "
                f"got {sig2_f_val}."
            )
        train_data = _core.TrainingData()
        train_data.X_ndim_ntrain_mat = X_train_points_mat.T
        train_data.f_train_vec = f_train_values_vec
        train_data.sig2_f_val = float(sig2_f_val)
        return train_data

    # ── Fit, or make a posterior from given hyperparameters ─────────────────

    def fit(self, X_train_mat: ArrayLike, f_train_vec: ArrayLike, *, sig2_f_val: float = 0.0) -> Self:
        """Fit the hyperparameters to the (n_train, n_dim) points X_train_mat and their values f_train_vec.

        sig2_f_val is the noise variance on the values, by its sign as in C++'s TrainingData: 0 for
        noise-free data, a positive value for a known variance held fixed, a negative value to estimate it.
        """
        train_data = self._make_train_data(X_train_mat, f_train_vec, sig2_f_val)
        mll_optz_result = _core.select_hp_with_max_mll(train_data, self._optz_opts)
        self._set_posterior(_core.make_posterior_from_fit(train_data, mll_optz_result), mll_optz_result, sig2_f_val)
        return self

    def make_posterior(
        self, X_train_mat: ArrayLike, f_train_vec: ArrayLike, hp_val_struct: _core.HpVal, *, sig2_f_val: float = 0.0
    ) -> Self:
        """The posterior at given hyperparameters, without fitting.

        hp_val_struct must hold hp_sig2_k, and hp_sig2_f_val when sig2_f_val < 0 asks for the noise to be estimated.
        """
        train_data = self._make_train_data(X_train_mat, f_train_vec, sig2_f_val)
        posterior = _core.make_posterior(train_data, hp_val_struct, self._optz_opts.model_options)
        self._set_posterior(posterior, None, sig2_f_val)
        return self

    def eval_mll(
        self,
        X_train_mat: ArrayLike,
        f_train_vec: ArrayLike,
        hp_val_struct: _core.HpVal,
        *,
        return_mll_gradient: bool = False,
        sig2_f_val: float = 0.0,
    ) -> _core.MllResult:
        """The MLL at given hyperparameters, and its gradient when return_mll_gradient is True."""
        train_data = self._make_train_data(X_train_mat, f_train_vec, sig2_f_val)
        return _core.eval_mll(train_data, hp_val_struct, self._optz_opts.model_options, return_mll_gradient)

    # ── Predict ──────────────────────────────────────────────────────────────

    def predict(
        self,
        X_post_mat: ArrayLike,
        *,
        return_mean: bool = True,
        return_var: bool = True,
        return_mean_gradient: bool = False,
        return_var_gradient: bool = False,
        check_query_points: bool = True,
    ) -> PosteriorPrediction:
        """The posterior at the (n_post, n_dim) points X_post_mat, in one pass.

        The flags and their defaults are C++'s PosteriorOptions; a quantity not asked for is None in the
        result. The variance costs O(n_train^2) per point against O(n_train) for the mean, so pass
        return_var=False when only the mean is needed, and for a posterior saved without variance data.
        check_query_points=False skips the check of the points' dimension and finiteness, for speed.
        """
        posterior = self._fitted_posterior()
        X_post_points_mat = _as_points_mat(X_post_mat, "X_post_mat")
        n_dim = X_post_points_mat.shape[1]
        if n_dim != posterior.n_dim():
            raise _core.InvalidInputError(
                f"X_post_mat has {n_dim} columns; the model was fitted in {posterior.n_dim()} dimensions."
            )
        post_opts = _core.PosteriorOptions()
        post_opts.return_mean = bool(return_mean)
        post_opts.return_var = bool(return_var)
        post_opts.return_mean_gradient = bool(return_mean_gradient)
        post_opts.return_var_gradient = bool(return_var_gradient)
        post_opts.check_query_points = bool(check_query_points)
        posterior_result = posterior.eval(X_post_points_mat.T, post_opts)

        # What was not asked for is None rather than C++'s empty array. The gradients are transposed to one per row.
        mean_post_vec = None
        var_post_vec = None
        d_mean_post_d_x_post_mat = None
        d_var_post_d_x_post_mat = None
        if return_mean:
            mean_post_vec = posterior_result.mean_post_vec
        if return_var:
            var_post_vec = posterior_result.var_post_vec
        if return_mean_gradient:
            d_mean_post_d_x_post_mat = posterior_result.d_mean_post_d_x_post_ndim_npost_mat.T
        if return_var_gradient:
            d_var_post_d_x_post_mat = posterior_result.d_var_post_d_x_post_ndim_npost_mat.T
        return PosteriorPrediction(mean_post_vec, var_post_vec, d_mean_post_d_x_post_mat, d_var_post_d_x_post_mat)

    # ── The fitted model ─────────────────────────────────────────────────────

    def hp_val_struct(self) -> _core.HpVal:
        """The hyperparameters, as a copy: the fit's, or for a posterior made or loaded without one, the posterior's.

        As in C++, hp_sig2_f_val is set only when the data asked for the noise to be estimated; a posterior
        loaded without its data cannot say, so it stays unset there.
        """
        if self._mll_optz_result is not None:
            return _core.HpVal(self._mll_optz_result.hp_val_struct)
        posterior = self._fitted_posterior()
        hp_val_struct = _core.HpVal()
        hp_val_struct.hp_gamma_vec = posterior.hp_gamma_vec()
        hp_val_struct.hp_kern_extra_vec = posterior.hp_kern_extra_vec()
        hp_val_struct.hp_sig2_k = posterior.hp_sig2_k()
        if self._train_sig2_f_val is not None and self._train_sig2_f_val < 0.0:
            hp_val_struct.hp_sig2_f_val = posterior.post_model_info().sig2_f_val
        return hp_val_struct

    def posterior(self) -> _core.GpPosterior:
        """The posterior predict() evaluates, for the C++ interface's queries and diagnostics."""
        return self._fitted_posterior()

    def mll_optz_result(self) -> _core.MllOptzResult:
        """The fit: every optimization, where each started and stopped, and why."""
        if self._mll_optz_result is None:
            raise RuntimeError("mll_optz_result() needs a fit; a posterior made or loaded without one has none.")
        return self._mll_optz_result

    def optz_opts(self) -> _core.OptzOptions:
        """The options, as a copy: set once, in the constructor."""
        return _core.OptzOptions(self._optz_opts)

    def hp_beta_vec(self) -> NDArray[np.float64]:
        """The mean model weights, solved in closed form."""
        return self._fitted_posterior().hp_beta_vec()

    def mll(self) -> float:
        """The MLL at the posterior's hyperparameters."""
        return self._fitted_posterior().mll()

    def kernel_id(self) -> _core.KernelId:
        """The posterior's kernel, or before one exists, the kernel the options name."""
        if self._posterior is not None:
            return self._posterior.kernel_id()
        return self._optz_opts.model_options.kernel_id

    def to_summary_string(self, label: str = "") -> str:
        """A short report of the fit."""
        return self.mll_optz_result().to_summary_string(label)

    # ── Save and load ────────────────────────────────────────────────────────

    def save_precompute(self, path: str | os.PathLike[str], note: str = "", include_variance_data: bool = True) -> None:
        """Save the posterior to an HDF5 file that load_posterior, in Python or C++, reads back.

        include_variance_data=False stores only what the mean needs, a smaller file that cannot return variances.
        """
        self._fitted_posterior().save_precompute(os.fspath(path), note, include_variance_data)

    @classmethod
    def load_posterior(cls, path: str | os.PathLike[str]) -> Self:
        """Load a posterior saved with save_precompute. It predicts; it holds no fit.

        The file is checked, not which fit wrote it; load_posterior_verified checks that too. The file
        records the kernel and whether the mean model has a constant, which optz_opts() then holds; the
        mean model degrees and the fit options stay at their defaults.
        """
        cache_info = _core.peek_info(os.fspath(path))
        optz_opts = _core.OptzOptions()
        optz_opts.model_options.kernel_id = cache_info.post_model_info.kernel_id
        optz_opts.model_options.has_mean_model_constant = cache_info.has_mean_model_constant
        gp = cls(optz_opts)
        gp._set_posterior(_core.load_posterior(os.fspath(path)), None, None)
        return gp

    def load_posterior_verified(
        self,
        path: str | os.PathLike[str],
        X_train_mat: ArrayLike,
        f_train_vec: ArrayLike,
        hp_val_struct: _core.HpVal,
        *,
        sig2_f_val: float = 0.0,
    ) -> Self:
        """Load a posterior, refusing a file that this model's options, these data and hp_val_struct did not produce."""
        train_data = self._make_train_data(X_train_mat, f_train_vec, sig2_f_val)
        posterior = _core.load_posterior_verified(
            os.fspath(path), train_data, hp_val_struct, self._optz_opts.model_options
        )
        self._set_posterior(posterior, None, sig2_f_val)
        return self

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _fitted_posterior(self) -> _core.GpPosterior:
        if self._posterior is None:
            raise RuntimeError("This GaussianProcess has no posterior; call fit(X_train_mat, f_train_vec) first.")
        return self._posterior

    def __repr__(self) -> str:
        state = "with a posterior" if self._posterior is not None else "not fitted"
        return f"<GaussianProcess kernel_id={self.kernel_id()}, {state}>"
