"""Fit diagnostics and cache-file tools: what a C++ user reads after a fit or from a saved file."""

import numpy as np
import pytest

import gpflash


@pytest.fixture(scope="module")
def fit():
    rng = np.random.default_rng(0)
    X_train_mat = rng.random((30, 2))
    f_train_vec = np.sin(3.0 * X_train_mat[:, 0]) + X_train_mat[:, 1] ** 2
    optz_opts = gpflash.OptzOptions()
    optz_opts.model_options.kernel_id = gpflash.KernelId.RatQuad
    optz_opts.use_multistart = True
    optz_opts.n_ind_optz = 3
    gp = gpflash.GaussianProcess(optz_opts).fit(X_train_mat, f_train_vec)
    return gp, X_train_mat, f_train_vec


def test_best_local_optz_result(fit):
    gp, _, _ = fit
    best = gp.mll_optz_result().build_best_local_optz_result()
    assert best.stop_reason in gpflash.OptzStopReason.__members__.values()
    assert best.stop_reason != gpflash.OptzStopReason.Failed
    assert best.n_eval > 0
    # One log-hyperparameter per searched entry: two inverse length scales and the rat_quad shape.
    assert best.ln_hp_vec.shape == (3,)


def test_hp_at_bound_agrees_with_the_counts(fit):
    gp, _, _ = fit
    mll_optz_result = gp.mll_optz_result()
    for i_ind_optz in range(mll_optz_result.n_ind_optz_run()):
        at_bound = mll_optz_result.calc_hp_at_bound_struct(i_ind_optz)
        flags = mll_optz_result.calc_at_bound_flags_vec(i_ind_optz)
        n_at_bound = (
            at_bound.hp_gamma_at_bound_vec.sum()
            + at_bound.hp_kern_extra_at_bound_vec.sum()
            + at_bound.hp_sig2_k_at_bound
            + at_bound.hp_sig2_f_val_at_bound
        )
        assert n_at_bound == flags.sum() == mll_optz_result.calc_n_at_bound(i_ind_optz)
        assert at_bound.has_any() == (n_at_bound > 0)
    best = mll_optz_result.calc_hp_at_bound_struct()
    np.testing.assert_array_equal(
        best.hp_gamma_at_bound_vec,
        mll_optz_result.calc_hp_at_bound_struct(mll_optz_result.idx_best()).hp_gamma_at_bound_vec,
    )
    assert "hp_gamma" in best.to_summary_string()


def test_multistart_report(fit, tmp_path, capsys):
    gp, _, _ = fit
    report = gpflash.to_multistart_report_string(gp.mll_optz_result(), "test fit")
    assert "test fit" in report
    assert "Per optimization" in report

    # Printed through Python, so a notebook or capsys sees it.
    gpflash.print_multistart_report(gp.mll_optz_result(), "test fit")
    assert capsys.readouterr().out == report

    path = tmp_path / "report.txt"
    gpflash.save_multistart_report(str(path), gp.mll_optz_result(), "test fit")
    assert path.read_bytes() == report.encode()


def test_peek_info(fit, tmp_path):
    gp, X_train_mat, _ = fit
    path = str(tmp_path / "model.h5")
    gp.save_precompute(path, note="peek test")
    cache_info = gpflash.peek_info(path)
    assert (cache_info.n_train, cache_info.n_dim) == X_train_mat.shape
    assert cache_info.note == "peek test"
    assert cache_info.has_variance_data
    assert cache_info.symmetry_kind == gpflash.SymmetryKind.None_
    assert (cache_info.version_major, cache_info.version_minor, cache_info.version_patch) == tuple(
        int(part) for part in gpflash.__version__.split(".")
    )
    assert cache_info.post_model_info.kernel_id == gpflash.KernelId.RatQuad
    assert gpflash.peek_kernel_id(path) == gpflash.KernelId.RatQuad
    assert "peek test" in cache_info.to_summary_string()

    gp.save_precompute(str(tmp_path / "mean_only.h5"), include_variance_data=False)
    assert not gpflash.peek_info(str(tmp_path / "mean_only.h5")).has_variance_data


def test_print_cache_info(fit, tmp_path, capsys):
    gp, _, _ = fit
    path = str(tmp_path / "model.h5")
    gp.save_precompute(path)
    cache_info = gpflash.peek_info(path)
    gpflash.print_cache_info(cache_info, "label")
    assert capsys.readouterr().out == cache_info.to_summary_string("label")


def test_peek_info_on_a_missing_file_raises(tmp_path):
    with pytest.raises(gpflash.InvalidInputError):
        gpflash.peek_info(str(tmp_path / "missing.h5"))


def test_post_model_info_matches_the_posterior_and_its_file(fit, tmp_path):
    gp, _, _ = fit
    post_model_info = gp.posterior().post_model_info()
    assert post_model_info.kernel_id == gp.posterior().kernel_id()
    assert post_model_info.input_hash == gp.posterior().input_hash()
    assert post_model_info.mll == gp.posterior().mll()
    assert post_model_info.sig2_f_val == 0.0
    assert post_model_info.cond_max == gpflash.COND_MAX_DEFAULT
    path = str(tmp_path / "model.h5")
    gp.save_precompute(path)
    assert gpflash.peek_info(path).post_model_info.input_hash == post_model_info.input_hash
    assert "rat_quad" in post_model_info.to_summary_string()


def test_calc_default_cache_path(fit):
    _, X_train_mat, f_train_vec = fit
    train_data = gpflash.TrainingData()
    train_data.X_ndim_ntrain_mat = X_train_mat.T
    train_data.f_train_vec = f_train_vec
    path = gpflash.calc_default_cache_path(gpflash.KernelId.RatQuad, train_data, "out")
    assert path == "out/gpflash_rat_quad_ntrain30_ndim2.h5"
