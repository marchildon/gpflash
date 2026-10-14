"""Fit diagnostics and cache-file tools: what a C++ user reads after a fit or from a saved file."""

import numpy as np
import pytest

import gpflash


@pytest.fixture(scope="module")
def fit():
    rng = np.random.default_rng(0)
    X = rng.random((30, 2))
    y = np.sin(3.0 * X[:, 0]) + X[:, 1] ** 2
    gp = gpflash.GaussianProcess("rat_quad", n_starts=3).fit(X, y)
    return gp, X, y


def test_best_local_optz_result(fit):
    gp, _, _ = fit
    best = gp.fit_result_.build_best_local_optz_result()
    assert best.stop_reason in gpflash.OptzStopReason.__members__.values()
    assert best.stop_reason != gpflash.OptzStopReason.Failed
    assert best.n_eval > 0
    # One log-hyperparameter per searched entry: two inverse length scales and the rat_quad shape.
    assert best.ln_hp_vec.shape == (3,)


def test_hp_at_bound_agrees_with_the_counts(fit):
    gp, _, _ = fit
    result = gp.fit_result_
    for i_ind_optz in range(result.n_ind_optz_run()):
        at_bound = result.calc_hp_at_bound_struct(i_ind_optz)
        flags = result.calc_at_bound_flags_vec(i_ind_optz)
        n_at_bound = (
            at_bound.hp_gamma_at_bound_vec.sum()
            + at_bound.hp_kern_extra_at_bound_vec.sum()
            + at_bound.hp_sig2_k_at_bound
            + at_bound.hp_sig2_f_val_at_bound
        )
        assert n_at_bound == flags.sum() == result.calc_n_at_bound(i_ind_optz)
        assert at_bound.has_any() == (n_at_bound > 0)
    best = result.calc_hp_at_bound_struct()
    np.testing.assert_array_equal(
        best.hp_gamma_at_bound_vec, result.calc_hp_at_bound_struct(result.idx_best()).hp_gamma_at_bound_vec
    )
    assert "hp_gamma" in best.to_summary_string()


def test_multistart_report(fit, tmp_path, capsys):
    gp, _, _ = fit
    report = gpflash.to_multistart_report_string(gp.fit_result_, "test fit")
    assert "test fit" in report
    assert "Per optimization" in report

    # Printed through Python, so a notebook or capsys sees it.
    gpflash.print_multistart_report(gp.fit_result_, "test fit")
    assert capsys.readouterr().out == report

    path = tmp_path / "report.txt"
    gpflash.save_multistart_report(str(path), gp.fit_result_, "test fit")
    assert path.read_bytes() == report.encode()


def test_peek_info(fit, tmp_path):
    gp, X, _ = fit
    path = str(tmp_path / "model.h5")
    gp.save(path, note="peek test")
    info = gpflash.peek_info(path)
    assert (info.n_train, info.n_dim) == X.shape
    assert info.note == "peek test"
    assert info.has_variance_data
    assert info.symmetry_kind == gpflash.SymmetryKind.None_
    assert (info.version_major, info.version_minor, info.version_patch) == tuple(
        int(part) for part in gpflash.__version__.split(".")
    )
    assert info.post_model_info.kernel_id == gpflash.KernelId.RatQuad
    assert gpflash.peek_kernel_id(path) == gpflash.KernelId.RatQuad
    assert "peek test" in info.to_summary_string()

    gp.save(str(tmp_path / "mean_only.h5"), include_variance=False)
    assert not gpflash.peek_info(str(tmp_path / "mean_only.h5")).has_variance_data


def test_print_cache_info(fit, tmp_path, capsys):
    gp, _, _ = fit
    path = str(tmp_path / "model.h5")
    gp.save(path)
    info = gpflash.peek_info(path)
    gpflash.print_cache_info(info, "label")
    assert capsys.readouterr().out == info.to_summary_string("label")


def test_peek_info_on_a_missing_file_raises(tmp_path):
    with pytest.raises(gpflash.InvalidInputError):
        gpflash.peek_info(str(tmp_path / "missing.h5"))


def test_post_model_info_matches_the_posterior_and_its_file(fit, tmp_path):
    gp, _, _ = fit
    info = gp.posterior_.post_model_info()
    assert info.kernel_id == gp.posterior_.kernel_id()
    assert info.input_hash == gp.posterior_.input_hash()
    assert info.mll == gp.posterior_.mll()
    assert info.sig2_f_val == 0.0
    assert info.cond_max == gpflash.COND_MAX_DEFAULT
    path = str(tmp_path / "model.h5")
    gp.save(path)
    assert gpflash.peek_info(path).post_model_info.input_hash == info.input_hash
    assert "rat_quad" in info.to_summary_string()


def test_calc_default_cache_path(fit):
    _, X, y = fit
    train_data = gpflash.TrainingData()
    train_data.X_ndim_ntrain_mat = X.T
    train_data.f_train_vec = y
    path = gpflash.calc_default_cache_path(gpflash.KernelId.RatQuad, train_data, "out")
    assert path == "out/gpflash_rat_quad_ntrain30_ndim2.h5"
