"""The GaussianProcess wrapper: NumPy (n_points, n_dim) layout, fitting, prediction, saving."""

import threading

import numpy as np
import pytest

import gpflash


def target(X_mat):
    return np.sin(3.0 * X_mat[:, 0]) + X_mat[:, 1] ** 2


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X_train_mat = rng.random((40, 2))
    return X_train_mat, target(X_train_mat), rng.random((25, 2))


@pytest.fixture(scope="module")
def gp(data):
    X_train_mat, f_train_vec, _ = data
    return gpflash.GaussianProcess(make_optz_opts(gpflash.KernelId.SqExp)).fit(X_train_mat, f_train_vec)


def make_optz_opts(kernel_id=gpflash.KernelId.Matern52):
    optz_opts = gpflash.OptzOptions()
    optz_opts.model_options.kernel_id = kernel_id
    return optz_opts


def predict_mean(gp, X_post_mat):
    return gp.predict(X_post_mat, return_var=False).mean_post_vec


def test_noise_free_fit_interpolates(gp, data):
    # Nearly, not exactly: the nugget that keeps K factorizable, sized from cond_max, smooths the fit slightly.
    X_train_mat, f_train_vec, _ = data
    prediction = gp.predict(X_train_mat)
    np.testing.assert_allclose(prediction.mean_post_vec, f_train_vec, atol=1e-3)
    assert np.all(prediction.var_post_vec < 1e-6)


def test_predicts_held_out_points(gp, data):
    _, _, X_post_mat = data
    np.testing.assert_allclose(predict_mean(gp, X_post_mat), target(X_post_mat), atol=1e-2)


def test_defaults_match_posterior_options(gp, data):
    # The mean and the variance by default, as in C++; the gradients on request; anything not asked for is None.
    _, _, X_post_mat = data
    prediction = gp.predict(X_post_mat)
    assert prediction.mean_post_vec.shape == prediction.var_post_vec.shape == (25,)
    assert prediction.d_mean_post_d_x_post_mat is None
    assert prediction.d_var_post_d_x_post_mat is None
    mean_only = gp.predict(X_post_mat, return_var=False)
    assert mean_only.var_post_vec is None
    var_only = gp.predict(X_post_mat, return_mean=False)
    assert var_only.mean_post_vec is None
    np.testing.assert_array_equal(var_only.var_post_vec, prediction.var_post_vec)


def test_one_pass_matches_separate_requests(gp, data):
    _, _, X_post_mat = data
    everything = gp.predict(X_post_mat, return_mean_gradient=True, return_var_gradient=True)
    np.testing.assert_array_equal(everything.mean_post_vec, predict_mean(gp, X_post_mat))
    np.testing.assert_array_equal(everything.var_post_vec, gp.predict(X_post_mat).var_post_vec)
    assert everything.d_mean_post_d_x_post_mat.shape == everything.d_var_post_d_x_post_mat.shape == (25, 2)


def test_matches_core_interface(gp, data):
    X_train_mat, f_train_vec, X_post_mat = data
    train_data = gpflash.TrainingData()
    train_data.X_ndim_ntrain_mat = X_train_mat.T
    train_data.f_train_vec = f_train_vec
    optz_opts = gpflash.OptzOptions()
    optz_opts.model_options.kernel_id = gpflash.KernelId.SqExp
    posterior = gpflash.fit_and_make_posterior(train_data, optz_opts)
    # Two fits of the same problem, so they agree to rounding: BLAS reduction order can differ between runs, which
    # moves where each optimization stops by ~1e-11, not bitwise.
    np.testing.assert_allclose(predict_mean(gp, X_post_mat), posterior.eval(X_post_mat.T).mean_post_vec, rtol=1e-8)


def test_memory_layout_does_not_change_results(gp, data):
    # Every layout becomes the same buffer before the C++ call, so the results are bitwise equal.
    _, _, X_post_mat = data
    expected = predict_mean(gp, X_post_mat)
    np.testing.assert_array_equal(predict_mean(gp, np.asfortranarray(X_post_mat)), expected)
    np.testing.assert_array_equal(predict_mean(gp, np.repeat(X_post_mat, 2, axis=0)[::2]), expected)
    np.testing.assert_array_equal(predict_mean(gp, X_post_mat.tolist()), expected)
    # A single point is a batch of one, which BLAS runs through different kernels than a batch of 25, so it agrees to
    # rounding rather than bitwise.
    np.testing.assert_allclose(predict_mean(gp, X_post_mat[:1]), expected[:1], rtol=1e-10)


def test_points_must_be_2d(gp, data):
    # A 1-D array could be one point or a 1-D problem, so neither reading is guessed.
    X_train_mat, f_train_vec, X_post_mat = data
    with pytest.raises(gpflash.InvalidInputError, match="2-D"):
        gp.predict(X_post_mat[0])
    with pytest.raises(gpflash.InvalidInputError, match="2-D"):
        gpflash.GaussianProcess().fit(X_train_mat[:, 0], f_train_vec)
    one_dim = gpflash.GaussianProcess().fit(X_train_mat[:, :1], f_train_vec)
    assert one_dim.predict(X_post_mat[:, :1]).mean_post_vec.shape == (25,)


def test_gradient_matches_finite_difference(gp, data):
    _, _, X_post_mat = data
    prediction = gp.predict(
        X_post_mat[:3], return_mean=False, return_var=False, return_mean_gradient=True, return_var_gradient=True
    )
    assert prediction.d_mean_post_d_x_post_mat.shape == prediction.d_var_post_d_x_post_mat.shape == (3, 2)
    step = 1e-6
    x_plus_mat, x_minus_mat = X_post_mat[:1].copy(), X_post_mat[:1].copy()
    x_plus_mat[0, 1] += step
    x_minus_mat[0, 1] -= step
    fd = (predict_mean(gp, x_plus_mat)[0] - predict_mean(gp, x_minus_mat)[0]) / (2.0 * step)
    assert prediction.d_mean_post_d_x_post_mat[0, 1] == pytest.approx(fd, rel=1e-4)


def test_unchecked_query_matches_checked(gp, data):
    _, _, X_post_mat = data
    np.testing.assert_array_equal(
        gp.predict(X_post_mat, check_query_points=False).mean_post_vec, predict_mean(gp, X_post_mat)
    )


def test_hyperparameters_and_fit(gp):
    hp_val_struct = gp.hp_val_struct()
    assert hp_val_struct.hp_sig2_k > 0.0
    assert hp_val_struct.hp_sig2_f_val < 0.0  # unset: noise-free data has none to estimate
    assert hp_val_struct.hp_gamma_vec.shape == (2,)
    assert hp_val_struct.hp_kern_extra_vec.shape == (0,)
    assert gp.hp_beta_vec().shape == (1,)
    assert gp.mll() == pytest.approx(gp.mll_optz_result().mll())
    assert gp.kernel_id() == gpflash.KernelId.SqExp
    assert "marginal log lik" in gp.to_summary_string()
    assert repr(gp) == "<GaussianProcess kernel_id=KernelId.SqExp, with a posterior>"
    assert repr(hp_val_struct).startswith("HpVal(hp_sig2_k=")


def test_hp_val_struct_is_a_copy(gp):
    hp_val_struct = gp.hp_val_struct()
    hp_val_struct.hp_sig2_k = 123.0
    assert gp.hp_val_struct().hp_sig2_k != 123.0


def test_estimated_noise():
    rng = np.random.default_rng(1)
    X_train_mat = rng.random((60, 1))
    f_train_vec = np.sin(6.0 * X_train_mat[:, 0]) + 0.1 * rng.standard_normal(60)
    gp = gpflash.GaussianProcess().fit(X_train_mat, f_train_vec, sig2_f_val=-1.0)
    hp_sig2_f_val = gp.hp_val_struct().hp_sig2_f_val
    assert 1e-3 < hp_sig2_f_val < 1e-1
    # A posterior made at those hyperparameters, with the noise still estimated, reports the same noise.
    remade = gpflash.GaussianProcess().make_posterior(X_train_mat, f_train_vec, gp.hp_val_struct(), sig2_f_val=-1.0)
    assert remade.hp_val_struct().hp_sig2_f_val == pytest.approx(hp_sig2_f_val)
    # Evaluating the MLL under another noise rule leaves the posterior, and so the reported noise, as it was.
    # hp_sig2_f_val is set only when sig2_f_val < 0 asks for an estimate; C++ refuses it otherwise.
    hp_without_noise = gp.hp_val_struct()
    hp_without_noise.hp_sig2_f_val = -1.0
    remade.eval_mll(X_train_mat, f_train_vec, hp_without_noise, sig2_f_val=0.0)
    assert remade.hp_val_struct().hp_sig2_f_val == pytest.approx(hp_sig2_f_val)
    noise_free = gpflash.GaussianProcess().make_posterior(X_train_mat, f_train_vec, hp_without_noise)
    noise_free.eval_mll(X_train_mat, f_train_vec, gp.hp_val_struct(), sig2_f_val=-1.0)
    assert noise_free.hp_val_struct().hp_sig2_f_val < 0.0
    # With a known noise, nothing is estimated, so the field stays unset.
    known = gpflash.GaussianProcess().fit(X_train_mat, f_train_vec, sig2_f_val=0.01)
    assert known.hp_val_struct().hp_sig2_f_val < 0.0


def test_rat_quad_and_multistart(data):
    X_train_mat, f_train_vec, _ = data
    optz_opts = make_optz_opts(gpflash.KernelId.RatQuad)
    optz_opts.use_multistart = True
    optz_opts.n_ind_optz = 3
    optz_opts.hypercube_sampling_seed = 11
    gp = gpflash.GaussianProcess(optz_opts).fit(X_train_mat, f_train_vec)
    assert gp.mll_optz_result().n_ind_optz_run() == 3
    assert gp.hp_val_struct().hp_kern_extra_vec.shape == (1,)


def test_mean_models(data):
    X_train_mat, f_train_vec, _ = data
    zero_mean_opts = make_optz_opts()
    zero_mean_opts.model_options.has_mean_model_constant = False
    zero_mean = gpflash.GaussianProcess(zero_mean_opts).fit(X_train_mat, f_train_vec)
    assert zero_mean.hp_beta_vec().shape == (0,)
    linear_mean_opts = make_optz_opts()
    linear_mean_opts.model_options.mean_model_degree_vec = [1]
    linear_mean = gpflash.GaussianProcess(linear_mean_opts).fit(X_train_mat, f_train_vec)
    assert linear_mean.hp_beta_vec().shape == (3,)  # the constant, then one weight per dimension


def test_eval_mll_and_make_posterior_at_given_hyperparameters(gp, data):
    X_train_mat, f_train_vec, X_post_mat = data
    hp_val_struct = gp.hp_val_struct()
    model = gpflash.GaussianProcess(make_optz_opts(gpflash.KernelId.SqExp))
    mll_result = model.eval_mll(X_train_mat, f_train_vec, hp_val_struct, return_mll_gradient=True)
    assert mll_result.mll == pytest.approx(gp.mll())
    assert mll_result.mll_hp_grad_struct.d_mll_d_hp_gamma_vec.shape == (2,)
    model.make_posterior(X_train_mat, f_train_vec, hp_val_struct)
    with pytest.raises(RuntimeError, match="needs a fit"):
        model.mll_optz_result()
    # The same factorization built twice; on some BLAS builds (Accelerate on x86_64) the two differ by rounding.
    np.testing.assert_allclose(predict_mean(model, X_post_mat), predict_mean(gp, X_post_mat), rtol=1e-8)
    with pytest.raises(RuntimeError, match="fit"):
        model.to_summary_string()


def test_options_are_set_once(data):
    X_train_mat, f_train_vec, _ = data
    optz_opts = make_optz_opts(gpflash.KernelId.Matern32)
    optz_opts.numerical_optz_algorithm = gpflash.NumericalOptzAlgorithm.Ccsaq
    # A bound holds in the fit: the unbounded optimum is above 0.5 in both dimensions.
    optz_opts.hp_gamma_ub_vec = np.array([0.5])
    gp = gpflash.GaussianProcess(optz_opts).fit(X_train_mat, f_train_vec, sig2_f_val=1e-6)
    assert gp.kernel_id() == gpflash.KernelId.Matern32
    assert np.all(gp.hp_val_struct().hp_gamma_vec <= 0.5 * (1.0 + 1e-12))
    # The constructor copied the options, and optz_opts() returns a copy, so neither edit reaches the model.
    optz_opts.model_options.kernel_id = gpflash.KernelId.SqExp
    gp.optz_opts().model_options.kernel_id = gpflash.KernelId.SqExp
    assert gp.optz_opts().model_options.kernel_id == gpflash.KernelId.Matern32
    assert gpflash.GaussianProcess().optz_opts().model_options.kernel_id == gpflash.KernelId.Matern52


def test_save_and_load(gp, data, tmp_path):
    _, _, X_post_mat = data
    path = tmp_path / "gp.h5"
    gp.save_precompute(path, note="wrapper test")
    loaded = gpflash.GaussianProcess.load_posterior(path)
    assert loaded.kernel_id() == gpflash.KernelId.SqExp
    assert loaded.optz_opts().model_options.has_mean_model_constant
    loaded_prediction = loaded.predict(X_post_mat)
    expected_prediction = gp.predict(X_post_mat)
    np.testing.assert_array_equal(loaded_prediction.mean_post_vec, expected_prediction.mean_post_vec)
    np.testing.assert_array_equal(loaded_prediction.var_post_vec, expected_prediction.var_post_vec)
    assert loaded.hp_val_struct().hp_sig2_k == gp.hp_val_struct().hp_sig2_k
    with pytest.raises(RuntimeError, match="fit"):
        loaded.to_summary_string()


def test_load_restores_mean_model_constant(data, tmp_path):
    # The file records whether the mean model has a constant, so the loaded model's options say so too.
    X_train_mat, f_train_vec, _ = data
    path = tmp_path / "zero_mean.h5"
    optz_opts = make_optz_opts(gpflash.KernelId.SqExp)
    optz_opts.model_options.has_mean_model_constant = False
    gpflash.GaussianProcess(optz_opts).fit(X_train_mat, f_train_vec).save_precompute(path)
    loaded_opts = gpflash.GaussianProcess.load_posterior(path).optz_opts()
    assert loaded_opts.model_options.kernel_id == gpflash.KernelId.SqExp
    assert not loaded_opts.model_options.has_mean_model_constant


def test_load_posterior_verified(gp, data, tmp_path):
    X_train_mat, f_train_vec, X_post_mat = data
    path = tmp_path / "gp.h5"
    gp.save_precompute(path)
    verified = gpflash.GaussianProcess(make_optz_opts(gpflash.KernelId.SqExp)).load_posterior_verified(
        path, X_train_mat, f_train_vec, gp.hp_val_struct()
    )
    np.testing.assert_array_equal(predict_mean(verified, X_post_mat), predict_mean(gp, X_post_mat))
    with pytest.raises(gpflash.InvalidInputError):
        gpflash.GaussianProcess(make_optz_opts(gpflash.KernelId.SqExp)).load_posterior_verified(
            path, X_train_mat, f_train_vec + 1.0, gp.hp_val_struct()
        )


def test_mean_only_file_cannot_return_variance(gp, data, tmp_path):
    _, _, X_post_mat = data
    path = tmp_path / "mean_only.h5"
    gp.save_precompute(path, include_variance_data=False)
    loaded = gpflash.GaussianProcess.load_posterior(path)
    np.testing.assert_array_equal(predict_mean(loaded, X_post_mat), predict_mean(gp, X_post_mat))
    with pytest.raises(gpflash.InvalidInputError):
        loaded.predict(X_post_mat)  # the variance is asked for by default


def test_input_errors(gp, data):
    X_train_mat, f_train_vec, _ = data
    with pytest.raises(RuntimeError, match="no posterior"):
        gpflash.GaussianProcess().predict(X_train_mat)
    # Every bad input raises InvalidInputError, a ValueError, from either layer.
    with pytest.raises(gpflash.InvalidInputError, match="columns"):
        gp.predict(np.ones((3, 5)))
    with pytest.raises(gpflash.InvalidInputError, match="one value per row"):
        gpflash.GaussianProcess().fit(X_train_mat, f_train_vec[:-1])
    with pytest.raises(gpflash.InvalidInputError, match="sig2_f_val"):
        gpflash.GaussianProcess().fit(X_train_mat, f_train_vec, sig2_f_val=float("nan"))
    with pytest.raises(gpflash.InvalidInputError, match="sig2_f_val"):
        gpflash.GaussianProcess().fit(X_train_mat, f_train_vec, sig2_f_val=float("inf"))
    with pytest.raises(TypeError, match="parse_kernel_id"):
        gpflash.GaussianProcess("sq_exp")
    X_nan_mat = X_train_mat.copy()
    X_nan_mat[0, 0] = np.nan
    with pytest.raises(gpflash.InvalidInputError):
        gpflash.GaussianProcess().fit(X_nan_mat, f_train_vec)


def test_concurrent_predictions_agree(gp, data):
    # The GIL is released during a query, so threads overlap; each must still get the serial answer.
    _, _, X_post_mat = data
    expected = predict_mean(gp, X_post_mat)
    results = [None] * 8

    def worker(i_thread):
        results[i_thread] = predict_mean(gp, X_post_mat)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(results))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for result in results:
        np.testing.assert_array_equal(result, expected)
