"""The GaussianProcess wrapper: NumPy (n_points, n_dim) layout, fitting, prediction, saving."""

import threading

import numpy as np
import pytest

import gpflash


def target(X):
    return np.sin(3.0 * X[:, 0]) + X[:, 1] ** 2


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = rng.random((40, 2))
    return X, target(X), rng.random((25, 2))


@pytest.fixture(scope="module")
def gp(data):
    X, y, _ = data
    return gpflash.GaussianProcess("sq_exp").fit(X, y)


def test_noise_free_fit_interpolates(gp, data):
    # Nearly, not exactly: the nugget that keeps K factorizable, sized from cond_max, smooths the fit slightly.
    X, y, _ = data
    mean, var = gp.predict(X, return_var=True)
    np.testing.assert_allclose(mean, y, atol=1e-3)
    assert np.all(var < 1e-6)


def test_predicts_held_out_points(gp, data):
    _, _, X_test = data
    np.testing.assert_allclose(gp.predict(X_test), target(X_test), atol=1e-2)


def test_matches_core_interface(gp, data):
    X, y, X_test = data
    train_data = gpflash.TrainingData()
    train_data.X_ndim_ntrain_mat = X.T
    train_data.f_train_vec = y
    optz_opts = gpflash.OptzOptions()
    optz_opts.model_options.kernel_id = gpflash.KernelId.SqExp
    posterior = gpflash.fit_and_make_posterior(train_data, optz_opts)
    # Two fits of the same problem, so they agree to rounding: BLAS reduction order can differ between runs, which
    # moves where each optimization stops by ~1e-11, not bitwise.
    np.testing.assert_allclose(gp.predict(X_test), posterior.eval(X_test.T).mean_post_vec, rtol=1e-8)


def test_memory_layout_does_not_change_results(gp, data):
    # Every layout becomes the same buffer before the C++ call, so the results are bitwise equal.
    _, _, X_test = data
    expected = gp.predict(X_test)
    np.testing.assert_array_equal(gp.predict(np.asfortranarray(X_test)), expected)
    np.testing.assert_array_equal(gp.predict(np.repeat(X_test, 2, axis=0)[::2]), expected)
    np.testing.assert_array_equal(gp.predict(X_test.tolist()), expected)
    # A single point is a batch of one, which BLAS runs through different kernels than a batch of 25, so it agrees to
    # rounding rather than bitwise.
    np.testing.assert_allclose(gp.predict(X_test[0]), expected[:1], rtol=1e-10)


def test_predict_gradient_shapes_and_values(gp, data):
    _, _, X_test = data
    d_mean, d_var = gp.predict_gradient(X_test[:3])
    assert d_mean.shape == d_var.shape == (3, 2)
    step = 1e-6
    x_plus, x_minus = X_test[:1].copy(), X_test[:1].copy()
    x_plus[0, 1] += step
    x_minus[0, 1] -= step
    fd = (gp.predict(x_plus)[0] - gp.predict(x_minus)[0]) / (2.0 * step)
    assert d_mean[0, 1] == pytest.approx(fd, rel=1e-4)


def test_hyperparameters(gp):
    hp = gp.hyperparameters_
    assert hp["sig2_k"] > 0.0
    assert hp["sig2_f"] is None
    assert hp["gamma"].shape == (2,)
    assert hp["kern_extra"].shape == (0,)
    assert hp["beta"].shape == (1,)
    assert gp.log_marginal_likelihood_ == pytest.approx(gp.fit_result_.mll())
    assert "marginal log lik" in gp.summary()


def test_estimated_noise():
    rng = np.random.default_rng(1)
    X = rng.random((60, 1))
    y = np.sin(6.0 * X[:, 0]) + 0.1 * rng.standard_normal(60)
    gp = gpflash.GaussianProcess("matern_5_2", noise="estimate").fit(X, y)
    assert 1e-3 < gp.hyperparameters_["sig2_f"] < 1e-1


def test_rat_quad_and_multistart(data):
    X, y, _ = data
    gp = gpflash.GaussianProcess("rat_quad", n_starts=3, seed=11).fit(X, y)
    assert gp.fit_result_.n_ind_optz_run() == 3
    assert gp.hyperparameters_["kern_extra"].shape == (1,)


def test_zero_mean_has_no_weights(data):
    X, y, _ = data
    gp = gpflash.GaussianProcess(mean="zero").fit(X, y)
    assert gp.hyperparameters_["beta"].shape == (0,)


def test_from_options(data):
    X, y, _ = data
    optz_opts = gpflash.OptzOptions()
    optz_opts.model_options.kernel_id = gpflash.KernelId.Matern32
    optz_opts.numerical_optz_algorithm = gpflash.NumericalOptzAlgorithm.Ccsaq
    gp = gpflash.GaussianProcess.from_options(optz_opts, noise=1e-6).fit(X, y)
    assert gp.kernel == "matern_3_2"
    # The options were copied: a later edit does not reach the model.
    optz_opts.model_options.kernel_id = gpflash.KernelId.SqExp
    assert gp.kernel == "matern_3_2"


def test_save_and_load(gp, data, tmp_path):
    _, _, X_test = data
    path = tmp_path / "gp.h5"
    gp.save(path, note="wrapper test")
    loaded = gpflash.GaussianProcess.load(path)
    assert loaded.kernel == "sq_exp"
    mean, var = loaded.predict(X_test, return_var=True)
    expected_mean, expected_var = gp.predict(X_test, return_var=True)
    np.testing.assert_array_equal(mean, expected_mean)
    np.testing.assert_array_equal(var, expected_var)
    with pytest.raises(RuntimeError, match="fit"):
        loaded.summary()


def test_mean_only_file_cannot_return_variance(gp, data, tmp_path):
    _, _, X_test = data
    path = tmp_path / "mean_only.h5"
    gp.save(path, include_variance=False)
    loaded = gpflash.GaussianProcess.load(path)
    np.testing.assert_array_equal(loaded.predict(X_test), gp.predict(X_test))
    with pytest.raises(gpflash.InvalidInputError):
        loaded.predict(X_test, return_var=True)


def test_input_errors(gp, data):
    X, y, _ = data
    with pytest.raises(RuntimeError, match="not fitted"):
        gpflash.GaussianProcess().predict(X)
    with pytest.raises(ValueError, match="columns"):
        gp.predict(np.ones((3, 5)))
    with pytest.raises(ValueError, match="one value per row"):
        gpflash.GaussianProcess().fit(X, y[:-1])
    with pytest.raises(ValueError, match="noise"):
        gpflash.GaussianProcess(noise=-1.0)
    with pytest.raises(ValueError, match="noise"):
        gpflash.GaussianProcess(noise="guess")
    with pytest.raises(gpflash.InvalidInputError):
        gpflash.GaussianProcess("no_such_kernel")
    X_nan = X.copy()
    X_nan[0, 0] = np.nan
    with pytest.raises(gpflash.InvalidInputError):
        gpflash.GaussianProcess().fit(X_nan, y)


def test_concurrent_predictions_agree(gp, data):
    # The GIL is released during a query, so threads overlap; each must still get the serial answer.
    _, _, X_test = data
    expected = gp.predict(X_test)
    results = [None] * 8

    def worker(i_thread):
        results[i_thread] = gp.predict(X_test)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(results))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for result in results:
        np.testing.assert_array_equal(result, expected)
