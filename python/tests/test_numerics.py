"""The C++ interface's numbers, against a direct NumPy implementation and finite differences.

The reference is the textbook GP with a squared-exponential kernel, k = exp(-0.5 ||gamma * (x - y)||^2), a
constant mean model whose weight is solved by generalized least squares, and a known noise variance. GpFlash adds a
nugget of about n_train / cond_max to K, which shifts these values by far less than the tolerances below.
"""

import numpy as np
import pytest

import gpflash

N_DIM, N_TRAIN = 2, 30
HP_GAMMA = np.array([1.7, 0.6])
HP_SIG2_K = 1.3
SIG2_F_VAL = 1e-3


@pytest.fixture(scope="module")
def problem():
    rng = np.random.default_rng(42)
    X_ndim_ntrain_mat = rng.random((N_DIM, N_TRAIN))
    f_train_vec = np.sin(3.0 * X_ndim_ntrain_mat[0]) + X_ndim_ntrain_mat[1] ** 2 + 0.01 * rng.standard_normal(N_TRAIN)
    train_data = gpflash.TrainingData()
    train_data.X_ndim_ntrain_mat = X_ndim_ntrain_mat
    train_data.f_train_vec = f_train_vec
    train_data.sig2_f_val = SIG2_F_VAL
    hp_val_struct = gpflash.HpVal()
    hp_val_struct.hp_gamma_vec = HP_GAMMA
    hp_val_struct.hp_sig2_k = HP_SIG2_K
    model_opts = gpflash.ModelOptions()
    model_opts.kernel_id = gpflash.KernelId.SqExp
    return train_data, hp_val_struct, model_opts


def sq_exp(X_a_mat, X_b_mat, hp_gamma_vec):
    diff = (X_a_mat[:, :, None] - X_b_mat[:, None, :]) * hp_gamma_vec[:, None, None]
    return np.exp(-0.5 * np.sum(diff**2, axis=0))


def reference_fit(X_ndim_ntrain_mat, f_train_vec, hp_gamma_vec, hp_sig2_k, sig2_f_val):
    n_train = X_ndim_ntrain_mat.shape[1]
    Sigma = hp_sig2_k * sq_exp(X_ndim_ntrain_mat, X_ndim_ntrain_mat, hp_gamma_vec) + sig2_f_val * np.eye(n_train)
    L = np.linalg.cholesky(Sigma)
    ones = np.ones(n_train)
    Sigma_inv_ones = np.linalg.solve(Sigma, ones)
    beta = Sigma_inv_ones @ f_train_vec / (Sigma_inv_ones @ ones)
    alpha = np.linalg.solve(Sigma, f_train_vec - beta)
    mll = -0.5 * (f_train_vec - beta) @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * n_train * np.log(2.0 * np.pi)
    return Sigma, beta, alpha, mll


def test_mll_matches_reference(problem):
    train_data, hp_val_struct, model_opts = problem
    X_ndim_ntrain_mat, f_train_vec = train_data.X_ndim_ntrain_mat, train_data.f_train_vec
    _, beta, _, mll = reference_fit(X_ndim_ntrain_mat, f_train_vec, HP_GAMMA, HP_SIG2_K, SIG2_F_VAL)
    mll_result = gpflash.eval_mll(train_data, hp_val_struct, model_opts)
    assert mll_result.mll == pytest.approx(mll, rel=1e-6)
    np.testing.assert_allclose(mll_result.hp_beta_vec, [beta], rtol=1e-6)


def test_mll_gradient_matches_finite_difference(problem):
    train_data, hp_val_struct, model_opts = problem
    mll_result = gpflash.eval_mll(train_data, hp_val_struct, model_opts, return_mll_gradient=True)
    mll_hp_grad_struct = mll_result.mll_hp_grad_struct
    step = 1e-6
    for i_dim in range(N_DIM):
        hp_plus_struct, hp_minus_struct = gpflash.HpVal(hp_val_struct), gpflash.HpVal(hp_val_struct)
        hp_plus_struct.hp_gamma_vec = HP_GAMMA + step * np.eye(N_DIM)[i_dim]
        hp_minus_struct.hp_gamma_vec = HP_GAMMA - step * np.eye(N_DIM)[i_dim]
        fd = (
            gpflash.eval_mll(train_data, hp_plus_struct, model_opts).mll
            - gpflash.eval_mll(train_data, hp_minus_struct, model_opts).mll
        ) / (2.0 * step)
        assert mll_hp_grad_struct.d_mll_d_hp_gamma_vec[i_dim] == pytest.approx(fd, rel=1e-5)
    hp_plus_struct, hp_minus_struct = gpflash.HpVal(hp_val_struct), gpflash.HpVal(hp_val_struct)
    hp_plus_struct.hp_sig2_k = HP_SIG2_K + step
    hp_minus_struct.hp_sig2_k = HP_SIG2_K - step
    fd = (
        gpflash.eval_mll(train_data, hp_plus_struct, model_opts).mll
        - gpflash.eval_mll(train_data, hp_minus_struct, model_opts).mll
    ) / (2.0 * step)
    assert mll_hp_grad_struct.d_mll_d_hp_sig2_k == pytest.approx(fd, rel=1e-5)


def test_posterior_matches_reference(problem):
    train_data, hp_val_struct, model_opts = problem
    X_ndim_ntrain_mat, f_train_vec = train_data.X_ndim_ntrain_mat, train_data.f_train_vec
    Sigma, beta, alpha, _ = reference_fit(X_ndim_ntrain_mat, f_train_vec, HP_GAMMA, HP_SIG2_K, SIG2_F_VAL)
    X_ndim_npost_mat = np.random.default_rng(7).random((N_DIM, 10))
    k_star = HP_SIG2_K * sq_exp(X_ndim_ntrain_mat, X_ndim_npost_mat, HP_GAMMA)
    mean_ref = beta + k_star.T @ alpha
    var_ref = HP_SIG2_K - np.sum(k_star * np.linalg.solve(Sigma, k_star), axis=0)

    posterior = gpflash.make_posterior(train_data, hp_val_struct, model_opts)
    posterior_result = posterior.eval(X_ndim_npost_mat)
    np.testing.assert_allclose(posterior_result.mean_post_vec, mean_ref, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(posterior_result.var_post_vec, var_ref, rtol=1e-4, atol=1e-7)


def test_posterior_gradient_matches_finite_difference(problem):
    train_data, hp_val_struct, model_opts = problem
    posterior = gpflash.make_posterior(train_data, hp_val_struct, model_opts)
    post_opts = gpflash.PosteriorOptions()
    post_opts.return_mean_gradient = True
    post_opts.return_var_gradient = True
    x_post_vec = np.array([0.3, 0.8])
    posterior_result = posterior.eval(x_post_vec, post_opts)
    step = 1e-6
    for i_dim in range(N_DIM):
        x_plus_vec, x_minus_vec = x_post_vec.copy(), x_post_vec.copy()
        x_plus_vec[i_dim] += step
        x_minus_vec[i_dim] -= step
        posterior_result_plus, posterior_result_minus = posterior.eval(x_plus_vec), posterior.eval(x_minus_vec)
        fd_mean = (posterior_result_plus.mean_post_vec[0] - posterior_result_minus.mean_post_vec[0]) / (2.0 * step)
        fd_var = (posterior_result_plus.var_post_vec[0] - posterior_result_minus.var_post_vec[0]) / (2.0 * step)
        d_mean = posterior_result.d_mean_post_d_x_post_ndim_npost_mat[i_dim, 0]
        d_var = posterior_result.d_var_post_d_x_post_ndim_npost_mat[i_dim, 0]
        assert d_mean == pytest.approx(fd_mean, rel=1e-5, abs=1e-8)
        assert d_var == pytest.approx(fd_var, rel=1e-4, abs=1e-8)


def test_fit_improves_mll_and_feeds_posterior(problem):
    train_data, hp_val_struct, model_opts = problem
    optz_opts = gpflash.OptzOptions()
    optz_opts.model_options = model_opts
    mll_optz_result = gpflash.select_hp_with_max_mll(train_data, optz_opts)
    assert mll_optz_result.mll() >= gpflash.eval_mll(train_data, hp_val_struct, model_opts).mll
    assert mll_optz_result.calc_n_failed_start() == 0
    assert mll_optz_result.calc_n_at_bound(mll_optz_result.idx_best()) == 0
    assert mll_optz_result.model_options.kernel_id == gpflash.KernelId.SqExp
    # The fit's own MLL, re-evaluated at its hyperparameters.
    mll_result = gpflash.eval_mll(train_data, mll_optz_result.hp_val_struct, mll_optz_result.model_options)
    assert mll_result.mll == pytest.approx(mll_optz_result.mll())
    posterior = gpflash.make_posterior_from_fit(train_data, mll_optz_result)
    assert posterior.mll() == pytest.approx(mll_optz_result.mll())
    assert "marginal log lik" in mll_optz_result.to_summary_string()


def test_save_and_load(problem, tmp_path):
    train_data, hp_val_struct, model_opts = problem
    posterior = gpflash.make_posterior(train_data, hp_val_struct, model_opts)
    path = str(tmp_path / "model.h5")
    posterior.save_precompute(path, "test note")
    X_ndim_npost_mat = np.random.default_rng(3).random((N_DIM, 8))
    expected_result = posterior.eval(X_ndim_npost_mat)

    loaded = gpflash.load_posterior(path)
    assert loaded.kernel_id() == gpflash.KernelId.SqExp
    assert loaded.input_hash() == posterior.input_hash()
    np.testing.assert_array_equal(loaded.eval(X_ndim_npost_mat).mean_post_vec, expected_result.mean_post_vec)
    np.testing.assert_array_equal(loaded.eval(X_ndim_npost_mat).var_post_vec, expected_result.var_post_vec)

    verified = gpflash.load_posterior_verified(path, train_data, hp_val_struct, model_opts)
    np.testing.assert_array_equal(verified.eval(X_ndim_npost_mat).mean_post_vec, expected_result.mean_post_vec)
    other_data = gpflash.TrainingData(train_data)
    other_data.f_train_vec = train_data.f_train_vec + 1.0
    with pytest.raises(gpflash.InvalidInputError):
        gpflash.load_posterior_verified(path, other_data, hp_val_struct, model_opts)
