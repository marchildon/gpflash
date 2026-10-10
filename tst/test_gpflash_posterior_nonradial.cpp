// Posterior evaluation for the kernels test_gpflash_posterior.cpp cannot reach. What this file asserts:
//
//   1) quadratic_vs_dense    Quadratic's mean and variance match a dense reference built from the definition
//   2) quadratic_grad_vs_fd  both Quadratic posterior gradients match finite differences
//   3) periodic_grad_vs_fd   both Periodic posterior gradients match finite differences, in raw input space
//   4) batch_matches_single  the batch path agrees with the single-query path at every column, for both kernels
//   5) load_posterior        the runtime factory reloads a saved fit and reproduces its predictions
//
// test_gpflash_posterior.cpp sweeps the four radial kernels against a dense reference whose diagonal is 1 and
// whose points are raw, so neither kernel here fits it: Quadratic's k(x, x) varies with the point, and
// Periodic is evaluated on mapped features. The gradients are the reason the file exists:
//
//   - Quadratic reaches calc_d_kern_diag_val_d_x_post with a dk/d(inner_prod) that varies with the query
//     point, which Linear's constant 1 cannot exercise (test_gpflash_linear.cpp covers Linear);
//   - Periodic folds its feature-space sums back onto a raw query point, the one gradient in the package
//     whose inputs and outputs live in different coordinate systems. A mistake there leaves the kernel
//     values correct, so test_gpflash_periodic.cpp cannot see it.
#include <algorithm>  // std::max over an initializer list
#include <armadillo>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>

#include "GpFlashTestSupport.hpp"
#include "gpflash/kernels/KernelDispatch.hpp"  // kernel_by_id
#include "gpflash/posterior_build/PosteriorBuild.hpp"
#include "gpflash/posterior_eval/GpPosterior.hpp"  // GpPosterior

using gpflash_test::draw_uniform_vec;
using gpflash_test::kDetailLen;
using gpflash_test::kFdScaleFloor;
using gpflash_test::kFdStepKern;
using gpflash_test::kTolAltPath;
using gpflash_test::kTolFd;
using gpflash_test::make_points;
using gpflash_test::report;

namespace {

// Every field, so one call yields the values and both gradients.
const gpflash::PosteriorOptions kOptsAll{
        /*return_mean=*/true, /*return_var=*/true, /*return_mean_gradient=*/true, /*return_var_gradient=*/true};
const gpflash::PosteriorOptions kOptsMeanVar{true, true, false, false};

// The inputs shared by the cases below. hp_sig2_k and the noise are non-unit and mutually unequal so a
// dropped or swapped factor cannot cancel; cond_max stays well below the default because these references
// invert Sigma densely in absolute coordinates, whose conditioning the nugget does not bound.
struct PostCase {
    gpflash::TrainingData train_data;
    gpflash::HpVal hp_val_struct;
    gpflash::ModelOptions model_opts;
    arma::vec x_post_vec;
};

PostCase make_post_case(
        const gpflash::kernel::KernelId kernel_id, const int n_dim, const int n_train, const bool has_hp_amplitude) {
    PostCase test_case;
    test_case.train_data.X_ndim_ntrain_mat = make_points(n_dim, n_train);
    std::mt19937_64 rng(19);
    test_case.train_data.f_train_vec = draw_uniform_vec(rng, n_train, 1.5, 2.5);
    test_case.train_data.sig2_f_val = 0.017;

    test_case.hp_val_struct.hp_gamma_vec.set_size(n_dim);
    for (int i_dim = 0; i_dim < n_dim; ++i_dim) test_case.hp_val_struct.hp_gamma_vec(i_dim) = 0.7 + 0.25 * i_dim;
    if (has_hp_amplitude) {
        // The periodic amplitudes, distinct per dimension so a tie error shows.
        test_case.hp_val_struct.hp_kern_extra_vec.set_size(n_dim);
        for (int i_dim = 0; i_dim < n_dim; ++i_dim)
            test_case.hp_val_struct.hp_kern_extra_vec(i_dim) = 0.8 + 0.2 * i_dim;
    }
    test_case.hp_val_struct.hp_sig2_k = 1.9;

    test_case.model_opts.kernel_id = kernel_id;
    test_case.model_opts.cond_max = 1.0e6;
    // Away from the training points, so the gradients are not near zero where a relative check is meaningless.
    test_case.x_post_vec = 0.35 + 0.25 * make_points(n_dim, 1).col(0);

    return test_case;
}

// Asserts Quadratic's mean and variance match a dense reference formed from the kernel definition.
//
// The variance is the check that matters: its leading term is k(x', x') = (||xdot'||^2 + 1)^2 rather than the
// 1 every stationary kernel contributes, so a reference with a unit prior variance would not catch a missing
// or misplaced diagonal term.
bool test_quadratic_vs_dense(const char* name) {
    const int n_dim = 3, n_train = 14;
    const PostCase test_case =
            make_post_case(gpflash::kernel::KernelId::Quadratic, n_dim, n_train, /*has_hp_amplitude=*/false);
    const auto posterior = gpflash::make_posterior(test_case.train_data, test_case.hp_val_struct, test_case.model_opts);

    // Reference: K from the definition (g + 1)^2, then Sigma = hp_sig2_k (K + eta P P) + sigma_f^2 I.
    const arma::vec& hp_gamma_vec = test_case.hp_val_struct.hp_gamma_vec;
    const double hp_sig2_k = test_case.hp_val_struct.hp_sig2_k;
    const arma::mat Xdot_ndim_ntrain_mat = test_case.train_data.X_ndim_ntrain_mat.each_col() % hp_gamma_vec;
    const arma::mat gram_mat = Xdot_ndim_ntrain_mat.t() * Xdot_ndim_ntrain_mat;
    const arma::mat K_full_mat = arma::square(gram_mat + 1.0);

    arma::mat K_corr_mat_lower = arma::trimatl(K_full_mat);
    const arma::vec precon_scale_vec = gpflash::detail::calc_precon_scale_vec(K_corr_mat_lower, "test");
    gpflash::detail::scale_to_unit_diag_inplace(K_corr_mat_lower, 1.0 / precon_scale_vec);
    gpflash::detail::EtaTerms kmat_eta_terms;
    gpflash::detail::fill_eta_bound_data(K_corr_mat_lower, /*want_active_row=*/false, kmat_eta_terms);
    const double eta = gpflash::detail::calc_eta(kmat_eta_terms.eig_max_ub_M_precon, test_case.model_opts.cond_max);

    arma::mat Sigma_mat = hp_sig2_k * K_full_mat;
    Sigma_mat.diag() += hp_sig2_k * eta * arma::square(precon_scale_vec) + test_case.train_data.sig2_f_val;
    const arma::mat Sigma_inv_mat = arma::inv_sympd(Sigma_mat);
    // The reference is the inaccurate side: it inverts Sigma in absolute coordinates while the library
    // factorizes the preconditioned matrix, so the tolerance follows the reference's own conditioning.
    const double tol_post = std::max(1.0e-10, 1.0e2 * arma::cond(Sigma_mat) * arma::datum::eps);

    const arma::vec ones_vec(n_train, arma::fill::ones);
    const double hp_beta = arma::dot(ones_vec, Sigma_inv_mat * test_case.train_data.f_train_vec)
            / arma::dot(ones_vec, Sigma_inv_mat * ones_vec);
    const arma::vec lin_sol_vec = Sigma_inv_mat * (test_case.train_data.f_train_vec - hp_beta * ones_vec);

    const arma::vec xdot_post_vec = hp_gamma_vec % test_case.x_post_vec;
    arma::vec k_post_vec(n_train);
    for (int i_train = 0; i_train < n_train; ++i_train) {
        const double xdot_inner_prod = arma::dot(Xdot_ndim_ntrain_mat.col(i_train), xdot_post_vec);
        k_post_vec(i_train) = (xdot_inner_prod + 1.0) * (xdot_inner_prod + 1.0);
    }
    const double mean_ref = hp_beta + hp_sig2_k * arma::dot(k_post_vec, lin_sol_vec);
    // k(x', x') = (||xdot'||^2 + 1)^2, the term a unit-diagonal reference would get wrong.
    const double prior_var = arma::dot(xdot_post_vec, xdot_post_vec) + 1.0;
    const double var_ref =
            hp_sig2_k * (prior_var * prior_var - hp_sig2_k * arma::dot(k_post_vec, Sigma_inv_mat * k_post_vec));

    const gpflash::PosteriorResult post_result = posterior.eval(test_case.x_post_vec, kOptsMeanVar);
    const double mean_err = std::fabs(post_result.mean_post_vec(0) - mean_ref) / std::max(std::fabs(mean_ref), 1.0);
    // Normalized by the prior variance: with n_train well above the kernel's rank the posterior variance is
    // genuinely near zero, so dividing by it would measure cancellation rather than agreement.
    const double var_err = std::fabs(post_result.var_post_vec(0) - var_ref) / (hp_sig2_k * prior_var * prior_var);

    char detail[kDetailLen];
    std::snprintf(detail, sizeof(detail), "mean %.2e  var %.2e  (tol %.1e)", mean_err, var_err, tol_post);

    return report(name, mean_err < tol_post && var_err < tol_post, detail);
}

// Asserts both posterior gradients match central finite differences of the library's own mean and variance.
//
// Differencing the library's own values rather than a reference isolates the gradient code: the values are
// already checked elsewhere, so any disagreement here is in the gradient path alone. For Periodic this is the
// only check that the feature-space sums fold back onto the raw query point correctly.
bool test_posterior_grad_vs_fd(const char* name, const PostCase& test_case) {
    const int n_dim = static_cast<int>(test_case.x_post_vec.n_elem);
    const auto posterior = gpflash::make_posterior(test_case.train_data, test_case.hp_val_struct, test_case.model_opts);
    const gpflash::PosteriorResult grad_result = posterior.eval(test_case.x_post_vec, kOptsAll);

    double worst_mean = 0.0, worst_var = 0.0;
    for (int i_dim = 0; i_dim < n_dim; ++i_dim) {
        arma::vec x_post_plus_vec = test_case.x_post_vec, x_post_minus_vec = test_case.x_post_vec;
        x_post_plus_vec(i_dim) += kFdStepKern;
        x_post_minus_vec(i_dim) -= kFdStepKern;
        const gpflash::PosteriorResult post_result_plus = posterior.eval(x_post_plus_vec, kOptsMeanVar);
        const gpflash::PosteriorResult post_result_minus = posterior.eval(x_post_minus_vec, kOptsMeanVar);

        const double fd_mean =
                (post_result_plus.mean_post_vec(0) - post_result_minus.mean_post_vec(0)) / (2.0 * kFdStepKern);
        const double fd_var =
                (post_result_plus.var_post_vec(0) - post_result_minus.var_post_vec(0)) / (2.0 * kFdStepKern);
        worst_mean = std::max(
                worst_mean,
                std::fabs(grad_result.d_mean_post_d_x_post_ndim_npost_mat(i_dim, 0) - fd_mean)
                        / std::max(std::fabs(fd_mean), kFdScaleFloor));
        worst_var = std::max(
                worst_var,
                std::fabs(grad_result.d_var_post_d_x_post_ndim_npost_mat(i_dim, 0) - fd_var)
                        / std::max(std::fabs(fd_var), kFdScaleFloor));
    }
    // The gradient must be reported in raw input space, so its height is the query width even where the kernel
    // is evaluated on wider features.
    const bool shape_ok = grad_result.d_mean_post_d_x_post_ndim_npost_mat.n_rows == static_cast<arma::uword>(n_dim)
            && grad_result.d_var_post_d_x_post_ndim_npost_mat.n_rows == static_cast<arma::uword>(n_dim);

    char detail[kDetailLen];
    std::snprintf(detail, sizeof(detail), "mean %.2e  var %.2e  shape %d", worst_mean, worst_var, shape_ok ? 1 : 0);

    return report(name, shape_ok && worst_mean < kTolFd && worst_var < kTolFd, detail);
}

// Asserts the batch path agrees with the single-query path it duplicates, values and gradients, at every column.
//
// The two reach the same answer through different BLAS levels, so this is where a transposition, a stride, or a
// per-column scaling error in the batch path shows up.
bool test_batch_matches_single(const char* name, const PostCase& test_case) {
    const int n_dim = static_cast<int>(test_case.x_post_vec.n_elem), n_post = 4;
    const auto posterior = gpflash::make_posterior(test_case.train_data, test_case.hp_val_struct, test_case.model_opts);

    // Column 0 is the case's own query point; the rest are distinct, so a column mix-up changes the answer.
    arma::mat X_ndim_npost_mat(n_dim, n_post);
    X_ndim_npost_mat.col(0) = test_case.x_post_vec;
    for (int i_post = 1; i_post < n_post; ++i_post)
        for (int i_dim = 0; i_dim < n_dim; ++i_dim)
            X_ndim_npost_mat(i_dim, i_post) = 0.3 * std::cos(1.3 * i_post + 0.7 * i_dim);
    const gpflash::PosteriorResult post_result_batch = posterior.eval(X_ndim_npost_mat, kOptsAll);

    // Each column against a single query at that column's point.
    double worst = 0.0;
    for (int i_post = 0; i_post < n_post; ++i_post) {
        const arma::uword i_col = static_cast<arma::uword>(i_post);
        const arma::vec x_post_vec = X_ndim_npost_mat.col(i_col);
        const gpflash::PosteriorResult post_result_single = posterior.eval(x_post_vec, kOptsAll);
        worst = std::max(
                {worst,
                 std::fabs(post_result_batch.mean_post_vec(i_col) - post_result_single.mean_post_vec(0)),
                 std::fabs(post_result_batch.var_post_vec(i_col) - post_result_single.var_post_vec(0)),
                 arma::abs(
                         post_result_batch.d_mean_post_d_x_post_ndim_npost_mat.col(i_col)
                         - post_result_single.d_mean_post_d_x_post_ndim_npost_mat.col(0))
                         .max(),
                 arma::abs(
                         post_result_batch.d_var_post_d_x_post_ndim_npost_mat.col(i_col)
                         - post_result_single.d_var_post_d_x_post_ndim_npost_mat.col(0))
                         .max()});
    }

    const arma::uword n_post_u = static_cast<arma::uword>(n_post);
    const bool shape_ok = post_result_batch.mean_post_vec.n_elem == n_post_u
            && post_result_batch.var_post_vec.n_elem == n_post_u
            && post_result_batch.d_mean_post_d_x_post_ndim_npost_mat.n_cols == n_post_u
            && post_result_batch.d_var_post_d_x_post_ndim_npost_mat.n_cols == n_post_u;
    char detail[kDetailLen];
    std::snprintf(detail, sizeof(detail), "worst abs %.2e  shape %d", worst, shape_ok ? 1 : 0);
    return report(name, shape_ok && worst < kTolAltPath, detail);
}

// Asserts load_posterior reloads a saved fit and reproduces its predictions.
//
// This is the one-call path a caller uses when the kernel is not known at compile time: the id comes from the
// file's own header, so nothing in the call names the kernel. Run on Periodic because a reloaded periodic
// model must still map its query points, which needs the stored hyperparameters rather than just L.
bool test_load_posterior(const char* name) {
    const int n_dim = 2, n_train = 12;
    const PostCase test_case =
            make_post_case(gpflash::kernel::KernelId::Periodic, n_dim, n_train, /*has_hp_amplitude=*/true);
    const std::string path = "test_posterior_nonradial_load.bin";

    const auto saved = gpflash::make_posterior(test_case.train_data, test_case.hp_val_struct, test_case.model_opts);
    saved.save_precompute(path);
    const auto loaded = gpflash::load_posterior(path);
    std::remove(path.c_str());

    const gpflash::PosteriorResult post_result_saved = saved.eval(test_case.x_post_vec, kOptsAll);
    const gpflash::PosteriorResult post_result_loaded = loaded.eval(test_case.x_post_vec, kOptsAll);

    const double worst = std::max(
            {std::fabs(post_result_loaded.mean_post_vec(0) - post_result_saved.mean_post_vec(0)),
             std::fabs(post_result_loaded.var_post_vec(0) - post_result_saved.var_post_vec(0)),
             arma::abs(
                     post_result_loaded.d_mean_post_d_x_post_ndim_npost_mat
                     - post_result_saved.d_mean_post_d_x_post_ndim_npost_mat)
                     .max(),
             arma::abs(
                     post_result_loaded.d_var_post_d_x_post_ndim_npost_mat
                     - post_result_saved.d_var_post_d_x_post_ndim_npost_mat)
                     .max()});
    const bool dims_ok = loaded.n_train() == saved.n_train() && loaded.n_dim() == n_dim;

    char detail[kDetailLen];
    std::snprintf(detail, sizeof(detail), "worst abs %.2e  dims %d", worst, dims_ok ? 1 : 0);

    return report(name, dims_ok && worst < kTolAltPath, detail);
}

}  // namespace

int main() {
    std::printf("== GpFlash posterior, non-radial kernels ==\n");
    bool ok = true;
    const PostCase quadratic_case = make_post_case(
            gpflash::kernel::KernelId::Quadratic, /*n_dim=*/3, /*n_train=*/14, /*has_hp_amplitude=*/false);
    const PostCase periodic_case =
            make_post_case(gpflash::kernel::KernelId::Periodic, /*n_dim=*/3, /*n_train=*/16, /*has_hp_amplitude=*/true);

    ok &= test_quadratic_vs_dense("quadratic_vs_dense");
    ok &= test_posterior_grad_vs_fd("grad_vs_fd/Quadratic", quadratic_case);
    ok &= test_posterior_grad_vs_fd("grad_vs_fd/Periodic", periodic_case);
    ok &= test_batch_matches_single("batch_matches_single/Quadratic", quadratic_case);
    ok &= test_batch_matches_single("batch_matches_single/Periodic", periodic_case);
    ok &= test_load_posterior("load_posterior");
    std::printf("\n%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");

    return ok ? 0 : 1;
}
