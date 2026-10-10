// The linear kernel end to end: kernel matrix, conditioning of a singular K, posterior, input-space
// gradient, recovery of a linear function, a training point at the origin, and runtime dispatch.
//
// Every check here is one a stationary kernel cannot reach: k(x, x) is constant for those and varies per point
// for the inner-product family. Quadratic shares that property; the family's analytic hyperparameter gradient
// is checked against finite differences in test_gpflash_inner_product_gradient.cpp.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>

#include "GpFlashTestSupport.hpp"
#include "gpflash/assembly/KernelMatrix.hpp"
#include "gpflash/inputs/HpVal.hpp"
#include "gpflash/inputs/ModelOptions.hpp"
#include "gpflash/inputs/TrainingData.hpp"
#include "gpflash/kernels/KernelDispatch.hpp"
#include "gpflash/posterior_build/PosteriorBuild.hpp"
#include "gpflash/posterior_eval/GpPosterior.hpp"
#include "gpflash/training/EvalMllDispatch.hpp"  // eval_mll

using gpflash_test::draw_uniform_vec;
using gpflash_test::kCondBoundSlack;
using gpflash_test::kDetailLen;
using gpflash_test::kFdScaleFloor;
using gpflash_test::kRelScaleFloor;
using gpflash_test::kTolAltPath;
using gpflash_test::kTolExact;
using gpflash_test::kTolFd;
using gpflash_test::make_points;
using gpflash_test::report;

namespace {

// A linear GP recovers a linear function exactly only as the nugget goes to zero; cond_max = 1e12 leaves a bias
// that scales as 1/cond_max. The bound sits two orders above it, so a nugget a hundred times heavier fails.
constexpr double kTolLinearRecovery = 1e-9;

// The kernel's offset, stated here rather than read from the library so that changing the library's value fails the
// references below.
constexpr double kXdotInnerProdOffset = 1.0;

// Seeds each test's own std::mt19937_64, so its data depends neither on the Armadillo version nor on test order.
constexpr std::uint64_t kDataSeed = 31;

// The posterior mean is linear and the variance quadratic in x', so central differences carry no truncation error
// and only rounding, ~eps |f| / h, remains: a step far above kFdStepKern shrinks it.
constexpr double kFdStepLinear = 1e-2;

arma::vec make_hp_gamma_vec(const int n_dim) {
    arma::vec hp_gamma_vec(n_dim);
    for (int i_dim = 0; i_dim < n_dim; ++i_dim) hp_gamma_vec(i_dim) = 0.7 + 0.4 * i_dim;
    return hp_gamma_vec;
}

// The kernel matrix against the definition sum_d hp_gamma_d^2 x_id x_jd + kXdotInnerProdOffset, diagonal included.
bool test_kern_mat_vs_definition(const char* name) {
    const int n_dim = 3, n_train = 9;
    const arma::mat X_ndim_ntrain_mat = make_points(n_dim, n_train);
    const arma::vec hp_gamma_vec = make_hp_gamma_vec(n_dim);
    const arma::mat Ydot_nfeature_ntrain_mat = X_ndim_ntrain_mat.each_col() % hp_gamma_vec;

    // K comes back raw, so this compares against the definition.
    const arma::mat K_mat = gpflash::detail::build_kmat_lower_tri(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Linear),
            Ydot_nfeature_ntrain_mat,
            nullptr,
            /*store_hp_grad_factors=*/true);

    double worst = 0.0, worst_diag = 0.0;
    for (int i_col = 0; i_col < n_train; ++i_col)
        for (int i_row = i_col; i_row < n_train; ++i_row) {
            double kern_ref = kXdotInnerProdOffset;
            for (int i_dim = 0; i_dim < n_dim; ++i_dim)
                kern_ref += hp_gamma_vec(i_dim) * hp_gamma_vec(i_dim) * X_ndim_ntrain_mat(i_dim, i_row)
                        * X_ndim_ntrain_mat(i_dim, i_col);
            const double max_err =
                    std::fabs(K_mat(i_row, i_col) - kern_ref) / std::max(std::fabs(kern_ref), kRelScaleFloor);
            if (i_row == i_col)
                worst_diag = std::max(worst_diag, max_err);
            else
                worst = std::max(worst, max_err);
        }

    // Rank at most n_dim + 1, so K is singular whenever n_train > n_dim + 1. This is the property that makes the
    // nugget load-bearing for the Cholesky rather than a safety net.
    const arma::uword rank_k = arma::rank(arma::symmatl(K_mat));
    const bool rank_ok = rank_k == static_cast<arma::uword>(n_dim + 1);

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "off-diag %.2e  diag %.2e  rank %llu of n_train %d",
            worst,
            worst_diag,
            static_cast<unsigned long long>(rank_k),
            n_train);

    return report(name, worst < kTolExact && worst_diag < kTolExact && rank_ok, detail);
}

// A singular K must still factorize, and the nugget must deliver the promised conditioning. Neither holds
// without the preconditioned path: eta measured on K rather than K_corr does not bound kappa.
bool test_singular_kmat_is_conditioned(const char* name) {
    const int n_dim = 3, n_train = 20;
    const arma::mat X_ndim_ntrain_mat = make_points(n_dim, n_train);
    const arma::vec hp_gamma_vec = make_hp_gamma_vec(n_dim);
    const arma::mat Ydot_nfeature_ntrain_mat = X_ndim_ntrain_mat.each_col() % hp_gamma_vec;

    double worst_ratio = 0.0, worst_unit_diag = 0.0, worst_eta_err = 0.0;
    bool all_ok = true;
    for (const double cond_max : {1.0e4, 1.0e8}) {
        gpflash::detail::EtaTerms kmat_eta_terms;
        arma::mat K_corr_mat_lower = gpflash::detail::build_kmat_lower_tri(
                gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Linear),
                Ydot_nfeature_ntrain_mat,
                nullptr,
                /*store_hp_grad_factors=*/false);
        kmat_eta_terms = gpflash::detail::build_eta_terms(
                gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Linear).is_kern_diag_unity(),
                K_corr_mat_lower,
                cond_max,
                gpflash::EigMaxUbMethod::Gershgorin,
                /*want_active_row=*/false);
        // The whole point of the mapping: whatever diag(K) was, K_corr's is exactly 1.
        for (int i_diag = 0; i_diag < n_train; ++i_diag)
            worst_unit_diag = std::max(worst_unit_diag, std::fabs(K_corr_mat_lower(i_diag, i_diag) - 1.0));
        if (kmat_eta_terms.precon_scales.precon_scale_vec.n_elem != static_cast<arma::uword>(n_train)) all_ok = false;

        // Gershgorin eta from the dense K_corr: the largest full absolute row sum over (cond_max - 1). kappa below
        // bounds eta from one side only, which a nugget too heavy also satisfies.
        const double eta_ref = arma::sum(arma::abs(arma::symmatl(K_corr_mat_lower)), 1).max() / (cond_max - 1.0);
        const double eta_err = std::fabs(kmat_eta_terms.eta - eta_ref) / eta_ref;
        worst_eta_err = std::max(worst_eta_err, eta_err);
        if (!(eta_err <= kTolAltPath)) all_ok = false;

        arma::mat Sigma_corr_mat_lower = K_corr_mat_lower;
        gpflash::detail::add_nugget_and_noise_to_diag_inplace(
                Sigma_corr_mat_lower,
                kmat_eta_terms.eta,
                /*sig2_f_val=*/0.0,
                /*hp_sig2_k=*/1.0,
                kmat_eta_terms.precon_scales.inv_precon_scale_vec);
        const arma::vec eig_vec = arma::eig_sym(arma::symmatl(Sigma_corr_mat_lower));
        const double kappa = eig_vec.max() / eig_vec.min();
        worst_ratio = std::max(worst_ratio, kappa / cond_max);
        if (!(kappa <= cond_max * (1.0 + kCondBoundSlack))) all_ok = false;
    }
    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "max |diag(K_corr) - 1| %.2e  worst kappa/cond_max %.4f  eta rel err %.1e",
            worst_unit_diag,
            worst_ratio,
            worst_eta_err);

    return report(name, all_ok && worst_unit_diag < kTolExact, detail);
}

// The posterior mean and variance against a dense reference built from the definition. The variance is the
// check that matters here: its leading term is k(x', x') = ||ydot'||^2 + kXdotInnerProdOffset rather than 1.
bool test_posterior_vs_dense_reference(const char* name, const double sig2_f_val) {  // NOLINT
    const int n_dim = 3, n_train = 15, n_post = 4;
    const arma::mat X_ndim_ntrain_mat = make_points(n_dim, n_train);
    std::mt19937_64 rng(kDataSeed);
    const arma::vec f_train_vec = draw_uniform_vec(rng, n_train, 1.5, 2.5);
    const arma::vec hp_gamma_vec = make_hp_gamma_vec(n_dim);
    const arma::mat X_ndim_npost_mat = 0.4 + 0.3 * make_points(n_dim, n_post);

    gpflash::TrainingData train_data;
    train_data.X_ndim_ntrain_mat = X_ndim_ntrain_mat;
    train_data.f_train_vec = f_train_vec;
    train_data.sig2_f_val = sig2_f_val;
    gpflash::HpVal hp_val_struct;
    hp_val_struct.hp_gamma_vec = hp_gamma_vec;
    hp_val_struct.hp_sig2_k = 1.7;
    gpflash::ModelOptions model_opts;
    model_opts.kernel_id = gpflash::kernel::KernelId::Linear;
    model_opts.cond_max = 1.0e6;

    const auto posterior = gpflash::make_posterior(train_data, hp_val_struct, model_opts);

    // Reference: Sigma = hp_sig2_k (K + eta P P) + sigma_f^2 I, formed densely from the definition.
    const arma::mat Ydot_nfeature_ntrain_mat = X_ndim_ntrain_mat.each_col() % hp_gamma_vec;
    const arma::mat K_full_mat = Ydot_nfeature_ntrain_mat.t() * Ydot_nfeature_ntrain_mat + kXdotInnerProdOffset;
    arma::mat K_corr_mat_lower = arma::trimatl(K_full_mat);
    const arma::vec precon_scale_vec = gpflash::detail::calc_precon_scale_vec(K_corr_mat_lower, "test");
    const arma::vec inv_precon_scale_vec = 1.0 / precon_scale_vec;
    gpflash::detail::scale_to_unit_diag_inplace(K_corr_mat_lower, inv_precon_scale_vec);
    gpflash::detail::EtaTerms kmat_eta_terms;
    gpflash::detail::fill_eta_bound_data(K_corr_mat_lower, /*want_active_row=*/false, kmat_eta_terms);
    const double eta = gpflash::detail::calc_eta(kmat_eta_terms.eig_max_ub_M_precon, model_opts.cond_max);

    arma::mat Sigma_mat = hp_val_struct.hp_sig2_k * K_full_mat;
    Sigma_mat.diag() += hp_val_struct.hp_sig2_k * eta * arma::square(precon_scale_vec) + sig2_f_val;
    // The reference is the inaccurate side here: it inverts Sigma in absolute coordinates,
    // whose condition number the nugget does not bound, while the library factorizes the preconditioned
    // matrix whose condition number it does. The tolerance therefore scales with the reference's own
    // conditioning rather than being a fixed number.
    const arma::mat Sigma_inv_mat = arma::inv_sympd(Sigma_mat);
    const double cond_sigma = arma::cond(Sigma_mat);
    const double tol_post = std::max(1.0e-11, 1.0e2 * cond_sigma * arma::datum::eps);
    const arma::vec one_vec(n_train, arma::fill::ones);
    const double hp_beta =
            arma::dot(one_vec, Sigma_inv_mat * f_train_vec) / arma::dot(one_vec, Sigma_inv_mat * one_vec);
    const arma::vec lin_sol_vec = Sigma_inv_mat * (f_train_vec - hp_beta * one_vec);

    double worst_mean = 0.0, worst_var = 0.0;
    for (int i_post = 0; i_post < n_post; ++i_post) {
        const arma::vec ydot_post_vec = hp_gamma_vec % X_ndim_npost_mat.col(i_post);
        // k_post_vec excludes hp_sig2_k, so the cross-covariance hp_sig2_k k_post_vec carries one factor of it.
        const arma::vec k_post_vec = Ydot_nfeature_ntrain_mat.t() * ydot_post_vec + kXdotInnerProdOffset;
        const double mean_ref = hp_beta + hp_val_struct.hp_sig2_k * arma::dot(k_post_vec, lin_sol_vec);
        // k(x', x') = ||ydot'||^2 + kXdotInnerProdOffset leads it, where a correlation matrix would put 1.
        const double var_ref = hp_val_struct.hp_sig2_k
                * (arma::dot(ydot_post_vec, ydot_post_vec) + kXdotInnerProdOffset
                   - hp_val_struct.hp_sig2_k * arma::dot(k_post_vec, Sigma_inv_mat * k_post_vec));

        gpflash::PosteriorOptions post_opts;
        post_opts.return_mean = true;
        post_opts.return_var = true;
        const auto post_result = posterior.eval(arma::vec(X_ndim_npost_mat.col(i_post)), post_opts);
        worst_mean = std::max(
                worst_mean,
                std::fabs(post_result.mean_post_vec(0) - mean_ref) / std::max(std::fabs(mean_ref), kFdScaleFloor));
        // Normalized by the prior variance, not the posterior one. With n_train > n_dim the training points
        // span the input space, so a linear kernel's posterior variance is genuinely near zero everywhere and
        // dividing by it would measure cancellation rather than agreement.
        const double prior_var =
                hp_val_struct.hp_sig2_k * (arma::dot(ydot_post_vec, ydot_post_vec) + kXdotInnerProdOffset);
        worst_var = std::max(worst_var, std::fabs(post_result.var_post_vec(0) - var_ref) / prior_var);
    }

    // The batch shape must agree with the single-query shape it duplicates.
    gpflash::PosteriorOptions batch_opts;
    batch_opts.return_mean = true;
    batch_opts.return_var = true;
    const auto batch_result = posterior.eval(X_ndim_npost_mat, batch_opts);
    // Normalized the same way as above, because the two shapes reach the same answer through different BLAS
    // calls and so differ by rounding that grows with kappa(Sigma), not by anything structural.
    double worst_batch = 0.0;
    for (int i_post = 0; i_post < n_post; ++i_post) {
        gpflash::PosteriorOptions one_opts;
        one_opts.return_mean = true;
        one_opts.return_var = true;
        const auto one_result = posterior.eval(arma::vec(X_ndim_npost_mat.col(i_post)), one_opts);
        const arma::vec ydot_post_vec = hp_gamma_vec % X_ndim_npost_mat.col(i_post);
        const double prior_var =
                hp_val_struct.hp_sig2_k * (arma::dot(ydot_post_vec, ydot_post_vec) + kXdotInnerProdOffset);
        worst_batch = std::max(
                worst_batch,
                std::fabs(batch_result.mean_post_vec(i_post) - one_result.mean_post_vec(0))
                        / std::max(std::fabs(one_result.mean_post_vec(0)), kFdScaleFloor));
        worst_batch = std::max(
                worst_batch, std::fabs(batch_result.var_post_vec(i_post) - one_result.var_post_vec(0)) / prior_var);
    }

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "sigma_f^2 %.3f  mean %.2e  var %.2e  batch %.2e  (kappa(Sigma) %.1e, tol %.1e)",
            sig2_f_val,
            worst_mean,
            worst_var,
            worst_batch,
            cond_sigma,
            tol_post);

    return report(name, worst_mean < tol_post && worst_var < tol_post && worst_batch < tol_post, detail);
}

// The posterior gradients against finite differences of the mean and variance. The variance gradient is the
// one that needs the prior variance's own term: d(||ydot'||^2)/d(x') is zero for no stationary kernel.
bool test_posterior_grad_vs_fd(const char* name) {
    const int n_dim = 3, n_train = 15;
    const arma::mat X_ndim_ntrain_mat = make_points(n_dim, n_train);
    std::mt19937_64 rng(kDataSeed + 1);
    const arma::vec f_train_vec = draw_uniform_vec(rng, n_train, 1.5, 2.5);

    gpflash::TrainingData train_data;
    train_data.X_ndim_ntrain_mat = X_ndim_ntrain_mat;
    train_data.f_train_vec = f_train_vec;
    train_data.sig2_f_val = 0.02;
    gpflash::HpVal hp_val_struct;
    hp_val_struct.hp_gamma_vec = make_hp_gamma_vec(n_dim);
    hp_val_struct.hp_sig2_k = 1.7;
    gpflash::ModelOptions model_opts;
    model_opts.kernel_id = gpflash::kernel::KernelId::Linear;
    model_opts.cond_max = 1.0e6;
    const auto posterior = gpflash::make_posterior(train_data, hp_val_struct, model_opts);

    const arma::vec x_post_vec = 0.45 + 0.2 * make_points(n_dim, 1).col(0);
    gpflash::PosteriorOptions grad_opts;
    grad_opts.return_mean_gradient = true;
    grad_opts.return_var_gradient = true;
    const auto grad_result = posterior.eval(x_post_vec, grad_opts);

    gpflash::PosteriorOptions val_opts;
    val_opts.return_mean = true;
    val_opts.return_var = true;

    double worst_mean = 0.0, worst_var = 0.0;
    for (int i_dim = 0; i_dim < n_dim; ++i_dim) {
        arma::vec x_post_plus_vec = x_post_vec, x_post_minus_vec = x_post_vec;
        x_post_plus_vec(i_dim) += kFdStepLinear;
        x_post_minus_vec(i_dim) -= kFdStepLinear;
        const auto post_result_plus = posterior.eval(x_post_plus_vec, val_opts);
        const auto post_result_minus = posterior.eval(x_post_minus_vec, val_opts);
        const double fd_mean =
                (post_result_plus.mean_post_vec(0) - post_result_minus.mean_post_vec(0)) / (2.0 * kFdStepLinear);
        const double fd_var =
                (post_result_plus.var_post_vec(0) - post_result_minus.var_post_vec(0)) / (2.0 * kFdStepLinear);
        worst_mean = std::max(
                worst_mean,
                std::fabs(grad_result.d_mean_post_d_x_post_ndim_npost_mat(i_dim, 0) - fd_mean)
                        / std::max(std::fabs(fd_mean), kFdScaleFloor));
        worst_var = std::max(
                worst_var,
                std::fabs(grad_result.d_var_post_d_x_post_ndim_npost_mat(i_dim, 0) - fd_var)
                        / std::max(std::fabs(fd_var), kFdScaleFloor));
    }

    // The batch gradients must agree with the single-query ones.
    const arma::mat X_ndim_npost_mat(x_post_vec);
    const auto batch_grad = posterior.eval(X_ndim_npost_mat, grad_opts);
    double worst_batch = 0.0;
    for (int i_dim = 0; i_dim < n_dim; ++i_dim) {
        worst_batch = std::max(
                worst_batch,
                std::fabs(
                        batch_grad.d_mean_post_d_x_post_ndim_npost_mat(i_dim, 0)
                        - grad_result.d_mean_post_d_x_post_ndim_npost_mat(i_dim, 0)));
        worst_batch = std::max(
                worst_batch,
                std::fabs(
                        batch_grad.d_var_post_d_x_post_ndim_npost_mat(i_dim, 0)
                        - grad_result.d_var_post_d_x_post_ndim_npost_mat(i_dim, 0)));
    }

    char detail[kDetailLen];
    std::snprintf(
            detail, sizeof(detail), "mean %.2e  var %.2e  batch vs single %.2e", worst_mean, worst_var, worst_batch);
    return report(name, worst_mean < kTolFd && worst_var < kTolFd && worst_batch < kTolAltPath, detail);
}

// A GP with a linear kernel and no noise reproduces a linear function exactly, which no amount of
// hyperparameter tuning lets a stationary kernel do. The nugget is the only source of error.
bool test_recovers_linear_function(const char* name) {
    const int n_dim = 3, n_train = 12;
    const arma::mat X_ndim_ntrain_mat = make_points(n_dim, n_train);
    const arma::vec coeff_vec = arma::vec("0.8 -1.3 0.45");
    const double offset = 2.25;
    const arma::vec f_train_vec = X_ndim_ntrain_mat.t() * coeff_vec + offset;

    gpflash::TrainingData train_data;
    train_data.X_ndim_ntrain_mat = X_ndim_ntrain_mat;
    train_data.f_train_vec = f_train_vec;
    train_data.sig2_f_val = 0.0;
    gpflash::HpVal hp_val_struct;
    hp_val_struct.hp_gamma_vec = arma::vec(n_dim, arma::fill::ones);
    hp_val_struct.hp_sig2_k = 1.0;  // the posterior always needs it supplied, noise or not
    gpflash::ModelOptions model_opts;
    model_opts.kernel_id = gpflash::kernel::KernelId::Linear;
    model_opts.cond_max = 1.0e12;  // a light nugget: the recovery is exact only as eta goes to zero
    const auto posterior = gpflash::make_posterior(train_data, hp_val_struct, model_opts);

    const arma::mat X_ndim_npost_mat = 1.5 * make_points(n_dim, 6);  // outside the training hull, where a
                                                                     // stationary kernel reverts to its prior mean
    gpflash::PosteriorOptions post_opts;
    post_opts.return_mean = true;
    const auto post_result = posterior.eval(X_ndim_npost_mat, post_opts);
    const arma::vec mean_ref_vec = X_ndim_npost_mat.t() * coeff_vec + offset;
    const double worst = arma::abs(post_result.mean_post_vec - mean_ref_vec).max() / arma::abs(mean_ref_vec).max();

    char detail[kDetailLen];
    std::snprintf(detail, sizeof(detail), "worst rel extrapolation error %.2e", worst);
    return report(name, worst < kTolLinearRecovery, detail);
}

// A training point at the origin keeps the offset's prior variance, k(0, 0) = kXdotInnerProdOffset, so the
// preconditioned path accepts it: the MLL and the posterior are finite, and the posterior variance at the origin is
// positive.
bool test_origin_training_point_accepted(const char* name) {
    const int n_dim = 3, n_train = 8;
    gpflash::TrainingData train_data;
    train_data.X_ndim_ntrain_mat = make_points(n_dim, n_train);
    train_data.X_ndim_ntrain_mat.col(0).zeros();
    std::mt19937_64 rng(kDataSeed + 2);
    train_data.f_train_vec = draw_uniform_vec(rng, n_train);
    train_data.sig2_f_val = 0.02;
    gpflash::HpVal hp_val_struct;
    hp_val_struct.hp_gamma_vec = make_hp_gamma_vec(n_dim);
    hp_val_struct.hp_sig2_k = 1.7;
    gpflash::ModelOptions model_opts;
    model_opts.kernel_id = gpflash::kernel::KernelId::Linear;

    bool raised = false;
    double mll = arma::datum::nan, mean_post = arma::datum::nan, var_post = arma::datum::nan;
    try {
        mll = gpflash::eval_mll(train_data, hp_val_struct, model_opts).mll;
        const auto posterior = gpflash::make_posterior(train_data, hp_val_struct, model_opts);
        const auto post_result = posterior.eval(arma::vec(n_dim, arma::fill::zeros));
        mean_post = post_result.mean_post_vec(0);
        var_post = post_result.var_post_vec(0);
    } catch (...) {
        raised = true;
    }
    const bool finite_ok = std::isfinite(mll) && std::isfinite(mean_post) && var_post > 0.0;

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "raised=%d mll %.4g mean %.4g var %.3e",
            static_cast<int>(raised),
            mll,
            mean_post,
            var_post);
    return report(name, !raised && finite_ok, detail);
}

// The name round-trips and the id resolves to the right kernel instance, which is all that wiring a kernel
// into the dispatch means.
bool test_dispatch_round_trip(const char* name) {
    const bool name_ok = gpflash::kernel::parse_kernel_id("linear") == gpflash::kernel::KernelId::Linear
            && std::string(gpflash::kernel::to_kernel_name(gpflash::kernel::KernelId::Linear)) == "linear";
    const gpflash::kernel::KernelBase& kern_fun = gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Linear);
    // The value is the Gram entry plus the offset, and c^(rdot) is 1.
    const double xdot_inner_prod = 2.5;
    double kern_val = xdot_inner_prod, kern_common_rdot = 0.0;
    kern_fun.calc_kern_val_and_common_rdot_inplace(&kern_val, 1, nullptr, &kern_common_rdot);
    // A varying diagonal marks the inner-product family; the value and c^(rdot) pin the linear closed form.
    const bool behavior_ok = !kern_fun.is_kern_diag_unity() && kern_val == xdot_inner_prod + kXdotInnerProdOffset
            && kern_common_rdot == 1.0;
    char detail[kDetailLen];
    std::snprintf(
            detail, sizeof(detail), "name=%d behavior=%d", static_cast<int>(name_ok), static_cast<int>(behavior_ok));

    return report(name, name_ok && behavior_ok, detail);
}

}  // namespace

int main() {
    std::printf("== GpFlash linear kernel ==\n");
    bool ok = true;
    ok &= test_kern_mat_vs_definition("kern_mat_vs_definition");
    ok &= test_singular_kmat_is_conditioned("singular_kmat_is_conditioned");
    ok &= test_posterior_vs_dense_reference("posterior_vs_dense_reference/noise_free", 0.0);
    ok &= test_posterior_vs_dense_reference("posterior_vs_dense_reference/noisy", 0.02);
    ok &= test_posterior_grad_vs_fd("posterior_grad_vs_fd");
    ok &= test_recovers_linear_function("recovers_linear_function");
    ok &= test_origin_training_point_accepted("origin_training_point_accepted");
    ok &= test_dispatch_round_trip("dispatch_round_trip");
    std::printf("\n%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");

    return ok ? 0 : 1;
}
