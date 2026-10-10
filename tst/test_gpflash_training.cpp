// The marginal log likelihood and its gradient, end to end. What this file asserts:
//
//   1) mll_and_grad_vs_reference   eval_mll and its gradient match a brute-force reference that forms
//                                  Sigma^-1 explicitly, for the four radial kernels across a range of n_train
//   2) mll_only_matches_full       the value-only fast path returns the same MLL as the gradient path
//   3) kernel_id_resolves_same     the public entry point resolves model_opts.kernel_id to the same kernel
//   4) noise_free_reference       the profiled noise-free MLL, solved hp_sig2_k and gamma gradient match a
//                                  dense implementation, for the zero, constant and {2, 1, 0} mean models
//   5) mean_model_mll_and_grad    the zero, constant and {2, 1, 0} mean models: MLL and gamma gradient against the
//                                  dense reference, and every gradient against a central difference at the default
//                                  cond_max, with the kernel matrix nugget from either bound. {2, 1, 0} again at three
//                                  lower cond_max under either mean model bound, where the gap to the central
//                                  difference, the neglected term, must scale as 1 / (cond_max - 1)
//   6) ill_conditioned_mean_model_basis  x_1 and x_2 on two nearly equal dimensions: MLL and gamma gradient against
//                                  the dense reference across three hp_gamma scales
//   7) extras_and_variance_grad    eval_mll's hp_rho and two variance gradients, by central
//                                  difference of the MLL, the terms the reference above does not form
//   8) kern_term_hp_grad_vs_dense  calc_kern_term_hp_grad alone against a dense Psi/C/h reference
//   9) hp_val_round_trip          build_hp_val carries both kernel blocks through unchanged
//  10) public_eval_mll             the struct entry point reports only what it solved for
//  11) indefinite_sigma_throws     a failed Cholesky raises a catchable exception
//  12) degenerate_noise_free_stays_finite  noise-free data that collapses the closed-form hp_sig2_k keeps the MLL and
//                                  its gradient finite
//  13) positive_min_bounds_positivity_checks  a value clamped to POSITIVE_MIN passes every positivity check
//
// Check 8 is the one the end-to-end checks cannot make: it pins the packed-triangle layout (c^(rdot) read
// from the strict upper triangle at column n_train - 1 - j) and the factor of two for the strict lower triangle.
#include <algorithm>
#include <armadillo>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>

#include "GpFlashTestSupport.hpp"
#include "gpflash/assembly/KernelMatrix.hpp"
#include "gpflash/assembly/KernelMatrixGrad.hpp"
#include "gpflash/assembly/SigmaMatrix.hpp"
#include "gpflash/core/Errors.hpp"
#include "gpflash/inputs/SymmetrySpec.hpp"      // SymmetrySpec, the no-symmetry argument
#include "gpflash/kernels/KernelDispatch.hpp"   // kernel_by_id
#include "gpflash/kernels/radial/Matern52.hpp"  // Matern52::calc_kern_val_unfactored, the dense reference
#include "gpflash/linear_algebra/Cholesky.hpp"
#include "gpflash/linear_algebra/RecursiveInv.hpp"
#include "gpflash/preconditioning/ConditionedCholesky.hpp"  // build_eta_terms, calc_L_M_precon_to_L_M_inplace
#include "gpflash/training/EvalMll.hpp"                     // eval_mll
#include "gpflash/training/EvalMllDispatch.hpp"             // eval_mll, the KernelId form of eval_mll

using gpflash_test::draw_normal_vec;
using gpflash_test::draw_uniform_mat;
using gpflash_test::draw_uniform_unit;
using gpflash_test::draw_uniform_vec;
using gpflash_test::kCondMaxStrongNugget;
using gpflash_test::kRelScaleFloor;
using gpflash_test::kTolAltPath;
using gpflash_test::kTolExact;
using gpflash_test::kTolReferenceImpl;
using gpflash_test::report;

// Non-unit and unequal: hp_sig2_k = 1 hides a dropped variance factor, and equal values hide a swap.
constexpr double kHpSig2KNonUnit = 2.7;  // absolute sigma_k^2
constexpr double kSig2FNonUnit = 0.013;  // absolute sigma_f^2
// HpVal's "no estimate supplied" sentinel.
constexpr double kHpSig2FNoEstimate = -1.0;
// Keeps a uniform-drawn length scale positive and off a flat kernel.
constexpr double kHpGammaFloor = 0.7;
// The factor either side of one within which a ratio is "of order one".
constexpr double kOrderOfMagnitude = 10.0;

struct MeanModelCase {
    const char* label;
    bool has_mean_model_constant;
    arma::uvec mean_model_degree_vec;
};

// The zero, constant and {2, 1, 0} mean models, for n_dim = 3. Only {2, 1, 0} has more than one weight, and so a
// mean model nugget.
static const MeanModelCase kZeroMeanModelCase{"zero", false, arma::uvec{0}};
static const MeanModelCase kConstantMeanModelCase{"constant", true, arma::uvec{0}};
static const MeanModelCase kMonomialMeanModelCase{"{2,1,0}", true, arma::uvec{2, 1, 0}};
static const MeanModelCase kMeanModelCases[] = {kZeroMeanModelCase, kConstantMeanModelCase, kMonomialMeanModelCase};

// Brute-force reference (independent of the library's tiled loops). Its gradient, like the library's, holds the mean
// model weights fixed.
static void reference_mll_grad(
        const gpflash::kernel::KernelBase& kern_fun,
        const arma::mat& X_ndim_ntrain_mat,
        const arma::vec& f_train_vec,
        const arma::mat& Basis_ntrain_nbasis_mat,
        const arma::vec& hp_gamma_vec,
        const double hp_rho,
        const gpflash::detail::EvalVariances& eval_variances,
        const double cond_max,
        double& mll_ref,
        arma::vec& grad_ref_vec_out,
        const gpflash::EigMaxUbMethod mean_model_eig_max_ub_method = gpflash::EigMaxUbMethod::Trace,
        const gpflash::EigMaxUbMethod kmat_eig_max_ub_method = gpflash::EigMaxUbMethod::Gershgorin) {
    const int n_dim = X_ndim_ntrain_mat.n_rows;
    const int n_train = X_ndim_ntrain_mat.n_cols;

    arma::mat K_mat(n_train, n_train, arma::fill::none);   // full symmetric kernel matrix
    arma::mat C_mat(n_train, n_train, arma::fill::zeros);  // c^(rdot)
    for (int i_row = 0; i_row < n_train; ++i_row) {
        K_mat(i_row, i_row) = 1.0;
        for (int i_col = i_row + 1; i_col < n_train; ++i_col) {
            double rdot_sq_norm = 0.0;
            for (int i_dim = 0; i_dim < n_dim; ++i_dim) {
                const double r_d =
                        (X_ndim_ntrain_mat(i_dim, i_row) - X_ndim_ntrain_mat(i_dim, i_col)) * hp_gamma_vec(i_dim);
                rdot_sq_norm += r_d * r_d;
            }
            double kern_val = rdot_sq_norm;
            double kern_common_rdot = 0.0;
            kern_fun.calc_kern_val_and_common_rdot_inplace(&kern_val, 1, &hp_rho, &kern_common_rdot);
            K_mat(i_row, i_col) = K_mat(i_col, i_row) = kern_val;
            C_mat(i_row, i_col) = C_mat(i_col, i_row) = kern_common_rdot;
        }
    }
    // eta = lambda_max_bound/(cond_max - 1). The Gershgorin bound is the
    // largest full absolute row sum, max_i sum_j |K_ij| -- stated here straight from the
    // dense matrix, diagonal included, rather than as k(0) + off-diagonal max, so that it
    // is an independent check of the library's factored form. The trace bound is tr(K).
    const bool is_kmat_trace = kmat_eig_max_ub_method == gpflash::EigMaxUbMethod::Trace;
    const double kmat_eig_max_ub = is_kmat_trace ? arma::trace(K_mat) : arma::sum(arma::abs(K_mat), 1).max();
    const double eta = kmat_eig_max_ub / (cond_max - 1.0);
    // Off-diagonal row sums separately: only these depend on the hyperparameters, so it is
    // their argmax that the eta derivative below sums over.
    const arma::vec row_sum_vec = arma::sum(arma::abs(K_mat), 1) - arma::abs(K_mat.diag());

    // The GpFlash paper's covariance, built with the absolute variances so the reference is
    // valid for any hp_sig2_k rather than only for hp_sig2_k = 1:
    //   Sigma = hp_sig2_k (K + eta I) + sig2_f_val I.
    arma::mat Sigma_mat = eval_variances.hp_sig2_k * K_mat;
    Sigma_mat.diag() += eval_variances.hp_sig2_k * eta + eval_variances.sig2_f_val;

    arma::mat L_Sigma_mat = arma::chol(Sigma_mat, "lower");
    // The reference mean model weights, then the residual solve. None for the zero mean model.
    auto sigma_solve = [&](const arma::mat& rhs_mat) {
        return arma::mat(arma::solve(arma::trimatu(L_Sigma_mat.t()), arma::solve(arma::trimatl(L_Sigma_mat), rhs_mat)));
    };
    const arma::vec lin_sol_sigma_f_train_vec = sigma_solve(f_train_vec);
    const arma::mat lin_sol_sigma_Basis_ntrain_nbasis_mat = sigma_solve(Basis_ntrain_nbasis_mat);
    const arma::vec hp_beta_ref_vec = gpflash_test::calc_hp_beta_ref_vec(
            Basis_ntrain_nbasis_mat,
            lin_sol_sigma_Basis_ntrain_nbasis_mat,
            lin_sol_sigma_f_train_vec,
            cond_max,
            mean_model_eig_max_ub_method);
    const arma::vec residual_vec = f_train_vec - Basis_ntrain_nbasis_mat * hp_beta_ref_vec;
    // Sigma^-1 (f_train - Basis hp_beta_vec)
    arma::vec lin_sol_sigma_residual_vec =
            lin_sol_sigma_f_train_vec - lin_sol_sigma_Basis_ntrain_nbasis_mat * hp_beta_ref_vec;
    double log_det = 0.0;
    for (int i_diag = 0; i_diag < n_train; ++i_diag) log_det += 2.0 * std::log(L_Sigma_mat(i_diag, i_diag));
    // The full Gaussian log density, normalizer included, matching eval_mll.
    // Written as log(2 pi) here rather than through arma::datum::log_sqrt2pi so the reference does not reuse the
    // library's constant.
    mll_ref = -0.5
            * (arma::dot(residual_vec, lin_sol_sigma_residual_vec) + log_det
               + n_train * std::log(2.0 * arma::datum::pi));

    const arma::mat Sigma_inv_mat = arma::inv_sympd(Sigma_mat);
    arma::mat Psi_mat = 0.5 * (lin_sol_sigma_residual_vec * lin_sol_sigma_residual_vec.t() - Sigma_inv_mat);

    // Both terms of dSigma/d(hp_gamma_d) = hp_sig2_k ( dK/d(hp_gamma_d) + d(eta)/d(hp_gamma_d) I ). The
    // eta derivative is found the naive way, argmax row of the full symmetric matrix then sum it, so it is
    // independent of the library's packed-triangle traversal rather than a restatement of it.
    const arma::uword i_star = row_sum_vec.index_max();
    const double trace_psi = arma::trace(Psi_mat);
    grad_ref_vec_out.set_size(n_dim);
    for (int i_dim = 0; i_dim < n_dim; ++i_dim) {
        double kern_sum = 0.0;
        for (int i_row = 0; i_row < n_train; ++i_row)
            for (int i_col = 0; i_col < n_train; ++i_col) {
                const double r_d = X_ndim_ntrain_mat(i_dim, i_row) - X_ndim_ntrain_mat(i_dim, i_col);
                kern_sum += Psi_mat(i_row, i_col) * C_mat(i_row, i_col) * r_d * r_d;
            }
        // d(eta)/d(hp_gamma_d) = 1/(cond_max - 1) * sum_{j != i*} dK_{i* j}/d(hp_gamma_d)
        double d_eta_sum = 0.0;
        for (int i_col = 0; i_col < n_train; ++i_col) {
            if (i_col == static_cast<int>(i_star)) continue;
            const double r_d = X_ndim_ntrain_mat(i_dim, i_star) - X_ndim_ntrain_mat(i_dim, i_col);
            d_eta_sum += C_mat(i_star, i_col) * r_d * r_d;
        }
        // h^(gamma,l) = hp_gamma_l r_l^2, its r_l^2 taken in the sums above and its hp_gamma_l here.
        const double d_gamma_pref = hp_gamma_vec(i_dim);
        // The trace bound tr(K) = n_train does not depend on hp_gamma.
        const double d_eta_d_gamma = is_kmat_trace ? 0.0 : d_gamma_pref * d_eta_sum / (cond_max - 1.0);
        grad_ref_vec_out(i_dim) = eval_variances.hp_sig2_k * (d_gamma_pref * kern_sum + d_eta_d_gamma * trace_psi);
    }
}

static bool run_kernel(
        const gpflash::kernel::KernelBase& kern_fun, const std::string& name, double rel_tol, double hp_rho = 0.0) {
    std::mt19937_64 rng(12345);
    gpflash::detail::EvalVariances eval_variances;
    eval_variances.hp_sig2_k = kHpSig2KNonUnit;
    eval_variances.sig2_f_val = kSig2FNonUnit;
    const double cond_max = gpflash::COND_MAX_DEFAULT;

    bool ok = true;
    std::printf("== Kernel: %s ==\n", name.c_str());
    // Spanning gpflash_test::kNtrainAboveOmpThresholds, so both the serial and the OpenMP column sweeps run.
    for (int n_train : {30, 60, gpflash_test::kNtrainAboveOmpThresholds, 600}) {
        const int n_dim = 10;
        arma::mat X_ndim_ntrain_mat(n_dim, n_train);
        for (auto& v : X_ndim_ntrain_mat) v = draw_uniform_unit(rng);
        arma::vec hp_gamma_vec(n_dim);
        for (auto& v : hp_gamma_vec) v = 0.5 + draw_uniform_unit(rng);  // positive length-scale inv
        arma::vec f_train_vec(n_train);
        for (auto& v : f_train_vec) v = draw_uniform_unit(rng) - 0.5;

        arma::vec hp_kern_extra_vec(kern_fun.n_hp_kern_extra(n_dim));
        if (!hp_kern_extra_vec.is_empty()) hp_kern_extra_vec(0) = hp_rho;

        const gpflash::TrainingData train_data =
                gpflash::TrainingData{X_ndim_ntrain_mat, f_train_vec, eval_variances.sig2_f_val};
        const gpflash::HpVal hp_val_struct =
                gpflash::detail::build_hp_val(hp_gamma_vec, hp_kern_extra_vec, eval_variances.hp_sig2_k);
        gpflash::ModelOptions model_opts;
        model_opts.cond_max = cond_max;

        double mll_ref;
        arma::vec grad_ref_vec;
        const arma::mat Basis_ntrain_nbasis_mat(n_train, 1, arma::fill::ones);  // the default, constant mean model
        reference_mll_grad(
                kern_fun,
                X_ndim_ntrain_mat,
                f_train_vec,
                Basis_ntrain_nbasis_mat,
                hp_gamma_vec,
                hp_rho,
                eval_variances,
                cond_max,
                mll_ref,
                grad_ref_vec);

        const double mll_only =
                gpflash::detail::eval_mll(kern_fun, train_data, hp_val_struct, model_opts, /*eval_grad=*/false).mll;
        const gpflash::MllResult full_result =
                gpflash::detail::eval_mll(kern_fun, train_data, hp_val_struct, model_opts, /*eval_grad=*/true);
        const double mll_full = full_result.mll;
        const arma::vec& d_mll_d_hp_gamma_lib_vec = full_result.mll_hp_grad_struct.d_mll_d_hp_gamma_vec;

        // The public form resolves model_opts.kernel_id to the instance above, so it adds nothing of its own. Above
        // PERF_OMP_MIN_N_POINT the OpenMP accumulation is order-dependent, so two identical calls agree only to
        // kTolAltPath.
        gpflash::ModelOptions model_opts_by_id = model_opts;
        model_opts_by_id.kernel_id = kern_fun.kernel_id();
        const double mll_by_id_only =
                gpflash::eval_mll(train_data, hp_val_struct, model_opts_by_id, /*eval_grad=*/false).mll;
        const gpflash::MllResult by_id_full_result =
                gpflash::eval_mll(train_data, hp_val_struct, model_opts_by_id, /*eval_grad=*/true);
        const double mll_by_id_full = by_id_full_result.mll;
        const arma::vec& d_mll_d_hp_gamma_by_id_vec = by_id_full_result.mll_hp_grad_struct.d_mll_d_hp_gamma_vec;
        const double by_id_mll_diff = std::max(std::abs(mll_by_id_only - mll_only), std::abs(mll_by_id_full - mll_full))
                / std::max(1.0, std::abs(mll_full));
        const double by_id_grad_diff = arma::norm(d_mll_d_hp_gamma_by_id_vec - d_mll_d_hp_gamma_lib_vec, "inf")
                / std::max(kRelScaleFloor, arma::norm(d_mll_d_hp_gamma_lib_vec, "inf"));
        const double by_id_diff = std::max(by_id_mll_diff, by_id_grad_diff);
        const double by_id_tol = kTolAltPath;

        const double mll_rel = std::abs(mll_full - mll_ref) / std::max(1.0, std::abs(mll_ref));
        const double mll_only_vs_full_rel = std::abs(mll_full - mll_only) / std::max(1.0, std::abs(mll_full));
        const double grad_ref_norm = std::max(kRelScaleFloor, arma::norm(grad_ref_vec));
        const double grad_rel = arma::norm(d_mll_d_hp_gamma_lib_vec - grad_ref_vec) / grad_ref_norm;

        const bool pass = (mll_rel < rel_tol) && (grad_rel < rel_tol) && (mll_only_vs_full_rel < kTolExact)
                && (by_id_diff <= by_id_tol);
        ok = ok && pass;
        std::printf(
                "  n_train=%4d  mll_rel=%.1e  grad_rel=%.1e  mll_only_vs_full=%.1e  by_id_diff=%.1e (tol %.0e)"
                "  %s\n",
                n_train,
                mll_rel,
                grad_rel,
                mll_only_vs_full_rel,
                by_id_diff,
                by_id_tol,
                pass ? "PASS" : "*** FAIL ***");
    }
    return ok;
}

// Must throw ErrorNumericalFailure specifically: the multistart treats that type alone as a bad start and carries
// on, and rethrows anything else as a defect. eval_mll's own eta guarantees conditioning, so reaching a real dpotrf
// failure means driving the stages directly with an eta (0.0) that understates the off-diagonal size.
static bool test_indefinite_sigma_throws() {
    arma::mat Sigma_mat = {
            {1.0, 10.0, 10.0},
            {10.0, 1.0, 10.0},
            {10.0, 10.0, 1.0}};  // indefinite: off-diagonals dominate the diagonal
    // Explicit noise/signal variances so the matrix handed to the factorization is the
    // indefinite one above plus a known diagonal, independent of the library defaults.
    gpflash::detail::EvalVariances eval_variances;
    eval_variances.sig2_f_val = kSig2FNonUnit;
    eval_variances.hp_sig2_k = kHpSig2KNonUnit;  // non-unit: scaling the matrix leaves it indefinite

    bool threw = false;
    try {
        gpflash::detail::build_sigma_corr_inplace(
                Sigma_mat,
                /*eta=*/0.0,
                eval_variances.sig2_f_val,
                eval_variances.hp_sig2_k,
                /*inv_precon_scale_vec=*/arma::vec());
        gpflash::detail::factorize_chol_inplace(Sigma_mat, "test_gpflash_training");
    } catch (const gpflash::ErrorNumericalFailure&) {
        threw = true;
    }
    report("indefinite_sigma_throws", threw);

    return threw;
}

// Degenerate noise-free data collapses the closed-form hp_sig2_k. That is a real perfect fit, so it is
// reported rather than rejected, but POSITIVE_MIN must keep it finite -- unfloored, log(0) makes the MLL
// +inf and the gradient divides by zero. Three triggers: constant f_train (eps-sized residual), all-zero
// (exact), n_train == 1. The gradient is requested too, since its 1 / hp_sig2_k is what sets the floor.
static bool test_degenerate_noise_free_stays_finite() {
    std::mt19937_64 rng(24601);
    // eval_mll's postconditions throw on a non-finite result, so a throw is this trigger's failure, not the run's.
    const auto stays_finite = [&rng](const int n_train, const double f_const) {
        try {
            gpflash::TrainingData train_data;
            train_data.X_ndim_ntrain_mat = draw_uniform_mat(rng, 2, n_train);
            train_data.f_train_vec = f_const * arma::ones<arma::vec>(n_train);
            gpflash::HpVal hp_val_struct;
            hp_val_struct.hp_gamma_vec = arma::ones<arma::vec>(2);
            const gpflash::MllResult mll_result =
                    gpflash::eval_mll(train_data, hp_val_struct, gpflash::ModelOptions{}, /*return_mll_gradient=*/true);
            const gpflash::MllHpGrad& mll_hp_grad_struct = mll_result.mll_hp_grad_struct;
            return std::isfinite(mll_result.mll) && mll_result.hp_sig2_k > 0.0 && mll_result.hp_beta_vec.is_finite()
                    && std::isfinite(mll_hp_grad_struct.d_mll_d_hp_sig2_k)
                    && mll_hp_grad_struct.d_mll_d_hp_gamma_vec.is_finite();
        } catch (const std::exception&) {
            return false;
        }
    };

    const bool constant_finite = stays_finite(/*n_train=*/12, /*f_const=*/3.7);
    const bool all_zero_finite = stays_finite(/*n_train=*/12, /*f_const=*/0.0);
    const bool single_point_finite = stays_finite(/*n_train=*/1, /*f_const=*/3.7);

    // The three above leave a residual of zero or of rounding size. A nonzero one far below rounding size is where
    // the gradient's 1 / hp_sig2_k would overflow: the floor, not cancellation, keeps this one finite.
    bool tiny_nonzero_finite = false;
    try {
        gpflash::TrainingData train_data;
        train_data.X_ndim_ntrain_mat = draw_uniform_mat(rng, 2, 12);
        train_data.f_train_vec = 1.0e-160 * draw_uniform_vec(rng, 12);
        gpflash::HpVal hp_val_struct;
        hp_val_struct.hp_gamma_vec = arma::ones<arma::vec>(2);
        const gpflash::MllResult mll_result =
                gpflash::eval_mll(train_data, hp_val_struct, gpflash::ModelOptions{}, /*return_mll_gradient=*/true);
        // Clamped, so the reported variance is POSITIVE_MIN itself rather than the ~1e-320 the data implies.
        tiny_nonzero_finite = std::isfinite(mll_result.mll) && mll_result.hp_sig2_k == gpflash::detail::POSITIVE_MIN
                && mll_result.mll_hp_grad_struct.d_mll_d_hp_gamma_vec.is_finite();
    } catch (const std::exception&) {
        tiny_nonzero_finite = false;
    }

    const bool pass = constant_finite && all_zero_finite && single_point_finite && tiny_nonzero_finite;
    char detail[gpflash_test::kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "constant %d  all_zero %d  single_point %d  tiny_nonzero %d",
            constant_finite ? 1 : 0,
            all_zero_finite ? 1 : 0,
            single_point_finite ? 1 : 0,
            tiny_nonzero_finite ? 1 : 0);
    report("degenerate_noise_free_stays_finite", pass, detail);

    return pass;
}

// POSITIVE_MIN is one constant with two jobs: it floors the closed-form hp_sig2_k above, and it is the bound
// every positivity check tests against. What ties them is inclusive acceptance -- a clamped value is exactly
// POSITIVE_MIN, so a validator rejecting it would refuse a fit's own output. That case is asserted first.
static bool test_positive_min_bounds_positivity_checks() {
    using gpflash::detail::check_hp_val;
    using gpflash::detail::check_supplied_hp_sig2_k;
    using gpflash::detail::POSITIVE_MIN;
    using gpflash_test::accepts;
    using gpflash_test::rejects;

    // The checkers take structs, so each case is a one-line HpVal built by build_hp_val.
    const auto hp_sig2_k_of = [](const double hp_sig2_k) {
        return gpflash::detail::build_hp_val(arma::ones<arma::vec>(2), arma::vec(), hp_sig2_k);
    };
    const auto hp_blocks_of = [](const arma::vec& hp_gamma_vec, const arma::vec& hp_kern_extra_vec) {
        return gpflash::detail::build_hp_val(hp_gamma_vec, hp_kern_extra_vec, 1.0);
    };

    // One ulp below the bound, so a validator whose bound drifted lower would accept it.
    const double below_positive_min = std::nextafter(POSITIVE_MIN, 0.0);

    // The invariant: the exact value the clamp produces must survive validation.
    const bool floor_accepted = accepts([&] { check_supplied_hp_sig2_k(hp_sig2_k_of(POSITIVE_MIN), "test"); });
    const bool below_floor_rejected =
            rejects([&] { check_supplied_hp_sig2_k(hp_sig2_k_of(below_positive_min), "test"); });
    // Zero, the unset value and NaN are rejected.
    const bool zero_and_unset_rejected = rejects([&] { check_supplied_hp_sig2_k(hp_sig2_k_of(0.0), "test"); })
            && rejects([&] { check_supplied_hp_sig2_k(hp_sig2_k_of(-1.0), "test"); })
            && rejects([&] { check_supplied_hp_sig2_k(hp_sig2_k_of(arma::datum::nan), "test"); });

    // The same bound governs both kernel hyperparameter blocks. Vectors named rather than inlined, so the
    // lambdas stay on one line each.
    const arma::vec hp_gamma_ok_vec = arma::ones<arma::vec>(2);
    const arma::vec hp_gamma_tiny_vec{1.0, below_positive_min};
    const arma::vec hp_extra_tiny_vec{below_positive_min};
    const arma::vec hp_gamma_floor_vec{1.0, POSITIVE_MIN};
    const arma::vec hp_extra_floor_vec{POSITIVE_MIN};
    const bool hp_gamma_rejected =
            rejects([&] { check_hp_val(hp_blocks_of(hp_gamma_tiny_vec, arma::vec()), 2, 0, "test"); });
    const bool hp_extra_rejected =
            rejects([&] { check_hp_val(hp_blocks_of(hp_gamma_ok_vec, hp_extra_tiny_vec), 2, 1, "test"); });
    const bool hp_blocks_rejected = hp_gamma_rejected && hp_extra_rejected;
    const bool hp_blocks_at_floor_accepted =
            accepts([&] { check_hp_val(hp_blocks_of(hp_gamma_floor_vec, hp_extra_floor_vec), 2, 1, "test"); });

    const bool pass = floor_accepted && below_floor_rejected && zero_and_unset_rejected && hp_blocks_rejected
            && hp_blocks_at_floor_accepted;
    char detail[gpflash_test::kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "floor_ok %d  below_rejected %d  zero_unset_nan %d  hp_blocks %d  hp_at_floor %d",
            floor_accepted ? 1 : 0,
            below_floor_rejected ? 1 : 0,
            zero_and_unset_rejected ? 1 : 0,
            hp_blocks_rejected ? 1 : 0,
            hp_blocks_at_floor_accepted ? 1 : 0);
    report("positive_min_bounds_positivity_checks", pass, detail);

    return pass;
}

// eval_mll's struct entry point: hp_sig2_k is reported, and flagged, only on the noise-free path, and each noise
// request and its two invalid forms behave as TrainingData's sign rule says.
static bool test_public_eval_mll_reports_what_it_solved() {
    const int n_dim = 2, n_train = 12;
    std::mt19937_64 rng(8675309);
    gpflash::TrainingData train_data;
    train_data.X_ndim_ntrain_mat = draw_uniform_mat(rng, n_dim, n_train);
    train_data.f_train_vec = draw_uniform_vec(rng, n_train);

    gpflash::HpVal hp_val_struct;
    hp_val_struct.hp_gamma_vec = arma::vec(n_dim, arma::fill::ones);
    const gpflash::ModelOptions model_opts;  // Matern52

    // Noise-free: sigma_k^2 has a closed form and comes back solved.
    train_data.sig2_f_val = 0.0;
    const gpflash::MllResult noise_free_no_grad = gpflash::eval_mll(train_data, hp_val_struct, model_opts);
    const gpflash::MllResult noise_free_with_grad =
            gpflash::eval_mll(train_data, hp_val_struct, model_opts, /*return_mll_gradient=*/true);

    // Known noise: sigma_k^2 is supplied, so it is not reported back.
    train_data.sig2_f_val = 1e-2;
    hp_val_struct.hp_sig2_k = 1.5;
    const gpflash::MllResult known_noise = gpflash::eval_mll(train_data, hp_val_struct, model_opts);

    // The third case of TrainingData's sign rule: negative takes the variance from HpVal::hp_sig2_f_val.
    train_data.sig2_f_val = -1.0;
    hp_val_struct.hp_sig2_f_val = 2.5e-2;
    const gpflash::MllResult estimated_noise = gpflash::eval_mll(train_data, hp_val_struct, model_opts);
    // The same MLL as supplying that variance directly. The estimate is cleared here: both at once is rejected.
    gpflash::TrainingData train_data_fixed = train_data;
    train_data_fixed.sig2_f_val = hp_val_struct.hp_sig2_f_val;
    gpflash::HpVal hp_val_fixed = hp_val_struct;
    hp_val_fixed.hp_sig2_f_val = kHpSig2FNoEstimate;
    const gpflash::MllResult fixed_noise = gpflash::eval_mll(train_data_fixed, hp_val_fixed, model_opts);
    const bool estimate_matches_fixed =
            std::fabs(estimated_noise.mll - fixed_noise.mll) <= kTolExact * std::max(1.0, std::fabs(fixed_noise.mll));

    // A negative request with no estimate to read is an error, not a silent fallback.
    bool estimate_without_value_throws = false;
    gpflash::HpVal hp_val_no_estimate = hp_val_struct;
    hp_val_no_estimate.hp_sig2_f_val = kHpSig2FNoEstimate;
    try {
        gpflash::eval_mll(train_data, hp_val_no_estimate, model_opts);
    } catch (const gpflash::ErrorInvalidInput&) {
        estimate_without_value_throws = true;
    }

    // The two noise sources are mutually exclusive: a supplied variance plus an estimate is ambiguous.
    bool both_noise_sources_throw = false;
    try {
        gpflash::eval_mll(train_data_fixed, hp_val_struct, model_opts);
    } catch (const gpflash::ErrorInvalidInput&) {
        both_noise_sources_throw = true;
    }

    // The default is the constant mean model, its one weight solved for on every path.
    const bool is_noise_free_ok = noise_free_no_grad.hp_sig2_k_from_closed_form && noise_free_no_grad.hp_sig2_k > 0.0
            && noise_free_no_grad.mll_hp_grad_struct.d_mll_d_hp_gamma_vec.is_empty()
            && noise_free_with_grad.mll_hp_grad_struct.d_mll_d_hp_gamma_vec.n_elem == static_cast<arma::uword>(n_dim)
            && noise_free_no_grad.hp_beta_vec.n_elem == 1 && noise_free_no_grad.hp_beta_vec.is_finite();
    const bool is_known_noise_ok = !known_noise.hp_sig2_k_from_closed_form && known_noise.hp_sig2_k == 0.0
            && known_noise.hp_beta_vec.n_elem == 1 && known_noise.hp_beta_vec.is_finite();
    const bool is_estimated_noise_ok = !estimated_noise.hp_sig2_k_from_closed_form && estimate_matches_fixed;

    const bool pass = is_noise_free_ok && is_known_noise_ok && is_estimated_noise_ok && estimate_without_value_throws
            && both_noise_sources_throw;
    char detail[gpflash_test::kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "noise_free %d  known %d  estimated %d  estimate_without_value %d  both_sources %d",
            is_noise_free_ok ? 1 : 0,
            is_known_noise_ok ? 1 : 0,
            is_estimated_noise_ok ? 1 : 0,
            estimate_without_value_throws ? 1 : 0,
            both_noise_sources_throw ? 1 : 0);
    report("public_eval_mll_reports_what_it_solved", pass, detail);

    return pass;
}

// calc_kern_term_hp_grad on its own, which the end-to-end checks above reach only through the whole
// pipeline. Two things are its alone and are not covered elsewhere:
//
//   - it builds A = Psi .* C^(rdot) one column at a time, reading c^(rdot) from Sigma_mat's strict upper
//     triangle at column n_train - 1 - j, and applies the factor of two for the strict lower triangle;
//   - it harvests the active row's c^(rdot), which is scattered one entry per column.
//
// The reference forms Psi, C and h densely and sums over the strict lower triangle directly.
static bool test_kern_term_hp_grad_vs_dense(const gpflash::kernel::KernelBase& kern_fun, const char* name) {
    const int n_dim = 3, n_train = 17;
    std::mt19937_64 rng(4242);
    const arma::mat X_ndim_ntrain_mat = draw_uniform_mat(rng, n_dim, n_train);
    const arma::vec f_train_vec = draw_normal_vec(rng, n_train);
    // Offset so every length scale is positive; see kHpGammaFloor.
    const arma::vec hp_gamma_vec = draw_uniform_vec(rng, n_dim) + kHpGammaFloor;

    gpflash::detail::EvalVariances eval_variances;
    // Non-unit for the reason kHpSig2KNonUnit states; the reference is read from this Sigma.
    eval_variances.hp_sig2_k = kHpSig2KNonUnit;
    eval_variances.sig2_f_val = 0.3;  // supplied noise, so hp_sig2_k is not re-solved
    const double cond_max = gpflash::COND_MAX_DEFAULT;

    // Drive the pipeline's own stages up to the point calc_kern_term_hp_grad is called.
    const arma::mat Ydot_nfeature_ntrain_mat = X_ndim_ntrain_mat.each_col() % hp_gamma_vec;
    gpflash::detail::EtaTerms kmat_eta_terms;
    arma::mat Sigma_mat = gpflash::detail::build_kmat_lower_tri(
            kern_fun, Ydot_nfeature_ntrain_mat, nullptr, /*store_hp_grad_factors=*/true);
    kmat_eta_terms = gpflash::detail::build_eta_terms(
            kern_fun.is_kern_diag_unity(),
            Sigma_mat,
            cond_max,
            gpflash::EigMaxUbMethod::Gershgorin,
            /*want_active_row=*/true);
    const arma::mat K_corr_upper_mat = Sigma_mat;  // keep c^(rdot) before the factorization overwrites it
    const double eta = kmat_eta_terms.eta;
    gpflash::detail::build_sigma_corr_inplace(
            Sigma_mat,
            eta,
            eval_variances.sig2_f_val,
            eval_variances.hp_sig2_k,
            kmat_eta_terms.precon_scales.inv_precon_scale_vec);
    gpflash::detail::factorize_chol_inplace(Sigma_mat, "test_gpflash_training");
    gpflash::detail::calc_L_M_precon_to_L_M_inplace(Sigma_mat, kmat_eta_terms.precon_scales.precon_scale_vec);
    // hp_sig2_k is supplied here, so this drives the noisy case: the factor is L_Sigma and no rescale to
    // absolute coordinates applies.
    const arma::mat Basis_ntrain_nbasis_mat(n_train, 1, arma::fill::ones);  // the constant mean model
    const gpflash::detail::CholTerms chol_terms = gpflash::detail::calc_chol_terms(
            Sigma_mat, f_train_vec, Basis_ntrain_nbasis_mat, cond_max, gpflash::EigMaxUbMethod::Trace);
    arma::vec lin_sol_sigma_residual_vec = chol_terms.lin_sol_L_residual_vec;
    gpflash::detail::solve_chol_transpose_vec_inplace(Sigma_mat, lin_sol_sigma_residual_vec);
    // Reads of L are done, so the inverse may now overwrite it.
    gpflash::detail::recursive_dpotri_lower(Sigma_mat.memptr(), static_cast<int>(Sigma_mat.n_rows));

    const gpflash::detail::KernTermHpGrad kern_term_hp_grad = gpflash::detail::calc_kern_term_hp_grad(
            kern_fun,
            gpflash::kernel::make_hp_grad_inputs(X_ndim_ntrain_mat, X_ndim_ntrain_mat, hp_gamma_vec, arma::vec()),
            lin_sol_sigma_residual_vec,
            Sigma_mat,
            kmat_eta_terms.idx_M_precon_max_abs_row_sum,
            gpflash::SymmetrySpec{},
            arma::vec{});
    const arma::vec& d_mll_d_hp_gamma_vec = kern_term_hp_grad.d_mll_d_hp_gamma_vec;
    const arma::vec& active_row_c_vec = kern_term_hp_grad.active_row_kern_common_rdot_vec;

    // c^(rdot) for pair (i, j) with i > j is not at (i, j): assembly packs column j's values into the strict
    // upper triangle of column n_train - 1 - j so the gradient reads them at unit stride. Stated here from that
    // rule so the test pins the packing rather than inheriting it.
    const auto kern_common_rdot_at = [&](const int i, const int j) {
        const int i_below = std::max(i, j) - std::min(i, j) - 1;
        return K_corr_upper_mat(i_below, n_train - 1 - std::min(i, j));
    };

    // Dense reference: Psi = (v v^T - Sigma^-1) / 2, then 2 sum_{i>j} Psi_ij C_ij h_ij.
    const arma::vec& v_vec = lin_sol_sigma_residual_vec;
    arma::vec grad_ref_vec(n_dim, arma::fill::zeros);
    for (int i_row = 0; i_row < n_train; ++i_row)
        for (int i_col = 0; i_col < i_row; ++i_col) {
            const double psi_ij = 0.5 * (v_vec(i_row) * v_vec(i_col) - Sigma_mat(i_row, i_col));
            const double c_ij = kern_common_rdot_at(i_row, i_col);
            for (int i_dim = 0; i_dim < n_dim; ++i_dim) {
                const double r_d = X_ndim_ntrain_mat(i_dim, i_row) - X_ndim_ntrain_mat(i_dim, i_col);
                grad_ref_vec(i_dim) += 2.0 * psi_ij * c_ij * (hp_gamma_vec(i_dim) * r_d * r_d);
            }
        }

    // The harvested active row, from the shared transcription of the packed layout.
    const int idx_active = kmat_eta_terms.idx_M_precon_max_abs_row_sum;
    const arma::vec active_row_ref_vec =
            gpflash_test::harvest_active_row_kern_common_rdot(K_corr_upper_mat, idx_active);

    const double grad_err = arma::abs(d_mll_d_hp_gamma_vec - grad_ref_vec).max()
            / std::max(kRelScaleFloor, arma::abs(grad_ref_vec).max());
    const double row_err = arma::abs(active_row_c_vec - active_row_ref_vec).max()
            / std::max(kRelScaleFloor, arma::abs(active_row_ref_vec).max());
    const bool pass = (grad_err <= kTolReferenceImpl) && (row_err <= kTolExact) && (idx_active >= 0);
    char detail[gpflash_test::kDetailLen];
    std::snprintf(detail, sizeof(detail), "grad_rel=%.1e  active_row_rel=%.1e", grad_err, row_err);
    report(name, pass, detail);

    return pass;
}

// eval_mll with eval_grad, which fills the kernel-extras and variance gradients. Each is checked
// against a central difference of the MLL, which reference_mll_grad above does not cover for these three terms.
static bool test_extras_and_variance_grad_vs_fd(const char* name) {
    const int n_dim = 3, n_train = 14;
    std::mt19937_64 rng(97531);
    const arma::mat X_ndim_ntrain_mat = draw_uniform_mat(rng, n_dim, n_train);
    const arma::vec f_train_vec = draw_normal_vec(rng, n_train);

    gpflash::detail::EvalVariances eval_variances;
    eval_variances.hp_sig2_k = kHpSig2KNonUnit;
    eval_variances.sig2_f_val = kSig2FNonUnit;
    const double cond_max = gpflash::COND_MAX_DEFAULT;

    arma::vec hp_gamma_vec(n_dim);
    for (int i_dim = 0; i_dim < n_dim; ++i_dim) hp_gamma_vec(i_dim) = 0.6 + 0.3 * i_dim;
    const arma::vec hp_kern_extra_vec{gpflash_test::kRatQuadHpRho};

    const gpflash::kernel::KernelBase& kern_fun = gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::RatQuad);
    gpflash::ModelOptions model_opts;
    model_opts.cond_max = cond_max;
    model_opts.kernel_id = kern_fun.kernel_id();
    const gpflash::MllResult mll_result = gpflash::detail::eval_mll(
            kern_fun,
            gpflash::TrainingData{X_ndim_ntrain_mat, f_train_vec, eval_variances.sig2_f_val},
            gpflash::detail::build_hp_val(hp_gamma_vec, hp_kern_extra_vec, eval_variances.hp_sig2_k),
            model_opts,
            /*eval_grad=*/true);

    // The MLL as a function of one scalar, everything else held.
    const auto calc_mll_at = [&](const arma::vec& hp_gamma_at,
                                 const arma::vec& hp_kern_extra_at,
                                 const gpflash::detail::EvalVariances& opts) {
        return gpflash::detail::eval_mll(
                       kern_fun,
                       gpflash::TrainingData{X_ndim_ntrain_mat, f_train_vec, opts.sig2_f_val},
                       gpflash::detail::build_hp_val(hp_gamma_at, hp_kern_extra_at, opts.hp_sig2_k),
                       model_opts,
                       /*eval_grad=*/false)
                .mll;
    };
    const auto central_diff = [](const double f_plus, const double f_minus, const double step) {
        return (f_plus - f_minus) / (2.0 * step);
    };

    // d(MLL)/d(hp_rho).
    arma::vec hp_kern_extra_plus_vec = hp_kern_extra_vec, hp_kern_extra_minus_vec = hp_kern_extra_vec;
    hp_kern_extra_plus_vec(0) += gpflash_test::kFdStepObj;
    hp_kern_extra_minus_vec(0) -= gpflash_test::kFdStepObj;
    const double d_rho_fd = central_diff(
            calc_mll_at(hp_gamma_vec, hp_kern_extra_plus_vec, eval_variances),
            calc_mll_at(hp_gamma_vec, hp_kern_extra_minus_vec, eval_variances),
            gpflash_test::kFdStepObj);
    const bool extras_shape_ok = mll_result.mll_hp_grad_struct.d_mll_d_hp_kern_extra_vec.n_elem == 1;
    const double d_rho_err = extras_shape_ok
            ? std::abs(mll_result.mll_hp_grad_struct.d_mll_d_hp_kern_extra_vec(0) - d_rho_fd)
                    / std::max(std::abs(d_rho_fd), gpflash_test::kFdScaleFloor)
            : 1.0;

    // d(MLL)/d(sigma_f^2) and d(MLL)/d(sigma_k^2). The variances are not kernel hyperparameters, so these are
    // stepped through EvalVariances rather than through either kernel block.
    //
    // Different steps because the two variances differ by kHpSig2KNonUnit / kSig2FNonUnit, about 200: the objective
    // step would leave sigma_f^2's difference truncation-limited, and the kernel step sigma_k^2's roundoff-limited.
    gpflash::detail::EvalVariances opts_plus = eval_variances, opts_minus = eval_variances;
    opts_plus.sig2_f_val += gpflash_test::kFdStepKern;
    opts_minus.sig2_f_val -= gpflash_test::kFdStepKern;
    const double d_sig2_f_fd = central_diff(
            calc_mll_at(hp_gamma_vec, hp_kern_extra_vec, opts_plus),
            calc_mll_at(hp_gamma_vec, hp_kern_extra_vec, opts_minus),
            gpflash_test::kFdStepKern);
    const double d_sig2_f_err = std::abs(mll_result.mll_hp_grad_struct.d_mll_d_hp_sig2_f_val - d_sig2_f_fd)
            / std::max(std::abs(d_sig2_f_fd), gpflash_test::kFdScaleFloor);

    opts_plus = eval_variances;
    opts_minus = eval_variances;
    opts_plus.hp_sig2_k += gpflash_test::kFdStepObj;
    opts_minus.hp_sig2_k -= gpflash_test::kFdStepObj;
    const double d_sig2_k_fd = central_diff(
            calc_mll_at(hp_gamma_vec, hp_kern_extra_vec, opts_plus),
            calc_mll_at(hp_gamma_vec, hp_kern_extra_vec, opts_minus),
            gpflash_test::kFdStepObj);
    const double d_sig2_k_err = std::abs(mll_result.mll_hp_grad_struct.d_mll_d_hp_sig2_k - d_sig2_k_fd)
            / std::max(std::abs(d_sig2_k_fd), gpflash_test::kFdScaleFloor);

    const bool pass = extras_shape_ok && d_rho_err < gpflash_test::kTolFdObj && d_sig2_f_err < gpflash_test::kTolFd
            && d_sig2_k_err < gpflash_test::kTolFdObj;
    char detail[gpflash_test::kDetailLen];
    std::snprintf(detail, sizeof(detail), "hp_rho=%.1e sig2_f=%.1e sig2_k=%.1e", d_rho_err, d_sig2_f_err, d_sig2_k_err);
    report(name, pass, detail);

    return pass;
}

// Noise-free MLL and gradient against a dense implementation. The reference first solves hp_sig2_k from
// K + eta I, then evaluates the same profiled objective; by the envelope theorem its gamma gradient is the
// fixed-hp_sig2_k derivative at that optimum.
static bool test_noise_free_mll_and_grad_vs_reference(const char* name) {
    const int n_dim = 3, n_train = 14;
    const double cond_max = 1.0e6;

    arma::mat X_ndim_ntrain_mat(n_dim, n_train);
    for (int i_train = 0; i_train < n_train; ++i_train)
        for (int i_dim = 0; i_dim < n_dim; ++i_dim)
            X_ndim_ntrain_mat(i_dim, i_train) =
                    0.3 * std::sin(0.7 * i_train + 0.4 * i_dim) + 0.2 * std::cos(0.2 * i_train - 0.9 * i_dim);
    arma::vec f_train_vec(n_train);
    for (int i_train = 0; i_train < n_train; ++i_train) f_train_vec(i_train) = std::sin(0.5 * i_train) + 0.15 * i_train;
    const arma::vec hp_gamma_vec{0.7, 1.1, 1.5};
    const arma::vec hp_kern_extra_vec;

    // Dense K + eta I, independent of the library's triangular assembly.
    arma::mat K_mat(n_train, n_train, arma::fill::ones);
    for (int i_row = 0; i_row < n_train; ++i_row)
        for (int i_col = i_row + 1; i_col < n_train; ++i_col) {
            double rdot_sq_norm = 0.0;
            for (int i_dim = 0; i_dim < n_dim; ++i_dim) {
                const double r_d =
                        (X_ndim_ntrain_mat(i_dim, i_row) - X_ndim_ntrain_mat(i_dim, i_col)) * hp_gamma_vec(i_dim);
                rdot_sq_norm += r_d * r_d;
            }
            const double kern_val = gpflash::kernel::Matern52::calc_kern_val_unfactored(rdot_sq_norm, nullptr);
            K_mat(i_row, i_col) = K_mat(i_col, i_row) = kern_val;
        }
    const double eta = arma::sum(arma::abs(K_mat), 1).max() / (cond_max - 1.0);
    arma::mat K_nugget_mat = K_mat;
    K_nugget_mat.diag() += eta;
    const arma::mat K_nugget_inv_mat = arma::inv_sympd(K_nugget_mat);
    const gpflash::kernel::KernelBase& kern_fun = gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Matern52);

    // The noise-free closed form reads hp_sig2_k off the residual, so it is where a mean model that changed the
    // residual wrongly would show.
    bool pass = true;
    for (const MeanModelCase& mean_model_case : kMeanModelCases) {
        // The reference mean model weights, then hp_sig2_k from the quadratic form they leave.
        const arma::mat Basis_ntrain_nbasis_mat = gpflash_test::build_mean_model_basis_ref_mat(
                X_ndim_ntrain_mat, mean_model_case.has_mean_model_constant, mean_model_case.mean_model_degree_vec);
        const arma::mat lin_sol_K_nugget_Basis_ntrain_nbasis_mat = K_nugget_inv_mat * Basis_ntrain_nbasis_mat;
        const arma::vec lin_sol_K_nugget_f_train_vec = K_nugget_inv_mat * f_train_vec;
        const arma::vec hp_beta_ref_vec = gpflash_test::calc_hp_beta_ref_vec(
                Basis_ntrain_nbasis_mat,
                lin_sol_K_nugget_Basis_ntrain_nbasis_mat,
                lin_sol_K_nugget_f_train_vec,
                cond_max);
        const arma::vec residual_vec = f_train_vec - Basis_ntrain_nbasis_mat * hp_beta_ref_vec;
        const double hp_sig2_k_ref = arma::dot(residual_vec, K_nugget_inv_mat * residual_vec) / n_train;
        gpflash::detail::EvalVariances ref_variances;
        ref_variances.hp_sig2_k = hp_sig2_k_ref;
        double mll_ref = 0.0;
        arma::vec grad_ref_vec;
        reference_mll_grad(
                kern_fun,
                X_ndim_ntrain_mat,
                f_train_vec,
                Basis_ntrain_nbasis_mat,
                hp_gamma_vec,
                /*hp_rho=*/0.0,
                ref_variances,
                cond_max,
                mll_ref,
                grad_ref_vec);

        const gpflash::detail::EvalVariances noise_free_variances;
        gpflash::ModelOptions model_opts;
        model_opts.cond_max = cond_max;
        model_opts.kernel_id = kern_fun.kernel_id();
        model_opts.has_mean_model_constant = mean_model_case.has_mean_model_constant;
        model_opts.mean_model_degree_vec = mean_model_case.mean_model_degree_vec;
        const gpflash::MllResult mll_result = gpflash::detail::eval_mll(
                kern_fun,
                gpflash::TrainingData{X_ndim_ntrain_mat, f_train_vec, noise_free_variances.sig2_f_val},
                gpflash::detail::build_hp_val(hp_gamma_vec, hp_kern_extra_vec, noise_free_variances.hp_sig2_k),
                model_opts,
                /*eval_grad=*/true);

        const double mll_err = std::fabs(mll_result.mll - mll_ref) / std::max(std::fabs(mll_ref), kRelScaleFloor);
        const double sig2_k_err = std::fabs(mll_result.hp_sig2_k - hp_sig2_k_ref) / hp_sig2_k_ref;
        const double grad_err = arma::norm(mll_result.mll_hp_grad_struct.d_mll_d_hp_gamma_vec - grad_ref_vec, "inf")
                / std::max(arma::norm(grad_ref_vec, "inf"), kRelScaleFloor);
        const bool case_pass = mll_result.hp_sig2_k_from_closed_form && mll_err < kTolReferenceImpl
                && sig2_k_err < kTolReferenceImpl && grad_err < kTolReferenceImpl;
        pass = pass && case_pass;

        char detail[gpflash_test::kDetailLen];
        std::snprintf(
                detail,
                sizeof(detail),
                "%s mll=%.1e sig2_k=%.1e grad=%.1e",
                mean_model_case.label,
                mll_err,
                sig2_k_err,
                grad_err);
        report(name, case_pass, detail);
    }

    return pass;
}

// build_hp_val carries both kernel blocks and hp_sig2_k through unchanged, with and without a kernel extra, and
// leaves the noise estimate at its sentinel.
static bool test_hp_val_round_trip(const char* name) {
    bool pass = true;
    for (const int n_hp_kern_extra : {0, 1}) {
        const int n_dim = 4;
        gpflash::HpVal hp_val_struct;
        hp_val_struct.hp_gamma_vec = arma::vec(n_dim);
        for (int i_dim = 0; i_dim < n_dim; ++i_dim) hp_val_struct.hp_gamma_vec(i_dim) = 0.4 + 0.35 * i_dim;
        hp_val_struct.hp_kern_extra_vec = arma::vec(n_hp_kern_extra);
        if (n_hp_kern_extra > 0) hp_val_struct.hp_kern_extra_vec(0) = gpflash_test::kRatQuadHpRho;
        hp_val_struct.hp_sig2_k = kHpSig2KNonUnit;

        const gpflash::HpVal round_tripped = gpflash::detail::build_hp_val(
                hp_val_struct.hp_gamma_vec, hp_val_struct.hp_kern_extra_vec, hp_val_struct.hp_sig2_k);

        const bool widths_ok = round_tripped.hp_gamma_vec.n_elem == static_cast<arma::uword>(n_dim)
                && round_tripped.hp_kern_extra_vec.n_elem == static_cast<arma::uword>(n_hp_kern_extra);
        // Compared by value, so a swapped pair fails rather than only a wrong width.
        const bool gamma_ok =
                arma::approx_equal(round_tripped.hp_gamma_vec, hp_val_struct.hp_gamma_vec, "absdiff", 0.0);
        const bool extra_ok =
                arma::approx_equal(round_tripped.hp_kern_extra_vec, hp_val_struct.hp_kern_extra_vec, "absdiff", 0.0);
        const bool sig2_k_ok = round_tripped.hp_sig2_k == hp_val_struct.hp_sig2_k;
        // build_hp_val is given no noise variance, so the estimate must stay at its sentinel.
        const bool noise_left_unset = round_tripped.hp_sig2_f_val < 0.0;

        const bool case_pass = widths_ok && gamma_ok && extra_ok && sig2_k_ok && noise_left_unset;
        pass = pass && case_pass;
        char detail[gpflash_test::kDetailLen];
        std::snprintf(
                detail,
                sizeof(detail),
                "n_hp_kern_extra=%d widths=%d gamma=%d extra=%d sig2_k=%d noise_unset=%d",
                n_hp_kern_extra,
                static_cast<int>(widths_ok),
                static_cast<int>(gamma_ok),
                static_cast<int>(extra_ok),
                static_cast<int>(sig2_k_ok),
                static_cast<int>(noise_left_unset));
        report(name, case_pass, detail);
    }

    return pass;
}

// Asserts the MLL and hp_gamma gradient against a dense reference, for one mean model at the given cond_max and
// the two nuggets' bounds, and the gradients against central differences.
static bool test_mean_model_mll_and_grad(
        const char* name,
        const MeanModelCase& mean_model_case,
        const double cond_max,
        const gpflash::EigMaxUbMethod mean_model_eig_max_ub_method = gpflash::EigMaxUbMethod::Trace,
        const gpflash::EigMaxUbMethod kmat_eig_max_ub_method = gpflash::EigMaxUbMethod::Gershgorin) {
    const int n_dim = 3, n_train = 40;
    std::mt19937_64 rng(777);
    arma::mat X_ndim_ntrain_mat(n_dim, n_train);
    for (double& x_val : X_ndim_ntrain_mat) x_val = draw_uniform_unit(rng);

    // A prior mean that the {2, 1, 0} mean model basis, [1, x_1^2, x_2], spans, plus what the kernel must fit.
    arma::vec f_train_vec(n_train);
    for (int i_train = 0; i_train < n_train; ++i_train) {
        const double x_1 = X_ndim_ntrain_mat(0, i_train), x_2 = X_ndim_ntrain_mat(1, i_train);
        const double x_3 = X_ndim_ntrain_mat(2, i_train);
        f_train_vec(i_train) = 1.5 - 3.0 * x_1 * x_1 + x_2 + 0.2 * std::sin(9.0 * x_3);
    }

    gpflash::detail::EvalVariances eval_variances;
    eval_variances.hp_sig2_k = kHpSig2KNonUnit;
    eval_variances.sig2_f_val = kSig2FNonUnit;
    const arma::vec hp_gamma_vec{1.1, 0.8, 1.4};
    const gpflash::kernel::KernelBase& kern_fun = gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::SqExp);
    const gpflash::TrainingData train_data{X_ndim_ntrain_mat, f_train_vec, eval_variances.sig2_f_val};

    gpflash::ModelOptions model_opts;
    model_opts.kernel_id = kern_fun.kernel_id();
    model_opts.cond_max = cond_max;
    model_opts.has_mean_model_constant = mean_model_case.has_mean_model_constant;
    model_opts.mean_model_degree_vec = mean_model_case.mean_model_degree_vec;
    model_opts.mean_model_eig_max_ub_method = mean_model_eig_max_ub_method;
    model_opts.kmat_eig_max_ub_method = kmat_eig_max_ub_method;
    const auto calc_mll_at = [&](const arma::vec& hp_gamma_at_vec, const bool eval_grad) {
        const gpflash::HpVal hp_val_struct =
                gpflash::detail::build_hp_val(hp_gamma_at_vec, arma::vec(), eval_variances.hp_sig2_k);
        return gpflash::eval_mll(train_data, hp_val_struct, model_opts, eval_grad);
    };
    const gpflash::MllResult mll_result = calc_mll_at(hp_gamma_vec, /*eval_grad=*/true);
    const arma::vec& d_mll_d_hp_gamma_vec = mll_result.mll_hp_grad_struct.d_mll_d_hp_gamma_vec;

    const arma::mat Basis_ntrain_nbasis_mat = gpflash_test::build_mean_model_basis_ref_mat(
            X_ndim_ntrain_mat, mean_model_case.has_mean_model_constant, mean_model_case.mean_model_degree_vec);

    // The value and hp_gamma gradient from the dense reference on the same basis.
    double mll_ref = 0.0;
    arma::vec grad_ref_vec;
    reference_mll_grad(
            kern_fun,
            X_ndim_ntrain_mat,
            f_train_vec,
            Basis_ntrain_nbasis_mat,
            hp_gamma_vec,
            gpflash_test::kHpRhoUnused,
            eval_variances,
            model_opts.cond_max,
            mll_ref,
            grad_ref_vec,
            mean_model_eig_max_ub_method,
            kmat_eig_max_ub_method);

    // The hp_gamma gradient against a central difference of the MLL.
    arma::vec grad_fd_vec(n_dim);
    for (int i_dim = 0; i_dim < n_dim; ++i_dim) {
        arma::vec hp_gamma_plus_vec = hp_gamma_vec, hp_gamma_minus_vec = hp_gamma_vec;
        hp_gamma_plus_vec(i_dim) += gpflash_test::kFdStepObj;
        hp_gamma_minus_vec(i_dim) -= gpflash_test::kFdStepObj;
        const double mll_plus = calc_mll_at(hp_gamma_plus_vec, /*eval_grad=*/false).mll;
        const double mll_minus = calc_mll_at(hp_gamma_minus_vec, /*eval_grad=*/false).mll;
        grad_fd_vec(i_dim) = (mll_plus - mll_minus) / (2.0 * gpflash_test::kFdStepObj);
    }

    // The two variance gradients against central differences, stepped as extras_and_variance_grad_vs_fd steps
    // them: hp_sig2_k through the hyperparameters, sig2_f_val through the data.
    const auto calc_mll_at_variances = [&](const double hp_sig2_k, const double sig2_f_val) {
        const gpflash::TrainingData train_data_at{X_ndim_ntrain_mat, f_train_vec, sig2_f_val};
        const gpflash::HpVal hp_val_at_struct = gpflash::detail::build_hp_val(hp_gamma_vec, arma::vec(), hp_sig2_k);
        return gpflash::eval_mll(train_data_at, hp_val_at_struct, model_opts, /*return_mll_gradient=*/false).mll;
    };
    const double hp_sig2_k = eval_variances.hp_sig2_k, sig2_f_val = eval_variances.sig2_f_val;
    const double d_mll_d_hp_sig2_k_fd = (calc_mll_at_variances(hp_sig2_k + gpflash_test::kFdStepObj, sig2_f_val)
                                         - calc_mll_at_variances(hp_sig2_k - gpflash_test::kFdStepObj, sig2_f_val))
            / (2.0 * gpflash_test::kFdStepObj);
    const double d_mll_d_hp_sig2_f_val_fd = (calc_mll_at_variances(hp_sig2_k, sig2_f_val + gpflash_test::kFdStepKern)
                                             - calc_mll_at_variances(hp_sig2_k, sig2_f_val - gpflash_test::kFdStepKern))
            / (2.0 * gpflash_test::kFdStepKern);
    const double d_mll_d_hp_sig2_k_err =
            std::abs(mll_result.mll_hp_grad_struct.d_mll_d_hp_sig2_k - d_mll_d_hp_sig2_k_fd)
            / std::max(std::abs(d_mll_d_hp_sig2_k_fd), gpflash_test::kFdScaleFloor);
    const double d_mll_d_hp_sig2_f_val_err =
            std::abs(mll_result.mll_hp_grad_struct.d_mll_d_hp_sig2_f_val - d_mll_d_hp_sig2_f_val_fd)
            / std::max(std::abs(d_mll_d_hp_sig2_f_val_fd), gpflash_test::kFdScaleFloor);

    const bool width_ok = mll_result.hp_beta_vec.n_elem == Basis_ntrain_nbasis_mat.n_cols;
    const double mll_err = std::abs(mll_result.mll - mll_ref) / std::max(std::abs(mll_ref), kRelScaleFloor);
    const double grad_ref_err =
            arma::norm(d_mll_d_hp_gamma_vec - grad_ref_vec) / std::max(kRelScaleFloor, arma::norm(grad_ref_vec));
    const double grad_fd_err = arma::norm(d_mll_d_hp_gamma_vec - grad_fd_vec)
            / std::max(gpflash_test::kFdScaleFloor, arma::norm(grad_fd_vec));

    // The gradient neglects a term that, for this well-conditioned basis, scales as 1 / (cond_max - 1). At the default
    // cond_max it is below the central differences' tolerance. At a lower cond_max the central differences see it, so
    // its relative size times (cond_max - 1) must be of order one.
    const bool is_default_cond_max = cond_max == gpflash::COND_MAX_DEFAULT;
    const double grad_fd_err_scaled = grad_fd_err * (cond_max - 1.0);
    const bool is_order_one = grad_fd_err_scaled > 1.0 / kOrderOfMagnitude && grad_fd_err_scaled < kOrderOfMagnitude;
    bool grad_fd_ok = false;
    if (is_default_cond_max) {
        grad_fd_ok = grad_fd_err < gpflash_test::kTolFdObj && d_mll_d_hp_sig2_k_err < gpflash_test::kTolFdObj
                && d_mll_d_hp_sig2_f_val_err < gpflash_test::kTolFd;
    } else {
        grad_fd_ok = is_order_one;
    }
    const bool case_pass = width_ok && mll_err < kTolReferenceImpl && grad_ref_err < kTolReferenceImpl && grad_fd_ok;
    char detail[gpflash_test::kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "%s %s cond_max=%.0e n_basis=%d mll=%.1e grad=%.1e/%.1e (x(cond_max-1)=%.2f) sig2_k=%.1e sig2_f=%.1e",
            mean_model_case.label,
            gpflash::to_eig_max_ub_method_name(mean_model_eig_max_ub_method),
            cond_max,
            static_cast<int>(mll_result.hp_beta_vec.n_elem),
            mll_err,
            grad_ref_err,
            grad_fd_err,
            grad_fd_err_scaled,
            d_mll_d_hp_sig2_k_err,
            d_mll_d_hp_sig2_f_val_err);
    return report(name, case_pass, detail);
}

// Asserts the MLL and hp_gamma gradient on a nearly dependent mean model basis against the dense reference: x_1 and x_2
// on two dimensions that differ by a small perturbation, swept over hp_gamma.
static bool test_ill_conditioned_mean_model_basis(const char* name) {
    const int n_dim = 2, n_train = 40;
    std::mt19937_64 rng(4242);
    arma::mat X_ndim_ntrain_mat(n_dim, n_train);
    for (double& x_val : X_ndim_ntrain_mat) x_val = draw_uniform_unit(rng);
    // x_2 = x_1 + x_2_perturbation u, u on [0, 1], which leaves the two columns dependent to beyond cond_max.
    const double x_2_perturbation = 1.0e-6;
    X_ndim_ntrain_mat.row(1) = X_ndim_ntrain_mat.row(0) + x_2_perturbation * X_ndim_ntrain_mat.row(1);
    arma::vec f_train_vec(n_train);
    for (int i_train = 0; i_train < n_train; ++i_train)
        f_train_vec(i_train) = 1.0 + std::sin(3.0 * X_ndim_ntrain_mat(1, i_train));

    gpflash::detail::EvalVariances eval_variances;
    eval_variances.hp_sig2_k = 1.0;
    eval_variances.sig2_f_val = 1e-4;
    const gpflash::TrainingData train_data{X_ndim_ntrain_mat, f_train_vec, eval_variances.sig2_f_val};
    const gpflash::kernel::KernelBase& kern_fun = gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::SqExp);
    gpflash::ModelOptions model_opts;
    model_opts.kernel_id = kern_fun.kernel_id();
    model_opts.mean_model_degree_vec = {1, 1};
    const arma::mat Basis_ntrain_nbasis_mat =
            gpflash_test::build_mean_model_basis_ref_mat(X_ndim_ntrain_mat, true, model_opts.mean_model_degree_vec);

    // The preconditioned mean model Gram matrix's condition number is bounded only by cond_max here, which costs the
    // dense reference a decade of accuracy. Still tight enough to fail a mean model nugget off by a factor of two at
    // every hp_gamma scale.
    const double tol_ill_conditioned = 10.0 * kTolReferenceImpl;

    bool pass = true;
    for (const double hp_gamma_scale : {1e-2, 1.0, 1e2}) {
        const arma::vec hp_gamma_vec{hp_gamma_scale, hp_gamma_scale};
        const gpflash::HpVal hp_val_struct =
                gpflash::detail::build_hp_val(hp_gamma_vec, arma::vec(), eval_variances.hp_sig2_k);
        double mll_err = arma::datum::inf, grad_err = arma::datum::inf;
        std::string error_message;
        try {
            const gpflash::MllResult mll_result =
                    gpflash::eval_mll(train_data, hp_val_struct, model_opts, /*return_mll_gradient=*/true);
            double mll_ref = 0.0;
            arma::vec grad_ref_vec;
            reference_mll_grad(
                    kern_fun,
                    X_ndim_ntrain_mat,
                    f_train_vec,
                    Basis_ntrain_nbasis_mat,
                    hp_gamma_vec,
                    gpflash_test::kHpRhoUnused,
                    eval_variances,
                    model_opts.cond_max,
                    mll_ref,
                    grad_ref_vec);
            mll_err = std::fabs(mll_result.mll - mll_ref) / std::max(std::fabs(mll_ref), kRelScaleFloor);
            grad_err = arma::norm(mll_result.mll_hp_grad_struct.d_mll_d_hp_gamma_vec - grad_ref_vec)
                    / std::max(arma::norm(grad_ref_vec), kRelScaleFloor);
        } catch (const std::exception& error) {
            error_message = error.what();
        }

        const bool case_pass = mll_err < tol_ill_conditioned && grad_err < tol_ill_conditioned;
        pass = pass && case_pass;
        char detail[gpflash_test::kDetailLen];
        std::snprintf(
                detail, sizeof(detail), "hp_gamma_scale=%g mll=%.1e grad=%.1e", hp_gamma_scale, mll_err, grad_err);
        // Appended outside the buffer, which a library message would overrun.
        const std::string detail_with_raise =
                error_message.empty() ? std::string(detail) : std::string(detail) + " raised: " + error_message;
        report(name, case_pass, detail_with_raise.c_str());
    }

    return pass;
}

int main() {
    std::printf("== GpFlash marginal log likelihood ==\n");
    const double rel_tol = kTolReferenceImpl;
    bool ok = true;
    ok = run_kernel(gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::SqExp), "SqExp", rel_tol) && ok;
    ok = run_kernel(gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Matern52), "Matern52", rel_tol) && ok;
    ok = run_kernel(gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Matern32), "Matern32", rel_tol) && ok;
    ok = run_kernel(
                 gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::RatQuad),
                 "RatQuad",
                 rel_tol,
                 gpflash_test::kRatQuadHpRho)
            && ok;
    ok = test_noise_free_mll_and_grad_vs_reference("noise_free_mll_and_grad_vs_reference") && ok;
    for (const MeanModelCase& mean_model_case : kMeanModelCases)
        ok = test_mean_model_mll_and_grad("mean_model_mll_and_grad", mean_model_case, gpflash::COND_MAX_DEFAULT) && ok;
    // The kernel matrix nugget sized from the trace bound, which every other case leaves at Gershgorin.
    for (const MeanModelCase& mean_model_case : kMeanModelCases)
        ok = test_mean_model_mll_and_grad(
                     "mean_model_mll_and_grad/kmat_trace",
                     mean_model_case,
                     gpflash::COND_MAX_DEFAULT,
                     /*mean_model_eig_max_ub_method=*/gpflash::EigMaxUbMethod::Trace,
                     /*kmat_eig_max_ub_method=*/gpflash::EigMaxUbMethod::Trace)
                && ok;
    // Three lower cond_max, two decades apart end to end, where the neglected gradient term is large enough for the
    // central differences to measure how it scales.
    for (const double cond_max : {kCondMaxStrongNugget, 10.0 * kCondMaxStrongNugget, 100.0 * kCondMaxStrongNugget})
        for (const gpflash::EigMaxUbMethod eig_max_ub_method :
             {gpflash::EigMaxUbMethod::Trace, gpflash::EigMaxUbMethod::Gershgorin})
            ok = test_mean_model_mll_and_grad(
                         "mean_model_mll_and_grad", kMonomialMeanModelCase, cond_max, eig_max_ub_method)
                    && ok;
    ok = test_ill_conditioned_mean_model_basis("ill_conditioned_mean_model_basis") && ok;
    std::printf("== Gradient stages in isolation ==\n");
    ok = test_extras_and_variance_grad_vs_fd("extras_and_variance_grad_vs_fd/RatQuad") && ok;
    ok = test_kern_term_hp_grad_vs_dense(
                 gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::SqExp), "kern_term_hp_grad_vs_dense/SqExp")
            && ok;
    ok = test_kern_term_hp_grad_vs_dense(
                 gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Matern52),
                 "kern_term_hp_grad_vs_dense/Matern52")
            && ok;
    std::printf("== Public interface ==\n");
    ok = test_hp_val_round_trip("hp_val_round_trip") && ok;
    ok = test_public_eval_mll_reports_what_it_solved() && ok;
    std::printf("== Error handling ==\n");
    ok = test_indefinite_sigma_throws() && ok;
    ok = test_degenerate_noise_free_stays_finite() && ok;
    ok = test_positive_min_bounds_positivity_checks() && ok;
    std::printf("\n%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");

    return ok ? 0 : 1;
}
