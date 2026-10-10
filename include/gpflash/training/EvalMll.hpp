/**
 * EvalMll.hpp -- the marginal log-likelihood, for any kernel implementing kernel::KernelBase.
 *
 * O(n_train^3) in a single (n_train, n_train) buffer, which arrives from assembly holding K_corr and is then
 * overwritten in place by every later stage. Its lower triangle and diagonal carry that chain while, when a gradient
 * follows, its strict upper holds c^(rdot) -- written once by assembly, read once by the gradient, untouched between:
 *
 *   lower with diag memory: IN: K_corr, OUT: Sigma_corr (noisy) or K_corr + eta I (noise-free), then its Cholesky
 *                           factor, then Sigma^-1
 *
 * The inverse is the GpFlash paper's recursive one, not the vendor DPOTRI.
 *
 * One driver per noise case (see "The two noise cases" below), each a sequence of the shared stages here and
 * in EvalMllGrad.hpp. Validation and every checked entry point are EvalMllDispatch.hpp's, including the
 * public one:
 *   gpflash::MllResult mll_result = gpflash::eval_mll(train_data, hp_val_struct, model_opts, return_mll_gradient);
 */
#pragma once

#include <algorithm>
#include <armadillo>
#include <cmath>
#include <string>

#include "gpflash/assembly/CholTerms.hpp"                   // CholTerms, calc_chol_terms
#include "gpflash/assembly/KernelAssembly.hpp"              // KernelAssembly, build_kern_assembly
#include "gpflash/assembly/SigmaGrad.hpp"                   // calc_trace_psi_sigma
#include "gpflash/assembly/SigmaMatrix.hpp"                 // build_sigma_corr_inplace, scale_lower_with_diag_inplace
#include "gpflash/core/Errors.hpp"                          // ErrorNumericalFailure, is_finite_val, POSITIVE_MIN
#include "gpflash/core/NoiseModel.hpp"                      // NoiseRequest, classify_noise_request
#include "gpflash/core/Platform.hpp"                        // GPFLASH_CHECK
#include "gpflash/inputs/HpVal.hpp"                         // HpVal
#include "gpflash/inputs/ModelOptions.hpp"                  // ModelOptions
#include "gpflash/inputs/TrainingData.hpp"                  // TrainingData
#include "gpflash/kernels/HpGradInputs.hpp"                 // make_hp_grad_inputs
#include "gpflash/kernels/KernelBase.hpp"                   // kernel::KernelBase
#include "gpflash/linear_algebra/Cholesky.hpp"              // solve_chol_transpose_vec_inplace
#include "gpflash/linear_algebra/RecursiveInv.hpp"          // recursive_dpotri_lower
#include "gpflash/preconditioning/ConditionedCholesky.hpp"  // the nugget and the conditioned factorization
#include "gpflash/preconditioning/EtaTerms.hpp"             // EtaTerms
#include "gpflash/training/EvalMllGrad.hpp"                 // calc_mll_hp_grad
#include "gpflash/training/EvalVariances.hpp"               // EvalVariances
#include "gpflash/training/MllResult.hpp"                   // MllResult
#include "gpflash/training/StageTiming.hpp"                 // GPFLASH_STAGE_BEGIN, GPFLASH_STAGE_MARK

namespace gpflash {
namespace detail {

// ── The two noise cases ──────────────────────────────────────────────────────
// One driver each, differing in whether hp_sig2_k factors out of the covariance:
//
//   noise-free  sig2_f_val == 0:  Sigma = hp_sig2_k (K + eta W)
//   noisy       sig2_f_val  > 0:  Sigma = hp_sig2_k (K + eta W) + sig2_f_val I
//
// Noise-free: the factor is L_K_nugget, the MLL solves hp_sig2_k in closed form, and the gradient inputs need
// a rescale into absolute coordinates. Noisy: the factor is L_Sigma, hp_sig2_k is the caller's, nothing
// rescales.

// ── The likelihood, read off the Cholesky factor ─────────────────────────────

// The hp_sig2_k that maximizes the MLL, for L_K_nugget, which leaves it outside the matrix. One division on a
// norm calc_chol_terms already took.
inline double calc_closed_form_hp_sig2_k(const CholTerms& chol_terms, const int n_train) {
    const double hp_sig2_k = chol_terms.lin_sol_L_residual_sq_norm / n_train;
    if (!is_finite_val(hp_sig2_k))
        throw ErrorNumericalFailure(
                "gpflash::eval_mll: the closed-form signal variance is not finite (hp_sig2_k = "
                + std::to_string(hp_sig2_k) + "), so the triangular solves failed at these inputs.");
    return std::max(hp_sig2_k, POSITIVE_MIN);
}

// Off L_Sigma: every variance is inside the matrix, so the determinant and the quadratic form stand as read,
// and hp_sig2_k is not read again here.
inline double calc_mll_from_L_Sigma(const CholTerms& chol_terms, const int n_train) {
    return -0.5 * (chol_terms.lin_sol_L_residual_sq_norm + chol_terms.ln_det) - n_train * arma::datum::log_sqrt2pi;
}

// Off L_K_nugget, at the hp_sig2_k above: hp_sig2_k was outside the matrix, so substituting it makes the
// quadratic form exactly n_train and the variance the matrix left out enters ln|Sigma| here.
inline double calc_mll_from_L_K_nugget(const CholTerms& chol_terms, const double hp_sig2_k, const int n_train) {
    const double ln_det_sigma = chol_terms.ln_det + n_train * std::log(hp_sig2_k);
    return -0.5 * (n_train + ln_det_sigma) - n_train * arma::datum::log_sqrt2pi;
}

// ── The two drivers ──────────────────────────────────────────────────────────

// Noise-free data: Sigma = hp_sig2_k (K + eta W). hp_sig2_k is an output -- solved in closed form and
// reported in the result -- so no variance arrives here at all. Until the rescale near the end, everything
// read off is L_K_nugget = chol(K + eta W), which differs from L_Sigma by a factor of sqrt(hp_sig2_k).
inline MllResult eval_mll_noise_free(
        const kernel::KernelBase& kern_fun,
        const arma::vec& x_center_vec,
        const arma::mat& Basis_ntrain_nbasis_mat,
        const TrainingData& train_data,
        const HpVal& hp_val_struct,
        const ModelOptions& model_opts,
        const bool eval_grad) {
    GPFLASH_STAGE_BEGIN();
    KernelAssembly kern_assembly = build_kern_assembly(
            kern_fun,
            train_data.X_ndim_ntrain_mat,
            x_center_vec,
            hp_val_struct.hp_gamma_vec,
            hp_val_struct.hp_kern_extra_vec,
            model_opts.symmetry_spec,
            model_opts.cond_max,
            model_opts.kmat_eig_max_ub_method,
            eval_grad);
    // Training never reads the scaled points again; only a posterior build does. Freed here so they do not sit
    // beside the gradient's own copy of X.
    kern_assembly.Ydot_nfeature_ntrain_mat.reset();
    GPFLASH_STAGE_MARK(kern_assembly_s);

    // Buffer_mat: K_corr here, then K_corr + eta I, L_K_nugget, Sigma^-1. Each call names both its
    // states, so no alias is bound before the final contents.
    const EtaTerms& kmat_eta_terms = kern_assembly.kmat_eta_terms;

    // K_corr + eta I is the whole matrix to factorize: the signal variance sits outside it.
    const double eta = kmat_eta_terms.eta;
    add_nugget_to_diag_inplace(kern_assembly.Buffer_mat, eta);
    factorize_M_precon_to_L_M_inplace(kern_assembly.Buffer_mat, kmat_eta_terms.precon_scales, "eval_mll/dpotrf");
    GPFLASH_STAGE_MARK(chol_s);

    // The buffer now holds L_K_nugget, short one factor of hp_sig2_k, which the reads below correct.
    const CholTerms chol_terms = calc_chol_terms(
            kern_assembly.Buffer_mat,
            train_data.f_train_vec,
            Basis_ntrain_nbasis_mat,
            model_opts.cond_max,
            model_opts.mean_model_eig_max_ub_method);
    const int n_train = static_cast<int>(train_data.f_train_vec.n_elem);
    const double hp_sig2_k = calc_closed_form_hp_sig2_k(chol_terms, n_train);
    // calc_trace_psi_sigma is exactly zero here: the closed-form hp_sig2_k makes r^T Sigma^-1 r = n_train.
    const double trace_psi_sigma = 0.0;
    GPFLASH_STAGE_MARK(lin_sol_s);

    MllResult mll_result;
    mll_result.mll = calc_mll_from_L_K_nugget(chol_terms, hp_sig2_k, n_train);
    mll_result.hp_beta_vec = chol_terms.hp_beta_vec;
    mll_result.hp_sig2_k_from_closed_form = true;
    mll_result.hp_sig2_k = hp_sig2_k;  // solved here, so reported back

    if (!eval_grad) {  // MLL-only fast path: Sigma^-1 is never formed
        GPFLASH_CHECK(check_mll_result(mll_result, "eval_mll_noise_free"));
        return mll_result;
    }

    // The gradient of the MLL is now calculated.
    //
    // lin_sol_sigma_residual_vec memory: IN: L_K_nugget^-1 residual, the half solve calc_chol_terms left behind,
    // OUT: (K + eta W)^-1 residual, rescaled to Sigma^-1 residual below. Named for the OUT state, as
    // Sigma_inv_mat_lower is.
    arma::vec lin_sol_sigma_residual_vec = chol_terms.lin_sol_L_residual_vec;
    solve_chol_transpose_vec_inplace(kern_assembly.Buffer_mat, lin_sol_sigma_residual_vec);
    // Last read of L_K_nugget was the transpose solve above; dpotri overwrites it, so no read of it may follow.
    recursive_dpotri_lower(kern_assembly.Buffer_mat.memptr(), n_train);

    // Final contents, so this name cannot go stale. Lower only: the strict upper holds c^(rdot).
    arma::mat& Sigma_inv_mat_lower = kern_assembly.Buffer_mat;

    // Both came off the K + eta W factor, so both are short the hp_sig2_k it left outside the matrix: divide
    // it out and they are the real Sigma's.
    lin_sol_sigma_residual_vec /= hp_sig2_k;
    scale_lower_with_diag_inplace(Sigma_inv_mat_lower, 1.0 / hp_sig2_k);
    const kernel::HpGradInputs hp_grad_inputs = kernel::make_hp_grad_inputs(
            train_data.X_ndim_ntrain_mat,
            kern_assembly.Y_nfeature_ntrain_mat,
            hp_val_struct.hp_gamma_vec,
            hp_val_struct.hp_kern_extra_vec);
    mll_result.mll_hp_grad_struct = calc_mll_hp_grad(
            kern_fun,
            hp_grad_inputs,
            kmat_eta_terms,
            lin_sol_sigma_residual_vec,
            Sigma_inv_mat_lower,
            trace_psi_sigma,
            /*sig2_f_val=*/0.0,
            hp_sig2_k,
            model_opts.symmetry_spec,
            kern_assembly.kern_common_rdot_mirror_packed_vec);
    GPFLASH_CHECK(check_mll_result(mll_result, "eval_mll_noise_free"));

    return mll_result;
}

// Noisy data: Sigma = hp_sig2_k (K + eta W) + sig2_f_val I. Both variances arrive from the caller, the factor is
// L_Sigma, and nothing is ever rescaled: every read off it is already absolute.
inline MllResult eval_mll_noisy(
        const kernel::KernelBase& kern_fun,
        const arma::vec& x_center_vec,
        const arma::mat& Basis_ntrain_nbasis_mat,
        const TrainingData& train_data,
        const HpVal& hp_val_struct,
        const EvalVariances& eval_variances,
        const ModelOptions& model_opts,
        const bool eval_grad) {
    GPFLASH_STAGE_BEGIN();
    KernelAssembly kern_assembly = build_kern_assembly(
            kern_fun,
            train_data.X_ndim_ntrain_mat,
            x_center_vec,
            hp_val_struct.hp_gamma_vec,
            hp_val_struct.hp_kern_extra_vec,
            model_opts.symmetry_spec,
            model_opts.cond_max,
            model_opts.kmat_eig_max_ub_method,
            eval_grad);
    // Training never reads the scaled points again; only a posterior build does. Freed here so they do not sit
    // beside the gradient's own copy of X.
    kern_assembly.Ydot_nfeature_ntrain_mat.reset();
    GPFLASH_STAGE_MARK(kern_assembly_s);

    // Buffer_mat: K_corr here, then Sigma_corr, L_Sigma, Sigma^-1. Each call names both its states, so
    // no alias is bound before the final contents.
    const EtaTerms& kmat_eta_terms = kern_assembly.kmat_eta_terms;

    // K_corr -> Sigma_corr, both variances applied here and nowhere else.
    const double eta = kmat_eta_terms.eta;
    build_sigma_corr_inplace(
            kern_assembly.Buffer_mat,
            eta,
            eval_variances.sig2_f_val,
            eval_variances.hp_sig2_k,
            kmat_eta_terms.precon_scales.inv_precon_scale_vec);
    factorize_M_precon_to_L_M_inplace(kern_assembly.Buffer_mat, kmat_eta_terms.precon_scales, "eval_mll/dpotrf");
    GPFLASH_STAGE_MARK(chol_s);

    // The buffer now holds L_Sigma, so every quantity read off it is already absolute.
    const CholTerms chol_terms = calc_chol_terms(
            kern_assembly.Buffer_mat,
            train_data.f_train_vec,
            Basis_ntrain_nbasis_mat,
            model_opts.cond_max,
            model_opts.mean_model_eig_max_ub_method);
    const int n_train = static_cast<int>(train_data.f_train_vec.n_elem);
    const double trace_psi_sigma = calc_trace_psi_sigma(chol_terms.lin_sol_L_residual_sq_norm, n_train);
    GPFLASH_STAGE_MARK(lin_sol_s);

    MllResult mll_result;
    mll_result.mll = calc_mll_from_L_Sigma(chol_terms, n_train);
    mll_result.hp_beta_vec = chol_terms.hp_beta_vec;
    mll_result.hp_sig2_k_from_closed_form = false;  // the caller supplied it, so it is not reported back
    if (!eval_grad) {                               // MLL-only fast path: Sigma^-1 is never formed
        GPFLASH_CHECK(check_mll_result(mll_result, "eval_mll_noisy"));
        return mll_result;
    }

    // The gradient of the MLL is now calculated.
    // lin_sol_sigma_residual_vec memory: IN: L_Sigma^-1 residual, OUT: Sigma^-1 residual. Named for the OUT state,
    // as Sigma_inv_mat_lower is.
    arma::vec lin_sol_sigma_residual_vec = chol_terms.lin_sol_L_residual_vec;
    solve_chol_transpose_vec_inplace(kern_assembly.Buffer_mat, lin_sol_sigma_residual_vec);
    // Last read of L_Sigma was the transpose solve above; dpotri overwrites it, so no read of it may follow.
    recursive_dpotri_lower(kern_assembly.Buffer_mat.memptr(), n_train);

    // Final contents, so this name cannot go stale. Lower only: the strict upper holds c^(rdot).
    arma::mat& Sigma_inv_mat_lower = kern_assembly.Buffer_mat;

    const kernel::HpGradInputs hp_grad_inputs = kernel::make_hp_grad_inputs(
            train_data.X_ndim_ntrain_mat,
            kern_assembly.Y_nfeature_ntrain_mat,
            hp_val_struct.hp_gamma_vec,
            hp_val_struct.hp_kern_extra_vec);
    mll_result.mll_hp_grad_struct = calc_mll_hp_grad(
            kern_fun,
            hp_grad_inputs,
            kmat_eta_terms,
            lin_sol_sigma_residual_vec,
            Sigma_inv_mat_lower,
            trace_psi_sigma,
            eval_variances.sig2_f_val,
            eval_variances.hp_sig2_k,
            model_opts.symmetry_spec,
            kern_assembly.kern_common_rdot_mirror_packed_vec);
    GPFLASH_CHECK(check_mll_result(mll_result, "eval_mll_noisy"));

    return mll_result;
}

// Checks nothing: the caller guarantees every input. EvalMllDispatch.hpp's eval_mll is the same evaluation
// with the checks, and is what every caller outside optimization/ should use.
//
// x_center_vec is the kernel's calc_x_center_vec of the training data, and Basis_ntrain_nbasis_mat the mean model
// basis at the training points; both are fixed through a fit.
inline MllResult eval_mll_inputs_unchecked(
        const kernel::KernelBase& kern_fun,
        const arma::vec& x_center_vec,
        const arma::mat& Basis_ntrain_nbasis_mat,
        const TrainingData& train_data,
        const HpVal& hp_val_struct,
        const EvalVariances& eval_variances,
        const ModelOptions& model_opts,
        const bool eval_grad) {
    if (classify_noise_request(eval_variances.sig2_f_val) == NoiseRequest::NoiseFree) {
        return eval_mll_noise_free(
                kern_fun, x_center_vec, Basis_ntrain_nbasis_mat, train_data, hp_val_struct, model_opts, eval_grad);
    } else {
        return eval_mll_noisy(
                kern_fun,
                x_center_vec,
                Basis_ntrain_nbasis_mat,
                train_data,
                hp_val_struct,
                eval_variances,
                model_opts,
                eval_grad);
    }
}

}  // namespace detail
}  // namespace gpflash
