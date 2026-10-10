/**
 * CholTerms.hpp -- what one Cholesky factor L yields: the mean model weights, the half residual, its norm, and
 * ln|L L^T|.
 *
 * L is the caller's factor:
 *   noise-free: L_K_nugget = chol(K + eta W), hp_sig2_k not yet known
 *   noisy:      L_Sigma = chol(Sigma)
 */
#pragma once

#include <armadillo>

#include "gpflash/core/Errors.hpp"              // the field checks below compose
#include "gpflash/core/Platform.hpp"            // GPFLASH_CHECK
#include "gpflash/inputs/EigMaxUbMethod.hpp"    // EigMaxUbMethod
#include "gpflash/linear_algebra/Cholesky.hpp"  // solve_chol_forward_vec, solve_chol_forward_mat, calc_ln_det_from_chol
#include "gpflash/mean/MeanModelWeights.hpp"    // calc_mean_model_weights, calc_lin_sol_L_residual_vec

namespace gpflash {
namespace detail {

struct CholTerms {
    arma::vec hp_beta_vec;                    // (n_basis) the mean model weights, the same solve on either path
    arma::vec lin_sol_L_residual_vec;         // L^-1 (f_train - Basis hp_beta_vec), the GpFlash paper's v_res^(1/2)
    double lin_sol_L_residual_sq_norm = 0.0;  // its squared norm
    // ln|L L^T|.
    //   noise-free: ln|K + eta W|
    //   noisy:      ln|Sigma|
    double ln_det = 0.0;
};

// Postcondition.
inline void check_chol_terms(const CholTerms& chol_terms, const char* caller) {
    check_all_finite_vals(chol_terms.hp_beta_vec, "CholTerms::hp_beta_vec", caller);
    check_positive_count(
            static_cast<int>(chol_terms.lin_sol_L_residual_vec.n_elem), "CholTerms::lin_sol_L_residual_vec", caller);
    check_all_finite_vals(chol_terms.lin_sol_L_residual_vec, "CholTerms::lin_sol_L_residual_vec", caller);

    // A squared norm, so non-negative; ln|.| of a positive-definite matrix is finite.
    check_finite_non_negative_val(
            chol_terms.lin_sol_L_residual_sq_norm, "CholTerms::lin_sol_L_residual_sq_norm", caller);
    check_finite_val(chol_terms.ln_det, "CholTerms::ln_det", caller);
}

// Reads L_mat_lower, L_K_nugget or L_Sigma, into CholTerms, leaving it intact.
// cond_max and mean_model_eig_max_ub_method size the mean model nugget.
inline CholTerms calc_chol_terms(
        const arma::mat& L_mat_lower,
        const arma::vec& f_train_vec,
        const arma::mat& Basis_ntrain_nbasis_mat,
        const double cond_max,
        const EigMaxUbMethod mean_model_eig_max_ub_method) {
    CholTerms chol_terms;

    // The forward solves; every quantity below is a combination of them
    const arma::vec lin_sol_L_f_train_vec = solve_chol_forward_vec(L_mat_lower, f_train_vec);
    const arma::mat lin_sol_L_Basis_ntrain_nbasis_mat = solve_chol_forward_mat(L_mat_lower, Basis_ntrain_nbasis_mat);

    // The mean model weights
    chol_terms.hp_beta_vec = mean::detail::calc_mean_model_weights(
            lin_sol_L_f_train_vec, lin_sol_L_Basis_ntrain_nbasis_mat, cond_max, mean_model_eig_max_ub_method);

    // Calculate remaining terms
    chol_terms.lin_sol_L_residual_vec = mean::detail::calc_lin_sol_L_residual_vec(
            lin_sol_L_f_train_vec, lin_sol_L_Basis_ntrain_nbasis_mat, chol_terms.hp_beta_vec);
    chol_terms.lin_sol_L_residual_sq_norm =
            arma::dot(chol_terms.lin_sol_L_residual_vec, chol_terms.lin_sol_L_residual_vec);
    chol_terms.ln_det = calc_ln_det_from_chol(L_mat_lower);

    GPFLASH_CHECK(check_chol_terms(chol_terms, "calc_chol_terms"));
    return chol_terms;
}

}  // namespace detail
}  // namespace gpflash
