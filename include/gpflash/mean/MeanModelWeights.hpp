/**
 * MeanModelWeights.hpp -- the mean model weights in closed form, and the residual they leave.
 *
 * Both read forward solves against the caller's Cholesky factor L, L_K_nugget noise-free and L_Sigma noisy:
 *
 *   lin_sol_L_f_train_vec               L^-1 f_train
 *   lin_sol_L_Basis_ntrain_nbasis_mat   L^-1 Basis, the mean model basis at the training points
 *
 * With more than one weight, the mean model Gram matrix Z = Basis^T (L L^T)^-1 Basis is conditioned as the kernel
 * matrix is: scaled to unit diagonal, given the nugget mean_model_eta from the bound mean_model_eig_max_ub_method
 * selects, factorized, and scaled back. The weights then solve
 *
 *   (Z + mean_model_eta diag(Z)) hp_beta_vec = Basis^T (L L^T)^-1 f_train
 */
#pragma once

#include <armadillo>

#include "gpflash/core/Errors.hpp"                          // ErrorNumericalFailure, error_prefix
#include "gpflash/inputs/EigMaxUbMethod.hpp"                // EigMaxUbMethod
#include "gpflash/linear_algebra/Cholesky.hpp"              // the solve_chol_*_inplace solves
#include "gpflash/preconditioning/ConditionedCholesky.hpp"  // build_eta_terms and the conditioned factorization
#include "gpflash/preconditioning/EtaTerms.hpp"             // EtaTerms

namespace gpflash {
namespace mean {
namespace detail {

// (n_basis) the mean model weights, solved as the file header states.
//
// Each argument carries one L^-1, so every inner product below is (L^-1 a)^T (L^-1 b) = a^T (L L^T)^-1 b. A column is
// read through unsafe_col, a read-only alias measured faster than col() inside arma::dot at small n_train.
inline arma::vec calc_mean_model_weights(
        const arma::vec& lin_sol_L_f_train_vec,
        const arma::mat& lin_sol_L_Basis_ntrain_nbasis_mat,
        const double cond_max,
        const EigMaxUbMethod mean_model_eig_max_ub_method) {
    const arma::uword n_basis = lin_sol_L_Basis_ntrain_nbasis_mat.n_cols;
    arma::vec hp_beta_vec;

    if (n_basis == 0) {
        // The zero mean model has no weights.
        return hp_beta_vec;
    } else if (n_basis == 1) {
        // One weight: the mean model Gram matrix is a scalar, so a division replaces the solve.
        const arma::vec lin_sol_L_basis_vec = lin_sol_L_Basis_ntrain_nbasis_mat.unsafe_col(0);
        const double lin_sol_L_basis_f_train_inner_prod_val = arma::dot(lin_sol_L_f_train_vec, lin_sol_L_basis_vec);
        const double mean_model_gram_val = arma::dot(lin_sol_L_basis_vec, lin_sol_L_basis_vec);
        // A nonzero basis this small underflows to zero here; more weights meet the same check in the preconditioner.
        if (!(mean_model_gram_val > 0.0))
            throw gpflash::ErrorNumericalFailure(
                    gpflash::detail::error_prefix("calc_mean_model_weights", "the mean model Gram matrix")
                    + " is zero, so the weight cannot be solved for.");
        hp_beta_vec = {lin_sol_L_basis_f_train_inner_prod_val / mean_model_gram_val};
    } else {
        // The lower triangle of Z and the right-hand side, one dot per entry: for a basis this narrow, measured faster
        // than DSYRK plus DGEMV.
        arma::mat mean_model_gram_mat_lower(n_basis, n_basis, arma::fill::zeros);
        arma::vec lin_sol_L_Basis_f_train_inner_prod_vec(n_basis);
        for (arma::uword i_basis = 0; i_basis < n_basis; ++i_basis) {
            const arma::vec lin_sol_L_basis_vec = lin_sol_L_Basis_ntrain_nbasis_mat.unsafe_col(i_basis);
            lin_sol_L_Basis_f_train_inner_prod_vec(i_basis) = arma::dot(lin_sol_L_f_train_vec, lin_sol_L_basis_vec);
            for (arma::uword i_other_basis = 0; i_other_basis <= i_basis; ++i_other_basis) {
                const arma::vec lin_sol_L_other_basis_vec = lin_sol_L_Basis_ntrain_nbasis_mat.unsafe_col(i_other_basis);
                const double mean_model_gram_val = arma::dot(lin_sol_L_basis_vec, lin_sol_L_other_basis_vec);
                mean_model_gram_mat_lower(i_basis, i_other_basis) = mean_model_gram_val;
            }
        }

        // mean_model_gram_mat_lower memory: IN Z, then Z_precon, Z_precon + mean_model_eta I,
        // OUT L_Z_nugget = chol(Z + mean_model_eta diag(Z)).
        const gpflash::detail::EtaTerms mean_model_eta_terms = gpflash::detail::build_eta_terms(
                /*is_unit_diag=*/false,
                mean_model_gram_mat_lower,
                cond_max,
                mean_model_eig_max_ub_method,
                /*want_active_row=*/false);
        gpflash::detail::add_nugget_to_diag_inplace(mean_model_gram_mat_lower, mean_model_eta_terms.eta);
        gpflash::detail::factorize_M_precon_to_L_M_inplace(
                mean_model_gram_mat_lower, mean_model_eta_terms.precon_scales, "calc_mean_model_weights/dpotrf");

        hp_beta_vec = lin_sol_L_Basis_f_train_inner_prod_vec;
        gpflash::detail::solve_chol_forward_vec_inplace(mean_model_gram_mat_lower, hp_beta_vec);
        gpflash::detail::solve_chol_transpose_vec_inplace(mean_model_gram_mat_lower, hp_beta_vec);
    }

    return hp_beta_vec;
}

// L^-1 (f_train - Basis hp_beta_vec), the GpFlash paper's v_res^(1/2): the solve is linear, so it is a
// combination of the solves already run rather than another.
inline arma::vec calc_lin_sol_L_residual_vec(
        const arma::vec& lin_sol_L_f_train_vec,
        const arma::mat& lin_sol_L_Basis_ntrain_nbasis_mat,
        const arma::vec& hp_beta_vec) {
    arma::vec lin_sol_L_residual_vec;

    if (hp_beta_vec.n_elem == 1) {
        // One weight: a scaled column, which skips the matrix-vector product.
        const arma::vec lin_sol_L_basis_vec = lin_sol_L_Basis_ntrain_nbasis_mat.unsafe_col(0);
        lin_sol_L_residual_vec = lin_sol_L_f_train_vec - hp_beta_vec(0) * lin_sol_L_basis_vec;
    } else {
        // Any other count: one DGEMV. With no weights the (n_train, 0) product is zero, leaving L^-1 f_train.
        lin_sol_L_residual_vec = lin_sol_L_f_train_vec - lin_sol_L_Basis_ntrain_nbasis_mat * hp_beta_vec;
    }

    return lin_sol_L_residual_vec;
}

}  // namespace detail
}  // namespace mean
}  // namespace gpflash
