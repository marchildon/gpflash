/**
 * SigmaMatrix.hpp -- the last assembly step when hp_sig2_k is supplied, in place on KernelAssembly's Buffer_mat:
 * K_corr becomes Sigma_corr. Only the lower triangle and the diagonal are written, so c^(rdot) survives
 * above it.
 */
#pragma once

#include <armadillo>

#include "gpflash/core/Platform.hpp"  // GPFLASH_RESTRICT

namespace gpflash {
namespace detail {

// Multiply the lower triangle, diagonal included, by scale_factor. Strict upper: untouched
//
// mat_to_scale lower with diag memory: IN: the matrix to scale, OUT: the same, scaled.
inline void scale_lower_with_diag_inplace(arma::mat& mat_to_scale, const double scale_factor) {
    const int n_train = static_cast<int>(mat_to_scale.n_rows);
    for (int i_col = 0; i_col < n_train; ++i_col) {
        double* GPFLASH_RESTRICT col_ptr = mat_to_scale.colptr(i_col);
        for (int i_row = i_col; i_row < n_train; ++i_row) col_ptr[i_row] *= scale_factor;
    }
}

// Add the diagonal of Sigma_corr = hp_sig2_k (K_corr + eta I) + sigma_f^2 P^-2, whose off-diagonal entries must
// already carry hp_sig2_k. inv_precon_scale_vec is 1 / diag(P), empty for P = I.
inline void add_nugget_and_noise_to_diag_inplace(
        arma::mat& Sigma_corr_mat_lower,
        const double eta,
        const double sig2_f_val,
        const double hp_sig2_k,
        const arma::vec& inv_precon_scale_vec) {
    // One value for every entry when P = I, which an empty scale means, or when there is no noise.
    if (inv_precon_scale_vec.is_empty() || sig2_f_val == 0.0) {
        Sigma_corr_mat_lower.diag() += hp_sig2_k * eta + sig2_f_val;

        return;
    }
    // Only sigma_f^2 varies with the point: it belongs to Sigma, which carries no P. The nugget does not,
    // eta W with W = P P giving P^-1 (eta W) P^-1 = eta I.
    const double* GPFLASH_RESTRICT inv_precon_scale_ptr = inv_precon_scale_vec.memptr();
    const int n_train = static_cast<int>(Sigma_corr_mat_lower.n_rows);
    const double nugget = hp_sig2_k * eta;
    for (int i_diag = 0; i_diag < n_train; ++i_diag)
        Sigma_corr_mat_lower(i_diag, i_diag) +=
                nugget + sig2_f_val * inv_precon_scale_ptr[i_diag] * inv_precon_scale_ptr[i_diag];
}

// The noisy case, hp_sig2_k supplied: Sigma_corr = hp_sig2_k (K_corr + eta I) + sigma_f^2 P^-2. The posterior
// takes this path at any sigma_f^2, its hp_sig2_k coming from a finished fit.
//
// K_corr_in_Sigma_corr_out_mat_lower lower with diag memory: IN: K_corr, OUT: Sigma_corr
inline void build_sigma_corr_inplace(
        arma::mat& K_corr_in_Sigma_corr_out_mat_lower,
        const double eta,
        const double sig2_f_val,
        const double hp_sig2_k,
        const arma::vec& inv_precon_scale_vec) {
    scale_lower_with_diag_inplace(K_corr_in_Sigma_corr_out_mat_lower, hp_sig2_k);
    add_nugget_and_noise_to_diag_inplace(
            K_corr_in_Sigma_corr_out_mat_lower, eta, sig2_f_val, hp_sig2_k, inv_precon_scale_vec);
}

}  // namespace detail
}  // namespace gpflash
