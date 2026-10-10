/**
 * SigmaGrad.hpp -- the gradient mirror of SigmaMatrix.hpp: Psi, the weight every derivative of Sigma is
 * contracted against, its trace, and the two variance derivatives that trace feeds.
 *
 *   d(MLL)/d(hp) = <Psi, d(Sigma)/d(hp)>_F
 *   Psi          = (v_res v_res^T - Sigma^-1) / 2
 *   v_res        = Sigma^-1 (f_train - Basis hp_beta_vec),   Basis the mean model basis at the training points
 *
 * The gradient neglects the term in d(hp_beta_vec)/d(hp). It is nonzero only with more than one mean model weight,
 * because of the mean model nugget, and scales as 1 / (cond_max - 1) while the mean model basis is well conditioned.
 *
 * Psi is never formed. The kernel term takes one entry at a time, and only the diagonal is materialized, for the
 * terms that read every entry of it.
 */
#pragma once

#include <armadillo>

namespace gpflash {
namespace detail {

// Psi_ij.
inline double calc_psi_val(const double lin_sol_i, const double lin_sol_j, const double sigma_inv_ij) {
    return 0.5 * (lin_sol_i * lin_sol_j - sigma_inv_ij);
}

// diag(Psi), the i == j case over the whole diagonal.
inline arma::vec calc_psi_diag_vec(const arma::vec& lin_sol_sigma_residual_vec, const arma::mat& Sigma_inv_mat_lower) {
    return 0.5 * (arma::square(lin_sol_sigma_residual_vec) - Sigma_inv_mat_lower.diag());
}

// tr(Psi). Only Sigma^-1's diagonal is read, so the work is O(n_train), and neither argument is written. The one
// quantity both variance derivatives below are built from.
inline double calc_trace_psi(const arma::vec& lin_sol_sigma_residual_vec, const arma::mat& Sigma_inv_mat_lower) {
    const int n_train = static_cast<int>(Sigma_inv_mat_lower.n_rows);
    double trace_sigma_inv = 0.0;
    for (int i_diag = 0; i_diag < n_train; ++i_diag) trace_sigma_inv += Sigma_inv_mat_lower(i_diag, i_diag);

    return 0.5 * (arma::dot(lin_sol_sigma_residual_vec, lin_sol_sigma_residual_vec) - trace_sigma_inv);
}

// d(MLL)/d(sigma_f^2) = <Psi, d(Sigma)/d(sigma_f^2)>. That derivative is the identity, so the contraction is
// tr(Psi) unchanged. Named rather than assigned at the call site, so the equality is stated where it holds.
inline double calc_d_mll_d_sig2_f(const double trace_psi) { return trace_psi; }

// <Psi, Sigma> = (r^T Sigma^-1 r - n_train) / 2, from <v_res v_res^T, Sigma> = r^T Sigma^-1 r and
// <Sigma^-1, Sigma> = n_train. r^T Sigma^-1 r is lin_sol_L_residual_sq_norm, read off L_Sigma. Named rather than
// assigned at the call site, as calc_d_mll_d_sig2_f is.
inline double calc_trace_psi_sigma(const double lin_sol_L_residual_sq_norm, const int n_train) {
    return 0.5 * (lin_sol_L_residual_sq_norm - n_train);
}

// d(MLL)/d(sigma_k^2) = <Psi, K + eta W>, taken as <Psi, (Sigma - sig2_f_val I) / hp_sig2_k> so Sigma^-1 is
// not read again here: the caller supplies <Psi, Sigma>, and the noise part is sig2_f_val tr(Psi). Where hp_sig2_k was
// solved this is exactly zero, the stationarity defining its closed form.
inline double calc_d_mll_d_sig2_k(
        const double trace_psi_sigma, const double trace_psi, const double sig2_f_val, const double hp_sig2_k) {
    return (trace_psi_sigma - sig2_f_val * trace_psi) / hp_sig2_k;
}

}  // namespace detail
}  // namespace gpflash
