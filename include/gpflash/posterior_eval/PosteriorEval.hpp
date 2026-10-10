/**
 * PosteriorEval.hpp -- the posterior mean and variance:
 *
 *     mean(x')     = basis(x')^T hp_beta_vec + sigma_k^2 k_*^T v,   v = Sigma^-1 ( f_train - Basis hp_beta_vec )
 *     variance(x') = sigma_k^2 ( k(x', x') - sigma_k^2 ||L_Sigma^-1 k_*||^2 )
 *
 * v is solved once at fit time and stored, so a mean costs the prior mean plus one dot product against it. basis(x') is
 * the mean model basis at the query point, Basis the mean model basis at the training points, and hp_beta_vec the mean
 * model weights the fit solved in closed form.
 *
 * Each quantity has a single-point form and a batch counterpart, differing only in BLAS level.
 */
#pragma once

#include <algorithm>  // std::max
#include <armadillo>

#include "gpflash/assembly/SymmetryMatrix.hpp"              // calc_symmetry_kern_diag_val_vec
#include "gpflash/kernels/KernelBase.hpp"                   // kernel::KernelBase
#include "gpflash/posterior_cache/PosteriorPrecompute.hpp"  // PosteriorPrecompute

namespace gpflash {
namespace detail {

// ── Mean ─────────────────────────────────────────────────────────────────────

// One query: the prior mean at x_post_vec, plus one dot against the fit-time residual solve.
inline double calc_mean_post_val(
        const PosteriorPrecompute& post_precompute, const arma::vec& x_post_vec, const arma::vec& k_post_vec) {
    const double prior_mean_post_val =
            post_precompute.mean_model.calc_prior_mean_val(x_post_vec, post_precompute.hp_beta_vec);

    return prior_mean_post_val
            + post_precompute.hp_sig2_k * arma::dot(k_post_vec, post_precompute.lin_sol_sigma_residual_vec);
}

// Batch counterpart: the prior mean at every query, plus one DGEMV.
inline arma::vec calc_mean_post_val_vec(
        const PosteriorPrecompute& post_precompute,
        const arma::mat& X_ndim_npost_mat,
        const arma::mat& K_ntrain_npost_mat) {
    arma::vec mean_post_vec =
            post_precompute.hp_sig2_k * (K_ntrain_npost_mat.t() * post_precompute.lin_sol_sigma_residual_vec);
    mean_post_vec += post_precompute.mean_model.calc_prior_mean_val_vec(X_ndim_npost_mat, post_precompute.hp_beta_vec);

    return mean_post_vec;
}

// ── The prior variance, k(x', x') ────────────────────────────────────────────
// What the posterior variance starts from. A symmetry model's is the two terms summed, which the kernel alone
// cannot report, so the model is asked rather than the kernel.

inline arma::vec calc_kern_diag_post_val_vec(
        const kernel::KernelBase& kern_fun,
        const PosteriorPrecompute& post_precompute,
        const arma::mat& Ydot_nfeature_npost_mat,
        const arma::vec& ydot_post_sq_norm_vec,
        const double* hp_kern_extra_ptr) {
    if (post_precompute.symmetry_spec.is_on())
        return calc_symmetry_kern_diag_val_vec(
                kern_fun,
                Ydot_nfeature_npost_mat,
                ydot_post_sq_norm_vec,
                post_precompute.symmetry_spec,
                hp_kern_extra_ptr);

    return kern_fun.calc_kern_diag_val_vec(ydot_post_sq_norm_vec, hp_kern_extra_ptr);
}

// The single-query counterpart, which reuses the batch form over a one-column matrix where symmetry is on: the
// mirrored diagonal needs the point's flipped-dimension norm, which the scalar interface does not carry.
inline double calc_kern_diag_post_val(
        const kernel::KernelBase& kern_fun,
        const PosteriorPrecompute& post_precompute,
        const arma::vec& ydot_post_vec,
        const double ydot_post_sq_norm,
        const double* hp_kern_extra_ptr) {
    if (post_precompute.symmetry_spec.is_on())
        return calc_kern_diag_post_val_vec(
                kern_fun, post_precompute, arma::mat(ydot_post_vec), arma::vec{ydot_post_sq_norm}, hp_kern_extra_ptr)(
                0);

    return kern_fun.calc_kern_diag_val(ydot_post_sq_norm, hp_kern_extra_ptr);
}

// ── Variance ─────────────────────────────────────────────────────────────────

// One query, against the prior variance the caller resolved above.
inline double calc_var_post_val(
        const PosteriorPrecompute& post_precompute, const double kern_diag_val, const arma::vec& lin_sol_L_k_post_vec) {
    const double var_post_val = post_precompute.hp_sig2_k
            * (kern_diag_val - post_precompute.hp_sig2_k * arma::dot(lin_sol_L_k_post_vec, lin_sol_L_k_post_vec));

    // Clamped because the two terms cancel at a training point and may leave a small negative value. A NaN passes
    // through, as in the batch clamp: std::max returns its first argument when the comparison fails.
    return std::max(var_post_val, 0.0);
}

// Batch counterpart. lin_sol_L_K_ntrain_npost_mat holds the forward half solves as columns.
inline arma::vec calc_var_post_val_vec(
        const PosteriorPrecompute& post_precompute,
        const arma::vec& kern_diag_val_vec,
        const arma::mat& lin_sol_L_K_ntrain_npost_mat) {
    const arma::vec quad_form_vec = arma::sum(arma::square(lin_sol_L_K_ntrain_npost_mat), 0).t();
    const arma::vec var_post_vec =
            post_precompute.hp_sig2_k * (kern_diag_val_vec - post_precompute.hp_sig2_k * quad_form_vec);

    // Clamped for the same reason as the single-query form above.
    return arma::clamp(var_post_vec, 0.0, arma::datum::inf);
}

}  // namespace detail
}  // namespace gpflash
