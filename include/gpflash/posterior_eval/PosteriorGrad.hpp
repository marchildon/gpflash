/**
 * PosteriorGrad.hpp -- the input-space gradients of the GP posterior.
 *
 * Notation is the GpFlash paper's, with the mean model in source symbols. For a query point x', training points
 * x_i, signal variance sigma_k^2, prior mean basis(x')^T hp_beta_vec with hp_beta_vec the mean model weights and
 * Basis the mean model basis at the training points, and k_* the cross-covariance vector k(x', x_i):
 *
 *     d(mean)/d(x')     = d(basis(x')^T hp_beta_vec)/d(x') + sigma_k^2 q_mu^T H^(r)
 *     d(variance)/d(x') = sigma_k^2 d(k(x', x'))/d(x') - 2 sigma_k^4 q_sigma^T H^(r)
 *
 * Both weight vectors fold the kernel's shared derivative factor c^(rdot) = s^(rdot) c^(k), which the
 * cross-covariance already left behind, into one of the two solves:
 *
 *     q_mu    = v_res         (.) c^(rdot),  v_res = Sigma^-1 (f_train - Basis hp_beta_vec), solved once at fit time
 *     q_sigma = Sigma^-1 k_*  (.) c^(rdot)
 *
 * with (.) elementwise. One function sums q against H^(r) for either, so outside it the two gradients differ only
 * in their solve, their leading factor, and the term each adds: the prior mean's gradient to the mean,
 * d(k(x', x'))/d(x') to the variance.
 *
 * H^(r) is the (n_train, n_dim) matrix of kernel-specific factors h^(r,l)_i. It is summed over the training
 * index, never formed:
 *
 *     (q^T H^(r))_l = sum_i q_i h^(r,l)_i
 *
 * A single query returns (n_dim); the batch forms (n_dim, n_post), one column per query.
 *
 * A symmetry model adds the mirrored term's share to each gradient. Its h^(r,l) is gamma_l^2 (x'_l - s_l x_il),
 * which is s_l times the base form taken at the mirrored query point S x', so the same kernel method serves it
 * and its result is scaled back into the query's own dimensions by s. Its q is formed from the mirrored
 * c^(rdot) the cross-covariance captured alongside the base one, against the same solve.
 */
#pragma once

#include <armadillo>

#include "gpflash/assembly/SymmetryMatrix.hpp"              // the mirrored points and diagonal derivative
#include "gpflash/inputs/HpVal.hpp"                         // calc_hp_kern_extra_ptr
#include "gpflash/kernels/KernelBase.hpp"                   // kernel::KernelBase
#include "gpflash/posterior_cache/PosteriorPrecompute.hpp"  // PosteriorPrecompute

namespace gpflash {
namespace detail {

// ── The prior-variance term, d(k(x', x'))/d(x') ──────────────────────────────
// Empty where the diagonal does not move with the query. A symmetry model's does, through its mirrored term,
// even under a base kernel whose own diagonal is constant -- which is the only kind a mirror wraps, so the
// two sources are alternatives rather than a sum.

inline arma::mat calc_d_kern_diag_post_val_d_x_post_mat(
        const kernel::KernelBase& kern_fun,
        const PosteriorPrecompute& post_precompute,
        const arma::mat& Ydot_nfeature_npost_mat) {
    const double* hp_kern_extra_ptr = calc_hp_kern_extra_ptr(post_precompute.hp_kern_extra_vec);
    if (post_precompute.symmetry_spec.is_on())
        return calc_symmetry_d_kern_diag_val_d_x_post_mat(
                kern_fun,
                Ydot_nfeature_npost_mat,
                post_precompute.hp_gamma_vec,
                post_precompute.symmetry_spec,
                hp_kern_extra_ptr);

    return kern_fun.calc_d_kern_diag_val_d_x_post_batch(
            Ydot_nfeature_npost_mat, post_precompute.hp_gamma_vec, hp_kern_extra_ptr);
}

// The single-query counterpart, which reuses the batch form over a one-column matrix where symmetry is on: the
// mirrored diagonal needs the point's flipped-dimension norm, which the scalar interface does not carry.
inline arma::vec calc_d_kern_diag_post_val_d_x_post_vec(
        const kernel::KernelBase& kern_fun,
        const PosteriorPrecompute& post_precompute,
        const arma::vec& ydot_post_vec) {
    if (post_precompute.symmetry_spec.is_on())
        return arma::vec(calc_d_kern_diag_post_val_d_x_post_mat(kern_fun, post_precompute, arma::mat(ydot_post_vec)));

    return kern_fun.calc_d_kern_diag_val_d_x_post(
            ydot_post_vec, post_precompute.hp_gamma_vec, calc_hp_kern_extra_ptr(post_precompute.hp_kern_extra_vec));
}

// ── Single query ─────────────────────────────────────────────────────────────

// q^T H^(r), the product both gradients are built from, returned (n_dim). lin_sol_vec is v_res for the mean,
// making q_mu, or Sigma^-1 k_* for the variance, making q_sigma; the caller applies the power of sigma_k^2.
// kern_common_rdot_mirror_post_vec is read only where symmetry is on.
inline arma::vec calc_q_h_r_prod_vec(
        const kernel::KernelBase& kern_fun,
        const PosteriorPrecompute& post_precompute,
        const arma::vec& ydot_post_vec,
        const arma::vec& kern_common_rdot_post_vec,
        const arma::vec& kern_common_rdot_mirror_post_vec,
        const arma::vec& lin_sol_vec) {
    // q, formed once. The kernel sums it against its own h^(r,l), so H^(r) is never materialized.
    const arma::vec q_vec = lin_sol_vec % kern_common_rdot_post_vec;
    arma::vec q_h_r_prod_vec = kern_fun.calc_d_kern_sum_d_x_post(
            post_precompute.Ydot_ntrain_nfeature_mat, ydot_post_vec, post_precompute.hp_gamma_vec, q_vec);
    if (!post_precompute.symmetry_spec.is_on()) return q_h_r_prod_vec;

    const arma::vec& mirror_sign_vec = post_precompute.symmetry_spec.mirror_sign_vec;
    const arma::vec q_mirror_vec = lin_sol_vec % kern_common_rdot_mirror_post_vec;
    const arma::vec q_h_r_prod_mirror_vec = kern_fun.calc_d_kern_sum_d_x_post(
            post_precompute.Ydot_ntrain_nfeature_mat,
            build_mirrored_point_vec(ydot_post_vec, kern_fun.calc_feature_mirror_sign_vec(mirror_sign_vec)),
            post_precompute.hp_gamma_vec,
            q_mirror_vec);
    q_h_r_prod_vec += post_precompute.symmetry_spec.parity_sign() * (mirror_sign_vec % q_h_r_prod_mirror_vec);

    return q_h_r_prod_vec;
}

// d(mean)/d(x') at one query, (n_dim). q_mu's solve is v_res, taken straight off the precompute, so a mean
// gradient adds no solve of its own.
inline arma::vec calc_d_mean_post_d_x_post_vec(
        const kernel::KernelBase& kern_fun,
        const PosteriorPrecompute& post_precompute,
        const arma::vec& x_post_vec,
        const arma::vec& ydot_post_vec,
        const arma::vec& kern_common_rdot_post_vec,
        const arma::vec& kern_common_rdot_mirror_post_vec) {
    const arma::vec q_h_r_prod_vec = calc_q_h_r_prod_vec(
            kern_fun,
            post_precompute,
            ydot_post_vec,
            kern_common_rdot_post_vec,
            kern_common_rdot_mirror_post_vec,
            post_precompute.lin_sol_sigma_residual_vec);

    arma::vec d_mean_post_d_x_post_vec = post_precompute.hp_sig2_k * q_h_r_prod_vec;

    // The prior mean's own gradient.
    post_precompute.mean_model.add_d_prior_mean_d_x_post_inplace(
            x_post_vec, post_precompute.hp_beta_vec, d_mean_post_d_x_post_vec);

    return d_mean_post_d_x_post_vec;
}

// d(variance)/d(x') at one query, (n_dim). q_sigma's solve is Sigma^-1 k_*, which is what takes the leading
// factor to sigma_k^4 rather than sigma_k^2.
inline arma::vec calc_d_var_post_d_x_post_vec(
        const kernel::KernelBase& kern_fun,
        const PosteriorPrecompute& post_precompute,
        const arma::vec& ydot_post_vec,
        const arma::vec& kern_common_rdot_post_vec,
        const arma::vec& kern_common_rdot_mirror_post_vec,
        const arma::vec& lin_sol_sigma_k_post_vec) {
    const double hp_sig4_k = post_precompute.hp_sig2_k * post_precompute.hp_sig2_k;
    const arma::vec q_h_r_prod_vec = calc_q_h_r_prod_vec(
            kern_fun,
            post_precompute,
            ydot_post_vec,
            kern_common_rdot_post_vec,
            kern_common_rdot_mirror_post_vec,
            lin_sol_sigma_k_post_vec);
    arma::vec d_var_post_d_x_post_vec = (-2.0 * hp_sig4_k) * q_h_r_prod_vec;

    // The prior-variance term. An empty return is the kernel stating d(k(x', x'))/d(x') == 0.
    const arma::vec d_kern_diag_val_vec =
            calc_d_kern_diag_post_val_d_x_post_vec(kern_fun, post_precompute, ydot_post_vec);
    if (!d_kern_diag_val_vec.is_empty()) d_var_post_d_x_post_vec += post_precompute.hp_sig2_k * d_kern_diag_val_vec;

    return d_var_post_d_x_post_vec;
}

// ── Batch ────────────────────────────────────────────────────────────────────

// Batch counterpart, one column of q per query. The contraction becomes one DGEMM, S = Ydot_train Q, handed to the
// kernel with Q's column sums, since a radial h^(r,l) needs the weighted point sum and the plain weight sum both.
// model_scale is applied inside, so the result needs no second pass. q_ntrain_npost_mat is Q, formed by each
// caller from its own solve and dead afterwards.
inline arma::mat calc_q_h_r_prod_batch_mat(
        const kernel::KernelBase& kern_fun,
        const PosteriorPrecompute& post_precompute,
        const arma::mat& Ydot_nfeature_npost_mat,
        const arma::mat& q_ntrain_npost_mat,
        const double model_scale) {
    return kern_fun.calc_d_kern_sum_d_x_post_batch(
            Ydot_nfeature_npost_mat,
            post_precompute.Ydot_nfeature_ntrain_mat * q_ntrain_npost_mat,  // DGEMM, S = Ydot_train Q
            arma::sum(q_ntrain_npost_mat, 0),
            post_precompute.hp_gamma_vec,
            model_scale);
}

// The mirrored term's share of the same product, (n_dim, n_post). q_mirror_ntrain_npost_mat is the caller's own
// solve against the mirrored c^(rdot), formed exactly as its base Q was.
inline arma::mat calc_q_h_r_prod_mirror_batch_mat(
        const kernel::KernelBase& kern_fun,
        const PosteriorPrecompute& post_precompute,
        const arma::mat& Ydot_nfeature_npost_mat,
        const arma::mat& q_mirror_ntrain_npost_mat,
        const double model_scale) {
    const arma::vec& mirror_sign_vec = post_precompute.symmetry_spec.mirror_sign_vec;
    arma::mat q_h_r_prod_mirror_ndim_npost_mat = calc_q_h_r_prod_batch_mat(
            kern_fun,
            post_precompute,
            build_mirrored_points_mat(Ydot_nfeature_npost_mat, kern_fun.calc_feature_mirror_sign_vec(mirror_sign_vec)),
            q_mirror_ntrain_npost_mat,
            post_precompute.symmetry_spec.parity_sign() * model_scale);
    q_h_r_prod_mirror_ndim_npost_mat.each_col() %= mirror_sign_vec;

    return q_h_r_prod_mirror_ndim_npost_mat;
}

// d(mean)/d(x') for every query, (n_dim, n_post). Q_mu is c^(rdot) with every column scaled by the one residual
// solve, so copying c^(rdot) and scaling in place does it in one pass, with no broadcast of that single vector.
inline arma::mat calc_d_mean_post_d_x_post_batch_mat(
        const kernel::KernelBase& kern_fun,
        const PosteriorPrecompute& post_precompute,
        const arma::mat& X_ndim_npost_mat,
        const arma::mat& Ydot_nfeature_npost_mat,
        const arma::mat& kern_common_rdot_ntrain_npost_mat,
        const arma::mat& kern_common_rdot_mirror_ntrain_npost_mat) {
    arma::mat q_ntrain_npost_mat = kern_common_rdot_ntrain_npost_mat;
    q_ntrain_npost_mat.each_col() %= post_precompute.lin_sol_sigma_residual_vec;

    // The kernel term.
    arma::mat d_mean_post_d_x_post_ndim_npost_mat = calc_q_h_r_prod_batch_mat(
            kern_fun, post_precompute, Ydot_nfeature_npost_mat, q_ntrain_npost_mat, post_precompute.hp_sig2_k);

    // Its mirrored counterpart, where symmetry is on.
    if (post_precompute.symmetry_spec.is_on()) {
        arma::mat q_mirror_ntrain_npost_mat = kern_common_rdot_mirror_ntrain_npost_mat;
        q_mirror_ntrain_npost_mat.each_col() %= post_precompute.lin_sol_sigma_residual_vec;
        d_mean_post_d_x_post_ndim_npost_mat += calc_q_h_r_prod_mirror_batch_mat(
                kern_fun,
                post_precompute,
                Ydot_nfeature_npost_mat,
                q_mirror_ntrain_npost_mat,
                post_precompute.hp_sig2_k);
    }

    // The prior mean's own gradient.
    post_precompute.mean_model.add_d_prior_mean_d_x_post_inplace(
            X_ndim_npost_mat, post_precompute.hp_beta_vec, d_mean_post_d_x_post_ndim_npost_mat);

    return d_mean_post_d_x_post_ndim_npost_mat;
}

// d(variance)/d(x') for every query, (n_dim, n_post). lin_sol_sigma_K_ntrain_npost_mat holds Sigma^-1 k_* per column
// and is dead after this, so Q_sigma is formed in place on it.
inline arma::mat calc_d_var_post_d_x_post_batch_mat(
        const kernel::KernelBase& kern_fun,
        const PosteriorPrecompute& post_precompute,
        const arma::mat& Ydot_nfeature_npost_mat,
        const arma::mat& kern_common_rdot_ntrain_npost_mat,
        const arma::mat& kern_common_rdot_mirror_ntrain_npost_mat,
        arma::mat lin_sol_sigma_K_ntrain_npost_mat) {
    const double hp_sig4_k = post_precompute.hp_sig2_k * post_precompute.hp_sig2_k;
    const double var_model_scale = -2.0 * hp_sig4_k;

    // Both terms share the one solve, so the mirrored Q comes off it before the base Q overwrites it.
    arma::mat q_mirror_ntrain_npost_mat;
    if (post_precompute.symmetry_spec.is_on())
        q_mirror_ntrain_npost_mat = lin_sol_sigma_K_ntrain_npost_mat % kern_common_rdot_mirror_ntrain_npost_mat;

    lin_sol_sigma_K_ntrain_npost_mat %= kern_common_rdot_ntrain_npost_mat;  // now Q_sigma
    arma::mat d_var_post_d_x_post_ndim_npost_mat = calc_q_h_r_prod_batch_mat(
            kern_fun, post_precompute, Ydot_nfeature_npost_mat, lin_sol_sigma_K_ntrain_npost_mat, var_model_scale);
    if (post_precompute.symmetry_spec.is_on())
        d_var_post_d_x_post_ndim_npost_mat += calc_q_h_r_prod_mirror_batch_mat(
                kern_fun, post_precompute, Ydot_nfeature_npost_mat, q_mirror_ntrain_npost_mat, var_model_scale);

    // The prior-variance term, as in the single-query form above.
    const arma::mat d_kern_diag_val_ndim_npost_mat =
            calc_d_kern_diag_post_val_d_x_post_mat(kern_fun, post_precompute, Ydot_nfeature_npost_mat);
    if (!d_kern_diag_val_ndim_npost_mat.is_empty())
        d_var_post_d_x_post_ndim_npost_mat += post_precompute.hp_sig2_k * d_kern_diag_val_ndim_npost_mat;

    return d_var_post_d_x_post_ndim_npost_mat;
}

}  // namespace detail
}  // namespace gpflash
