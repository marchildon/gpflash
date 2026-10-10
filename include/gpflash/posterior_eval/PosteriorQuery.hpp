/**
 * PosteriorQuery.hpp -- the two query drivers, one per operand shape.
 *
 * x_post_vec (length n_dim) for a single query, X_ndim_npost_mat (n_dim, n_post) for a batch. Both run the same
 * four stages, numbered in each body so the two read side by side:
 *
 *   1) cross-covariance   k_* / K_*  against the training points
 *   2) mean               the prior mean, plus one dot or one DGEMV
 *   3) variance           one forward triangular solve, plus a back solve only where its gradient is wanted
 *   4) gradients          one call per quantity
 *
 * They differ only in BLAS level, and are separate implementations rather than one forwarding to the other:
 * routing a single query through the batch path's dgemm/dtrsm is slower than its own level-2 path. The resulting
 * _vec/_mat pair in each stage is intentional and must not be merged.
 *
 * post_opts selects the fields to fill; the work behind an unselected one is skipped, not computed and dropped.
 */
#pragma once

#include <armadillo>
#include <utility>  // std::move

#include "gpflash/assembly/PosteriorCovariance.hpp"         // build_post_vec, build_post_mat
#include "gpflash/assembly/SymmetryMatrix.hpp"              // build_mirrored_point_vec, build_mirrored_points_mat
#include "gpflash/core/Errors.hpp"                          // ErrorInvalidInput
#include "gpflash/core/Platform.hpp"                        // GPFLASH_CHECK
#include "gpflash/inputs/HpVal.hpp"                         // calc_hp_kern_extra_ptr
#include "gpflash/kernels/KernelBase.hpp"                   // kernel::KernelBase
#include "gpflash/linear_algebra/Cholesky.hpp"              // the solve_chol_* triangular solves
#include "gpflash/posterior_cache/PosteriorPrecompute.hpp"  // PosteriorPrecompute
#include "gpflash/posterior_eval/PosteriorEval.hpp"         // the mean and variance, single and batch
#include "gpflash/posterior_eval/PosteriorGrad.hpp"         // the input-space gradients
#include "gpflash/posterior_eval/PosteriorOptions.hpp"      // PosteriorOptions, is_empty_request
#include "gpflash/posterior_eval/PosteriorResult.hpp"       // PosteriorResult, check_posterior_result

namespace gpflash {
namespace detail {

// ── Validation ───────────────────────────────────────────────────────────────

// The query points, before any of them reaches the kernel. A point of another dimension would read past the training
// features, and a non-finite one has no distance to them, so a stationary kernel would return its far-field answer.
inline void check_post_points(
        const arma::mat& X_ndim_npost_mat, const PosteriorPrecompute& post_precompute, const char* field_name) {
    const char* caller = "GpPosterior::eval";
    check_mat_n_rows(
            X_ndim_npost_mat,
            static_cast<arma::uword>(post_precompute.n_dim),
            field_name,
            caller,
            "One row per input dimension the posterior was built with.");
    check_all_finite_vals(X_ndim_npost_mat, field_name, caller);
}

// What the options alone cannot say: whether this precompute can serve them.
inline void check_posterior_request(const PosteriorOptions& post_opts, const PosteriorPrecompute& post_precompute) {
    if ((post_opts.return_var || post_opts.return_var_gradient) && !post_precompute.has_variance_data())
        throw ErrorInvalidInput(
                "gpflash posterior eval: this post_precompute came from a mean-only cache file, which does not store "
                "the Cholesky factor the variance needs. Clear PosteriorOptions::return_var and "
                "return_var_gradient to query the mean and its gradient, or save the file again with "
                "save_precompute's include_variance_data set.");
}

// ── The two query entry points ───────────────────────────────────────────────

inline PosteriorResult eval_precompute(
        const kernel::KernelBase& kern_fun,
        const PosteriorPrecompute& post_precompute,
        const arma::vec& x_post_vec,
        const PosteriorOptions& post_opts) {
    // Skipped only when the caller vouches for the points; see PosteriorOptions::check_query_points.
    if (post_opts.check_query_points) check_post_points(x_post_vec, post_precompute, "x_post_vec");
    // Nothing asked for, nothing to build.
    if (is_empty_request(post_opts)) return PosteriorResult{};
    check_posterior_request(post_opts, post_precompute);

    const bool needs_gradient = post_opts.return_mean_gradient || post_opts.return_var_gradient;
    const bool needs_solve = post_opts.return_var || post_opts.return_var_gradient;
    const double* hp_kern_extra_ptr = calc_hp_kern_extra_ptr(post_precompute.hp_kern_extra_vec);

    // 1. cross-covariance k_* between the training points and this query, measured from the same center.
    const arma::vec ydot_post_vec = kern_fun.calc_scaled_feature_point_vec(
            x_post_vec, post_precompute.x_center_vec, post_precompute.hp_gamma_vec, post_precompute.hp_kern_extra_vec);
    // Formed once: the cross-covariance reads it, and stage 3 reads it again as the prior variance.
    const double ydot_post_sq_norm = arma::dot(ydot_post_vec, ydot_post_vec);
    arma::vec kern_common_rdot_post_vec;
    // Null skips c^(rdot), which only a gradient reads.
    arma::vec* kern_common_rdot_post_vec_out = needs_gradient ? &kern_common_rdot_post_vec : nullptr;
    arma::vec k_post_vec = build_post_vec(
            kern_fun,
            post_precompute.Ydot_nfeature_ntrain_mat,
            post_precompute.ydot_train_sq_norm_vec,
            ydot_post_vec,
            ydot_post_sq_norm,
            hp_kern_extra_ptr,
            kern_common_rdot_post_vec_out);
    // The mirrored term, against the query point S sends this one to. S preserves the norm, so the same norms
    // complete its squared distance.
    arma::vec kern_common_rdot_mirror_post_vec;
    if (post_precompute.symmetry_spec.is_on()) {
        const arma::vec k_mirror_post_vec = build_post_vec(
                kern_fun,
                post_precompute.Ydot_nfeature_ntrain_mat,
                post_precompute.ydot_train_sq_norm_vec,
                build_mirrored_point_vec(
                        ydot_post_vec,
                        kern_fun.calc_feature_mirror_sign_vec(post_precompute.symmetry_spec.mirror_sign_vec)),
                ydot_post_sq_norm,
                hp_kern_extra_ptr,
                needs_gradient ? &kern_common_rdot_mirror_post_vec : nullptr);
        k_post_vec += post_precompute.symmetry_spec.parity_sign() * k_mirror_post_vec;
    }

    // 2. mean: the prior mean at this query, plus one dot against the fit-time residual solve.
    PosteriorResult post_result;
    if (post_opts.return_mean)
        post_result.mean_post_vec = arma::vec{calc_mean_post_val(post_precompute, x_post_vec, k_post_vec)};

    // 3. variance: one forward triangular solve, plus a back solve when its gradient is wanted, since that
    //    needs Sigma^-1 k_* rather than L_Sigma^-1 k_*.
    arma::vec lin_sol_sigma_k_post_vec;
    if (needs_solve) {
        // k_post_vec is dead once the mean has read it, so it becomes the solve's buffer.
        arma::vec lin_sol_L_k_post_vec = std::move(k_post_vec);
        solve_chol_forward_vec_inplace(post_precompute.L_Sigma_mat_lower, lin_sol_L_k_post_vec);
        if (post_opts.return_var) {
            const double kern_diag_val = calc_kern_diag_post_val(
                    kern_fun, post_precompute, ydot_post_vec, ydot_post_sq_norm, hp_kern_extra_ptr);
            post_result.var_post_vec =
                    arma::vec{calc_var_post_val(post_precompute, kern_diag_val, lin_sol_L_k_post_vec)};
        }
        if (post_opts.return_var_gradient) {
            // Dead after the variance dot, so the back solve runs in place: no second buffer.
            solve_chol_transpose_vec_inplace(post_precompute.L_Sigma_mat_lower, lin_sol_L_k_post_vec);
            lin_sol_sigma_k_post_vec = std::move(lin_sol_L_k_post_vec);
        }
    }

    // 4. gradients, one quantity per call. Each comes back (n_dim), the n_post == 1 case of the field.
    if (post_opts.return_mean_gradient)
        post_result.d_mean_post_d_x_post_ndim_npost_mat = calc_d_mean_post_d_x_post_vec(
                kern_fun,
                post_precompute,
                x_post_vec,
                ydot_post_vec,
                kern_common_rdot_post_vec,
                kern_common_rdot_mirror_post_vec);
    if (post_opts.return_var_gradient)
        post_result.d_var_post_d_x_post_ndim_npost_mat = calc_d_var_post_d_x_post_vec(
                kern_fun,
                post_precompute,
                ydot_post_vec,
                kern_common_rdot_post_vec,
                kern_common_rdot_mirror_post_vec,
                lin_sol_sigma_k_post_vec);

    GPFLASH_CHECK(check_posterior_result(post_result, "eval_precompute"));
    return post_result;
}

// Batch counterpart, on DGEMM and DTRSM.
inline PosteriorResult eval_precompute(
        const kernel::KernelBase& kern_fun,
        const PosteriorPrecompute& post_precompute,
        const arma::mat& X_ndim_npost_mat,
        const PosteriorOptions& post_opts) {
    if (post_opts.check_query_points) check_post_points(X_ndim_npost_mat, post_precompute, "X_ndim_npost_mat");
    // Nothing asked for, nothing to build.
    if (is_empty_request(post_opts)) return PosteriorResult{};
    check_posterior_request(post_opts, post_precompute);

    const bool needs_gradient = post_opts.return_mean_gradient || post_opts.return_var_gradient;
    const bool needs_solve = post_opts.return_var || post_opts.return_var_gradient;
    const double* hp_kern_extra_ptr = calc_hp_kern_extra_ptr(post_precompute.hp_kern_extra_vec);

    // 1. cross-covariance K_* for every query at once, measured from the same center as the training points.
    // The unscaled features are not requested: only a hyperparameter gradient reads those, and a query never
    // differentiates by one.
    const arma::mat Ydot_nfeature_npost_mat = kern_fun.calc_scaled_feature_points(
            X_ndim_npost_mat,
            post_precompute.x_center_vec,
            post_precompute.hp_gamma_vec,
            post_precompute.hp_kern_extra_vec);
    // Formed once: the cross-covariance reads these, and stage 3 reads them again as the prior variance.
    const arma::vec ydot_post_sq_norm_vec = arma::sum(arma::square(Ydot_nfeature_npost_mat), 0).t();
    arma::mat kern_common_rdot_ntrain_npost_mat;
    // Null skips c^(rdot), which only a gradient reads.
    arma::mat* kern_common_rdot_ntrain_npost_mat_out = needs_gradient ? &kern_common_rdot_ntrain_npost_mat : nullptr;
    arma::mat K_ntrain_npost_mat = build_post_mat(
            kern_fun,
            post_precompute.Ydot_nfeature_ntrain_mat,
            post_precompute.ydot_train_sq_norm_vec,
            Ydot_nfeature_npost_mat,
            ydot_post_sq_norm_vec,
            hp_kern_extra_ptr,
            kern_common_rdot_ntrain_npost_mat_out);
    // The mirrored term, against the query points S sends these to; the norms are unchanged by S.
    arma::mat kern_common_rdot_mirror_ntrain_npost_mat;
    if (post_precompute.symmetry_spec.is_on()) {
        const arma::mat K_mirror_ntrain_npost_mat = build_post_mat(
                kern_fun,
                post_precompute.Ydot_nfeature_ntrain_mat,
                post_precompute.ydot_train_sq_norm_vec,
                build_mirrored_points_mat(
                        Ydot_nfeature_npost_mat,
                        kern_fun.calc_feature_mirror_sign_vec(post_precompute.symmetry_spec.mirror_sign_vec)),
                ydot_post_sq_norm_vec,
                hp_kern_extra_ptr,
                needs_gradient ? &kern_common_rdot_mirror_ntrain_npost_mat : nullptr);
        K_ntrain_npost_mat += post_precompute.symmetry_spec.parity_sign() * K_mirror_ntrain_npost_mat;
    }

    // 2. means: the prior mean at every query, plus one DGEMV.
    PosteriorResult post_result;
    if (post_opts.return_mean)
        post_result.mean_post_vec = calc_mean_post_val_vec(post_precompute, X_ndim_npost_mat, K_ntrain_npost_mat);

    // 3. variances, one triangular solve over all the columns.
    arma::mat lin_sol_sigma_K_ntrain_npost_mat;
    if (needs_solve) {
        // K_ntrain_npost_mat is dead once the means have read it, so it becomes the solve's buffer.
        arma::mat lin_sol_L_K_ntrain_npost_mat = std::move(K_ntrain_npost_mat);
        solve_chol_forward_mat_inplace(post_precompute.L_Sigma_mat_lower, lin_sol_L_K_ntrain_npost_mat);
        if (post_opts.return_var) {
            const arma::vec kern_diag_val_vec = calc_kern_diag_post_val_vec(
                    kern_fun, post_precompute, Ydot_nfeature_npost_mat, ydot_post_sq_norm_vec, hp_kern_extra_ptr);
            post_result.var_post_vec =
                    calc_var_post_val_vec(post_precompute, kern_diag_val_vec, lin_sol_L_K_ntrain_npost_mat);
        }
        if (post_opts.return_var_gradient) {
            // In place: the forward half is dead once the variance above has read it.
            solve_chol_transpose_mat_inplace(post_precompute.L_Sigma_mat_lower, lin_sol_L_K_ntrain_npost_mat);
            lin_sol_sigma_K_ntrain_npost_mat = std::move(lin_sol_L_K_ntrain_npost_mat);
        }
    }

    // 4. gradients, one quantity per call.
    if (post_opts.return_mean_gradient)
        post_result.d_mean_post_d_x_post_ndim_npost_mat = calc_d_mean_post_d_x_post_batch_mat(
                kern_fun,
                post_precompute,
                X_ndim_npost_mat,
                Ydot_nfeature_npost_mat,
                kern_common_rdot_ntrain_npost_mat,
                kern_common_rdot_mirror_ntrain_npost_mat);
    if (post_opts.return_var_gradient)
        post_result.d_var_post_d_x_post_ndim_npost_mat = calc_d_var_post_d_x_post_batch_mat(
                kern_fun,
                post_precompute,
                Ydot_nfeature_npost_mat,
                kern_common_rdot_ntrain_npost_mat,
                kern_common_rdot_mirror_ntrain_npost_mat,
                std::move(lin_sol_sigma_K_ntrain_npost_mat));

    GPFLASH_CHECK(check_posterior_result(post_result, "eval_precompute"));
    return post_result;
}

}  // namespace detail
}  // namespace gpflash
