/**
 * The fitted model a posterior query reads, built by build_posterior_precompute or read back from a cache file.
 *
 * Internal, not supported API: the fit's derived quantities and the mean model they were fit with. The cache format
 * versions this layout.
 */
#pragma once

#include <armadillo>
#include <string>

#include "gpflash/core/Errors.hpp"          // ErrorInvalidInput, for check_posterior_precompute below
#include "gpflash/inputs/SymmetrySpec.hpp"  // SymmetrySpec
#include "gpflash/kernels/KernelBase.hpp"   // kernel::KernelBase, whose feature map the checker reads
#include "gpflash/mean/MeanModel.hpp"       // mean::MeanModel

namespace gpflash {
namespace detail {

struct PosteriorPrecompute {
    // Shapes every field below is dimensioned by.
    int n_dim = 0;      // the query point's dimension
    int n_feature = 0;  // rows the kernel evaluates on, which a feature map raises above n_dim
    int n_train = 0;

    // The factorization a query solves against.
    arma::mat L_Sigma_mat_lower;           // (n_train, n_train) Cholesky factor of Sigma
    arma::vec lin_sol_sigma_residual_vec;  // (n_train) Sigma^-1 (f_train - Basis hp_beta_vec)

    // The points the kernel evaluates on, not the caller's inputs: a feature-map kernel stores its mapped
    // features here, so the leading dimension is n_feature, not n_dim. The two coincide for a kernel that
    // evaluates on raw points.
    arma::mat Ydot_nfeature_ntrain_mat;  // the gamma-scaled points the kernel evaluates on
    arma::mat Ydot_ntrain_nfeature_mat;  // its transpose; the gradient sum reads one feature at unit stride
    arma::vec ydot_train_sq_norm_vec;    // (n_train) squared column norms of Ydot, see KernelMatrix.hpp

    // The mirror the fit was built around, so a query mirrors its own point the same way.
    SymmetrySpec symmetry_spec;

    // The center and hyperparameters the fit used, so a query centers, maps and scales its own point the same way.
    arma::vec x_center_vec;       // (n_dim) subtracted from every point the kernel sees
    arma::vec hp_gamma_vec;       // (n_dim)
    arma::vec hp_kern_extra_vec;  // (n_hp_kern_extra)
    double hp_sig2_k = 1.0;       // signal variance, scales the query's kernel values in the mean and the variance

    // The mean model, whose basis a query evaluates at its own point, and the mean model weights the fit solved.
    mean::MeanModel mean_model;
    arma::vec hp_beta_vec;  // (n_basis)

    // Whether the variance is available: it needs L_Sigma, the mean and its gradient do not.
    // False for a precompute loaded from a mean-only cache file.
    bool has_variance_data() const { return !L_Sigma_mat_lower.is_empty(); }
};

// Catches a file saved for a different kernel whose id happens to match: the hyperparameter widths are the
// kernel's, so they disagree even where the id does not.
inline void check_posterior_precompute(
        const kernel::KernelBase& kern_fun, const PosteriorPrecompute& post_precompute, const char* caller) {
    const int n_dim = post_precompute.n_dim;
    const char* kernel_hint = "The file was probably saved for a different kernel.";

    check_positive_count(n_dim, "PosteriorPrecompute::n_dim", caller);
    check_vec_len(post_precompute.x_center_vec, n_dim, "PosteriorPrecompute::x_center_vec", caller);
    check_all_finite_vals(post_precompute.x_center_vec, "PosteriorPrecompute::x_center_vec", caller);
    check_vec_len(post_precompute.hp_gamma_vec, n_dim, "PosteriorPrecompute::hp_gamma_vec", caller, kernel_hint);
    check_vec_len(
            post_precompute.hp_kern_extra_vec,
            kern_fun.n_hp_kern_extra(n_dim),
            "PosteriorPrecompute::hp_kern_extra_vec",
            caller,
            kernel_hint);

    // The mean model sizes the weights, so it is checked first.
    mean::check_mean_model(post_precompute.mean_model, n_dim, caller);
    check_vec_len(
            post_precompute.hp_beta_vec,
            post_precompute.mean_model.n_basis(),
            "PosteriorPrecompute::hp_beta_vec",
            caller);
    check_all_finite_vals(post_precompute.hp_beta_vec, "PosteriorPrecompute::hp_beta_vec", caller);

    if (kern_fun.calc_n_dim_input(post_precompute.n_feature) != n_dim)
        throw ErrorInvalidInput(
                error_prefix(caller, "PosteriorPrecompute::n_feature") + " of "
                + std::to_string(post_precompute.n_feature) + " does not map back to n_dim of " + std::to_string(n_dim)
                + " for " + kern_fun.kernel_name() + ".");

    check_symmetry_spec(post_precompute.symmetry_spec, caller);
    if (post_precompute.symmetry_spec.is_on()) {
        check_vec_len(
                post_precompute.symmetry_spec.mirror_sign_vec,
                n_dim,
                "PosteriorPrecompute::symmetry_spec.mirror_sign_vec",
                caller,
                kernel_hint);
        // A fit cannot produce either of these, so only a hand-edited file holds one.
        if (!kern_fun.is_symmetrizable())
            throw ErrorInvalidInput(
                    error_prefix(caller, "PosteriorPrecompute::symmetry_spec") + " sets a mirror on "
                    + kern_fun.kernel_name() + ", which does not admit one.");
        mean::check_mean_model_fits_symmetry(post_precompute.mean_model, post_precompute.symmetry_spec, caller);
    }
}

// The one place Ydot_ntrain_nfeature_mat is derived, since no file stores it. Every construction path must call
// this; skipping it leaves the single-point gradient reading an empty matrix.
inline void fill_ydot_transpose(PosteriorPrecompute& post_precompute) {
    post_precompute.Ydot_ntrain_nfeature_mat = post_precompute.Ydot_nfeature_ntrain_mat.t();
}

}  // namespace detail
}  // namespace gpflash
