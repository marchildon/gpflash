/**
 * The points and hyperparameters a hyperparameter-gradient evaluation reads, and the one function that
 * builds them.
 *
 * Built once per gradient evaluation, then read unmodified by every thread, so every member is owned by
 * value. One definition for every kernel; a field a given kernel does not use stays empty. Read on the
 * marginal-likelihood gradient path only, never by the posterior.
 *
 * Internal. It holds derived quantities in the layout the gradient loops want, not a supported API.
 */
#pragma once

#include <armadillo>

#include "gpflash/core/Errors.hpp"    // the field checks below compose
#include "gpflash/core/Platform.hpp"  // GPFLASH_CHECK

namespace gpflash {
namespace kernel {

struct HpGradInputs {
    arma::vec hp_gamma_vec;       // (n_dim) inverse length scales, or inverse periods for Periodic
    arma::vec hp_kern_extra_vec;  // the kernel's own extras, empty when it has none

    // The training points, one per column. Not owned: the caller's matrix, which must outlive this struct.
    const arma::mat* X_ndim_ntrain_mat_ptr = nullptr;
    arma::mat X_ntrain_ndim_mat;  // the same points transposed, one dimension per column

    // The features a kernel maps X onto, empty for a kernel evaluated from X directly. Periodic maps each
    // dimension onto a cosine and a sine axis, so n_feature = 2 n_dim for it.
    arma::mat Y_ntrain_nfeature_mat;
};

// Postcondition. The field checks live in gpflash::detail, so they are qualified.
inline void check_hp_grad_inputs(const HpGradInputs& hp_grad_inputs, const char* caller) {
    detail::check_mat_not_empty(hp_grad_inputs.X_ntrain_ndim_mat, "HpGradInputs::X_ntrain_ndim_mat", caller);
    detail::check_vec_len(
            hp_grad_inputs.hp_gamma_vec,
            hp_grad_inputs.X_ntrain_ndim_mat.n_cols,
            "HpGradInputs::hp_gamma_vec",
            caller,
            "One hp_gamma per column of X_ntrain_ndim_mat.");
    if (hp_grad_inputs.X_ndim_ntrain_mat_ptr == nullptr)
        throw ErrorInvalidInput(detail::error_prefix(caller, "HpGradInputs::X_ndim_ntrain_mat_ptr") + " is null.");
    detail::check_mat_n_cols(
            *hp_grad_inputs.X_ndim_ntrain_mat_ptr,
            hp_grad_inputs.X_ntrain_ndim_mat.n_rows,
            "HpGradInputs::X_ndim_ntrain_mat_ptr",
            caller,
            "One column per row of X_ntrain_ndim_mat.");

    // The two optional fields: a kernel with no extras, and one evaluated from X directly, leave them empty.
    detail::check_all_positive_vals(
            hp_grad_inputs.hp_kern_extra_vec,
            "HpGradInputs::hp_kern_extra_vec",
            caller,
            "The kernel's extras are natural-space hyperparameters.");
    detail::check_all_finite_vals(hp_grad_inputs.Y_ntrain_nfeature_mat, "HpGradInputs::Y_ntrain_nfeature_mat", caller);
}

// Holds X_train in both layouts, the caller's and a transpose, so a gradient loop can run over points or over
// dimensions at unit stride, and transposes Y_train so each feature becomes a unit-stride column. Y_train is carried
// only where the kernel mapped its points, which calc_feature_points signals by returning empty. Nothing here depends
// on the kernel.
inline HpGradInputs make_hp_grad_inputs(
        const arma::mat& X_ndim_ntrain_mat,
        const arma::mat& Y_nfeature_ntrain_mat,
        const arma::vec& hp_gamma_vec,
        const arma::vec& hp_kern_extra_vec) {
    HpGradInputs hp_grad_inputs;
    hp_grad_inputs.hp_gamma_vec = hp_gamma_vec;
    hp_grad_inputs.hp_kern_extra_vec = hp_kern_extra_vec;

    hp_grad_inputs.X_ndim_ntrain_mat_ptr = &X_ndim_ntrain_mat;
    hp_grad_inputs.X_ntrain_ndim_mat = X_ndim_ntrain_mat.t();

    // "X was not mapped" has two signals: an empty Y matrix, and the Y-aliases-X identity below. Prefer
    // emptiness -- a kernel returning a copy of X would slip past the identity test.
    if (Y_nfeature_ntrain_mat.n_elem != 0 && Y_nfeature_ntrain_mat.memptr() != X_ndim_ntrain_mat.memptr())
        hp_grad_inputs.Y_ntrain_nfeature_mat = Y_nfeature_ntrain_mat.t();

    GPFLASH_CHECK(check_hp_grad_inputs(hp_grad_inputs, "make_hp_grad_inputs"));
    return hp_grad_inputs;
}

// A temporary X would leave X_ndim_ntrain_mat_ptr dangling, so it does not compile.
HpGradInputs make_hp_grad_inputs(
        const arma::mat&& X_ndim_ntrain_mat,
        const arma::mat& Y_nfeature_ntrain_mat,
        const arma::vec& hp_gamma_vec,
        const arma::vec& hp_kern_extra_vec) = delete;

}  // namespace kernel

}  // namespace gpflash
