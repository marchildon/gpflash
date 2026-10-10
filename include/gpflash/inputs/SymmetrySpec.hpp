/**
 * SymmetrySpec.hpp -- the mirror a symmetry kernel is built around.
 *
 * k_sym(x, y) = k(x, y) + parity_sign k(x, S y), for a stationary base kernel k and an axis-aligned mirror S.
 * S is its own inverse and preserves norms, so the mirrored term is symmetric and assembly keeps it to one
 * triangle. The parity selects which function the model carries: even about the mirror plane, or odd. The mean
 * model basis must have the same parity. The constant is even, so an odd model needs
 * ModelOptions::has_mean_model_constant = false; check_mean_model_fits_symmetry enforces both.
 *
 * The paper writes parity_sign as lambda.
 */
#pragma once

#include <armadillo>
#include <cmath>   // std::sqrt
#include <string>  // std::to_string

#include "gpflash/core/Errors.hpp"  // the field checks check_symmetry_spec composes

namespace gpflash {

struct SymmetrySpec {
    arma::vec mirror_sign_vec;     // (n_dim) diagonal of S, each +1 or -1; empty means no symmetry
    bool is_even_symmetry = true;  // read only where the mirror is set

    bool is_on() const { return !mirror_sign_vec.is_empty(); }

    // The weight the mirrored term enters K with.
    double parity_sign() const { return is_even_symmetry ? 1.0 : -1.0; }

    bool is_odd_symmetry() const { return is_on() && !is_even_symmetry; }
};

namespace detail {

// Length against n_dim needs the training data, so check_symmetry_applies checks it.
inline void check_symmetry_spec(const SymmetrySpec& symmetry_spec, const char* caller) {
    if (!symmetry_spec.is_on()) return;

    bool has_flipped_dim = false;
    for (arma::uword i_dim = 0; i_dim < symmetry_spec.mirror_sign_vec.n_elem; ++i_dim) {
        const double mirror_sign = symmetry_spec.mirror_sign_vec(i_dim);
        if (mirror_sign != 1.0 && mirror_sign != -1.0)
            throw ErrorInvalidInput(
                    error_prefix(caller, "SymmetrySpec::mirror_sign_vec")
                    + " entries must be +1 or -1, the diagonal of an axis-aligned mirror.");
        if (mirror_sign == -1.0) has_flipped_dim = true;
    }

    // Every sign +1 makes S the identity, so k_sym collapses to a rescaled base kernel for an even parity and to
    // zero for an odd one.
    if (!has_flipped_dim)
        throw ErrorInvalidInput(
                error_prefix(caller, "SymmetrySpec::mirror_sign_vec") + " flips no dimension, so it is not a mirror.");
}

// Rejects an odd model's training point on the mirror plane, where the model is zero by construction: k_sym(x, x) = 0
// there and the point's row of K vanishes.
//
// Near the plane counts as on it. Relative to each flipped dimension's largest |x|, sqrt(eps) is where, at a length
// scale of that size, k_sym(x, x) ~ (distance / scale)^2 drops below eps and rounds to zero.
inline void check_training_points_off_mirror_plane(
        const SymmetrySpec& symmetry_spec, const arma::mat& X_ndim_ntrain_mat, const char* caller) {
    if (symmetry_spec.is_odd_symmetry()) {
        const arma::uvec idx_flipped_vec = arma::find(symmetry_spec.mirror_sign_vec < 0.0);
        const arma::mat X_flipped_abs_mat = arma::abs(X_ndim_ntrain_mat.rows(idx_flipped_vec));
        const arma::vec x_flipped_abs_max_vec = arma::max(X_flipped_abs_mat, 1);
        const double mirror_plane_rel_tol = std::sqrt(arma::datum::eps);
        const arma::vec x_flipped_tol_vec = mirror_plane_rel_tol * x_flipped_abs_max_vec;

        for (arma::uword i_train = 0; i_train < X_flipped_abs_mat.n_cols; ++i_train) {
            const bool is_on_mirror_plane = arma::all(X_flipped_abs_mat.col(i_train) <= x_flipped_tol_vec);
            if (is_on_mirror_plane)
                throw ErrorInvalidInput(
                        error_prefix(caller, "TrainingData::X_ndim_ntrain_mat") + " has training point "
                        + std::to_string(i_train)
                        + " on the mirror plane of an odd symmetry model, where the model is zero by construction. "
                          "Drop the point.");
        }
    }
}

// What the spec cannot know: the dimension it must match, whether this base kernel can carry a mirror, and whether
// the training points sit where the model is fixed. The kernel's own facts arrive as values so the input tier names
// no kernel.
inline void check_symmetry_applies(
        const SymmetrySpec& symmetry_spec,
        const arma::mat& X_ndim_ntrain_mat,
        const bool is_symmetrizable,
        const char* kernel_name,
        const char* caller) {
    if (!symmetry_spec.is_on()) return;

    check_vec_len(
            symmetry_spec.mirror_sign_vec,
            X_ndim_ntrain_mat.n_rows,
            "SymmetrySpec::mirror_sign_vec",
            caller,
            "One sign per input dimension, matching hp_gamma.");

    if (!is_symmetrizable)
        throw ErrorInvalidInput(
                error_prefix(caller, "ModelOptions::symmetry_spec") + " cannot wrap " + kernel_name
                + ": a symmetry model needs a stationary base kernel.");

    check_training_points_off_mirror_plane(symmetry_spec, X_ndim_ntrain_mat, caller);
}

}  // namespace detail
}  // namespace gpflash
