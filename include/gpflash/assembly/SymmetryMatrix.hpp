/**
 * SymmetryMatrix.hpp -- the symmetry kernel's K, from one base kernel and a mirror.
 *
 *   k_sym(x, y) = k(x, y) + parity_sign k(x, S y)
 *
 * S negates a subset of the dimensions, so it changes only the sign that subset contributes to the Gram cross
 * term. Splitting the feature rows into the dimensions S preserves and the dimensions it flips therefore yields
 * both cross terms from one Gram's work:
 *
 *   base cross      = preserved + flipped
 *   mirrored cross = preserved - flipped
 *
 * Only the preserved half is a matrix, built by one DSYRK and overwritten with K_sym. The flipped half is formed
 * one column at a time inside the sweep, in the memory the mirrored factor is about to occupy, so the build
 * holds one n_train by n_train buffer and the packed half below.
 *
 * S preserves norms, so both squared distances read the same per-point norms, and the mirrored term is
 * symmetric in its two points, which keeps it to one triangle like the base term.
 *
 * The base kernel is evaluated twice per pair; beyond that it supplies only the feature-space mirror and the
 * mirrored gradients KernelBase declares.
 */
#pragma once

#include <armadillo>
#include <cstddef>
#include <string>  // std::to_string

#include "gpflash/core/Errors.hpp"                   // ErrorNumericalFailure
#include "gpflash/core/Platform.hpp"                 // GPFLASH_RESTRICT, OMP constants
#include "gpflash/inputs/SymmetrySpec.hpp"           // SymmetrySpec
#include "gpflash/kernels/HpGradInputs.hpp"          // kernel::HpGradInputs
#include "gpflash/kernels/KernelBase.hpp"            // kernel::KernelBase
#include "gpflash/linear_algebra/InnerProducts.hpp"  // fill_inner_prod_self_lower_tri, clamp_sq_dist

namespace gpflash {
namespace detail {

// ── The packed mirrored derivative factor ──────────────────────────────────
// C^(rdot) for the mirrored term, packed column-major over the lower triangle with the diagonal: column i_col
// holds rows [i_col, n_train) contiguously. The base term's factor goes in the buffer's strict upper triangle,
// which the mirrored term cannot share, and a second n_train by n_train matrix would store a triangle's worth
// of nothing. The diagonal entries are live: the symmetry diagonal varies, so its gradient needs them.

inline std::ptrdiff_t calc_packed_lower_n_elem(const int n_train) {
    return static_cast<std::ptrdiff_t>(n_train) * (n_train + 1) / 2;
}

// Where column i_col starts, its own diagonal entry first.
inline std::ptrdiff_t calc_packed_lower_offset(const int i_col, const int n_train) {
    return static_cast<std::ptrdiff_t>(i_col) * n_train - static_cast<std::ptrdiff_t>(i_col) * (i_col - 1) / 2;
}

// ── The mirrored cross term, a column at a time ────────────────────────────

// One column of the mirrored Gram half: entries [i_col + 1, n_train) of column i_col, over the flipped
// dimensions alone. Written where the mirrored factor will overwrite it, so the half never exists as a matrix.
//
// An explicit loop, not a BLAS call: the contracted dimension is the number of flipped dimensions, usually one
// or two, so there is nothing to block and the work is bound by memory traffic, and a BLAS call here would sit
// inside the threaded column sweep, which a pthread OpenBLAS handles badly.
inline void fill_gram_flipped_strict_column(
        const arma::mat& Ydot_flipped_nflip_ntrain_mat,
        const int i_col,
        const int n_train,
        double* GPFLASH_RESTRICT gram_flipped_col_ptr_out) {
    const int n_flip = static_cast<int>(Ydot_flipped_nflip_ntrain_mat.n_rows);
    const double* GPFLASH_RESTRICT ydot_col_ptr = Ydot_flipped_nflip_ntrain_mat.colptr(i_col);

    for (int i_row = i_col + 1; i_row < n_train; ++i_row) {
        const double* GPFLASH_RESTRICT ydot_row_ptr = Ydot_flipped_nflip_ntrain_mat.colptr(i_row);
        double inner_prod = 0.0;
        GPFLASH_OMP_SIMD_REDUCTION(+ : inner_prod)
        for (int i_flip = 0; i_flip < n_flip; ++i_flip) inner_prod += ydot_row_ptr[i_flip] * ydot_col_ptr[i_flip];
        gram_flipped_col_ptr_out[i_row - i_col - 1] = inner_prod;
    }
}

// ── The preserved Gram half to K_sym ────────────────────────────────────────

/**
 * Turn the preserved Gram half into K_sym, one column at a time, forming the mirrored cross term as it goes.
 *
 * The buffer arrives holding the preserved half and leaves holding K_sym in its lower triangle with its own
 * diagonal, and the base term's c^(rdot) in the strict upper where the single-term build also leaves it.
 *
 * Where the gradient factors are wanted, each column's mirrored cross term is written into that column's slice
 * of the packed vector and the mirrored factor is written over it: the entry is read to form the distance and
 * dead immediately after, so one slice serves both and no mirrored Gram matrix is allocated. Where they are not
 * wanted, a per-thread scratch column stands in and the packed vector is never sized at all.
 *
 * Gram_preserved_in_kmat_out_mat lower with diag memory: IN: the preserved Gram half, OUT: K_sym
 * Gram_preserved_in_kmat_out_mat strict upper memory:    OUT: the base term's c^(rdot), where asked for
 */
inline void calc_gram_preserved_to_kmat_inplace(
        const kernel::KernelBase& kern_fun,
        arma::mat& Gram_preserved_in_kmat_out_mat,
        const arma::mat& Ydot_flipped_nflip_ntrain_mat,
        const arma::vec& ydot_train_sq_norm_vec,
        const arma::vec& ydot_flipped_sq_norm_vec,
        const double parity_sign,
        const double* hp_kern_extra_ptr,
        arma::vec* kern_common_rdot_mirror_packed_vec_out) {
    const int n_train = static_cast<int>(Gram_preserved_in_kmat_out_mat.n_cols);
    const double* GPFLASH_RESTRICT sq_norm_ptr = ydot_train_sq_norm_vec.memptr();
    const bool store_hp_grad_factors = kern_common_rdot_mirror_packed_vec_out != nullptr;

    double* kern_common_rdot_mirror_packed_ptr = nullptr;
    if (store_hp_grad_factors) {
        kern_common_rdot_mirror_packed_vec_out->set_size(calc_packed_lower_n_elem(n_train));
        kern_common_rdot_mirror_packed_ptr = kern_common_rdot_mirror_packed_vec_out->memptr();
    }

    // S x_i differs from x_i in the flipped dimensions alone, so the mirrored diagonal's squared distance is
    // four times their squared norm. One call over the whole diagonal, outside the column loop, which writes
    // off-diagonal entries only.
    arma::vec kern_mirror_diag_vec = 4.0 * ydot_flipped_sq_norm_vec;
    arma::vec kern_common_rdot_mirror_diag_vec;
    if (store_hp_grad_factors) {
        kern_common_rdot_mirror_diag_vec.set_size(n_train);
        kern_fun.calc_kern_val_and_common_rdot_inplace(
                kern_mirror_diag_vec.memptr(), n_train, hp_kern_extra_ptr, kern_common_rdot_mirror_diag_vec.memptr());
    } else {
        kern_fun.calc_kern_val_inplace(kern_mirror_diag_vec.memptr(), n_train, hp_kern_extra_ptr);
    }

    // Two columns per thread: the mirrored run, and the mirrored cross term where no packed slice holds it.
    // Allocated before the region opens.
    const int n_thread = (n_train >= PERF_OMP_MIN_N_POINT) ? GPFLASH_OMP_MAX_THREADS() : 1;
    arma::vec kern_input_mirror_thread_vec(static_cast<arma::uword>(n_thread) * n_train, arma::fill::none);
    arma::vec gram_flipped_thread_vec(
            store_hp_grad_factors ? 0 : static_cast<arma::uword>(n_thread) * n_train, arma::fill::none);

#pragma omp parallel for schedule(dynamic, gpflash::PERF_OMP_COL_CHUNK) if (n_train >= gpflash::PERF_OMP_MIN_N_POINT) \
        num_threads(n_thread)
    for (int i_col = 0; i_col < n_train; ++i_col) {
        const std::ptrdiff_t idx_thread_slice = static_cast<std::ptrdiff_t>(GPFLASH_OMP_THREAD_NUM()) * n_train;
        double* GPFLASH_RESTRICT kern_input_mirror_ptr = kern_input_mirror_thread_vec.memptr() + idx_thread_slice;
        const int n_below = n_train - i_col - 1;

        if (n_below > 0) {
            const double sq_norm_col = sq_norm_ptr[i_col];
            double* GPFLASH_RESTRICT kern_input_base_ptr = Gram_preserved_in_kmat_out_mat.colptr(i_col) + (i_col + 1);
            // Past the packed column's diagonal entry, which the diagonal pass below fills.
            double* GPFLASH_RESTRICT gram_flipped_ptr = store_hp_grad_factors
                    ? kern_common_rdot_mirror_packed_ptr + calc_packed_lower_offset(i_col, n_train) + 1
                    : gram_flipped_thread_vec.memptr() + idx_thread_slice;
            fill_gram_flipped_strict_column(Ydot_flipped_nflip_ntrain_mat, i_col, n_train, gram_flipped_ptr);

            // Both squared distances by the polarization identity, differing only in the flipped half's sign.
            GPFLASH_OMP_SIMD
            for (int i_below = 0; i_below < n_below; ++i_below) {
                const double sq_norm_sum = sq_norm_ptr[i_col + 1 + i_below] + sq_norm_col;
                const double gram_preserved = kern_input_base_ptr[i_below];
                const double gram_flipped = gram_flipped_ptr[i_below];
                kern_input_base_ptr[i_below] = clamp_sq_dist(sq_norm_sum - 2.0 * (gram_preserved + gram_flipped));
                kern_input_mirror_ptr[i_below] = clamp_sq_dist(sq_norm_sum - 2.0 * (gram_preserved - gram_flipped));
            }

            if (store_hp_grad_factors) {
                // The base factor goes to the transposed upper column; the mirrored factor replaces the cross
                // term just read, which the distance above was the last use of.
                double* const kern_common_rdot_base_ptr = Gram_preserved_in_kmat_out_mat.colptr(n_train - 1 - i_col);
                kern_fun.calc_kern_val_and_common_rdot_inplace(
                        kern_input_base_ptr, n_below, hp_kern_extra_ptr, kern_common_rdot_base_ptr);
                kern_fun.calc_kern_val_and_common_rdot_inplace(
                        kern_input_mirror_ptr, n_below, hp_kern_extra_ptr, gram_flipped_ptr);
            } else {
                kern_fun.calc_kern_val_inplace(kern_input_base_ptr, n_below, hp_kern_extra_ptr);
                kern_fun.calc_kern_val_inplace(kern_input_mirror_ptr, n_below, hp_kern_extra_ptr);
            }

            // The two terms, summed into the column the base term already occupies.
            GPFLASH_OMP_SIMD
            for (int i_below = 0; i_below < n_below; ++i_below)
                kern_input_base_ptr[i_below] += parity_sign * kern_input_mirror_ptr[i_below];
        }
    }

    // The two diagonals summed, and the mirrored factor's own diagonal entries, which its gradient reads.
    const arma::vec kern_diag_base_vec = kern_fun.calc_kern_diag_val_vec(ydot_train_sq_norm_vec, hp_kern_extra_ptr);
    for (int i_col = 0; i_col < n_train; ++i_col) {
        Gram_preserved_in_kmat_out_mat(i_col, i_col) =
                kern_diag_base_vec(i_col) + parity_sign * kern_mirror_diag_vec(i_col);

        // An odd model's k_sym(x, x) can still round to zero near the mirror plane at this hp_gamma.
        if (!(Gram_preserved_in_kmat_out_mat(i_col, i_col) > 0.0))
            throw ErrorNumericalFailure(
                    "gpflash::detail::calc_gram_preserved_to_kmat_inplace: training point " + std::to_string(i_col)
                    + " is within rounding of the mirror plane of an odd symmetry model at this hp_gamma, so "
                      "k_sym(x, x) rounds to zero.");

        if (store_hp_grad_factors)
            kern_common_rdot_mirror_packed_ptr[calc_packed_lower_offset(i_col, n_train)] =
                    kern_common_rdot_mirror_diag_vec(i_col);
    }
}

/**
 * Build K_sym for every pair of points, and return it.
 *
 * The lower triangle with the diagonal holds K_sym, the strict upper the base term's c^(rdot). The mirrored
 * term's c^(rdot) comes back packed in kern_common_rdot_mirror_packed_vec_out, and the squared norms K_sym was
 * built from in ydot_train_sq_norm_vec_out.
 *
 * Only one Gram half is ever a matrix: the norms come off the points, and the mirrored cross term is formed a
 * column at a time inside the sweep.
 */
inline arma::mat build_symmetry_kmat_lower_tri(
        const kernel::KernelBase& kern_fun,
        const arma::mat& Ydot_nfeature_ntrain_mat,
        const SymmetrySpec& symmetry_spec,
        const double* hp_kern_extra_ptr,
        arma::vec& ydot_train_sq_norm_vec_out,
        arma::vec* kern_common_rdot_mirror_packed_vec_out) {
    // The preserved and flipped halves are rows of Ydot, so the split reads the mirror in feature space.
    const arma::vec feature_mirror_sign_vec = kern_fun.calc_feature_mirror_sign_vec(symmetry_spec.mirror_sign_vec);
    const arma::uvec idx_preserved_vec = arma::find(feature_mirror_sign_vec > 0.0);
    const arma::uvec idx_flipped_vec = arma::find(feature_mirror_sign_vec < 0.0);
    // Gathered once so a column of the flipped dimensions is contiguous, which the per-column cross term reads.
    const arma::mat Ydot_flipped_nflip_ntrain_mat = Ydot_nfeature_ntrain_mat.rows(idx_flipped_vec);

    arma::mat K_sym_mat_lower;
    fill_inner_prod_self_lower_tri(Ydot_nfeature_ntrain_mat.rows(idx_preserved_vec), K_sym_mat_lower);

    // Straight off the points: the preserved half's diagonal alone is not the full norm, and the flipped half
    // has no diagonal to read.
    ydot_train_sq_norm_vec_out = arma::sum(arma::square(Ydot_nfeature_ntrain_mat), 0).t();
    const arma::vec ydot_flipped_sq_norm_vec = arma::sum(arma::square(Ydot_flipped_nflip_ntrain_mat), 0).t();

    calc_gram_preserved_to_kmat_inplace(
            kern_fun,
            K_sym_mat_lower,
            Ydot_flipped_nflip_ntrain_mat,
            ydot_train_sq_norm_vec_out,
            ydot_flipped_sq_norm_vec,
            symmetry_spec.parity_sign(),
            hp_kern_extra_ptr,
            kern_common_rdot_mirror_packed_vec_out);

    return K_sym_mat_lower;
}

// ── What the gradient reads back ────────────────────────────────────────────

// d(K_sym_ii)/d(hp), (n_train, n_dim + n_hp_kern_extra), the hp_gamma columns then the extras. The base term's
// diagonal is constant, so only the mirrored term k(x_i, S x_i) contributes, and its derivative is the mirrored
// pair sum at the one pair (x_i, S x_i), weighted by its own c^(rdot) and the parity.
inline arma::mat calc_symmetry_kern_diag_hp_grad_mat(
        const kernel::KernelBase& kern_fun,
        const kernel::HpGradInputs& hp_grad_inputs,
        const SymmetrySpec& symmetry_spec,
        const arma::vec& kern_common_rdot_mirror_packed_vec) {
    const int n_train = static_cast<int>(hp_grad_inputs.X_ntrain_ndim_mat.n_rows);
    const int n_dim = static_cast<int>(hp_grad_inputs.hp_gamma_vec.n_elem);
    const int n_hp_kern_extra = kern_fun.n_hp_kern_extra(n_dim);

    arma::mat d_kern_diag_d_hp_ntrain_nhp_mat(n_train, n_dim + n_hp_kern_extra, arma::fill::none);
    arma::vec d_kern_diag_d_hp_vec(n_dim + n_hp_kern_extra);
    for (int i_train = 0; i_train < n_train; ++i_train) {
        const double diag_weight = symmetry_spec.parity_sign()
                * kern_common_rdot_mirror_packed_vec(calc_packed_lower_offset(i_train, n_train));
        double* const d_kern_diag_d_hp_kern_extra_ptr =
                (n_hp_kern_extra > 0) ? d_kern_diag_d_hp_vec.memptr() + n_dim : nullptr;

        d_kern_diag_d_hp_vec.zeros();
        kern_fun.add_hp_gradient_contribution_mirrored(
                hp_grad_inputs,
                i_train,
                i_train,
                i_train + 1,
                &diag_weight,
                symmetry_spec.mirror_sign_vec,
                d_kern_diag_d_hp_vec.memptr(),
                d_kern_diag_d_hp_kern_extra_ptr);
        d_kern_diag_d_hp_ntrain_nhp_mat.row(i_train) = d_kern_diag_d_hp_vec.t();
    }

    return d_kern_diag_d_hp_ntrain_nhp_mat;
}

// ── What a query mirrors ───────────────────────────────────────────────────

// Ydot(S x): the flipped features negated. The cross-covariance against this is the mirrored term, and S
// preserves the norm, so the squared distance it feeds reads the same norms the base term does.
inline arma::vec build_mirrored_point_vec(const arma::vec& ydot_vec, const arma::vec& feature_mirror_sign_vec) {
    return ydot_vec % feature_mirror_sign_vec;
}

inline arma::mat build_mirrored_points_mat(
        const arma::mat& Ydot_nfeature_npoint_mat, const arma::vec& feature_mirror_sign_vec) {
    arma::mat Ydot_mirrored_nfeature_npoint_mat = Ydot_nfeature_npoint_mat;
    Ydot_mirrored_nfeature_npoint_mat.each_col() %= feature_mirror_sign_vec;

    return Ydot_mirrored_nfeature_npoint_mat;
}

// k_sym(x, x) = k(0) + parity_sign k(||ydot - S ydot||^2), and S moves a point only in the flipped features, so the
// mirrored distance is four times their squared norm. One value loop call covers every point.
inline arma::vec calc_symmetry_kern_diag_val_vec(
        const kernel::KernelBase& kern_fun,
        const arma::mat& Ydot_nfeature_npoint_mat,
        const arma::vec& ydot_sq_norm_vec,
        const SymmetrySpec& symmetry_spec,
        const double* hp_kern_extra_ptr) {
    const arma::vec feature_mirror_sign_vec = kern_fun.calc_feature_mirror_sign_vec(symmetry_spec.mirror_sign_vec);
    const arma::uvec idx_flipped_vec = arma::find(feature_mirror_sign_vec < 0.0);
    arma::vec kern_mirror_diag_vec =
            4.0 * arma::sum(arma::square(Ydot_nfeature_npoint_mat.rows(idx_flipped_vec)), 0).t();
    kern_fun.calc_kern_val_inplace(
            kern_mirror_diag_vec.memptr(), static_cast<std::ptrdiff_t>(kern_mirror_diag_vec.n_elem), hp_kern_extra_ptr);

    return kern_fun.calc_kern_diag_val_vec(ydot_sq_norm_vec, hp_kern_extra_ptr)
            + symmetry_spec.parity_sign() * kern_mirror_diag_vec;
}

// d(k_sym(x', x'))/d(x'), (n_dim, n_post). Only the mirrored term moves with the query, and the base kernel is
// stationary, so k(x', S x') is a function of x' - S x' and its total derivative is (I - S) times the derivative in
// its first point alone: twice that in a flipped dimension, zero in a preserved one. The first-point derivative is
// the kernel's own weighted sum, taken against the one point S x' with the mirrored c^(rdot) as its weight.
inline arma::mat calc_symmetry_d_kern_diag_val_d_x_post_mat(
        const kernel::KernelBase& kern_fun,
        const arma::mat& Ydot_nfeature_npost_mat,
        const arma::vec& hp_gamma_vec,
        const SymmetrySpec& symmetry_spec,
        const double* hp_kern_extra_ptr) {
    const arma::vec feature_mirror_sign_vec = kern_fun.calc_feature_mirror_sign_vec(symmetry_spec.mirror_sign_vec);
    const arma::uvec idx_flipped_vec = arma::find(feature_mirror_sign_vec < 0.0);
    const arma::uword n_post = Ydot_nfeature_npost_mat.n_cols;

    // The mirrored diagonal's own c^(rdot). Its value comes back in the input buffer and is not wanted here.
    arma::vec kern_mirror_diag_vec =
            4.0 * arma::sum(arma::square(Ydot_nfeature_npost_mat.rows(idx_flipped_vec)), 0).t();
    arma::vec kern_common_rdot_mirror_diag_vec(n_post);
    kern_fun.calc_kern_val_and_common_rdot_inplace(
            kern_mirror_diag_vec.memptr(),
            static_cast<std::ptrdiff_t>(n_post),
            hp_kern_extra_ptr,
            kern_common_rdot_mirror_diag_vec.memptr());

    // Each query's one "training" point is its own mirror image, weighted by that c^(rdot).
    const arma::rowvec kern_common_rdot_mirror_diag_row = kern_common_rdot_mirror_diag_vec.t();
    arma::mat q_ydot_mirror_nfeature_npost_mat =
            build_mirrored_points_mat(Ydot_nfeature_npost_mat, feature_mirror_sign_vec);
    q_ydot_mirror_nfeature_npost_mat.each_row() %= kern_common_rdot_mirror_diag_row;
    arma::mat d_kern_diag_val_ndim_npost_mat = kern_fun.calc_d_kern_sum_d_x_post_batch(
            Ydot_nfeature_npost_mat,
            q_ydot_mirror_nfeature_npost_mat,
            kern_common_rdot_mirror_diag_row,
            hp_gamma_vec,
            symmetry_spec.parity_sign());

    // (I - S) in input dimensions: 1 - s_l is 2 where flipped and 0 where preserved.
    const arma::vec one_minus_mirror_sign_vec = 1.0 - symmetry_spec.mirror_sign_vec;
    d_kern_diag_val_ndim_npost_mat.each_col() %= one_minus_mirror_sign_vec;

    return d_kern_diag_val_ndim_npost_mat;
}

// Row idx_active of the mirrored c^(rdot), gathered from the packed lower triangle that stores it. The
// mirrored term is symmetric, so an entry above the diagonal is read at its transpose.
inline arma::vec gather_active_row_packed_vec(
        const arma::vec& kern_common_rdot_mirror_packed_vec, const int idx_active, const int n_train) {
    arma::vec active_row_vec(n_train, arma::fill::zeros);
    for (int i_col = 0; i_col < idx_active; ++i_col)
        active_row_vec(i_col) =
                kern_common_rdot_mirror_packed_vec(calc_packed_lower_offset(i_col, n_train) + (idx_active - i_col));
    for (int i_row = idx_active + 1; i_row < n_train; ++i_row)
        active_row_vec(i_row) = kern_common_rdot_mirror_packed_vec(
                calc_packed_lower_offset(idx_active, n_train) + (i_row - idx_active));

    return active_row_vec;
}

}  // namespace detail
}  // namespace gpflash
