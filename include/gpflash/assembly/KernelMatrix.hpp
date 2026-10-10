/**
 * KernelMatrix.hpp -- build the training kernel matrix K from the scaled feature points Ydot.
 *
 * Three steps on one buffer, in two functions:
 *
 *   1) build_gram_lower_tri fills the lower triangle, diagonal included, with the Gram cross terms.
 *   2) Each cross term becomes the single scalar the kernel reads.
 *   3) That is overwritten with the kernel value.
 *
 * Steps 2 and 3 are calc_gram_to_kmat_inplace, which hands one column at a time to the kernel's
 * calc_kmat_column_inplace and does both there. build_kmat_lower_tri runs all three; they are separate so a
 * caller can read the squared norms off the Gram diagonal before step 3 replaces it with K's own.
 *
 * K comes out raw, with no nugget and no noise. A caller that goes on to factorize conditions it first.
 */
#pragma once

#include <armadillo>

#include "gpflash/core/Platform.hpp"                 // GPFLASH_RESTRICT, OMP constants
#include "gpflash/kernels/KernelBase.hpp"            // kernel::KernelBase
#include "gpflash/linear_algebra/InnerProducts.hpp"  // fill_inner_prod_self_lower_tri

namespace gpflash {
namespace detail {

// Turn the Gram cross terms into kernel values, one column at a time, and replace the diagonal with
// calc_kern_diag_val. The norms come from ydot_sq_norm_vec, not from the diagonal each column overwrites,
// because the kernel's SIMD loop needs them contiguous.
//
// Where store_hp_grad_factors asks, the per-value derivative factor c^(rdot) is kept in the strict upper
// triangle, column-major, so the hyperparameter gradient need not evaluate the kernel again.
//
// ydot_inner_prod_mat strict lower memory: IN: ydot_inner_prod, OUT: kernel values
// ydot_inner_prod_mat diag memory:         IN: ydot_sq_norm,    OUT: kern_diag_val
inline void calc_gram_to_kmat_inplace(
        const kernel::KernelBase& kern_fun,
        arma::mat& ydot_inner_prod_mat,
        const arma::vec& ydot_sq_norm_vec,
        const double* hp_kern_extra_ptr,
        const bool store_hp_grad_factors) {
    const int n_train = static_cast<int>(ydot_inner_prod_mat.n_cols);
    const double* GPFLASH_RESTRICT sq_norm_ptr = ydot_sq_norm_vec.memptr();

#pragma omp parallel for schedule(dynamic, gpflash::PERF_OMP_COL_CHUNK) if (n_train >= gpflash::PERF_OMP_MIN_N_POINT)
    for (int i_col = 0; i_col < n_train; ++i_col) {
        const int n_below = n_train - i_col - 1;
        if (n_below > 0) {
            // Null asks the kernel to discard c^(rdot) rather than store it.
            double* const kern_common_rdot_col_ptr =
                    store_hp_grad_factors ? ydot_inner_prod_mat.colptr(n_train - 1 - i_col) : nullptr;
            kern_fun.calc_kmat_column_inplace(
                    ydot_inner_prod_mat.colptr(i_col) + (i_col + 1),
                    n_below,
                    sq_norm_ptr[i_col],
                    &sq_norm_ptr[i_col + 1],
                    hp_kern_extra_ptr,
                    kern_common_rdot_col_ptr);
        }
        // The diagonal entry, one scalar per column: the vector form would allocate n_train to deliver one value.
        ydot_inner_prod_mat(i_col, i_col) = kern_fun.calc_kern_diag_val(sq_norm_ptr[i_col], hp_kern_extra_ptr);
    }
}

// The Gram cross terms in the lower triangle, DSYRK leaving ||ydot_i||^2 on the diagonal.
inline arma::mat build_gram_lower_tri(const arma::mat& Ydot_nfeature_ntrain_mat) {
    const int n_train = static_cast<int>(Ydot_nfeature_ntrain_mat.n_cols);

    arma::mat Gram_mat_lower(n_train, n_train, arma::fill::none);
    fill_inner_prod_self_lower_tri(Ydot_nfeature_ntrain_mat, Gram_mat_lower);

    return Gram_mat_lower;
}

/**
 * Build K for every pair of points in Ydot_nfeature_ntrain_mat, and return it.
 *
 * The lower triangle, diagonal included, holds K; the strict upper holds c^(rdot) when store_hp_grad_factors.
 * Each step names its parameter for what it receives, so one buffer arrives as ydot_inner_prod_mat and leaves
 * as K_mat_lower.
 */
inline arma::mat build_kmat_lower_tri(
        const kernel::KernelBase& kern_fun,
        const arma::mat& Ydot_nfeature_ntrain_mat,
        const double* hp_kern_extra_ptr,
        const bool store_hp_grad_factors) {
    arma::mat K_mat_lower = build_gram_lower_tri(Ydot_nfeature_ntrain_mat);
    const arma::vec ydot_sq_norm_vec = K_mat_lower.diag();
    calc_gram_to_kmat_inplace(kern_fun, K_mat_lower, ydot_sq_norm_vec, hp_kern_extra_ptr, store_hp_grad_factors);

    return K_mat_lower;
}

}  // namespace detail
}  // namespace gpflash
