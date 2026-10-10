/**
 * KernelStationary.hpp -- the squared-distance machinery for a kernel that reads its points only through the
 * scaled distance ||rdot_vec|| in the space it evaluates on.
 *
 * This base owns the path from points to the kernel's input scalar and back:
 *
 *     the scaled squared distance as the input, over a matrix, a run, and a K column
 *     the unit diagonal every such kernel has, k(x, x) = 1
 *
 * A derived kernel still writes the value loops that turn that distance into k and c^(rdot):
 *
 *     calc_kern_val_inplace, calc_kern_val_and_common_rdot_inplace
 *
 * so K assembly makes a fixed number of virtual calls per column, with the per-element math inlined inside the
 * kernel.
 */
#pragma once

#include <armadillo>  // arma::mat, arma::vec

#include "gpflash/core/Platform.hpp"                 // GPFLASH_RESTRICT, GPFLASH_OMP_SIMD
#include "gpflash/kernels/KernelBase.hpp"            // KernelBase
#include "gpflash/linear_algebra/InnerProducts.hpp"  // fill_sq_dist_mat, clamp_sq_dist

namespace gpflash {
namespace kernel {

class KernelStationary : public KernelBase {
   public:
    bool is_stationary() const final { return true; }

    // K is a correlation matrix: k(0) == 1 for every kernel on this base.
    bool is_kern_diag_unity() const final { return true; }

    // ── The scalar every stationary kernel reads ──────────────────────────────
    void fill_kern_input_post_mat(
            const arma::mat& Ydot_nfeature_ntrain_mat,
            const arma::vec& ydot_train_sq_norm_vec,
            const arma::mat& Ydot_nfeature_npost_mat,
            const arma::vec& ydot_post_sq_norm_vec,
            arma::mat& kern_input_ntrain_npost_mat_out) const final {
        detail::fill_sq_dist_mat(
                Ydot_nfeature_ntrain_mat,
                Ydot_nfeature_npost_mat,
                ydot_train_sq_norm_vec,
                ydot_post_sq_norm_vec,
                kern_input_ntrain_npost_mat_out);
    }

    // The run form, for the single-query posterior path.
    void calc_kern_input_inplace(
            double* kern_input_ptr,
            const int n_elem,
            const double* sq_norm_ptr,
            const double sq_norm_shared) const final {
        calc_sq_dist_run_inplace(kern_input_ptr, n_elem, sq_norm_ptr, sq_norm_shared);
    }

    // One column of K: the scaled distances, then this kernel's value loop, written once here for every kernel on
    // this base.
    void calc_kmat_column_inplace(
            double* GPFLASH_RESTRICT ydot_inner_prod_col_ptr,
            const int n_below,
            const double sq_norm_col,
            const double* GPFLASH_RESTRICT sq_norm_below_ptr,
            const double* hp_kern_extra_ptr,
            double* GPFLASH_RESTRICT kern_common_rdot_col_ptr_out) const final {
        calc_sq_dist_run_inplace(ydot_inner_prod_col_ptr, n_below, sq_norm_below_ptr, sq_norm_col);
        if (kern_common_rdot_col_ptr_out)
            calc_kern_val_and_common_rdot_inplace(
                    ydot_inner_prod_col_ptr, n_below, hp_kern_extra_ptr, kern_common_rdot_col_ptr_out);
        else
            calc_kern_val_inplace(ydot_inner_prod_col_ptr, n_below, hp_kern_extra_ptr);
    }

    // ── The diagonal, constant for every kernel on this base ──────────────────
    double calc_kern_diag_val(const double /*ydot_sq_norm*/, const double* /*hp_kern_extra_ptr*/) const final {
        return 1.0;
    }

    arma::vec calc_kern_diag_val_vec(
            const arma::vec& ydot_sq_norm_vec, const double* /*hp_kern_extra_ptr*/) const final {
        return arma::ones<arma::vec>(ydot_sq_norm_vec.n_elem);
    }

   protected:
    KernelStationary() = default;

    // The scaled squared distance, by the polarization identity, over a run of pairs sharing one point, clamped by
    // detail::clamp_sq_dist. Non-virtual, so a kernel's column inlines it.
    //
    // kern_input_ptr memory: IN: ydot_inner_prod, OUT: the scaled squared distance
    static void calc_sq_dist_run_inplace(
            double* GPFLASH_RESTRICT kern_input_ptr,
            const int n_elem,
            const double* GPFLASH_RESTRICT sq_norm_ptr,
            const double sq_norm_shared) {
        GPFLASH_OMP_SIMD
        for (int i_elem = 0; i_elem < n_elem; ++i_elem)
            kern_input_ptr[i_elem] =
                    detail::clamp_sq_dist(sq_norm_ptr[i_elem] + sq_norm_shared - 2.0 * kern_input_ptr[i_elem]);
    }
};

}  // namespace kernel
}  // namespace gpflash
