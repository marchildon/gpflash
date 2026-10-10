/**
 * PosteriorCovariance.hpp -- build the train-against-query cross-covariance k_post.
 *
 * KernelMatrix.hpp's three steps, rectangular:
 *
 *   1) fill_inner_prod_vec for a single query, fill_inner_prod_mat for a batch.
 *   2) The kernel's calc_kern_input_inplace turns each into the single scalar it reads.
 *   3) Its calc_kern_val_inplace overwrites that with the kernel value.
 *
 * The single-query path runs all three on one buffer; the batch path fuses steps 1 and 2 into one
 * fill_sq_dist_mat or fill_inner_prod_mat.
 */
#pragma once

#include <armadillo>
#include <cstddef>

#include "gpflash/core/Platform.hpp"                 // GPFLASH_RESTRICT, GPFLASH_BLAS_SKINNY_GEMV_IS_SLOW
#include "gpflash/kernels/KernelBase.hpp"            // kernel::KernelBase
#include "gpflash/linear_algebra/InnerProducts.hpp"  // fill_inner_prod_vec, fill_inner_prod_vec_simd

namespace gpflash {
namespace detail {

/**
 * The n_train training points against one query: entry i_train = k(train column i_train, ydot_post_vec). Returned
 * by value like build_post_mat below; only the optional c^(rdot) is an out-param, in both.
 */
inline arma::vec build_post_vec(
        const kernel::KernelBase& kern_fun,
        const arma::mat& Ydot_nfeature_ntrain_mat,
        const arma::vec& ydot_train_sq_norm_vec,
        const arma::vec& ydot_post_vec,
        const double ydot_post_sq_norm,
        const double* hp_kern_extra_ptr,
        arma::vec* kern_common_rdot_post_vec_out) {
    const int n_train = static_cast<int>(Ydot_nfeature_ntrain_mat.n_cols);

    // One query makes this a transposed tall-narrow DGEMV, which some BLAS scale super-linearly. The two calls
    // below compute the same product; Platform.hpp selects between them, and the batch overload needs no such
    // choice because DGEMM has no equivalent pathology. Both size the output themselves, so the products land
    // in the returned vector and the two steps after this overwrite them in place.
    arma::vec k_post_vec;
#if GPFLASH_BLAS_SKINNY_GEMV_IS_SLOW
    fill_inner_prod_vec_simd(Ydot_nfeature_ntrain_mat, ydot_post_vec, /*scale_factor=*/1.0, k_post_vec);
#else
    fill_inner_prod_vec(Ydot_nfeature_ntrain_mat, ydot_post_vec, /*scale_factor=*/1.0, k_post_vec);
#endif
    double* GPFLASH_RESTRICT k_post_ptr = k_post_vec.memptr();

    double* kern_common_rdot_post_ptr = nullptr;
    if (kern_common_rdot_post_vec_out) {
        kern_common_rdot_post_vec_out->set_size(n_train);
        kern_common_rdot_post_ptr = kern_common_rdot_post_vec_out->memptr();
    }

    // Every training point is paired against the one query, so its norm is the shared side of the run.
    kern_fun.calc_kern_input_inplace(k_post_ptr, n_train, ydot_train_sq_norm_vec.memptr(), ydot_post_sq_norm);
    if (kern_common_rdot_post_ptr)
        kern_fun.calc_kern_val_and_common_rdot_inplace(
                k_post_ptr, n_train, hp_kern_extra_ptr, kern_common_rdot_post_ptr);
    else
        kern_fun.calc_kern_val_inplace(k_post_ptr, n_train, hp_kern_extra_ptr);

    return k_post_vec;
}

/**
 * The n_train training points against n_post queries: entry (i_train, i_post) = k(train column i_train, post
 * column i_post). One DGEMM covers every inner product, so this needs no platform split.
 */
inline arma::mat build_post_mat(
        const kernel::KernelBase& kern_fun,
        const arma::mat& Ydot_nfeature_ntrain_mat,
        const arma::vec& ydot_train_sq_norm_vec,
        const arma::mat& Ydot_nfeature_npost_mat,
        const arma::vec& ydot_post_sq_norm_vec,
        const double* hp_kern_extra_ptr,
        arma::mat* kern_common_rdot_ntrain_npost_mat_out) {
    const int n_train = static_cast<int>(Ydot_nfeature_ntrain_mat.n_cols);
    const int n_post = static_cast<int>(Ydot_nfeature_npost_mat.n_cols);

    // The scalar this kernel reads, a matrix at a time. Its family decides which, so a non-stationary kernel
    // skips the two axis adds a squared distance needs.
    arma::mat kern_input_ntrain_npost_mat;
    kern_fun.fill_kern_input_post_mat(
            Ydot_nfeature_ntrain_mat,
            ydot_train_sq_norm_vec,
            Ydot_nfeature_npost_mat,
            ydot_post_sq_norm_vec,
            kern_input_ntrain_npost_mat);

    double* kern_common_rdot_post_ptr = nullptr;
    if (kern_common_rdot_ntrain_npost_mat_out) {
        kern_common_rdot_ntrain_npost_mat_out->set_size(n_train, n_post);
        kern_common_rdot_post_ptr = kern_common_rdot_ntrain_npost_mat_out->memptr();
    }
    // Every entry of the (n_train, n_post) buffer in one call. ptrdiff_t before the multiply: the product
    // overflows int for a large batch against many training points.
    const std::ptrdiff_t n_elem = static_cast<std::ptrdiff_t>(n_train) * n_post;
    if (kern_common_rdot_post_ptr)
        kern_fun.calc_kern_val_and_common_rdot_inplace(
                kern_input_ntrain_npost_mat.memptr(), n_elem, hp_kern_extra_ptr, kern_common_rdot_post_ptr);
    else
        kern_fun.calc_kern_val_inplace(kern_input_ntrain_npost_mat.memptr(), n_elem, hp_kern_extra_ptr);

    return kern_input_ntrain_npost_mat;  // the inputs, overwritten with the kernel values
}

}  // namespace detail
}  // namespace gpflash
