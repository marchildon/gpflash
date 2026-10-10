/**
 * InnerProducts.hpp -- lhs^T rhs, for column-major operands.
 *
 * Free of GP vocabulary: these are BLAS wrappers, so lhs/rhs name each operand's position in the
 * product, not what a caller puts there. The operands share their row count, the contracted dimension, and each
 * has its own column count. Callers here pass scaled feature points one per column, but nothing depends on that.
 *
 * scale_factor is BLAS's alpha: fill_sq_dist_mat passes -2, so adding the two squared norms completes
 * ||lhs - rhs||^2.
 *
 * Three shapes, one BLAS call each: a matrix against a vector (DGEMV), against another matrix (DGEMM), and
 * against itself (DSYRK, which fills one triangle because the result is symmetric).
 */
#pragma once

#include <algorithm>  // std::max, std::min
#include <cstddef>

#include <armadillo>
#include "gpflash/core/Platform.hpp"              // GPFLASH_RESTRICT, GPFLASH_OMP_SIMD_REDUCTION
#include "gpflash/linear_algebra/BlasLapack.hpp"  // gpflash::blas and gpflash::lapack

namespace gpflash {
namespace detail {

// inner_prod_mat_lower_out with diag = lhs^T lhs
//
// One DSYRK over the lower triangle and the diagonal. The diagonal comes out holding ||lhs_i||^2, which a
// caller gathers as the squared norms.
inline void fill_inner_prod_self_lower_tri(const arma::mat& lhs_mat, arma::mat& inner_prod_mat_lower_out) {
    const char uplo = 'L', trans = 'T';
    const arma::blas_int N = static_cast<arma::blas_int>(lhs_mat.n_cols);
    const arma::blas_int K_dim = static_cast<arma::blas_int>(lhs_mat.n_rows);
    const double blas_alpha = 1.0, blas_beta = 0.0;
    const arma::blas_int lda = K_dim, ldc = N;

    inner_prod_mat_lower_out.set_size(lhs_mat.n_cols, lhs_mat.n_cols);
    gpflash::blas::syrk(
            uplo,
            trans,
            N,
            K_dim,
            blas_alpha,
            lhs_mat.memptr(),
            lda,
            blas_beta,
            inner_prod_mat_lower_out.memptr(),
            ldc);
}

// inner_prod_vec_out = scale_factor * lhs^T rhs
inline void fill_inner_prod_vec(
        const arma::mat& lhs_mat, const arma::vec& rhs_vec, const double scale_factor, arma::vec& inner_prod_vec_out) {
    const char trans = 'T';
    const arma::blas_int M = static_cast<arma::blas_int>(lhs_mat.n_rows);
    const arma::blas_int N = static_cast<arma::blas_int>(lhs_mat.n_cols), lda = M, inc = 1;
    const double blas_alpha = scale_factor, blas_beta = 0.0;

    inner_prod_vec_out.set_size(lhs_mat.n_cols);
    gpflash::blas::gemv(
            trans,
            M,
            N,
            blas_alpha,
            lhs_mat.memptr(),
            lda,
            rhs_vec.memptr(),
            inc,
            blas_beta,
            inner_prod_vec_out.memptr(),
            inc);
}

// inner_prod_mat_out = scale_factor * lhs^T rhs
inline void fill_inner_prod_mat(
        const arma::mat& lhs_mat, const arma::mat& rhs_mat, const double scale_factor, arma::mat& inner_prod_mat_out) {
    const char trans_a = 'T', trans_b = 'N';
    const arma::blas_int M = static_cast<arma::blas_int>(lhs_mat.n_cols);
    const arma::blas_int N = static_cast<arma::blas_int>(rhs_mat.n_cols);
    const arma::blas_int K_dim = static_cast<arma::blas_int>(lhs_mat.n_rows);
    const arma::blas_int lda = K_dim, ldb = K_dim, ldc = M;
    const double blas_alpha = scale_factor, blas_beta = 0.0;

    inner_prod_mat_out.set_size(lhs_mat.n_cols, rhs_mat.n_cols);
    gpflash::blas::gemm(
            trans_a,
            trans_b,
            M,
            N,
            K_dim,
            blas_alpha,
            lhs_mat.memptr(),
            lda,
            rhs_mat.memptr(),
            ldb,
            blas_beta,
            inner_prod_mat_out.memptr(),
            ldc);
}

// The same product as fill_inner_prod_vec, as one explicit dot per column rather than a DGEMV. Interchangeable
// with it, so a caller can select on the platform and a test can assert the two agree.
inline void fill_inner_prod_vec_simd(
        const arma::mat& lhs_mat, const arma::vec& rhs_vec, const double scale_factor, arma::vec& inner_prod_vec_out) {
    const int n_row = static_cast<int>(lhs_mat.n_rows);
    const std::size_t n_lhs = static_cast<std::size_t>(lhs_mat.n_cols);
    const double* GPFLASH_RESTRICT lhs_ptr = lhs_mat.memptr();
    const double* GPFLASH_RESTRICT rhs_ptr = rhs_vec.memptr();

    inner_prod_vec_out.set_size(n_lhs);
    double* GPFLASH_RESTRICT inner_prod_ptr = inner_prod_vec_out.memptr();
    for (std::size_t i_lhs = 0; i_lhs < n_lhs; ++i_lhs) {
        const double* GPFLASH_RESTRICT col_ptr = lhs_ptr + i_lhs * static_cast<std::size_t>(n_row);
        double inner_prod = 0.0;
        GPFLASH_OMP_SIMD_REDUCTION(+ : inner_prod)
        for (int i_row = 0; i_row < n_row; ++i_row) inner_prod += col_ptr[i_row] * rhs_ptr[i_row];
        inner_prod_ptr[i_lhs] = scale_factor * inner_prod;
    }
}

// A scaled squared distance at which the exponential kernels are 0 and every kernel's arithmetic stays finite.
constexpr double SQ_DIST_MAX = 1e200;

// A scaled squared distance from the polarization identity, held to [0, SQ_DIST_MAX]:
// 1. below 0, where cancellation takes coincident points and a kernel's sqrt would return NaN;
// 2. above SQ_DIST_MAX, including a norm that overflowed to inf, or to inf - inf = NaN. std::min returns its first
//    argument when the comparison fails, so a NaN maps to SQ_DIST_MAX.
inline double clamp_sq_dist(const double sq_dist) { return std::max(0.0, std::min(SQ_DIST_MAX, sq_dist)); }

// sq_dist_mat_out(i, j) = ||lhs_i - rhs_j||^2, by the polarization identity
//   ||lhs - rhs||^2 = ||lhs||^2 + ||rhs||^2 - 2 lhs . rhs,
// whose cross term is the -2 the product carries. Both norm vectors are the caller's, already held.
inline void fill_sq_dist_mat(
        const arma::mat& lhs_mat,
        const arma::mat& rhs_mat,
        const arma::vec& lhs_sq_norm_vec,
        const arma::vec& rhs_sq_norm_vec,
        arma::mat& sq_dist_mat_out) {
    fill_inner_prod_mat(lhs_mat, rhs_mat, /*scale_factor=*/-2.0, sq_dist_mat_out);
    sq_dist_mat_out.each_col() += lhs_sq_norm_vec;
    sq_dist_mat_out.each_row() += rhs_sq_norm_vec.t();
    sq_dist_mat_out.transform(clamp_sq_dist);
}

}  // namespace detail
}  // namespace gpflash
