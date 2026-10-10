/**
 * Cholesky.hpp -- everything done with a lower Cholesky factor L: produce it, read its determinant, solve
 * against it.
 *
 * The solves are named solve_chol_<forward|transpose>_<vec|mat>[_inplace]: forward gives L^-1 b, transpose
 * gives L^-T b, and a full solve (L L^T)^-1 b is one of each. Only forward has a plain form that leaves the
 * right-hand side intact; transpose is _inplace only, which is what its callers want.
 */
#pragma once

#include <armadillo>
#include <cmath>

#include "gpflash/core/Errors.hpp"                // check_lapack_info
#include "gpflash/linear_algebra/BlasLapack.hpp"  // gpflash::blas and gpflash::lapack

namespace gpflash {
namespace detail {

// Replace a symmetric positive definite matrix by the lower Cholesky factor L that the solves below take.
// Throws ErrorNumericalFailure when the matrix is not positive definite.
//
// mat_to_factorize lower with diag memory: IN: the matrix to factorize, OUT: L
inline void factorize_chol_inplace(arma::mat& mat_to_factorize, const char* blas_call_site) {
    const int n_row = static_cast<int>(mat_to_factorize.n_rows);
    const char uplo = 'L';
    const arma::blas_int N = n_row, lda = N;
    arma::blas_int info = 0;

    gpflash::lapack::potrf(uplo, N, mat_to_factorize.memptr(), lda, &info);
    check_lapack_info(info, blas_call_site, n_row, "the matrix is not positive definite, so it was not factorized");
}

// ln|L L^T|, which is 2 sum_i ln L_ii.
inline double calc_ln_det_from_chol(const arma::mat& L_mat_lower) {
    const int n_row = static_cast<int>(L_mat_lower.n_rows);
    double ln_det = 0.0;
    for (int i_diag = 0; i_diag < n_row; ++i_diag) ln_det += 2.0 * std::log(L_mat_lower(i_diag, i_diag));

    return ln_det;
}

// The one BLAS call the solves below wrap. trans is 'N' for L x = b and 'T' for L^T x = b; a single right-hand
// side takes the Level-2 DTRSV and several take the Level-3 DTRSM.
//
// rhs_ptr memory: IN: right-hand side, column major with leading dimension L.n_rows, OUT: solution
inline void solve_chol_blas_inplace(
        const arma::mat& L_mat_lower, double* rhs_ptr, const int n_rhs_col, const char trans) {
    const char uplo = 'L', diag_n = 'N';
    const arma::blas_int M = static_cast<arma::blas_int>(L_mat_lower.n_rows), lda = M;

    if (n_rhs_col == 1) {
        const arma::blas_int incx = 1;

        gpflash::blas::trsv(uplo, trans, diag_n, M, L_mat_lower.memptr(), lda, rhs_ptr, incx);
        return;
    }
    const char side = 'L';
    const arma::blas_int N = static_cast<arma::blas_int>(n_rhs_col), ldb = M;
    const double blas_alpha = 1.0;

    gpflash::blas::trsm(side, uplo, trans, diag_n, M, N, blas_alpha, L_mat_lower.memptr(), lda, rhs_ptr, ldb);
}

// rhs_vec memory: IN: b, OUT: L^-1 b
inline void solve_chol_forward_vec_inplace(const arma::mat& L_mat_lower, arma::vec& rhs_vec) {
    solve_chol_blas_inplace(L_mat_lower, rhs_vec.memptr(), 1, 'N');
}

// rhs_mat memory: IN: B, OUT: L^-1 B
inline void solve_chol_forward_mat_inplace(const arma::mat& L_mat_lower, arma::mat& rhs_mat) {
    solve_chol_blas_inplace(L_mat_lower, rhs_mat.memptr(), static_cast<int>(rhs_mat.n_cols), 'N');
}

// Called on the result of a forward solve this completes (L L^T)^-1 b, which is how a full solve is taken.
//
// rhs_vec memory: IN: b, OUT: L^-T b
inline void solve_chol_transpose_vec_inplace(const arma::mat& L_mat_lower, arma::vec& rhs_vec) {
    solve_chol_blas_inplace(L_mat_lower, rhs_vec.memptr(), 1, 'T');
}

// rhs_mat memory: IN: B, OUT: L^-T B
inline void solve_chol_transpose_mat_inplace(const arma::mat& L_mat_lower, arma::mat& rhs_mat) {
    solve_chol_blas_inplace(L_mat_lower, rhs_mat.memptr(), static_cast<int>(rhs_mat.n_cols), 'T');
}

// L^-1 b, leaving b intact.
inline arma::vec solve_chol_forward_vec(const arma::mat& L_mat_lower, const arma::vec& b_vec) {
    arma::vec lin_sol_L_b_vec = b_vec;
    solve_chol_forward_vec_inplace(L_mat_lower, lin_sol_L_b_vec);

    return lin_sol_L_b_vec;
}

// L^-1 B, leaving B intact.
inline arma::mat solve_chol_forward_mat(const arma::mat& L_mat_lower, const arma::mat& B_mat) {
    arma::mat lin_sol_L_B_mat = B_mat;
    solve_chol_forward_mat_inplace(L_mat_lower, lin_sol_L_B_mat);

    return lin_sol_L_B_mat;
}

}  // namespace detail
}  // namespace gpflash
