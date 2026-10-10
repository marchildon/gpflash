/**
 * RecursiveInv.hpp -- DPOTRI's two halves as divide-and-conquer over the tuned BLAS-3 panels
 * DTRSM, DTRMM and DSYRK, which is the recursive case the GpFlash paper prescribes. Same operation count as the
 * vendor DPOTRI.
 *
 *   recursive_trtri_lower:  A(lower) holds L on entry, W = L^-1 on exit.
 *   recursive_lauum_lower:  A(lower) holds W on entry, W^T W on exit.
 *
 * Both touch only the lower triangle and the subdiagonal blocks; the training pipeline keeps C^(rdot) in
 * the strict upper triangle.
 *
 * Naming here follows the LAPACK convention (A, A11, A21, ld, n, nb, alpha, beta) rather than the
 * package's descriptive style, so the blocks read against the DTRTRI/DLAUUM documentation they mirror.
 */
#pragma once

#include <cstddef>

#include <armadillo>

#include "gpflash/core/Errors.hpp"                // check_lapack_info
#include "gpflash/linear_algebra/BlasLapack.hpp"  // gpflash::blas and gpflash::lapack

// info < 0 is the status worth guarding: it means a sub-block dimension was computed wrongly in this file, so
// check_lapack_info raises logic_error rather than ErrorNumericalFailure.

namespace gpflash {
namespace detail {

// Measured recursion block size; kept with the algorithm it tunes.
constexpr int PERF_RECURSIVE_INV_BLOCK_SIZE = 128;

// Recursion split point: half of n rounded up to a multiple of nb. The guard falls back to n/2 for n <= nb,
// which the recursion never passes, so both sub-blocks stay non-empty.
inline int recursive_split(const int n, const int nb) {
    int n1 = ((n / 2 + nb - 1) / nb) * nb;  // half, rounded up to nb multiple
    if (n1 >= n) n1 = n / 2;                // n <= nb
    return n1;
}

// Invert a lower-triangular block by divide-and-conquer: recurse on the diagonal blocks A11, A22, then
// W21 = -W22 A21 W11 via two DTRMMs.
//
// A lower with diag memory: IN: L, OUT: W = L^-1
inline void recursive_trtri_lower(double* A, const arma::blas_int ld, const int n, const int nb) {
    if (n <= nb) {
        const char uplo = 'L', diag = 'N';
        const arma::blas_int N = n, LDA = ld;
        arma::blas_int info = 0;

        gpflash::lapack::trtri(uplo, diag, N, A, LDA, &info);
        // info > 0 is a zero diagonal in the block, which makes it singular.
        check_lapack_info(
                info,
                "recursive_trtri_lower/dtrtri",
                n,
                "a diagonal element of the triangular block is exactly "
                "zero, so it is singular and was not inverted");

        return;
    }
    const int n1 = recursive_split(n, nb);
    const int n2 = n - n1;
    double* A11 = A;
    double* A21 = A + n1;
    double* A22 = A + static_cast<std::ptrdiff_t>(n1) * ld + n1;

    recursive_trtri_lower(A11, ld, n1, nb);  // A11 -> W11
    recursive_trtri_lower(A22, ld, n2, nb);  // A22 -> W22

    // Both DTRMMs below differ only in side, the triangular block and the sign of alpha; everything else is shared.
    const char uplo = 'L', trans = 'N', diag = 'N';
    const arma::blas_int M = n2, N1 = n1, LDA = ld, LDB = ld;
    {  // A21 := A21 * W11
        const char side = 'R';
        const double blas_alpha = 1.0;
        gpflash::blas::trmm(side, uplo, trans, diag, M, N1, blas_alpha, A11, LDA, A21, LDB);
    }
    {  // A21 := -W22 * A21
        const char side = 'L';
        const double blas_alpha = -1.0;
        gpflash::blas::trmm(side, uplo, trans, diag, M, N1, blas_alpha, A22, LDA, A21, LDB);
    }
}

// Form the lower triangle of W^T W from W, the DLAUUM half: recurse on A11, fold in A11 += W21^T W21
// (DSYRK), set A21 = W22^T A21 (DTRMM), recurse on A22.
//
// A lower with diag memory: IN: W, OUT: W^T W
inline void recursive_lauum_lower(double* A, const arma::blas_int ld, const int n, const int nb) {
    if (n <= nb) {
        const char uplo = 'L';
        const arma::blas_int N = n, LDA = ld;
        arma::blas_int info = 0;

        gpflash::lapack::lauum(uplo, N, A, LDA, &info);
        check_lapack_info(
                info, "recursive_lauum_lower/dlauum", n, "unexpected non-zero status from a triangular product");
        return;
    }
    const int n1 = recursive_split(n, nb);
    const int n2 = n - n1;
    double* A11 = A;
    double* A21 = A + n1;
    double* A22 = A + static_cast<std::ptrdiff_t>(n1) * ld + n1;

    recursive_lauum_lower(A11, ld, n1, nb);  // A11 -> W11^T W11
    {                                        // A11 += W21^T W21   (A21 still holds W21)
        const char uplo = 'L', trans = 'T';
        const arma::blas_int N1 = n1, K = n2, LDA = ld, LDC = ld;
        const double blas_alpha = 1.0, blas_beta = 1.0;  // beta of 1 accumulates into A11
        gpflash::blas::syrk(uplo, trans, N1, K, blas_alpha, A21, LDA, blas_beta, A11, LDC);
    }
    {  // A21 := W22^T * A21  (A22 still holds W22)
        const char side = 'L', uplo = 'L', trans = 'T', diag = 'N';
        const arma::blas_int M = n2, N1 = n1, LDA = ld, LDB = ld;
        const double blas_alpha = 1.0;
        gpflash::blas::trmm(side, uplo, trans, diag, M, N1, blas_alpha, A22, LDA, A21, LDB);
    }
    recursive_lauum_lower(A22, ld, n2, nb);  // A22 -> W22^T W22
}

// Form the lower triangle of Sigma^-1 from its Cholesky factor. Equivalent to LAPACK dpotri(UPLO='L')
// but routed through the recursive BLAS-3 panels.
//
// A lower with diag memory: IN: L from dpotrf(UPLO='L'), OUT: Sigma^-1
inline void recursive_dpotri_lower(double* A, const int n) {
    const int nb = PERF_RECURSIVE_INV_BLOCK_SIZE;
    recursive_trtri_lower(A, n, n, nb);  // L      -> W = L^-1
    recursive_lauum_lower(A, n, n, nb);  // W      -> W^T W = Sigma^-1 (lower)
}

}  // namespace detail
}  // namespace gpflash
