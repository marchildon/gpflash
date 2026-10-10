/**
 * BlasLapack.hpp -- the BLAS/LAPACK routines GpFlash calls directly, as inline wrappers over the Fortran symbols.
 *
 * Armadillo declares dgemv_, dgemm_, dsyrk_, dpotrf_ and dtrtri_ too, and C++ requires every declaration of a
 * C-linkage function to agree, so the prototypes follow Armadillo's: arma::blas_int integers, one hidden
 * arma::blas_len length after the arguments for each CHARACTER argument (ARMA_USE_FORTRAN_HIDDEN_ARGS, Armadillo's
 * default; gfortran-built routines read them), and lowercase names with a trailing underscore. The checks below
 * reject an Armadillo configured otherwise.
 *
 * The wrappers take scalars by value and arrays by pointer and pass the hidden lengths, so a call site writes neither.
 * No argument is checked here: the vendor's XERBLA reports a BLAS argument error, and LAPACK returns info.
 */
#pragma once

#include <armadillo>  // arma::blas_int, arma::blas_len and the ARMA_* configuration macros

#if !defined(ARMA_VERSION_MAJOR) || (ARMA_VERSION_MAJOR < 9) || (ARMA_VERSION_MAJOR == 9 && ARMA_VERSION_MINOR < 870)
#error "GpFlash requires Armadillo 9.870 or later"
#endif
#if !defined(ARMA_USE_BLAS) || !defined(ARMA_USE_LAPACK)
#error "GpFlash requires ARMA_USE_BLAS and ARMA_USE_LAPACK (linking GpFlash::GpFlash defines them)"
#endif
#if !defined(ARMA_DONT_USE_WRAPPER)
#error "GpFlash binds the BLAS/LAPACK symbols itself and requires ARMA_DONT_USE_WRAPPER (linking GpFlash::GpFlash defines it)"
#endif
#if !defined(ARMA_BLAS_UNDERSCORE) || defined(ARMA_BLAS_CAPITALS)
#error "GpFlash calls BLAS/LAPACK as dgemv_ etc.; ARMA_BLAS_UNDERSCORE must be on and ARMA_BLAS_CAPITALS off"
#endif
#if !defined(ARMA_USE_FORTRAN_HIDDEN_ARGS) || defined(ARMA_BLAS_NOEXCEPT) || defined(ARMA_LAPACK_NOEXCEPT)
#error "GpFlash's prototypes match Armadillo's defaults; leave hidden arguments on and the noexcept options off"
#endif
// An n_train by n_train buffer passes 2^31 elements at n_train ~ 46,000. Armadillo defaults to a 64-bit uword on a
// 64-bit target, so this catches only a build that overrode it.
static_assert(sizeof(arma::uword) == 8, "GpFlash needs a 64-bit arma::uword; build with ARMA_64BIT_WORD");

// Each trailing blas_len is the hidden length of the CHARACTER argument of the same name.
extern "C" {
// ── BLAS level 2 ──────────────────────────────────────────────────────────
void dgemv_(
        const char* trans,
        const arma::blas_int* m,
        const arma::blas_int* n,
        const double* blas_alpha,
        const double* a,
        const arma::blas_int* lda,
        const double* x,
        const arma::blas_int* incx,
        const double* blas_beta,
        double* y,
        const arma::blas_int* incy,
        arma::blas_len trans_len);
void dtrsv_(
        const char* uplo,
        const char* trans,
        const char* diag,
        const arma::blas_int* n,
        const double* a,
        const arma::blas_int* lda,
        double* x,
        const arma::blas_int* incx,
        arma::blas_len uplo_len,
        arma::blas_len trans_len,
        arma::blas_len diag_len);
// ── BLAS level 3 ──────────────────────────────────────────────────────────
void dgemm_(
        const char* transa,
        const char* transb,
        const arma::blas_int* m,
        const arma::blas_int* n,
        const arma::blas_int* k,
        const double* blas_alpha,
        const double* a,
        const arma::blas_int* lda,
        const double* b,
        const arma::blas_int* ldb,
        const double* blas_beta,
        double* c,
        const arma::blas_int* ldc,
        arma::blas_len transa_len,
        arma::blas_len transb_len);
void dsyrk_(
        const char* uplo,
        const char* trans,
        const arma::blas_int* n,
        const arma::blas_int* k,
        const double* blas_alpha,
        const double* a,
        const arma::blas_int* lda,
        const double* blas_beta,
        double* c,
        const arma::blas_int* ldc,
        arma::blas_len uplo_len,
        arma::blas_len trans_len);
void dtrsm_(
        const char* side,
        const char* uplo,
        const char* transa,
        const char* diag,
        const arma::blas_int* m,
        const arma::blas_int* n,
        const double* blas_alpha,
        const double* a,
        const arma::blas_int* lda,
        double* b,
        const arma::blas_int* ldb,
        arma::blas_len side_len,
        arma::blas_len uplo_len,
        arma::blas_len transa_len,
        arma::blas_len diag_len);
void dtrmm_(
        const char* side,
        const char* uplo,
        const char* transa,
        const char* diag,
        const arma::blas_int* m,
        const arma::blas_int* n,
        const double* blas_alpha,
        const double* a,
        const arma::blas_int* lda,
        double* b,
        const arma::blas_int* ldb,
        arma::blas_len side_len,
        arma::blas_len uplo_len,
        arma::blas_len transa_len,
        arma::blas_len diag_len);
// ── LAPACK ────────────────────────────────────────────────────────────────
void dpotrf_(
        const char* uplo,
        const arma::blas_int* n,
        double* a,
        const arma::blas_int* lda,
        arma::blas_int* info,
        arma::blas_len uplo_len);
void dtrtri_(
        const char* uplo,
        const char* diag,
        const arma::blas_int* n,
        double* a,
        const arma::blas_int* lda,
        arma::blas_int* info,
        arma::blas_len uplo_len,
        arma::blas_len diag_len);
void dlauum_(
        const char* uplo,
        const arma::blas_int* n,
        double* a,
        const arma::blas_int* lda,
        arma::blas_int* info,
        arma::blas_len uplo_len);
}

namespace gpflash {
namespace blas {
// Every CHARACTER argument GpFlash passes is one character, hence the hidden lengths of 1.
// ── Level 2 ───────────────────────────────────────────────────────────────
// y := alpha op(A) x + beta y                         (DGEMV)
inline void gemv(
        const char trans,
        const arma::blas_int m,
        const arma::blas_int n,
        const double blas_alpha,
        const double* a,
        const arma::blas_int lda,
        const double* x,
        const arma::blas_int incx,
        const double blas_beta,
        double* y,
        const arma::blas_int incy) {
    dgemv_(&trans, &m, &n, &blas_alpha, a, &lda, x, &incx, &blas_beta, y, &incy, 1);
}
// x := op(A)^-1 x, A triangular                       (DTRSV)
inline void trsv(
        const char uplo,
        const char trans,
        const char diag,
        const arma::blas_int n,
        const double* a,
        const arma::blas_int lda,
        double* x,
        const arma::blas_int incx) {
    dtrsv_(&uplo, &trans, &diag, &n, a, &lda, x, &incx, 1, 1, 1);
}
// ── Level 3 ───────────────────────────────────────────────────────────────
// C := alpha op(A) op(B) + beta C                     (DGEMM)
inline void gemm(
        const char transa,
        const char transb,
        const arma::blas_int m,
        const arma::blas_int n,
        const arma::blas_int k,
        const double blas_alpha,
        const double* a,
        const arma::blas_int lda,
        const double* b,
        const arma::blas_int ldb,
        const double blas_beta,
        double* c,
        const arma::blas_int ldc) {
    dgemm_(&transa, &transb, &m, &n, &k, &blas_alpha, a, &lda, b, &ldb, &blas_beta, c, &ldc, 1, 1);
}
// C := alpha op(A) op(A)^T + beta C, one triangle      (DSYRK)
inline void syrk(
        const char uplo,
        const char trans,
        const arma::blas_int n,
        const arma::blas_int k,
        const double blas_alpha,
        const double* a,
        const arma::blas_int lda,
        const double blas_beta,
        double* c,
        const arma::blas_int ldc) {
    dsyrk_(&uplo, &trans, &n, &k, &blas_alpha, a, &lda, &blas_beta, c, &ldc, 1, 1);
}
// B := alpha op(A)^-1 B or alpha B op(A)^-1            (DTRSM)
inline void trsm(
        const char side,
        const char uplo,
        const char transa,
        const char diag,
        const arma::blas_int m,
        const arma::blas_int n,
        const double blas_alpha,
        const double* a,
        const arma::blas_int lda,
        double* b,
        const arma::blas_int ldb) {
    dtrsm_(&side, &uplo, &transa, &diag, &m, &n, &blas_alpha, a, &lda, b, &ldb, 1, 1, 1, 1);
}
// B := alpha op(A) B or alpha B op(A)                  (DTRMM)
inline void trmm(
        const char side,
        const char uplo,
        const char transa,
        const char diag,
        const arma::blas_int m,
        const arma::blas_int n,
        const double blas_alpha,
        const double* a,
        const arma::blas_int lda,
        double* b,
        const arma::blas_int ldb) {
    dtrmm_(&side, &uplo, &transa, &diag, &m, &n, &blas_alpha, a, &lda, b, &ldb, 1, 1, 1, 1);
}
}  // namespace blas

namespace lapack {
// A := its Cholesky factor                            (DPOTRF)
inline void potrf(const char uplo, const arma::blas_int n, double* a, const arma::blas_int lda, arma::blas_int* info) {
    dpotrf_(&uplo, &n, a, &lda, info, 1);
}
// A := A^-1, A triangular                             (DTRTRI)
inline void trtri(
        const char uplo,
        const char diag,
        const arma::blas_int n,
        double* a,
        const arma::blas_int lda,
        arma::blas_int* info) {
    dtrtri_(&uplo, &diag, &n, a, &lda, info, 1, 1);
}
// A := U U^T or L^T L, one triangle                   (DLAUUM)
inline void lauum(const char uplo, const arma::blas_int n, double* a, const arma::blas_int lda, arma::blas_int* info) {
    dlauum_(&uplo, &n, a, &lda, info, 1);
}
}  // namespace lapack

}  // namespace gpflash
