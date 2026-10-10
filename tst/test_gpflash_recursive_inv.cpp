// The recursive replacement for LAPACK dpotri. What this file asserts:
//
//   1) split_point          recursive_split keeps both sub-blocks non-empty and nb-aligned where it can
//   2) trtri_vs_dense       recursive_trtri_lower turns L into L^-1
//   3) lauum_vs_dense       recursive_lauum_lower turns W into the lower triangle of W^T W
//   4) dpotri_vs_dense      the pair gives Sigma^-1 from L
//
// main runs 2 to 4 at sizes either side of PERF_RECURSIVE_INV_BLOCK_SIZE, so the base case and at least two
// levels of recursion are each covered; the sizes are chosen so that a split lands on an unaligned remainder.
// Each of 2 to 4 also asserts that nothing above the diagonal is written.
#include <armadillo>
#include <cmath>
#include <cstdio>

#include "GpFlashTestSupport.hpp"
#include "gpflash/linear_algebra/RecursiveInv.hpp"

using gpflash::detail::PERF_RECURSIVE_INV_BLOCK_SIZE;
using gpflash::detail::recursive_dpotri_lower;
using gpflash::detail::recursive_lauum_lower;
using gpflash::detail::recursive_split;
using gpflash::detail::recursive_trtri_lower;
using gpflash_test::is_strict_upper_intact;
using gpflash_test::kDetailLen;
using gpflash_test::kTolAltPath;
using gpflash_test::kUpperSentinel;
using gpflash_test::lower_rel_err;
using gpflash_test::make_points;
using gpflash_test::report;

namespace {

// A well-conditioned Sigma and its lower Cholesky factor, with a sentinel written above the diagonal so any
// stray write there is detectable.
arma::mat make_chol_factor(const int n_train) {
    const arma::mat A_mat = make_points(n_train, n_train);
    const arma::mat Sigma_mat = A_mat.t() * A_mat + n_train * arma::eye(n_train, n_train);
    arma::mat L_mat = arma::chol(Sigma_mat, "lower");
    for (int i_col = 1; i_col < n_train; ++i_col)
        for (int i_row = 0; i_row < i_col; ++i_row) L_mat(i_row, i_col) = kUpperSentinel;

    return L_mat;
}

// Reference W = L^-1 by plain forward substitution, reading only the lower triangle. Not
// arma::inv(arma::trimatl(...)): that runs one full-size LAPACK dtrtri, which some OpenBLAS builds segfault
// inside at n_train = 333, exactly the unaligned-remainder size tested here.
arma::mat inv_lower_ref(const arma::mat& L_mat) {
    const int n_train = static_cast<int>(L_mat.n_rows);
    arma::mat W_mat(n_train, n_train, arma::fill::zeros);
    for (int i_col = 0; i_col < n_train; ++i_col) {
        W_mat(i_col, i_col) = 1.0 / L_mat(i_col, i_col);
        for (int i_row = i_col + 1; i_row < n_train; ++i_row) {
            double row_sum = 0.0;
            for (int i_mid = i_col; i_mid < i_row; ++i_mid) row_sum += L_mat(i_row, i_mid) * W_mat(i_mid, i_col);
            W_mat(i_row, i_col) = -row_sum / L_mat(i_row, i_row);
        }
    }

    return W_mat;
}

// Asserts recursive_trtri_lower turns L into L^-1, leaving the strict upper triangle alone.
bool test_trtri_vs_dense(const char* name, const int n_train) {
    const arma::mat L_mat = make_chol_factor(n_train);
    const arma::mat W_expected_mat = inv_lower_ref(L_mat);

    arma::mat W_mat = L_mat;
    recursive_trtri_lower(W_mat.memptr(), n_train, n_train, PERF_RECURSIVE_INV_BLOCK_SIZE);

    const double err = lower_rel_err(W_mat, W_expected_mat);
    const bool upper_ok = is_strict_upper_intact(W_mat);
    char detail[kDetailLen];
    std::snprintf(
            detail, sizeof(detail), "n_train=%d err=%.1e upper_intact=%d", n_train, err, static_cast<int>(upper_ok));

    return report(name, err < kTolAltPath && upper_ok, detail);
}

// Asserts recursive_lauum_lower turns W into the lower triangle of W^T W.
bool test_lauum_vs_dense(const char* name, const int n_train) {
    const arma::mat L_mat = make_chol_factor(n_train);
    const arma::mat W_dense_mat = inv_lower_ref(L_mat);
    const arma::mat product_ref_mat = W_dense_mat.t() * W_dense_mat;

    // Start from L so the sentinel is in place, then overwrite the lower triangle with W.
    arma::mat A_mat = L_mat;
    for (int i_col = 0; i_col < n_train; ++i_col)
        for (int i_row = i_col; i_row < n_train; ++i_row) A_mat(i_row, i_col) = W_dense_mat(i_row, i_col);
    recursive_lauum_lower(A_mat.memptr(), n_train, n_train, PERF_RECURSIVE_INV_BLOCK_SIZE);

    const double err = lower_rel_err(A_mat, product_ref_mat);
    const bool upper_ok = is_strict_upper_intact(A_mat);
    char detail[kDetailLen];
    std::snprintf(
            detail, sizeof(detail), "n_train=%d err=%.1e upper_intact=%d", n_train, err, static_cast<int>(upper_ok));

    return report(name, err < kTolAltPath && upper_ok, detail);
}

// Asserts recursive_dpotri_lower gives Sigma^-1 from L, against an independent symmetric inverse.
bool test_dpotri_vs_dense(const char* name, const int n_train) {
    const arma::mat L_mat = make_chol_factor(n_train);
    const arma::mat Sigma_mat = arma::trimatl(L_mat) * arma::trimatl(L_mat).t();
    // Sigma^-1 = W^T W off the reference inverse; arma::inv_sympd would reach the same full-size dtrtri.
    const arma::mat W_dense_mat = inv_lower_ref(L_mat);
    const arma::mat Sigma_inv_expected_mat = W_dense_mat.t() * W_dense_mat;

    arma::mat Sigma_inv_mat_lower = L_mat;
    recursive_dpotri_lower(Sigma_inv_mat_lower.memptr(), n_train);

    const double err = lower_rel_err(Sigma_inv_mat_lower, Sigma_inv_expected_mat);
    const bool upper_ok = is_strict_upper_intact(Sigma_inv_mat_lower);
    // Sigma^-1 Sigma = I over the whole matrix, formed by mirroring the lower triangle back.
    const arma::mat identity_err_mat =
            arma::symmatl(Sigma_inv_mat_lower.submat(0, 0, n_train - 1, n_train - 1)) * Sigma_mat
            - arma::eye(n_train, n_train);
    const double identity_err = arma::abs(identity_err_mat).max();

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "n_train=%d err=%.1e identity=%.1e upper_intact=%d",
            n_train,
            err,
            identity_err,
            static_cast<int>(upper_ok));

    return report(name, err < kTolAltPath && identity_err < kTolAltPath && upper_ok, detail);
}

// Asserts recursive_split never returns an empty sub-block, and aligns to nb once n is large enough for that.
bool test_split_point(const char* name) {
    const int nb = PERF_RECURSIVE_INV_BLOCK_SIZE;
    int first_failing_n = -1, first_failing_n1 = -1;
    for (int n = 2; n <= 5 * nb; ++n) {
        const int n1 = recursive_split(n, nb);
        const bool is_both_non_empty = n1 > 0 && n1 < n;
        const bool is_aligned = n < 2 * nb || n1 % nb == 0;  // aligned once half of n reaches nb
        if (!(is_both_non_empty && is_aligned) && first_failing_n < 0) {
            first_failing_n = n;
            first_failing_n1 = n1;
        }
    }
    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "nb=%d, n in [2, %d], first failing n=%d n1=%d",
            nb,
            5 * nb,
            first_failing_n,
            first_failing_n1);

    return report(name, first_failing_n < 0, detail);
}

}  // namespace

int main() {
    std::printf("== GpFlash recursive inverse ==\n");
    bool ok = true;
    ok &= test_split_point("split_point");
    // 7 and 128 stay in the dtrtri/dlauum base case; 129 and 333 recurse, 333 with an unaligned remainder.
    for (const int n_train : {7, PERF_RECURSIVE_INV_BLOCK_SIZE, PERF_RECURSIVE_INV_BLOCK_SIZE + 1, 333}) {
        ok &= test_trtri_vs_dense("trtri_vs_dense", n_train);
        ok &= test_lauum_vs_dense("lauum_vs_dense", n_train);
        ok &= test_dpotri_vs_dense("dpotri_vs_dense", n_train);
    }
    std::printf("\n%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");

    return ok ? 0 : 1;
}
