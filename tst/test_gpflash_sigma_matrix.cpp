// The last two assembly steps, K_corr -> the matrix to factorize -> L. What this file asserts:
//
//   1) sigma_with_noise         Sigma_corr = hp_sig2_k (K_corr + eta I) + sigma_f^2 I
//   2) nugget_only              add_nugget_to_diag_inplace yields K_corr + eta I, applying no hp_sig2_k
//   3) sigma_preconditioned     a non-unit diag(P) puts the noise in as sigma_f^2 / p_i^2
//   4) chol_reproduces_sigma    L L^T equals what build_sigma_corr_inplace produced
//   5) chol_rejects_indefinite  factorize_chol_inplace throws ErrorNumericalFailure, not runtime_error, naming the
//                               call site, the size and the refused pivot
//   6) psi                      Psi = (v_res v_res^T - Sigma^-1) / 2, entrywise and along the diagonal
//
// Checks 1 to 4 also assert strict_upper_intact, the invariant the hyperparameter gradient depends on:
// assembly leaves c^(rdot) in the strict upper triangle of this same buffer, and dpotrf must not touch it.
#include <armadillo>
#include <cmath>
#include <cstdio>
#include <string>

#include "GpFlashTestSupport.hpp"
#include "gpflash/assembly/SigmaGrad.hpp"
#include "gpflash/assembly/SigmaMatrix.hpp"
#include "gpflash/core/Errors.hpp"
#include "gpflash/linear_algebra/Cholesky.hpp"
#include "gpflash/preconditioning/ConditionedCholesky.hpp"

using gpflash::detail::add_nugget_to_diag_inplace;
using gpflash::detail::build_sigma_corr_inplace;
using gpflash::detail::factorize_chol_inplace;
using gpflash_test::is_strict_upper_intact;
using gpflash_test::kDetailLen;
using gpflash_test::kRelScaleFloor;
using gpflash_test::kTolAltPath;
using gpflash_test::kTolExact;
using gpflash_test::kUpperSentinel;
using gpflash_test::lower_rel_err;
using gpflash_test::make_points;
using gpflash_test::report;

namespace {

// A well-conditioned K in the lower triangle, with a sentinel above the diagonal standing in for c^(rdot).
arma::mat make_kmat_with_sentinel(const int n_train) {
    const arma::mat A_mat = make_points(n_train, n_train);
    const arma::mat K_full_mat = A_mat.t() * A_mat + n_train * arma::eye(n_train, n_train);
    arma::mat K_mat(n_train, n_train, arma::fill::none);
    for (int i_col = 0; i_col < n_train; ++i_col) {
        for (int i_row = 0; i_row < i_col; ++i_row) K_mat(i_row, i_col) = kUpperSentinel;
        for (int i_row = i_col; i_row < n_train; ++i_row) K_mat(i_row, i_col) = K_full_mat(i_row, i_col);
    }
    return K_mat;
}

// Asserts build_sigma_corr_inplace matches a dense hp_sig2_k (K + eta I) + sigma_f^2 I, and leaves c^(rdot)
// alone.
//
// The reference scales the whole matrix and adds both terms to the diagonal in one expression, sharing no step
// with the two in-place passes under test.
bool test_sigma_with_noise(const char* name, const int n_train) {
    const double eta = 0.037, sig2_f_val = 0.021, hp_sig2_k = 2.7;
    const arma::mat K_mat = make_kmat_with_sentinel(n_train);

    arma::mat Sigma_expected_mat = hp_sig2_k * K_mat;
    Sigma_expected_mat.diag() += hp_sig2_k * eta + sig2_f_val;

    arma::mat Sigma_mat = K_mat;
    build_sigma_corr_inplace(Sigma_mat, eta, sig2_f_val, hp_sig2_k, arma::vec());

    const double err = lower_rel_err(Sigma_mat, Sigma_expected_mat);
    const bool upper_ok = is_strict_upper_intact(Sigma_mat);
    char detail[kDetailLen];
    std::snprintf(
            detail, sizeof(detail), "n_train=%d err=%.1e upper_intact=%d", n_train, err, static_cast<int>(upper_ok));

    return report(name, err < kTolExact && upper_ok, detail);
}

// Asserts add_nugget_to_diag_inplace adds eta and nothing else, producing K_corr + eta I.
//
// This is the form the fit builds before hp_sig2_k is known, so the matrix that comes out is Sigma_corr
// divided by it. The reference below carries no hp_sig2_k at all, so calling build_sigma_corr_inplace here
// instead, with any hp_sig2_k other than one, fails the check.
bool test_nugget_only(const char* name, const int n_train) {
    const double eta = 0.037;
    const arma::mat K_mat = make_kmat_with_sentinel(n_train);

    arma::mat Sigma_expected_mat = K_mat;
    Sigma_expected_mat.diag() += eta;

    arma::mat Sigma_mat = K_mat;
    add_nugget_to_diag_inplace(Sigma_mat, eta);

    const double err = lower_rel_err(Sigma_mat, Sigma_expected_mat);
    const bool upper_ok = is_strict_upper_intact(Sigma_mat);
    char detail[kDetailLen];
    std::snprintf(
            detail, sizeof(detail), "n_train=%d err=%.1e upper_intact=%d", n_train, err, static_cast<int>(upper_ok));

    return report(name, err < kTolExact && upper_ok, detail);
}

// Asserts a non-unit preconditioner puts the noise in as sigma_f^2 / p_i^2 per point.
//
// The nugget is measured in preconditioned coordinates so it is the same for every point, while sigma_f^2
// belongs to Sigma and so picks up the row scaling.
bool test_sigma_preconditioned(const char* name, const int n_train) {
    const double eta = 0.037, sig2_f_val = 0.021, hp_sig2_k = 2.7;
    const arma::mat K_mat = make_kmat_with_sentinel(n_train);

    arma::vec precon_scale_vec(n_train);
    for (int i_diag = 0; i_diag < n_train; ++i_diag) precon_scale_vec(i_diag) = 0.5 + 0.25 * i_diag;
    const arma::vec inv_precon_scale_vec = 1.0 / precon_scale_vec;

    arma::mat Sigma_expected_mat = hp_sig2_k * K_mat;
    for (int i_diag = 0; i_diag < n_train; ++i_diag)
        Sigma_expected_mat(i_diag, i_diag) +=
                hp_sig2_k * eta + sig2_f_val / (precon_scale_vec(i_diag) * precon_scale_vec(i_diag));

    arma::mat Sigma_mat = K_mat;
    build_sigma_corr_inplace(Sigma_mat, eta, sig2_f_val, hp_sig2_k, inv_precon_scale_vec);

    const double err = lower_rel_err(Sigma_mat, Sigma_expected_mat);
    const bool upper_ok = is_strict_upper_intact(Sigma_mat);
    char detail[kDetailLen];
    std::snprintf(
            detail, sizeof(detail), "n_train=%d err=%.1e upper_intact=%d", n_train, err, static_cast<int>(upper_ok));

    return report(name, err < kTolExact && upper_ok, detail);
}

// Asserts factorize_chol_inplace produces a factor whose product is the matrix it replaced.
bool test_chol_reproduces_sigma(const char* name, const int n_train) {
    const double eta = 0.037, sig2_f_val = 0.021, hp_sig2_k = 2.7;
    arma::mat Sigma_mat = make_kmat_with_sentinel(n_train);
    build_sigma_corr_inplace(Sigma_mat, eta, sig2_f_val, hp_sig2_k, arma::vec());
    const arma::mat Sigma_before_mat = Sigma_mat;

    factorize_chol_inplace(Sigma_mat, "test_gpflash_sigma_matrix");

    const arma::mat product_mat = arma::trimatl(Sigma_mat) * arma::trimatl(Sigma_mat).t();
    const double err = lower_rel_err(product_mat, arma::symmatl(Sigma_before_mat));
    const bool upper_ok = is_strict_upper_intact(Sigma_mat);
    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "n_train=%d ||L L^T - Sigma||=%.1e upper_intact=%d",
            n_train,
            err,
            static_cast<int>(upper_ok));

    return report(name, err < kTolAltPath && upper_ok, detail);
}

// Asserts an indefinite matrix raises ErrorNumericalFailure, not a bare runtime_error: run_multistart records
// that type as one failed start, and raises any other as a defect.
bool test_chol_rejects_indefinite(const char* name) {
    const int n_train = 6;
    arma::mat Sigma_mat(n_train, n_train, arma::fill::zeros);
    Sigma_mat.diag().ones();
    const int i_negative_pivot = 3;
    Sigma_mat(i_negative_pivot, i_negative_pivot) = -1.0;  // one negative pivot is enough for dpotrf to refuse
    const char* blas_call_site = "test_gpflash_sigma_matrix";
    // The message names the caller's call site, the size, and dpotrf's info: the 1-based index of the refused pivot.
    char expected_message[kDetailLen];
    std::snprintf(
            expected_message,
            sizeof(expected_message),
            "%s (n=%d) failed (info=%d)",
            blas_call_site,
            n_train,
            i_negative_pivot + 1);

    bool threw_numerical = false, threw_other = false, has_expected_message = false;
    try {
        factorize_chol_inplace(Sigma_mat, blas_call_site);
    } catch (const gpflash::ErrorNumericalFailure& error) {
        threw_numerical = true;
        has_expected_message = std::string(error.what()).find(expected_message) != std::string::npos;
    } catch (...) {
        threw_other = true;
    }
    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "ErrorNumericalFailure=%d other_exception=%d expected_message=%d",
            static_cast<int>(threw_numerical),
            static_cast<int>(threw_other),
            static_cast<int>(has_expected_message));

    return report(name, threw_numerical && !threw_other && has_expected_message, detail);
}

// Asserts calc_psi_val is Psi_ij entry by entry, and that calc_psi_diag_vec agrees with it on the diagonal.
//
// The two are separate implementations of the same expression -- one scalar, one vectorized -- so the
// diagonal is where they can be held against each other.
bool test_psi(const char* name, const int n_train) {
    const arma::mat A_mat = make_points(n_train, n_train);
    const arma::mat Sigma_inv_mat = arma::symmatl(A_mat.t() * A_mat + n_train * arma::eye(n_train, n_train));
    const arma::vec lin_sol_vec = make_points(n_train, 1);

    const arma::mat Psi_expected_mat = 0.5 * (lin_sol_vec * lin_sol_vec.t() - Sigma_inv_mat);
    double entry_err = 0.0;
    for (int i_col = 0; i_col < n_train; ++i_col)
        for (int i_row = 0; i_row < n_train; ++i_row) {
            const double psi_val =
                    gpflash::detail::calc_psi_val(lin_sol_vec(i_row), lin_sol_vec(i_col), Sigma_inv_mat(i_row, i_col));
            entry_err = std::max(entry_err, std::fabs(psi_val - Psi_expected_mat(i_row, i_col)));
        }
    entry_err /= std::max(arma::abs(Psi_expected_mat).max(), kRelScaleFloor);

    const arma::vec psi_diag_vec = gpflash::detail::calc_psi_diag_vec(lin_sol_vec, Sigma_inv_mat);
    const double diag_err = arma::abs(psi_diag_vec - Psi_expected_mat.diag()).max()
            / std::max(arma::abs(Psi_expected_mat.diag()).max(), kRelScaleFloor);

    char detail[kDetailLen];
    std::snprintf(detail, sizeof(detail), "n_train=%d entry=%.1e diag=%.1e", n_train, entry_err, diag_err);

    return report(name, entry_err < kTolExact && diag_err < kTolExact, detail);
}

}  // namespace

int main() {
    std::printf("== GpFlash Sigma assembly ==\n");
    bool ok = true;
    for (const int n_train : {5, 40}) {
        ok &= test_sigma_with_noise("sigma_with_noise", n_train);
        ok &= test_nugget_only("nugget_only", n_train);
        ok &= test_sigma_preconditioned("sigma_preconditioned", n_train);
        ok &= test_chol_reproduces_sigma("chol_reproduces_sigma", n_train);
        ok &= test_psi("psi", n_train);
    }
    ok &= test_chol_rejects_indefinite("chol_rejects_indefinite");
    std::printf("\n%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");

    return ok ? 0 : 1;
}
