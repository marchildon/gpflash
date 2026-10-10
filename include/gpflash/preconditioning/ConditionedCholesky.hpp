/**
 * ConditionedCholesky.hpp -- conditioning a symmetric positive-semidefinite matrix M with positive diagonal before
 * its Cholesky factorization: scale it to unit diagonal, add a nugget sized to bound the condition number, factorize,
 * and scale the factor back.
 *
 *   P          = diag(sqrt(diag(M)))
 *   M_precon   = P^-1 M P^-1
 *   eta        = eig_max_ub_M_precon / (cond_max - 1),   eig_max_ub_M_precon an upper bound on lambda_max(M_precon)
 *   L_M_nugget = P chol(M_precon + eta I),      so L_M_nugget L_M_nugget^T = M + eta P P
 *
 * M_precon's unit diagonal is what makes a cheap bound on lambda_max available, and lambda_min(M_precon + eta I) >=
 * eta then gives kappa(M_precon + eta I) <= cond_max. Nothing here knows what M is. build_eta_terms returns
 * everything in EtaTerms, with the bound the caller's EigMaxUbMethod selects: Gershgorin's, or the trace n_row.
 */
#pragma once

#include <armadillo>
#include <cmath>
#include <cstddef>  // std::ptrdiff_t

#include "gpflash/core/Errors.hpp"                   // ErrorNumericalFailure, ErrorInvalidInput, check_enum_val
#include "gpflash/core/Platform.hpp"                 // GPFLASH_RESTRICT, the OMP macros and constants, GPFLASH_CHECK
#include "gpflash/inputs/EigMaxUbMethod.hpp"         // EigMaxUbMethod
#include "gpflash/linear_algebra/Cholesky.hpp"       // factorize_chol_inplace
#include "gpflash/preconditioning/EtaTerms.hpp"      // EtaTerms
#include "gpflash/preconditioning/PreconScales.hpp"  // PreconScales

namespace gpflash {
namespace detail {

// ── The preconditioner P ────────────────────────────────────────────────────

// diag(P) for the preconditioner P = diag(sqrt(diag(M))), which is all of P that is ever needed.
inline arma::vec calc_precon_scale_vec(const arma::mat& M_mat_lower, const char* caller) {
    const int n_row = static_cast<int>(M_mat_lower.n_rows);
    arma::vec precon_scale_vec(n_row, arma::fill::none);
    for (int i_diag = 0; i_diag < n_row; ++i_diag) {
        const double diag_val = M_mat_lower(i_diag, i_diag);
        if (!(diag_val > 0.0))
            throw gpflash::ErrorNumericalFailure(
                    error_prefix(caller, "the matrix to precondition")
                    + " has a diagonal entry <= 0, so the preconditioner is singular.");
        precon_scale_vec(i_diag) = std::sqrt(diag_val);
    }
    return precon_scale_vec;
}

// Scale M to unit diagonal, which the bound on lambda_max relies on.
//
// M_mat_lower lower with diag memory: IN: M, OUT: M_precon = P^-1 M P^-1
inline void scale_to_unit_diag_inplace(arma::mat& M_mat_lower, const arma::vec& inv_precon_scale_vec) {
    if (inv_precon_scale_vec.is_empty()) return;
    const double* GPFLASH_RESTRICT inv_precon_scale_ptr = inv_precon_scale_vec.memptr();

    const int n_row = static_cast<int>(M_mat_lower.n_rows);
    for (int i_col = 0; i_col < n_row; ++i_col) {
        double* GPFLASH_RESTRICT col_ptr = M_mat_lower.colptr(i_col);
        const double inv_precon_scale_col = inv_precon_scale_ptr[i_col];
        col_ptr[i_col] = 1.0;
        GPFLASH_OMP_SIMD
        for (int i_row = i_col + 1; i_row < n_row; ++i_row)
            col_ptr[i_row] *= inv_precon_scale_col * inv_precon_scale_ptr[i_row];
    }
}

// Scale M to unit diagonal in place, and return the scales used.
//
// M_mat_lower lower with diag memory: IN: M, OUT: M_precon
inline PreconScales precondition_to_unit_diag_inplace(arma::mat& M_mat_lower, const char* caller) {
    PreconScales precon_scales;
    precon_scales.M_diag_vec = M_mat_lower.diag();  // W, read before the rescale assigns the diagonal 1
    precon_scales.precon_scale_vec = calc_precon_scale_vec(M_mat_lower, caller);
    precon_scales.inv_precon_scale_vec = 1.0 / precon_scales.precon_scale_vec;  // once, so consumers multiply
    scale_to_unit_diag_inplace(M_mat_lower, precon_scales.inv_precon_scale_vec);

    GPFLASH_CHECK(check_precon_scales(precon_scales, caller));
    return precon_scales;
}

// Exact: a unit diagonal is assigned 1.0, never computed near it.
inline void check_unit_diag(const arma::mat& M_mat_lower, const char* caller) {
    if (!arma::all(M_mat_lower.diag() == 1.0))
        throw gpflash::ErrorInvalidInput(
                error_prefix(caller, "the matrix declared unit-diagonal") + " has a diagonal entry other than 1.");
}

// Scale M to unit diagonal in place unless it already has one, and return the scales used.
//
// M_mat_lower lower with diag memory: IN: M, OUT: M_precon
inline PreconScales precondition_unless_unit_diag_inplace(
        const bool is_unit_diag, arma::mat& M_mat_lower, const char* caller) {
    if (is_unit_diag) {
        // A unit diagonal is already M_precon, so M is untouched and the scales come back empty.
        GPFLASH_CHECK(check_unit_diag(M_mat_lower, caller));
        return PreconScales{};
    } else {
        return precondition_to_unit_diag_inplace(M_mat_lower, caller);
    }
}

// ── The Gershgorin bound and its active row ─────────────────────────────────

// Row idx_M_precon_max_abs_row_sum of a symmetric matrix, gathered from the lower triangle that stores it.
inline arma::vec gather_active_row_vec(const arma::mat& M_precon_mat_lower, const int idx_M_precon_max_abs_row_sum) {
    const int n_row = static_cast<int>(M_precon_mat_lower.n_rows);
    arma::vec active_row_M_precon_vec(n_row, arma::fill::zeros);
    if (idx_M_precon_max_abs_row_sum == -1) return active_row_M_precon_vec;

    for (int i_col = 0; i_col < idx_M_precon_max_abs_row_sum; ++i_col)
        active_row_M_precon_vec(i_col) = M_precon_mat_lower(idx_M_precon_max_abs_row_sum, i_col);
    for (int i_row = idx_M_precon_max_abs_row_sum + 1; i_row < n_row; ++i_row)
        active_row_M_precon_vec(i_row) = M_precon_mat_lower(i_row, idx_M_precon_max_abs_row_sum);

    return active_row_M_precon_vec;
}

// Fill Gershgorin's bound on lambda_max(M_precon) and the row attaining it. Read before M_precon is overwritten.
// Fills the bound, its row index and the active row; the preconditioner and the resolved eta are filled elsewhere.
inline void fill_eta_bound_data(
        const arma::mat& M_precon_mat_lower, const bool want_active_row, EtaTerms& eta_terms_out) {
    const int n_row = static_cast<int>(M_precon_mat_lower.n_rows);
    if (n_row <= 1) return;  // already perfectly conditioned: no nugget, no active row

    // Each row's off-diagonal absolute sum. One buffer per thread, allocated before the region opens because an
    // exception escaping an OpenMP block terminates.
    const int n_thread = (n_row >= PERF_OMP_MIN_N_POINT) ? GPFLASH_OMP_MAX_THREADS() : 1;
    arma::vec abs_row_sum_M_precon_thread_vec(static_cast<arma::uword>(n_thread) * n_row, arma::fill::zeros);
    arma::vec abs_row_sum_M_precon_vec(n_row, arma::fill::zeros);
#pragma omp parallel if (n_row >= PERF_OMP_MIN_N_POINT) num_threads(n_thread)
    {
        double* GPFLASH_RESTRICT abs_row_sum_M_precon_local_ptr = abs_row_sum_M_precon_thread_vec.memptr()
                + static_cast<std::ptrdiff_t>(GPFLASH_OMP_THREAD_NUM()) * n_row;
#pragma omp for schedule(dynamic, PERF_OMP_ROW_SUM_CHUNK) nowait
        for (int i_col = 0; i_col < n_row; ++i_col) {
            const double* GPFLASH_RESTRICT col_ptr = M_precon_mat_lower.colptr(i_col);
            double col_sum = 0.0;

            GPFLASH_OMP_SIMD_REDUCTION(+ : col_sum)
            for (int i_row = i_col + 1; i_row < n_row; ++i_row) {
                const double abs_M_precon_ij = std::fabs(col_ptr[i_row]);
                col_sum += abs_M_precon_ij;
                abs_row_sum_M_precon_local_ptr[i_row] += abs_M_precon_ij;
            }
            abs_row_sum_M_precon_local_ptr[i_col] += col_sum;
        }
    }
    for (int i_thread = 0; i_thread < n_thread; ++i_thread) {
        const double* GPFLASH_RESTRICT thread_ptr =
                abs_row_sum_M_precon_thread_vec.memptr() + static_cast<std::ptrdiff_t>(i_thread) * n_row;
        for (int i_row = 0; i_row < n_row; ++i_row) abs_row_sum_M_precon_vec(i_row) += thread_ptr[i_row];
    }

    // The bound is max_i (M_ii + sum_{j != i} |M_ij|). Every M_ii of M_precon is 1, so it factors out of the
    // max and the sums above skip the diagonal.
    constexpr double M_precon_diag = 1.0;
    const int idx_M_precon_max_abs_row_sum = static_cast<int>(abs_row_sum_M_precon_vec.index_max());
    eta_terms_out.idx_M_precon_max_abs_row_sum = idx_M_precon_max_abs_row_sum;
    eta_terms_out.eig_max_ub_M_precon = M_precon_diag + abs_row_sum_M_precon_vec(idx_M_precon_max_abs_row_sum);

    // One gather, for the hyperparameter gradient alone.
    if (want_active_row)
        eta_terms_out.active_row_M_precon_vec = gather_active_row_vec(M_precon_mat_lower, idx_M_precon_max_abs_row_sum);
}

// ── The nugget ───────────────────────────────────────────────────────────────

// The nugget from an upper bound on lambda_max(M_precon).
inline double calc_eta(const double eig_max_ub_M_precon, const double cond_max) {
    return eig_max_ub_M_precon / (cond_max - 1.0);
}

// M_precon_mat_lower lower with diag memory: IN: M_precon, OUT: M_precon + eta I
inline void add_nugget_to_diag_inplace(arma::mat& M_precon_mat_lower, const double eta) {
    M_precon_mat_lower.diag() += eta;
}

// Everything needed to use and differentiate eta for this evaluation.
// Returns the eta data but also preconditions its argument, so the parameter name carries the mutation:
// M_in_M_precon_out_mat_lower, lower with diag, IN M, OUT M_precon.
inline EtaTerms build_eta_terms(
        const bool is_unit_diag,
        arma::mat& M_in_M_precon_out_mat_lower,
        const double cond_max,
        const EigMaxUbMethod eig_max_ub_method,
        const bool want_active_row) {
    // Before M is preconditioned, so a rejected method leaves it untouched.
    check_enum_val(eig_max_ub_method, "EigMaxUbMethod", "eig_max_ub_method", "build_eta_terms");

    EtaTerms eta_terms;
    eta_terms.precon_scales =
            precondition_unless_unit_diag_inplace(is_unit_diag, M_in_M_precon_out_mat_lower, "build_eta_terms");

    // The bound on lambda_max(M_precon). Only Gershgorin's reads the matrix and has an active row.
    switch (eig_max_ub_method) {
        case EigMaxUbMethod::Gershgorin:
            fill_eta_bound_data(M_in_M_precon_out_mat_lower, want_active_row, eta_terms);
            break;
        case EigMaxUbMethod::Trace: {
            // tr(M_precon) = n_row, constant, so no row attains it. n_row <= 1 is already perfectly conditioned.
            const int n_row = static_cast<int>(M_in_M_precon_out_mat_lower.n_rows);
            if (n_row > 1) eta_terms.eig_max_ub_M_precon = static_cast<double>(n_row);
            break;
        }
        case EigMaxUbMethod::End:
            break;  // rejected by check_enum_val above
    }

    eta_terms.cond_max = cond_max;
    eta_terms.eta = calc_eta(eta_terms.eig_max_ub_M_precon, cond_max);

    GPFLASH_CHECK(check_eta_terms(eta_terms, "build_eta_terms"));
    return eta_terms;
}

// ── Undoing the preconditioning in the factor ───────────────────────────────

// Scale row i of chol(M_precon) by diag(P)_i, turning it into chol(M). M here includes whatever the caller added after
// preconditioning: the nugget, or Sigma's noise. An empty scale is P = I.
//
// L_M_precon_in_L_M_out_mat_lower with diag memory: IN chol(M_precon), OUT chol(M) = P chol(M_precon).
inline void calc_L_M_precon_to_L_M_inplace(
        arma::mat& L_M_precon_in_L_M_out_mat_lower, const arma::vec& precon_scale_vec) {
    if (precon_scale_vec.is_empty()) return;
    const double* GPFLASH_RESTRICT precon_scale_ptr = precon_scale_vec.memptr();
    const int n_row = static_cast<int>(L_M_precon_in_L_M_out_mat_lower.n_rows);
    for (int i_col = 0; i_col < n_row; ++i_col) {
        double* GPFLASH_RESTRICT col_ptr = L_M_precon_in_L_M_out_mat_lower.colptr(i_col);
        for (int i_row = i_col; i_row < n_row; ++i_row) col_ptr[i_row] *= precon_scale_ptr[i_row];
    }
}

// Factorize, then undo the preconditioning the factor inherited.
//
// M_precon_in_L_M_out_mat_lower with diag memory: IN M_precon, its nugget included, OUT chol(M).
inline void factorize_M_precon_to_L_M_inplace(
        arma::mat& M_precon_in_L_M_out_mat_lower, const PreconScales& precon_scales, const char* blas_call_site) {
    factorize_chol_inplace(M_precon_in_L_M_out_mat_lower, blas_call_site);
    calc_L_M_precon_to_L_M_inplace(M_precon_in_L_M_out_mat_lower, precon_scales.precon_scale_vec);
}

}  // namespace detail
}  // namespace gpflash
