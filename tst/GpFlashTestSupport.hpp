#pragma once

#include <armadillo>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>

#include "gpflash/core/Errors.hpp"            // gpflash::ErrorInvalidInput, for rejects below
#include "gpflash/core/Platform.hpp"          // gpflash::PERF_OMP_MIN_N_POINT
#include "gpflash/inputs/EigMaxUbMethod.hpp"  // EigMaxUbMethod, for the mean model nugget references

namespace gpflash_test {

// ── Agreement tolerances ────────────────────────────────────────────────────
constexpr double kTolExact = 1e-13;         // the same arithmetic in a different summation order
constexpr double kTolAltPath = 1e-11;       // one quantity by two paths, each free to reassociate or use FMA
constexpr double kTolReferenceImpl = 1e-9;  // the library against an independent brute-force reimplementation

// ── Finite differences ──────────────────────────────────────────────────────
//
//   kernel       f is a closed-form kernel value, noisy only at machine precision
//   objective    f is the MLL, carrying the noise of an O(n_train^3) factorization, so a larger step wins
//   oscillatory  the periodic MLL, still roundoff-limited at the objective step, so larger again
//
// Each step is the largest of its regime that stays truncation-free, established by shrinking it until the
// agreement stops improving. A step smaller than its regime's makes agreement worse, not better.
constexpr double kFdStepKern = 1e-6;    // kernel: step, in a hyperparameter, coordinate or squared distance
constexpr double kTolFd = 1e-7;         // kernel: tolerance
constexpr double kFdStepObj = 1e-4;     // objective: step
constexpr double kTolFdObj = 1e-6;      // objective: tolerance
constexpr double kFdStepObjOsc = 1e-3;  // oscillatory: step
constexpr double kTolFdObjOsc = 1e-5;   // oscillatory: tolerance

// Denominator guards for a relative comparison, not agreement tolerances.
constexpr double kFdScaleFloor = 1e-7;     // a difference smaller than this is itself noise
constexpr double kRelScaleFloor = 1e-300;  // avoids dividing by a genuinely zero reference

// kappa is itself computed from a factorization, so a conditioning bound is checked with relative
// slack rather than exactly.
constexpr double kCondBoundSlack = 1e-8;
// A cond_max low enough that the mean model nugget moves any result with more than one mean model weight past the
// test tolerances.
constexpr double kCondMaxStrongNugget = 1e3;

// ── Shared inputs ───────────────────────────────────────────────────────────
constexpr int kDetailLen = 128;                 // report() detail buffer
constexpr double kRatQuadHpRho = 1.7;           // non-unit: hp_rho == 1 collapses the s^(k) powers
constexpr double kRatQuadHpRhoNearZero = 1e-3;  // where 1/(2 hp_rho) is largest
constexpr double kHpRhoUnused = 0.0;            // a kernel with no extra hyperparameters never reads it

// Above PERF_OMP_MIN_N_POINT, so the OpenMP column sweep runs instead of its serial fallback.
constexpr int kNtrainAboveOmpThresholds = 300;
static_assert(kNtrainAboveOmpThresholds > gpflash::PERF_OMP_MIN_N_POINT, "OpenMP column sweep never exercised.");

// ── Portable random draws ───────────────────────────────────────────────────
// std::mt19937_64's output sequence is fixed by the standard; std::*_distribution's is not, and libc++, libstdc++
// and MSVC differ. Test data is therefore drawn from the raw generator.

// Uniform on [0, 1): the top 53 bits of one draw, divided by 2^53. Exact, so identical on every platform.
inline double draw_uniform_unit(std::mt19937_64& rng) {
    return static_cast<double>(static_cast<std::uint64_t>(rng()) >> 11) / 9007199254740992.0;
}

// Uniform on [lo, hi).
inline double draw_uniform(std::mt19937_64& rng, const double lo, const double hi) {
    return lo + (hi - lo) * draw_uniform_unit(rng);
}

// Standard normal by Box-Muller, cosine branch only, so every call takes exactly two draws whatever came before.
// The draws are separate statements because argument evaluation order is unspecified. Identical to within the
// last ulp across platforms: std::log and std::cos are not correctly rounded.
inline double draw_normal(std::mt19937_64& rng) {
    const double u_radius = 1.0 - draw_uniform_unit(rng);  // (0, 1], so the log is finite
    const double u_angle = draw_uniform_unit(rng);
    return std::sqrt(-2.0 * std::log(u_radius)) * std::cos(2.0 * arma::datum::pi * u_angle);
}

// Portable stand-ins for arma::randu and arma::randn, whose streams change with the Armadillo version. Filled in
// column-major order.
inline arma::mat draw_uniform_mat(
        std::mt19937_64& rng,
        const arma::uword n_rows,
        const arma::uword n_cols,
        const double lo = 0.0,
        const double hi = 1.0) {
    arma::mat A_mat(n_rows, n_cols);
    for (double& a_val : A_mat) a_val = draw_uniform(rng, lo, hi);
    return A_mat;
}

inline arma::vec draw_uniform_vec(
        std::mt19937_64& rng, const arma::uword n_elem, const double lo = 0.0, const double hi = 1.0) {
    arma::vec a_vec(n_elem);
    for (double& a_val : a_vec) a_val = draw_uniform(rng, lo, hi);
    return a_vec;
}

inline arma::vec draw_normal_vec(std::mt19937_64& rng, const arma::uword n_elem) {
    arma::vec a_vec(n_elem);
    for (double& a_val : a_vec) a_val = draw_normal(rng);
    return a_vec;
}

inline bool report(const char* name, bool pass, const char* detail = "") {
    std::printf("  %-42s %s%s%s\n", name, pass ? "PASS" : "*** FAIL ***", *detail ? "  " : "", detail);
    return pass;
}

// ── Checker assertions ──────────────────────────────────────────────────────

// Whether function threw ErrorInvalidInput specifically. Any other exception counts as a failure: the type is what
// tells a caller whether retrying at a different point could help.
template <class Function>
bool rejects(Function&& function) {
    try {
        function();
    } catch (const gpflash::ErrorInvalidInput&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

// Whether function returns without throwing, for the accept half of each rule.
template <class Function>
bool accepts(Function&& function) {
    try {
        function();
    } catch (...) {
        return false;
    }
    return true;
}

// ── In-place lower-triangular checks ────────────────────────────────────────
// Written into a strict upper triangle before an in-place operation on the lower one; any stray write above
// the diagonal then changes it, which is_strict_upper_intact detects.
constexpr double kUpperSentinel = -1234.5;

inline bool is_strict_upper_intact(const arma::mat& A_mat) {
    const int n_train = static_cast<int>(A_mat.n_rows);
    for (int i_col = 1; i_col < n_train; ++i_col)
        for (int i_row = 0; i_row < i_col; ++i_row)
            if (A_mat(i_row, i_col) != kUpperSentinel) return false;

    return true;
}

// Largest absolute discrepancy over the lower triangle and diagonal -- the only region an in-place
// lower-triangular routine defines -- normalized by the largest reference magnitude there. One global ratio,
// not a per-entry relative error.
inline double lower_rel_err(const arma::mat& actual_mat, const arma::mat& expected_mat) {
    const int n_train = static_cast<int>(actual_mat.n_rows);
    double max_err = 0.0, max_ref = kRelScaleFloor;
    for (int i_col = 0; i_col < n_train; ++i_col)
        for (int i_row = i_col; i_row < n_train; ++i_row) {
            max_err = std::max(max_err, std::fabs(actual_mat(i_row, i_col) - expected_mat(i_row, i_col)));
            max_ref = std::max(max_ref, std::fabs(expected_mat(i_row, i_col)));
        }
    return max_err / max_ref;
}

// Same ratio as lower_rel_err, over every entry.
inline double rel_err(const arma::mat& actual_mat, const arma::mat& expected_mat) {
    return arma::abs(actual_mat - expected_mat).max() / std::max(arma::abs(expected_mat).max(), kRelScaleFloor);
}

// Deterministic, so a failure reproduces from the source alone. The coefficients only give a
// non-degenerate spread: distinct points, varied pairwise distances, away from zero and the kernel tail.
inline arma::mat make_points(const int n_dim, const int n_train) {
    arma::mat X_ndim_ntrain_mat(n_dim, n_train);
    for (int i_train = 0; i_train < n_train; ++i_train)
        for (int i_dim = 0; i_dim < n_dim; ++i_dim)
            X_ndim_ntrain_mat(i_dim, i_train) =
                    std::sin(0.7 * i_train + 1.3 * i_dim) + 0.25 * std::cos(2.1 * i_train - 0.4 * i_dim);
    return X_ndim_ntrain_mat;
}

// The mean model basis at X_ndim_npoint_mat from its definition: the constant if has_mean_model_constant, then
// x_d^degree_d for each non-zero degree. One degree per dimension, never broadcast.
inline arma::mat build_mean_model_basis_ref_mat(
        const arma::mat& X_ndim_npoint_mat,
        const bool has_mean_model_constant,
        const arma::uvec& mean_model_degree_vec) {
    const arma::uword n_point = X_ndim_npoint_mat.n_cols;
    arma::mat Basis_npoint_nbasis_mat(n_point, 0);
    if (has_mean_model_constant) Basis_npoint_nbasis_mat.ones(n_point, 1);
    for (arma::uword i_dim = 0; i_dim < mean_model_degree_vec.n_elem; ++i_dim) {
        if (mean_model_degree_vec(i_dim) == 0) continue;
        const double degree = static_cast<double>(mean_model_degree_vec(i_dim));
        const arma::vec x_power_vec = arma::pow(X_ndim_npoint_mat.row(i_dim).t(), degree);
        Basis_npoint_nbasis_mat = arma::join_horiz(Basis_npoint_nbasis_mat, x_power_vec);
    }
    return Basis_npoint_nbasis_mat;
}

// The mean model nugget on the preconditioned mean model Gram matrix, n_basis / (cond_max - 1) for more than one weight
// and zero otherwise, stated here rather than read from the library.
inline double calc_mean_model_eta_ref(const arma::uword n_basis, const double cond_max) {
    double mean_model_eta_ref = 0.0;
    if (n_basis > 1) mean_model_eta_ref = static_cast<double>(n_basis) / (cond_max - 1.0);

    return mean_model_eta_ref;
}

// The mean model nugget for the given bound on the preconditioned mean model Gram matrix, zero for one weight or none.
// Gershgorin's is the largest full absolute row sum, unit diagonal included, formed densely rather than by the
// library's sweep over the lower triangle.
inline double calc_mean_model_eta_ref(
        const arma::mat& mean_model_gram_mat, const double cond_max, const gpflash::EigMaxUbMethod eig_max_ub_method) {
    double mean_model_eta_ref = calc_mean_model_eta_ref(mean_model_gram_mat.n_cols, cond_max);
    if (eig_max_ub_method == gpflash::EigMaxUbMethod::Gershgorin && mean_model_gram_mat.n_cols > 1) {
        const arma::vec inv_scale_vec = 1.0 / arma::sqrt(mean_model_gram_mat.diag());
        const arma::mat mean_model_gram_precon_mat = mean_model_gram_mat % (inv_scale_vec * inv_scale_vec.t());
        const double eig_max_ub_ref = arma::sum(arma::abs(mean_model_gram_precon_mat), 1).max();
        mean_model_eta_ref = eig_max_ub_ref / (cond_max - 1.0);
    }

    return mean_model_eta_ref;
}

// The mean model weights (Z + mean_model_eta diag(Z))^-1 Basis^T Sigma^-1 f_train, Z = Basis^T Sigma^-1 Basis, from the
// caller's Sigma^-1 Basis and Sigma^-1 f_train, and empty for a basis with no columns. A dense equilibrated solve, not
// the library's preconditioned Cholesky.
inline arma::vec calc_hp_beta_ref_vec(
        const arma::mat& Basis_ntrain_nbasis_mat,
        const arma::mat& lin_sol_sigma_Basis_ntrain_nbasis_mat,
        const arma::vec& lin_sol_sigma_f_train_vec,
        const double cond_max,
        const gpflash::EigMaxUbMethod eig_max_ub_method = gpflash::EigMaxUbMethod::Trace) {
    arma::vec hp_beta_ref_vec(Basis_ntrain_nbasis_mat.n_cols);

    if (Basis_ntrain_nbasis_mat.n_cols > 0) {
        arma::mat mean_model_gram_mat = Basis_ntrain_nbasis_mat.t() * lin_sol_sigma_Basis_ntrain_nbasis_mat;
        mean_model_gram_mat.diag() *= 1.0 + calc_mean_model_eta_ref(mean_model_gram_mat, cond_max, eig_max_ub_method);
        const arma::vec Basis_sigma_inv_f_train_vec = Basis_ntrain_nbasis_mat.t() * lin_sol_sigma_f_train_vec;
        hp_beta_ref_vec = arma::solve(mean_model_gram_mat, Basis_sigma_inv_f_train_vec, arma::solve_opts::equilibrate);
    }

    return hp_beta_ref_vec;
}

// A well-conditioned SPD matrix. The n_train shift dominates the Gram term, so the factorizations and dense
// inverses the tests compare against carry no error of their own.
inline arma::mat make_spd_matrix(const int n_train) {
    const arma::mat A_mat = make_points(n_train, n_train);
    return A_mat.t() * A_mat + n_train * arma::eye(n_train, n_train);
}

// The c^(rdot) values build_kmat_lower_tri left in the strict upper triangle, for one row of K, gathered the
// way the MLL gradient loop gathers them: column i_col's run lives in upper column n_train - 1 - i_col, offset one
// past the diagonal. A hand transcription of that layout, independent of the library's own sweep, so a test
// can use it either as the expected value or as an input.
inline arma::vec harvest_active_row_kern_common_rdot(
        const arma::mat& kern_common_rdot_mat_upper, const int idx_active_row) {
    const int n_train = static_cast<int>(kern_common_rdot_mat_upper.n_rows);
    arma::vec active_row_vec(n_train, arma::fill::zeros);
    for (int i_col = 0; i_col < n_train; ++i_col) {
        const int n_below = n_train - i_col - 1;
        if (n_below <= 0) continue;
        const double* kern_common_rdot_col_ptr =
                kern_common_rdot_mat_upper.memptr() + static_cast<std::size_t>(n_train - 1 - i_col) * n_train;
        if (i_col < idx_active_row)
            active_row_vec(i_col) = kern_common_rdot_col_ptr[idx_active_row - i_col - 1];
        else if (i_col == idx_active_row)
            for (int i_below = 0; i_below < n_below; ++i_below)
                active_row_vec(i_col + 1 + i_below) = kern_common_rdot_col_ptr[i_below];
    }
    return active_row_vec;
}

}  // namespace gpflash_test
