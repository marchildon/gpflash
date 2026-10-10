// The linear_algebra tier: the dense Cholesky solves and inner products the fit and the posterior share.
// What this file asserts:
//
//   1) forward_solves                L^-1 b and L^-1 B equal a dense triangular solve
//   2) inplace_solves                all four in-place solves match, over both directions and both BLAS paths
//   3) full_solve                    a forward solve followed by solve_chol_transpose_vec_inplace gives Sigma^-1 b
//   4) inner_products                the DGEMV and DGEMM wrappers equal a dense transposed product
//   5) factorize_and_ln_det          L L^T reproduces the matrix, and ln|A| matches a dense log determinant
//   6) beta_and_lin_sol_residual     the mean model weights hp_beta_vec and the residual L^-1 (f - Basis hp_beta_vec)
//   7) variance_hp_grad              Psi contracted against each of the two variance derivatives of Sigma
//   8) duplicate_basis_columns_split_weight
//                                    identical basis columns split their weight evenly under the nugget
//   9) underflowing_basis_rejected   a nonzero basis whose mean model Gram matrix underflows raises, for one weight
//                                    and for two
#include <armadillo>
#include <cmath>
#include <cstdio>

#include "GpFlashTestSupport.hpp"
#include "gpflash/assembly/SigmaGrad.hpp"
#include "gpflash/core/Errors.hpp"          // ErrorNumericalFailure
#include "gpflash/inputs/ModelOptions.hpp"  // COND_MAX_DEFAULT
#include "gpflash/linear_algebra/Cholesky.hpp"
#include "gpflash/linear_algebra/InnerProducts.hpp"
#include "gpflash/mean/MeanModelWeights.hpp"

using gpflash::detail::calc_ln_det_from_chol;
using gpflash::detail::factorize_chol_inplace;
using gpflash::detail::fill_inner_prod_mat;
using gpflash::detail::fill_inner_prod_vec;
using gpflash::detail::fill_inner_prod_vec_simd;
using gpflash::detail::solve_chol_forward_mat;
using gpflash::detail::solve_chol_forward_mat_inplace;
using gpflash::detail::solve_chol_forward_vec;
using gpflash::detail::solve_chol_forward_vec_inplace;
using gpflash::detail::solve_chol_transpose_mat_inplace;
using gpflash::detail::solve_chol_transpose_vec_inplace;
using gpflash_test::kCondMaxStrongNugget;
using gpflash_test::kDetailLen;
using gpflash_test::kRelScaleFloor;
using gpflash_test::kTolAltPath;
using gpflash_test::make_points;
using gpflash_test::make_spd_matrix;
using gpflash_test::rel_err;
using gpflash_test::report;

namespace {

// A well-conditioned Sigma and its lower Cholesky factor, so the reference solves carry no error of their own.
struct CholCase {
    arma::mat Sigma_mat;
    arma::mat L_mat_lower;
};

CholCase make_chol_case(const int n_train) {
    CholCase one_case;
    one_case.Sigma_mat = make_spd_matrix(n_train);
    one_case.L_mat_lower = arma::chol(one_case.Sigma_mat, "lower");

    return one_case;
}

// Asserts L^-1 b and L^-1 B equal a dense triangular solve.
bool test_forward_solves(const char* name, const int n_train) {
    const CholCase one_case = make_chol_case(n_train);
    const arma::vec b_vec = make_points(n_train, 1);
    const arma::mat B_mat = make_points(n_train, 3);

    const arma::vec lin_sol_vec = solve_chol_forward_vec(one_case.L_mat_lower, b_vec);
    const arma::mat lin_sol_mat = solve_chol_forward_mat(one_case.L_mat_lower, B_mat);

    const arma::mat L_dense_mat = arma::trimatl(one_case.L_mat_lower);
    const double vec_err = rel_err(lin_sol_vec, arma::solve(L_dense_mat, b_vec));
    const double mat_err = rel_err(lin_sol_mat, arma::solve(L_dense_mat, B_mat));

    char detail[kDetailLen];
    std::snprintf(detail, sizeof(detail), "n_train=%d vec=%.1e mat=%.1e", n_train, vec_err, mat_err);

    return report(name, vec_err < kTolAltPath && mat_err < kTolAltPath, detail);
}

// Asserts calc_trace_psi, and the two variance derivatives built from it, each against its own definition.
//
// Sigma^-1 and v_res are built here from a dense inverse rather than from the in-place solve that produces them
// in the fit, so the references share no step with the routine under test. hp_sig2_k is supplied rather than
// solved, so d(MLL)/d(sigma_k^2) is not zero by construction and both references are the plain definitions.
bool test_variance_hp_grad(const char* name, const int n_train) {
    const CholCase one_case = make_chol_case(n_train);
    const arma::vec f_train_vec = make_points(n_train, 1) + 2.0;
    const double hp_sig2_k = 1.7, sig2_f_val = 0.25;

    const arma::mat Sigma_inv_mat = arma::inv_sympd(arma::symmatl(one_case.Sigma_mat));
    const arma::vec ones_vec(n_train, arma::fill::ones);
    const double hp_beta =
            arma::dot(ones_vec, Sigma_inv_mat * f_train_vec) / arma::dot(ones_vec, Sigma_inv_mat * ones_vec);
    const arma::vec residual_vec = f_train_vec - hp_beta * ones_vec;
    const arma::vec lin_sol_sigma_residual_vec = Sigma_inv_mat * residual_vec;
    const double residual_quad_form = arma::dot(residual_vec, lin_sol_sigma_residual_vec);

    const double trace_psi = gpflash::detail::calc_trace_psi(lin_sol_sigma_residual_vec, arma::trimatl(Sigma_inv_mat));

    // zeta = tr(Psi), checked against the trace of the dense Psi rather than calc_trace_psi's closed form.
    const arma::mat Psi_mat = 0.5 * (lin_sol_sigma_residual_vec * lin_sol_sigma_residual_vec.t() - Sigma_inv_mat);
    const double trace_psi_ref = arma::trace(Psi_mat);
    const double trace_psi_err =
            std::fabs(trace_psi - trace_psi_ref) / std::max(std::fabs(trace_psi_ref), kRelScaleFloor);

    // d(Sigma)/d(sigma_f^2) is the identity, so this derivative must be tr(Psi) exactly, not merely close.
    const bool d_sig2_f_is_trace_psi = gpflash::detail::calc_d_mll_d_sig2_f(trace_psi) == trace_psi;

    // <Psi, K + eta W>, formed here as <Psi, Sigma - sig2_f_val I> / hp_sig2_k straight from the dense Psi, so
    // the reference does not reuse the identity the routine folds the trace into.
    const arma::mat d_sigma_d_sig2_k_mat =
            (arma::symmatl(one_case.Sigma_mat) - sig2_f_val * arma::eye(n_train, n_train)) / hp_sig2_k;
    const double d_sig2_k_ref = arma::accu(Psi_mat % d_sigma_d_sig2_k_mat);
    const double trace_psi_sigma = gpflash::detail::calc_trace_psi_sigma(residual_quad_form, n_train);
    const double d_sig2_k = gpflash::detail::calc_d_mll_d_sig2_k(trace_psi_sigma, trace_psi, sig2_f_val, hp_sig2_k);
    const double d_sig2_k_err = std::fabs(d_sig2_k - d_sig2_k_ref) / std::max(std::fabs(d_sig2_k_ref), kRelScaleFloor);

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "n_train=%d trace_psi=%.1e d_sig2_f_exact=%d d_sig2_k=%.1e",
            n_train,
            trace_psi_err,
            static_cast<int>(d_sig2_f_is_trace_psi),
            d_sig2_k_err);

    return report(name, trace_psi_err < kTolAltPath && d_sig2_f_is_trace_psi && d_sig2_k_err < kTolAltPath, detail);
}

// Asserts the four in-place solves match a dense solve, over both directions and both BLAS paths.
//
// n_rhs of 1 and 4 exercise the DTRSV and DTRSM branches inside solve_chol_blas_inplace. The vec and mat
// entry points are separate functions, so both are called rather than one standing in for the other.
bool test_inplace_solves(const char* name, const int n_train) {
    const CholCase one_case = make_chol_case(n_train);
    const arma::mat L_dense_mat = arma::trimatl(one_case.L_mat_lower);

    // Each solve feeds its direction's worst and its BLAS call's worst, so a failure names both.
    double worst_forward_err = 0.0, worst_transpose_err = 0.0, worst_dtrsv_err = 0.0, worst_dtrsm_err = 0.0;
    for (const int n_rhs : {1, 4}) {
        const arma::mat B_mat = make_points(n_train, n_rhs);
        const bool is_dtrsm = n_rhs > 1;
        double& worst_blas_err = is_dtrsm ? worst_dtrsm_err : worst_dtrsv_err;

        arma::mat forward_mat = B_mat;
        solve_chol_forward_mat_inplace(one_case.L_mat_lower, forward_mat);
        const double forward_mat_err = rel_err(forward_mat, arma::solve(L_dense_mat, B_mat));
        worst_forward_err = std::max(worst_forward_err, forward_mat_err);
        worst_blas_err = std::max(worst_blas_err, forward_mat_err);

        arma::mat transpose_mat = B_mat;
        solve_chol_transpose_mat_inplace(one_case.L_mat_lower, transpose_mat);
        const double transpose_mat_err = rel_err(transpose_mat, arma::solve(L_dense_mat.t(), B_mat));
        worst_transpose_err = std::max(worst_transpose_err, transpose_mat_err);
        worst_blas_err = std::max(worst_blas_err, transpose_mat_err);
    }
    const arma::vec b_vec = make_points(n_train, 1);

    arma::vec forward_vec = b_vec;
    solve_chol_forward_vec_inplace(one_case.L_mat_lower, forward_vec);
    const double forward_vec_err = rel_err(forward_vec, arma::solve(L_dense_mat, b_vec));
    worst_forward_err = std::max(worst_forward_err, forward_vec_err);
    worst_dtrsv_err = std::max(worst_dtrsv_err, forward_vec_err);

    arma::vec transpose_vec = b_vec;
    solve_chol_transpose_vec_inplace(one_case.L_mat_lower, transpose_vec);
    const double transpose_vec_err = rel_err(transpose_vec, arma::solve(L_dense_mat.t(), b_vec));
    worst_transpose_err = std::max(worst_transpose_err, transpose_vec_err);
    worst_dtrsv_err = std::max(worst_dtrsv_err, transpose_vec_err);

    const double worst_err = std::max(worst_forward_err, worst_transpose_err);
    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "n_train=%d forward=%.1e transpose=%.1e dtrsv=%.1e dtrsm=%.1e",
            n_train,
            worst_forward_err,
            worst_transpose_err,
            worst_dtrsv_err,
            worst_dtrsm_err);

    return report(name, worst_err < kTolAltPath, detail);
}

// Asserts the forward solve followed by solve_chol_transpose_vec_inplace gives Sigma^-1 b.
bool test_full_solve(const char* name, const int n_train) {
    const CholCase one_case = make_chol_case(n_train);
    const arma::vec b_vec = make_points(n_train, 1);

    arma::vec lin_sol_vec = solve_chol_forward_vec(one_case.L_mat_lower, b_vec);
    solve_chol_transpose_vec_inplace(one_case.L_mat_lower, lin_sol_vec);

    // Residual against Sigma itself, so the reference shares no step with the two triangular solves.
    const double residual = rel_err(one_case.Sigma_mat * lin_sol_vec, b_vec);
    char detail[kDetailLen];
    std::snprintf(detail, sizeof(detail), "n_train=%d ||Sigma x - b||/||b|| = %.1e", n_train, residual);

    return report(name, residual < kTolAltPath, detail);
}

// Asserts the two inner-product wrappers equal a dense transposed product, and that scale_factor is applied.
//
// The -2 case is the one assembly uses for a stationary kernel, so both scales are exercised.
bool test_inner_products(const char* name, const int n_train) {
    const int n_dim = 4, n_rhs = 3;
    const arma::mat lhs_mat = make_points(n_dim, n_train);
    const arma::mat rhs_mat = make_points(n_dim, n_rhs);
    const arma::vec rhs_vec = make_points(n_dim, 1);

    double worst_err = 0.0, worst_simd_err = 0.0;
    for (const double scale_factor : {1.0, -2.0}) {
        arma::vec inner_prod_vec;
        fill_inner_prod_vec(lhs_mat, rhs_vec, scale_factor, inner_prod_vec);
        worst_err = std::max(worst_err, rel_err(inner_prod_vec, scale_factor * lhs_mat.t() * rhs_vec));

        // The platform alternative to the DGEMV above. Only one of the two is compiled into build_post_vec, so
        // this is the check that keeps the unselected one honest.
        arma::vec inner_prod_simd_vec;
        fill_inner_prod_vec_simd(lhs_mat, rhs_vec, scale_factor, inner_prod_simd_vec);
        worst_simd_err = std::max(worst_simd_err, rel_err(inner_prod_simd_vec, inner_prod_vec));

        arma::mat inner_prod_mat;
        fill_inner_prod_mat(lhs_mat, rhs_mat, scale_factor, inner_prod_mat);
        worst_err = std::max(worst_err, rel_err(inner_prod_mat, scale_factor * lhs_mat.t() * rhs_mat));
    }
    char detail[kDetailLen];
    std::snprintf(
            detail, sizeof(detail), "n_train=%d dense=%.1e simd_vs_dgemv=%.1e", n_train, worst_err, worst_simd_err);

    return report(name, worst_err < kTolAltPath && worst_simd_err < kTolAltPath, detail);
}

// Asserts factorize_chol_inplace reproduces the matrix it replaced, and that ln|A| from the factor's diagonal
// matches a dense log determinant.
//
// Both are checked here because the second reads the first's output and nothing else.
bool test_factorize_and_ln_det(const char* name, const int n_train) {
    const CholCase one_case = make_chol_case(n_train);

    arma::mat factor_mat = arma::trimatl(one_case.Sigma_mat);  // lower triangle only, as dpotrf reads
    factorize_chol_inplace(factor_mat, "test_gpflash_chol_solves");
    const arma::mat product_mat = arma::trimatl(factor_mat) * arma::trimatl(factor_mat).t();
    const double factor_err = rel_err(product_mat, arma::symmatl(one_case.Sigma_mat));

    const double ln_det = calc_ln_det_from_chol(factor_mat);
    double ln_det_ref = 0.0, sign = 0.0;
    arma::log_det(ln_det_ref, sign, arma::symmatl(one_case.Sigma_mat));
    const double ln_det_err = std::fabs(ln_det - ln_det_ref) / std::max(std::fabs(ln_det_ref), kRelScaleFloor);

    char detail[kDetailLen];
    std::snprintf(detail, sizeof(detail), "n_train=%d L L^T=%.1e ln_det=%.1e", n_train, factor_err, ln_det_err);

    return report(name, factor_err < kTolAltPath && ln_det_err < kTolAltPath, detail);
}

// Asserts hp_beta_vec holds the generalized-least-squares mean model weights with the nugget cond_max sizes, and the
// output vector is L^-1 (f - Basis hp_beta_vec). The reference mean model weights come from a dense inverse rather than
// from the forward solves both library quantities are combinations of.
bool test_beta_and_lin_sol_residual(const char* name, const int n_train, const int n_basis, const double cond_max) {
    const CholCase one_case = make_chol_case(n_train);
    const arma::vec f_train_vec = make_points(n_train, 1) + 2.0;

    // A constant column, then x_d^(d + 1) per coordinate, so the columns are independent but correlated; f_train is
    // drawn separately and does not lie in their span.
    const arma::uword n_dim = 2;  // enough dimensions for n_basis up to 3
    const arma::mat X_ndim_ntrain_mat = make_points(static_cast<int>(n_dim), n_train);
    arma::uvec mean_model_degree_vec(n_dim, arma::fill::zeros);
    for (int i_dim = 0; i_dim < n_basis - 1; ++i_dim)
        mean_model_degree_vec(i_dim) = static_cast<arma::uword>(i_dim + 1);
    const arma::mat Basis_ntrain_nbasis_mat = gpflash_test::build_mean_model_basis_ref_mat(
            X_ndim_ntrain_mat, /*has_mean_model_constant=*/true, mean_model_degree_vec);

    const arma::vec lin_sol_L_f_train_vec = gpflash::detail::solve_chol_forward_vec(one_case.L_mat_lower, f_train_vec);
    const arma::mat lin_sol_L_Basis_ntrain_nbasis_mat =
            gpflash::detail::solve_chol_forward_mat(one_case.L_mat_lower, Basis_ntrain_nbasis_mat);
    const arma::vec hp_beta_vec = gpflash::mean::detail::calc_mean_model_weights(
            lin_sol_L_f_train_vec, lin_sol_L_Basis_ntrain_nbasis_mat, cond_max, gpflash::EigMaxUbMethod::Trace);
    const arma::vec lin_sol_L_residual_vec = gpflash::mean::detail::calc_lin_sol_L_residual_vec(
            lin_sol_L_f_train_vec, lin_sol_L_Basis_ntrain_nbasis_mat, hp_beta_vec);

    const arma::mat Sigma_inv_mat = arma::inv_sympd(one_case.Sigma_mat);
    const arma::mat lin_sol_sigma_Basis_ntrain_nbasis_mat = Sigma_inv_mat * Basis_ntrain_nbasis_mat;
    const arma::vec lin_sol_sigma_f_train_vec = Sigma_inv_mat * f_train_vec;
    const arma::vec hp_beta_ref_vec = gpflash_test::calc_hp_beta_ref_vec(
            Basis_ntrain_nbasis_mat, lin_sol_sigma_Basis_ntrain_nbasis_mat, lin_sol_sigma_f_train_vec, cond_max);
    const double hp_beta_err = rel_err(hp_beta_vec, hp_beta_ref_vec);
    const arma::vec f_train_residual_ref_vec = f_train_vec - Basis_ntrain_nbasis_mat * hp_beta_ref_vec;
    const arma::vec residual_ref_vec = arma::solve(arma::trimatl(one_case.L_mat_lower), f_train_residual_ref_vec);
    const double residual_err = rel_err(lin_sol_L_residual_vec, residual_ref_vec);

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "n_train=%d n_basis=%d cond_max=%.0e beta=%.1e lin_sol_residual=%.1e",
            n_train,
            n_basis,
            cond_max,
            hp_beta_err,
            residual_err);
    return report(name, hp_beta_err < kTolAltPath && residual_err < kTolAltPath, detail);
}

// Asserts two identical basis columns, whose mean model Gram matrix is exactly singular, are solved rather than
// rejected: the mean model nugget splits the one weight the data determine evenly between them, and the split weights
// sum to that weight shrunk by the nugget.
bool test_duplicate_basis_columns_split_weight(const char* name) {
    const int n_train = 6;
    const arma::vec lin_sol_L_f_train_vec = make_points(n_train, 1);
    const arma::vec lin_sol_L_basis_col_vec = make_points(n_train, 1) + 1.0;
    const arma::mat lin_sol_L_Basis_ntrain_nbasis_mat =
            arma::join_horiz(lin_sol_L_basis_col_vec, lin_sol_L_basis_col_vec);
    const double hp_beta_single_ref = arma::dot(lin_sol_L_f_train_vec, lin_sol_L_basis_col_vec)
            / arma::dot(lin_sol_L_basis_col_vec, lin_sol_L_basis_col_vec);

    bool raised = false;
    arma::vec hp_beta_vec;
    try {
        hp_beta_vec = gpflash::mean::detail::calc_mean_model_weights(
                lin_sol_L_f_train_vec,
                lin_sol_L_Basis_ntrain_nbasis_mat,
                gpflash::COND_MAX_DEFAULT,
                gpflash::EigMaxUbMethod::Trace);
    } catch (...) {
        raised = true;
    }
    // The nugget adds mean_model_eta to each unit diagonal entry, so the pair solves
    // [[1 + mean_model_eta, 1], [1, 1 + mean_model_eta]] and their sum is the single weight times
    // 2 / (2 + mean_model_eta). Their difference lies along the null direction, which the nugget determines only to
    // about cond_max eps.
    const double mean_model_eta_ref = gpflash_test::calc_mean_model_eta_ref(2, gpflash::COND_MAX_DEFAULT);
    const double hp_beta_sum_ref = hp_beta_single_ref * 2.0 / (2.0 + mean_model_eta_ref);
    const double tol_null_direction = gpflash::COND_MAX_DEFAULT * arma::datum::eps;
    const bool split_ok = !raised && hp_beta_vec.n_elem == 2
            && std::fabs(hp_beta_vec(0) + hp_beta_vec(1) - hp_beta_sum_ref) <= kTolAltPath * std::fabs(hp_beta_sum_ref)
            && std::fabs(hp_beta_vec(0) - hp_beta_vec(1)) <= tol_null_direction * std::fabs(hp_beta_sum_ref);

    char detail[kDetailLen];
    std::snprintf(detail, sizeof(detail), "raised=%d", static_cast<int>(raised));
    return report(name, split_ok, detail);
}

// Asserts a basis small enough that its squared norm underflows raises ErrorNumericalFailure rather than returning a
// non-finite weight, on the one-weight division and on the preconditioned solve.
bool test_underflowing_basis_rejected(const char* name) {
    const int n_train = 6;
    const arma::vec lin_sol_L_f_train_vec = make_points(n_train, 1) + 1.0;
    // Each entry squares to 1e-400, below the smallest subnormal double.
    const arma::vec lin_sol_L_tiny_basis_col_vec = 1e-200 * (make_points(n_train, 1) + 1.0);
    const auto raises_numerical_failure = [&](const arma::mat& lin_sol_L_Basis_ntrain_nbasis_mat) {
        try {
            gpflash::mean::detail::calc_mean_model_weights(
                    lin_sol_L_f_train_vec,
                    lin_sol_L_Basis_ntrain_nbasis_mat,
                    gpflash::COND_MAX_DEFAULT,
                    gpflash::EigMaxUbMethod::Trace);
        } catch (const gpflash::ErrorNumericalFailure&) {
            return true;
        } catch (...) {
            return false;
        }
        return false;
    };
    const arma::mat lin_sol_L_two_col_Basis_mat =
            arma::join_horiz(lin_sol_L_tiny_basis_col_vec, lin_sol_L_f_train_vec + 1.0);
    const bool one_weight_raised = raises_numerical_failure(lin_sol_L_tiny_basis_col_vec);
    const bool two_weights_raised = raises_numerical_failure(lin_sol_L_two_col_Basis_mat);
    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "one_weight=%d two_weights=%d",
            static_cast<int>(one_weight_raised),
            static_cast<int>(two_weights_raised));
    return report(name, one_weight_raised && two_weights_raised, detail);
}
}  // namespace
int main() {
    std::printf("== GpFlash Cholesky solves ==\n");
    bool ok = true;
    for (const int n_train : {5, 40}) {
        ok &= test_forward_solves("forward_solves", n_train);
        ok &= test_inplace_solves("inplace_solves", n_train);
        ok &= test_full_solve("full_solve", n_train);
        ok &= test_inner_products("inner_products", n_train);
        ok &= test_factorize_and_ln_det("factorize_and_ln_det", n_train);
        // One weight has no mean model nugget, so cond_max cannot reach it. At the default cond_max the nugget barely
        // moves the n_basis = 3 weights; at kCondMaxStrongNugget it moves them well past the tolerance, so a reference
        // without it would fail.
        ok &= test_beta_and_lin_sol_residual(
                "beta_and_lin_sol_residual", n_train, /*n_basis=*/1, gpflash::COND_MAX_DEFAULT);
        for (const double cond_max : {gpflash::COND_MAX_DEFAULT, kCondMaxStrongNugget})
            ok &= test_beta_and_lin_sol_residual("beta_and_lin_sol_residual", n_train, /*n_basis=*/3, cond_max);
        ok &= test_variance_hp_grad("variance_hp_grad", n_train);
    }
    ok &= test_duplicate_basis_columns_split_weight("duplicate_basis_columns_split_weight");
    ok &= test_underflowing_basis_rejected("underflowing_basis_rejected");
    std::printf("\n%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");

    return ok ? 0 : 1;
}
