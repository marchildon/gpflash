// The nugget: the Gershgorin bound and its active row against a dense reference, eta bounding kappa on the
// unit-diagonal path under either bound and on the preconditioned path under Gershgorin's, and build_eta_terms
// rejecting a bound method outside the enum or a false unit-diagonal claim.
#include <cmath>
#include <cstdio>

#include "GpFlashTestSupport.hpp"
#include "gpflash/assembly/KernelMatrix.hpp"
#include "gpflash/assembly/SigmaMatrix.hpp"
#include "gpflash/kernels/KernelDispatch.hpp"  // kernel_by_id
#include "gpflash/kernels/radial/KernelRadial.hpp"
#include "gpflash/preconditioning/ConditionedCholesky.hpp"  // build_eta_terms, fill_eta_bound_data

using gpflash_test::kCondBoundSlack;
using gpflash_test::kDetailLen;
using gpflash_test::kHpRhoUnused;
using gpflash_test::kRatQuadHpRho;
using gpflash_test::kTolAltPath;
using gpflash_test::kTolExact;
using gpflash_test::make_points;
using gpflash_test::report;

namespace {

// How far the unpreconditioned kappa must exceed cond_max for the negative result to mean anything. Three
// orders of magnitude: enough that no plausible rounding explains it.
constexpr double kMinUnpreconditionedKappaRatio = 1.0e3;

// Asserts the bound and the index of the row attaining it, read off a real kernel matrix, equal a direct
// max_i (K_ii + sum_{j != i} |K_ij|) and its arg max over a dense reference.
bool test_eta_terms_vs_dense_reference(
        const gpflash::kernel::KernelBase& kern_fun, const char* name, const double hp_rho, const int n_train = 23) {
    const int n_dim = 4;
    const arma::mat X_ndim_ntrain_mat = make_points(n_dim, n_train);
    arma::vec hp_gamma_vec(n_dim);
    for (int i_dim = 0; i_dim < n_dim; ++i_dim) hp_gamma_vec(i_dim) = 0.6 + 0.35 * i_dim;
    const double* hp_kern_extra_ptr = (kern_fun.n_hp_kern_extra(n_dim) > 0) ? &hp_rho : nullptr;

    const arma::mat Ydot_nfeature_ntrain_mat = X_ndim_ntrain_mat.each_col() % hp_gamma_vec;
    arma::mat K_built_mat = gpflash::detail::build_kmat_lower_tri(
            kern_fun, Ydot_nfeature_ntrain_mat, hp_kern_extra_ptr, /*store_hp_grad_factors=*/true);
    // Every kernel here has a unit diagonal, so this reads the bound without altering K.
    const gpflash::detail::EtaTerms kmat_eta_terms = gpflash::detail::build_eta_terms(
            kern_fun.is_kern_diag_unity(),
            K_built_mat,
            /*cond_max=*/1.0e4,
            gpflash::EigMaxUbMethod::Gershgorin,
            /*want_active_row=*/true);

    // fill_eta_bound_data walks the lower triangle column-major into two accumulators per entry; this reference
    // shares no traversal with it, summing whole rows of the full symmetric matrix.
    const arma::mat K_sym_abs_mat = arma::abs(arma::symmatl(K_built_mat));
    const arma::vec off_diag_sum_vec = arma::sum(K_sym_abs_mat, 1) - K_sym_abs_mat.diag();

    double ref_max_abs_row_sum = 0.0;
    int ref_row_idx = -1, n_ties = 0;
    for (int i_row = 0; i_row < n_train; ++i_row) {
        const double off_diag_sum = off_diag_sum_vec(i_row);
        // Tie detection is relative: the sum runs over n_train terms, so its rounding grows with n_train. An absolute
        // threshold would read rows agreeing only to the last bit as a unique maximum and arm the arg-max
        // assertion on an arg max that is undefined.
        const double tie_tol = kTolExact * std::max(ref_max_abs_row_sum, 1.0);
        if (off_diag_sum > ref_max_abs_row_sum + tie_tol) {
            ref_max_abs_row_sum = off_diag_sum;
            ref_row_idx = i_row;
            n_ties = 1;
        } else if (std::fabs(off_diag_sum - ref_max_abs_row_sum) <= tie_tol) {
            ++n_ties;
        }
    }
    ref_max_abs_row_sum += 1.0;
    // Relative, since the error of a sum of n_train terms grows with n_train while each kernel value stays O(1).
    const double row_sum_err =
            std::abs(kmat_eta_terms.eig_max_ub_M_precon - ref_max_abs_row_sum) / std::max(ref_max_abs_row_sum, 1.0);
    // The row index feeds the eta derivative, so a wrong one corrupts the gradient. Asserted only where the
    // maximum is unique; the skip is reported so it cannot be mistaken for a pass.
    const bool argmax_unique = (n_ties == 1);
    const bool argmax_ok = !argmax_unique || (kmat_eta_terms.idx_M_precon_max_abs_row_sum == ref_row_idx);

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "n_train=%d row_sum_err=%.1e argmax=%s",
            n_train,
            row_sum_err,
            argmax_unique ? (argmax_ok ? "ok" : "WRONG") : "skipped (tie)");

    return report(name, row_sum_err < kTolExact && argmax_ok, detail);
}

// fill_eta_bound_data on a hand-built matrix, the only way to reach two of its behaviors: every
// GpFlash kernel is strictly positive, so no real kernel matrix exercises the sign capture, and the
// single-point case cannot arise from an assembly the other tests drive.
bool test_eta_terms(const char* name) {
    using gpflash::detail::fill_eta_bound_data;

    // Lower triangle holds the matrix; the strict upper is filled with a value that would dominate every row
    // sum if read, proving it is not.
    const int n_train = 4;
    arma::mat K_mat(n_train, n_train, arma::fill::zeros);
    K_mat.diag().ones();
    K_mat(1, 0) = -0.5;
    K_mat(2, 0) = 0.25;
    K_mat(3, 0) = 0.125;
    K_mat(2, 1) = -0.75;  // row 2 has the largest absolute off-diagonal sum: 0.25 + 0.75 + 0.5 = 1.5
    K_mat(3, 1) = 0.0625;
    K_mat(3, 2) = 0.5;
    for (int i_col = 0; i_col < n_train; ++i_col)
        for (int i_row = 0; i_row < i_col; ++i_row) K_mat(i_row, i_col) = 1.0e6;

    gpflash::detail::EtaTerms kmat_eta_terms;
    fill_eta_bound_data(K_mat, /*want_active_row=*/true, kmat_eta_terms);
    const bool bound_ok = std::fabs(kmat_eta_terms.eig_max_ub_M_precon - (1.0 + 1.5)) < kTolExact;
    const bool idx_ok = kmat_eta_terms.idx_M_precon_max_abs_row_sum == 2;
    // Row 2 gathered off the lower triangle is [0.25, -0.75, 0, 0.5]: -0.75 keeps its sign, no entry is the 1e6 in
    // the strict upper triangle, and the row's own entry stays zero because the bound's sum excludes it.
    const arma::vec active_row_ref_vec = {0.25, -0.75, 0.0, 0.5};
    const bool row_ok = kmat_eta_terms.active_row_M_precon_vec.n_elem == static_cast<arma::uword>(n_train)
            && arma::abs(kmat_eta_terms.active_row_M_precon_vec - active_row_ref_vec).max() < kTolExact;

    // Not asking for the row leaves it empty, which is what makes it free on the MLL-only and posterior paths.
    gpflash::detail::EtaTerms no_row_data;
    fill_eta_bound_data(K_mat, /*want_active_row=*/false, no_row_data);
    const bool no_row_ok = no_row_data.active_row_M_precon_vec.is_empty()
            && std::fabs(no_row_data.eig_max_ub_M_precon - (1.0 + 1.5)) < kTolExact;

    // P = diag(sqrt(diag(M))) and M_precon = P^-1 M P^-1 on a 2x2 with a non-unit diagonal, so a wrong power of
    // the diagonal or a swapped scale changes the result. The diagonal is assigned 1; the off-diagonal is divided by
    // both scales.
    const arma::vec precon_scale_ref_vec{2.0, 3.0};
    const double M_off_diagonal = 3.0;
    arma::mat M_mat_lower(2, 2, arma::fill::zeros);
    M_mat_lower.diag() = arma::square(precon_scale_ref_vec);
    M_mat_lower(1, 0) = M_off_diagonal;
    const arma::vec precon_scale_vec = gpflash::detail::calc_precon_scale_vec(M_mat_lower, "test");
    const arma::vec inv_precon_scale_vec = 1.0 / precon_scale_vec;
    arma::mat M_precon_mat_lower = M_mat_lower;
    gpflash::detail::scale_to_unit_diag_inplace(M_precon_mat_lower, inv_precon_scale_vec);
    const double M_precon_off_diagonal_ref = M_off_diagonal / (precon_scale_ref_vec(0) * precon_scale_ref_vec(1));
    const bool precon_scale_ok = arma::approx_equal(precon_scale_vec, precon_scale_ref_vec, "absdiff", 0.0)
            && M_precon_mat_lower(0, 0) == 1.0 && M_precon_mat_lower(1, 1) == 1.0
            && std::fabs(M_precon_mat_lower(1, 0) - M_precon_off_diagonal_ref) < kTolExact;

    // A single point is already perfectly conditioned: no nugget, and no active row, on which index the eta
    // driver returns.
    arma::mat K_one_mat(1, 1);
    K_one_mat(0, 0) = 1.0;
    gpflash::detail::EtaTerms one_data;
    fill_eta_bound_data(K_one_mat, /*want_active_row=*/true, one_data);
    const bool one_ok = one_data.idx_M_precon_max_abs_row_sum == -1 && one_data.eig_max_ub_M_precon == 0.0;

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "bound=%d idx=%d row=%d no_row=%d precon_scale=%d one_point=%d",
            bound_ok,
            idx_ok,
            static_cast<int>(row_ok),
            static_cast<int>(no_row_ok),
            static_cast<int>(precon_scale_ok),
            static_cast<int>(one_ok));

    return report(name, bound_ok && idx_ok && row_ok && no_row_ok && precon_scale_ok && one_ok, detail);
}

// With diag(K) != 1 the nugget bounds kappa only in the coordinates it was measured in, so the pipeline
// takes K to K_corr, factorizes there, and scales L's rows back. Checked here on a Gram matrix, which stands
// in for the linear kernel: symmetric positive semi-definite, singular because n_train > n_dim, and with a
// diagonal spanning many orders of magnitude.
bool test_precon_path_bounds_condition(const char* name) {
    using gpflash::detail::add_nugget_and_noise_to_diag_inplace;
    using gpflash::detail::calc_eta;
    using gpflash::detail::calc_L_M_precon_to_L_M_inplace;
    using gpflash::detail::calc_precon_scale_vec;
    using gpflash::detail::fill_eta_bound_data;
    using gpflash::detail::scale_to_unit_diag_inplace;

    const int n_dim = 3, n_train = 25;
    arma::mat X_ndim_ntrain_mat = make_points(n_dim, n_train);
    for (int i_train = 0; i_train < n_train; ++i_train)
        X_ndim_ntrain_mat.col(i_train) *= std::pow(10.0, -2.0 + 4.0 * i_train / (n_train - 1.0));
    const arma::mat K_full_mat = X_ndim_ntrain_mat.t() * X_ndim_ntrain_mat;

    double worst_ratio = 0.0, worst_reconstruct = 0.0, worst_unpreconditioned_ratio = 0.0;
    int n_chol_failed = 0;
    bool all_ok = true;
    for (const double cond_max : {1.0e3, 1.0e6, 1.0e10}) {
        arma::mat K_corr_mat_lower = arma::trimatl(K_full_mat);
        const arma::vec precon_scale_vec = calc_precon_scale_vec(K_corr_mat_lower, "test");
        const arma::vec inv_precon_scale_vec = 1.0 / precon_scale_vec;
        scale_to_unit_diag_inplace(K_corr_mat_lower, inv_precon_scale_vec);
        gpflash::detail::EtaTerms kmat_eta_terms;
        fill_eta_bound_data(K_corr_mat_lower, /*want_active_row=*/false, kmat_eta_terms);
        const double eta = calc_eta(kmat_eta_terms.eig_max_ub_M_precon, cond_max);

        // The matrix dpotrf actually sees.
        arma::mat Sigma_corr_mat_lower = K_corr_mat_lower;
        add_nugget_and_noise_to_diag_inplace(
                Sigma_corr_mat_lower, eta, /*sig2_f_val=*/0.0, /*hp_sig2_k=*/1.0, inv_precon_scale_vec);
        const arma::vec eig_corr_vec = arma::eig_sym(arma::symmatl(Sigma_corr_mat_lower));
        const double kappa_corr = eig_corr_vec.max() / eig_corr_vec.min();
        worst_ratio = std::max(worst_ratio, kappa_corr / cond_max);
        if (!(kappa_corr <= cond_max * (1.0 + kCondBoundSlack))) all_ok = false;

        // L_Sigma = P chol(Sigma_corr) must reconstruct Sigma = K + eta P P in absolute coordinates. The bool form,
        // so a Sigma_corr that is not positive definite fails this case instead of aborting the run.
        arma::mat L_Sigma_mat;
        const bool is_chol_ok = arma::chol(L_Sigma_mat, arma::symmatl(Sigma_corr_mat_lower), "lower");
        if (!is_chol_ok) {
            all_ok = false;
            ++n_chol_failed;
            continue;
        }
        calc_L_M_precon_to_L_M_inplace(L_Sigma_mat, precon_scale_vec);
        arma::mat Sigma_ref_mat = K_full_mat;
        Sigma_ref_mat.diag() += eta * arma::square(precon_scale_vec);
        worst_reconstruct = std::max(
                worst_reconstruct,
                arma::abs(L_Sigma_mat * L_Sigma_mat.t() - Sigma_ref_mat).max() / arma::abs(Sigma_ref_mat).max());

        // Why the mapping is not optional: Sigma = P Sigma_corr P, in absolute coordinates, misses the bound outright.
        const arma::vec eig_vec = arma::eig_sym(Sigma_ref_mat);
        worst_unpreconditioned_ratio =
                std::max(worst_unpreconditioned_ratio, (eig_vec.max() / eig_vec.min()) / cond_max);
    }
    if (!(worst_reconstruct < kTolAltPath)) all_ok = false;
    // The negative result: if this ever drops to 1 the test has stopped covering the case it was written for.
    if (!(worst_unpreconditioned_ratio > kMinUnpreconditionedKappaRatio)) all_ok = false;

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "diag_span=%.1e kappa_corr/cond_max=%.4f LLt_err=%.1e kappa/cond_max=%.1e chol_failed=%d",
            K_full_mat.diag().max() / K_full_mat.diag().min(),
            worst_ratio,
            worst_reconstruct,
            worst_unpreconditioned_ratio,
            n_chol_failed);

    return report(name, all_ok, detail);
}

// eta must deliver kappa(K + eta I) <= cond_max. On real eigenvalues of a real kernel matrix, at a cond_max
// small enough that the nugget is doing visible work.
bool test_eta_bounds_condition(const gpflash::kernel::KernelBase& kern_fun, const char* name, const double hp_rho) {
    const int n_dim = 3, n_train = 25;
    const arma::mat X_ndim_ntrain_mat = make_points(n_dim, n_train);
    arma::vec hp_gamma_vec(n_dim);
    // Small hp_gamma: long length scales make K nearly singular, the case eta must rescue.
    for (int i_dim = 0; i_dim < n_dim; ++i_dim) hp_gamma_vec(i_dim) = 0.05 + 0.01 * i_dim;
    const double* hp_kern_extra_ptr = (kern_fun.n_hp_kern_extra(n_dim) > 0) ? &hp_rho : nullptr;

    const arma::mat Ydot_nfeature_ntrain_mat = X_ndim_ntrain_mat.each_col() % hp_gamma_vec;
    const arma::mat K_mat_lower = gpflash::detail::build_kmat_lower_tri(
            kern_fun, Ydot_nfeature_ntrain_mat, hp_kern_extra_ptr, /*store_hp_grad_factors=*/false);

    double worst_ratio = 0.0;
    bool all_ok = true;
    for (const double cond_max : {1.0e3, 1.0e6, 1.0e10}) {
        arma::mat K_corr_mat_lower = K_mat_lower;
        const gpflash::detail::EtaTerms kmat_eta_terms = gpflash::detail::build_eta_terms(
                kern_fun.is_kern_diag_unity(),
                K_corr_mat_lower,
                cond_max,
                gpflash::EigMaxUbMethod::Gershgorin,
                /*want_active_row=*/false);
        arma::mat Sigma_mat = arma::symmatl(K_corr_mat_lower);
        Sigma_mat.diag() += kmat_eta_terms.eta;
        const arma::vec eig_vec = arma::eig_sym(Sigma_mat);
        const double kappa = eig_vec.max() / eig_vec.min();
        worst_ratio = std::max(worst_ratio, kappa / cond_max);
        if (!(kappa <= cond_max * (1.0 + kCondBoundSlack))) all_ok = false;
    }
    char detail[kDetailLen];
    std::snprintf(detail, sizeof(detail), "worst kappa/cond_max %.4f", worst_ratio);

    return report(name, all_ok, detail);
}

// The trace bound on a real kernel matrix: eta = n_train / (cond_max - 1) with no active row, even when one is
// asked for, and it still delivers kappa(K + eta I) <= cond_max.
bool test_trace_eta_bounds_condition(const char* name) {
    const gpflash::kernel::KernelBase& kern_fun = gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::SqExp);
    const int n_dim = 3, n_train = 25;
    const arma::mat X_ndim_ntrain_mat = make_points(n_dim, n_train);
    arma::vec hp_gamma_vec(n_dim);
    for (int i_dim = 0; i_dim < n_dim; ++i_dim) hp_gamma_vec(i_dim) = 0.05 + 0.01 * i_dim;

    const arma::mat Ydot_nfeature_ntrain_mat = X_ndim_ntrain_mat.each_col() % hp_gamma_vec;
    const arma::mat K_mat_lower = gpflash::detail::build_kmat_lower_tri(
            kern_fun, Ydot_nfeature_ntrain_mat, /*hp_kern_extra_ptr=*/nullptr, /*store_hp_grad_factors=*/false);

    double worst_ratio = 0.0, worst_eta_rel_err = 0.0;
    bool all_ok = true;
    for (const double cond_max : {1.0e3, 1.0e6, 1.0e10}) {
        arma::mat K_corr_mat_lower = K_mat_lower;
        const gpflash::detail::EtaTerms kmat_eta_terms = gpflash::detail::build_eta_terms(
                kern_fun.is_kern_diag_unity(),
                K_corr_mat_lower,
                cond_max,
                gpflash::EigMaxUbMethod::Trace,
                /*want_active_row=*/true);
        const double eta_expected = n_train / (cond_max - 1.0);
        const double eta_rel_err = std::fabs(kmat_eta_terms.eta - eta_expected) / eta_expected;
        worst_eta_rel_err = std::max(worst_eta_rel_err, eta_rel_err);
        const bool no_row_ok =
                kmat_eta_terms.idx_M_precon_max_abs_row_sum == -1 && kmat_eta_terms.active_row_M_precon_vec.is_empty();
        if (!(kmat_eta_terms.eig_max_ub_M_precon == n_train && eta_rel_err < kTolExact && no_row_ok)) all_ok = false;

        arma::mat Sigma_mat = arma::symmatl(K_corr_mat_lower);
        Sigma_mat.diag() += kmat_eta_terms.eta;
        const arma::vec eig_vec = arma::eig_sym(Sigma_mat);
        const double kappa = eig_vec.max() / eig_vec.min();
        worst_ratio = std::max(worst_ratio, kappa / cond_max);
        if (!(kappa <= cond_max * (1.0 + kCondBoundSlack))) all_ok = false;
    }

    // A single point needs no nugget under either bound.
    arma::mat K_one_mat(1, 1);
    K_one_mat(0, 0) = 1.0;
    const gpflash::detail::EtaTerms one_eta_terms = gpflash::detail::build_eta_terms(
            /*is_unit_diag=*/true,
            K_one_mat,
            /*cond_max=*/1.0e6,
            gpflash::EigMaxUbMethod::Trace,
            /*want_active_row=*/true);
    const bool one_ok = one_eta_terms.eig_max_ub_M_precon == 0.0 && one_eta_terms.eta == 0.0;
    if (!one_ok) all_ok = false;

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "worst kappa/cond_max %.4f, eta rel err %.2e, one_point=%d",
            worst_ratio,
            worst_eta_rel_err,
            static_cast<int>(one_ok));

    return report(name, all_ok, detail);
}

// build_eta_terms refuses a bound method outside the enum, and a unit-diagonal claim the matrix contradicts.
bool test_build_eta_terms_rejects(const char* name) {
    using gpflash::detail::build_eta_terms;

    arma::mat M_mat_lower(2, 2, arma::fill::zeros);
    M_mat_lower.diag().ones();
    M_mat_lower(1, 0) = 0.5;
    const auto built = [&](const bool is_unit_diag, const gpflash::EigMaxUbMethod eig_max_ub_method) {
        arma::mat M_in_M_precon_out_mat_lower = M_mat_lower;
        build_eta_terms(
                is_unit_diag,
                M_in_M_precon_out_mat_lower,
                /*cond_max=*/1.0e6,
                eig_max_ub_method,
                /*want_active_row=*/false);
    };
    const bool end_rejected = gpflash_test::rejects([&] { built(true, gpflash::EigMaxUbMethod::End); });
    const bool cast_rejected = gpflash_test::rejects([&] { built(true, static_cast<gpflash::EigMaxUbMethod>(-1)); });

    // A diagonal of 2 claimed as unit.
    M_mat_lower.diag().fill(2.0);
    const bool false_claim_rejected = gpflash_test::rejects([&] { built(true, gpflash::EigMaxUbMethod::Trace); });
    const bool unclaimed_ok = gpflash_test::accepts([&] { built(false, gpflash::EigMaxUbMethod::Trace); });

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "end=%d cast=%d false_claim=%d unclaimed=%d",
            static_cast<int>(end_rejected),
            static_cast<int>(cast_rejected),
            static_cast<int>(false_claim_rejected),
            static_cast<int>(unclaimed_ok));

    return report(name, end_rejected && cast_rejected && false_claim_rejected && unclaimed_ok, detail);
}

}  // namespace

int main() {
    std::printf("== GpFlash kernel matrix nugget ==\n");
    bool ok = true;
    ok &= test_eta_terms("eta_terms");
    ok &= test_eta_terms_vs_dense_reference(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::SqExp), "eta_terms_vs_dense/SqExp", kHpRhoUnused);
    ok &= test_eta_terms_vs_dense_reference(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Matern52),
            "eta_terms_vs_dense/Matern52",
            kHpRhoUnused);
    ok &= test_eta_terms_vs_dense_reference(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Matern32),
            "eta_terms_vs_dense/Matern32",
            kHpRhoUnused);
    ok &= test_eta_terms_vs_dense_reference(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::RatQuad),
            "eta_terms_vs_dense/RatQuad",
            kRatQuadHpRho);
    // Above gpflash::PERF_OMP_MIN_N_POINT, so the parallel reduction is compared against the serial reference.
    ok &= test_eta_terms_vs_dense_reference(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::SqExp),
            "eta_terms_vs_dense/SqExp_parallel",
            kHpRhoUnused,
            gpflash_test::kNtrainAboveOmpThresholds);
    ok &= test_eta_bounds_condition(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::SqExp), "eta_bounds_kappa/SqExp", kHpRhoUnused);
    ok &= test_eta_bounds_condition(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Matern52),
            "eta_bounds_kappa/Matern52",
            kHpRhoUnused);
    ok &= test_eta_bounds_condition(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::RatQuad),
            "eta_bounds_kappa/RatQuad",
            kRatQuadHpRho);
    ok &= test_trace_eta_bounds_condition("trace_eta_bounds_kappa/SqExp");
    ok &= test_build_eta_terms_rejects("build_eta_terms_rejects");
    ok &= test_precon_path_bounds_condition("precon_path_bounds_kappa/non_unit_diagonal");
    std::printf("\n%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");

    return ok ? 0 : 1;
}
