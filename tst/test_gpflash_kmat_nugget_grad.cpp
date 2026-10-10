// KmatNuggetGrad.hpp on its own. What this file asserts:
//
//   1) precon_scale_grad  calc_eta_precon_scale_grad is zero for a unit-diagonal kernel, nonzero for Quadratic
//   2) eta_grad_vs_fd     calc_eta_term_hp_grad equals a finite difference of eta
//   3) extras_without_diag_grad_refused  a preconditioned kernel with extras but no d(diag K)/d(extras) is refused
//
// Check 2 passes an MLL sensitivity of 1, so the driver's output is the bare derivative of eta. The bound is an
// arg max over rows, so it also asserts the winning row is unchanged at both perturbed points; a step that
// moved it would compare two different functions.
//
// Quadratic's varying diagonal is the one case where calc_eta_precon_scale_grad's d(P)/d(gamma) term is nonzero,
// so it is what pins that term on both checks.
#include <armadillo>
#include <cmath>
#include <cstdio>
#include <string>
#include <utility>

#include "GpFlashTestSupport.hpp"
#include "gpflash/assembly/KernelMatrix.hpp"
#include "gpflash/assembly/KmatNuggetGrad.hpp"
#include "gpflash/core/Errors.hpp"          // ErrorNotImplemented
#include "gpflash/inputs/HpVal.hpp"         // calc_hp_kern_extra_ptr
#include "gpflash/inputs/SymmetrySpec.hpp"  // SymmetrySpec, the no-symmetry argument
#include "gpflash/kernels/HpGradInputs.hpp"
#include "gpflash/kernels/KernelDispatch.hpp"  // kernel_by_id
#include "gpflash/kernels/radial/KernelRadial.hpp"
#include "gpflash/preconditioning/ConditionedCholesky.hpp"  // build_eta_terms, fill_eta_bound_data

using gpflash_test::harvest_active_row_kern_common_rdot;
using gpflash_test::kDetailLen;
using gpflash_test::kFdScaleFloor;
using gpflash_test::kFdStepKern;
using gpflash_test::kHpRhoUnused;
using gpflash_test::kRatQuadHpRho;
using gpflash_test::make_points;
using gpflash_test::report;

namespace {

// Looser than the support header's kTolFd: eta's bound rebuilds a full K per finite-difference point, so
// the difference carries assembly roundoff a closed-form kernel value does not.
constexpr double kTolFdEta = 1.0e-6;

bool test_eta_grad_vs_fd(const gpflash::kernel::KernelBase& kern_fun, const char* name, const double hp_rho) {
    const int n_dim = 3, n_train = 15;
    const arma::mat X_ndim_ntrain_mat = make_points(n_dim, n_train);
    arma::vec hp_gamma_vec(n_dim);
    arma::vec hp_kern_extra_vec(kern_fun.n_hp_kern_extra(n_dim));
    for (int i_dim = 0; i_dim < n_dim; ++i_dim) hp_gamma_vec(i_dim) = 0.8 + 0.3 * i_dim;
    if (!hp_kern_extra_vec.is_empty()) hp_kern_extra_vec(0) = hp_rho;
    const double cond_max = 1.0e4;

    // The eta data, and the buffer whose strict upper triangle holds C^(rdot), at a given hp_gamma.
    const auto build_at_hp = [&](const arma::vec& hp_gamma_at) {
        const double* hp_kern_extra_ptr = gpflash::detail::calc_hp_kern_extra_ptr(hp_kern_extra_vec);
        const arma::mat Ydot_nfeature_ntrain_mat = X_ndim_ntrain_mat.each_col() % hp_gamma_at;
        gpflash::detail::EtaTerms kmat_eta_terms;
        arma::mat kern_common_rdot_mat_upper = gpflash::detail::build_kmat_lower_tri(
                kern_fun, Ydot_nfeature_ntrain_mat, hp_kern_extra_ptr, /*store_hp_grad_factors=*/true);
        kmat_eta_terms = gpflash::detail::build_eta_terms(
                kern_fun.is_kern_diag_unity(),
                kern_common_rdot_mat_upper,
                cond_max,
                gpflash::EigMaxUbMethod::Gershgorin,
                /*want_active_row=*/true);

        return std::make_pair(std::move(kmat_eta_terms), std::move(kern_common_rdot_mat_upper));
    };

    const auto base_case = build_at_hp(hp_gamma_vec);
    const gpflash::detail::EtaTerms& kmat_eta_terms = base_case.first;
    const int idx_K_corr_max_abs_row_sum = kmat_eta_terms.idx_M_precon_max_abs_row_sum;

    // The active row's c^(rdot) values, gathered out of the buffer the way the MLL gradient loop gathers them.
    const arma::vec active_row_kern_common_rdot_vec =
            harvest_active_row_kern_common_rdot(base_case.second, idx_K_corr_max_abs_row_sum);

    const auto hp_grad_inputs =
            gpflash::kernel::make_hp_grad_inputs(X_ndim_ntrain_mat, X_ndim_ntrain_mat, hp_gamma_vec, hp_kern_extra_vec);
    const arma::vec d_eta_d_hp_gamma_vec = gpflash::detail::calc_eta_term_hp_grad(
                                                   kern_fun,
                                                   hp_grad_inputs,
                                                   kmat_eta_terms,
                                                   kern_fun.calc_d_kern_diag_d_hp_gamma_mat(hp_grad_inputs),
                                                   /*d_kern_diag_d_hp_kern_extra_ntrain_nextra_mat=*/arma::mat{},
                                                   active_row_kern_common_rdot_vec,
                                                   /*trace_psi_w=*/1.0,
                                                   gpflash::SymmetrySpec{},
                                                   arma::vec{})
                                                   .d_mll_d_hp_gamma_vec;

    double worst = 0.0;
    bool row_stable = true;
    for (int i_dim = 0; i_dim < n_dim; ++i_dim) {
        arma::vec hp_gamma_plus_vec = hp_gamma_vec, hp_gamma_minus_vec = hp_gamma_vec;
        hp_gamma_plus_vec(i_dim) += kFdStepKern;
        hp_gamma_minus_vec(i_dim) -= kFdStepKern;
        const auto case_plus = build_at_hp(hp_gamma_plus_vec);
        const auto case_minus = build_at_hp(hp_gamma_minus_vec);
        if (case_plus.first.idx_M_precon_max_abs_row_sum != idx_K_corr_max_abs_row_sum
            || case_minus.first.idx_M_precon_max_abs_row_sum != idx_K_corr_max_abs_row_sum) {
            row_stable = false;  // the arg max moved; the derivative is not defined across the fd_step
            continue;
        }
        const double d_bound =
                (case_plus.first.eig_max_ub_M_precon - case_minus.first.eig_max_ub_M_precon) / (2.0 * kFdStepKern);
        const double grad_fd = d_bound / (cond_max - 1.0);
        worst = std::max(
                worst, std::fabs(d_eta_d_hp_gamma_vec(i_dim) - grad_fd) / std::max(std::fabs(grad_fd), kFdScaleFloor));
    }
    char detail[kDetailLen];
    std::snprintf(detail, sizeof(detail), "worst rel %.2e  active row stable %d", worst, static_cast<int>(row_stable));

    return report(name, row_stable && worst < kTolFdEta, detail);
}

// Asserts calc_eta_precon_scale_grad returns zero for a unit-diagonal kernel and something nonzero for a
// varying-diagonal one.
bool test_precon_scale_grad(
        const gpflash::kernel::KernelBase& kern_fun, const char* name, const bool expect_contribution) {
    const int n_dim = 3, n_train = 12;
    const arma::mat X_ndim_ntrain_mat = make_points(n_dim, n_train);
    arma::vec hp_gamma_vec(n_dim);
    arma::vec hp_kern_extra_vec(kern_fun.n_hp_kern_extra(n_dim));
    for (int i_dim = 0; i_dim < n_dim; ++i_dim) hp_gamma_vec(i_dim) = 0.9 + 0.2 * i_dim;
    if (!hp_kern_extra_vec.is_empty()) hp_kern_extra_vec(0) = kRatQuadHpRho;

    const arma::mat Ydot_nfeature_ntrain_mat = X_ndim_ntrain_mat.each_col() % hp_gamma_vec;
    arma::mat K_mat = gpflash::detail::build_kmat_lower_tri(
            kern_fun,
            Ydot_nfeature_ntrain_mat,
            gpflash::detail::calc_hp_kern_extra_ptr(hp_kern_extra_vec),
            /*store_hp_grad_factors=*/true);
    const gpflash::detail::EtaTerms kmat_eta_terms = gpflash::detail::build_eta_terms(
            kern_fun.is_kern_diag_unity(),
            K_mat,
            /*cond_max=*/1.0e4,
            gpflash::EigMaxUbMethod::Gershgorin,
            /*want_active_row=*/true);

    const auto hp_grad_inputs =
            gpflash::kernel::make_hp_grad_inputs(X_ndim_ntrain_mat, X_ndim_ntrain_mat, hp_gamma_vec, hp_kern_extra_vec);

    const arma::vec precon_scale_grad_vec = gpflash::detail::calc_eta_precon_scale_grad(
            kern_fun.calc_d_kern_diag_d_hp_gamma_mat(hp_grad_inputs), kmat_eta_terms, n_dim);
    const bool precon_scale_contributed = arma::any(precon_scale_grad_vec != 0.0);

    // Still n_dim long on the zero path, so a caller can add it unconditionally.
    const bool shapes_ok = precon_scale_grad_vec.n_elem == static_cast<arma::uword>(n_dim);

    const bool pass = shapes_ok && precon_scale_contributed == expect_contribution;
    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "precon_scale=%d expected=%d",
            static_cast<int>(precon_scale_contributed),
            static_cast<int>(expect_contribution));

    return report(name, pass, detail);
}

}  // namespace

// Test 3: the eta gradient needs d(diag K)/d(hp_kern_extra) whenever P is not the identity, and only the symmetry
// path supplies it. A RatQuad with hand-made preconditioning scales and an empty extras derivative must be refused
// by name, since silently dropping the d(P) term would return a wrong gradient. No library path builds this case
// today, so the terms are assembled by hand.
bool test_extras_without_diag_grad_refused(const char* name) {
    const gpflash::kernel::KernelBase& kern_fun = gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::RatQuad);
    const int n_dim = 2, n_train = 3;
    const arma::mat X_ndim_ntrain_mat = arma::ones<arma::mat>(n_dim, n_train);
    const arma::vec hp_gamma_vec = arma::ones<arma::vec>(n_dim);
    const arma::vec hp_kern_extra_vec{kRatQuadHpRho};
    const auto hp_grad_inputs =
            gpflash::kernel::make_hp_grad_inputs(X_ndim_ntrain_mat, X_ndim_ntrain_mat, hp_gamma_vec, hp_kern_extra_vec);

    gpflash::detail::EtaTerms kmat_eta_terms;
    kmat_eta_terms.cond_max = 1.0e4;
    kmat_eta_terms.precon_scales.M_diag_vec = arma::ones<arma::vec>(n_train);  // non-empty: P is not the identity
    kmat_eta_terms.precon_scales.precon_scale_vec = arma::ones<arma::vec>(n_train);
    kmat_eta_terms.precon_scales.inv_precon_scale_vec = arma::ones<arma::vec>(n_train);

    // ErrorNotImplemented by type, naming the kernel, so the refusal is this one and not an input check.
    bool refused = false;
    try {
        gpflash::detail::calc_eta_term_hp_grad(
                kern_fun,
                hp_grad_inputs,
                kmat_eta_terms,
                kern_fun.calc_d_kern_diag_d_hp_gamma_mat(hp_grad_inputs),
                /*d_kern_diag_d_hp_kern_extra_ntrain_nextra_mat=*/arma::mat{},
                arma::zeros<arma::vec>(n_train),
                /*trace_psi_w=*/1.0,
                gpflash::SymmetrySpec{},
                arma::vec{});
    } catch (const gpflash::ErrorNotImplemented& error) {
        refused = std::string(error.what()).find(kern_fun.kernel_name()) != std::string::npos;
    }
    return report(name, refused);
}

int main() {
    std::printf("== GpFlash kernel matrix nugget gradient ==\n");
    bool ok = true;
    ok &= test_extras_without_diag_grad_refused("extras_without_diag_grad_refused");
    ok &= test_precon_scale_grad(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::SqExp),
            "precon_scale_grad/SqExp",
            /*expect_contribution=*/false);
    ok &= test_precon_scale_grad(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Quadratic),
            "precon_scale_grad/Quadratic",
            /*expect_contribution=*/true);
    ok &= test_eta_grad_vs_fd(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::SqExp), "eta_grad_vs_fd/SqExp", kHpRhoUnused);
    ok &= test_eta_grad_vs_fd(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Matern52),
            "eta_grad_vs_fd/Matern52",
            kHpRhoUnused);
    ok &= test_eta_grad_vs_fd(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Matern32),
            "eta_grad_vs_fd/Matern32",
            kHpRhoUnused);
    ok &= test_eta_grad_vs_fd(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::RatQuad), "eta_grad_vs_fd/RatQuad", kRatQuadHpRho);
    ok &= test_eta_grad_vs_fd(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Quadratic),
            "eta_grad_vs_fd/Quadratic",
            kHpRhoUnused);
    std::printf("\n%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");

    return ok ? 0 : 1;
}
