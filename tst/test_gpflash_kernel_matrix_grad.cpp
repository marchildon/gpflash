// KernelMatrixGrad.hpp on its own. What this file asserts:
//
//   1) kern_diag_term      calc_kern_diag_hp_grad is zero for a unit-diagonal kernel, nonzero for Quadratic
//   2) kern_grad_vs_fd     the pair term plus the diagonal term equal a finite difference of <Psi, K>, in hp_gamma
//                          and in the kernel extras
//   3) active_row_harvest  the harvested c^(rdot) matches the strict upper triangle it was read from
//
// Check 2 is what pins the factor of two: the pair term covers the strict lower triangle doubled and
// calc_kern_diag_hp_grad the diagonal, so their sum is the whole d<Psi, K>/d(hp_gamma) and any missing or
// duplicated half shows up against the difference. Quadratic's varying diagonal is what reaches
// calc_kern_diag_hp_grad's nonzero branch.
#include <armadillo>
#include <cmath>
#include <cstdio>

#include "GpFlashTestSupport.hpp"
#include "gpflash/assembly/KernelMatrix.hpp"
#include "gpflash/assembly/KernelMatrixGrad.hpp"
#include "gpflash/inputs/HpVal.hpp"         // calc_hp_kern_extra_ptr
#include "gpflash/inputs/SymmetrySpec.hpp"  // SymmetrySpec, the no-symmetry argument
#include "gpflash/kernels/HpGradInputs.hpp"
#include "gpflash/kernels/KernelDispatch.hpp"  // kernel_by_id
#include "gpflash/kernels/radial/KernelRadial.hpp"

using gpflash_test::harvest_active_row_kern_common_rdot;
using gpflash_test::kDetailLen;
using gpflash_test::kFdStepKern;
using gpflash_test::kHpRhoUnused;
using gpflash_test::kRatQuadHpRho;
using gpflash_test::kRelScaleFloor;
using gpflash_test::kTolFd;
using gpflash_test::make_points;
using gpflash_test::report;

namespace {

// The buffer the gradient reads, in the layout the pipeline hands it over in, plus the symmetric matrix its
// lower triangle stands for.
struct GradFixture {
    arma::mat Sigma_inv_mat_lower;  // lower with diag Sigma^-1, strict upper c^(rdot)
    arma::mat Sigma_inv_full_mat;   // the same Sigma^-1, mirrored, for the dense reference
};

// Fill the strict upper triangle with real c^(rdot) values from an assembly pass, then put an arbitrary
// symmetric stand-in for Sigma^-1 below and on the diagonal. The stand-in is a closed form in the indices, so
// it is the same at every point of a finite difference and only K varies.
GradFixture make_grad_fixture(
        const gpflash::kernel::KernelBase& kern_fun,
        const arma::mat& X_ndim_ntrain_mat,
        const arma::vec& hp_gamma_vec,
        const arma::vec& hp_kern_extra_vec) {
    const int n_train = static_cast<int>(X_ndim_ntrain_mat.n_cols);
    const arma::mat Ydot_nfeature_ntrain_mat = X_ndim_ntrain_mat.each_col() % hp_gamma_vec;
    GradFixture grad_fixture;
    grad_fixture.Sigma_inv_mat_lower = gpflash::detail::build_kmat_lower_tri(
            kern_fun,
            Ydot_nfeature_ntrain_mat,
            gpflash::detail::calc_hp_kern_extra_ptr(hp_kern_extra_vec),
            /*store_hp_grad_factors=*/true);

    for (int i_col = 0; i_col < n_train; ++i_col)
        for (int i_row = i_col; i_row < n_train; ++i_row)
            grad_fixture.Sigma_inv_mat_lower(i_row, i_col) = 1.0 / (2.0 + i_row + i_col) + (i_row == i_col ? 1.5 : 0.0);
    grad_fixture.Sigma_inv_full_mat = arma::symmatl(grad_fixture.Sigma_inv_mat_lower);

    return grad_fixture;
}

// Asserts calc_kern_diag_hp_grad returns zero for a unit-diagonal kernel and something nonzero for a
// varying-diagonal one, at the right length either way.
bool test_kern_diag_term(
        const gpflash::kernel::KernelBase& kern_fun, const char* name, const bool expect_contribution) {
    const int n_dim = 3, n_train = 12;
    const arma::mat X_ndim_ntrain_mat = make_points(n_dim, n_train);
    arma::vec hp_gamma_vec(n_dim);
    arma::vec hp_kern_extra_vec(kern_fun.n_hp_kern_extra(n_dim));
    for (int i_dim = 0; i_dim < n_dim; ++i_dim) hp_gamma_vec(i_dim) = 0.9 + 0.2 * i_dim;
    if (!hp_kern_extra_vec.is_empty()) hp_kern_extra_vec(0) = kRatQuadHpRho;

    const auto grad_fixture = make_grad_fixture(kern_fun, X_ndim_ntrain_mat, hp_gamma_vec, hp_kern_extra_vec);
    const auto hp_grad_inputs =
            gpflash::kernel::make_hp_grad_inputs(X_ndim_ntrain_mat, X_ndim_ntrain_mat, hp_gamma_vec, hp_kern_extra_vec);
    const arma::vec psi_diag_vec =
            gpflash::detail::calc_psi_diag_vec(make_points(n_train, 1), grad_fixture.Sigma_inv_mat_lower);

    const arma::vec kern_diag_grad_vec = gpflash::detail::calc_kern_diag_hp_grad(
            kern_fun.calc_d_kern_diag_d_hp_gamma_mat(hp_grad_inputs), psi_diag_vec, n_dim);

    const bool contributed = arma::any(kern_diag_grad_vec != 0.0);
    const bool shape_ok = kern_diag_grad_vec.n_elem == static_cast<arma::uword>(n_dim);
    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "contributed=%d expected=%d shape_ok=%d",
            static_cast<int>(contributed),
            static_cast<int>(expect_contribution),
            static_cast<int>(shape_ok));

    return report(name, shape_ok && contributed == expect_contribution, detail);
}

// Asserts the pair term and the diagonal term together equal d<Psi, K>/d(hp_gamma) by central difference.
bool test_kern_grad_vs_fd(const gpflash::kernel::KernelBase& kern_fun, const char* name, const double hp_rho) {
    const int n_dim = 3, n_train = 14;
    const arma::mat X_ndim_ntrain_mat = make_points(n_dim, n_train);
    arma::vec hp_gamma_vec(n_dim);
    arma::vec hp_kern_extra_vec(kern_fun.n_hp_kern_extra(n_dim));
    for (int i_dim = 0; i_dim < n_dim; ++i_dim) hp_gamma_vec(i_dim) = 0.8 + 0.3 * i_dim;
    if (!hp_kern_extra_vec.is_empty()) hp_kern_extra_vec(0) = hp_rho;

    const auto grad_fixture = make_grad_fixture(kern_fun, X_ndim_ntrain_mat, hp_gamma_vec, hp_kern_extra_vec);
    const arma::vec lin_sol_vec = make_points(n_train, 1);
    const arma::mat Psi_mat = 0.5 * (lin_sol_vec * lin_sol_vec.t() - grad_fixture.Sigma_inv_full_mat);

    // The analytic gradient: the doubled strict lower triangle, then the diagonal.
    const auto hp_grad_inputs =
            gpflash::kernel::make_hp_grad_inputs(X_ndim_ntrain_mat, X_ndim_ntrain_mat, hp_gamma_vec, hp_kern_extra_vec);
    const gpflash::detail::KernTermHpGrad kern_term_hp_grad = gpflash::detail::calc_kern_term_hp_grad(
            kern_fun,
            hp_grad_inputs,
            lin_sol_vec,
            grad_fixture.Sigma_inv_mat_lower,
            /*idx_K_corr_max_abs_row_sum=*/-1,
            gpflash::SymmetrySpec{},
            arma::vec{});
    // Every kernel with extras here has a unit diagonal, so the pair term is the whole extras gradient.
    const arma::vec& d_mll_d_hp_kern_extra_vec = kern_term_hp_grad.d_mll_d_hp_kern_extra_vec;
    arma::vec d_mll_d_hp_gamma_vec = kern_term_hp_grad.d_mll_d_hp_gamma_vec;
    d_mll_d_hp_gamma_vec += gpflash::detail::calc_kern_diag_hp_grad(
            kern_fun.calc_d_kern_diag_d_hp_gamma_mat(hp_grad_inputs),
            gpflash::detail::calc_psi_diag_vec(lin_sol_vec, grad_fixture.Sigma_inv_mat_lower),
            n_dim);

    // <Psi, K> at shifted hyperparameters, with Psi held fixed.
    const auto calc_psi_dot_kmat = [&](const arma::vec& hp_gamma_at, const arma::vec& hp_kern_extra_at) {
        const arma::mat Ydot_nfeature_ntrain_mat = X_ndim_ntrain_mat.each_col() % hp_gamma_at;
        const arma::mat K_mat = gpflash::detail::build_kmat_lower_tri(
                kern_fun,
                Ydot_nfeature_ntrain_mat,
                gpflash::detail::calc_hp_kern_extra_ptr(hp_kern_extra_at),
                /*store_hp_grad_factors=*/false);

        return arma::accu(Psi_mat % arma::symmatl(K_mat));
    };

    double worst = 0.0;
    for (int i_dim = 0; i_dim < n_dim; ++i_dim) {
        arma::vec hp_gamma_plus_vec = hp_gamma_vec, hp_gamma_minus_vec = hp_gamma_vec;
        hp_gamma_plus_vec(i_dim) += kFdStepKern;
        hp_gamma_minus_vec(i_dim) -= kFdStepKern;
        const double psi_dot_kmat_plus = calc_psi_dot_kmat(hp_gamma_plus_vec, hp_kern_extra_vec);
        const double psi_dot_kmat_minus = calc_psi_dot_kmat(hp_gamma_minus_vec, hp_kern_extra_vec);
        const double grad_fd = (psi_dot_kmat_plus - psi_dot_kmat_minus) / (2.0 * kFdStepKern);
        worst = std::max(
                worst, std::fabs(d_mll_d_hp_gamma_vec(i_dim) - grad_fd) / std::max(std::fabs(grad_fd), kRelScaleFloor));
    }

    // The extras gradient, whose factor of two the pair term applies separately from the hp_gamma one.
    for (arma::uword i_extra = 0; i_extra < hp_kern_extra_vec.n_elem; ++i_extra) {
        arma::vec hp_kern_extra_plus_vec = hp_kern_extra_vec, hp_kern_extra_minus_vec = hp_kern_extra_vec;
        hp_kern_extra_plus_vec(i_extra) += kFdStepKern;
        hp_kern_extra_minus_vec(i_extra) -= kFdStepKern;
        const double psi_dot_kmat_plus = calc_psi_dot_kmat(hp_gamma_vec, hp_kern_extra_plus_vec);
        const double psi_dot_kmat_minus = calc_psi_dot_kmat(hp_gamma_vec, hp_kern_extra_minus_vec);
        const double grad_fd = (psi_dot_kmat_plus - psi_dot_kmat_minus) / (2.0 * kFdStepKern);
        worst = std::max(
                worst,
                std::fabs(d_mll_d_hp_kern_extra_vec(i_extra) - grad_fd) / std::max(std::fabs(grad_fd), kRelScaleFloor));
    }
    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "worst rel %.2e over %d hp_gamma, %d extras",
            worst,
            n_dim,
            static_cast<int>(hp_kern_extra_vec.n_elem));

    return report(name, worst < kTolFd, detail);
}

// Asserts the harvested active row equals the c^(rdot) entries the strict upper triangle holds for that row.
//
// The sweep collects them one per column, so an off-by-one in the column offset is what this catches.
bool test_active_row_harvest(const char* name, const int idx_active_row) {
    const gpflash::kernel::KernelBase& kern_fun = gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Matern52);
    const int n_dim = 2, n_train = 9;
    const arma::mat X_ndim_ntrain_mat = make_points(n_dim, n_train);
    arma::vec hp_gamma_vec(n_dim);
    for (int i_dim = 0; i_dim < n_dim; ++i_dim) hp_gamma_vec(i_dim) = 1.1 + 0.2 * i_dim;
    const arma::vec hp_kern_extra_vec;  // Matern52 declares none

    const auto grad_fixture = make_grad_fixture(kern_fun, X_ndim_ntrain_mat, hp_gamma_vec, hp_kern_extra_vec);
    const auto hp_grad_inputs =
            gpflash::kernel::make_hp_grad_inputs(X_ndim_ntrain_mat, X_ndim_ntrain_mat, hp_gamma_vec, hp_kern_extra_vec);
    const arma::vec active_row_vec = gpflash::detail::calc_kern_term_hp_grad(
                                             kern_fun,
                                             hp_grad_inputs,
                                             make_points(n_train, 1),
                                             grad_fixture.Sigma_inv_mat_lower,
                                             idx_active_row,
                                             gpflash::SymmetrySpec{},
                                             arma::vec{})
                                             .active_row_kern_common_rdot_vec;

    // The same values read straight out of the buffer by a hand transcription of the layout, which shares no
    // code with the library's sweep.
    const arma::vec expected_vec =
            harvest_active_row_kern_common_rdot(grad_fixture.Sigma_inv_mat_lower, idx_active_row);
    const double err =
            arma::abs(active_row_vec - expected_vec).max() / std::max(arma::abs(expected_vec).max(), kRelScaleFloor);

    char detail[kDetailLen];
    std::snprintf(detail, sizeof(detail), "active row %d rel %.1e", idx_active_row, err);

    return report(name, err == 0.0, detail);
}

}  // namespace

int main() {
    std::printf("== GpFlash kernel matrix gradient ==\n");
    bool ok = true;
    ok &= test_kern_diag_term(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::SqExp),
            "kern_diag_term/SqExp",
            /*expect_contribution=*/false);
    ok &= test_kern_diag_term(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Quadratic),
            "kern_diag_term/Quadratic",
            /*expect_contribution=*/true);
    ok &= test_kern_grad_vs_fd(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::SqExp), "kern_grad_vs_fd/SqExp", kHpRhoUnused);
    ok &= test_kern_grad_vs_fd(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Matern52),
            "kern_grad_vs_fd/Matern52",
            kHpRhoUnused);
    ok &= test_kern_grad_vs_fd(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Matern32),
            "kern_grad_vs_fd/Matern32",
            kHpRhoUnused);
    ok &= test_kern_grad_vs_fd(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::RatQuad),
            "kern_grad_vs_fd/RatQuad",
            kRatQuadHpRho);
    ok &= test_kern_grad_vs_fd(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Quadratic),
            "kern_grad_vs_fd/Quadratic",
            kHpRhoUnused);
    for (const int idx_active_row : {0, 4, 8}) ok &= test_active_row_harvest("active_row_harvest", idx_active_row);
    std::printf("\n%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");

    return ok ? 0 : 1;
}
