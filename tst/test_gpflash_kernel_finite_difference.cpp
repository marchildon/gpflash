// What each kernel computes for itself: its values against its own unfactorized forms, and its analytic
// derivatives against central finite differences.
//
//   1) factorization                     kern_val and kern_common_rdot match the unfactorized forms
//   2) coincident_points                 duplicate points give kern_val = 1 and no NaN, and a negative polarized
//                                        distance clamps to zero
//   3) kern_common_rdot_is_derivative    the shared factor is 2 d(kern_val)/d(rdot_sq_norm)
//   4) hp_gamma_grad_vs_fd               add_hp_gradient_contribution is d(sum k)/d(hp_gamma_l)
//   5) rat_quad_shape_grad_vs_fd         the extra-hyperparameter gradient is d(sum k)/d(hp_rho)
//
// Each finite-difference regime needs its own step size and tolerance; GpFlashTestSupport.hpp holds those.
#include <armadillo>
#include <cmath>
#include <cstdio>
#include <vector>

#include "GpFlashTestSupport.hpp"
#include "gpflash/assembly/KernelMatrix.hpp"
#include "gpflash/kernels/HpGradInputs.hpp"
#include "gpflash/kernels/KernelDispatch.hpp"  // kernel_by_id
#include "gpflash/kernels/radial/KernelRadial.hpp"
#include "gpflash/kernels/radial/Matern32.hpp"
#include "gpflash/kernels/radial/Matern52.hpp"
#include "gpflash/kernels/radial/RatQuad.hpp"
#include "gpflash/kernels/radial/SqExp.hpp"

using gpflash_test::kDetailLen;
using gpflash_test::kFdScaleFloor;
using gpflash_test::kFdStepKern;
using gpflash_test::kRatQuadHpRho;
using gpflash_test::kRatQuadHpRhoNearZero;
using gpflash_test::kTolExact;
using gpflash_test::kTolFd;
using gpflash_test::make_points;
using gpflash_test::report;

namespace {

// Asserts the factorized value loop agrees with the kernel's own unfactorized forms.
//
// Every other reference in the package calls the factorized path, so an error in it would agree with itself
// everywhere. Each kernel states its unfactorized definition beside itself, as a test-only static rather than an
// interface method, which is why these arrive as function pointers. No per-kernel math lives here.
using KernValRefFn = double (*)(double, const double*);

bool test_factorization(
        const gpflash::kernel::KernelBase& kern_fun,
        const KernValRefFn calc_kern_val_ref,
        const KernValRefFn calc_kern_common_rdot_ref,
        const char* name,
        const double hp_kern_extra_val) {
    const double* hp_kern_extra_ptr = (kern_fun.n_hp_kern_extra(1) > 0) ? &hp_kern_extra_val : nullptr;
    // Separations spanning the range assembly produces, including exactly zero.
    const std::vector<double> rdot_norm_vec = {0.0, 1.0e-4, 0.22, 0.71, 1.0, 1.92, 3.46, 7.75};

    double worst_kern_val = 0.0, worst_kern_common_rdot = 0.0;
    for (const double rdot_norm : rdot_norm_vec) {
        const double rdot_sq_norm = rdot_norm * rdot_norm;
        double kern_val = 0.0, kern_common_rdot = 0.0;
        kern_val = rdot_sq_norm;
        kern_fun.calc_kern_val_and_common_rdot_inplace(&kern_val, 1, hp_kern_extra_ptr, &kern_common_rdot);
        const double kern_val_ref = calc_kern_val_ref(rdot_sq_norm, hp_kern_extra_ptr);
        const double kern_common_rdot_ref = calc_kern_common_rdot_ref(rdot_sq_norm, hp_kern_extra_ptr);

        worst_kern_val = std::max(
                worst_kern_val, std::fabs(kern_val - kern_val_ref) / std::max(std::fabs(kern_val_ref), kFdScaleFloor));
        worst_kern_common_rdot = std::max(
                worst_kern_common_rdot,
                std::fabs(kern_common_rdot - kern_common_rdot_ref)
                        / std::max(std::fabs(kern_common_rdot_ref), kFdScaleFloor));
    }
    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "worst rel: kern_val %.2e  kern_common_rdot %.2e",
            worst_kern_val,
            worst_kern_common_rdot);

    return report(name, worst_kern_val < kTolExact && worst_kern_common_rdot < kTolExact, detail);
}

// Asserts duplicate training points give k = 1 with no NaN anywhere in K, and that a polarized distance just
// below zero is clamped to zero.
//
// The polarized distance ||y_i||^2 + ||y_j||^2 - 2 y_i . y_j cancels only to rounding: where the norms and the
// cross term differ by an ulp it lands below zero, where the Matern kernels' sqrt returns NaN and it spreads into
// K, the eigenvalue bound and the factorization. Assembly reads its norms off the same DSYRK diagonal as the cross
// terms, so its duplicates need not go negative; the probe below feeds the negative case to both clamping entry
// points directly.
bool test_coincident_points(
        const gpflash::kernel::KernelBase& kern_fun, const char* name, const double hp_kern_extra_val) {
    const int n_dim = 4, n_train = 300;  // above PERF_OMP_MIN_N_POINT, so the parallel sweep runs too
    arma::mat X_ndim_ntrain_mat = make_points(n_dim, n_train);
    // Every second point duplicates its predecessor, so n_train / 2 pairs are coincident.
    for (int i_train = 1; i_train < n_train; i_train += 2)
        X_ndim_ntrain_mat.col(i_train) = X_ndim_ntrain_mat.col(i_train - 1);

    arma::vec hp_gamma_vec(n_dim);
    for (int i_dim = 0; i_dim < n_dim; ++i_dim) hp_gamma_vec(i_dim) = 0.6 + 0.35 * i_dim;
    const double* hp_kern_extra_ptr = (kern_fun.n_hp_kern_extra(n_dim) > 0) ? &hp_kern_extra_val : nullptr;

    const arma::mat Ydot_nfeature_ntrain_mat = X_ndim_ntrain_mat.each_col() % hp_gamma_vec;
    const arma::mat K_mat = gpflash::detail::build_kmat_lower_tri(
            kern_fun, Ydot_nfeature_ntrain_mat, hp_kern_extra_ptr, /*store_hp_grad_factors=*/true);

    int n_nan = 0;
    double worst_coincident_err = 0.0;
    for (int i_col = 0; i_col < n_train; ++i_col) {
        for (int i_row = i_col; i_row < n_train; ++i_row) {
            if (std::isnan(K_mat(i_row, i_col))) ++n_nan;
        }
        // The duplicate of column i_col, where one exists: zero distance up to rounding, so k matches the diagonal's 1.
        const int i_twin = (i_col % 2 == 0) ? i_col + 1 : i_col - 1;
        if (i_twin > i_col && i_twin < n_train)
            worst_coincident_err = std::max(worst_coincident_err, std::fabs(K_mat(i_twin, i_col) - 1.0));
    }

    // A cross term one ulp above both norms polarizes to -2 ulp, which the clamp must turn into a zero distance.
    const double probe_sq_norm = 1.0;
    const double probe_inner_prod = std::nextafter(probe_sq_norm, arma::datum::inf);
    double probe_kern_val = probe_inner_prod, probe_kern_common_rdot = 0.0;
    kern_fun.calc_kmat_column_inplace(
            &probe_kern_val, 1, probe_sq_norm, &probe_sq_norm, hp_kern_extra_ptr, &probe_kern_common_rdot);
    double probe_kern_input = probe_inner_prod;
    kern_fun.calc_kern_input_inplace(&probe_kern_input, 1, &probe_sq_norm, probe_sq_norm);
    const bool is_clamped = probe_kern_val == 1.0 && std::isfinite(probe_kern_common_rdot) && probe_kern_input == 0.0;

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "NaN=%d  worst |k(dup) - 1| %.2e  probe k=%.17g input=%.2e",
            n_nan,
            worst_coincident_err,
            probe_kern_val,
            probe_kern_input);

    return report(name, n_nan == 0 && worst_coincident_err < kTolExact && is_clamped, detail);
}

// Asserts kern_common_rdot == 2 d(kern_val)/d(rdot_sq_norm) for a stationary kernel.
//
// The 2 is a chain rule the gradient path splits in half. rdot_sq_norm = sum_l (hp_gamma_l r_l)^2, so
// d(rdot_sq_norm)/d(hp_gamma_l) = 2 hp_gamma_l r_l^2. add_hp_gradient_contribution multiplies by
// hp_gamma_l r_l^2 with no 2, which leaves kern_common_rdot to carry it. Lose the 2 and every
// hyperparameter gradient comes out halved.
//
// expected_common_rdot_at_zero is hard-coded per kernel by the caller, so this function never asks the kernel
// under test what its own answer should be.
bool test_kern_common_rdot_is_derivative(
        const gpflash::kernel::KernelBase& kern_fun,
        const char* name,
        const double hp_kern_extra_val,
        const double expected_common_rdot_at_zero) {
    const double* hp_kern_extra_ptr = (kern_fun.n_hp_kern_extra(1) == 1) ? &hp_kern_extra_val : nullptr;

    // Zero distance takes hard-coded expectations rather than a finite difference: a value carrying an odd
    // power of ||rdot_vec|| converges only as O(sqrt(fd_step)) there, so no step reaches the tolerance.
    {
        double kern_at_zero, common_rdot_at_zero;
        kern_at_zero = 0.0;
        kern_fun.calc_kern_val_and_common_rdot_inplace(&kern_at_zero, 1, hp_kern_extra_ptr, &common_rdot_at_zero);
        if (std::fabs(kern_at_zero - 1.0) > kTolExact
            || std::fabs(common_rdot_at_zero - expected_common_rdot_at_zero) > kTolExact)
            return report(name, false, "kern_val or kern_common_rdot at zero distance is not the expected value");
    }

    // Away from zero, differenced in rdot_sq_norm directly. Each sample is far enough from 0 that fd_step is
    // small against it, which the central form needs.
    const double rdot_sq_norm_list[] = {0.05, 0.5, 1.0, 3.7, 12.0};
    double worst = 0.0;
    for (const double rdot_sq_norm : rdot_sq_norm_list) {
        double kern_val = 0.0, kern_common_rdot_analytic = 0.0;
        kern_val = rdot_sq_norm;
        kern_fun.calc_kern_val_and_common_rdot_inplace(&kern_val, 1, hp_kern_extra_ptr, &kern_common_rdot_analytic);

        const double fd_step = kFdStepKern;
        double kern_val_plus = 0.0, kern_val_minus = 0.0, kern_common_rdot_unused = 0.0;
        kern_val_plus = rdot_sq_norm + fd_step;
        kern_fun.calc_kern_val_and_common_rdot_inplace(&kern_val_plus, 1, hp_kern_extra_ptr, &kern_common_rdot_unused);
        kern_val_minus = rdot_sq_norm - fd_step;
        kern_fun.calc_kern_val_and_common_rdot_inplace(&kern_val_minus, 1, hp_kern_extra_ptr, &kern_common_rdot_unused);
        const double reference = 2.0 * (kern_val_plus - kern_val_minus) / (2.0 * fd_step);
        const double scale = std::max(std::fabs(reference), kFdScaleFloor);
        worst = std::max(worst, std::fabs(kern_common_rdot_analytic - reference) / scale);
    }
    char detail[64];
    std::snprintf(detail, sizeof(detail), "worst rel %.2e", worst);

    return report(name, worst < kTolFd, detail);
}

// Asserts add_hp_gradient_contribution accumulates d(sum_{i>j} kern_ij)/d(hp_gamma_l).
//
// The weights fed in are kern_common_rdot, exactly as the training pipeline supplies them, so this covers the
// two halves together: the kernel's derivative factor and the per-dimension term the base class applies to it.
// Feeding unit weights instead would not satisfy the identity, which is what makes the pairing load-bearing.
bool test_hp_gamma_grad_vs_fd(
        const gpflash::kernel::KernelBase& kern_fun, const char* name, const double hp_kern_extra_val) {
    const int n_dim = 3, n_train = 7;
    const double* hp_kern_extra_ptr = (kern_fun.n_hp_kern_extra(n_dim) == 1) ? &hp_kern_extra_val : nullptr;
    const arma::mat X_ndim_ntrain_mat = make_points(n_dim, n_train);
    arma::vec hp_gamma_vec(n_dim);
    for (int i_dim = 0; i_dim < n_dim; ++i_dim) hp_gamma_vec(i_dim) = 0.8 + 0.3 * i_dim;

    // sum_{i>j} k(i, j), the scalar the gradient below must differentiate.
    const auto kern_sum = [&](const arma::vec& hp_gamma_vec) {
        const arma::mat Ydot_nfeature_ntrain_mat = X_ndim_ntrain_mat.each_col() % hp_gamma_vec;
        double total = 0.0;
        for (int i_col = 0; i_col < n_train; ++i_col)
            for (int i_row = i_col + 1; i_row < n_train; ++i_row) {
                double kern_val, kern_common_rdot_unused;
                kern_val = arma::accu(
                        arma::square(Ydot_nfeature_ntrain_mat.col(i_row) - Ydot_nfeature_ntrain_mat.col(i_col)));
                kern_fun.calc_kern_val_and_common_rdot_inplace(
                        &kern_val, 1, hp_kern_extra_ptr, &kern_common_rdot_unused);
                total += kern_val;
            }
        return total;
    };

    // One precompute signature serves every kernel: interface hyperparameters and evaluated points, which for a
    // radial kernel are the training points themselves.
    arma::vec hp_kern_extra_vec;
    if (kern_fun.n_hp_kern_extra(n_dim) > 0) hp_kern_extra_vec = arma::vec(1).fill(hp_kern_extra_val);
    const auto hp_grad_inputs =
            gpflash::kernel::make_hp_grad_inputs(X_ndim_ntrain_mat, X_ndim_ntrain_mat, hp_gamma_vec, hp_kern_extra_vec);
    arma::vec d_sum_d_hp_gamma_vec(n_dim, arma::fill::zeros);

    const arma::mat Ydot_nfeature_ntrain_mat = X_ndim_ntrain_mat.each_col() % hp_gamma_vec;
    for (int i_col = 0; i_col < n_train; ++i_col) {
        const int n_below = n_train - i_col - 1;
        if (n_below <= 0) continue;
        std::vector<double> A_vec(n_below);
        for (int i_below = 0; i_below < n_below; ++i_below) {
            double kern_val = 0.0, kern_common_rdot = 0.0;
            kern_val = arma::accu(arma::square(
                    Ydot_nfeature_ntrain_mat.col(i_col + 1 + i_below) - Ydot_nfeature_ntrain_mat.col(i_col)));
            kern_fun.calc_kern_val_and_common_rdot_inplace(&kern_val, 1, hp_kern_extra_ptr, &kern_common_rdot);
            A_vec[i_below] = kern_common_rdot;
        }
        kern_fun.add_hp_gradient_contribution(
                hp_grad_inputs,
                i_col,
                i_col + 1,
                i_col + 1 + n_below,
                A_vec.data(),
                d_sum_d_hp_gamma_vec.memptr(),
                nullptr);
    }

    const double fd_step = kFdStepKern;
    double worst = 0.0;
    for (int i_dim = 0; i_dim < n_dim; ++i_dim) {
        arma::vec hp_gamma_plus_vec = hp_gamma_vec, hp_gamma_minus_vec = hp_gamma_vec;
        hp_gamma_plus_vec(i_dim) += fd_step;
        hp_gamma_minus_vec(i_dim) -= fd_step;
        const double grad_fd = (kern_sum(hp_gamma_plus_vec) - kern_sum(hp_gamma_minus_vec)) / (2.0 * fd_step);
        const double scale = std::max(std::fabs(grad_fd), kFdScaleFloor);
        worst = std::max(worst, std::fabs(d_sum_d_hp_gamma_vec(i_dim) - grad_fd) / scale);
    }
    char detail[64];
    std::snprintf(detail, sizeof(detail), "worst rel %.2e", worst);

    return report(name, worst < kTolFd, detail);
}

// Asserts RatQuad::add_hp_kern_extra_gradient_contribution accumulates d(sum_{i>j} kern_ij)/d(hp_rho).
//
// Run twice from main, once at a normal hp_rho and once near zero. Near zero is the hard case: the kernel's
// leading coefficient grows without bound, so the finite difference is worst conditioned there.
bool test_rat_quad_shape_grad_vs_fd(const char* name, const double hp_rho) {
    const int n_dim = 3, n_train = 7;
    const arma::mat X_ndim_ntrain_mat = make_points(n_dim, n_train);
    arma::vec hp_gamma_vec(n_dim);
    for (int i_dim = 0; i_dim < n_dim; ++i_dim) hp_gamma_vec(i_dim) = 0.8 + 0.3 * i_dim;
    const arma::mat Ydot_nfeature_ntrain_mat = X_ndim_ntrain_mat.each_col() % hp_gamma_vec;
    const arma::vec hp_kern_extra_vec{hp_rho};
    const gpflash::kernel::KernelBase& rat_quad = gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::RatQuad);
    const auto hp_grad_inputs =
            gpflash::kernel::make_hp_grad_inputs(X_ndim_ntrain_mat, X_ndim_ntrain_mat, hp_gamma_vec, hp_kern_extra_vec);

    const auto kern_sum = [&](const double rho) {
        double total = 0.0;
        for (int i_col = 0; i_col < n_train; ++i_col)
            for (int i_row = i_col + 1; i_row < n_train; ++i_row) {
                double kern_val = arma::accu(
                        arma::square(Ydot_nfeature_ntrain_mat.col(i_row) - Ydot_nfeature_ntrain_mat.col(i_col)));
                rat_quad.calc_kern_val_inplace(&kern_val, 1, &rho);
                total += kern_val;
            }
        return total;
    };

    double grad_analytic = 0.0;
    for (int i_col = 0; i_col < n_train; ++i_col) {
        const int n_below = n_train - i_col - 1;
        if (n_below <= 0) continue;
        std::vector<double> A_vec(n_below);
        for (int i_below = 0; i_below < n_below; ++i_below) {
            double kern_val = arma::accu(arma::square(
                    Ydot_nfeature_ntrain_mat.col(i_col + 1 + i_below) - Ydot_nfeature_ntrain_mat.col(i_col)));
            double kern_common_rdot = 0.0;
            rat_quad.calc_kern_val_and_common_rdot_inplace(&kern_val, 1, &hp_rho, &kern_common_rdot);
            A_vec[i_below] = kern_common_rdot;  // A_ij with Psi_ij set to 1
        }
        rat_quad.add_hp_kern_extra_gradient_contribution(
                hp_grad_inputs, i_col, i_col + 1, i_col + 1 + n_below, A_vec.data(), &grad_analytic);
    }

    const double fd_step = kFdStepKern;
    const double grad_fd = (kern_sum(hp_rho + fd_step) - kern_sum(hp_rho - fd_step)) / (2.0 * fd_step);
    const double rel_err = std::fabs(grad_analytic - grad_fd) / std::max(std::fabs(grad_fd), kFdScaleFloor);
    char detail[72];
    std::snprintf(detail, sizeof(detail), "hp_rho %.3g  worst rel %.2e", hp_rho, rel_err);
    return report(name, rel_err < kTolFd, detail);
}

}  // namespace

int main() {
    std::printf("== GpFlash kernel values and derivatives ==\n");
    bool ok = true;
    ok &= test_factorization(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::SqExp),
            &gpflash::kernel::SqExp::calc_kern_val_unfactored,
            &gpflash::kernel::SqExp::calc_kern_common_rdot_unfactored,
            "factorization/SqExp",
            0.0);
    ok &= test_factorization(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Matern52),
            &gpflash::kernel::Matern52::calc_kern_val_unfactored,
            &gpflash::kernel::Matern52::calc_kern_common_rdot_unfactored,
            "factorization/Matern52",
            0.0);
    ok &= test_factorization(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Matern32),
            &gpflash::kernel::Matern32::calc_kern_val_unfactored,
            &gpflash::kernel::Matern32::calc_kern_common_rdot_unfactored,
            "factorization/Matern32",
            0.0);
    ok &= test_factorization(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::RatQuad),
            &gpflash::kernel::RatQuad::calc_kern_val_unfactored,
            &gpflash::kernel::RatQuad::calc_kern_common_rdot_unfactored,
            "factorization/RatQuad",
            kRatQuadHpRho);
    ok &= test_factorization(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::RatQuad),
            &gpflash::kernel::RatQuad::calc_kern_val_unfactored,
            &gpflash::kernel::RatQuad::calc_kern_common_rdot_unfactored,
            "factorization/RatQuad_near_zero",
            kRatQuadHpRhoNearZero);
    ok &= test_coincident_points(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Matern52), "coincident_points/Matern52", 0.0);
    ok &= test_coincident_points(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Matern32), "coincident_points/Matern32", 0.0);
    ok &= test_coincident_points(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::SqExp), "coincident_points/SqExp", 0.0);
    ok &= test_coincident_points(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::RatQuad),
            "coincident_points/RatQuad",
            kRatQuadHpRho);
    // The last argument is s^(rdot) at zero distance, hard-coded per kernel: -1 for the squared exponential,
    // rational quadratic and periodic, -3 for the Matern 3/2, and -5/3 (1 + 0) for the Matern 5/2. Every
    // stationary kernel is checked, so none can adopt a sign convention of its own.
    ok &= test_kern_common_rdot_is_derivative(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::SqExp),
            "kern_common_rdot_is_derivative/SqExp",
            0.0,
            -1.0);
    ok &= test_kern_common_rdot_is_derivative(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Matern52),
            "kern_common_rdot_is_derivative/Matern52",
            0.0,
            -5.0 / 3.0);
    ok &= test_kern_common_rdot_is_derivative(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Matern32),
            "kern_common_rdot_is_derivative/Matern32",
            0.0,
            -3.0);
    ok &= test_kern_common_rdot_is_derivative(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::RatQuad),
            "kern_common_rdot_is_derivative/RatQuad",
            kRatQuadHpRho,
            -1.0);
    // Periodic's value is exp(-0.5 rdot_sq_norm) on its features; its amplitudes enter through the feature map.
    ok &= test_kern_common_rdot_is_derivative(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Periodic),
            "kern_common_rdot_is_derivative/Periodic",
            0.0,
            -1.0);
    ok &= test_hp_gamma_grad_vs_fd(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::SqExp), "hp_gamma_grad_vs_fd/SqExp", 0.0);
    ok &= test_hp_gamma_grad_vs_fd(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Matern52), "hp_gamma_grad_vs_fd/Matern52", 0.0);
    ok &= test_hp_gamma_grad_vs_fd(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Matern32), "hp_gamma_grad_vs_fd/Matern32", 0.0);
    ok &= test_hp_gamma_grad_vs_fd(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::RatQuad),
            "hp_gamma_grad_vs_fd/RatQuad",
            kRatQuadHpRho);
    ok &= test_rat_quad_shape_grad_vs_fd("rat_quad_shape_grad_vs_fd", kRatQuadHpRho);
    ok &= test_rat_quad_shape_grad_vs_fd("rat_quad_shape_grad_vs_fd/near_zero", kRatQuadHpRhoNearZero);
    std::printf("\n%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");

    return ok ? 0 : 1;
}
