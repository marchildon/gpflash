// Kernel-matrix assembly. What this file asserts:
//
//   1) K_mat_vs_brute_force   build_kmat_lower_tri's lower triangle equals a reference that scales the points,
//                             forms each squared distance directly, and evaluates the kernel per pair, and its
//                             diagonal is 1
//   2) K_mat_parallel         the same, at n_train above gpflash::PERF_OMP_MIN_N_POINT, so the OpenMP column sweep is
//                             compared against the serial reference rather than against itself
//   3) val_map_paths_agree    the two buffer entry points reproduce the kernel's per-pair value
//   4) sq_dist_clamp          a squared distance below 0 becomes 0, and one above SQ_DIST_MAX, inf or NaN becomes
//                             SQ_DIST_MAX, at which SqExp and the Matern kernels are 0 and RatQuad is finite
//
// Only radial kernels appear here. The kernel-independent parts of assembly are the same for every family; the
// non-radial kernels' assembly is checked in test_gpflash_linear, _periodic and _posterior_nonradial.
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

#include "GpFlashTestSupport.hpp"
#include "gpflash/assembly/KernelMatrix.hpp"
#include "gpflash/kernels/KernelDispatch.hpp"  // kernel_by_id
#include "gpflash/kernels/radial/KernelRadial.hpp"
#include "gpflash/linear_algebra/InnerProducts.hpp"           // clamp_sq_dist, SQ_DIST_MAX
#include "gpflash/optimization/search_space/OptzOptions.hpp"  // OptzOptions, for the default shape bound

using gpflash_test::kDetailLen;
using gpflash_test::kNtrainAboveOmpThresholds;
using gpflash_test::kRatQuadHpRho;
using gpflash_test::kRelScaleFloor;
using gpflash_test::kTolExact;
using gpflash_test::make_points;
using gpflash_test::report;

namespace {

// The reference K: scale the points by hp_gamma, form each squared distance directly, evaluate the kernel.
// Independent of assembly, which reaches the same numbers through DSYRK and polarization.
arma::mat brute_force_K_mat(
        const gpflash::kernel::KernelBase& kern_fun,
        const arma::mat& X_ndim_ntrain_mat,
        const arma::vec& hp_gamma_vec,
        const double* hp_kern_extra_ptr) {
    const int n_dim = static_cast<int>(X_ndim_ntrain_mat.n_rows), n_train = static_cast<int>(X_ndim_ntrain_mat.n_cols);
    arma::mat K_mat(n_train, n_train, arma::fill::zeros);
    for (int i_row = 0; i_row < n_train; ++i_row) {
        for (int i_col = 0; i_col <= i_row; ++i_col) {
            double rdot_sq_norm = 0.0;
            for (int i_dim = 0; i_dim < n_dim; ++i_dim) {
                const double rdot_d =
                        hp_gamma_vec(i_dim) * (X_ndim_ntrain_mat(i_dim, i_row) - X_ndim_ntrain_mat(i_dim, i_col));
                rdot_sq_norm += rdot_d * rdot_d;
            }
            double kern_val = 0.0, kern_common_rdot = 0.0;
            kern_val = rdot_sq_norm;
            kern_fun.calc_kern_val_and_common_rdot_inplace(&kern_val, 1, hp_kern_extra_ptr, &kern_common_rdot);
            K_mat(i_row, i_col) = kern_val;
        }
    }
    return K_mat;
}

// Largest lower-triangle discrepancy. The builder keeps c^(rdot) in the strict upper triangle, so only the
// lower half holds K.
double max_lower_tri_err(const arma::mat& A_mat, const arma::mat& B_mat) {
    const int n_train = static_cast<int>(A_mat.n_rows);
    double max_err = 0.0;
    for (int i_row = 0; i_row < n_train; ++i_row)
        for (int i_col = 0; i_col <= i_row; ++i_col)
            max_err = std::max(max_err, std::abs(A_mat(i_row, i_col) - B_mat(i_row, i_col)));
    return max_err;
}

// Asserts build_kmat_lower_tri's lower triangle and unit diagonal match the reference above.
//
// main runs this on both sides of gpflash::PERF_OMP_MIN_N_POINT, so the OpenMP column sweep is compared against the
// serial reference rather than against itself.
bool test_kernel_matrix(
        const gpflash::kernel::KernelBase& kern_fun, const char* name, const double hp_rho, const int n_train = 23) {
    const int n_dim = 4;
    const arma::mat X_ndim_ntrain_mat = make_points(n_dim, n_train);
    arma::vec hp_gamma_vec(n_dim);
    // Distinct and non-unit: equal length scales hide a per-dimension indexing error, unit ones hide a
    // missing scale factor.
    for (int i_dim = 0; i_dim < n_dim; ++i_dim) hp_gamma_vec(i_dim) = 0.6 + 0.35 * i_dim;
    const double* hp_kern_extra_ptr = (kern_fun.n_hp_kern_extra(n_dim) > 0) ? &hp_rho : nullptr;

    // The rescale every radial kernel inherits; a radial kernel reads no extras there.
    const arma::mat Ydot_nfeature_ntrain_mat = kern_fun.calc_scaled_feature_points(
            X_ndim_ntrain_mat, arma::zeros<arma::vec>(n_dim), hp_gamma_vec, /*hp_kern_extra_vec=*/arma::vec());
    // store_hp_grad_factors = true exercises the strict-upper write, which shares a buffer with the lower
    // triangle checked below.
    const arma::mat K_built_mat = gpflash::detail::build_kmat_lower_tri(
            kern_fun, Ydot_nfeature_ntrain_mat, hp_kern_extra_ptr, /*store_hp_grad_factors=*/true);
    // The scaled points must agree with a direct rescale, since everything above reads them. Written as an
    // explicit loop so the reference shares no Armadillo expression with the code under test.
    arma::mat Ydot_ref_nfeature_ntrain_mat(n_dim, n_train, arma::fill::none);
    for (int i_train = 0; i_train < n_train; ++i_train)
        for (int i_dim = 0; i_dim < n_dim; ++i_dim)
            Ydot_ref_nfeature_ntrain_mat(i_dim, i_train) = X_ndim_ntrain_mat(i_dim, i_train) * hp_gamma_vec(i_dim);
    const double scale_err = arma::abs(Ydot_nfeature_ntrain_mat - Ydot_ref_nfeature_ntrain_mat).max();
    const arma::mat K_ref_mat = brute_force_K_mat(kern_fun, X_ndim_ntrain_mat, hp_gamma_vec, hp_kern_extra_ptr);

    const double k_err = max_lower_tri_err(K_built_mat, K_ref_mat);
    // Unit diagonal, which a radial kernel declares through is_kern_diag_unity.
    double diag_err = 0.0;
    for (int i_diag = 0; i_diag < n_train; ++i_diag)
        diag_err = std::max(diag_err, std::abs(K_built_mat(i_diag, i_diag) - 1.0));

    char detail[kDetailLen];
    std::snprintf(detail, sizeof(detail), "K_err=%.1e diag_err=%.1e scale_err=%.1e", k_err, diag_err, scale_err);

    return report(name, k_err < kTolExact && diag_err < kTolExact && scale_err < kTolExact, detail);
}

// Asserts the two buffer entry points agree with each other, value and c^(rdot).
//
// They enter at different stages, which is the whole point: calc_kmat_column_inplace receives the Gram cross
// terms and must polarize them itself, while calc_kern_val_and_common_rdot_inplace receives squared distances
// formed directly by subtraction. Agreement is therefore a real check on the polarization identity
// ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b as assembly applies it. The kernels' closed forms are not rechecked
// here; test_gpflash_kernel_finite_difference.cpp pins those against independent transcriptions.
bool test_val_map_paths_agree(
        const gpflash::kernel::KernelBase& kern_fun, const char* name, const double hp_kern_extra_val) {
    const int n_dim = 3, n_train = 9;
    const double* hp_kern_extra_ptr = (kern_fun.n_hp_kern_extra(n_dim) == 1) ? &hp_kern_extra_val : nullptr;
    const arma::mat X_ndim_ntrain_mat = make_points(n_dim, n_train);
    arma::vec hp_gamma_vec(n_dim);
    for (int i_dim = 0; i_dim < n_dim; ++i_dim) hp_gamma_vec(i_dim) = 0.8 + 0.3 * i_dim;

    const arma::mat Ydot_nfeature_ntrain_mat = X_ndim_ntrain_mat.each_col() % hp_gamma_vec;
    const arma::vec sq_norm_vec = arma::sum(arma::square(Ydot_nfeature_ntrain_mat), 0).t();

    double worst_kern_val = 0.0, worst_kern_common_rdot = 0.0;
    for (int i_col = 0; i_col < n_train - 1; ++i_col) {
        const int n_below = n_train - i_col - 1;
        std::vector<double> kern_val_from_polar(n_below), kern_common_rdot_from_polar(n_below),
                kern_val_from_distance(n_below), kern_common_rdot_from_distance(n_below);

        // Path 1: the Gram cross terms, as DSYRK leaves them, polarized inside the kernel.
        for (int i_below = 0; i_below < n_below; ++i_below)
            kern_val_from_polar[i_below] =
                    arma::dot(Ydot_nfeature_ntrain_mat.col(i_col + 1 + i_below), Ydot_nfeature_ntrain_mat.col(i_col));
        kern_fun.calc_kmat_column_inplace(
                kern_val_from_polar.data(),
                n_below,
                sq_norm_vec(i_col),
                sq_norm_vec.memptr() + i_col + 1,
                hp_kern_extra_ptr,
                kern_common_rdot_from_polar.data());

        // Path 2: the squared distances formed directly by subtraction.
        for (int i_below = 0; i_below < n_below; ++i_below)
            kern_val_from_distance[i_below] = arma::accu(arma::square(
                    Ydot_nfeature_ntrain_mat.col(i_col + 1 + i_below) - Ydot_nfeature_ntrain_mat.col(i_col)));
        kern_fun.calc_kern_val_and_common_rdot_inplace(
                kern_val_from_distance.data(), n_below, hp_kern_extra_ptr, kern_common_rdot_from_distance.data());

        for (int i_below = 0; i_below < n_below; ++i_below) {
            const double kern_val_scale = std::max(std::fabs(kern_val_from_distance[i_below]), kRelScaleFloor);
            worst_kern_val = std::max(
                    worst_kern_val,
                    std::fabs(kern_val_from_polar[i_below] - kern_val_from_distance[i_below]) / kern_val_scale);
            const double rdot_scale = std::max(std::fabs(kern_common_rdot_from_distance[i_below]), kRelScaleFloor);
            worst_kern_common_rdot = std::max(
                    worst_kern_common_rdot,
                    std::fabs(kern_common_rdot_from_polar[i_below] - kern_common_rdot_from_distance[i_below])
                            / rdot_scale);
        }
    }
    char detail[kDetailLen];
    std::snprintf(
            detail, sizeof(detail), "kern_val %.1e  kern_common_rdot %.1e", worst_kern_val, worst_kern_common_rdot);

    return report(name, worst_kern_val < kTolExact && worst_kern_common_rdot < kTolExact, detail);
}

bool test_sq_dist_clamp(const char* name) {
    using gpflash::detail::clamp_sq_dist;
    using gpflash::detail::SQ_DIST_MAX;
    const bool clamp_ok = clamp_sq_dist(-1e-17) == 0.0 && clamp_sq_dist(2.5) == 2.5
            && clamp_sq_dist(SQ_DIST_MAX * 10.0) == SQ_DIST_MAX && clamp_sq_dist(arma::datum::inf) == SQ_DIST_MAX
            && clamp_sq_dist(arma::datum::nan) == SQ_DIST_MAX;

    // The kernels' values at the clamp: 0 for the exponential ones, finite for RatQuad at its default lower bound.
    bool val_ok = true;
    for (const gpflash::kernel::KernelId kernel_id :
         {gpflash::kernel::KernelId::SqExp, gpflash::kernel::KernelId::Matern32, gpflash::kernel::KernelId::Matern52}) {
        double kern_val = SQ_DIST_MAX;
        gpflash::kernel::kernel_by_id(kernel_id).calc_kern_val_inplace(&kern_val, 1, nullptr);
        val_ok = val_ok && kern_val == 0.0;
    }
    const double hp_rho_lb = gpflash::OptzOptions().hp_kern_extra_lb_vec(0);
    double rat_quad_val = SQ_DIST_MAX;
    gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::RatQuad)
            .calc_kern_val_inplace(&rat_quad_val, 1, &hp_rho_lb);
    val_ok = val_ok && std::isfinite(rat_quad_val);

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "clamp %d  exponential kernels 0 %d  RatQuad at shape %g: %.3g",
            static_cast<int>(clamp_ok),
            static_cast<int>(val_ok),
            hp_rho_lb,
            rat_quad_val);
    return report(name, clamp_ok && val_ok, detail);
}

}  // namespace

int main() {
    std::printf("== GpFlash kernel assembly ==\n");
    bool ok = true;
    ok &= test_kernel_matrix(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::SqExp), "K_mat_vs_brute_force/SqExp", 0.0);
    ok &= test_kernel_matrix(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Matern52), "K_mat_vs_brute_force/Matern52", 0.0);
    ok &= test_kernel_matrix(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Matern32), "K_mat_vs_brute_force/Matern32", 0.0);
    ok &= test_kernel_matrix(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::RatQuad),
            "K_mat_vs_brute_force/RatQuad",
            kRatQuadHpRho);
    // Above PERF_OMP_MIN_N_POINT, against the same reference as the serial path, with and without an extra
    // hyperparameter so both argument paths run at that size.
    ok &= test_kernel_matrix(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::SqExp),
            "K_mat_parallel/SqExp",
            0.0,
            kNtrainAboveOmpThresholds);
    ok &= test_kernel_matrix(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Matern52),
            "K_mat_parallel/Matern52",
            0.0,
            kNtrainAboveOmpThresholds);
    ok &= test_kernel_matrix(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::RatQuad),
            "K_mat_parallel/RatQuad",
            kRatQuadHpRho,
            kNtrainAboveOmpThresholds);
    ok &= test_val_map_paths_agree(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::SqExp), "val_map_paths_agree/SqExp", 0.0);
    ok &= test_val_map_paths_agree(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Matern52), "val_map_paths_agree/Matern52", 0.0);
    ok &= test_val_map_paths_agree(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Matern32), "val_map_paths_agree/Matern32", 0.0);
    ok &= test_val_map_paths_agree(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::RatQuad),
            "val_map_paths_agree/RatQuad",
            kRatQuadHpRho);
    ok &= test_sq_dist_clamp("sq_dist_clamp");
    std::printf("\n%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");

    return ok ? 0 : 1;
}
