// Posterior assembly shapes: the single-query cross-covariance and its batch counterpart agree with each
// other and with a brute-force reference, for every radial kernel. Coincident queries also pin the squared-distance
// clamp that keeps the two shapes finite and consistent.
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>

#include "GpFlashTestSupport.hpp"
#include "gpflash/assembly/KernelMatrix.hpp"
#include "gpflash/assembly/PosteriorCovariance.hpp"
#include "gpflash/kernels/KernelDispatch.hpp"  // kernel_by_id
#include "gpflash/kernels/radial/KernelRadial.hpp"

using gpflash_test::draw_uniform_mat;
using gpflash_test::kDetailLen;
using gpflash_test::kHpRhoUnused;
using gpflash_test::kRatQuadHpRho;
using gpflash_test::kRelScaleFloor;
using gpflash_test::kTolExact;
using gpflash_test::make_points;
using gpflash_test::report;

namespace {

// build_post_vec (one query point) and build_post_mat (many, one DGEMM) against the same brute-force
// cross-covariance and against each other. Both polarise from a different BLAS call than the fit path, so a
// sign or transposition error here is invisible to the kernel-matrix test.
bool test_post_shapes(const gpflash::kernel::KernelBase& kern_fun, const char* name, const double hp_rho) {
    const int n_dim = 4, n_train = 17, n_post = 5;
    const arma::mat X_ndim_ntrain_mat = make_points(n_dim, n_train);
    // Query points from a different generator, so none coincides with a training point by accident.
    arma::mat X_ndim_npost_mat(n_dim, n_post);
    for (int i_post = 0; i_post < n_post; ++i_post)
        for (int i_dim = 0; i_dim < n_dim; ++i_dim)
            X_ndim_npost_mat(i_dim, i_post) =
                    0.4 * std::cos(1.1 * i_post + 0.9 * i_dim) - 0.6 * std::sin(0.5 * i_post - 1.4 * i_dim);

    arma::vec hp_gamma_vec(n_dim);
    for (int i_dim = 0; i_dim < n_dim; ++i_dim) hp_gamma_vec(i_dim) = 0.6 + 0.35 * i_dim;
    const double* hp_kern_extra_ptr = (kern_fun.n_hp_kern_extra(n_dim) > 0) ? &hp_rho : nullptr;

    const arma::mat Ydot_nfeature_ntrain_mat = X_ndim_ntrain_mat.each_col() % hp_gamma_vec;
    const arma::mat Ydot_nfeature_npost_mat = X_ndim_npost_mat.each_col() % hp_gamma_vec;
    const arma::vec ydot_train_sq_norm_vec = arma::sum(arma::square(Ydot_nfeature_ntrain_mat), 0).t();

    // Reference: the definition, one pair at a time.
    arma::mat K_ref_ntrain_npost_mat(n_train, n_post), kern_common_rdot_ref_ntrain_npost_mat(n_train, n_post);
    for (int i_post = 0; i_post < n_post; ++i_post)
        for (int i_train = 0; i_train < n_train; ++i_train) {
            const double rdot_sq_norm = arma::accu(
                    arma::square(Ydot_nfeature_ntrain_mat.col(i_train) - Ydot_nfeature_npost_mat.col(i_post)));
            double kern_val, kern_common_rdot;
            kern_val = rdot_sq_norm;
            kern_fun.calc_kern_val_and_common_rdot_inplace(&kern_val, 1, hp_kern_extra_ptr, &kern_common_rdot);
            K_ref_ntrain_npost_mat(i_train, i_post) = kern_val;
            kern_common_rdot_ref_ntrain_npost_mat(i_train, i_post) = kern_common_rdot;
        }

    // Shape 1: one query point at a time. build_post_vec takes both sides' squared norms, as the batch form
    // takes the training ones.
    const arma::vec ydot_post_sq_norm_vec = arma::sum(arma::square(Ydot_nfeature_npost_mat), 0).t();
    double worst_post_vec_err = 0.0, worst_post_vec_common_rdot_err = 0.0;
    for (int i_post = 0; i_post < n_post; ++i_post) {
        arma::vec kern_common_rdot_post_vec;
        const arma::vec k_post_vec = gpflash::detail::build_post_vec(
                kern_fun,
                Ydot_nfeature_ntrain_mat,
                ydot_train_sq_norm_vec,
                Ydot_nfeature_npost_mat.col(i_post),
                ydot_post_sq_norm_vec(i_post),
                hp_kern_extra_ptr,
                &kern_common_rdot_post_vec);
        for (int i_train = 0; i_train < n_train; ++i_train) {
            worst_post_vec_err = std::max(
                    worst_post_vec_err,
                    std::fabs(k_post_vec(i_train) - K_ref_ntrain_npost_mat(i_train, i_post))
                            / std::max(std::fabs(K_ref_ntrain_npost_mat(i_train, i_post)), kRelScaleFloor));
            worst_post_vec_common_rdot_err = std::max(
                    worst_post_vec_common_rdot_err,
                    std::fabs(
                            kern_common_rdot_post_vec(i_train) - kern_common_rdot_ref_ntrain_npost_mat(i_train, i_post))
                            / std::max(
                                    std::fabs(kern_common_rdot_ref_ntrain_npost_mat(i_train, i_post)), kRelScaleFloor));
        }
    }

    // Shape 2: every query point at once.
    arma::mat kern_common_rdot_ntrain_npost_mat;
    const arma::mat K_ntrain_npost_mat = gpflash::detail::build_post_mat(
            kern_fun,
            Ydot_nfeature_ntrain_mat,
            ydot_train_sq_norm_vec,
            Ydot_nfeature_npost_mat,
            ydot_post_sq_norm_vec,
            hp_kern_extra_ptr,
            &kern_common_rdot_ntrain_npost_mat);
    const bool shape_ok = K_ntrain_npost_mat.n_rows == static_cast<arma::uword>(n_train)
            && K_ntrain_npost_mat.n_cols == static_cast<arma::uword>(n_post);
    double worst_post_mat_err = 0.0, worst_post_mat_common_rdot_err = 0.0;
    if (shape_ok)
        for (int i_post = 0; i_post < n_post; ++i_post)
            for (int i_train = 0; i_train < n_train; ++i_train) {
                worst_post_mat_err = std::max(
                        worst_post_mat_err,
                        std::fabs(K_ntrain_npost_mat(i_train, i_post) - K_ref_ntrain_npost_mat(i_train, i_post))
                                / std::max(std::fabs(K_ref_ntrain_npost_mat(i_train, i_post)), kRelScaleFloor));
                worst_post_mat_common_rdot_err = std::max(
                        worst_post_mat_common_rdot_err,
                        std::fabs(
                                kern_common_rdot_ntrain_npost_mat(i_train, i_post)
                                - kern_common_rdot_ref_ntrain_npost_mat(i_train, i_post))
                                / std::max(
                                        std::fabs(kern_common_rdot_ref_ntrain_npost_mat(i_train, i_post)),
                                        kRelScaleFloor));
            }

    const double worst = std::max(
            std::max(worst_post_vec_err, worst_post_vec_common_rdot_err),
            std::max(worst_post_mat_err, worst_post_mat_common_rdot_err));
    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "post_vec %.1e/%.1e  post_mat %.1e/%.1e",
            worst_post_vec_err,
            worst_post_vec_common_rdot_err,
            worst_post_mat_err,
            worst_post_mat_common_rdot_err);
    return report(name, shape_ok && worst < kTolExact, detail);
}

// Coincident queries expose cancellation in the polarized squared distance. Both posterior shapes must clamp
// the same negative roundoff to zero before Matern52 takes a square root.
bool test_coincident_query_clamp(const char* name) {
    const gpflash::kernel::KernelBase& kern_fun = gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Matern52);
    const int n_dim = 32, n_train = 8, n_post = 3;
    std::mt19937_64 rng(7);
    const arma::mat Ydot_nfeature_ntrain_mat = draw_uniform_mat(rng, n_dim, n_train, 0.0, 1.0e3);
    const arma::mat Ydot_nfeature_npost_mat = Ydot_nfeature_ntrain_mat.cols(0, n_post - 1);
    arma::vec ydot_train_sq_norm_vec = arma::sum(arma::square(Ydot_nfeature_ntrain_mat), 0).t();
    // The norm sum and the GEMM cross term round in opposite directions, so a coincident pair can yield a
    // slightly negative squared distance. Walking the norms down a few ULPs forces that here rather than
    // leaving it to one BLAS's summation order. The count only has to be enough to cross zero, and
    // cancellation_reached below fails the test if it is not, so it needs no tuning.
    constexpr int kNormUlpStepsDown = 16;
    for (int i_post = 0; i_post < n_post; ++i_post)
        for (int i_ulp = 0; i_ulp < kNormUlpStepsDown; ++i_ulp)
            ydot_train_sq_norm_vec(i_post) =
                    std::nextafter(ydot_train_sq_norm_vec(i_post), -std::numeric_limits<double>::infinity());

    // Confirm this fixture reaches the cancellation case before relying on it as a regression test.
    arma::mat raw_sq_dist_mat = -2.0 * Ydot_nfeature_ntrain_mat.t() * Ydot_nfeature_npost_mat;
    raw_sq_dist_mat.each_col() += ydot_train_sq_norm_vec;
    raw_sq_dist_mat.each_row() += arma::sum(arma::square(Ydot_nfeature_npost_mat), 0);
    const bool cancellation_reached = raw_sq_dist_mat.min() < 0.0;

    const arma::vec ydot_post_sq_norm_vec = arma::sum(arma::square(Ydot_nfeature_npost_mat), 0).t();
    const arma::mat K_ntrain_npost_mat = gpflash::detail::build_post_mat(
            kern_fun,
            Ydot_nfeature_ntrain_mat,
            ydot_train_sq_norm_vec,
            Ydot_nfeature_npost_mat,
            ydot_post_sq_norm_vec,
            /*hp_kern_extra_ptr=*/nullptr,
            /*kern_common_rdot_ntrain_npost_mat_out=*/nullptr);
    double worst_shape_diff = 0.0, worst_coincident_err = 0.0;
    bool is_single_finite = true;
    for (int i_post = 0; i_post < n_post; ++i_post) {
        const arma::vec k_post_vec = gpflash::detail::build_post_vec(
                kern_fun,
                Ydot_nfeature_ntrain_mat,
                ydot_train_sq_norm_vec,
                Ydot_nfeature_npost_mat.col(i_post),
                ydot_post_sq_norm_vec(i_post),
                /*hp_kern_extra_ptr=*/nullptr,
                /*kern_common_rdot_post_vec_out=*/nullptr);
        // The inf-norm and std::max both drop a NaN, so the single shape is checked directly as well.
        is_single_finite = is_single_finite && k_post_vec.is_finite();
        worst_shape_diff = std::max(worst_shape_diff, arma::norm(k_post_vec - K_ntrain_npost_mat.col(i_post), "inf"));
        worst_coincident_err = std::max(worst_coincident_err, std::fabs(K_ntrain_npost_mat(i_post, i_post) - 1.0));
        worst_coincident_err = std::max(worst_coincident_err, std::fabs(k_post_vec(i_post) - 1.0));
    }

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "raw_min=%.1e shape=%.1e coincident=%.1e single_finite=%d",
            raw_sq_dist_mat.min(),
            worst_shape_diff,
            worst_coincident_err,
            is_single_finite ? 1 : 0);
    return report(
            name,
            cancellation_reached && K_ntrain_npost_mat.is_finite() && is_single_finite && worst_shape_diff < kTolExact
                    && worst_coincident_err < kTolExact,
            detail);
}

}  // namespace

int main() {
    std::printf("== GpFlash posterior assembly ==\n");
    bool ok = true;
    ok &= test_post_shapes(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::SqExp), "post_shapes/SqExp", kHpRhoUnused);
    ok &= test_post_shapes(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Matern52), "post_shapes/Matern52", kHpRhoUnused);
    ok &= test_post_shapes(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Matern32), "post_shapes/Matern32", kHpRhoUnused);
    ok &= test_post_shapes(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::RatQuad), "post_shapes/RatQuad", kRatQuadHpRho);
    ok &= test_coincident_query_clamp("coincident_query_clamp/Matern52");
    std::printf("\n%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");

    return ok ? 0 : 1;
}
