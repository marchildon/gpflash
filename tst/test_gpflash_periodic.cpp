// The periodic kernel's feature map: the cosine and sine features reproduce the kernel exactly, and the
// map reached through calc_scaled_feature_points matches the direct construction.
#include <armadillo>
#include <cmath>
#include <cstdio>

#include "GpFlashTestSupport.hpp"
#include "gpflash/assembly/KernelMatrix.hpp"   // build_kmat_lower_tri
#include "gpflash/kernels/KernelDispatch.hpp"  // kernel_by_id

using gpflash_test::kDetailLen;
using gpflash_test::kRelScaleFloor;
using gpflash_test::kTolExact;
using gpflash_test::make_points;
using gpflash_test::report;

namespace {

// The periodic kernel's closed form for the pair (i_row, i_col), straight from the definition:
// exp(-2 sum_d hp_amplitude_d^2 sin^2(pi hp_gamma_d (x_id - x_jd))). The independent reference both tests compare to.
double periodic_closed_form(
        const arma::mat& X_ndim_ntrain_mat,
        const int i_row,
        const int i_col,
        const arma::vec& hp_gamma_vec,
        const arma::vec& hp_amplitude_vec) {
    double sum_sin_sq = 0.0;
    for (int i_dim = 0; i_dim < static_cast<int>(hp_gamma_vec.n_elem); ++i_dim) {
        const double theta = arma::datum::pi * hp_gamma_vec(i_dim)
                * (X_ndim_ntrain_mat(i_dim, i_row) - X_ndim_ntrain_mat(i_dim, i_col));
        const double sin_theta = std::sin(theta);
        sum_sin_sq += hp_amplitude_vec(i_dim) * hp_amplitude_vec(i_dim) * sin_theta * sin_theta;
    }

    return std::exp(-2.0 * sum_sin_sq);
}

// SqExp on Periodic::calc_feature_points output must reproduce the periodic closed form
// exp(-2 sum_d hp_amplitude_d^2 sin^2(pi hp_gamma_d r_d)) -- the equality that lets the periodic kernel reuse the
// shared assembly. Includes a pair exactly one period apart, where the feature-space distance is zero.
bool test_periodic_feature_map_is_exact(const char* name) {
    const int n_dim = 3;
    arma::vec hp_gamma_vec(n_dim), hp_amplitude_vec(n_dim);
    for (int i_dim = 0; i_dim < n_dim; ++i_dim) {
        hp_gamma_vec(i_dim) = 0.6 + 0.35 * i_dim;      // distinct frequencies per dimension
        hp_amplitude_vec(i_dim) = 0.7 + 0.25 * i_dim;  // distinct amplitudes, so a tie error shows
    }

    // Fixed pairs. The last is one full period apart in every dimension: r_d = 1 / hp_gamma_d.
    const int n_train = 7;
    arma::mat X_ndim_ntrain_mat(n_dim, n_train);
    for (int i_train = 0; i_train < n_train - 1; ++i_train)
        for (int i_dim = 0; i_dim < n_dim; ++i_dim)
            X_ndim_ntrain_mat(i_dim, i_train) =
                    std::sin(0.9 * i_train + 0.6 * i_dim) + 0.3 * std::cos(1.7 * i_train - 0.5 * i_dim);
    for (int i_dim = 0; i_dim < n_dim; ++i_dim)
        X_ndim_ntrain_mat(i_dim, n_train - 1) = X_ndim_ntrain_mat(i_dim, 0) + 1.0 / hp_gamma_vec(i_dim);

    const arma::mat Y_nfeature_ntrain_mat =
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Periodic)
                    .calc_feature_points(
                            X_ndim_ntrain_mat, arma::zeros<arma::vec>(n_dim), hp_gamma_vec, hp_amplitude_vec);
    // The feature-space length scales tie both axes of dimension d to hp_amplitude_d.
    const arma::vec hp_amplitude_tied_vec = arma::join_cols(hp_amplitude_vec, hp_amplitude_vec);

    // The assembly the periodic model runs: scale by the tied amplitudes, then SqExp on the squared distance.
    const arma::mat Ydot_nfeature_ntrain_mat = Y_nfeature_ntrain_mat.each_col() % hp_amplitude_tied_vec;

    double worst = 0.0, period_pair_val = 0.0, period_pair_ref = 0.0;
    for (int i_col = 0; i_col < n_train; ++i_col)
        for (int i_row = i_col + 1; i_row < n_train; ++i_row) {
            double via_features =
                    arma::accu(arma::square(Ydot_nfeature_ntrain_mat.col(i_row) - Ydot_nfeature_ntrain_mat.col(i_col)));
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::SqExp)
                    .calc_kern_val_inplace(&via_features, 1, nullptr);

            const double closed_form =
                    periodic_closed_form(X_ndim_ntrain_mat, i_row, i_col, hp_gamma_vec, hp_amplitude_vec);

            worst = std::max(
                    worst, std::fabs(via_features - closed_form) / std::max(std::fabs(closed_form), kRelScaleFloor));
            if (i_col == 0 && i_row == n_train - 1) {
                period_pair_val = via_features;
                period_pair_ref = closed_form;
            }
        }

    // One period apart is the same point on every circle, so both sides must give the zero-distance value.
    const bool period_ok = std::fabs(period_pair_val - 1.0) < kTolExact && std::fabs(period_pair_ref - 1.0) < kTolExact;

    char detail[kDetailLen];
    std::snprintf(detail, sizeof(detail), "worst rel %.2e  one_period=%.15g", worst, period_pair_val);
    return report(name, worst < kTolExact && period_ok, detail);
}

// The periodic kernel reaches the assembly through calc_scaled_feature_points, then evaluates its own value formula
// in the feature coordinates. Building the features by hand would bypass that route, so this uses only the members a
// generic caller does.
bool test_periodic_through_calc_scaled_feature_points(const char* name) {
    const gpflash::kernel::KernelBase& periodic = gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Periodic);
    const int n_dim = 3, n_train = 21;

    const arma::mat X_ndim_ntrain_mat = make_points(n_dim, n_train);
    const int n_hp_kern_extra = periodic.n_hp_kern_extra(n_dim);
    arma::vec hp_gamma_vec(n_dim), hp_kern_extra_vec(n_hp_kern_extra);
    const arma::vec x_center_vec = arma::zeros<arma::vec>(n_dim);
    for (int i_dim = 0; i_dim < n_dim; ++i_dim) {
        hp_gamma_vec(i_dim) = 0.6 + 0.35 * i_dim;       // hp_gamma: the phase frequencies
        hp_kern_extra_vec(i_dim) = 0.7 + 0.25 * i_dim;  // hp_amplitude: the feature-space length scales
    }

    // Exactly what training and the posterior do, through the generic members only: the unscaled features from
    // one, the scaled points from the other, at the same hyperparameters.
    const arma::mat Y_nfeature_ntrain_mat =
            periodic.calc_feature_points(X_ndim_ntrain_mat, x_center_vec, hp_gamma_vec, hp_kern_extra_vec);
    const arma::mat Ydot_nfeature_ntrain_mat =
            periodic.calc_scaled_feature_points(X_ndim_ntrain_mat, x_center_vec, hp_gamma_vec, hp_kern_extra_vec);
    const double* hp_kern_extra_ptr = hp_kern_extra_vec.is_empty() ? nullptr : hp_kern_extra_vec.memptr();
    const arma::mat K_built_mat = gpflash::detail::build_kmat_lower_tri(
            periodic, Ydot_nfeature_ntrain_mat, hp_kern_extra_ptr, /*store_hp_grad_factors=*/true);

    // The map must raise the dimension, and the round trip back must recover the input width.
    const bool dims_ok = Y_nfeature_ntrain_mat.n_rows == static_cast<arma::uword>(2 * n_dim)
            && periodic.calc_n_dim_input(static_cast<int>(Y_nfeature_ntrain_mat.n_rows)) == n_dim
            && n_hp_kern_extra == n_dim;

    double worst = 0.0, diag_err = 0.0;
    for (int i_col = 0; i_col < n_train; ++i_col) {
        diag_err = std::max(diag_err, std::fabs(K_built_mat(i_col, i_col) - 1.0));
        for (int i_row = i_col + 1; i_row < n_train; ++i_row) {
            const double closed_form =
                    periodic_closed_form(X_ndim_ntrain_mat, i_row, i_col, hp_gamma_vec, hp_kern_extra_vec);
            worst = std::max(
                    worst, std::fabs(K_built_mat(i_row, i_col) - closed_form) / std::max(closed_form, kRelScaleFloor));
        }
    }
    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "worst rel %.2e  diag_err %.1e  dims %d",
            worst,
            diag_err,
            static_cast<int>(dims_ok));

    return report(name, dims_ok && worst < kTolExact && diag_err < kTolExact, detail);
}

}  // namespace

int main() {
    std::printf("== GpFlash periodic kernel ==\n");
    bool ok = true;
    ok &= test_periodic_feature_map_is_exact("periodic_feature_map_is_exact");
    ok &= test_periodic_through_calc_scaled_feature_points("periodic_through_calc_scaled_feature_points");
    std::printf("\n%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return ok ? 0 : 1;
}
