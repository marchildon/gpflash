// The center a kernel measures its points from. What this file asserts:
//
//   1) center_by_kernel          a stationary kernel's center is the training points' per-dimension median, which
//                                one extreme point does not move, and is 0 in a dimension the mirror flips; any
//                                other kernel's center is 0
//   2) stationary_declaration    a kernel declaring itself stationary builds the same K from shifted points, and an
//                                inner-product kernel does not
//   3) offset_leaves_model_unchanged  shifting every point by 1e6, outside the dimensions a mirror flips, leaves the
//                                MLL, its gradient, and the batch and single-point posteriors and their gradients bit
//                                for bit the same
//   4) offset_survives_cache     a posterior built on shifted points serves the same queries after a save and a load
//   5) offset_leaves_fit_unchanged  a fit on shifted points finds the same hyperparameters and MLL bit for bit
//
// The points are multiples of 1/64 and the shift an integer, so shifted points, their median and the shift back are
// all exact, and the kernel sees the same bits either way. Measured from 0 instead, the squared distances at that
// offset lose enough digits that 3) and 5) fail.
#include <algorithm>
#include <armadillo>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include "GpFlashTestSupport.hpp"
#include "gpflash/assembly/KernelAssembly.hpp"   // build_kern_assembly
#include "gpflash/kernels/KernelDispatch.hpp"    // kernel_by_id
#include "gpflash/optimization/HpOptimizer.hpp"  // select_hp_with_max_mll
#include "gpflash/posterior_build/PosteriorBuild.hpp"
#include "gpflash/training/EvalMllDispatch.hpp"
using gpflash_test::kDetailLen;
using gpflash_test::kTolExact;
using gpflash_test::report;

namespace {

using gpflash::kernel::KernelId;

constexpr double kOffset = 1e6;  // far enough that distances measured from 0 lose their leading digits
constexpr int kNTrain = 21;      // odd, so the median is one of the points
// A shift moves an inner-product kernel's K at leading order, far above any roundoff.
constexpr double kInnerProductKmatChangeMin = 1e-3;

// Two dimensions on a 1/64 grid, every coordinate non-zero so an odd model's points sit off its mirror plane.
arma::mat make_grid_points(const int n_point, const int stride) {
    arma::mat X_ndim_npoint_mat(2, n_point);
    for (int i_point = 0; i_point < n_point; ++i_point) {
        X_ndim_npoint_mat(0, i_point) = (1 + (i_point * stride) % 61) / 64.0;
        X_ndim_npoint_mat(1, i_point) = (1 + (i_point * (stride + 4)) % 59) / 64.0;
    }
    return X_ndim_npoint_mat;
}

gpflash::TrainingData make_train_data(const arma::mat& X_ndim_ntrain_mat) {
    gpflash::TrainingData train_data;
    train_data.X_ndim_ntrain_mat = X_ndim_ntrain_mat;
    train_data.f_train_vec.set_size(X_ndim_ntrain_mat.n_cols);
    for (arma::uword i_train = 0; i_train < X_ndim_ntrain_mat.n_cols; ++i_train) {
        const double x_0 = X_ndim_ntrain_mat(0, i_train), x_1 = X_ndim_ntrain_mat(1, i_train);
        train_data.f_train_vec(i_train) = std::sin(3.0 * x_0) * (1.0 + x_1) + std::cos(2.0 * x_1);
    }
    return train_data;
}

gpflash::HpVal make_hp_val(const KernelId kernel_id) {
    const gpflash::kernel::KernelBase& kern_fun = gpflash::kernel::kernel_by_id(kernel_id);
    gpflash::HpVal hp_val_struct;
    hp_val_struct.hp_gamma_vec = {3.0, 2.0};
    hp_val_struct.hp_kern_extra_vec = arma::vec(kern_fun.n_hp_kern_extra(2), arma::fill::ones);
    return hp_val_struct;
}

// The shift applied to every point: kOffset, except in a dimension the mirror flips, whose plane is part of the model.
arma::vec make_offset_vec(const gpflash::SymmetrySpec& symmetry_spec) {
    arma::vec offset_vec(2);
    offset_vec.fill(kOffset);
    if (symmetry_spec.is_on()) offset_vec.elem(arma::find(symmetry_spec.mirror_sign_vec < 0.0)).zeros();
    return offset_vec;
}

double calc_max_abs_diff(const arma::mat& A_mat, const arma::mat& B_mat) {
    if (A_mat.n_rows != B_mat.n_rows || A_mat.n_cols != B_mat.n_cols) return arma::datum::inf;
    if (A_mat.is_empty()) return 0.0;
    return arma::abs(A_mat - B_mat).max();
}

// Every stationary kernel alone, then the radial and periodic families each under an even and an odd mirror.
struct OffsetCase {
    KernelId kernel_id;
    gpflash::SymmetrySpec symmetry_spec;
    bool has_mean_model_constant;
};

std::vector<OffsetCase> make_offset_cases() {
    std::vector<OffsetCase> offset_case_list;
    for (int i_kernel = 0; i_kernel < static_cast<int>(KernelId::End); ++i_kernel) {
        const KernelId kernel_id = static_cast<KernelId>(i_kernel);
        if (gpflash::kernel::kernel_by_id(kernel_id).is_stationary())
            offset_case_list.push_back({kernel_id, gpflash::SymmetrySpec(), true});
    }
    for (const KernelId kernel_id : {KernelId::Matern52, KernelId::Periodic}) {
        for (const bool is_even_symmetry : {true, false}) {
            gpflash::SymmetrySpec symmetry_spec;
            symmetry_spec.mirror_sign_vec = {-1.0, 1.0};
            symmetry_spec.is_even_symmetry = is_even_symmetry;
            offset_case_list.push_back({kernel_id, symmetry_spec, is_even_symmetry});
        }
    }
    return offset_case_list;
}

gpflash::ModelOptions make_model_opts(const OffsetCase& offset_case) {
    gpflash::ModelOptions model_opts;
    model_opts.kernel_id = offset_case.kernel_id;
    model_opts.symmetry_spec = offset_case.symmetry_spec;
    model_opts.has_mean_model_constant = offset_case.has_mean_model_constant;
    return model_opts;
}

std::string case_label(const OffsetCase& offset_case) {
    std::string label = gpflash::kernel::kernel_by_id(offset_case.kernel_id).kernel_name();
    if (offset_case.symmetry_spec.is_on()) label += offset_case.symmetry_spec.is_even_symmetry ? "/even" : "/odd";
    return label;
}

// ── 1) center_by_kernel ──────────────────────────────────────────────────────

bool test_center_by_kernel(const char* name) {
    // Dimension 0: six points in [0.1, 0.5] and one at 1e6, whose median is 0.3 where the mean sits near 1.4e5 and
    // the range's midpoint near 5e5. Dimension 1: 2, 4, ..., 14, median 8.
    const arma::mat X_ndim_ntrain_mat = {{0.5, 0.1, 1e6, 0.3, 0.2, 0.4, 0.25}, {12.0, 2.0, 8.0, 4.0, 6.0, 10.0, 14.0}};
    const arma::vec x_median_vec = {0.3, 8.0};

    bool stationary_ok = true, other_ok = true;
    for (int i_kernel = 0; i_kernel < static_cast<int>(KernelId::End); ++i_kernel) {
        const gpflash::kernel::KernelBase& kern_fun = gpflash::kernel::kernel_by_id(static_cast<KernelId>(i_kernel));
        const arma::vec x_center_vec = kern_fun.calc_x_center_vec(X_ndim_ntrain_mat, gpflash::SymmetrySpec());
        if (kern_fun.is_stationary())
            stationary_ok &= calc_max_abs_diff(x_center_vec, x_median_vec) == 0.0;
        else
            other_ok &= calc_max_abs_diff(x_center_vec, arma::zeros<arma::vec>(2)) == 0.0;
    }

    gpflash::SymmetrySpec symmetry_spec;
    symmetry_spec.mirror_sign_vec = {-1.0, 1.0};
    const arma::vec x_mirror_center_vec =
            gpflash::kernel::kernel_by_id(KernelId::Matern52).calc_x_center_vec(X_ndim_ntrain_mat, symmetry_spec);
    const arma::vec x_mirror_expected_vec = {0.0, 8.0};
    const bool mirror_ok = calc_max_abs_diff(x_mirror_center_vec, x_mirror_expected_vec) == 0.0;

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "stationary at median %d  others at 0 %d  flipped dimension at 0 %d",
            static_cast<int>(stationary_ok),
            static_cast<int>(other_ok),
            static_cast<int>(mirror_ok));
    return report(name, stationary_ok && other_ok && mirror_ok, detail);
}

// ── 2) stationary_declaration ────────────────────────────────────────────────

// K's lower triangle from the points as given, with no centering of its own.
arma::mat build_kmat_lower(const gpflash::kernel::KernelBase& kern_fun, const arma::mat& X_ndim_ntrain_mat) {
    const gpflash::HpVal hp_val_struct = make_hp_val(kern_fun.kernel_id());
    const gpflash::detail::KernelAssembly kern_assembly = gpflash::detail::build_kern_assembly(
            kern_fun,
            X_ndim_ntrain_mat,
            arma::zeros<arma::vec>(X_ndim_ntrain_mat.n_rows),
            hp_val_struct.hp_gamma_vec,
            hp_val_struct.hp_kern_extra_vec,
            gpflash::SymmetrySpec(),
            gpflash::ModelOptions().cond_max,
            gpflash::EigMaxUbMethod::Gershgorin,
            /*eval_grad=*/false);
    return arma::trimatl(kern_assembly.Buffer_mat);
}

bool test_stationary_declaration(const char* name) {
    const arma::mat X_ndim_ntrain_mat = make_grid_points(kNTrain, 7);
    const arma::mat X_shifted_ndim_ntrain_mat = X_ndim_ntrain_mat + 0.75;

    bool stationary_ok = true, other_ok = true;
    double stationary_diff_max = 0.0, other_diff_min = arma::datum::inf;
    for (int i_kernel = 0; i_kernel < static_cast<int>(KernelId::End); ++i_kernel) {
        const gpflash::kernel::KernelBase& kern_fun = gpflash::kernel::kernel_by_id(static_cast<KernelId>(i_kernel));
        const double kmat_diff = calc_max_abs_diff(
                build_kmat_lower(kern_fun, X_ndim_ntrain_mat), build_kmat_lower(kern_fun, X_shifted_ndim_ntrain_mat));
        if (kern_fun.is_stationary()) {
            stationary_diff_max = std::max(stationary_diff_max, kmat_diff);
            stationary_ok &= kmat_diff <= kTolExact;
        } else {
            other_diff_min = std::min(other_diff_min, kmat_diff);
            other_ok &= kmat_diff > kInnerProductKmatChangeMin;
        }
    }

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "stationary K max change %.3g  inner-product K min change %.3g",
            stationary_diff_max,
            other_diff_min);
    return report(name, stationary_ok && other_ok, detail);
}

// ── 3) offset_leaves_model_unchanged ─────────────────────────────────────────

double calc_mll_grad_diff(const gpflash::MllResult& a_result, const gpflash::MllResult& b_result) {
    const gpflash::MllHpGrad& a_grad = a_result.mll_hp_grad_struct;
    const gpflash::MllHpGrad& b_grad = b_result.mll_hp_grad_struct;
    double diff_max = std::fabs(a_result.mll - b_result.mll);
    diff_max = std::max(diff_max, std::fabs(a_result.hp_sig2_k - b_result.hp_sig2_k));
    diff_max = std::max(diff_max, calc_max_abs_diff(a_result.hp_beta_vec, b_result.hp_beta_vec));
    diff_max = std::max(diff_max, calc_max_abs_diff(a_grad.d_mll_d_hp_gamma_vec, b_grad.d_mll_d_hp_gamma_vec));
    diff_max =
            std::max(diff_max, calc_max_abs_diff(a_grad.d_mll_d_hp_kern_extra_vec, b_grad.d_mll_d_hp_kern_extra_vec));
    return diff_max;
}

double calc_posterior_diff(const gpflash::PosteriorResult& a_result, const gpflash::PosteriorResult& b_result) {
    double diff_max = calc_max_abs_diff(a_result.mean_post_vec, b_result.mean_post_vec);
    diff_max = std::max(diff_max, calc_max_abs_diff(a_result.var_post_vec, b_result.var_post_vec));
    diff_max = std::max(
            diff_max,
            calc_max_abs_diff(
                    a_result.d_mean_post_d_x_post_ndim_npost_mat, b_result.d_mean_post_d_x_post_ndim_npost_mat));
    diff_max = std::max(
            diff_max,
            calc_max_abs_diff(
                    a_result.d_var_post_d_x_post_ndim_npost_mat, b_result.d_var_post_d_x_post_ndim_npost_mat));
    return diff_max;
}

gpflash::PosteriorOptions all_posterior_fields() {
    gpflash::PosteriorOptions post_opts;
    post_opts.return_mean_gradient = true;
    post_opts.return_var_gradient = true;
    return post_opts;
}

bool test_offset_leaves_model_unchanged(const char* name) {
    const arma::mat X_ndim_ntrain_mat = make_grid_points(kNTrain, 7);
    const arma::mat X_ndim_npost_mat = make_grid_points(9, 11) + 1.0 / 128.0;
    const gpflash::PosteriorOptions post_opts = all_posterior_fields();

    bool ok = true;
    std::string worst_label;
    double diff_max = 0.0;
    for (const OffsetCase& offset_case : make_offset_cases()) {
        const gpflash::ModelOptions model_opts = make_model_opts(offset_case);
        const arma::vec offset_vec = make_offset_vec(offset_case.symmetry_spec);
        const gpflash::TrainingData train_data = make_train_data(X_ndim_ntrain_mat);
        gpflash::TrainingData train_data_shifted = train_data;
        train_data_shifted.X_ndim_ntrain_mat.each_col() += offset_vec;
        gpflash::HpVal hp_val_struct = make_hp_val(offset_case.kernel_id);

        double case_diff = arma::datum::inf;
        try {
            const gpflash::MllResult mll_result = gpflash::eval_mll(train_data, hp_val_struct, model_opts, true);
            const gpflash::MllResult mll_result_shifted =
                    gpflash::eval_mll(train_data_shifted, hp_val_struct, model_opts, true);
            hp_val_struct.hp_sig2_k = mll_result.hp_sig2_k;
            const gpflash::GpPosterior posterior = gpflash::make_posterior(train_data, hp_val_struct, model_opts);
            const gpflash::GpPosterior posterior_shifted =
                    gpflash::make_posterior(train_data_shifted, hp_val_struct, model_opts);
            arma::mat X_shifted_ndim_npost_mat = X_ndim_npost_mat;
            X_shifted_ndim_npost_mat.each_col() += offset_vec;
            case_diff = std::max(
                    calc_mll_grad_diff(mll_result, mll_result_shifted),
                    calc_posterior_diff(
                            posterior.eval(X_ndim_npost_mat, post_opts),
                            posterior_shifted.eval(X_shifted_ndim_npost_mat, post_opts)));
            for (arma::uword i_post = 0; i_post < X_ndim_npost_mat.n_cols; ++i_post) {
                const arma::vec x_post_vec = X_ndim_npost_mat.col(i_post);
                const arma::vec x_shifted_post_vec = X_shifted_ndim_npost_mat.col(i_post);
                case_diff = std::max(
                        case_diff,
                        calc_posterior_diff(
                                posterior.eval(x_post_vec, post_opts),
                                posterior_shifted.eval(x_shifted_post_vec, post_opts)));
            }
        } catch (...) {
        }
        if (!(case_diff <= diff_max)) {
            diff_max = case_diff;
            worst_label = case_label(offset_case);
        }
        ok &= case_diff == 0.0;
    }

    char detail[kDetailLen];
    std::snprintf(detail, sizeof(detail), "largest change %.3g (%s)", diff_max, worst_label.c_str());
    return report(name, ok, detail);
}

// ── 4) offset_survives_cache ─────────────────────────────────────────────────

bool test_offset_survives_cache(const char* name) {
    const std::string path = "test_kernel_center_cache.h5";
    const OffsetCase offset_case{KernelId::Matern52, gpflash::SymmetrySpec(), true};
    const gpflash::ModelOptions model_opts = make_model_opts(offset_case);
    const arma::vec offset_vec = make_offset_vec(offset_case.symmetry_spec);
    gpflash::TrainingData train_data_shifted = make_train_data(make_grid_points(kNTrain, 7));
    train_data_shifted.X_ndim_ntrain_mat.each_col() += offset_vec;
    gpflash::HpVal hp_val_struct = make_hp_val(offset_case.kernel_id);
    hp_val_struct.hp_sig2_k = gpflash::eval_mll(train_data_shifted, hp_val_struct, model_opts).hp_sig2_k;
    arma::mat X_shifted_ndim_npost_mat = make_grid_points(9, 11) + 1.0 / 128.0;
    X_shifted_ndim_npost_mat.each_col() += offset_vec;
    const gpflash::PosteriorOptions post_opts = all_posterior_fields();

    double diff_max = arma::datum::inf;
    try {
        const gpflash::GpPosterior posterior = gpflash::make_posterior(train_data_shifted, hp_val_struct, model_opts);
        posterior.save_precompute(path);
        const gpflash::GpPosterior posterior_loaded = gpflash::load_posterior(path);
        diff_max = calc_posterior_diff(
                posterior.eval(X_shifted_ndim_npost_mat, post_opts),
                posterior_loaded.eval(X_shifted_ndim_npost_mat, post_opts));
    } catch (...) {
    }
    std::remove(path.c_str());

    char detail[kDetailLen];
    std::snprintf(detail, sizeof(detail), "loaded against built, largest change %.3g", diff_max);
    return report(name, diff_max == 0.0, detail);
}

// ── 5) offset_leaves_fit_unchanged ───────────────────────────────────────────

bool test_offset_leaves_fit_unchanged(const char* name) {
    const OffsetCase offset_case{KernelId::Matern52, gpflash::SymmetrySpec(), true};
    gpflash::OptzOptions optz_opts;
    optz_opts.model_options = make_model_opts(offset_case);
    const gpflash::TrainingData train_data = make_train_data(make_grid_points(kNTrain, 7));
    gpflash::TrainingData train_data_shifted = train_data;
    train_data_shifted.X_ndim_ntrain_mat.each_col() += make_offset_vec(offset_case.symmetry_spec);

    double diff_max = arma::datum::inf;
    try {
        const gpflash::MllOptzResult mll_optz_result = gpflash::select_hp_with_max_mll(train_data, optz_opts);
        const gpflash::MllOptzResult mll_optz_result_shifted =
                gpflash::select_hp_with_max_mll(train_data_shifted, optz_opts);
        diff_max = std::max(
                std::fabs(mll_optz_result.mll() - mll_optz_result_shifted.mll()),
                calc_max_abs_diff(
                        mll_optz_result.hp_val_struct.hp_gamma_vec,
                        mll_optz_result_shifted.hp_val_struct.hp_gamma_vec));
    } catch (...) {
    }

    char detail[kDetailLen];
    std::snprintf(detail, sizeof(detail), "MLL and hp_gamma, largest change %.3g", diff_max);
    return report(name, diff_max == 0.0, detail);
}

}  // namespace

int main() {
    std::printf("== GpFlash kernel center ==\n");
    bool ok = true;
    ok &= test_center_by_kernel("center_by_kernel");
    ok &= test_stationary_declaration("stationary_declaration");
    ok &= test_offset_leaves_model_unchanged("offset_leaves_model_unchanged");
    ok &= test_offset_survives_cache("offset_survives_cache");
    ok &= test_offset_leaves_fit_unchanged("offset_leaves_fit_unchanged");
    std::printf("\n%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");

    return ok ? 0 : 1;
}
