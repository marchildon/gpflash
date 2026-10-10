// The symmetry kernel k_sym(x, y) = k(x, y) + parity_sign k(x, S y), over each base kernel in kSymmetrizableKernelIds.
//
// Value references come from the base kernel's closed form in raw inputs, not from calling the kernel, so a wrong
// value loop cannot satisfy both sides; gradients are checked against central differences of the values. The mirrored
// term is what the checks are aimed at: its squared distance, its packed derivative factor, and the diagonal it
// makes vary.
#include <algorithm>
#include <armadillo>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <random>
#include <string>

#include "GpFlashTestSupport.hpp"
#include "gpflash/assembly/SymmetryMatrix.hpp"
#include "gpflash/inputs/SymmetrySpec.hpp"
#include "gpflash/kernels/KernelDispatch.hpp"
#include "gpflash/mean/MeanModel.hpp"
#include "gpflash/optimization/HpOptimizer.hpp"
#include "gpflash/posterior_build/PosteriorBuild.hpp"  // make_posterior, load_posterior_verified
#include "gpflash/posterior_cache/CacheInputHash.hpp"  // calc_posterior_input_hash
#include "gpflash/posterior_cache/PosteriorCache.hpp"  // peek_info
#include "gpflash/posterior_eval/GpPosterior.hpp"
#include "gpflash/training/EvalMllDispatch.hpp"

using gpflash_test::draw_uniform_vec;
using gpflash_test::kDetailLen;
using gpflash_test::kFdScaleFloor;
using gpflash_test::kFdStepObj;
using gpflash_test::kTolAltPath;
using gpflash_test::kTolExact;
using gpflash_test::kTolFdObj;
using gpflash_test::make_points;
using gpflash_test::rejects;
using gpflash_test::report;

namespace {

// Seeds each test's own std::mt19937_64, so its data depends neither on the Armadillo version nor on test order.
constexpr std::uint64_t kDataSeed = 57;

// One cosine-free closed form, the squared exponential on a scaled squared distance.
double calc_sq_exp(const double rdot_sq_norm) { return std::exp(-0.5 * rdot_sq_norm); }

// k(x_a, x_b) from each base kernel's closed form in raw inputs, independent of the library's feature maps.
double calc_kern_ref(
        const gpflash::kernel::KernelId kernel_id,
        const arma::vec& x_a_vec,
        const arma::vec& x_b_vec,
        const arma::vec& hp_gamma_vec,
        const arma::vec& hp_kern_extra_vec) {
    double rdot_sq_norm = 0.0, amplitude_sin_sq_sum = 0.0;
    for (arma::uword i_dim = 0; i_dim < hp_gamma_vec.n_elem; ++i_dim) {
        const double r_dim = x_a_vec(i_dim) - x_b_vec(i_dim);
        const double rdot_dim = hp_gamma_vec(i_dim) * r_dim;
        rdot_sq_norm += rdot_dim * rdot_dim;
        if (kernel_id == gpflash::kernel::KernelId::Periodic) {
            const double sin_theta = std::sin(arma::datum::pi * rdot_dim);
            amplitude_sin_sq_sum += hp_kern_extra_vec(i_dim) * hp_kern_extra_vec(i_dim) * sin_theta * sin_theta;
        }
    }

    double kern_val = calc_sq_exp(rdot_sq_norm);
    if (kernel_id == gpflash::kernel::KernelId::Matern52) {
        const double sqrt5_rdot_norm = std::sqrt(5.0 * rdot_sq_norm);
        kern_val = (1.0 + sqrt5_rdot_norm + sqrt5_rdot_norm * sqrt5_rdot_norm / 3.0) * std::exp(-sqrt5_rdot_norm);
    } else if (kernel_id == gpflash::kernel::KernelId::RatQuad) {
        const double hp_rho = hp_kern_extra_vec(0);
        kern_val = std::pow(1.0 + rdot_sq_norm / (2.0 * hp_rho), -hp_rho);
    } else if (kernel_id == gpflash::kernel::KernelId::Periodic) {
        kern_val = std::exp(-2.0 * amplitude_sin_sq_sum);
    }

    return kern_val;
}

// The base kernels these tests run a symmetry model on: one with each kind of extras, and Matern52, the default kernel.
const gpflash::kernel::KernelId kSymmetrizableKernelIds[] = {
        gpflash::kernel::KernelId::SqExp,
        gpflash::kernel::KernelId::Matern52,
        gpflash::kernel::KernelId::RatQuad,
        gpflash::kernel::KernelId::Periodic};

// Non-unit extras for kernel_id: RatQuad's shape, Periodic's per-dimension amplitudes, or none.
arma::vec make_hp_kern_extra_vec(const gpflash::kernel::KernelId kernel_id, const int n_dim) {
    arma::vec hp_kern_extra_vec;
    if (kernel_id == gpflash::kernel::KernelId::RatQuad) {
        hp_kern_extra_vec = {gpflash_test::kRatQuadHpRho};
    } else if (kernel_id == gpflash::kernel::KernelId::Periodic) {
        hp_kern_extra_vec.set_size(n_dim);
        for (int i_dim = 0; i_dim < n_dim; ++i_dim) hp_kern_extra_vec(i_dim) = 0.8 + 0.15 * i_dim;
    }

    return hp_kern_extra_vec;
}

arma::vec make_hp_gamma_vec(const int n_dim) {
    arma::vec hp_gamma_vec(n_dim);
    for (int i_dim = 0; i_dim < n_dim; ++i_dim) hp_gamma_vec(i_dim) = 0.7 + 0.3 * i_dim;
    return hp_gamma_vec;
}

// The mirrored term's scaled squared distance, sum_d hp_gamma_d^2 (x_id - s_d x_jd)^2, straight from the
// definition. The base term is the same with every sign +1.
double calc_mirrored_sq_dist(
        const arma::mat& X_ndim_ntrain_mat,
        const int i_row,
        const int i_col,
        const arma::vec& hp_gamma_vec,
        const arma::vec& mirror_sign_vec) {
    double sq_dist = 0.0;
    for (arma::uword i_dim = 0; i_dim < hp_gamma_vec.n_elem; ++i_dim) {
        const double rdot = hp_gamma_vec(i_dim)
                * (X_ndim_ntrain_mat(i_dim, i_row) - mirror_sign_vec(i_dim) * X_ndim_ntrain_mat(i_dim, i_col));
        sq_dist += rdot * rdot;
    }

    return sq_dist;
}

// Asserts K_sym, its varying diagonal, and the packed mirrored c^(rdot) against the closed form.
//
// The packed factor is the memory the mirrored term exists for: the base term's c^(rdot) lives in the strict
// upper triangle, which the mirrored term cannot share. For this kernel c^(rdot) = -k, so a packed entry that
// does not equal minus the mirrored value is either mispacked or captured from the wrong distance.
bool test_kmat_vs_closed_form(const char* name, const bool is_even_symmetry, const int n_train) {
    const int n_dim = 3;
    const arma::mat X_ndim_ntrain_mat = make_points(n_dim, n_train);
    const arma::vec hp_gamma_vec = make_hp_gamma_vec(n_dim);
    const arma::mat Ydot_nfeature_ntrain_mat = X_ndim_ntrain_mat.each_col() % hp_gamma_vec;

    // A mirror that flips one of three dimensions, so the preserved and flipped halves are both non-empty
    // and the two Gram halves differ.
    gpflash::SymmetrySpec symmetry_spec;
    symmetry_spec.mirror_sign_vec = arma::vec{1.0, -1.0, 1.0};
    symmetry_spec.is_even_symmetry = is_even_symmetry;
    // Spelled out rather than taken from the spec, so the reference does not read its weight through the code
    // under test.
    const double parity_sign = is_even_symmetry ? 1.0 : -1.0;

    arma::vec ydot_train_sq_norm_vec, kern_common_rdot_mirror_packed_vec;
    const arma::mat K_sym_mat_lower = gpflash::detail::build_symmetry_kmat_lower_tri(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::SqExp),
            Ydot_nfeature_ntrain_mat,
            symmetry_spec,
            nullptr,
            ydot_train_sq_norm_vec,
            &kern_common_rdot_mirror_packed_vec);

    const arma::vec no_mirror_sign_vec(n_dim, arma::fill::ones);
    double worst_pair = 0.0, worst_diag = 0.0, worst_packed = 0.0;
    // Walked in the packed order itself -- column by column, diagonal entry first -- rather than through the
    // offset the build uses, so a wrong offset cannot satisfy both sides.
    arma::uword idx_packed = 0;
    for (int i_col = 0; i_col < n_train; ++i_col) {
        for (int i_row = i_col; i_row < n_train; ++i_row) {
            const double kern_base = calc_sq_exp(
                    calc_mirrored_sq_dist(X_ndim_ntrain_mat, i_row, i_col, hp_gamma_vec, no_mirror_sign_vec));
            const double kern_mirror = calc_sq_exp(calc_mirrored_sq_dist(
                    X_ndim_ntrain_mat, i_row, i_col, hp_gamma_vec, symmetry_spec.mirror_sign_vec));
            const double kern_sym_ref = kern_base + parity_sign * kern_mirror;

            const double kern_err = std::fabs(K_sym_mat_lower(i_row, i_col) - kern_sym_ref);
            if (i_row == i_col)
                worst_diag = std::max(worst_diag, kern_err);
            else
                worst_pair = std::max(worst_pair, kern_err);

            // c^(rdot) = -k for this kernel, at the mirrored distance, in the packed lower triangle.
            worst_packed =
                    std::max(worst_packed, std::fabs(kern_common_rdot_mirror_packed_vec(idx_packed) + kern_mirror));
            ++idx_packed;
        }
    }

    // The packed factor spans the lower triangle with the diagonal and nothing more, which the walk above just
    // stepped through entry by entry.
    const bool packed_len_ok = kern_common_rdot_mirror_packed_vec.n_elem == idx_packed;
    // Summing the two terms moves the diagonal off 1, which is what makes the matrix need preconditioning.
    const bool diag_varies = std::fabs(K_sym_mat_lower(0, 0) - 1.0) > kTolExact;

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "parity %+.0f  pair %.2e  diag %.2e  packed %.2e  len_ok %d  diag_varies %d",
            parity_sign,
            worst_pair,
            worst_diag,
            worst_packed,
            static_cast<int>(packed_len_ok),
            static_cast<int>(diag_varies));

    return report(
            name,
            worst_pair < kTolExact && worst_diag < kTolExact && worst_packed < kTolExact && packed_len_ok
                    && diag_varies,
            detail);
}

// A query's k_* is built against the squared norms assembly returns, so they must be the norms of the scaled points
// it was handed, with every feature counted once. The reference sums hp_gamma_l^2 x_l^2 by hand, so it shares
// neither Ydot nor Armadillo's dot with the library.
bool test_sq_norms_are_the_points(const char* name) {
    const int n_dim = 4, n_train = 9;
    const arma::mat X_ndim_ntrain_mat = make_points(n_dim, n_train);
    const arma::vec hp_gamma_vec = make_hp_gamma_vec(n_dim);
    const arma::mat Ydot_nfeature_ntrain_mat = X_ndim_ntrain_mat.each_col() % hp_gamma_vec;

    gpflash::SymmetrySpec symmetry_spec;
    symmetry_spec.mirror_sign_vec = arma::vec{-1.0, 1.0, 1.0, -1.0};  // two flipped, two preserved

    arma::vec ydot_train_sq_norm_vec;
    const arma::mat K_sym_mat_lower = gpflash::detail::build_symmetry_kmat_lower_tri(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::SqExp),
            Ydot_nfeature_ntrain_mat,
            symmetry_spec,
            nullptr,
            ydot_train_sq_norm_vec,
            nullptr);
    (void)K_sym_mat_lower;

    double worst = 0.0;
    for (int i_train = 0; i_train < n_train; ++i_train) {
        double sq_norm_ref = 0.0;
        for (int i_dim = 0; i_dim < n_dim; ++i_dim) {
            const double ydot = hp_gamma_vec(i_dim) * X_ndim_ntrain_mat(i_dim, i_train);
            sq_norm_ref += ydot * ydot;
        }
        worst = std::max(worst, std::fabs(ydot_train_sq_norm_vec(i_train) - sq_norm_ref));
    }

    char detail[kDetailLen];
    std::snprintf(detail, sizeof(detail), "worst %.2e", worst);
    return report(name, worst < kTolExact, detail);
}

// A mirror that flips nothing is not a mirror, and an entry that is neither +1 nor -1 is not a sign. The parity
// needs no case: a bool cannot hold a third value.
bool test_rejects_degenerate_spec(const char* name) {
    gpflash::SymmetrySpec no_flip_spec;
    no_flip_spec.mirror_sign_vec = arma::vec{1.0, 1.0};
    gpflash::SymmetrySpec bad_sign_spec;
    bad_sign_spec.mirror_sign_vec = arma::vec{1.0, 0.5};

    const bool rejects_no_flip =
            gpflash_test::rejects([&] { gpflash::detail::check_symmetry_spec(no_flip_spec, "test"); });
    const bool rejects_bad_sign =
            gpflash_test::rejects([&] { gpflash::detail::check_symmetry_spec(bad_sign_spec, "test"); });

    // The public entry point must run the same check on the options it is handed.
    const gpflash::TrainingData train_data{make_points(2, 8), arma::vec(8, arma::fill::ones), /*sig2_f_val=*/0.0};
    gpflash::HpVal hp_val_struct;
    hp_val_struct.hp_gamma_vec = arma::vec{1.0, 1.0};
    hp_val_struct.hp_sig2_k = 1.0;
    gpflash::ModelOptions model_opts;
    model_opts.kernel_id = gpflash::kernel::KernelId::SqExp;
    model_opts.symmetry_spec = no_flip_spec;
    const bool eval_mll_rejects_no_flip =
            gpflash_test::rejects([&] { gpflash::eval_mll(train_data, hp_val_struct, model_opts); });
    model_opts.symmetry_spec = bad_sign_spec;
    const bool eval_mll_rejects_bad_sign =
            gpflash_test::rejects([&] { gpflash::eval_mll(train_data, hp_val_struct, model_opts); });

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "no_flip %d  bad_sign %d  eval_mll %d/%d",
            static_cast<int>(rejects_no_flip),
            static_cast<int>(rejects_bad_sign),
            static_cast<int>(eval_mll_rejects_no_flip),
            static_cast<int>(eval_mll_rejects_bad_sign));

    return report(
            name, rejects_no_flip && rejects_bad_sign && eval_mll_rejects_no_flip && eval_mll_rejects_bad_sign, detail);
}

// Asserts the MLL gradient in every hyperparameter the base kernel has, hp_gamma and its extras, against a central
// difference of the MLL.
//
// Every piece of the two-term gradient has to be right at once: the mirrored pair pass reading the packed factor,
// its h taken at x_il - s_l x_jl, the varying diagonal the sum creates, and the nugget's own two-term derivative.
// Differencing the MLL value, a separate code path from the gradient, makes it independent. n_train > n_dim keeps
// the nugget load-bearing, and n_train > PERF_GRAD_MIN_PAIR_PER_DIM n_dim sends the early columns down the
// pair-inner loop and the late ones down the dimension-inner loop.
bool test_mll_hp_grad_vs_fd(
        const char* name,
        const gpflash::kernel::KernelId kernel_id,
        const bool is_even_symmetry,
        const double sig2_f_val) {
    const int n_dim = 3, n_train = 20;
    const arma::mat X_ndim_ntrain_mat = make_points(n_dim, n_train);
    std::mt19937_64 rng(kDataSeed);
    const arma::vec f_train_vec = draw_uniform_vec(rng, n_train, 2.0, 3.0);
    const arma::vec hp_gamma_vec = make_hp_gamma_vec(n_dim);
    const arma::vec hp_kern_extra_vec = make_hp_kern_extra_vec(kernel_id, n_dim);
    const gpflash::TrainingData train_data{X_ndim_ntrain_mat, f_train_vec, sig2_f_val};
    gpflash::ModelOptions model_opts;
    model_opts.kernel_id = kernel_id;
    model_opts.cond_max = 1.0e4;
    model_opts.symmetry_spec.mirror_sign_vec = arma::vec{1.0, -1.0, -1.0};
    model_opts.symmetry_spec.is_even_symmetry = is_even_symmetry;
    // The constant is even, so the odd model fits x_1, the flipped dimension's odd power, instead. One weight each,
    // which leaves no neglected mean model term in the gradient at this cond_max.
    if (!is_even_symmetry) {
        model_opts.has_mean_model_constant = false;
        model_opts.mean_model_degree_vec = arma::uvec{0, 1, 0};
    }
    const double parity_sign = is_even_symmetry ? 1.0 : -1.0;
    const bool is_noise_free = gpflash::detail::classify_noise_request(sig2_f_val) == gpflash::NoiseRequest::NoiseFree;
    const double hp_sig2_k = is_noise_free ? 1.0 : 2.3;
    // Several periods across these points curve the periodic MLL enough that the objective step's truncation error
    // exceeds the tolerance. A central difference's error goes as the step squared, so a tenth of the step clears it.
    const bool is_periodic = kernel_id == gpflash::kernel::KernelId::Periodic;
    const double fd_step = is_periodic ? 0.1 * kFdStepObj : kFdStepObj;
    const double tol_fd = is_periodic ? gpflash_test::kTolFdObjOsc : kTolFdObj;

    // hp_gamma then the extras, the order the gradient returns them in.
    const arma::vec hp_vec = arma::join_cols(hp_gamma_vec, hp_kern_extra_vec);
    const auto mll_at_hp = [&](const arma::vec& hp_at_vec, const bool eval_grad) {
        gpflash::HpVal hp_val_struct;
        hp_val_struct.hp_gamma_vec = hp_at_vec.head(n_dim);
        hp_val_struct.hp_kern_extra_vec = hp_at_vec.tail(hp_at_vec.n_elem - n_dim);
        hp_val_struct.hp_sig2_k = hp_sig2_k;
        return gpflash::eval_mll(train_data, hp_val_struct, model_opts, eval_grad);
    };
    const gpflash::MllResult mll_result = mll_at_hp(hp_vec, /*eval_grad=*/true);
    const arma::vec d_mll_d_hp_vec = arma::join_cols(
            mll_result.mll_hp_grad_struct.d_mll_d_hp_gamma_vec,
            mll_result.mll_hp_grad_struct.d_mll_d_hp_kern_extra_vec);

    double worst = 0.0;
    for (arma::uword i_hp = 0; i_hp < hp_vec.n_elem; ++i_hp) {
        arma::vec hp_plus_vec = hp_vec, hp_minus_vec = hp_vec;
        hp_plus_vec(i_hp) += fd_step;
        hp_minus_vec(i_hp) -= fd_step;
        const double mll_plus = mll_at_hp(hp_plus_vec, /*eval_grad=*/false).mll;
        const double mll_minus = mll_at_hp(hp_minus_vec, /*eval_grad=*/false).mll;
        const double grad_fd = (mll_plus - mll_minus) / (2.0 * fd_step);
        const double grad_err = std::fabs(d_mll_d_hp_vec(i_hp) - grad_fd) / std::max(std::fabs(grad_fd), kFdScaleFloor);
        worst = std::max(worst, grad_err);
    }

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "parity %+.0f  sigma_f^2 %.3f  n_hp %d  worst rel %.2e",
            parity_sign,
            sig2_f_val,
            static_cast<int>(hp_vec.n_elem),
            worst);
    return report(name, worst < tol_fd, detail);
}

// Asserts the posterior mean and variance against a dense reference built from the two-term kernel.
//
// The query path mirrors the query point rather than the training points, which is the opposite side from
// assembly, so this is what catches a mirror applied to the wrong operand. The prior variance is the other
// target: it is 1 + parity_sign k(4 ||ydot' flipped||^2), not the base kernel's constant 1.
bool test_posterior_vs_dense_reference(
        const char* name, const gpflash::kernel::KernelId kernel_id, const bool is_even_symmetry) {
    const int n_dim = 3, n_train = 13, n_post = 5;
    const arma::mat X_ndim_ntrain_mat = make_points(n_dim, n_train);
    std::mt19937_64 rng(kDataSeed);
    const arma::vec f_train_vec = draw_uniform_vec(rng, n_train, 1.5, 2.5);
    const arma::vec hp_gamma_vec = make_hp_gamma_vec(n_dim);
    // Query 0 lies on the mirror plane, where an odd model is zero.
    arma::mat X_ndim_npost_mat = 0.35 + 0.3 * make_points(n_dim, n_post);
    X_ndim_npost_mat(1, 0) = 0.0;
    const arma::vec mirror_sign_vec{1.0, -1.0, 1.0};
    const double sig2_f_val = 0.02, hp_sig2_k = 1.7;
    const arma::vec hp_kern_extra_vec = make_hp_kern_extra_vec(kernel_id, n_dim);
    gpflash::TrainingData train_data;
    train_data.X_ndim_ntrain_mat = X_ndim_ntrain_mat;
    train_data.f_train_vec = f_train_vec;
    train_data.sig2_f_val = sig2_f_val;
    gpflash::HpVal hp_val_struct;
    hp_val_struct.hp_gamma_vec = hp_gamma_vec;
    hp_val_struct.hp_kern_extra_vec = hp_kern_extra_vec;
    hp_val_struct.hp_sig2_k = hp_sig2_k;
    gpflash::ModelOptions model_opts;
    model_opts.kernel_id = kernel_id;
    model_opts.cond_max = 1.0e6;
    model_opts.symmetry_spec.mirror_sign_vec = mirror_sign_vec;
    model_opts.symmetry_spec.is_even_symmetry = is_even_symmetry;
    // One weight each: the constant for the even model, and x_1, the flipped dimension's odd power, for the odd one.
    if (!is_even_symmetry) {
        model_opts.has_mean_model_constant = false;
        model_opts.mean_model_degree_vec = arma::uvec{0, 1, 0};
    }
    const double parity_sign = is_even_symmetry ? 1.0 : -1.0;
    const auto calc_basis_val = [is_even_symmetry](const arma::vec& x_vec) {
        return is_even_symmetry ? 1.0 : x_vec(1);
    };

    const auto posterior = gpflash::make_posterior(train_data, hp_val_struct, model_opts);

    // The reference: k_sym from the base kernel's closed form in raw inputs, then Sigma inverted densely. The
    // mirror is applied to the raw point, so no feature map is re-derived through the code under test.
    const auto calc_kern_sym = [&](const arma::vec& x_a_vec, const arma::vec& x_b_vec) {
        const arma::vec x_b_mirrored_vec = mirror_sign_vec % x_b_vec;
        const double kern_base = calc_kern_ref(kernel_id, x_a_vec, x_b_vec, hp_gamma_vec, hp_kern_extra_vec);
        const double kern_mirror = calc_kern_ref(kernel_id, x_a_vec, x_b_mirrored_vec, hp_gamma_vec, hp_kern_extra_vec);
        return kern_base + parity_sign * kern_mirror;
    };

    arma::mat K_sym_mat(n_train, n_train);
    for (int i_row = 0; i_row < n_train; ++i_row)
        for (int i_col = 0; i_col < n_train; ++i_col)
            K_sym_mat(i_row, i_col) = calc_kern_sym(X_ndim_ntrain_mat.col(i_row), X_ndim_ntrain_mat.col(i_col));

    // The library conditions K_sym through its preconditioned nugget, so the reference reproduces that nugget
    // from the same Gershgorin bound rather than assuming none.
    const arma::vec precon_scale_vec = arma::sqrt(K_sym_mat.diag());
    arma::mat K_corr_mat = K_sym_mat;
    K_corr_mat.each_col() /= precon_scale_vec;
    K_corr_mat.each_row() /= precon_scale_vec.t();
    double max_abs_row_sum = 0.0;
    for (int i_row = 0; i_row < n_train; ++i_row) {
        double abs_row_sum = 0.0;
        for (int i_col = 0; i_col < n_train; ++i_col)
            if (i_col != i_row) abs_row_sum += std::fabs(K_corr_mat(i_row, i_col));
        max_abs_row_sum = std::max(max_abs_row_sum, abs_row_sum);
    }
    const double eta = (1.0 + max_abs_row_sum) / (model_opts.cond_max - 1.0);

    arma::mat Sigma_mat = hp_sig2_k * K_sym_mat;
    Sigma_mat.diag() += hp_sig2_k * eta * arma::square(precon_scale_vec) + sig2_f_val;
    const arma::mat Sigma_inv_mat = arma::inv_sympd(Sigma_mat);
    // The one weight by generalized least squares.
    arma::vec basis_train_vec(n_train);
    for (int i_train = 0; i_train < n_train; ++i_train)
        basis_train_vec(i_train) = calc_basis_val(X_ndim_ntrain_mat.col(i_train));
    const arma::vec lin_sol_basis_vec = Sigma_inv_mat * basis_train_vec;
    const double hp_beta = arma::dot(lin_sol_basis_vec, f_train_vec) / arma::dot(lin_sol_basis_vec, basis_train_vec);
    const arma::vec lin_sol_vec = Sigma_inv_mat * (f_train_vec - hp_beta * basis_train_vec);

    const double cond_sigma = arma::cond(Sigma_mat);
    const double tol_post = std::max(kTolAltPath, 1.0e2 * cond_sigma * arma::datum::eps);

    gpflash::PosteriorOptions post_opts;
    post_opts.return_mean = true;
    post_opts.return_var = true;
    const auto batch_result = posterior.eval(X_ndim_npost_mat, post_opts);
    // The posterior mean has the model's parity, mean(S x) = parity_sign mean(x), prior mean included.
    const arma::mat X_mirrored_ndim_npost_mat = X_ndim_npost_mat.each_col() % mirror_sign_vec;
    const auto mirrored_result = posterior.eval(X_mirrored_ndim_npost_mat, post_opts);
    const double worst_mirrored_mean =
            arma::abs(mirrored_result.mean_post_vec - parity_sign * batch_result.mean_post_vec).max();

    double worst_mean = 0.0, worst_var = 0.0, worst_single = 0.0;
    for (int i_post = 0; i_post < n_post; ++i_post) {
        const arma::vec x_post_vec = X_ndim_npost_mat.col(i_post);
        arma::vec k_post_vec(n_train);
        for (int i_train = 0; i_train < n_train; ++i_train)
            k_post_vec(i_train) = calc_kern_sym(X_ndim_ntrain_mat.col(i_train), x_post_vec);

        const double mean_ref = hp_beta * calc_basis_val(x_post_vec) + hp_sig2_k * arma::dot(k_post_vec, lin_sol_vec);
        // k_sym(x', x') carries the mirrored term, which a stationary kernel's constant diagonal does not.
        const double prior_var_ref = calc_kern_sym(x_post_vec, x_post_vec);
        const double var_ref =
                hp_sig2_k * (prior_var_ref - hp_sig2_k * arma::dot(k_post_vec, Sigma_inv_mat * k_post_vec));

        worst_mean = std::max(
                worst_mean,
                std::fabs(batch_result.mean_post_vec(i_post) - mean_ref) / std::max(std::fabs(mean_ref), 1.0));
        worst_var = std::max(
                worst_var, std::fabs(batch_result.var_post_vec(i_post) - var_ref) / std::max(prior_var_ref, 1.0));

        // The single-query path is a separate implementation, so it must agree with the batch one.
        const auto single_result = posterior.eval(x_post_vec, post_opts);
        worst_single = std::max(worst_single, std::fabs(single_result.mean_post_vec(0) - mean_ref));
        worst_single = std::max(worst_single, std::fabs(single_result.var_post_vec(0) - var_ref));
    }

    // On the mirror plane k_post is exactly zero, and so is x_1, so an odd model's mean is exactly zero.
    const bool on_plane_mean_ok = is_even_symmetry || batch_result.mean_post_vec(0) == 0.0;

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "parity %+.0f  mean %.2e  var %.2e  single %.2e  mirrored %.2e  on_plane %d  (tol %.1e)",
            parity_sign,
            worst_mean,
            worst_var,
            worst_single,
            worst_mirrored_mean,
            static_cast<int>(on_plane_mean_ok),
            tol_post);

    return report(
            name,
            worst_mean < tol_post && worst_var < tol_post && worst_single < tol_post && worst_mirrored_mean < tol_post
                    && on_plane_mean_ok,
            detail);
}

// A symmetry posterior survives a save and a load: every query field comes back bit for bit, the file records the
// parity, and a verified load refuses the same inputs without the mirror. The mirror and the parity are both part of
// the input hash, so neither can be dropped from a request and still match.
bool test_cache_round_trip(const char* name, const gpflash::kernel::KernelId kernel_id, const bool is_even_symmetry) {
    const int n_dim = 3, n_train = 13, n_post = 4;
    gpflash::TrainingData train_data;
    train_data.X_ndim_ntrain_mat = make_points(n_dim, n_train);
    std::mt19937_64 rng(kDataSeed);
    train_data.f_train_vec = draw_uniform_vec(rng, n_train, 1.5, 2.5);
    train_data.sig2_f_val = 0.02;
    gpflash::HpVal hp_val_struct;
    hp_val_struct.hp_gamma_vec = make_hp_gamma_vec(n_dim);
    hp_val_struct.hp_kern_extra_vec = make_hp_kern_extra_vec(kernel_id, n_dim);
    hp_val_struct.hp_sig2_k = 1.7;
    gpflash::ModelOptions model_opts;
    model_opts.kernel_id = kernel_id;
    model_opts.symmetry_spec.mirror_sign_vec = {1.0, -1.0, 1.0};
    model_opts.symmetry_spec.is_even_symmetry = is_even_symmetry;
    // The odd model's one weight is on x_1, the flipped dimension's odd power.
    if (!is_even_symmetry) {
        model_opts.has_mean_model_constant = false;
        model_opts.mean_model_degree_vec = arma::uvec{0, 1, 0};
    }

    const gpflash::GpPosterior posterior = gpflash::make_posterior(train_data, hp_val_struct, model_opts);
    const std::string path = "test_symmetry_round_trip.h5";
    std::remove(path.c_str());
    posterior.save_precompute(path);
    const gpflash::cache::SymmetryKind stored_symmetry_kind = gpflash::cache::peek_info(path).symmetry_kind;
    const gpflash::GpPosterior posterior_loaded = gpflash::load_posterior(path);
    const gpflash::GpPosterior posterior_verified =
            gpflash::load_posterior_verified(path, train_data, hp_val_struct, model_opts);
    gpflash::ModelOptions model_opts_no_mirror = model_opts;
    model_opts_no_mirror.symmetry_spec = gpflash::SymmetrySpec{};
    const bool no_mirror_refused =
            rejects([&] { gpflash::load_posterior_verified(path, train_data, hp_val_struct, model_opts_no_mirror); });
    std::remove(path.c_str());

    const arma::mat X_ndim_npost_mat = 0.35 + 0.3 * make_points(n_dim, n_post);
    const gpflash::PosteriorOptions post_opts{true, true, true, true};
    const gpflash::PosteriorResult saved_result = posterior.eval(X_ndim_npost_mat, post_opts);
    const auto is_identical = [&](const gpflash::GpPosterior& other_posterior) {
        const gpflash::PosteriorResult other_result = other_posterior.eval(X_ndim_npost_mat, post_opts);
        return arma::all(other_result.mean_post_vec == saved_result.mean_post_vec)
                && arma::all(other_result.var_post_vec == saved_result.var_post_vec)
                && arma::all(arma::vectorise(
                        other_result.d_mean_post_d_x_post_ndim_npost_mat
                        == saved_result.d_mean_post_d_x_post_ndim_npost_mat))
                && arma::all(arma::vectorise(
                        other_result.d_var_post_d_x_post_ndim_npost_mat
                        == saved_result.d_var_post_d_x_post_ndim_npost_mat));
    };
    const bool loaded_identical = is_identical(posterior_loaded);
    const bool verified_identical = is_identical(posterior_verified);

    // The parity alone changes the hash, at a fixed mean model.
    gpflash::ModelOptions model_opts_other_parity = model_opts;
    model_opts_other_parity.symmetry_spec.is_even_symmetry = !is_even_symmetry;
    const gpflash::mean::MeanModel mean_model = posterior.mean_model();
    const bool parity_hashed =
            gpflash::cache::calc_posterior_input_hash(
                    kernel_id, mean_model, train_data, hp_val_struct, model_opts, train_data.sig2_f_val)
            != gpflash::cache::calc_posterior_input_hash(
                    kernel_id, mean_model, train_data, hp_val_struct, model_opts_other_parity, train_data.sig2_f_val);

    // Without a mirror the parity is never read, so it must not reach the hash either: a request that drops the mirror
    // but keeps the flag has to match a file fitted without symmetry. Flipped from model_opts_no_mirror's own flag,
    // which the reset above leaves at its default whatever the test's parity.
    gpflash::ModelOptions model_opts_no_mirror_flag_flipped = model_opts_no_mirror;
    model_opts_no_mirror_flag_flipped.symmetry_spec.is_even_symmetry =
            !model_opts_no_mirror.symmetry_spec.is_even_symmetry;
    const bool parity_unread_without_mirror =
            gpflash::cache::calc_posterior_input_hash(
                    kernel_id, mean_model, train_data, hp_val_struct, model_opts_no_mirror, train_data.sig2_f_val)
            == gpflash::cache::calc_posterior_input_hash(
                    kernel_id,
                    mean_model,
                    train_data,
                    hp_val_struct,
                    model_opts_no_mirror_flag_flipped,
                    train_data.sig2_f_val);
    const gpflash::cache::SymmetryKind expected_symmetry_kind =
            is_even_symmetry ? gpflash::cache::SymmetryKind::Even : gpflash::cache::SymmetryKind::Odd;
    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "kind %d  loaded %d  verified %d  no_mirror_refused %d  parity_hashed %d  parity_unread_without_mirror %d",
            static_cast<int>(stored_symmetry_kind),
            static_cast<int>(loaded_identical),
            static_cast<int>(verified_identical),
            static_cast<int>(no_mirror_refused),
            static_cast<int>(parity_hashed),
            static_cast<int>(parity_unread_without_mirror));
    return report(
            name,
            stored_symmetry_kind == expected_symmetry_kind && loaded_identical && verified_identical
                    && no_mirror_refused && parity_hashed && parity_unread_without_mirror,
            detail);
}

// Asserts the posterior mean and variance input-space gradients against central differences of the values.
//
// Differencing the value path, which the gradient shares nothing with beyond the precompute, is what makes this
// independent. The mirrored term is the target on both counts: its h is taken at x'_l - s_l x_il, so a sign
// dropped or applied to the wrong operand shows in the preserved and flipped dimensions differently, and the
// variance carries the extra mirrored prior term 4 parity_sign c^(rdot) hp_gamma_l ydot'_l, which a base kernel's
// constant diagonal does not have at all. A query point off the mirror plane keeps both non-zero.
bool test_posterior_grad_vs_fd(
        const char* name, const gpflash::kernel::KernelId kernel_id, const bool is_even_symmetry) {
    const int n_dim = 3, n_train = 13, n_post = 4;
    const arma::mat X_ndim_ntrain_mat = make_points(n_dim, n_train);
    std::mt19937_64 rng(kDataSeed);
    const arma::vec f_train_vec = draw_uniform_vec(rng, n_train, 1.5, 2.5);
    const arma::vec hp_gamma_vec = make_hp_gamma_vec(n_dim);
    const arma::mat X_ndim_npost_mat = 0.4 + 0.3 * make_points(n_dim, n_post);

    const gpflash::TrainingData train_data{X_ndim_ntrain_mat, f_train_vec, /*sig2_f_val=*/0.02};
    gpflash::HpVal hp_val_struct;
    hp_val_struct.hp_gamma_vec = hp_gamma_vec;
    hp_val_struct.hp_kern_extra_vec = make_hp_kern_extra_vec(kernel_id, n_dim);
    hp_val_struct.hp_sig2_k = 1.7;
    gpflash::ModelOptions model_opts;
    model_opts.kernel_id = kernel_id;
    model_opts.cond_max = 1.0e6;
    // Two flipped dimensions and one preserved, so a sign applied to the wrong set cannot pass.
    model_opts.symmetry_spec.mirror_sign_vec = arma::vec{1.0, -1.0, -1.0};
    model_opts.symmetry_spec.is_even_symmetry = is_even_symmetry;
    // Even: 1, x_0^2 on the preserved dimension and x_2^2 on a flipped one. Odd: x_1 and x_2, with no constant ahead.
    if (is_even_symmetry) {
        model_opts.mean_model_degree_vec = arma::uvec{2, 0, 2};
    } else {
        model_opts.has_mean_model_constant = false;
        model_opts.mean_model_degree_vec = arma::uvec{0, 1, 1};
    }
    const double parity_sign = is_even_symmetry ? 1.0 : -1.0;
    // Several periods across these points curve the periodic posterior enough that the objective step's truncation
    // error exceeds the tolerance, and Matern52's does too. A central difference's error goes as the step squared, so
    // a tenth of the step clears it.
    const bool is_periodic = kernel_id == gpflash::kernel::KernelId::Periodic;
    const bool uses_short_fd_step = is_periodic || kernel_id == gpflash::kernel::KernelId::Matern52;
    const double fd_step = uses_short_fd_step ? 0.1 * kFdStepObj : kFdStepObj;
    const double tol_fd = is_periodic ? gpflash_test::kTolFdObjOsc : kTolFdObj;
    const auto posterior = gpflash::make_posterior(train_data, hp_val_struct, model_opts);

    gpflash::PosteriorOptions grad_opts;
    grad_opts.return_mean_gradient = true;
    grad_opts.return_var_gradient = true;
    const auto batch_result = posterior.eval(X_ndim_npost_mat, grad_opts);

    gpflash::PosteriorOptions val_opts;
    val_opts.return_mean = true;
    val_opts.return_var = true;
    const auto eval_val_at = [&](const arma::vec& x_at_vec) { return posterior.eval(x_at_vec, val_opts); };

    double worst_mean_grad = 0.0, worst_var_grad = 0.0, worst_single_grad = 0.0;
    for (int i_post = 0; i_post < n_post; ++i_post) {
        const arma::vec x_post_vec = X_ndim_npost_mat.col(i_post);
        const auto single_result = posterior.eval(x_post_vec, grad_opts);

        for (int i_dim = 0; i_dim < n_dim; ++i_dim) {
            arma::vec x_plus_vec = x_post_vec, x_minus_vec = x_post_vec;
            x_plus_vec(i_dim) += fd_step;
            x_minus_vec(i_dim) -= fd_step;
            const auto result_plus = eval_val_at(x_plus_vec);
            const auto result_minus = eval_val_at(x_minus_vec);
            const double d_mean_fd = (result_plus.mean_post_vec(0) - result_minus.mean_post_vec(0)) / (2.0 * fd_step);
            const double d_var_fd = (result_plus.var_post_vec(0) - result_minus.var_post_vec(0)) / (2.0 * fd_step);

            const double d_mean = batch_result.d_mean_post_d_x_post_ndim_npost_mat(i_dim, i_post);
            const double d_var = batch_result.d_var_post_d_x_post_ndim_npost_mat(i_dim, i_post);
            worst_mean_grad = std::max(
                    worst_mean_grad, std::fabs(d_mean - d_mean_fd) / std::max(std::fabs(d_mean_fd), kFdScaleFloor));
            worst_var_grad = std::max(
                    worst_var_grad, std::fabs(d_var - d_var_fd) / std::max(std::fabs(d_var_fd), kFdScaleFloor));

            // The single-query path is a separate implementation, so it must agree with the batch one.
            worst_single_grad = std::max(
                    worst_single_grad, std::fabs(single_result.d_mean_post_d_x_post_ndim_npost_mat(i_dim, 0) - d_mean));
            worst_single_grad = std::max(
                    worst_single_grad, std::fabs(single_result.d_var_post_d_x_post_ndim_npost_mat(i_dim, 0) - d_var));
        }
    }

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "parity %+.0f  mean grad %.2e  var grad %.2e  single %.2e",
            parity_sign,
            worst_mean_grad,
            worst_var_grad,
            worst_single_grad);

    return report(name, worst_mean_grad < tol_fd && worst_var_grad < tol_fd && worst_single_grad < kTolAltPath, detail);
}

// A fit runs end to end with a mirror set, and the hyperparameters it returns reproduce the marginal log
// likelihood it reports.
//
// The optimizer reads only the MLL and its gradient, both checked above, so this is the wiring rather than the
// mathematics: the fit must carry the mirror all the way from the options into every evaluation. The
// self-consistency check is what catches it being dropped on one path but not the other.
bool test_fit_end_to_end(const char* name) {
    const int n_dim = 2, n_train = 40;
    arma::mat X_ndim_ntrain_mat = make_points(n_dim, n_train);
    // A function that is even about the second axis, which is the mirror the fit is given.
    arma::vec f_train_vec(n_train);
    for (int i_train = 0; i_train < n_train; ++i_train)
        f_train_vec(i_train) = std::cos(1.3 * X_ndim_ntrain_mat(0, i_train))
                + X_ndim_ntrain_mat(1, i_train) * X_ndim_ntrain_mat(1, i_train);

    const gpflash::TrainingData train_data{X_ndim_ntrain_mat, f_train_vec, /*sig2_f_val=*/0.0};
    gpflash::OptzOptions optz_opts;
    optz_opts.model_options.kernel_id = gpflash::kernel::KernelId::Matern52;
    optz_opts.model_options.symmetry_spec.mirror_sign_vec = arma::vec{1.0, -1.0};
    optz_opts.model_options.symmetry_spec.is_even_symmetry = true;  // matching the data above

    const gpflash::MllOptzResult mll_optz_result = gpflash::select_hp_with_max_mll(train_data, optz_opts);

    // The fit's own MLL against a fresh evaluation at the hyperparameters it chose: the same model, reached twice.
    const auto mll_result =
            gpflash::eval_mll(train_data, mll_optz_result.hp_val_struct, optz_opts.model_options, false);
    const double mll_err =
            std::fabs(mll_result.mll - mll_optz_result.mll()) / std::max(std::fabs(mll_optz_result.mll()), 1.0);
    const bool hp_finite = mll_optz_result.hp_val_struct.hp_gamma_vec.is_finite()
            && std::isfinite(mll_optz_result.hp_val_struct.hp_sig2_k);

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "mll %.6g  re-eval rel err %.2e  hp_finite %d",
            mll_optz_result.mll(),
            mll_err,
            static_cast<int>(hp_finite));

    return report(name, mll_err < kTolAltPath && hp_finite, detail);
}

// A base kernel whose own diagonal varies is not a mirror's to wrap.
bool test_rejects_unsupported_base_kernel(const char* name) {
    const int n_dim = 2;
    const arma::mat X_ndim_ntrain_mat = make_points(n_dim, 5);
    gpflash::SymmetrySpec symmetry_spec;
    symmetry_spec.mirror_sign_vec = arma::vec{1.0, -1.0};
    const auto check_applies = [&](const gpflash::kernel::KernelId kernel_id) {
        const gpflash::kernel::KernelBase& kern_fun = gpflash::kernel::kernel_by_id(kernel_id);
        gpflash::detail::check_symmetry_applies(
                symmetry_spec, X_ndim_ntrain_mat, kern_fun.is_symmetrizable(), kern_fun.kernel_name(), "test");
    };
    bool accepts_stationary = true;
    std::string refused_kernel_name = "none";  // the first one refused
    for (const gpflash::kernel::KernelId kernel_id : kSymmetrizableKernelIds) {
        const bool is_kernel_accepted = gpflash_test::accepts([&] { check_applies(kernel_id); });
        if (!is_kernel_accepted && accepts_stationary)
            refused_kernel_name = gpflash::kernel::kernel_by_id(kernel_id).kernel_name();
        accepts_stationary = accepts_stationary && is_kernel_accepted;
    }
    const bool rejects_linear = gpflash_test::rejects([&] { check_applies(gpflash::kernel::KernelId::Linear); });
    const bool rejects_quadratic = gpflash_test::rejects([&] { check_applies(gpflash::kernel::KernelId::Quadratic); });

    // A wrong-length mirror cannot be matched against hp_gamma any later than this.
    gpflash::SymmetrySpec wrong_len_spec;
    wrong_len_spec.mirror_sign_vec = arma::vec{1.0, -1.0, -1.0};
    const bool rejects_wrong_len = gpflash_test::rejects([&] {
        gpflash::detail::check_symmetry_applies(
                wrong_len_spec, X_ndim_ntrain_mat, /*is_symmetrizable=*/true, "sq_exp", "test");
    });

    // The fit evaluates its objective unchecked, so it must run the same checks before searching.
    const gpflash::TrainingData train_data{
            X_ndim_ntrain_mat, arma::vec(X_ndim_ntrain_mat.n_cols, arma::fill::ones), 0.0};
    gpflash::OptzOptions optz_opts;
    optz_opts.model_options.kernel_id = gpflash::kernel::KernelId::Linear;
    optz_opts.model_options.symmetry_spec = symmetry_spec;
    const bool fit_rejects_linear =
            gpflash_test::rejects([&] { gpflash::select_hp_with_max_mll(train_data, optz_opts); });
    optz_opts.model_options.kernel_id = gpflash::kernel::KernelId::SqExp;
    optz_opts.model_options.symmetry_spec = wrong_len_spec;
    const bool fit_rejects_wrong_len =
            gpflash_test::rejects([&] { gpflash::select_hp_with_max_mll(train_data, optz_opts); });

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "stationary %d (refused %s)  linear %d  quadratic %d  wrong_len %d  fit %d/%d",
            static_cast<int>(accepts_stationary),
            refused_kernel_name.c_str(),
            static_cast<int>(rejects_linear),
            static_cast<int>(rejects_quadratic),
            static_cast<int>(rejects_wrong_len),
            static_cast<int>(fit_rejects_linear),
            static_cast<int>(fit_rejects_wrong_len));
    return report(
            name,
            accepts_stationary && rejects_linear && rejects_quadratic && rejects_wrong_len && fit_rejects_linear
                    && fit_rejects_wrong_len,
            detail);
}

// Asserts an odd model rejects a training point on, or within rounding of, the mirror plane, where it is zero by
// construction, and accepts one just outside that tolerance. An even model has no such point.
bool test_rejects_training_point_on_mirror_plane(const char* name) {
    const int n_dim = 2;
    const arma::mat X_ndim_ntrain_mat = make_points(n_dim, 7);
    const auto with_flipped_coord = [&](const double x_flipped) {
        arma::mat X_moved_ndim_ntrain_mat = X_ndim_ntrain_mat;
        X_moved_ndim_ntrain_mat(1, 3) = x_flipped;
        return X_moved_ndim_ntrain_mat;
    };
    // Relative to the flipped dimension's largest |x|, which make_points keeps near 1: the rejected offset sits
    // below sqrt(eps) of it and the accepted one well above.
    const arma::mat X_on_plane_mat = with_flipped_coord(0.0);
    const arma::mat X_near_plane_mat = with_flipped_coord(1e-10);
    const arma::mat X_off_plane_mat = with_flipped_coord(1e-6);

    gpflash::SymmetrySpec odd_spec;
    odd_spec.mirror_sign_vec = arma::vec{1.0, -1.0};
    odd_spec.is_even_symmetry = false;
    gpflash::SymmetrySpec even_spec = odd_spec;
    even_spec.is_even_symmetry = true;
    const auto check_applies = [](const gpflash::SymmetrySpec& symmetry_spec,
                                  const arma::mat& X_moved_ndim_ntrain_mat) {
        gpflash::detail::check_symmetry_applies(
                symmetry_spec, X_moved_ndim_ntrain_mat, /*is_symmetrizable=*/true, "sq_exp", "test");
    };

    const bool rejects_on = gpflash_test::rejects([&] { check_applies(odd_spec, X_on_plane_mat); });
    const bool rejects_near = gpflash_test::rejects([&] { check_applies(odd_spec, X_near_plane_mat); });
    const bool accepts_off = gpflash_test::accepts([&] { check_applies(odd_spec, X_off_plane_mat); });
    const bool even_accepts_on = gpflash_test::accepts([&] { check_applies(even_spec, X_on_plane_mat); });

    // The fit reaches the same check.
    const gpflash::TrainingData train_data{X_on_plane_mat, arma::vec(X_on_plane_mat.n_cols, arma::fill::ones), 0.0};
    gpflash::OptzOptions optz_opts;
    optz_opts.model_options.kernel_id = gpflash::kernel::KernelId::SqExp;
    optz_opts.model_options.symmetry_spec = odd_spec;
    optz_opts.model_options.has_mean_model_constant = false;
    const bool fit_rejects_on = gpflash_test::rejects([&] { gpflash::select_hp_with_max_mll(train_data, optz_opts); });

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "on %d  near %d  off %d  even %d  fit %d",
            static_cast<int>(rejects_on),
            static_cast<int>(rejects_near),
            static_cast<int>(accepts_off),
            static_cast<int>(even_accepts_on),
            static_cast<int>(fit_rejects_on));
    return report(name, rejects_on && rejects_near && accepts_off && even_accepts_on && fit_rejects_on, detail);
}

// Asserts make_mean_model accepts a mean model basis whose every column has the model's parity about the mirror plane,
// and rejects any other. Dimension 0 is preserved and dimension 1 flipped.
bool test_rejects_mean_model_the_parity_does_not_admit(const char* name) {
    const int n_dim = 2;
    const auto make_at =
            [&](const bool is_even_symmetry, const bool has_mean_model_constant, const arma::uvec& degree_vec) {
                gpflash::ModelOptions model_opts;
                model_opts.has_mean_model_constant = has_mean_model_constant;
                model_opts.mean_model_degree_vec = degree_vec;
                model_opts.symmetry_spec.mirror_sign_vec = arma::vec{1.0, -1.0};
                model_opts.symmetry_spec.is_even_symmetry = is_even_symmetry;
                gpflash::mean::make_mean_model(model_opts, n_dim, "test");
            };
    const auto rejects_at =
            [&](const bool is_even_symmetry, const bool has_mean_model_constant, const arma::uvec& degree_vec) {
                return gpflash_test::rejects([&] { make_at(is_even_symmetry, has_mean_model_constant, degree_vec); });
            };
    const auto accepts_at =
            [&](const bool is_even_symmetry, const bool has_mean_model_constant, const arma::uvec& degree_vec) {
                return gpflash_test::accepts([&] { make_at(is_even_symmetry, has_mean_model_constant, degree_vec); });
            };

    // Odd: x_1 alone fits; the constant, x_1^2 and any power of x_0 are even.
    const bool odd_case_held[] = {
            accepts_at(/*is_even_symmetry=*/false, /*has_mean_model_constant=*/false, arma::uvec{0}),
            accepts_at(/*is_even_symmetry=*/false, /*has_mean_model_constant=*/false, arma::uvec{0, 1}),
            rejects_at(/*is_even_symmetry=*/false, /*has_mean_model_constant=*/true, arma::uvec{0}),
            rejects_at(/*is_even_symmetry=*/false, /*has_mean_model_constant=*/true, arma::uvec{0, 1}),
            rejects_at(/*is_even_symmetry=*/false, /*has_mean_model_constant=*/false, arma::uvec{0, 2}),
            rejects_at(/*is_even_symmetry=*/false, /*has_mean_model_constant=*/false, arma::uvec{1, 0})};
    // Even: the constant, any power of x_0 and x_1^2 fit; x_1 is odd.
    const bool even_case_held[] = {
            accepts_at(/*is_even_symmetry=*/true, /*has_mean_model_constant=*/true, arma::uvec{1, 0}),
            accepts_at(/*is_even_symmetry=*/true, /*has_mean_model_constant=*/true, arma::uvec{0, 2}),
            accepts_at(/*is_even_symmetry=*/true, /*has_mean_model_constant=*/false, arma::uvec{1, 2}),
            rejects_at(/*is_even_symmetry=*/true, /*has_mean_model_constant=*/true, arma::uvec{0, 1})};
    const bool odd_ok = std::find(std::begin(odd_case_held), std::end(odd_case_held), false) == std::end(odd_case_held);
    const bool even_ok =
            std::find(std::begin(even_case_held), std::end(even_case_held), false) == std::end(even_case_held);

    // One digit per case, in the order listed above.
    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "odd %d%d%d%d%d%d  even %d%d%d%d",
            static_cast<int>(odd_case_held[0]),
            static_cast<int>(odd_case_held[1]),
            static_cast<int>(odd_case_held[2]),
            static_cast<int>(odd_case_held[3]),
            static_cast<int>(odd_case_held[4]),
            static_cast<int>(odd_case_held[5]),
            static_cast<int>(even_case_held[0]),
            static_cast<int>(even_case_held[1]),
            static_cast<int>(even_case_held[2]),
            static_cast<int>(even_case_held[3]));
    return report(name, odd_ok && even_ok, detail);
}

// Asserts an odd model's assembly raises a numerical failure, not a wrong matrix, where k_sym(x, x) rounds to zero.
// The input check keeps this from the public entry points, so the assembly is called directly.
bool test_kmat_diag_rounding_to_zero_fails_numerically(const char* name) {
    const int n_dim = 2;
    arma::mat X_ndim_ntrain_mat = make_points(n_dim, 5);
    X_ndim_ntrain_mat(1, 2) = 0.0;
    const arma::vec hp_gamma_vec = make_hp_gamma_vec(n_dim);
    const arma::mat Ydot_nfeature_ntrain_mat = X_ndim_ntrain_mat.each_col() % hp_gamma_vec;
    gpflash::SymmetrySpec symmetry_spec;
    symmetry_spec.mirror_sign_vec = arma::vec{1.0, -1.0};
    symmetry_spec.is_even_symmetry = false;

    bool raised_numerical_failure = false;
    try {
        arma::vec ydot_train_sq_norm_vec;
        gpflash::detail::build_symmetry_kmat_lower_tri(
                gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::SqExp),
                Ydot_nfeature_ntrain_mat,
                symmetry_spec,
                nullptr,
                ydot_train_sq_norm_vec,
                nullptr);
    } catch (const gpflash::ErrorNumericalFailure&) {
        raised_numerical_failure = true;
    } catch (...) {
    }

    return report(name, raised_numerical_failure);
}

}  // namespace

int main() {
    std::printf("== GpFlash symmetry kernel ==\n");
    bool ok = true;
    ok &= test_kmat_vs_closed_form("kmat_vs_closed_form/even", /*is_even_symmetry=*/true, /*n_train=*/11);
    ok &= test_kmat_vs_closed_form("kmat_vs_closed_form/odd", /*is_even_symmetry=*/false, /*n_train=*/11);
    // Past PERF_OMP_MIN_N_POINT, so the column sweep runs threaded and each thread's own mirrored-run slice is
    // exercised. Below it the build is serial and a slice error cannot show.
    ok &= test_kmat_vs_closed_form("kmat_vs_closed_form/threaded", /*is_even_symmetry=*/true, /*n_train=*/300);
    ok &= test_sq_norms_are_the_points("sq_norms_are_the_points");
    for (const gpflash::kernel::KernelId kernel_id : kSymmetrizableKernelIds) {
        const std::string kernel_name = gpflash::kernel::kernel_by_id(kernel_id).kernel_name();
        for (const bool is_even_symmetry : {true, false}) {
            const std::string parity_name = is_even_symmetry ? "/even" : "/odd";
            for (const double sig2_f_val : {0.0, 0.031}) {
                const std::string noise_name = (sig2_f_val > 0.0) ? "_noisy" : "";
                const std::string mll_name = "mll_hp_grad_vs_fd/" + kernel_name + parity_name + noise_name;
                ok &= test_mll_hp_grad_vs_fd(mll_name.c_str(), kernel_id, is_even_symmetry, sig2_f_val);
            }
            const std::string post_name = "posterior_vs_dense_reference/" + kernel_name + parity_name;
            ok &= test_posterior_vs_dense_reference(post_name.c_str(), kernel_id, is_even_symmetry);
            const std::string grad_name = "posterior_grad_vs_fd/" + kernel_name + parity_name;
            ok &= test_posterior_grad_vs_fd(grad_name.c_str(), kernel_id, is_even_symmetry);
            const std::string cache_name = "cache_round_trip/" + kernel_name + parity_name;
            ok &= test_cache_round_trip(cache_name.c_str(), kernel_id, is_even_symmetry);
        }
    }
    ok &= test_fit_end_to_end("fit_end_to_end");
    ok &= test_rejects_degenerate_spec("rejects_degenerate_spec");
    ok &= test_rejects_unsupported_base_kernel("rejects_unsupported_base_kernel");
    ok &= test_rejects_training_point_on_mirror_plane("rejects_training_point_on_mirror_plane");
    ok &= test_kmat_diag_rounding_to_zero_fails_numerically("kmat_diag_rounding_to_zero_fails_numerically");
    ok &= test_rejects_mean_model_the_parity_does_not_admit("rejects_mean_model_the_parity_does_not_admit");
    std::printf("\n%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");

    return ok ? 0 : 1;
}
