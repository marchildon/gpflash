// The periodic kernel through select_hp_with_max_mll: log-space gradient against a finite difference, the fitted
// region carrying Periodic::calc_optz_seeding's start and sampling box, a fit at least as good as the true
// hyperparameters in 1D, determinism, and a valid posterior.
//
// Separate from test_gpflash_optimizer, which fits only radial kernels.
#include <armadillo>
#include <chrono>
#include <cstdio>
#include <random>
#include <string>

#include "GpFlashTestSupport.hpp"
#include "gpflash/kernels/KernelDispatch.hpp"         // kernel::KernelId::Periodic
#include "gpflash/optimization/HpOptimizer.hpp"       // select_hp_with_max_mll
#include "gpflash/optimization/MllOptzObjective.hpp"  // detail::MllOptzObjective
#include "gpflash/posterior_build/PosteriorBuild.hpp"
#include "gpflash/posterior_eval/GpPosterior.hpp"

using gpflash_test::draw_normal;
using gpflash_test::draw_uniform;
using gpflash_test::kFdStepObjOsc;
using gpflash_test::kRelScaleFloor;
using gpflash_test::kTolFdObjOsc;

using namespace gpflash;

// Slack on "the fit is at least as good as the truth": the optimizer stops on its own tolerances, so it
// may land a hair above the objective at the true hyperparameters without being wrong.
constexpr double kMllSlack = 1e-6;

// The periodic gradient is checked at the true hyperparameters, where hp_gamma is better conditioned
// than the amplitudes and the variances, so the two groups do not reach the same floor.
constexpr double kTolPeriodicFitGamma = 1e-5;

constexpr double kTolPeriodicFitOther = 1e-4;

// Local, not the shared objective-level pair: these cases are truncation-limited and so reach a tighter
// floor than the roundoff-limited noise-free objective those constants are sized for.
constexpr double kFdStepFit = 1e-6;

// Draw f_train ~ N(3, Sigma_mat) from a periodic GP:
// K_ij = exp(-2 sum_d hp_amplitude_d^2 sin^2(pi hp_gamma_d r_d)).
static arma::vec sample_gp_periodic(
        const arma::mat& X_ndim_ntrain_mat,
        const arma::vec& hp_gamma_vec,
        const arma::vec& hp_amplitude_vec,
        double hp_sig2_k,
        double sig2_f_val,
        std::mt19937_64& rng) {
    const int n_train = static_cast<int>(X_ndim_ntrain_mat.n_cols);
    const int n_dim = static_cast<int>(X_ndim_ntrain_mat.n_rows);
    arma::mat K_mat(n_train, n_train);
    double max_abs_row_sum = 0.0;
    for (int i_row = 0; i_row < n_train; ++i_row) {
        double row_sum = 0.0;
        for (int i_col = 0; i_col < n_train; ++i_col) {
            if (i_row == i_col) {
                K_mat(i_row, i_row) = 1.0;
                continue;
            }
            double kern_sum = 0.0;
            for (int i_dim = 0; i_dim < n_dim; ++i_dim) {
                const double sin_sum = std::sin(
                        arma::datum::pi * hp_gamma_vec(i_dim)
                        * (X_ndim_ntrain_mat(i_dim, i_row) - X_ndim_ntrain_mat(i_dim, i_col)));
                kern_sum += hp_amplitude_vec(i_dim) * hp_amplitude_vec(i_dim) * sin_sum * sin_sum;
            }
            const double kern_val = std::exp(-2.0 * kern_sum);
            K_mat(i_row, i_col) = kern_val;
            row_sum += std::abs(kern_val);
        }
        max_abs_row_sum = std::max(max_abs_row_sum, row_sum);
    }
    const double eta = max_abs_row_sum / (ModelOptions{}.cond_max - 1.0);
    arma::mat Sigma_mat = hp_sig2_k * K_mat;
    Sigma_mat.diag() += hp_sig2_k * eta + sig2_f_val;
    arma::mat L_Sigma_mat = arma::chol(Sigma_mat, "lower");
    arma::vec z_vec(n_train);
    for (int i_train = 0; i_train < n_train; ++i_train) z_vec(i_train) = draw_normal(rng);

    return L_Sigma_mat * z_vec + 3.0;
}

// Periodic test: every analytic gradient (hp_gamma from the raw inputs, hp_amplitude and the variances from the
// feature ones) against a finite difference, plus a deterministic fit and posterior. check_global is
// asserted only in 1D; the ARD periodic objective is strongly multimodal in the joint inverse-period
// space, which is a property of the kernel rather than of this machinery.
static bool run_periodic_case(
        const std::string& name,
        const arma::vec& hp_gamma_true_vec,
        const arma::vec& hp_amplitude_true_vec,
        double hp_sig2_k_true,
        double sig2_f_true,
        int n_ind_optz,
        bool check_global) {
    std::mt19937_64 rng(101);
    const int n_train = 120, n_dim = static_cast<int>(hp_gamma_true_vec.n_elem);
    arma::mat X_ndim_ntrain_mat(n_dim, n_train);
    {
        for (arma::uword i_elem = 0; i_elem < X_ndim_ntrain_mat.n_elem; ++i_elem)
            X_ndim_ntrain_mat(i_elem) = draw_uniform(rng, -3.0, 3.0);  // several periods span the data
    }
    arma::vec f_train_vec = sample_gp_periodic(
            X_ndim_ntrain_mat, hp_gamma_true_vec, hp_amplitude_true_vec, hp_sig2_k_true, sig2_f_true, rng);
    TrainingData train_data;
    train_data.X_ndim_ntrain_mat = X_ndim_ntrain_mat;
    train_data.f_train_vec = f_train_vec;

    OptzOptions optz_opts;
    optz_opts.model_options.kernel_id = gpflash::kernel::KernelId::Periodic;
    optz_opts.use_multistart = true;
    optz_opts.n_ind_optz = n_ind_optz;
    optz_opts.hypercube_sampling_seed = 5;
    // Exercise the periodic path with both variances fitted (the most general case): a
    // negative sig2_f_val is what asks for the noise to be estimated.
    train_data.sig2_f_val = -1.0;
    optz_opts.scale_variance_bounds_by_residual_var = false;
    optz_opts.hp_sig2_k_init = hp_sig2_k_true;
    // Non-unit, so the amplitude band's centre at log(hp_kern_extra_init) is not 0.
    optz_opts.hp_kern_extra_init = 1.2;

    // (1) Objective-gradient check on the shared MllOptzObjective at the periodic layout: every gradient
    // (hp_gamma, the extras block, sigma_k^2, sigma_f^2) analytic, vs a full finite difference.
    const detail::MllOptzObjective optz_objective(train_data, optz_opts, "test");
    arma::vec ln_hp_init_vec(optz_objective.layout().n_hp_optz());
    for (int i_dim = 0; i_dim < n_dim; ++i_dim) {
        ln_hp_init_vec(i_dim) = std::log(hp_gamma_true_vec(i_dim));
        ln_hp_init_vec(n_dim + i_dim) = std::log(hp_amplitude_true_vec(i_dim));
    }
    ln_hp_init_vec(optz_objective.layout().offset(HpBlock::Sig2K)) = std::log(hp_sig2_k_true);
    ln_hp_init_vec(optz_objective.layout().offset(HpBlock::Sig2FVal)) = std::log(sig2_f_true);
    arma::vec grad_analytic_vec;
    optz_objective.calc_objective(ln_hp_init_vec, &grad_analytic_vec);
    double gamma_err = 0.0, other_err = 0.0;
    // Step local, not kFdStepObjOsc: at the true hyperparameters the difference is truncation-limited, so the
    // larger shared step would degrade the reference.
    const double fd_step = kFdStepFit;
    for (int i_hp_optz = 0; i_hp_optz < optz_objective.layout().n_hp_optz(); ++i_hp_optz) {
        arma::vec ln_hp_plus_vec = ln_hp_init_vec, ln_hp_minus_vec = ln_hp_init_vec;
        ln_hp_plus_vec(i_hp_optz) += fd_step;
        ln_hp_minus_vec(i_hp_optz) -= fd_step;
        const double fd_deriv = (optz_objective.calc_objective(ln_hp_plus_vec, nullptr)
                                 - optz_objective.calc_objective(ln_hp_minus_vec, nullptr))
                / (2.0 * fd_step);
        if (i_hp_optz < n_dim)
            gamma_err = std::max(gamma_err, std::abs(fd_deriv - grad_analytic_vec(i_hp_optz)));  // hp_gamma (analytic)
        else
            other_err = std::max(
                    other_err,
                    std::abs(fd_deriv - grad_analytic_vec(i_hp_optz)));  // hp_amplitude + variances (analytic)
    }

    // (2) fit, (3) the fit's MLL against the truth's, (4) determinism.
    MllOptzResult mll_optz_result = select_hp_with_max_mll(train_data, optz_opts);
    MllOptzResult mll_optz_result_repeat = select_hp_with_max_mll(train_data, optz_opts);
    const double mll_true = -optz_objective.calc_objective(ln_hp_init_vec, nullptr);
    const double determinism_diff = std::abs(mll_optz_result.mll() - mll_optz_result_repeat.mll())
            + arma::norm(mll_optz_result.hp_val_struct.hp_gamma_vec
                         - mll_optz_result_repeat.hp_val_struct.hp_gamma_vec);

    // (5) posterior with the fitted hyperparameters. The periodic posterior takes raw points and raw
    // query points, applying the feature map itself, so this is the same call any other kernel makes.
    // A default ModelOptions would fall back to Matern52, so the periodic kernel is named explicitly.
    ModelOptions post_model_opts;
    post_model_opts.kernel_id = gpflash::kernel::KernelId::Periodic;
    auto posterior = make_posterior(train_data, mll_optz_result.hp_val_struct, post_model_opts);
    const arma::vec x_post_vec = X_ndim_ntrain_mat.col(0);  // named, so the vector overload is unambiguous
    const double var_post = posterior.eval(x_post_vec).var_post_vec(0);

    const bool global_ok = !check_global || (mll_optz_result.mll() >= mll_true - kMllSlack);

    // (6) The fit's region carries the kernel's seeding, against a reference from the data range. hp_gamma starts
    // at one period across each dimension's range and samples upward from there by one factor shared by every
    // dimension; the amplitudes sample a band centred on hp_kern_extra_init, narrower than their constraint box.
    const OptzRegion& optz_region = mll_optz_result.optz_region;
    const HpOptzLayout& hp_optz_layout = mll_optz_result.hp_optz_layout();
    const arma::uword n_dim_u = static_cast<arma::uword>(n_dim);
    arma::vec ln_hp_gamma_start_ref_vec(n_dim_u);
    for (arma::uword i_dim = 0; i_dim < n_dim_u; ++i_dim)
        ln_hp_gamma_start_ref_vec(i_dim) = -std::log(arma::range(X_ndim_ntrain_mat.row(i_dim)));
    const arma::vec gamma_sample_width_vec =
            optz_region.sample_ub_vec.head(n_dim_u) - optz_region.sample_lb_vec.head(n_dim_u);
    const bool gamma_seeded =
            arma::norm(optz_region.start_vec.head(n_dim_u) - ln_hp_gamma_start_ref_vec, "inf") < gpflash_test::kTolExact
            && arma::norm(optz_region.sample_lb_vec.head(n_dim_u) - ln_hp_gamma_start_ref_vec, "inf")
                    < gpflash_test::kTolExact
            && gamma_sample_width_vec.min() > 0.0
            && gamma_sample_width_vec.max() - gamma_sample_width_vec.min() < gpflash_test::kTolExact;
    const arma::uword idx_extra = static_cast<arma::uword>(hp_optz_layout.offset(HpBlock::KernExtra));
    const arma::vec extra_sample_lb_vec = optz_region.sample_lb_vec.subvec(idx_extra, idx_extra + n_dim_u - 1);
    const arma::vec extra_sample_ub_vec = optz_region.sample_ub_vec.subvec(idx_extra, idx_extra + n_dim_u - 1);
    const arma::vec extra_constraint_width_vec =
            optz_region.constraint_ub_vec.subvec(idx_extra, idx_extra + n_dim_u - 1)
            - optz_region.constraint_lb_vec.subvec(idx_extra, idx_extra + n_dim_u - 1);
    const arma::vec extra_band_centre_vec = 0.5 * (extra_sample_lb_vec + extra_sample_ub_vec);
    const bool amplitude_seeded =
            arma::norm(extra_band_centre_vec - std::log(optz_opts.hp_kern_extra_init), "inf") < gpflash_test::kTolExact
            && arma::all((extra_sample_ub_vec - extra_sample_lb_vec) > 0.0)
            && arma::all((extra_sample_ub_vec - extra_sample_lb_vec) < extra_constraint_width_vec);
    const bool seeding_ok = gamma_seeded && amplitude_seeded;

    const bool pass = gamma_err < kTolPeriodicFitGamma && other_err < kTolPeriodicFitOther && global_ok && seeding_ok
            && determinism_diff == 0.0 && var_post > 0.0
            && gpflash::detail::is_all_finite(
                              mll_optz_result.hp_val_struct.hp_gamma_vec.memptr(),
                              mll_optz_result.hp_val_struct.hp_gamma_vec.n_elem)
            && gpflash::detail::is_finite_val(mll_optz_result.hp_val_struct.hp_sig2_k);
    std::printf(
            "%-14s g_err=%.1e a_err=%.1e  seeding=%d  mll(fit=%.2f, true=%.2f)  hp_sig2_k=%.2f noise=%.4f  "
            "var_post=%.3f  determ=%.0e  %s\n",
            name.c_str(),
            gamma_err,
            other_err,
            static_cast<int>(seeding_ok),
            mll_optz_result.mll(),
            mll_true,
            mll_optz_result.hp_val_struct.hp_sig2_k,
            mll_optz_result.hp_val_struct.hp_sig2_f_val,
            var_post,
            determinism_diff,
            pass ? "PASS" : "*** FAIL ***");

    return pass;
}

// The periodic counterpart of run_exact_gradient_case in test_gpflash_optimizer. Two choices matter for
// this to measure anything: low frequencies, so the oscillatory periodic objective admits an accurate
// finite difference, and a point away from the optimum, so the gradient magnitude is not near zero (at
// the optimum the relative comparison is meaningless).
static bool run_exact_gradient_periodic_case() {
    std::mt19937_64 rng(11);
    const int n_dim = 2, n_train = 70;
    arma::mat X_ndim_ntrain_mat(n_dim, n_train);
    for (arma::uword i_elem = 0; i_elem < X_ndim_ntrain_mat.n_elem; ++i_elem)
        X_ndim_ntrain_mat(i_elem) = draw_uniform(rng, -1.0, 1.0);
    const arma::vec hp_gamma_true_vec = {0.22, 0.31}, hp_amplitude_true_vec = {0.5, 0.65};
    arma::vec f_train_vec = sample_gp_periodic(
            X_ndim_ntrain_mat, hp_gamma_true_vec, hp_amplitude_true_vec, /*hp_sig2_k=*/2.1, /*sig2_f_val=*/0.0, rng);
    TrainingData train_data;
    train_data.X_ndim_ntrain_mat = X_ndim_ntrain_mat;
    train_data.f_train_vec = f_train_vec;

    // Off-truth evaluation point.
    const arma::vec hp_gamma_at_vec = {0.35, 0.19}, hp_amplitude_at_vec = {0.8, 0.42};

    const auto calc_rel_err = [&]() {
        OptzOptions optz_opts;  // train_data.sig2_f_val is 0: noise-free
        optz_opts.model_options.kernel_id = gpflash::kernel::KernelId::Periodic;

        const detail::MllOptzObjective optz_objective(train_data, optz_opts, "test");

        arma::vec ln_hp_init_vec(2 * n_dim), grad_analytic_vec;
        for (int i_dim = 0; i_dim < n_dim; ++i_dim) {
            ln_hp_init_vec(i_dim) = std::log(hp_gamma_at_vec(i_dim));
            ln_hp_init_vec(n_dim + i_dim) = std::log(hp_amplitude_at_vec(i_dim));
        }
        optz_objective.calc_objective(ln_hp_init_vec, &grad_analytic_vec);
        const double fd_step = kFdStepObjOsc;  // roundoff-limited, so a larger fd_step is the better reference
        double grad_err = 0.0, grad_mag = 0.0;
        for (int i_hp_optz = 0; i_hp_optz < 2 * n_dim; ++i_hp_optz) {
            arma::vec ln_hp_plus_vec = ln_hp_init_vec, ln_hp_minus_vec = ln_hp_init_vec;
            ln_hp_plus_vec(i_hp_optz) += fd_step;
            ln_hp_minus_vec(i_hp_optz) -= fd_step;
            const double fd_deriv = (optz_objective.calc_objective(ln_hp_plus_vec, nullptr)
                                     - optz_objective.calc_objective(ln_hp_minus_vec, nullptr))
                    / (2.0 * fd_step);
            grad_err = std::max(grad_err, std::abs(fd_deriv - grad_analytic_vec(i_hp_optz)));
            grad_mag = std::max(grad_mag, std::abs(fd_deriv));
        }
        return grad_err / std::max(grad_mag, kRelScaleFloor);
    };

    // The nugget's eta terms are load-bearing in this analytic gradient: dropping them leaves an
    // fd_step-independent error far above the tolerance, so this case pins that contribution.
    const double rel_err = calc_rel_err();
    const bool pass = rel_err < kTolFdObjOsc;
    std::printf(
            "%-16s rel_err(analytic vs FD)=%.2e (tol 1e-05)  %s\n",
            "NuggetGradPeriodic",
            rel_err,
            pass ? "PASS" : "*** FAIL ***");

    return pass;
}

// Times each case, so a slow test suite points at the case responsible rather than needing to be
// bisected by hand. The fits dominate: each is n_ind_optz local solves, each of which is up to
// max_evals Cholesky factorizations of an (n_train, n_train) matrix.
template <class Function>
static bool timed(const char* label, Function&& function) {
    const auto t_start = std::chrono::steady_clock::now();
    const bool pass = function();
    const double elapsed_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    std::printf("%-22s %6.2f s\n", label, elapsed_s);

    return pass;
}

int main() {
    bool ok = true;
    ok &= timed("run_exact_gradient_periodic_case", [] { return run_exact_gradient_periodic_case(); });
    ok &= timed("Periodic1D", [] {
        return run_periodic_case(
                "Periodic1D",
                {1.15},
                {0.95},
                /*hp_sig2_k_true=*/1.5,
                /*sig2_f_true=*/0.03,
                /*n_ind_optz=*/12,
                /*check_global=*/true);
    });
    // Distinct and non-unit, but kept near the 1D case's scale: the periodic kernel is
    // oscillatory, so raising the inverse periods increases the finite-difference
    // truncation error of the gradient check without testing anything new.
    ok &= timed("Periodic2D", [] {
        return run_periodic_case(
                "Periodic2D",
                {0.92, 1.08},
                {0.95, 1.25},
                /*hp_sig2_k_true=*/1.5,
                /*sig2_f_true=*/0.03,
                /*n_ind_optz=*/16,
                /*check_global=*/false);
    });
    std::printf("\n%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");

    return ok ? 0 : 1;
}
