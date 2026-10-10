// Tests select_hp_with_max_mll end to end: every analytic log-space gradient against a finite difference of the
// objective, a fit at least as good as the known GP's own hyperparameters, optimality, determinism (the same fit run
// twice agrees exactly), a valid positive-variance posterior from the fit, the two posterior-from-fit entry points
// including the save/load round trip, a fit with the {1} mean model, the unbounded fit and its multistart rule,
// variance bounds scaled by the mean model's residual including the zero-residual fallback, and the MllOptzResult
// postcondition. Covers SqExp, Matern 5/2 and rational quadratic.
//
// The tier's units are tested on their own: test_gpflash_local_optz, test_gpflash_multistart,
// test_gpflash_search_space and test_gpflash_multistart_report.
#include <armadillo>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>

#include "GpFlashTestSupport.hpp"
#include "gpflash/kernels/KernelDispatch.hpp"  // kernel_by_id
#include "gpflash/mean/MeanModel.hpp"  // make_mean_model, build_train_basis_mat, calc_least_squares_residual_var
#include "gpflash/optimization/HpOptimizer.hpp"
#include "gpflash/optimization/MllOptzObjective.hpp"             // MllOptzObjective
#include "gpflash/optimization/multistart/MultistartResult.hpp"  // MultistartResult, StartField
#include "gpflash/optimization/multistart/RunMultistart.hpp"     // IndOptzOutcome, fill_ind_optz_column_inplace
#include "gpflash/optimization/search_space/HpOptzLayout.hpp"    // HpOptzLayout, HpBlock, build_hp_optz_layout
#include "gpflash/optimization/search_space/OptzOptions.hpp"     // OptzOptions
#include "gpflash/optimization/search_space/OptzRegion.hpp"      // calc_variance_scale
#include "gpflash/posterior_build/PosteriorBuild.hpp"
#include "gpflash/posterior_eval/GpPosterior.hpp"  // GpPosterior, load_posterior
#include "gpflash/training/EvalMllDispatch.hpp"    // detail::eval_mll, the checked form this test compares against

using gpflash_test::accepts;
using gpflash_test::draw_normal;
using gpflash_test::kDetailLen;
using gpflash_test::kFdStepObj;
using gpflash_test::kHpRhoUnused;
using gpflash_test::kRelScaleFloor;
using gpflash_test::kTolFdObj;
using gpflash_test::rejects;
using gpflash_test::report;

using namespace gpflash;

// ── Thresholds specific to fitting, each used by more than one case below ────
// Slack on "the fit is at least as good as the truth": the optimizer stops on its own tolerances, so it may land
// a hair above the objective at the true hyperparameters without being wrong.
constexpr double kMllSlack = 1e-6;

// The returned point must be first-order optimal relative to the objective's own scale.
constexpr double kRelOptimality = 1.0e-3;

// run_case defaults, local because these cases are truncation-limited and so reach a tighter floor than the
// roundoff-limited noise-free objective the shared constants are sized for.
constexpr double kTolFitGradRel = 1e-7;

constexpr double kFdStepFit = 1e-6;

// run_case's data and start points. A sweep of these criteria over data seeds 101-105 by hypercube seeds 5-8 (20
// pairs, every run_case case, libc++/Accelerate and libstdc++/OpenBLAS) passed on all 20. In it, hypercube seed 5
// puts one start in the corner of tiny length-scale inverses and tiny noise, where the backend fails at SqExp's and
// Matern 5/2's estimated-noise fits on every data seed (one failed start, recovered from), and RatQuad's shape
// finishes on its upper bound for data seeds 101-103 (fit MLL above the truth each time). This pair keeps both
// paths exercised.
constexpr unsigned kRunCaseDataSeed = 101;
constexpr unsigned kRunCaseHypercubeSeed = 5;

// A multistart start may fail numerically (the factorization or the backend gives up at a hostile start point);
// the fit records it and stands on the others. Allowed only for the estimated-noise rows that can lose one at a
// corner start (SqExp/Estimated, Matern52, RatQuad); every other row keeps zero. More than one in run_case's six would
// no longer be an occasional bad start but a robustness problem worth failing on.
constexpr int kMaxFailedStart = 1;
// The trend's residual is about 1e-5 of f_train, so the least-squares fit's cancellation costs about five digits.
constexpr double kTolResidualVar = 1e-8;

// Largest analytic-vs-finite-difference gradient error at ln_hp_vec, relative to the gradient magnitude. Relative
// because the MLL gradient scales with the data and the conditioning, so an absolute threshold is not portable
// across the noise models -- the noise-free case being the ill-conditioned one.
static double calc_grad_rel_err(
        const detail::MllOptzObjective& optz_objective, const arma::vec& ln_hp_vec, const double fd_step) {
    arma::vec grad_analytic_vec;
    optz_objective.calc_objective(ln_hp_vec, &grad_analytic_vec);

    double grad_err = 0.0;
    double grad_mag = 0.0;
    for (int i_hp_optz = 0; i_hp_optz < optz_objective.layout().n_hp_optz(); ++i_hp_optz) {
        arma::vec ln_hp_plus_vec = ln_hp_vec, ln_hp_minus_vec = ln_hp_vec;
        ln_hp_plus_vec(i_hp_optz) += fd_step;
        ln_hp_minus_vec(i_hp_optz) -= fd_step;
        const double fd_deriv = (optz_objective.calc_objective(ln_hp_plus_vec, nullptr)
                                 - optz_objective.calc_objective(ln_hp_minus_vec, nullptr))
                / (2.0 * fd_step);
        grad_err = std::max(grad_err, std::abs(fd_deriv - grad_analytic_vec(i_hp_optz)));
        grad_mag = std::max(grad_mag, std::abs(fd_deriv));
    }

    return grad_err / std::max(grad_mag, kRelScaleFloor);
}

// Draw f_train ~ N(3, Sigma_mat) with Sigma_mat = hp_sig2_k (K(hp_gamma) + eta I) + noise I for kernel K.
static arma::vec sample_gp(
        const gpflash::kernel::KernelBase& kern_fun,
        const arma::mat& X_ndim_ntrain_mat,
        const arma::vec& hp_gamma_vec,
        double hp_sig2_k,
        double sig2_f_val,
        double hp_kern_extra,
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
                double rdot = hp_gamma_vec(i_dim) * (X_ndim_ntrain_mat(i_dim, i_row) - X_ndim_ntrain_mat(i_dim, i_col));
                kern_sum += rdot * rdot;
            }
            double kern_val = kern_sum;
            double kern_common_rdot = 0.0;
            kern_fun.calc_kern_val_and_common_rdot_inplace(&kern_val, 1, &hp_kern_extra, &kern_common_rdot);
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

// Whether every failed start was reported and recovered from. Reported: its column says Failed, carries +inf and
// its own start point, and calc_n_failed_start counts it; every other column converged to a finite objective.
// Recovered: the winner is a start that converged, so the fit stands on the others.
static bool is_failed_start_record_ok(const MllOptzResult& mll_optz_result) {
    const MultistartResult& multistart_result = mll_optz_result.multistart_result;
    int n_failed_seen = 0;
    bool record_ok = true;
    for (int i_ind_optz = 0; i_ind_optz < multistart_result.n_ind_optz_run(); ++i_ind_optz) {
        const double objective_val = multistart_result.start_field_val(StartField::ObjectiveVal, i_ind_optz);
        const bool is_failed = static_cast<OptzStopReason>(static_cast<int>(
                                       multistart_result.start_field_val(StartField::StopReason, i_ind_optz)))
                == OptzStopReason::Failed;
        if (is_failed) {
            ++n_failed_seen;
            record_ok =
                    record_ok && std::isinf(objective_val) && objective_val > 0.0
                    && arma::all(
                            multistart_result.ln_hp_vec(i_ind_optz) == multistart_result.ln_hp_init_vec(i_ind_optz));
        } else {
            record_ok = record_ok && gpflash::detail::is_finite_val(objective_val);
        }
    }
    const int i_best = mll_optz_result.idx_best();
    const bool best_converged = static_cast<OptzStopReason>(static_cast<int>(
                                        multistart_result.start_field_val(StartField::StopReason, i_best)))
            != OptzStopReason::Failed;

    return record_ok && n_failed_seen == mll_optz_result.calc_n_failed_start() && best_converged;
}

// Takes the kernel as an id, not a reference: the fit reads it from optz_opts, so a reference argument could
// disagree with the kernel actually fitted.
static bool run_case(
        const gpflash::kernel::KernelId kernel_id,
        const std::string& name,
        const arma::vec& hp_gamma_ref_vec,
        double hp_sig2_k_ref,
        double sig2_f_ref,
        double hp_kern_extra_ref,
        // The noise request, sign-encoded as TrainingData::sig2_f_val: 0 noise-free,
        // > 0 known, < 0 estimate it.
        double sig2_f_request = -1.0,
        double grad_rel_tol = kTolFitGradRel,
        // Suits the truncation-limited noisy models; the noise-free model is roundoff-limited and passes
        // kFdStepObj.
        double fd_step = kFdStepFit,
        // Whether the kernel extra may finish on its bound. Only where the extra has a limit the model approaches
        // smoothly, so the bound is where the search box ends rather than a defect; the fit must still be at least as
        // good as the truth.
        const bool allow_kern_extra_at_bound = false,
        // How many of the six starts may fail and be recovered from: kMaxFailedStart only where a row needs it.
        const int max_failed_start = 0) {
    const gpflash::kernel::KernelBase& kern_fun = gpflash::kernel::kernel_by_id(kernel_id);
    std::mt19937_64 rng(kRunCaseDataSeed);
    const int n_train = 80;
    const int n_dim = static_cast<int>(hp_gamma_ref_vec.n_elem);
    arma::mat X_ndim_ntrain_mat(n_dim, n_train);
    {
        for (arma::uword i_elem = 0; i_elem < X_ndim_ntrain_mat.n_elem; ++i_elem)
            X_ndim_ntrain_mat(i_elem) = draw_normal(rng);
    }
    arma::vec f_train_vec =
            sample_gp(kern_fun, X_ndim_ntrain_mat, hp_gamma_ref_vec, hp_sig2_k_ref, sig2_f_ref, hp_kern_extra_ref, rng);
    TrainingData train_data;
    train_data.X_ndim_ntrain_mat = X_ndim_ntrain_mat;
    train_data.f_train_vec = f_train_vec;

    OptzOptions optz_opts;
    optz_opts.model_options.kernel_id = kernel_id;
    optz_opts.use_multistart = true;
    optz_opts.n_ind_optz = 6;
    optz_opts.hypercube_sampling_seed = kRunCaseHypercubeSeed;
    train_data.sig2_f_val = sig2_f_request;
    const bool noise_free = (sig2_f_request == 0.0);
    const bool estimate_noise = (sig2_f_request < 0.0);
    // Absolute variance bounds and inits, so ln_hp_init_vec below is exactly the point the analytic gradient
    // is checked at, independent of the residual variance.
    optz_opts.scale_variance_bounds_by_residual_var = false;
    optz_opts.hp_sig2_k_init = hp_sig2_k_ref;

    // (1) The real objective's gradient against a full finite difference. Every gradient is analytic: hp_gamma,
    // the kernel extra hp_kern_extra, and whichever variances sig2_f_request activates. Built through the same calls
    // select_hp_with_max_mll makes, so the layout under test is the one a fit produces.
    const int n_hp_kern_extra = kern_fun.n_hp_kern_extra(n_dim);
    const detail::MllOptzObjective optz_objective(train_data, optz_opts, "test");
    const HpOptzLayout& hp_optz_layout = optz_objective.layout();
    arma::vec ln_hp_init_vec(hp_optz_layout.n_hp_optz(), arma::fill::zeros);
    // hp_gamma at the reference rather than 1, which would hide a missing gamma_l chain-rule factor and leave every
    // dimension alike. Noise-free stays at 1: its truth sits near the optimum, where the relative error is noise.
    if (sig2_f_request != 0.0) ln_hp_init_vec.head(n_dim) = arma::log(hp_gamma_ref_vec);
    if (hp_optz_layout.is_searched(HpBlock::Sig2K))
        ln_hp_init_vec(hp_optz_layout.offset(HpBlock::Sig2K)) = std::log(hp_sig2_k_ref);
    if (hp_optz_layout.is_searched(HpBlock::Sig2FVal))
        ln_hp_init_vec(hp_optz_layout.offset(HpBlock::Sig2FVal)) = std::log(std::max(sig2_f_ref, 1e-6));
    // 1.7 rather than 1: a unit shape hides a scaling error in the chain rule.
    for (int i_hp_kern_extra = 0; i_hp_kern_extra < n_hp_kern_extra; ++i_hp_kern_extra)
        ln_hp_init_vec(hp_optz_layout.offset(HpBlock::KernExtra) + i_hp_kern_extra) = std::log(1.7);
    const double grad_rel_err = calc_grad_rel_err(optz_objective, ln_hp_init_vec, fd_step);

    // (2) fit + (3) optimality + (4) determinism.
    MllOptzResult mll_optz_result = select_hp_with_max_mll(train_data, optz_opts);
    MllOptzResult mll_optz_result_repeat = select_hp_with_max_mll(train_data, optz_opts);
    arma::vec hp_kern_extra_ref_vec(kern_fun.n_hp_kern_extra(n_dim));
    if (!hp_kern_extra_ref_vec.empty()) hp_kern_extra_ref_vec(0) = hp_kern_extra_ref;
    // Truth reference in the GpFlash paper's absolute parametrization. sig2_f_val alone decides where
    // sigma_k^2 comes from: 0 takes the closed form, non-zero the supplied value.
    detail::EvalVariances eval_variances;
    if (!noise_free) {
        eval_variances.sig2_f_val = sig2_f_ref;
        eval_variances.hp_sig2_k = hp_sig2_k_ref;
    }
    gpflash::ModelOptions model_opts_ref;
    model_opts_ref.cond_max = gpflash::COND_MAX_DEFAULT;
    double mll_ref = detail::eval_mll(
                             kern_fun,
                             gpflash::TrainingData{X_ndim_ntrain_mat, f_train_vec, eval_variances.sig2_f_val},
                             detail::build_hp_val(hp_gamma_ref_vec, hp_kern_extra_ref_vec, eval_variances.hp_sig2_k),
                             model_opts_ref,
                             /*eval_grad=*/false)
                             .mll;
    // Determinism, not numerical reproducibility: the same fit on the same data in the same process, twice.
    // What it catches is hidden state, uninitialised memory, and dependence on allocation addresses -- not
    // agreement with any other build. A flag that changes the arithmetic still passes, deterministically.
    const double determinism_diff = std::abs(mll_optz_result.mll() - mll_optz_result_repeat.mll())
            + arma::norm(mll_optz_result.hp_val_struct.hp_gamma_vec
                         - mll_optz_result_repeat.hp_val_struct.hp_gamma_vec);

    // (5) posterior from the same data struct the fit read: the noise travels with it, and the
    // estimate (when there was one) travels in the hyperparameters.
    const double sig2_f_fit = estimate_noise ? mll_optz_result.hp_val_struct.hp_sig2_f_val : sig2_f_request;
    auto posterior = make_posterior(train_data, mll_optz_result.hp_val_struct, mll_optz_result.model_options);
    const arma::vec x_post_vec = X_ndim_ntrain_mat.col(0);
    const double var_post = posterior.eval(x_post_vec).var_post_vec(0);

    // (6) MllOptzResult diagnostics. optimality is bounded relative to |mll| because the MLL gradient
    // scales with the data and the conditioning. A zero gradient count would mean the analytic gradient never
    // reached the optimizer.
    // n_eval == n_mll_eval + n_mll_grad_eval: every backend evaluation goes through the counted objective.
    const int n_eval_total = mll_optz_result.n_total_mll_eval() + mll_optz_result.n_total_mll_grad_eval();
    const LocalOptzResult best_local_optz_result = mll_optz_result.build_best_local_optz_result();
    int n_eval_backend_total = 0;  // summed by hand, so the fit-level total is checked against every start
    bool tally_matches_backend = true;
    for (int i_ind_optz = 0; i_ind_optz < mll_optz_result.n_ind_optz_run(); ++i_ind_optz) {
        const MultistartResult& multistart_result = mll_optz_result.multistart_result;
        const int n_eval_start = multistart_result.start_field_count(StartField::NEval, i_ind_optz);
        const int n_mll_total_start = multistart_result.start_field_count(StartField::NMllEval, i_ind_optz)
                + multistart_result.start_field_count(StartField::NMllGradEval, i_ind_optz);
        tally_matches_backend = tally_matches_backend && n_eval_start == n_mll_total_start;
        n_eval_backend_total += n_eval_start;
    }
    // Failed starts: each reported and recovered from, and at most max_failed_start of them.
    const bool failed_start_ok =
            is_failed_start_record_ok(mll_optz_result) && mll_optz_result.calc_n_failed_start() <= max_failed_start;
    // At the optimum, optimality is the projected gradient norm, which drops the components pushing against an
    // active bound, so "small projected gradient or an active bound" is one check. Which bounds may be active: none,
    // these cases' true parameters all lying well inside the box, except a kernel extra the caller allows, and then
    // only alongside the global "at least as good as the truth" below.
    const HpAtBound hp_at_bound_struct = mll_optz_result.calc_hp_at_bound_struct();
    HpAtBound hp_at_bound_disallowed_struct = hp_at_bound_struct;
    if (allow_kern_extra_at_bound) hp_at_bound_disallowed_struct.hp_kern_extra_at_bound_vec.zeros();
    const bool diag_ok = best_local_optz_result.stop_reason == OptzStopReason::ToleranceMet
            && mll_optz_result.n_total_mll_grad_eval() > 0 && failed_start_ok && tally_matches_backend
            && n_eval_total == n_eval_backend_total
            && n_eval_total <= optz_opts.max_evals * std::max(1, mll_optz_result.n_ind_optz_run())
            && gpflash::detail::is_finite_val(mll_optz_result.optimality)
            && mll_optz_result.optimality < kRelOptimality * std::max(1.0, std::abs(mll_optz_result.mll()))
            && hp_at_bound_struct.hp_gamma_at_bound_vec.n_elem == static_cast<arma::uword>(n_dim)
            && !hp_at_bound_disallowed_struct.has_any();

    const bool pass = grad_rel_err < grad_rel_tol && mll_optz_result.mll() >= mll_ref - kMllSlack
            && determinism_diff == 0.0 && var_post > 0.0
            && gpflash::detail::is_all_finite(
                              mll_optz_result.hp_val_struct.hp_gamma_vec.memptr(),
                              mll_optz_result.hp_val_struct.hp_gamma_vec.n_elem)
            && gpflash::detail::is_finite_val(mll_optz_result.hp_val_struct.hp_sig2_k) && diag_ok;
    std::printf(
            "%-16s grad_rel=%.1e  mll(fit=%.2f, true=%.2f)  hp_sig2_k=%.2f noise=%.4f  var_post=%.3f  "
            "determ=%.0e  "
            "opt=%.1e ev=%d+%d bnd=%d  diag=%d stop=%s failed=%d tally=%d  %s\n",
            name.c_str(),
            grad_rel_err,
            mll_optz_result.mll(),
            mll_ref,
            mll_optz_result.hp_val_struct.hp_sig2_k,
            sig2_f_fit,
            var_post,
            determinism_diff,
            mll_optz_result.optimality,
            mll_optz_result.n_total_mll_eval(),
            mll_optz_result.n_total_mll_grad_eval(),
            static_cast<int>(mll_optz_result.calc_hp_at_bound_struct().has_any()),
            static_cast<int>(diag_ok),
            to_stop_reason_name(best_local_optz_result.stop_reason),
            mll_optz_result.calc_n_failed_start(),
            static_cast<int>(tally_matches_backend),
            pass ? "PASS" : "*** FAIL ***");

    return pass;
}

// An unconstrained fit needs no bounds, because the heuristic start comes from the data and the init options, so
// use_bounds = false with one optimization must run. Multistart with it cannot: the initial points are drawn from
// inside the box, so every optimization would repeat the same start at n_ind_optz times the cost, and it is
// rejected. Both halves are asserted here, since the pair is the rule.
static bool run_unbounded_case() {
    std::mt19937_64 rng(31);
    const int n_dim = 2;
    const int n_train = 40;
    const arma::vec hp_gamma_ref_vec = {1.3, 0.7};
    arma::mat X_ndim_ntrain_mat(n_dim, n_train);
    {
        for (arma::uword i_elem = 0; i_elem < X_ndim_ntrain_mat.n_elem; ++i_elem)
            X_ndim_ntrain_mat(i_elem) = draw_normal(rng);
    }
    const arma::vec f_train_vec = sample_gp(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::SqExp),
            X_ndim_ntrain_mat,
            hp_gamma_ref_vec,
            /*hp_sig2_k=*/1.5,
            /*sig2_f_val=*/1e-2,
            /*hp_kern_extra=*/0.0,
            rng);

    TrainingData train_data;
    train_data.X_ndim_ntrain_mat = X_ndim_ntrain_mat;
    train_data.f_train_vec = f_train_vec;
    train_data.sig2_f_val = 1e-2;

    // (1) Unbounded, one optimization: runs, and reaches a finite optimum from the data-driven start.
    OptzOptions optz_opts;
    optz_opts.model_options.kernel_id = gpflash::kernel::KernelId::SqExp;  // the kernel f_train was drawn from
    optz_opts.use_bounds = false;                                          // every bound becomes +/-inf

    bool threw = false;
    MllOptzResult mll_optz_result;
    try {
        mll_optz_result = select_hp_with_max_mll(train_data, optz_opts);
    } catch (const std::exception&) {
        threw = true;
    }
    const bool single_ok = !threw
            && mll_optz_result.build_best_local_optz_result().stop_reason == OptzStopReason::ToleranceMet
            && mll_optz_result.n_ind_optz_run() == 1
            && gpflash::detail::is_all_finite(
                    mll_optz_result.hp_val_struct.hp_gamma_vec.memptr(),
                    mll_optz_result.hp_val_struct.hp_gamma_vec.n_elem)
            && gpflash::detail::is_finite_val(mll_optz_result.mll())
            && arma::all(mll_optz_result.optz_region.constraint_lb_vec == -arma::datum::inf)
            && arma::all(mll_optz_result.optz_region.constraint_ub_vec == arma::datum::inf);

    // (2) Unbounded plus multistart: rejected at the entry point, before any factorization.
    OptzOptions optz_opts_multistart = optz_opts;
    optz_opts_multistart.use_multistart = true;
    optz_opts_multistart.n_ind_optz = 3;
    const bool multistart_rejected = rejects([&] { select_hp_with_max_mll(train_data, optz_opts_multistart); });

    // (3) The same multistart with bounds, which the error message directs the caller to, so it has to work.
    OptzOptions optz_opts_bounded = optz_opts_multistart;
    optz_opts_bounded.use_bounds = true;
    const bool bounded_ok = accepts([&] { select_hp_with_max_mll(train_data, optz_opts_bounded); });

    const bool pass = single_ok && multistart_rejected && bounded_ok;
    std::printf(
            "%-14s single=%d multistart_rejected=%d bounded=%d  %s\n",
            "Unbounded",
            static_cast<int>(single_ok),
            static_cast<int>(multistart_rejected),
            static_cast<int>(bounded_ok),
            pass ? "PASS" : "*** FAIL ***");

    return pass;
}

// A constant f_train leaves the constant mean model no residual variance. The variance bounds are multiples of it
// and are log-transformed, so a scale of zero gives log(0) = -inf bounds and NaN reaches the gradient several layers
// away. Asserts the fallback to an absolute scale, that it is a fallback -- a real variance still scales -- and that
// the fit runs.
static bool run_constant_f_train_case() {
    const int n_dim = 2;
    const int n_train = 8;

    TrainingData train_data;
    train_data.X_ndim_ntrain_mat = gpflash_test::make_points(n_dim, n_train);
    train_data.f_train_vec = arma::vec(n_train, arma::fill::ones) * 3.5;  // constant
    train_data.sig2_f_val = -1.0;                                         // estimate the noise

    OptzOptions optz_opts;
    optz_opts.model_options.kernel_id = gpflash::kernel::KernelId::SqExp;

    const double scale_at_zero_var = gpflash::detail::calc_variance_scale(0.0, optz_opts);
    const double real_var = 0.75;
    const double scale_at_real_var = gpflash::detail::calc_variance_scale(real_var, optz_opts);

    const bool falls_back = scale_at_zero_var == 1.0;
    const bool still_scales = scale_at_real_var == real_var;

    bool threw = false;
    MllOptzResult mll_optz_result;
    try {
        mll_optz_result = select_hp_with_max_mll(train_data, optz_opts);
    } catch (const std::exception&) {
        threw = true;
    }
    const bool fit_ran = !threw && gpflash::detail::is_finite_val(mll_optz_result.mll());

    const bool pass = falls_back && still_scales && fit_ran;
    std::printf(
            "%-14s scale(var=0)=%.3g scale(var>0)=%.3g  threw=%d mll=%.6g  %s\n",
            "ConstantF",
            scale_at_zero_var,
            scale_at_real_var,
            static_cast<int>(threw),
            mll_optz_result.mll(),
            pass ? "PASS" : "*** FAIL ***");

    return pass;
}

// The hardest setting for the gradient: a noise-free fit, where eta is the only regularization, so
// Sigma_mat^-1 ~ 1/eta and any error in the eta terms is amplified.
static bool run_exact_gradient_case() {
    std::mt19937_64 rng(101);
    const int n_dim = 3;
    const int n_train = 90;
    arma::mat X_ndim_ntrain_mat(n_dim, n_train);
    for (arma::uword i_elem = 0; i_elem < X_ndim_ntrain_mat.n_elem; ++i_elem)
        X_ndim_ntrain_mat(i_elem) = draw_normal(rng);
    const arma::vec hp_gamma_ref_vec = {1.4, 0.6, 0.9};
    const double hp_sig2_k_ref = 2.3;
    arma::vec f_train_vec = sample_gp(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::SqExp),
            X_ndim_ntrain_mat,
            hp_gamma_ref_vec,
            hp_sig2_k_ref,
            /*sig2_f_val=*/0.0,
            /*hp_kern_extra=*/0.0,
            rng);
    TrainingData train_data;
    train_data.X_ndim_ntrain_mat = X_ndim_ntrain_mat;
    train_data.f_train_vec = f_train_vec;

    const auto calc_rel_err = [&]() {
        OptzOptions optz_opts;  // train_data.sig2_f_val is 0: noise-free, which the objective reads off the data
        optz_opts.model_options.kernel_id = gpflash::kernel::KernelId::SqExp;  // the kernel f_train was drawn from

        const detail::MllOptzObjective optz_objective(train_data, optz_opts, "test");

        arma::vec ln_hp_init_vec(n_dim);
        for (int i_dim = 0; i_dim < n_dim; ++i_dim) ln_hp_init_vec(i_dim) = std::log(hp_gamma_ref_vec(i_dim));

        // Roundoff-dominated: the error grows as fd_step shrinks, so the larger kFdStepObj is the more accurate
        // reference here.
        return calc_grad_rel_err(optz_objective, ln_hp_init_vec, kFdStepObj);
    };

    const double rel_err = calc_rel_err();
    // kTolFdObj is the finite-difference floor for this problem. Omitting the eta terms from the gradient
    // raises the error well above it, which is what this tolerance detects.
    const bool pass = rel_err < kTolFdObj;
    std::printf(
            "%-16s rel_err(analytic vs FD)=%.2e (tol %.0e)  %s\n",
            "NuggetGradient",
            rel_err,
            kTolFdObj,
            pass ? "PASS" : "*** FAIL ***");
    return pass;
}

// The two posterior-from-fit entry points, the only posterior calls that read a fit result:
//   1) the fit echoes the kernel it fitted, which is what makes (2) safe
//   2) make_posterior_from_fit equals unpacking the result by hand, exactly
//   3) fit_and_make_posterior gives a usable posterior
//   4) saving that posterior writes a file that reads back to the same one
static bool run_posterior_from_fit_case() {
    const std::string cache_path = "test_optimizer_fit_and_make_posterior.h5";
    std::remove(cache_path.c_str());  // a previous run may have left one

    std::mt19937_64 rng(7);
    const arma::vec hp_gamma_ref_vec = {1.4, 0.7};
    arma::mat X_ndim_ntrain_mat(2, 60);
    {
        for (arma::uword i_elem = 0; i_elem < X_ndim_ntrain_mat.n_elem; ++i_elem)
            X_ndim_ntrain_mat(i_elem) = draw_normal(rng);
    }
    TrainingData train_data;
    train_data.X_ndim_ntrain_mat = X_ndim_ntrain_mat;
    train_data.f_train_vec = sample_gp(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::SqExp),
            X_ndim_ntrain_mat,
            hp_gamma_ref_vec,
            /*hp_sig2_k=*/1.5,
            /*sig2_f_val=*/0.04,
            kHpRhoUnused,
            rng);
    train_data.sig2_f_val = 0.04;  // known noise: the cheapest layout, and the cache seam is what is under test

    // One optimization: the cache seam is what is under test, not the search, so the cheapest fit will do.
    OptzOptions optz_opts;
    optz_opts.model_options.kernel_id = gpflash::kernel::KernelId::SqExp;

    const arma::vec x_post_vec = X_ndim_ntrain_mat.col(0);  // named, so the vector overload is unambiguous

    // (1) + (2)
    const MllOptzResult mll_optz_result = select_hp_with_max_mll(train_data, optz_opts);
    const bool kernel_echoed = mll_optz_result.model_options.kernel_id == gpflash::kernel::KernelId::SqExp;
    const double mean_from_result =
            make_posterior_from_fit(train_data, mll_optz_result).eval(x_post_vec).mean_post_vec(0);
    const double mean_unpacked =
            make_posterior(train_data, mll_optz_result.hp_val_struct, mll_optz_result.model_options)
                    .eval(x_post_vec)
                    .mean_post_vec(0);
    const bool forwards_faithfully = mean_from_result == mean_unpacked;

    // (3)
    const gpflash::GpPosterior fitted = fit_and_make_posterior(train_data, optz_opts);
    const gpflash::PosteriorResult post_result_written = fitted.eval(x_post_vec);
    const bool fit_ok = gpflash::detail::is_finite_val(post_result_written.mean_post_vec(0))
            && post_result_written.var_post_vec(0) > 0.0;

    // (4) Reading the file back is what fails if the write is wrong; a second fit is seeded, so it would agree
    // whether or not the file was written. What a file records and refuses is test_gpflash_cache's remit.
    fitted.save_precompute(cache_path);
    const bool file_exists = std::ifstream(cache_path).good();
    bool round_trip_ok = false;
    if (file_exists) {  // guarded so a missing file reports a failure rather than throwing out of the case
        const gpflash::PosteriorResult post_result_loaded = load_posterior(cache_path).eval(x_post_vec);
        round_trip_ok = post_result_written.mean_post_vec(0) == post_result_loaded.mean_post_vec(0)
                && post_result_written.var_post_vec(0) == post_result_loaded.var_post_vec(0);
    }
    std::remove(cache_path.c_str());

    const bool pass = kernel_echoed && forwards_faithfully && fit_ok && round_trip_ok;
    std::printf(
            "%-16s echo=%d forward=%d fit=%d round_trip=%d  %s\n",
            "PosteriorFromFit",  // the case name, kept stable for log comparisons
            kernel_echoed,
            forwards_faithfully,
            fit_ok,
            round_trip_ok,
            pass ? "PASS" : "*** FAIL ***");

    return pass;
}

// The variance options scale with what the mean model leaves in f_train, not with var(f_train):
//   1) the residual variance matches hand-computed values for a strong trend, a zero mean model's offset, and a
//      basis with as many columns as points but dependent ones
//   2) fits on such data reach an hp_sig2_k that var(f_train)-scaled bounds exclude, and leave it inside the bounds
static bool run_residual_variance_scale_case() {
    const auto calc_residual_var_at = [](const bool has_mean_model_constant,
                                         const arma::uword mean_model_degree,
                                         const arma::mat& X_ndim_ntrain_mat,
                                         const arma::vec& f_train_vec) {
        ModelOptions model_options;
        model_options.has_mean_model_constant = has_mean_model_constant;
        model_options.mean_model_degree_vec = {mean_model_degree};
        const int n_dim = static_cast<int>(X_ndim_ntrain_mat.n_rows);
        const mean::MeanModel mean_model = mean::make_mean_model(model_options, n_dim, "test");
        const arma::mat Basis_ntrain_nbasis_mat = mean::build_train_basis_mat(mean_model, X_ndim_ntrain_mat, "test");
        return mean::detail::calc_least_squares_residual_var(
                Basis_ntrain_nbasis_mat, f_train_vec, has_mean_model_constant);
    };

    // (1) residual_pattern_vec sums to zero and is orthogonal to x, so the linear fit leaves exactly
    // residual_amplitude residual_pattern_vec, and the zero mean model's cross terms with the offset cancel.
    const arma::mat X_hand_mat = {{-3.0, -1.0, 1.0, 3.0}};
    const arma::vec x_hand_vec = X_hand_mat.row(0).t();
    const arma::vec residual_pattern_vec = {1.0, -1.0, -1.0, 1.0};
    const double residual_amplitude = 1e-3;
    const arma::vec f_trend_vec = 5.0 + 100.0 * x_hand_vec + residual_amplitude * residual_pattern_vec;
    const double offset = 1000.0;
    const arma::vec f_offset_vec = offset + residual_amplitude * residual_pattern_vec;
    // Four squared residuals over 4 - 2 degrees of freedom, and over 4.
    const double trend_residual_var_ref = 4.0 * residual_amplitude * residual_amplitude / 2.0;
    const double offset_residual_var_ref = offset * offset + residual_amplitude * residual_amplitude;
    const double trend_residual_var = calc_residual_var_at(true, 1, X_hand_mat, f_trend_vec);
    const double offset_residual_var = calc_residual_var_at(false, 0, X_hand_mat, f_offset_vec);
    const double trend_err = std::fabs(trend_residual_var - trend_residual_var_ref) / trend_residual_var_ref;
    const double offset_err = std::fabs(offset_residual_var - offset_residual_var_ref) / offset_residual_var_ref;

    // Equal x_1 and x_2 give the basis [1, x, x] rank 2 on three points, leaving one degree of freedom.
    // dependent_f_train_vec is orthogonal to 1 and x, so the fit leaves all of it.
    const arma::mat X_dependent_mat = {{-1.0, 0.0, 1.0}, {-1.0, 0.0, 1.0}};
    const arma::vec dependent_f_train_vec = {1.0, -2.0, 1.0};
    const double dependent_residual_var_ref = 1.0 + 4.0 + 1.0;  // its squares, over one degree of freedom
    const double dependent_residual_var = calc_residual_var_at(true, 1, X_dependent_mat, dependent_f_train_vec);
    const double dependent_err =
            std::fabs(dependent_residual_var - dependent_residual_var_ref) / dependent_residual_var_ref;
    const bool residual_var_ok =
            trend_err < kTolResidualVar && offset_err < kTolResidualVar && dependent_err < kTolResidualVar;

    // (2) The trend's var(f_train) is the trend's, which puts a var(f_train)-scaled hp_sig2_k_lb above the kernel
    // variance of its residual. The offset's var(f_train) excludes the offset, which puts hp_sig2_k_ub below what the
    // kernel must carry. Each fit is compared against the bound var(f_train) would have set.
    std::mt19937_64 rng(23);
    const int n_train = 20;
    TrainingData train_data;
    train_data.X_ndim_ntrain_mat = arma::linspace<arma::rowvec>(0.0, 1.0, n_train);
    train_data.sig2_f_val = -1.0;  // estimate the noise, so hp_sig2_k is searched
    const arma::vec x_train_vec = train_data.X_ndim_ntrain_mat.row(0).t();
    arma::vec noise_vec(n_train);
    for (double& noise_val : noise_vec) noise_val = 1e-4 * draw_normal(rng);
    const arma::vec smooth_vec = arma::sin(6.0 * x_train_vec);
    OptzOptions optz_options;
    optz_options.model_options.kernel_id = gpflash::kernel::KernelId::SqExp;

    train_data.f_train_vec = 5.0 + 100.0 * x_train_vec + 1e-3 * smooth_vec + noise_vec;
    optz_options.model_options.mean_model_degree_vec = {1};
    const double trend_var_f_train_lb = optz_options.hp_sig2_k_lb * arma::var(train_data.f_train_vec);
    const MllOptzResult trend_result = select_hp_with_max_mll(train_data, optz_options);
    const double trend_hp_sig2_k = trend_result.hp_val_struct.hp_sig2_k;

    train_data.f_train_vec = offset + smooth_vec + noise_vec;
    optz_options.model_options.has_mean_model_constant = false;
    optz_options.model_options.mean_model_degree_vec = {0};
    const double offset_var_f_train_ub = optz_options.hp_sig2_k_ub * arma::var(train_data.f_train_vec);
    const MllOptzResult offset_result = select_hp_with_max_mll(train_data, optz_options);
    const double offset_hp_sig2_k = offset_result.hp_val_struct.hp_sig2_k;

    const bool outside_var_f_train_bounds =
            trend_hp_sig2_k < trend_var_f_train_lb && offset_hp_sig2_k > offset_var_f_train_ub;
    const bool inside_bounds = !trend_result.calc_hp_at_bound_struct().hp_sig2_k_at_bound
            && !offset_result.calc_hp_at_bound_struct().hp_sig2_k_at_bound;
    const bool pass = residual_var_ok && outside_var_f_train_bounds && inside_bounds;
    std::printf(
            "%-16s trend_err=%.1e offset_err=%.1e dependent_err=%.1e  hp_sig2_k trend=%.2e (var lb %.2e) offset=%.2e "
            "(var ub %.2e) "
            "inside=%d  %s\n",
            "ResidualVarScale",
            trend_err,
            offset_err,
            dependent_err,
            trend_hp_sig2_k,
            trend_var_f_train_lb,
            offset_hp_sig2_k,
            offset_var_f_train_ub,
            static_cast<int>(inside_bounds),
            pass ? "PASS" : "*** FAIL ***");
    return pass;
}
// A fit with the {1} mean model carries it end to end:
//   1) the objective the search maximized is the MLL under the {1} mean model, not the constant mean model
//   2) the result echoes the mean model options, and both posterior-from-fit entry points build its basis
static bool run_mean_model_fit_case() {
    std::mt19937_64 rng(19);
    const int n_dim = 2, n_train = 50;
    arma::mat X_ndim_ntrain_mat(n_dim, n_train);
    for (double& x_val : X_ndim_ntrain_mat) x_val = draw_normal(rng);

    // A GP sample on a prior mean of degree 1 in each dimension, which the {1} mean model spans and the constant one
    // does not.
    const double sig2_f_val = 0.04;
    TrainingData train_data;
    train_data.X_ndim_ntrain_mat = X_ndim_ntrain_mat;
    const arma::vec prior_mean_train_vec =
            3.0 + 2.0 * X_ndim_ntrain_mat.row(0).t() - 1.5 * X_ndim_ntrain_mat.row(1).t();
    train_data.f_train_vec = prior_mean_train_vec
            + sample_gp(gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::SqExp),
                        X_ndim_ntrain_mat,
                        arma::vec{1.2, 0.9},
                        /*hp_sig2_k=*/0.5,
                        sig2_f_val,
                        kHpRhoUnused,
                        rng);
    train_data.sig2_f_val = sig2_f_val;

    const arma::uword mean_model_degree = 1;  // the {1} mean model
    OptzOptions optz_opts;
    optz_opts.model_options.kernel_id = gpflash::kernel::KernelId::SqExp;
    optz_opts.model_options.mean_model_degree_vec = {mean_model_degree};
    const MllOptzResult mll_optz_result = select_hp_with_max_mll(train_data, optz_opts);

    // (1) The winner's MLL, re-evaluated under the fitted {1} mean model and under the constant one.
    const double mll_fitted_mean_model =
            eval_mll(train_data, mll_optz_result.hp_val_struct, optz_opts.model_options).mll;
    ModelOptions model_opts_constant_mean_model = optz_opts.model_options;
    model_opts_constant_mean_model.mean_model_degree_vec = {0};
    const double mll_constant_mean_model =
            eval_mll(train_data, mll_optz_result.hp_val_struct, model_opts_constant_mean_model).mll;
    const double mll_fitted_mean_model_err = std::abs(mll_optz_result.mll() - mll_fitted_mean_model)
            / std::max(std::abs(mll_fitted_mean_model), gpflash_test::kRelScaleFloor);
    const bool objective_used_mean_model =
            mll_fitted_mean_model_err <= gpflash_test::kTolAltPath && mll_constant_mean_model < mll_fitted_mean_model;

    // (2) The mean model options reach the result and both posteriors.
    const bool options_echoed = mll_optz_result.model_options.has_mean_model_constant
            && arma::all(mll_optz_result.model_options.mean_model_degree_vec == arma::uvec{mean_model_degree});
    const int n_basis_ref = 1 + n_dim;  // the constant, and one column per dimension
    const bool from_fit_ok = make_posterior_from_fit(train_data, mll_optz_result).mean_model().n_basis() == n_basis_ref;
    const bool fit_and_make_ok =
            fit_and_make_posterior(train_data, optz_opts).hp_beta_vec().n_elem == static_cast<arma::uword>(n_basis_ref);

    const bool pass = objective_used_mean_model && options_echoed && from_fit_ok && fit_and_make_ok;
    std::printf(
            "%-16s objective=%d echo=%d from_fit=%d fit_and_make=%d  mll fitted=%.4f constant=%.4f  %s\n",
            "MeanModelFit",
            objective_used_mean_model,
            options_echoed,
            from_fit_ok,
            fit_and_make_ok,
            mll_fitted_mean_model,
            mll_constant_mean_model,
            pass ? "PASS" : "*** FAIL ***");

    return pass;
}

// The finished fit's postcondition, which GPFLASH_CHECK compiles out of a release build, so this is the only thing
// that drives its reject paths. The record underneath it is built the way run_multistart fills it.
static bool test_mll_optz_result_checks(const char* name) {
    using gpflash::NoiseRequest;
    using gpflash::detail::build_hp_optz_layout;
    using gpflash::detail::check_mll_optz_result;
    const int n_dim = 2;

    gpflash::LocalOptzResult local_optz_result;
    local_optz_result.ln_hp_vec = arma::vec(n_dim, arma::fill::zeros);
    local_optz_result.objective_val = -3.5;
    local_optz_result.n_eval = 12;
    local_optz_result.stop_reason = gpflash::OptzStopReason::ToleranceMet;

    const gpflash::HpOptzLayout hp_optz_layout = build_hp_optz_layout(n_dim, 0, NoiseRequest::NoiseFree, "test");
    const int n_ind_optz = 2;
    gpflash::MultistartResult multistart_result;
    multistart_result.hp_optz_layout = hp_optz_layout;
    multistart_result.ln_hp_mat.zeros(n_dim, n_ind_optz);
    multistart_result.ln_hp_init_mat.ones(n_dim, n_ind_optz);
    multistart_result.start_field_mat.zeros(static_cast<int>(gpflash::StartField::End), n_ind_optz);
    gpflash::detail::IndOptzOutcome ind_optz_outcome;
    ind_optz_outcome.ln_hp_init_vec = arma::vec(n_dim, arma::fill::ones);
    ind_optz_outcome.local_optz_result = local_optz_result;
    for (int i_ind_optz = 0; i_ind_optz < n_ind_optz; ++i_ind_optz) {
        ind_optz_outcome.local_optz_result.objective_val = -3.5 + static_cast<double>(i_ind_optz);
        gpflash::detail::fill_ind_optz_column_inplace(multistart_result, i_ind_optz, ind_optz_outcome);
    }

    // A finished fit over that record.
    gpflash::MllOptzResult mll_optz_result;
    mll_optz_result.multistart_result = multistart_result;
    mll_optz_result.optimality = 1e-7;
    // A region every recorded point lies inside.
    mll_optz_result.optz_region.constraint_lb_vec = arma::vec(n_dim, arma::fill::ones) * -5.0;
    mll_optz_result.optz_region.constraint_ub_vec = arma::vec(n_dim, arma::fill::ones) * 5.0;
    mll_optz_result.optz_region.sample_lb_vec = mll_optz_result.optz_region.constraint_lb_vec;
    mll_optz_result.optz_region.sample_ub_vec = mll_optz_result.optz_region.constraint_ub_vec;
    mll_optz_result.optz_region.start_vec = arma::vec(n_dim, arma::fill::ones);
    mll_optz_result.hp_val_struct =
            gpflash::detail::build_hp_val(arma::vec(n_dim, arma::fill::ones), arma::vec(), /*hp_sig2_k=*/1.5);

    const bool mll_accepted = accepts([&] { check_mll_optz_result(mll_optz_result, "test"); });
    // Interior point: on no bound.
    const bool on_no_bound = !mll_optz_result.calc_hp_at_bound_struct().has_any()
            && mll_optz_result.calc_n_at_bound(1) == 0
            && mll_optz_result.calc_hp_at_bound_struct().hp_gamma_at_bound_vec.n_elem
                    == static_cast<arma::uword>(n_dim);
    const bool record_rejected = rejects([&] {  // the record's checker composes
        gpflash::MllOptzResult broken = mll_optz_result;
        broken.multistart_result.ln_hp_mat(0, 0) = arma::datum::nan;
        check_mll_optz_result(broken, "test");
    });
    // The search reports a winner or raises, so the negative seed cannot reach a caller.
    const bool optimality_rejected = rejects([&] {
        gpflash::MllOptzResult broken = mll_optz_result;
        broken.optimality = -1.0;
        check_mll_optz_result(broken, "test");
    });
    const bool hp_width_rejected = rejects([&] {  // the hyperparameters must match the layout the fit carries
        gpflash::MllOptzResult broken = mll_optz_result;
        broken.hp_val_struct.hp_gamma_vec = arma::vec(n_dim + 1, arma::fill::ones);
        check_mll_optz_result(broken, "test");
    });
    const bool region_rejected = rejects([&] {  // the region's checker composes
        gpflash::MllOptzResult broken = mll_optz_result;
        broken.optz_region.constraint_ub_vec = arma::vec(n_dim + 1, arma::fill::ones);
        check_mll_optz_result(broken, "test");
    });
    const bool start_outside_rejected = rejects([&] {  // a start outside the constraint box
        gpflash::MllOptzResult broken = mll_optz_result;
        broken.optz_region.start_vec(0) = 10.0;
        check_mll_optz_result(broken, "test");
    });
    // A finished fit always resolves the signal variance, so the unset sentinel is a defect.
    const bool sig2_k_unset_rejected = rejects([&] {
        gpflash::MllOptzResult broken = mll_optz_result;
        broken.hp_val_struct.hp_sig2_k = -1.0;
        check_mll_optz_result(broken, "test");
    });

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "accept=%d no_bound=%d record=%d optimality=%d hp_width=%d region=%d start=%d sig2_k_unset=%d",
            static_cast<int>(mll_accepted),
            static_cast<int>(on_no_bound),
            static_cast<int>(record_rejected),
            static_cast<int>(optimality_rejected),
            static_cast<int>(hp_width_rejected),
            static_cast<int>(region_rejected),
            static_cast<int>(start_outside_rejected),
            static_cast<int>(sig2_k_unset_rejected));
    return report(
            name,
            mll_accepted && on_no_bound && record_rejected && optimality_rejected && hp_width_rejected
                    && region_rejected && start_outside_rejected && sig2_k_unset_rejected,
            detail);
}

// Times each case, so a slow suite points at the case responsible rather than needing to be bisected by hand.
// The fits dominate: each is n_ind_optz local solves, each up to max_evals Cholesky factorizations of an
// (n_train, n_train) matrix.
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
    ok &= timed("run_unbounded_case", [] { return run_unbounded_case(); });
    ok &= timed("run_exact_gradient_case", [] { return run_exact_gradient_case(); });
    ok &= timed("run_constant_f_train_case", [] { return run_constant_f_train_case(); });
    ok &= timed("run_residual_variance_scale_case", [] { return run_residual_variance_scale_case(); });
    // All three noise models, so every analytic variance gradient is checked: sigma_k^2 in closed form,
    // sigma_k^2 optimized, and both optimized. NoiseFree keeps the same tolerance but needs the larger fd_step,
    // which is a measurement constraint rather than an accuracy one.
    ok &= timed("SqExp/NoiseFree", [] {
        return run_case(
                gpflash::kernel::KernelId::SqExp,
                "SqExp/NoiseFree",
                {1.5, 0.5, 1.0},
                /*hp_sig2_k_ref=*/2.0,
                /*sig2_f_ref=*/0.0,
                /*hp_kern_extra_ref=*/0.0,
                /*sig2_f_request=*/0.0,
                /*grad_rel_tol=*/kTolFitGradRel,
                /*fd_step=*/kFdStepObj);
    });
    ok &= timed("SqExp/KnownVar", [] {
        return run_case(
                gpflash::kernel::KernelId::SqExp,
                "SqExp/KnownVar",
                {1.5, 0.5, 1.0},
                /*hp_sig2_k_ref=*/2.0,
                /*sig2_f_ref=*/0.05,
                /*hp_kern_extra_ref=*/0.0,
                /*sig2_f_request=*/0.05);
    });
    ok &= timed("SqExp/Estimated", [] {
        return run_case(
                gpflash::kernel::KernelId::SqExp,
                "SqExp/Estimated",
                {1.5, 0.5, 1.0},
                /*hp_sig2_k_ref=*/2.0,
                /*sig2_f_ref=*/0.05,
                /*hp_kern_extra_ref=*/0.0,
                /*sig2_f_request=*/-1.0,
                /*grad_rel_tol=*/kTolFitGradRel,
                /*fd_step=*/kFdStepFit,
                /*allow_kern_extra_at_bound=*/false,
                // SLSQP gives up at the corner start (ln-noise near -16), and the fit stands on the other five.
                /*max_failed_start=*/kMaxFailedStart);
    });
    ok &= timed("Matern52", [] {
        return run_case(
                gpflash::kernel::KernelId::Matern52,
                "Matern52",
                {1.5, 0.5, 1.0},
                /*hp_sig2_k_ref=*/2.0,
                /*sig2_f_ref=*/0.05,
                /*hp_kern_extra_ref=*/0.0,
                /*sig2_f_request=*/-1.0,
                /*grad_rel_tol=*/kTolFitGradRel,
                /*fd_step=*/kFdStepFit,
                /*allow_kern_extra_at_bound=*/false,
                // The same corner start as SqExp/Estimated.
                /*max_failed_start=*/kMaxFailedStart);
    });
    ok &= timed("RatQuad", [] {
        return run_case(
                gpflash::kernel::KernelId::RatQuad,
                "RatQuad",
                {1.2, 0.8},
                /*hp_sig2_k_ref=*/1.5,
                /*sig2_f_ref=*/0.03,
                /*hp_kern_extra_ref=*/2.0,
                /*sig2_f_request=*/-1.0,
                /*grad_rel_tol=*/kTolFitGradRel,
                /*fd_step=*/kFdStepFit,
                // The shape alpha may finish on its upper bound: as alpha grows the kernel tends smoothly to the
                // squared exponential, so on 80 points the MLL can keep rising slowly towards the box's edge.
                /*allow_kern_extra_at_bound=*/true,
                // MSVC with vcpkg's OpenBLAS fails the same corner start as SqExp/Estimated.
                /*max_failed_start=*/kMaxFailedStart);
    });
    ok &= timed("run_posterior_from_fit_case", [] { return run_posterior_from_fit_case(); });
    ok &= timed("run_mean_model_fit_case", [] { return run_mean_model_fit_case(); });
    ok &= test_mll_optz_result_checks("mll_optz_result_checks");
    std::printf("\n%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");

    return ok ? 0 : 1;
}
