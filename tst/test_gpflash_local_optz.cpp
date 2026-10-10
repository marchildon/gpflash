// The local-solve backend (optimization/local_optz): run_local_optz on a convex quadratic with an analytically
// known solution, so bounds, algorithms, the projected-gradient optimality and the evaluation budget are checked
// independently of any GP; and the LocalOptzResult postcondition, which GPFLASH_CHECK compiles out of a release
// build, so this is the only thing that drives its reject paths.
#include <armadillo>
#include <cmath>
#include <cstdio>
#include <string>

#include "GpFlashTestSupport.hpp"
#include "gpflash/optimization/local_optz/LocalOptzResult.hpp"  // LocalOptzResult, check_local_optz_result
#include "gpflash/optimization/local_optz/OptimizerNlopt.hpp"   // run_local_optz, to_optz_algorithm_name
#include "gpflash/optimization/search_space/ActiveBounds.hpp"   // calc_at_bound_flags_vec, calc_projected_gradient_norm
#include "gpflash/optimization/search_space/OptzOptions.hpp"    // OptzOptions, NumericalOptzAlgorithm

using gpflash_test::accepts;
using gpflash_test::kDetailLen;
using gpflash_test::rejects;
using gpflash_test::report;

using namespace gpflash;

// The quadratic probe has an analytically known solution, so the optimizer must land on it and drive the
// objective to its exact minimum of zero.
constexpr double kTolOptzPoint = 1e-6;

constexpr double kTolOptzObjective = 1e-10;

// The unprojected gradient norm must be clearly non-zero, or the projection is not being exercised: a point pushed
// well outside the box leaves a gradient of order the box offset, not of order the tolerance. The bar only has to
// separate "large" from "converged".
constexpr double kMinRawGradNorm = 1.0;

// The optimizer seam (detail::run_local_optz) on a convex quadratic f(x) = ||x - c||^2, independent of the
// GP machinery: c inside the box is recovered, and c outside lands on the active bound with the flags
// reporting it, which is what shows the bounds are enforced natively rather than by clamping.
static bool run_backend_case() {
    using namespace gpflash::detail;

    auto make_quadratic = [](const arma::vec& center_vec) -> LocalObjective {
        return [center_vec](const arma::vec& x_vec, arma::vec* grad_vec_out) -> double {
            if (grad_vec_out) *grad_vec_out = 2.0 * (x_vec - center_vec);
            return arma::dot(x_vec - center_vec, x_vec - center_vec);
        };
    };

    // Asymmetric, non-unit region: a symmetric [-1,1]^2 one would hide a swapped or mis-indexed bound.
    const arma::vec lb_vec = {-0.7, -1.3};
    const arma::vec ub_vec = {1.1, 0.4};

    // Stopping criteria and algorithm: the shared OptzOptions defaults, which run_local_optz reads whole.
    const OptzOptions optz_opts;

    // (1) interior optimum recovered.
    const arma::vec center_interior_vec = {0.45, -0.62};
    LocalOptzResult local_optz_result_interior_box =
            run_local_optz(make_quadratic(center_interior_vec), arma::vec{0.0, 0.0}, lb_vec, ub_vec, optz_opts);
    const bool interior_ok = local_optz_result_interior_box.stop_reason == OptzStopReason::ToleranceMet
            && arma::norm(local_optz_result_interior_box.ln_hp_vec - center_interior_vec) < kTolOptzPoint
            && local_optz_result_interior_box.objective_val < kTolOptzObjective;

    // (2) exterior optima: one parks on the upper bound in x0 (interior in x1), the other on the lower bound in x1
    // (interior in x0), so both an upper and a lower bound are exercised.
    const arma::vec center_outside_vec = {5.0, -0.62};
    LocalOptzResult local_optz_result_active_box =
            run_local_optz(make_quadratic(center_outside_vec), arma::vec{0.0, 0.0}, lb_vec, ub_vec, optz_opts);
    const arma::vec center_below_vec = {0.45, -5.0};
    LocalOptzResult local_optz_result_lower_box =
            run_local_optz(make_quadratic(center_below_vec), arma::vec{0.0, 0.0}, lb_vec, ub_vec, optz_opts);
    const arma::uvec at_upper_flags_vec =
            calc_at_bound_flags_vec(local_optz_result_active_box.ln_hp_vec, lb_vec, ub_vec);
    const arma::uvec at_lower_flags_vec =
            calc_at_bound_flags_vec(local_optz_result_lower_box.ln_hp_vec, lb_vec, ub_vec);
    const bool bound_ok = std::abs(local_optz_result_active_box.ln_hp_vec(0) - ub_vec(0)) < kTolOptzPoint
            && std::abs(local_optz_result_active_box.ln_hp_vec(1) - center_outside_vec(1)) < kTolOptzPoint
            && at_upper_flags_vec(0) == 1 && at_upper_flags_vec(1) == 0
            && std::abs(local_optz_result_lower_box.ln_hp_vec(0) - center_below_vec(0)) < kTolOptzPoint
            && std::abs(local_optz_result_lower_box.ln_hp_vec(1) - lb_vec(1)) < kTolOptzPoint
            && at_lower_flags_vec(0) == 0
            && at_lower_flags_vec(1) == 1
            // A bound vector of the wrong width is rejected rather than read past its end.
            && rejects([&] { calc_at_bound_flags_vec(local_optz_result_lower_box.ln_hp_vec, lb_vec.head(1), ub_vec); });

    // (3) Every selectable algorithm solves the same two problems and honors the same box, so selecting one
    // cannot silently return an infeasible or unconverged point. Each also maps to a distinct backend name.
    bool all_methods_ok = true;
    const NumericalOptzAlgorithm numerical_optz_algorithm_list[] = {
            NumericalOptzAlgorithm::Sqp, NumericalOptzAlgorithm::Ccsaq};
    std::string previous_algorithm_name;
    std::string failed_algorithm_names;
    for (const NumericalOptzAlgorithm numerical_optz_algorithm : numerical_optz_algorithm_list) {
        OptzOptions optz_opts_algorithm = optz_opts;
        optz_opts_algorithm.numerical_optz_algorithm = numerical_optz_algorithm;
        const LocalOptzResult local_optz_result_interior = run_local_optz(
                make_quadratic(center_interior_vec), arma::vec{0.0, 0.0}, lb_vec, ub_vec, optz_opts_algorithm);
        const LocalOptzResult local_optz_result_bound = run_local_optz(
                make_quadratic(center_outside_vec), arma::vec{0.0, 0.0}, lb_vec, ub_vec, optz_opts_algorithm);
        const std::string algorithm_name = to_optz_algorithm_name(numerical_optz_algorithm);
        const bool algorithm_ok = local_optz_result_interior.stop_reason == OptzStopReason::ToleranceMet
                && arma::norm(local_optz_result_interior.ln_hp_vec - center_interior_vec) < kTolOptzPoint
                // Feasibility is the load-bearing claim: the box must hold in both coordinates.
                && arma::all(local_optz_result_bound.ln_hp_vec >= lb_vec - kTolOptzPoint)
                && arma::all(local_optz_result_bound.ln_hp_vec <= ub_vec + kTolOptzPoint)
                && std::abs(local_optz_result_bound.ln_hp_vec(0) - ub_vec(0)) < kTolOptzPoint && !algorithm_name.empty()
                && algorithm_name != previous_algorithm_name;
        if (!algorithm_ok) failed_algorithm_names += " " + algorithm_name;
        all_methods_ok = all_methods_ok && algorithm_ok;
        previous_algorithm_name = algorithm_name;
    }
    if (!all_methods_ok)
        std::printf("  [backend] failed the box, optimum or name check:%s\n", failed_algorithm_names.c_str());

    // (4) The optimality measure must account for an active bound on either side. At an exterior-optimum solution
    // the raw gradient is large, but the bound holds the point there, so it is first-order optimal and the
    // projected norm is ~0. Without the projection a bounded fit would report itself unconverged.
    const auto calc_norms = [&](const arma::vec& center_vec, const arma::vec& ln_hp_vec, double& raw_norm_out) {
        arma::vec grad_at_bound_vec;
        make_quadratic(center_vec)(ln_hp_vec, &grad_at_bound_vec);
        raw_norm_out = arma::norm(grad_at_bound_vec);
        return calc_projected_gradient_norm(ln_hp_vec, grad_at_bound_vec, lb_vec, ub_vec);
    };
    double raw_norm_upper = 0.0;
    double raw_norm_lower = 0.0;
    const double projected_norm_upper =
            calc_norms(center_outside_vec, local_optz_result_active_box.ln_hp_vec, raw_norm_upper);
    const double projected_norm_lower =
            calc_norms(center_below_vec, local_optz_result_lower_box.ln_hp_vec, raw_norm_lower);
    // At a bound with the gradient pointing inward, stepping against it is a feasible descent, so the component
    // counts in full.
    const double projected_norm_inward_upper =
            calc_projected_gradient_norm(arma::vec{ub_vec(0), 0.0}, arma::vec{3.0, 0.0}, lb_vec, ub_vec);
    const double projected_norm_inward_lower =
            calc_projected_gradient_norm(arma::vec{0.0, lb_vec(1)}, arma::vec{0.0, -4.0}, lb_vec, ub_vec);
    const bool projection_ok =
            raw_norm_upper > kMinRawGradNorm && projected_norm_upper < kTolOptzPoint && raw_norm_lower > kMinRawGradNorm
            && projected_norm_lower < kTolOptzPoint && projected_norm_inward_upper == 3.0
            && projected_norm_inward_lower == 4.0 && rejects([&] {
                   calc_projected_gradient_norm(
                           local_optz_result_lower_box.ln_hp_vec, arma::vec(3, arma::fill::ones), lb_vec, ub_vec);
               });

    // (5) The evaluation budget is a stop reason of its own, not a failure.
    OptzOptions optz_opts_budget = optz_opts;
    optz_opts_budget.max_evals = 2;
    const LocalOptzResult local_optz_result_budget =
            run_local_optz(make_quadratic(center_interior_vec), arma::vec{0.0, 0.0}, lb_vec, ub_vec, optz_opts_budget);
    const bool budget_ok = local_optz_result_budget.stop_reason == OptzStopReason::EvalBudgetSpent
            && local_optz_result_budget.n_eval <= 2;

    const bool pass = interior_ok && bound_ok && projection_ok && all_methods_ok && budget_ok;
    std::printf(
            "%-14s interior=%d bound=%d proj=%d methods=%d budget=%d  %s\n",
            "Backend",
            interior_ok,
            bound_ok,
            projection_ok,
            all_methods_ok,
            budget_ok,
            pass ? "PASS" : "*** FAIL ***");

    return pass;
}

// The postcondition every backend result must satisfy.
static bool test_local_optz_result_checks(const char* name) {
    using gpflash::detail::check_local_optz_result;
    const int n_dim = 2;

    // A solve that finished, as a backend returns one.
    gpflash::LocalOptzResult local_optz_result;
    local_optz_result.ln_hp_vec = arma::vec(n_dim, arma::fill::zeros);
    local_optz_result.objective_val = -3.5;
    local_optz_result.n_eval = 12;
    local_optz_result.stop_reason = gpflash::OptzStopReason::ToleranceMet;

    const bool local_ok = accepts([&] { check_local_optz_result(local_optz_result, "test"); });
    // An empty minimizer first: check_all_finite_vals passes on an empty vector, so without the extent check a
    // zero-parameter result would satisfy the postcondition.
    const bool empty_rejected = rejects([&] {
        gpflash::LocalOptzResult broken = local_optz_result;
        broken.ln_hp_vec.reset();
        check_local_optz_result(broken, "test");
    });
    const bool nan_rejected = rejects([&] {
        gpflash::LocalOptzResult broken = local_optz_result;
        broken.ln_hp_vec(0) = arma::datum::nan;
        check_local_optz_result(broken, "test");
    });
    const bool n_eval_rejected = rejects([&] {
        gpflash::LocalOptzResult broken = local_optz_result;
        broken.n_eval = -1;
        check_local_optz_result(broken, "test");
    });
    const bool stop_reason_rejected = rejects([&] {  // outside the enum
        gpflash::LocalOptzResult broken = local_optz_result;
        broken.stop_reason = static_cast<gpflash::OptzStopReason>(99);
        check_local_optz_result(broken, "test");
    });

    // A failed start reports +inf and must pass: it is recorded as a column like any other.
    gpflash::LocalOptzResult local_optz_result_failed = local_optz_result;
    local_optz_result_failed.objective_val = arma::datum::inf;
    local_optz_result_failed.stop_reason = gpflash::OptzStopReason::Failed;
    const bool failed_start_accepted = accepts([&] { check_local_optz_result(local_optz_result_failed, "test"); });
    const bool nan_objective_rejected = rejects([&] {  // NaN would win a minimum comparison
        gpflash::LocalOptzResult broken = local_optz_result_failed;
        broken.objective_val = arma::datum::nan;
        check_local_optz_result(broken, "test");
    });

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "accept=%d empty=%d nan=%d n_eval=%d stop_reason=%d failed_accept=%d nan_objective=%d",
            static_cast<int>(local_ok),
            static_cast<int>(empty_rejected),
            static_cast<int>(nan_rejected),
            static_cast<int>(n_eval_rejected),
            static_cast<int>(stop_reason_rejected),
            static_cast<int>(failed_start_accepted),
            static_cast<int>(nan_objective_rejected));
    return report(
            name,
            local_ok && empty_rejected && nan_rejected && n_eval_rejected && stop_reason_rejected
                    && failed_start_accepted && nan_objective_rejected,
            detail);
}

int main() {
    std::printf("== GpFlash local optimization ==\n");
    bool ok = true;
    ok &= run_backend_case();
    ok &= test_local_optz_result_checks("local_optz_result_checks");
    std::printf("\n%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");

    return ok ? 0 : 1;
}
