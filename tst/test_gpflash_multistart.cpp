// The multistart search (optimization/multistart): the Latin-hypercube design, what run_multistart does with a
// start that raises, and the MultistartResult postcondition, which GPFLASH_CHECK compiles out of a release build so
// this is the only thing that drives its reject paths. Driven with synthetic objectives over a real region, so no
// case depends on optimizer numerics.
#include <armadillo>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>

#include "GpFlashTestSupport.hpp"
#include "gpflash/core/Errors.hpp"                               // the three error types a start may raise
#include "gpflash/core/ParallelContext.hpp"                      // ParallelContext
#include "gpflash/optimization/multistart/LatinHypercube.hpp"    // build_latin_hypercube_unit_cube
#include "gpflash/optimization/multistart/MultistartResult.hpp"  // MultistartResult, check_multistart_result
#include "gpflash/optimization/multistart/RunMultistart.hpp"     // run_multistart, fill_ind_optz_column_inplace
#include "gpflash/optimization/search_space/HpOptzLayout.hpp"    // build_hp_optz_layout
#include "gpflash/optimization/search_space/OptzRegion.hpp"      // build_optz_region

using gpflash_test::accepts;
using gpflash_test::draw_uniform_mat;
using gpflash_test::draw_uniform_vec;
using gpflash_test::kDetailLen;
using gpflash_test::rejects;
using gpflash_test::report;

// The Latin-hypercube design: one point per stratum on every axis, every point inside the unit cube, and the same
// seed gives the same design -- which is what lets every MPI process build it without communicating.
static bool test_latin_hypercube(const char* name) {
    const int n_sample = 7;
    const int n_dim_sample = 3;
    std::mt19937_64 rng(5);
    const arma::mat unit_cube_mat = gpflash::detail::build_latin_hypercube_unit_cube(n_sample, n_dim_sample, rng);

    const bool shape_ok = unit_cube_mat.n_rows == static_cast<arma::uword>(n_sample)
            && unit_cube_mat.n_cols == static_cast<arma::uword>(n_dim_sample);
    const bool inside_ok = unit_cube_mat.min() >= 0.0 && unit_cube_mat.max() < 1.0;

    // Stratified: on each axis the strata floor(n_sample * x) are a permutation of 0 .. n_sample - 1.
    bool stratified_ok = true;
    for (int i_dim_sample = 0; i_dim_sample < n_dim_sample; ++i_dim_sample) {
        arma::uvec stratum_vec(static_cast<arma::uword>(n_sample));
        for (int i_sample = 0; i_sample < n_sample; ++i_sample)
            stratum_vec(i_sample) =
                    static_cast<arma::uword>(std::floor(n_sample * unit_cube_mat(i_sample, i_dim_sample)));
        stratified_ok =
                stratified_ok && arma::all(arma::sort(stratum_vec) == arma::regspace<arma::uvec>(0, n_sample - 1));
    }

    std::mt19937_64 rng_same(5);
    std::mt19937_64 rng_other(6);
    const bool reproducible = arma::approx_equal(
            gpflash::detail::build_latin_hypercube_unit_cube(n_sample, n_dim_sample, rng_same),
            unit_cube_mat,
            "absdiff",
            0.0);
    const bool seed_matters = !arma::approx_equal(
            gpflash::detail::build_latin_hypercube_unit_cube(n_sample, n_dim_sample, rng_other),
            unit_cube_mat,
            "absdiff",
            0.0);

    // Each axis permuted on its own: one permutation shared by every axis passes every check above and puts every
    // point on the stratum diagonal.
    const arma::vec stratum_axis_0_vec = arma::floor(n_sample * unit_cube_mat.col(0));
    bool axes_differ = false;
    for (int i_dim_sample = 1; i_dim_sample < n_dim_sample; ++i_dim_sample)
        axes_differ =
                axes_differ || arma::any(arma::floor(n_sample * unit_cube_mat.col(i_dim_sample)) != stratum_axis_0_vec);

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "shape=%d inside=%d stratified=%d reproducible=%d seed_matters=%d axes_differ=%d",
            static_cast<int>(shape_ok),
            static_cast<int>(inside_ok),
            static_cast<int>(stratified_ok),
            static_cast<int>(reproducible),
            static_cast<int>(seed_matters),
            static_cast<int>(axes_differ));
    return report(name, shape_ok && inside_ok && stratified_ok && reproducible && seed_matters && axes_differ, detail);
}

// The design for a fixed seed, exactly, so a change to how it consumes the generator fails here on every platform.
// The literals were produced on libc++ and match libstdc++ bitwise. %.17g round-trips a double exactly, and C++14
// has no hex-float literals.
static bool test_latin_hypercube_pinned(const char* name) {
    // The generator first: the standard requires a default-constructed std::mt19937_64's 10000th output to be this
    // value, so a failure here is the generator's, not the design's.
    std::mt19937_64 rng_default;
    for (int i_draw = 1; i_draw < 10000; ++i_draw) rng_default();
    const bool generator_ok = rng_default() == 9981545732273789042ULL;

    std::mt19937_64 rng(5);
    const arma::mat unit_cube_mat = gpflash::detail::build_latin_hypercube_unit_cube(4, 3, rng);
    const arma::mat pinned_mat = {
            {0.91898304635819994, 0.76299172026797268, 0.11073405194918415},
            {0.022591722403859038, 0.64291980553220274, 0.74172744914365885},
            {0.27408560724403674, 0.069922874231837046, 0.9612121490320491},
            {0.53245640508902725, 0.48112902029107707, 0.35600348973197282}};
    const bool design_ok = unit_cube_mat.n_rows == pinned_mat.n_rows && unit_cube_mat.n_cols == pinned_mat.n_cols
            && arma::all(arma::vectorise(unit_cube_mat == pinned_mat));

    // Bounded integers, one generator across the bounds. Bound 1 always gives 0 and still consumes a draw, so the
    // sequence position does not depend on the bound.
    std::mt19937_64 rng_index(5);
    const std::uint64_t bound_list[] = {1, 2, 3, 7, 1000};
    const std::uint64_t pinned_index_list[] = {0, 0, 2, 6, 844};
    bool index_ok = true;
    for (int i_bound = 0; i_bound < 5; ++i_bound)
        index_ok = index_ok
                && gpflash::detail::draw_index_below_portable(rng_index, bound_list[i_bound])
                        == pinned_index_list[i_bound];

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "generator=%d design=%d bounded_index=%d",
            static_cast<int>(generator_ok),
            static_cast<int>(design_ok),
            static_cast<int>(index_ok));
    return report(name, generator_ok && design_ok && index_ok, detail);
}

// What run_multistart does with an exception from a start: a numerical failure is recorded and the fit stands on the
// other starts; anything else is a bug and propagates. Driven with synthetic objectives over a real region.
static bool test_failed_start_policy(const char* name) {
    const int n_dim = 2;
    const int n_train = 12;
    std::mt19937_64 rng(17);
    gpflash::TrainingData train_data;
    train_data.X_ndim_ntrain_mat = draw_uniform_mat(rng, n_dim, n_train);
    train_data.f_train_vec = draw_uniform_vec(rng, n_train);
    train_data.sig2_f_val = 1e-3;

    gpflash::OptzOptions optz_opts;
    optz_opts.use_multistart = true;
    optz_opts.n_ind_optz = 3;
    const gpflash::ParallelContext parallel_context;  // serial: this target is built without MPI
    const gpflash::HpOptzLayout hp_optz_layout =
            gpflash::detail::build_hp_optz_layout(n_dim, 0, gpflash::NoiseRequest::Fixed, "test");
    const double f_train_residual_var = arma::var(train_data.f_train_vec);  // the constant mean model's
    const arma::vec x_center_vec = arma::zeros<arma::vec>(train_data.X_ndim_ntrain_mat.n_rows);
    const gpflash::OptzRegion optz_region = gpflash::detail::build_optz_region(
            train_data, x_center_vec, f_train_residual_var, optz_opts, gpflash::OptzSeeding(), hp_optz_layout, "test");

    // A convex quadratic that raises `make_error()` at exactly the heuristic start, which only optimization 0 sees.
    const arma::vec ln_hp_start_vec = optz_region.start_vec;
    auto make_objective = [&](auto make_error) -> gpflash::detail::LocalObjective {
        return [ln_hp_start_vec, make_error](const arma::vec& ln_hp_vec, arma::vec* grad_vec_out) -> double {
            if (arma::approx_equal(ln_hp_vec, ln_hp_start_vec, "absdiff", 0.0)) throw make_error();
            if (grad_vec_out) *grad_vec_out = 2.0 * ln_hp_vec;
            return arma::dot(ln_hp_vec, ln_hp_vec);
        };
    };

    // ErrorNumericalFailure: recorded, the fit continues on the two other starts. Raised once, on the call after the
    // first value-only one, so the failed start has used both counters. Serial starts run in order, so that is
    // optimization 0.
    bool has_value_only_call = false;
    bool has_raised = false;
    int n_call_failed_start = 0;  // the raising call included
    const gpflash::detail::LocalObjective failing_after_value_only_objective =
            [&has_value_only_call, &has_raised, &n_call_failed_start](
                    const arma::vec& ln_hp_vec, arma::vec* grad_vec_out) -> double {
        if (!has_raised) ++n_call_failed_start;
        if (has_value_only_call && !has_raised) {
            has_raised = true;
            throw gpflash::ErrorNumericalFailure("seeded");
        }
        if (grad_vec_out)
            *grad_vec_out = 2.0 * ln_hp_vec;
        else
            has_value_only_call = true;
        return arma::dot(ln_hp_vec, ln_hp_vec);
    };
    bool numerical_recorded = false;
    bool failed_start_counts_agree = false;
    int n_eval = 0;
    int n_mll_eval = 0;
    int n_mll_grad_eval = 0;
    try {
        const gpflash::MultistartResult multistart_result = gpflash::detail::run_multistart(
                optz_region, hp_optz_layout, failing_after_value_only_objective, optz_opts, parallel_context, "test");
        numerical_recorded = multistart_result.calc_n_failed_start() == 1
                && multistart_result.stop_reason(0) == gpflash::OptzStopReason::Failed
                && multistart_result.idx_best() != 0;

        // The backend raised instead of reporting its own count, so the failed start's n_eval comes from the two
        // counters it is the sum of. Both are non-zero here, so neither alone can stand in for the sum, and the
        // total must include the call that raised.
        n_eval = multistart_result.start_field_count(gpflash::StartField::NEval, 0);
        n_mll_eval = multistart_result.start_field_count(gpflash::StartField::NMllEval, 0);
        n_mll_grad_eval = multistart_result.start_field_count(gpflash::StartField::NMllGradEval, 0);
        failed_start_counts_agree = n_mll_eval > 0 && n_mll_grad_eval > 0 && n_eval == n_mll_eval + n_mll_grad_eval
                && n_eval == n_call_failed_start;
    } catch (...) {
    }

    // ErrorNotImplemented and ErrorInvalidInput: bugs, so the fit raises with the original type.
    bool not_implemented_propagated = false;
    try {
        gpflash::detail::run_multistart(
                optz_region,
                hp_optz_layout,
                make_objective([] { return gpflash::ErrorNotImplemented("seeded"); }),
                optz_opts,
                parallel_context,
                "test");
    } catch (const gpflash::ErrorNotImplemented&) {
        not_implemented_propagated = true;
    } catch (...) {
    }
    bool invalid_input_propagated = false;
    try {
        gpflash::detail::run_multistart(
                optz_region,
                hp_optz_layout,
                make_objective([] { return gpflash::ErrorInvalidInput("seeded"); }),
                optz_opts,
                parallel_context,
                "test");
    } catch (const gpflash::ErrorInvalidInput&) {
        invalid_input_propagated = true;
    } catch (...) {
    }

    // Every start failing raises the first numerical failure itself.
    bool all_failed_raised = false;
    try {
        gpflash::detail::run_multistart(
                optz_region,
                hp_optz_layout,
                [](const arma::vec&, arma::vec*) -> double { throw gpflash::ErrorNumericalFailure("every start"); },
                optz_opts,
                parallel_context,
                "test");
    } catch (const gpflash::ErrorNumericalFailure& caught) {
        // The message, since the fallback for every start failing is an ErrorNumericalFailure too.
        all_failed_raised = std::string(caught.what()) == "every start";
    } catch (...) {
    }

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "recorded=%d counts=%d (n_eval=%d mll=%d grad=%d) not_implemented=%d invalid_input=%d all_failed=%d",
            static_cast<int>(numerical_recorded),
            static_cast<int>(failed_start_counts_agree),
            n_eval,
            n_mll_eval,
            n_mll_grad_eval,
            static_cast<int>(not_implemented_propagated),
            static_cast<int>(invalid_input_propagated),
            static_cast<int>(all_failed_raised));
    return report(
            name,
            numerical_recorded && failed_start_counts_agree && not_implemented_propagated && invalid_input_propagated
                    && all_failed_raised,
            detail);
}

// The record's postcondition, over a record built the way run_multistart fills it.
static bool test_multistart_result_checks(const char* name) {
    using gpflash::NoiseRequest;
    using gpflash::detail::build_hp_optz_layout;
    const int n_dim = 2;

    // A solve that finished, as a backend returns one, and one that failed.
    gpflash::LocalOptzResult local_optz_result;
    local_optz_result.ln_hp_vec = arma::vec(n_dim, arma::fill::zeros);
    local_optz_result.objective_val = -3.5;
    local_optz_result.n_eval = 12;
    local_optz_result.stop_reason = gpflash::OptzStopReason::ToleranceMet;
    gpflash::LocalOptzResult local_optz_result_failed = local_optz_result;
    local_optz_result_failed.objective_val = arma::datum::inf;
    local_optz_result_failed.stop_reason = gpflash::OptzStopReason::Failed;

    // The record over two starts, in the form run_multistart fills.
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
    ind_optz_outcome.n_mll_eval = 8;
    ind_optz_outcome.n_mll_grad_eval = 4;
    for (int i_ind_optz = 0; i_ind_optz < n_ind_optz; ++i_ind_optz) {
        // The second start is worse, so idx_best is a real choice.
        ind_optz_outcome.local_optz_result.objective_val = -3.5 + static_cast<double>(i_ind_optz);
        gpflash::detail::fill_ind_optz_column_inplace(multistart_result, i_ind_optz, ind_optz_outcome);
    }

    const bool record_accepted = accepts([&] { gpflash::detail::check_multistart_result(multistart_result, "test"); });
    // Everything the record reports is derived from those arrays.
    const bool record_derived_ok = multistart_result.idx_best() == 0 && multistart_result.mll() == 3.5
            && multistart_result.total_start_field_count(gpflash::StartField::NMllEval) == 16
            && multistart_result.calc_n_failed_start() == 0;
    const bool width_rejected = rejects([&] {
        gpflash::MultistartResult broken = multistart_result;
        broken.ln_hp_mat.zeros(n_dim, n_ind_optz + 1);
        gpflash::detail::check_multistart_result(broken, "test");
    });
    const bool nan_rejected = rejects([&] {
        gpflash::MultistartResult broken = multistart_result;
        broken.ln_hp_mat(0, 0) = arma::datum::nan;
        gpflash::detail::check_multistart_result(broken, "test");
    });
    const bool start_point_rejected = rejects([&] {  // the start points are checked too
        gpflash::MultistartResult broken = multistart_result;
        broken.ln_hp_init_mat(0, 1) = arma::datum::inf;
        gpflash::detail::check_multistart_result(broken, "test");
    });
    const bool other_start_count_rejected = rejects([&] {  // every start is checked, not only the winner
        gpflash::MultistartResult broken = multistart_result;
        broken.start_field_mat(static_cast<arma::uword>(gpflash::StartField::NMllGradEval), 1) = -1.0;
        gpflash::detail::check_multistart_result(broken, "test");
    });
    const bool other_count_rejected = rejects([&] {  // and every count
        gpflash::MultistartResult broken = multistart_result;
        broken.start_field_mat(static_cast<arma::uword>(gpflash::StartField::NMllEval), 0) = -1.0;
        gpflash::detail::check_multistart_result(broken, "test");
    });
    const bool empty_rejected = rejects([&] {  // no start recorded, so there is no winner to report
        gpflash::MultistartResult broken = multistart_result;
        broken.ln_hp_mat.reset();
        broken.ln_hp_init_mat.reset();
        broken.start_field_mat.reset();
        gpflash::detail::check_multistart_result(broken, "test");
    });
    const bool all_failed_rejected = rejects([&] {  // all failed: the winner is +inf
        gpflash::MultistartResult broken = multistart_result;
        gpflash::detail::IndOptzOutcome ind_optz_outcome_failed = ind_optz_outcome;
        ind_optz_outcome_failed.local_optz_result = local_optz_result_failed;
        for (int i_ind_optz = 0; i_ind_optz < n_ind_optz; ++i_ind_optz)
            gpflash::detail::fill_ind_optz_column_inplace(broken, i_ind_optz, ind_optz_outcome_failed);
        gpflash::detail::check_multistart_result(broken, "test");
    });

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "accept=%d derived=%d width=%d nan=%d start=%d other_start=%d other_count=%d empty=%d all_failed=%d",
            static_cast<int>(record_accepted),
            static_cast<int>(record_derived_ok),
            static_cast<int>(width_rejected),
            static_cast<int>(nan_rejected),
            static_cast<int>(start_point_rejected),
            static_cast<int>(other_start_count_rejected),
            static_cast<int>(other_count_rejected),
            static_cast<int>(empty_rejected),
            static_cast<int>(all_failed_rejected));
    return report(
            name,
            record_accepted && record_derived_ok && width_rejected && nan_rejected && start_point_rejected
                    && other_start_count_rejected && other_count_rejected && empty_rejected && all_failed_rejected,
            detail);
}

int main() {
    std::printf("== GpFlash multistart ==\n");
    bool ok = true;
    ok &= test_latin_hypercube("latin_hypercube");
    ok &= test_latin_hypercube_pinned("latin_hypercube_pinned");
    ok &= test_failed_start_policy("failed_start_policy");
    ok &= test_multistart_result_checks("multistart_result_checks");
    std::printf("\n%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");

    return ok ? 0 : 1;
}
