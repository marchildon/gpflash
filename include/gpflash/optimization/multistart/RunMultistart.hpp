/**
 * RunMultistart.hpp -- keep-best multistart over the log hyperparameters. Restarts come from a Latin-hypercube
 * design over the sampling box; each local solve goes through run_local_optz() in OptimizerNlopt.hpp.
 *
 * A start that raises a numerical failure, in its MLL or in the backend, is recorded as Failed at +inf, not
 * rethrown; only every start failing ends the fit. Any other exception is a bug and propagates.
 */
#pragma once

#include <algorithm>  // std::min
#include <armadillo>  // arma::mat
#include <exception>  // std::exception_ptr
#include <limits>     // std::numeric_limits, for a failed start's +inf
#include <random>     // std::mt19937_64
#include <string>     // std::to_string
#include <vector>     // the optimizations this process runs

#include "gpflash/core/Errors.hpp"                              // ErrorNumericalFailure, ErrorPeerFailure
#include "gpflash/core/ParallelContext.hpp"                     // ParallelContext, the processes a fit is spread over
#include "gpflash/core/Platform.hpp"                            // GPFLASH_CHECK
#include "gpflash/optimization/local_optz/LocalOptzResult.hpp"  // LocalObjective, LocalOptzResult, OptzStopReason
#include "gpflash/optimization/local_optz/OptimizerNlopt.hpp"   // run_local_optz (the optimizer backend)
#include "gpflash/optimization/multistart/LatinHypercube.hpp"   // build_latin_hypercube_unit_cube
#include "gpflash/optimization/multistart/MultistartResult.hpp"  // MultistartResult, StartField, fill_all_ind_optz_inplace
#include "gpflash/optimization/search_space/HpOptzLayout.hpp"  // HpOptzLayout
#include "gpflash/optimization/search_space/OptzOptions.hpp"   // OptzOptions
#include "gpflash/optimization/search_space/OptzRegion.hpp"    // OptzRegion

namespace gpflash {
namespace detail {

// ── Splitting the independent optimizations over processes ───────────────────

// Two counts, which are not the same: how many processes the fit is spread over, which is the context's, and how
// many of them optimize, which is this one. max_parallel_ind_optz bounds memory -- each concurrent solve holds its
// own (n_train, n_train) factorization -- so it caps the workers only: every process still reaches every
// collective, the ones past the cap with zeros to contribute.
inline int calc_n_optz_process(
        const OptzOptions& optz_opts, const ParallelContext& parallel_context, const int n_ind_optz) {
    return std::min(std::min(parallel_context.n_process(), optz_opts.max_parallel_ind_optz), n_ind_optz);
}

// The optimizations this process runs, round-robin: optimization i belongs to process i % n_optz_process, and a
// process beyond the cap runs none. A serial context runs them all.
inline std::vector<int> calc_ind_optz_idx_vec(
        const ParallelContext& parallel_context, const int n_optz_process, const int n_ind_optz) {
    const int i_process = parallel_context.i_process();
    std::vector<int> ind_optz_idx_vec;
    if (i_process >= n_optz_process) return ind_optz_idx_vec;
    for (int i_ind_optz = i_process; i_ind_optz < n_ind_optz; i_ind_optz += n_optz_process)
        ind_optz_idx_vec.push_back(i_ind_optz);

    return ind_optz_idx_vec;
}

// ── Leaving the fit the same way on every process ────────────────────────────

// What a process carries out of a stage of the fit, worst last: the agreement below takes the maximum, so a defect
// on one process outranks a clean finish on another. SetupFailed is the one error the stage before the search
// carries.
enum class FitErrorKind : int { None = 0, AllStartsFailed = 1, Defect = 2, SetupFailed = 3 };

// The text of an exception already caught and held. Only call this on a non-null pointer.
inline std::string to_error_message(const std::exception_ptr& error_ptr) {
    try {
        std::rethrow_exception(error_ptr);
    } catch (const std::exception& caught) {
        return caught.what();
    } catch (...) {
        return "an exception not derived from std::exception";
    }
}

// Agree on the error state across every process the fit is spread over, and give them all the message the
// lowest-numbered process holding that state reported. Without it a defect is asymmetric: the process whose start
// raised would throw while its peers returned, leaving the application split and its next collective deadlocked.
// Returns the number of the process whose message was adopted, or -1 when the agreed state is None.
inline int agree_fit_error_inplace(
        FitErrorKind& fit_error_kind_inout,
        std::string& fit_error_message_inout,
        const ParallelContext& parallel_context) {
    const int local_kind = static_cast<int>(fit_error_kind_inout);
    const int agreed_kind = parallel_context.calc_max(local_kind);
    fit_error_kind_inout = static_cast<FitErrorKind>(agreed_kind);
    if (agreed_kind == static_cast<int>(FitErrorKind::None)) return -1;

    // A process without the agreed state offers the process count, so the minimum is the lowest-numbered one that
    // does have it.
    const int local_root = (local_kind == agreed_kind) ? parallel_context.i_process() : parallel_context.n_process();
    const int i_root_process = parallel_context.calc_min(local_root);
    parallel_context.broadcast_inplace(fit_error_message_inout, i_root_process);

    return i_root_process;
}

// ── One independent optimization ─────────────────────────────────────────────

// The heuristic start for optimization 0; otherwise its row of the unit-cube design mapped onto the sampling box.
inline arma::vec calc_ln_hp_init_vec(
        const int i_ind_optz, const arma::mat& unit_cube_mat, const OptzRegion& optz_region, const int n_hp_optz) {
    arma::vec ln_hp_init_vec = optz_region.start_vec;
    if (i_ind_optz == 0) return ln_hp_init_vec;

    // The sampling corners are finite even for an unconstrained fit.
    for (int i_hp_optz = 0; i_hp_optz < n_hp_optz; ++i_hp_optz) {
        const double sample_lb = optz_region.sample_lb_vec(i_hp_optz);
        const double sample_ub = optz_region.sample_ub_vec(i_hp_optz);
        ln_hp_init_vec(i_hp_optz) = sample_lb + unit_cube_mat(i_ind_optz - 1, i_hp_optz) * (sample_ub - sample_lb);
    }

    return ln_hp_init_vec;
}

// One independent optimization: what the backend returned, plus what the driver knows around it.
struct IndOptzOutcome {
    arma::vec ln_hp_init_vec;           // where it started
    LocalOptzResult local_optz_result;  // Failed at +inf if the start raised
    int n_mll_eval = 0;                 // value-only MLL evaluations; the backend cannot tell these apart
    int n_mll_grad_eval = 0;            // MLL evaluations that also formed the gradient
};

// One local solve, its objective wrapped to count the MLL evaluations.
//
// The outcome is filled through a reference rather than returned, so a solve that raises still leaves its
// evaluation counts behind: what a failing start spent is what a report needs.
inline void run_ind_optz_inplace(
        const arma::vec& ln_hp_init_vec,
        const OptzRegion& optz_region,
        const LocalObjective& neg_mll_objective,
        const OptzOptions& optz_opts,
        IndOptzOutcome& ind_optz_outcome_out) {
    ind_optz_outcome_out.ln_hp_init_vec = ln_hp_init_vec;
    const LocalObjective counted_objective = [&neg_mll_objective, &ind_optz_outcome_out](
                                                     const arma::vec& ln_hp_vec, arma::vec* grad_vec_out) {
        if (grad_vec_out)
            ++ind_optz_outcome_out.n_mll_grad_eval;
        else
            ++ind_optz_outcome_out.n_mll_eval;

        return neg_mll_objective(ln_hp_vec, grad_vec_out);
    };

    // The backend enforces the bounds natively; the point comes back feasible.
    ind_optz_outcome_out.local_optz_result = run_local_optz(
            counted_objective, ln_hp_init_vec, optz_region.constraint_lb_vec, optz_region.constraint_ub_vec, optz_opts);
}

// ── The search ───────────────────────────────────────────────────────────────

// The only place a column is filled: a StartField added there is filled here.
inline void fill_ind_optz_column_inplace(
        MultistartResult& multistart_result, const int i_ind_optz, const IndOptzOutcome& ind_optz_outcome) {
    const arma::uword i_col = static_cast<arma::uword>(i_ind_optz);
    const LocalOptzResult& local_optz_result = ind_optz_outcome.local_optz_result;
    multistart_result.ln_hp_mat.col(i_col) = local_optz_result.ln_hp_vec;
    multistart_result.ln_hp_init_mat.col(i_col) = ind_optz_outcome.ln_hp_init_vec;

    multistart_result.start_field_mat(static_cast<arma::uword>(StartField::ObjectiveVal), i_col) =
            local_optz_result.objective_val;
    multistart_result.start_field_mat(static_cast<arma::uword>(StartField::NEval), i_col) =
            static_cast<double>(local_optz_result.n_eval);
    multistart_result.start_field_mat(static_cast<arma::uword>(StartField::NMllEval), i_col) =
            static_cast<double>(ind_optz_outcome.n_mll_eval);
    multistart_result.start_field_mat(static_cast<arma::uword>(StartField::NMllGradEval), i_col) =
            static_cast<double>(ind_optz_outcome.n_mll_grad_eval);
    multistart_result.start_field_mat(static_cast<arma::uword>(StartField::StopReason), i_col) =
            static_cast<double>(static_cast<int>(local_optz_result.stop_reason));
}

// A failed start records its start point (the only finite point it has) and LocalOptzResult's +inf, so it loses
// every comparison. The two counters the objective kept before it raised are left in place.
inline void mark_ind_optz_outcome_failed_inplace(
        const arma::vec& ln_hp_init_vec, IndOptzOutcome& ind_optz_outcome_out) {
    ind_optz_outcome_out.ln_hp_init_vec = ln_hp_init_vec;
    ind_optz_outcome_out.local_optz_result.ln_hp_vec = ln_hp_init_vec;
    ind_optz_outcome_out.local_optz_result.objective_val = std::numeric_limits<double>::infinity();
    ind_optz_outcome_out.local_optz_result.stop_reason = OptzStopReason::Failed;
    // The backend raised before reporting its count, so n_eval is the counters' sum, as for a start that succeeds.
    ind_optz_outcome_out.local_optz_result.n_eval =
            ind_optz_outcome_out.n_mll_eval + ind_optz_outcome_out.n_mll_grad_eval;
}

// Keep-best multistart. Returns a complete record: the winner and totals are derived from its arrays.
inline MultistartResult run_multistart(
        const OptzRegion& optz_region,
        const HpOptzLayout& hp_optz_layout,
        const LocalObjective& neg_mll_objective,
        const OptzOptions& optz_opts,
        const ParallelContext& parallel_context,
        const char* caller) {
    // Precondition: n_ind_optz >= 1.
    const int n_hp_optz = hp_optz_layout.n_hp_optz();
    const int n_ind_optz = optz_opts.use_multistart ? optz_opts.n_ind_optz : 1;

    // The two counts size every collective below and the seed places the starts, so a process that differs on
    // any of them is not running the same search.
    parallel_context.check_same_on_every_process(n_hp_optz, "the number of hyperparameters searched", caller);
    parallel_context.check_same_on_every_process(n_ind_optz, "the number of independent optimizations", caller);
    parallel_context.check_same_on_every_process(
            optz_opts.hypercube_sampling_seed, "OptzOptions::hypercube_sampling_seed", caller);

    std::mt19937_64 rng(optz_opts.hypercube_sampling_seed);
    const arma::mat unit_cube_mat =
            (n_ind_optz > 1) ? build_latin_hypercube_unit_cube(n_ind_optz - 1, n_hp_optz, rng) : arma::mat();

    // Zeroed: each process writes only its own columns and the gather sums.
    MultistartResult multistart_result;
    multistart_result.hp_optz_layout = hp_optz_layout;
    multistart_result.ln_hp_mat.zeros(n_hp_optz, n_ind_optz);
    multistart_result.ln_hp_init_mat.zeros(n_hp_optz, n_ind_optz);
    multistart_result.start_field_mat.zeros(static_cast<int>(StartField::End), n_ind_optz);

    const int n_optz_process = calc_n_optz_process(optz_opts, parallel_context, n_ind_optz);
    parallel_context.check_same_on_every_process(n_optz_process, "the number of processes optimizing", caller);

    // Both kinds of failure are recorded so the gather below always completes: ErrorNumericalFailure is a bad
    // start and the fit carries on, anything else is a defect and is rethrown after the gather. Every call that
    // can raise sits inside the try, since a process that unwound from this loop would leave the others in the
    // gather.
    std::exception_ptr first_numerical_failure_ptr;
    std::exception_ptr bug_ptr;

    for (const int i_ind_optz : calc_ind_optz_idx_vec(parallel_context, n_optz_process, n_ind_optz)) {
        // Sized here, so a start point exists to record even if calc_ln_hp_init_vec is what raises.
        arma::vec ln_hp_init_vec(n_hp_optz, arma::fill::zeros);
        IndOptzOutcome ind_optz_outcome;
        try {
            ln_hp_init_vec = calc_ln_hp_init_vec(i_ind_optz, unit_cube_mat, optz_region, n_hp_optz);
            run_ind_optz_inplace(ln_hp_init_vec, optz_region, neg_mll_objective, optz_opts, ind_optz_outcome);
            fill_ind_optz_column_inplace(multistart_result, i_ind_optz, ind_optz_outcome);
            continue;
        } catch (const ErrorNumericalFailure&) {
            if (!first_numerical_failure_ptr) first_numerical_failure_ptr = std::current_exception();
        } catch (...) {
            if (!bug_ptr) bug_ptr = std::current_exception();
        }
        mark_ind_optz_outcome_failed_inplace(ln_hp_init_vec, ind_optz_outcome);
        fill_ind_optz_column_inplace(multistart_result, i_ind_optz, ind_optz_outcome);
    }

    fill_all_ind_optz_inplace(multistart_result, parallel_context);

    // What this process would throw, and then what every process agrees to throw. The record is gathered by now,
    // so "every start failed" is the same verdict everywhere already; a defect is not, hence the agreement.
    FitErrorKind fit_error_kind = FitErrorKind::None;
    std::string fit_error_message;
    if (bug_ptr) {
        fit_error_kind = FitErrorKind::Defect;
        fit_error_message = to_error_message(bug_ptr);
    } else if (multistart_result.calc_n_failed_start() == n_ind_optz) {
        fit_error_kind = FitErrorKind::AllStartsFailed;
        fit_error_message = first_numerical_failure_ptr ? to_error_message(first_numerical_failure_ptr)
                                                        : "every independent optimization failed.";
    }
    const int i_failed_process = agree_fit_error_inplace(fit_error_kind, fit_error_message, parallel_context);

    // The last collective is done, so a process may unwind from here. Each throws: the one that saw the failure
    // rethrows it with its type intact, the rest report the agreed message so none of them returns a result.
    if (fit_error_kind == FitErrorKind::Defect) {
        if (bug_ptr) std::rethrow_exception(bug_ptr);
        throw ErrorPeerFailure(
                std::string(caller) + ": process " + std::to_string(i_failed_process) + " of "
                + std::to_string(parallel_context.n_process()) + " in this fit failed: " + fit_error_message);
    }
    if (fit_error_kind == FitErrorKind::AllStartsFailed) {
        if (first_numerical_failure_ptr) std::rethrow_exception(first_numerical_failure_ptr);
        throw ErrorNumericalFailure(
                std::string(caller) + ": all " + std::to_string(n_ind_optz) + " independent optimizations failed ("
                + fit_error_message + ").");
    }

    GPFLASH_CHECK(check_multistart_result(multistart_result, caller));
    return multistart_result;
}

}  // namespace detail
}  // namespace gpflash
