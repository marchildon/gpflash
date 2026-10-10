/**
 * LocalOptzResult.hpp -- the optimizer backend seam: the objective a local solve receives and the outcome it returns.
 * Another backend is a new file providing run_local_optz() against these two types.
 *
 * The outcome holds only what the backend sees. Start point and MLL tallies live in MultistartResult.hpp, and
 * MllOptzResult.hpp derives the at-bound flags.
 */
#pragma once

#include <armadillo>   // arma::vec
#include <functional>  // std::function
#include <limits>      // std::numeric_limits

#include "gpflash/core/Errors.hpp"  // the field checks

namespace gpflash {

namespace detail {

// Objective in log-hyperparameter space. Fills *grad_vec_out when it is non-null.
using LocalObjective = std::function<double(const arma::vec& ln_hp_vec, arma::vec* grad_vec_out)>;

}  // namespace detail

// Why a local solve stopped. Only EvalBudgetSpent is a reason to raise max_evals.
enum class OptzStopReason {
    ToleranceMet,     // the backend's step or objective tolerance fired
    EvalBudgetSpent,  // OptzOptions::max_evals ran out
    RoundoffLimited,  // the backend could not improve further in floating point
    Failed,           // the start raised a numerical failure, in the MLL or the backend; no minimizer
    End,              // one past the last, for check_enum_val
};

struct LocalOptzResult {
    arma::vec ln_hp_vec;                                             // the minimizer
    double objective_val = std::numeric_limits<double>::infinity();  // at the minimizer; +inf when Failed
    int n_eval = 0;                                                  // objective evaluations
    OptzStopReason stop_reason = OptzStopReason::Failed;             // why the solve returned; Failed until one does
};

inline const char* to_stop_reason_name(const OptzStopReason stop_reason) {
    switch (stop_reason) {
        case OptzStopReason::ToleranceMet:
            return "tolerance";
        case OptzStopReason::EvalBudgetSpent:
            return "eval_budget";  // one word: the report table is whitespace-aligned
        case OptzStopReason::RoundoffLimited:
            return "roundoff";
        case OptzStopReason::Failed:
            return "failed";
        case OptzStopReason::End:
            break;
    }
    return "unknown";
}

namespace detail {

// Postcondition.
inline void check_local_optz_result(const LocalOptzResult& local_optz_result, const char* caller) {
    check_vec_not_empty(local_optz_result.ln_hp_vec, "LocalOptzResult::ln_hp_vec", caller);
    check_all_finite_vals(local_optz_result.ln_hp_vec, "LocalOptzResult::ln_hp_vec", caller);

    check_enum_val(local_optz_result.stop_reason, "OptzStopReason", "LocalOptzResult::stop_reason", caller);
    // +inf is a failed start, so only NaN is rejected.
    check_not_nan_val(local_optz_result.objective_val, "LocalOptzResult::objective_val", caller);

    check_non_negative_count(local_optz_result.n_eval, "LocalOptzResult::n_eval", caller);
}

}  // namespace detail

}  // namespace gpflash
