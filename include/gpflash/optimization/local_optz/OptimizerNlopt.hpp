/**
 * OptimizerNlopt.hpp -- the only file in GpFlash that knows about NLopt. Everything else in optimization/ is
 * written against LocalObjective and LocalOptzResult, so another optimizer is a new file providing
 * run_local_optz() against those same types.
 */
#pragma once

#include <algorithm>  // std::copy
#include <armadillo>  // arma::vec
#include <exception>  // std::exception_ptr, std::current_exception
#include <limits>     // std::numeric_limits
#include <nlopt.hpp>  // the optimizer backend
#include <stdexcept>  // std::runtime_error, what NLopt flattens its own failures to
#include <string>     // std::to_string, in the unhandled-result message
#include <vector>     // NLopt's bound and step arguments

#include "gpflash/core/Errors.hpp"                              // ErrorNumericalFailure, ErrorNotImplemented
#include "gpflash/core/Platform.hpp"                            // GPFLASH_CHECK
#include "gpflash/optimization/local_optz/LocalOptzResult.hpp"  // LocalObjective, LocalOptzResult, OptzStopReason
#include "gpflash/optimization/search_space/OptzOptions.hpp"    // NumericalOptzAlgorithm

namespace gpflash {
namespace detail {

// The neutral method mapped onto NLopt's enum. Both are LD_* algorithms -- gradient plus native bounds -- and
// both come from NLopt's permissively licensed code.
inline nlopt::algorithm to_nlopt_algorithm(const NumericalOptzAlgorithm numerical_optz_algorithm) {
    switch (numerical_optz_algorithm) {
        case NumericalOptzAlgorithm::Sqp:
            return nlopt::LD_SLSQP;
        case NumericalOptzAlgorithm::Ccsaq:
            return nlopt::LD_CCSAQ;
        case NumericalOptzAlgorithm::End:
            break;
    }
    // Only End, or a value cast in from outside the enum, which check_optz_options rejects first.
    throw ErrorNotImplemented("to_nlopt_algorithm: unknown NumericalOptzAlgorithm.");
}

// NLopt's failure codes arrive as exceptions, so any other code here is one this mapping does not know.
inline OptzStopReason to_optz_stop_reason(const nlopt::result nlopt_result) {
    switch (nlopt_result) {
        case nlopt::SUCCESS:
        case nlopt::STOPVAL_REACHED:
        case nlopt::FTOL_REACHED:
        case nlopt::XTOL_REACHED:
            return OptzStopReason::ToleranceMet;
        case nlopt::MAXEVAL_REACHED:
        case nlopt::MAXTIME_REACHED:
            return OptzStopReason::EvalBudgetSpent;
        default:
            break;
    }
    throw ErrorNotImplemented(
            std::string("to_optz_stop_reason: unhandled nlopt::result ")
            + std::to_string(static_cast<int>(nlopt_result)) + ".");
}

// The backend's own name for the algorithm, so a recorded result stays interpretable if the backend is swapped.
inline const char* to_optz_algorithm_name(const NumericalOptzAlgorithm numerical_optz_algorithm) {
    switch (numerical_optz_algorithm) {
        case NumericalOptzAlgorithm::Sqp:
            return "nlopt_slsqp";
        case NumericalOptzAlgorithm::Ccsaq:
            return "nlopt_ccsaq";
        case NumericalOptzAlgorithm::End:
            break;
    }
    throw ErrorNotImplemented("to_optz_algorithm_name: unknown NumericalOptzAlgorithm.");
}

// What travels through NLopt's void* into the callback. The exception slot exists because NLopt flattens every
// exception into a bare std::runtime_error("nlopt failure"), so an error must travel out of band to keep its type.
struct NloptCallbackPayload {
    const LocalObjective* neg_mll_objective;  // non-owning; the solve below outlives the callback
    std::exception_ptr pending_error;         // the objective's own failure, held until the solve unwinds
};

// NLopt's C `func` callback, whose parameters carry NLopt's own names rather than the package's, so the signature
// can be read against its documentation. `x` is a no-copy view over NLopt's buffer; the gradient cannot be, since
// the objective fills a fresh vector that is then copied into NLopt's.
inline double calc_objective_for_nlopt(const unsigned n, const double* x, double* grad, void* data) {
    NloptCallbackPayload& payload = *static_cast<NloptCallbackPayload*>(data);
    const arma::vec x_vec(const_cast<double*>(x), n, /*copy_aux_mem=*/false, /*strict=*/true);
    try {
        if (grad) {
            arma::vec grad_vec;
            const double objective_val = (*payload.neg_mll_objective)(x_vec, &grad_vec);
            // The objective fills this, so an under-filled vector would be read past its allocation: ARMA_NO_DEBUG
            // removes the bounds check that would otherwise catch it.
            check_vec_len(grad_vec, n, "grad_vec", "calc_objective_for_nlopt", "The objective must fill every entry.");
            std::copy(grad_vec.memptr(), grad_vec.memptr() + n, grad);

            return objective_val;
        }
        return (*payload.neg_mll_objective)(x_vec, nullptr);
    } catch (...) {
        // Recorded before NLopt flattens it, then rethrown so the search stops rather than spending the rest of
        // its evaluation budget.
        if (!payload.pending_error) payload.pending_error = std::current_exception();
        throw;
    }
}

// Minimize `neg_mll_objective` from `ln_hp_init_vec` inside [lb_vec, ub_vec], which NLopt enforces as native bound
// constraints, reading +/-inf as "no bound". Convergence is by relative step and relative objective change.
//
// Fills every LocalOptzResult field, so its postcondition runs here.
inline LocalOptzResult run_local_optz(
        const LocalObjective& neg_mll_objective,
        const arma::vec& ln_hp_init_vec,
        const arma::vec& lb_vec,
        const arma::vec& ub_vec,
        const OptzOptions& optz_opts) {
    // At least one, since the layout sizes the vector from a validated n_dim.
    const unsigned n_hp_optz = static_cast<unsigned>(ln_hp_init_vec.n_elem);
    LocalOptzResult local_optz_result;

    nlopt::opt opt(to_nlopt_algorithm(optz_opts.numerical_optz_algorithm), n_hp_optz);
    NloptCallbackPayload payload{&neg_mll_objective, nullptr};
    opt.set_min_objective(&calc_objective_for_nlopt, &payload);
    opt.set_lower_bounds(std::vector<double>(lb_vec.begin(), lb_vec.end()));
    opt.set_upper_bounds(std::vector<double>(ub_vec.begin(), ub_vec.end()));
    opt.set_xtol_rel(optz_opts.rel_step_tol);
    opt.set_ftol_rel(optz_opts.rel_obj_tol);
    opt.set_maxeval(optz_opts.max_evals);

    std::vector<double> ln_hp_nlopt_vec(ln_hp_init_vec.begin(), ln_hp_init_vec.end());
    double objective_val_nlopt = std::numeric_limits<double>::infinity();
    nlopt::result nlopt_result = nlopt::FAILURE;
    bool is_roundoff_limited = false;
    try {
        nlopt_result = opt.optimize(ln_hp_nlopt_vec, objective_val_nlopt);
    } catch (const nlopt::roundoff_limited&) {
        // NLopt stopped at its best point; nlopt_result stays FAILURE, so the reason is recorded here.
        is_roundoff_limited = true;
    } catch (const nlopt::forced_stop&) {
        // Nothing here asks NLopt to stop, so reaching this is a defect rather than a bad start.
        if (payload.pending_error) std::rethrow_exception(payload.pending_error);
        throw;
    } catch (const std::runtime_error& nlopt_error) {
        // Where the objective raised, restore the original so it reaches the caller with its type intact.
        if (payload.pending_error) std::rethrow_exception(payload.pending_error);
        // Otherwise it is NLopt's own failure, already flattened to runtime_error: this start is numerically
        // unusable, which is the one recoverable outcome, so it is named as that here.
        throw ErrorNumericalFailure(
                std::string("run_local_optz: the optimizer failed at this start (") + nlopt_error.what() + ").");
    } catch (...) {
        // A malformed problem or an allocation failure: a defect, not a bad start.
        if (payload.pending_error) std::rethrow_exception(payload.pending_error);
        throw;
    }

    local_optz_result.ln_hp_vec = arma::vec(ln_hp_nlopt_vec.data(), n_hp_optz);  // copy out of the std::vector
    // Taken from the search rather than re-evaluated, which would cost a factorization. Valid on the
    // roundoff-limited path too, since NLopt assigns before it throws.
    local_optz_result.objective_val = objective_val_nlopt;
    local_optz_result.n_eval = opt.get_numevals();
    local_optz_result.stop_reason =
            is_roundoff_limited ? OptzStopReason::RoundoffLimited : to_optz_stop_reason(nlopt_result);

    GPFLASH_CHECK(check_local_optz_result(local_optz_result, "run_local_optz"));
    return local_optz_result;
}

}  // namespace detail
}  // namespace gpflash
