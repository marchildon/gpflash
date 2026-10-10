/**
 * MllOptzResult.hpp -- what the hyperparameter fit returns: the fit, plus the search that produced it.
 *
 * Built in one place (build_mll_optz_result in HpOptimizer.hpp). Everything derivable from the search is an
 * accessor, not a field.
 */
#pragma once

#include <string>  // std::string

#include "gpflash/core/Errors.hpp"                               // the field checks
#include "gpflash/core/StringFormat.hpp"                         // to_formatted_string
#include "gpflash/inputs/HpVal.hpp"                              // HpVal, check_hp_val
#include "gpflash/inputs/ModelOptions.hpp"                       // ModelOptions, check_model_options
#include "gpflash/optimization/multistart/MultistartResult.hpp"  // MultistartResult, check_multistart_result
#include "gpflash/optimization/search_space/ActiveBounds.hpp"    // calc_at_bound_flags_vec
#include "gpflash/optimization/search_space/HpAtBound.hpp"       // HpAtBound, to_hp_at_bound
#include "gpflash/optimization/search_space/HpOptzLayout.hpp"    // HpOptzLayout
#include "gpflash/optimization/search_space/OptzRegion.hpp"      // OptzRegion, check_optz_region

namespace gpflash {

struct MllOptzResult {
    // The fit.
    HpVal hp_val_struct;         // every fitted hyperparameter, closed-form ones included
    ModelOptions model_options;  // the options the fit ran with
    double optimality = -1.0;    // projected gradient norm at the winner

    MultistartResult multistart_result;  // the search
    OptzRegion optz_region;              // the log-space region the search was held to

    const HpOptzLayout& hp_optz_layout() const { return multistart_result.hp_optz_layout; }

    // Forwarded from the search.
    int n_ind_optz_run() const { return multistart_result.n_ind_optz_run(); }
    int idx_best() const { return multistart_result.idx_best(); }
    LocalOptzResult build_best_local_optz_result() const { return multistart_result.build_best_local_optz_result(); }
    double mll() const { return multistart_result.mll(); }
    int calc_n_failed_start() const { return multistart_result.calc_n_failed_start(); }
    int n_total_mll_eval() const { return multistart_result.total_start_field_count(StartField::NMllEval); }
    int n_total_mll_grad_eval() const { return multistart_result.total_start_field_count(StartField::NMllGradEval); }

    // Which entries optimization i_ind_optz left on a constraint bound, 1 or 0 each.
    arma::uvec calc_at_bound_flags_vec(const int i_ind_optz) const {
        return detail::calc_at_bound_flags_vec(
                multistart_result.ln_hp_vec(i_ind_optz), optz_region.constraint_lb_vec, optz_region.constraint_ub_vec);
    }

    HpAtBound calc_hp_at_bound_struct(const int i_ind_optz) const {
        return detail::to_hp_at_bound(hp_optz_layout(), calc_at_bound_flags_vec(i_ind_optz));
    }

    HpAtBound calc_hp_at_bound_struct() const { return calc_hp_at_bound_struct(idx_best()); }

    int calc_n_at_bound(const int i_ind_optz) const {
        return static_cast<int>(arma::accu(calc_at_bound_flags_vec(i_ind_optz)));
    }
};

namespace detail {

// Postcondition.
inline void check_mll_optz_result(const MllOptzResult& mll_optz_result, const char* caller) {
    check_multistart_result(mll_optz_result.multistart_result, caller);

    check_optz_region(mll_optz_result.optz_region, mll_optz_result.hp_optz_layout(), caller);
    check_model_options(mll_optz_result.model_options, caller);
    check_finite_non_negative_val(mll_optz_result.optimality, "MllOptzResult::optimality", caller);
    check_hp_val(
            mll_optz_result.hp_val_struct,
            mll_optz_result.hp_optz_layout().n_dim,
            mll_optz_result.hp_optz_layout().n_hp_kern_extra,
            caller);
    // A finished fit always resolves the signal variance, searched or closed-form, so it is never left unset.
    check_supplied_hp_sig2_k(mll_optz_result.hp_val_struct, caller);
}

}  // namespace detail

// Short fit report; ends with HpAtBound's own summary.
inline std::string to_summary_string(const MllOptzResult& mll_optz_result, const std::string& label = "") {
    // First, so an empty record raises in idx_best before the tallies below index its rows.
    const LocalOptzResult best_local_optz_result = mll_optz_result.build_best_local_optz_result();
    const double mll = mll_optz_result.mll();

    const std::string body_str = detail::to_formatted_string(
            "  marginal log lik   %.6g\n"
            "  optimality         %.6g\n"
            "  stopped on         %s\n"
            "  ind optz run       %d\n"
            "  failed starts      %d\n"
            "  evals (best)       %d\n"
            "  MLL evals          %d\n"
            "  MLL+grad evals     %d\n",
            mll,
            mll_optz_result.optimality,
            to_stop_reason_name(best_local_optz_result.stop_reason),
            mll_optz_result.n_ind_optz_run(),
            mll_optz_result.calc_n_failed_start(),
            best_local_optz_result.n_eval,
            mll_optz_result.n_total_mll_eval(),
            mll_optz_result.n_total_mll_grad_eval());

    const std::string summary_str = "GpFlash fit" + (label.empty() ? std::string() : ": " + label) + "\n" + body_str
            + to_summary_string(mll_optz_result.calc_hp_at_bound_struct());

    return summary_str;
}

}  // namespace gpflash
