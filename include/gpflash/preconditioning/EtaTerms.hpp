/**
 * EtaTerms.hpp -- the nugget eta of a matrix M and every term used to build or differentiate it.
 *
 * build_eta_terms in ConditionedCholesky.hpp preconditions M -> M_precon, takes the selected bound on lambda_max,
 * applies cond_max, and optionally records the active row needed by the hyperparameter gradient.
 */
#pragma once

#include <armadillo>

#include "gpflash/core/Errors.hpp"                   // the field checks below compose
#include "gpflash/preconditioning/PreconScales.hpp"  // PreconScales

namespace gpflash {
namespace detail {

struct EtaTerms {
    // The bound on lambda_max(M_precon) the EigMaxUbMethod selects, then the target and the nugget it sizes. The
    // bound is 1 + max_i sum_{j != i} |M_precon_ij| for Gershgorin, n_row for the trace.
    double eig_max_ub_M_precon = 0.0;
    double cond_max = 0.0;  // model target used to size eta
    double eta = 0.0;       // eig_max_ub_M_precon / (cond_max - 1)
    // The row attaining the bound, -1 where n_row <= 1 left it unmeasured or the trace was used. The row itself is
    // gathered only for the hyperparameter gradient.
    int idx_M_precon_max_abs_row_sum = -1;
    arma::vec active_row_M_precon_vec;  // (n_row) zero at the active row's own entry
    // The preconditioning that took M to M_precon, needed to undo it.
    PreconScales precon_scales;
};

// Postcondition.
inline void check_eta_terms(const EtaTerms& eta_terms, const char* caller) {
    // calc_eta divides by cond_max - 1.
    check_finite_above_val(eta_terms.cond_max, 1.0, "EtaTerms::cond_max", caller);
    check_finite_non_negative_val(eta_terms.eta, "EtaTerms::eta", caller);
    check_finite_non_negative_val(eta_terms.eig_max_ub_M_precon, "EtaTerms::eig_max_ub_M_precon", caller);

    // Empty where the gradient did not ask for it or the bound has no active row; the index is checked against it
    // only when present.
    if (!eta_terms.active_row_M_precon_vec.is_empty())
        check_idx_or_inactive(
                eta_terms.idx_M_precon_max_abs_row_sum,
                static_cast<int>(eta_terms.active_row_M_precon_vec.n_elem),
                "EtaTerms::idx_M_precon_max_abs_row_sum",
                caller);
    check_all_finite_vals(eta_terms.active_row_M_precon_vec, "EtaTerms::active_row_M_precon_vec", caller);

    check_precon_scales(eta_terms.precon_scales, caller);
}

}  // namespace detail
}  // namespace gpflash
