/**
 * ActiveBounds.hpp -- which constraint bounds are holding a finished solve, and what follows from that.
 *
 * A bound is active when the solve ended against it, so the optimizer could not move further that way. Both things
 * here rest on that one question: the flags report it, and the optimality norm drops the gradient components an
 * active bound holds, since the gradient need not vanish where a constraint stops the iterate.
 *
 * An infinite bound means "no bound".
 */
#pragma once

#include <algorithm>  // std::min, std::max
#include <armadillo>  // arma::vec, arma::uvec
#include <cmath>      // std::sqrt

#include "gpflash/core/Errors.hpp"  // is_finite_val, check_vec_len

namespace gpflash {

// A box constraint counts as active within this fraction of the bound's span.
constexpr double AT_BOUND_REL_TOL = 1.0e-6;

namespace detail {

// Which bound holds this component: -1 the lower, +1 the upper, 0 neither. Both functions below share it, so the
// tolerance cannot drift between them and disagree about which bound is active.
//
// Infinite is the only non-finite case: an unconstrained fit carries +/-infinity in every component, and a finite
// pair is the logarithm of one validated as 0 < lb < ub, so it always has an interior.
inline int calc_active_bound_side(const double ln_hp, const double lb, const double ub) {
    if (!is_finite_val(lb) || !is_finite_val(ub)) return 0;

    // The relative tolerance in this component's own units, which is what makes one constant serve every bound.
    const double at_bound_abs_tol = AT_BOUND_REL_TOL * (ub - lb);
    if (ln_hp <= lb + at_bound_abs_tol) return -1;
    if (ln_hp >= ub - at_bound_abs_tol) return 1;
    return 0;
}

// Per-component: 1 where the point sits against a bound, 0 where unconstrained. Diagnostic, not an error.
inline arma::uvec calc_at_bound_flags_vec(
        const arma::vec& ln_hp_vec, const arma::vec& lb_vec, const arma::vec& ub_vec) {
    const char* caller = "calc_at_bound_flags_vec";
    check_vec_len(lb_vec, ln_hp_vec.n_elem, "lb_vec", caller, "One bound per optimized entry.");
    check_vec_len(ub_vec, ln_hp_vec.n_elem, "ub_vec", caller, "One bound per optimized entry.");

    arma::uvec flags_vec(ln_hp_vec.n_elem, arma::fill::zeros);
    for (arma::uword i_hp_optz = 0; i_hp_optz < ln_hp_vec.n_elem; ++i_hp_optz)
        if (calc_active_bound_side(ln_hp_vec(i_hp_optz), lb_vec(i_hp_optz), ub_vec(i_hp_optz)) != 0)
            flags_vec(i_hp_optz) = 1;

    return flags_vec;
}

// Distance from the box-constrained (KKT) first-order conditions: ||g|| with the components pushing outward
// through an active bound removed, since the gradient need not vanish where a constraint holds the iterate. With
// no bound active this is ||g||.
inline double calc_projected_gradient_norm(
        const arma::vec& ln_hp_vec, const arma::vec& grad_vec, const arma::vec& lb_vec, const arma::vec& ub_vec) {
    const char* caller = "calc_projected_gradient_norm";
    check_vec_len(grad_vec, ln_hp_vec.n_elem, "grad_vec", caller, "One gradient entry per optimized entry.");
    check_vec_len(lb_vec, ln_hp_vec.n_elem, "lb_vec", caller, "One bound per optimized entry.");
    check_vec_len(ub_vec, ln_hp_vec.n_elem, "ub_vec", caller, "One bound per optimized entry.");

    double sum_sq = 0.0;
    for (arma::uword i_hp_optz = 0; i_hp_optz < ln_hp_vec.n_elem; ++i_hp_optz) {
        double grad_i = grad_vec(i_hp_optz);
        // Minimizing, so a feasible descent step needs grad < 0 at a lower bound and grad > 0 at an upper one.
        // The opposite sign is held by the constraint.
        const int bound_side = calc_active_bound_side(ln_hp_vec(i_hp_optz), lb_vec(i_hp_optz), ub_vec(i_hp_optz));
        if (bound_side < 0)
            grad_i = std::min(grad_i, 0.0);
        else if (bound_side > 0)
            grad_i = std::max(grad_i, 0.0);
        sum_sq += grad_i * grad_i;
    }

    return std::sqrt(sum_sq);
}

}  // namespace detail
}  // namespace gpflash
