/**
 * PreconScales.hpp -- the preconditioner P = diag(sqrt(diag(M))) of a symmetric positive-semidefinite matrix M with
 * positive diagonal, and its derived quantities.
 *
 * Every vector is empty where the caller skips preconditioning because M is already unit-diagonal, so P = I. Each
 * scaling reads that as "nothing to apply."
 */
#pragma once

#include <armadillo>

#include "gpflash/core/Errors.hpp"  // the field checks below compose

namespace gpflash {
namespace detail {

struct PreconScales {
    // All (n_row), the order of M, or all empty where P = I.
    arma::vec M_diag_vec;            // W = diag(M) = P P
    arma::vec precon_scale_vec;      // diag(P) = sqrt(W)
    arma::vec inv_precon_scale_vec;  // 1 / diag(P)

    // Whether the caller declared M unit-diagonal, so no scaling was applied and P = I.
    bool is_identity() const { return M_diag_vec.is_empty(); }
};

// Postcondition.
inline void check_precon_scales(const PreconScales& precon_scales, const char* caller) {
    // Diagonals of a positive-definite matrix and their square roots, so positive where present. Empty passes: P = I.
    check_all_positive_vals(precon_scales.M_diag_vec, "PreconScales::M_diag_vec", caller);
    check_all_positive_vals(precon_scales.precon_scale_vec, "PreconScales::precon_scale_vec", caller);
    check_all_positive_vals(precon_scales.inv_precon_scale_vec, "PreconScales::inv_precon_scale_vec", caller);
}

}  // namespace detail
}  // namespace gpflash
