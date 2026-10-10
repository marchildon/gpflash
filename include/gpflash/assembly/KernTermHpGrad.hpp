/**
 * KernTermHpGrad.hpp -- what the kernel matrix contributes to the hyperparameter gradient.
 *
 * Returned by calc_kern_term_hp_grad. The active row travels with the derivatives because the eta term needs
 * it and would otherwise gather it a second time.
 */
#pragma once

#include <armadillo>

#include "gpflash/core/Errors.hpp"  // the field checks below compose

namespace gpflash {
namespace detail {

struct KernTermHpGrad {
    // One block per hyperparameter group, matching HpVal's layout.
    arma::vec d_mll_d_hp_gamma_vec;       // (n_dim)
    arma::vec d_mll_d_hp_kern_extra_vec;  // (n_hp_kern_extra), empty for kernels that have none

    // Used to calculate the derivative of eta.
    arma::vec active_row_kern_common_rdot_vec;         // (n_train)
    arma::vec active_row_kern_common_rdot_mirror_vec;  // (n_train), empty without symmetry or an active row
};

// Postcondition. Derivatives take either sign, so finiteness is the whole
// invariant.
inline void check_kern_term_hp_grad(const KernTermHpGrad& kern_term_hp_grad, const char* caller) {
    check_all_finite_vals(kern_term_hp_grad.d_mll_d_hp_gamma_vec, "KernTermHpGrad::d_mll_d_hp_gamma_vec", caller);
    check_all_finite_vals(
            kern_term_hp_grad.d_mll_d_hp_kern_extra_vec, "KernTermHpGrad::d_mll_d_hp_kern_extra_vec", caller);
    check_all_finite_vals(
            kern_term_hp_grad.active_row_kern_common_rdot_vec,
            "KernTermHpGrad::active_row_kern_common_rdot_vec",
            caller);
    check_all_finite_vals(
            kern_term_hp_grad.active_row_kern_common_rdot_mirror_vec,
            "KernTermHpGrad::active_row_kern_common_rdot_mirror_vec",
            caller);
}

}  // namespace detail
}  // namespace gpflash
