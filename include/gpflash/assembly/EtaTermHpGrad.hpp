/**
 * EtaTermHpGrad.hpp -- what the nugget eta contributes to the hyperparameter gradient.
 */
#pragma once

#include <armadillo>

#include "gpflash/core/Errors.hpp"  // the field checks below compose

namespace gpflash {
namespace detail {

struct EtaTermHpGrad {
    // One block per hyperparameter group, matching HpVal's layout.
    arma::vec d_mll_d_hp_gamma_vec;       // (n_dim)
    arma::vec d_mll_d_hp_kern_extra_vec;  // (n_hp_kern_extra), empty for kernels that have none
};

// Postcondition. Derivatives take either sign, so finiteness is the whole invariant.
inline void check_eta_term_hp_grad(const EtaTermHpGrad& eta_term_hp_grad, const char* caller) {
    check_all_finite_vals(eta_term_hp_grad.d_mll_d_hp_gamma_vec, "EtaTermHpGrad::d_mll_d_hp_gamma_vec", caller);
    check_all_finite_vals(
            eta_term_hp_grad.d_mll_d_hp_kern_extra_vec, "EtaTermHpGrad::d_mll_d_hp_kern_extra_vec", caller);
}

}  // namespace detail
}  // namespace gpflash
