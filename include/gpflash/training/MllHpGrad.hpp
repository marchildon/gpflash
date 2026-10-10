/**
 * MllHpGrad.hpp -- the MLL's derivative with respect to each hyperparameter.
 *
 * One derivative per HpVal field. The posterior also reads HpVal, so the MLL-specific meaning belongs here
 * rather than on that shared input type. Populated only when a gradient was requested.
 */
#pragma once

#include <armadillo>

#include "gpflash/core/Errors.hpp"  // the field checks below compose

namespace gpflash {

struct MllHpGrad {
    // One block per hyperparameter group, matching HpVal's layout.
    arma::vec d_mll_d_hp_gamma_vec;       // (n_dim), every kernel
    arma::vec d_mll_d_hp_kern_extra_vec;  // (n_hp_kern_extra) in the kernel's order, empty when it has none

    // The variances. d_mll_d_hp_sig2_k is zero on noise-free data, where sigma_k^2 takes its closed form.
    double d_mll_d_hp_sig2_k = 0.0;
    double d_mll_d_hp_sig2_f_val = 0.0;
};

namespace detail {

// Postcondition. A derivative takes either sign, so finiteness is all that holds of every entry.
inline void check_mll_hp_grad(const MllHpGrad& mll_hp_grad_struct, const char* caller) {
    check_all_finite_vals(mll_hp_grad_struct.d_mll_d_hp_gamma_vec, "MllHpGrad::d_mll_d_hp_gamma_vec", caller);
    check_all_finite_vals(mll_hp_grad_struct.d_mll_d_hp_kern_extra_vec, "MllHpGrad::d_mll_d_hp_kern_extra_vec", caller);

    check_finite_val(mll_hp_grad_struct.d_mll_d_hp_sig2_k, "MllHpGrad::d_mll_d_hp_sig2_k", caller);
    check_finite_val(mll_hp_grad_struct.d_mll_d_hp_sig2_f_val, "MllHpGrad::d_mll_d_hp_sig2_f_val", caller);
}

}  // namespace detail
}  // namespace gpflash
