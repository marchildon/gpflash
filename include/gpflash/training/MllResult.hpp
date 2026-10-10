/**
 * MllResult.hpp -- values returned by eval_mll.
 *
 * Always the MLL and the mean model weights. Noise-free evaluations also return the closed-form signal
 * variance. The gradient block is populated only when requested; otherwise it stays default-empty.
 */
#pragma once

#include <armadillo>
#include <string>

#include "gpflash/core/Errors.hpp"         // the field checks below compose
#include "gpflash/inputs/HpVal.hpp"        // HpVal, which fill_closed_form_hp_inplace writes into
#include "gpflash/training/MllHpGrad.hpp"  // MllHpGrad

namespace gpflash {

struct MllResult {
    // What every evaluation returns.
    double mll = 0.0;
    arma::vec hp_beta_vec;  // (n_basis) the mean model weights, in closed form

    // Signal variance, returned only on noise-free data -- the one case with a closed form for it.
    bool hp_sig2_k_from_closed_form = false;
    double hp_sig2_k = 0.0;

    // The gradient, when one was asked for.
    MllHpGrad mll_hp_grad_struct;  // default-empty when no gradient was requested
};

namespace detail {

// Postcondition.
inline void check_mll_result(const MllResult& mll_result, const char* caller) {
    check_finite_val(mll_result.mll, "MllResult::mll", caller);
    check_all_finite_vals(mll_result.hp_beta_vec, "MllResult::hp_beta_vec", caller);

    // A variance, zero where the caller supplied it rather than the closed form resolving it.
    check_finite_non_negative_val(mll_result.hp_sig2_k, "MllResult::hp_sig2_k", caller);

    check_mll_hp_grad(mll_result.mll_hp_grad_struct, caller);
}

// Whatever this evaluation resolved in closed form rather than reading from its inputs, copied into a fitted HpVal.
// Beside the flags that report it, so which hyperparameters have a closed form is stated once here instead of at
// every caller assembling a fit result. A hyperparameter the evaluation did not resolve is left as it was.
inline void fill_closed_form_hp_inplace(const MllResult& mll_result, HpVal& hp_val_struct) {
    if (mll_result.hp_sig2_k_from_closed_form) hp_val_struct.hp_sig2_k = mll_result.hp_sig2_k;
}

}  // namespace detail
}  // namespace gpflash
