/**
 * EvalVariances.hpp -- the two variances an MLL evaluation reads, already resolved.
 */
#pragma once

#include "gpflash/core/Errors.hpp"  // the field checks check_eval_variances composes

namespace gpflash {
namespace detail {

// All variances
struct EvalVariances {
    // The noise variance in force, always >= 0. select_sig2_f_val reads the sign-encoded request on
    // TrainingData::sig2_f_val and returns the value here; this is a separate field, not a rewrite of that one.
    double sig2_f_val = 0.0;

    // Signal variance, from a fit.
    //   noise-free (sig2_f_val == 0): not read -- its closed form supplies it
    //   noisy      (sig2_f_val  > 0): the optimizer's trial value, or known variance
    double hp_sig2_k = -1.0;
};

// sig2_f_val's invariant. hp_sig2_k is checked where the noise case is known, in check_eval_mll_inputs.
inline void check_eval_variances(const EvalVariances& eval_variances, const char* caller) {
    // A resolved variance, so non-negative. This is what rejects a raw TrainingData::sig2_f_val copied in
    // without going through select_sig2_f_val, since a request for an estimate is negative.
    check_finite_non_negative_val(eval_variances.sig2_f_val, "EvalVariances::sig2_f_val", caller);
}

}  // namespace detail
}  // namespace gpflash
