/**
 * HpVal.hpp -- the hyperparameters, named.
 */
#pragma once

#include <armadillo>

#include "gpflash/core/Errors.hpp"  // the field checks check_hp_val composes

namespace gpflash {

// ── The hyperparameters, named. ──────────────────────────────────────────────
// Scalars default negative: every hyperparameter is positive, so a required-but-unset one is
// rejected rather than taken as plausible. Unset vectors are empty, which fails the length check.
struct HpVal {
    // The variances, both outputs of a fit.
    double hp_sig2_k = -1.0;      // signal variance, required by the posterior
    double hp_sig2_f_val = -1.0;  // noise variance, set only where TrainingData::sig2_f_val < 0 asked for it

    // The kernel's own blocks.
    arma::vec hp_gamma_vec;       // (n_dim) inverse length scales, or inverse periods for Periodic
    arma::vec hp_kern_extra_vec;  // (n_hp_kern_extra) whatever the kernel declares, or empty
};

namespace detail {

// The two blocks are checked separately so no split of a single combined length passes.
inline void check_hp_val(const HpVal& hp_val_struct, const int n_dim, const int n_hp_kern_extra, const char* caller) {
    check_vec_len(
            hp_val_struct.hp_gamma_vec,
            n_dim,
            "HpVal::hp_gamma_vec",
            caller,
            "An empty one usually means a default-constructed HpVal was passed.");
    check_vec_len(
            hp_val_struct.hp_kern_extra_vec,
            n_hp_kern_extra,
            "HpVal::hp_kern_extra_vec",
            caller,
            "The count is what this kernel declares.");
    check_all_positive_vals(hp_val_struct.hp_gamma_vec, "HpVal::hp_gamma_vec", caller);
    check_all_positive_vals(
            hp_val_struct.hp_kern_extra_vec,
            "HpVal::hp_kern_extra_vec",
            caller,
            "Every kernel extra is a shape or amplitude, so zero usually means it was left unset.");
}

// hp_sig2_k is read only where no closed form supplies it, so this is not part of check_hp_val.
inline void check_supplied_hp_sig2_k(const HpVal& hp_val_struct, const char* caller) {
    check_finite_positive_val(
            hp_val_struct.hp_sig2_k, "HpVal::hp_sig2_k", caller, "A negative one means it was never set.");
}

// The extras block as a raw pointer for the hot loops, null where the kernel has none.
// check_hp_val pins the length to the kernel's n_hp_kern_extra, so empty means none.
inline const double* calc_hp_kern_extra_ptr(const arma::vec& hp_kern_extra_vec) {
    return hp_kern_extra_vec.is_empty() ? nullptr : hp_kern_extra_vec.memptr();
}

// Leaves hp_sig2_f_val at its sentinel, so the result carries no noise estimate.
inline HpVal build_hp_val(const arma::vec& hp_gamma_vec, const arma::vec& hp_kern_extra_vec, const double hp_sig2_k) {
    HpVal hp_val_struct;
    hp_val_struct.hp_gamma_vec = hp_gamma_vec;
    hp_val_struct.hp_kern_extra_vec = hp_kern_extra_vec;
    hp_val_struct.hp_sig2_k = hp_sig2_k;
    return hp_val_struct;
}

}  // namespace detail
}  // namespace gpflash
