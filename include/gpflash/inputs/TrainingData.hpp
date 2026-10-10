/**
 * TrainingData.hpp -- what was measured. Fixed for the duration of a fit.
 */
#pragma once

#include <armadillo>
#include <string>

#include "gpflash/core/Errors.hpp"      // the field checks check_training_data composes
#include "gpflash/core/NoiseModel.hpp"  // NoiseRequest, classify_noise_request

namespace gpflash {

// ── What was measured. Fixed for the duration of a fit. ──────────────────────
struct TrainingData {
    // The points and what was measured at them.
    arma::mat X_ndim_ntrain_mat;  // one training point per column
    arma::vec f_train_vec;        // (n_train) function evaluations at those points

    // Noise variance on the f evaluations. Its sign says whether that variance is zero, known, or unknown:
    //   sig2_f_val = 0    zero     noise-free data; sigma_k^2 gets its closed form
    //   sig2_f_val > 0    known    this variance, held fixed through the fit
    //   sig2_f_val < 0    unknown  fit estimates it; read it back from the fit's HpVal::hp_sig2_f_val
    double sig2_f_val = 0.0;
};

namespace detail {

inline void check_training_data(const TrainingData& train_data, const char* caller) {
    check_mat_not_empty(
            train_data.X_ndim_ntrain_mat,
            "TrainingData::X_ndim_ntrain_mat",
            caller,
            "Expected (n_dim, n_train), one training point per column.");
    check_vec_len(
            train_data.f_train_vec,
            train_data.X_ndim_ntrain_mat.n_cols,
            "TrainingData::f_train_vec",
            caller,
            "One evaluation per column of X.");
    check_all_finite_vals(train_data.X_ndim_ntrain_mat, "TrainingData::X_ndim_ntrain_mat", caller);
    check_all_finite_vals(train_data.f_train_vec, "TrainingData::f_train_vec", caller);
    // Any sign is legal here; negative requests an estimate.
    check_finite_val(train_data.sig2_f_val, "TrainingData::sig2_f_val", caller);
}

// Decodes the sign rule on sig2_f_val above, returning the noise variance that is used, finite and >= 0.
// It rejects a known variance on the data alongside an estimate in the hyperparameters.
//
// hp_sig2_f_val_estimated is HpVal::hp_sig2_f_val, taken loose so this header need not depend on that peer
// struct. It is read only where sig2_f_val asked for an estimate.
inline double select_sig2_f_val(
        const TrainingData& train_data, const double hp_sig2_f_val_estimated, const char* caller) {
    check_finite_val(train_data.sig2_f_val, "TrainingData::sig2_f_val", caller);
    check_finite_val(hp_sig2_f_val_estimated, "HpVal::hp_sig2_f_val", caller);

    // An estimate was requested, so one must be present.
    if (classify_noise_request(train_data.sig2_f_val) == NoiseRequest::Estimate) {
        if (hp_sig2_f_val_estimated > 0.0) return hp_sig2_f_val_estimated;
        throw ErrorInvalidInput(
                std::string("gpflash::") + caller + ": TrainingData::sig2_f_val = "
                + std::to_string(train_data.sig2_f_val)
                + " asks for an estimated noise, but HpVal::hp_sig2_f_val holds none. Pass the hyperparameters "
                  "from select_hp_with_max_mll.");
    }

    // NoiseFree or Fixed: sig2_f_val supplies the noise, so the hyperparameters must not also carry one.
    if (hp_sig2_f_val_estimated > 0.0)
        throw ErrorInvalidInput(
                std::string("gpflash::") + caller
                + ": TrainingData::sig2_f_val = " + std::to_string(train_data.sig2_f_val)
                + " supplies the noise, but HpVal::hp_sig2_f_val = " + std::to_string(hp_sig2_f_val_estimated)
                + " also holds an estimate. It is one or the other, never both.");
    return train_data.sig2_f_val;
}

}  // namespace detail
}  // namespace gpflash
