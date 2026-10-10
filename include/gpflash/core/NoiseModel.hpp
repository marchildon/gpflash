/**
 * NoiseModel.hpp -- how TrainingData::sig2_f_val's sign selects the noise case, and the one function that reads it.
 */
#pragma once

namespace gpflash {

// The three noise cases, selected by the sign of TrainingData::sig2_f_val.
enum class NoiseRequest {
    NoiseFree,  // sig2_f_val == 0: no noise term, and hp_sig2_k then has a closed form
    Fixed,      // sig2_f_val  > 0: this known variance, held through the fit
    Estimate,   // sig2_f_val  < 0: fit it, and read the result from HpVal::hp_sig2_f_val
    End,        // one past the last, for check_enum_val
};

namespace detail {

inline NoiseRequest classify_noise_request(const double sig2_f_val) {
    if (sig2_f_val < 0.0) return NoiseRequest::Estimate;
    if (sig2_f_val > 0.0) return NoiseRequest::Fixed;
    return NoiseRequest::NoiseFree;
}

}  // namespace detail
}  // namespace gpflash
