/**
 * HpOptzLayout.hpp -- which hyperparameters a fit searches, and where each sits in the optimizer's flat vector.
 *
 * One block per HpBlock, in enum order, contiguous from 0. An unsearched block has width 0 and no entry:
 *
 *   Gamma      hp_gamma, one per input dimension
 *   KernExtra  the kernel's extras, if it declares any
 *   Sig2K      sigma_k^2, unless the data is noise-free (then closed form)
 *   Sig2FVal   sigma_f^2, only when the noise is estimated
 *
 * Widths and offsets are computed from the three stored fields. Append to HpBlock, never insert: a recorded fit is
 * sliced back against these offsets. The packing is in natural space.
 */
#pragma once

#include <armadillo>  // arma::vec
#include <string>     // std::string

#include "gpflash/core/Errors.hpp"      // the field checks
#include "gpflash/core/NoiseModel.hpp"  // NoiseRequest
#include "gpflash/core/Platform.hpp"    // GPFLASH_CHECK
#include "gpflash/inputs/HpVal.hpp"     // HpVal

namespace gpflash {

// The hyperparameter groups, in the order they occupy the flat vector.
enum class HpBlock {
    Gamma,
    KernExtra,
    Sig2K,
    Sig2FVal,
    End,  // one past the last, for loops over the blocks
};

constexpr int N_HP_BLOCK = static_cast<int>(HpBlock::End);

struct HpOptzLayout {
    int n_dim = 0;
    int n_hp_kern_extra = 0;
    NoiseRequest noise_request = NoiseRequest::NoiseFree;

    // Entries the block occupies; 0 when not searched.
    int width(const HpBlock hp_block) const {
        switch (hp_block) {
            case HpBlock::Gamma:
                return n_dim;
            case HpBlock::KernExtra:
                return n_hp_kern_extra;
            case HpBlock::Sig2K:
                return (noise_request != NoiseRequest::NoiseFree) ? 1 : 0;
            case HpBlock::Sig2FVal:
                return (noise_request == NoiseRequest::Estimate) ? 1 : 0;
            case HpBlock::End:
                break;
        }
        return 0;
    }

    bool is_searched(const HpBlock hp_block) const { return width(hp_block) > 0; }

    // First entry of the block, or -1 when not searched.
    int offset(const HpBlock hp_block) const {
        if (!is_searched(hp_block)) return -1;

        int i_hp_optz = 0;
        for (int i_block = 0; i_block < static_cast<int>(hp_block); ++i_block)
            i_hp_optz += width(static_cast<HpBlock>(i_block));

        return i_hp_optz;
    }

    // Length of the flat vector: the searched hyperparameters only.
    int n_hp_optz() const {
        int n_hp_optz_sum = 0;
        for (int i_block = 0; i_block < N_HP_BLOCK; ++i_block) n_hp_optz_sum += width(static_cast<HpBlock>(i_block));

        return n_hp_optz_sum;
    }
};

// The block name: the HpVal field without its _vec suffix. For messages and report columns.
inline const char* to_hp_block_name(const HpBlock hp_block) {
    switch (hp_block) {
        case HpBlock::Gamma:
            return "hp_gamma";
        case HpBlock::KernExtra:
            return "hp_kern_extra";
        case HpBlock::Sig2K:
            return "hp_sig2_k";
        case HpBlock::Sig2FVal:
            return "hp_sig2_f_val";
        case HpBlock::End:
            break;
    }
    return "unknown";
}

// Name of one entry: "hp_gamma[2]", or "hp_sig2_k" for a 1-wide block.
inline std::string to_hp_entry_name(const HpOptzLayout& hp_optz_layout, const int i_hp_optz) {
    for (int i_block = 0; i_block < N_HP_BLOCK; ++i_block) {
        const HpBlock hp_block = static_cast<HpBlock>(i_block);
        const int idx_block = hp_optz_layout.offset(hp_block);
        if (idx_block < 0 || i_hp_optz < idx_block || i_hp_optz >= idx_block + hp_optz_layout.width(hp_block)) continue;

        const std::string block_name_str = to_hp_block_name(hp_block);

        return (hp_optz_layout.width(hp_block) == 1)
                ? block_name_str
                : block_name_str + "[" + std::to_string(i_hp_optz - idx_block) + "]";
    }

    return "entry " + std::to_string(i_hp_optz);
}

namespace detail {

// Precondition: both widths come from the caller's data and kernel.
inline void check_hp_optz_layout(const HpOptzLayout& hp_optz_layout, const char* caller) {
    check_positive_count(hp_optz_layout.n_dim, "HpOptzLayout::n_dim", caller, "Every kernel fits one per dimension.");
    check_non_negative_count(hp_optz_layout.n_hp_kern_extra, "HpOptzLayout::n_hp_kern_extra", caller);
    check_enum_val(hp_optz_layout.noise_request, "NoiseRequest", "HpOptzLayout::noise_request", caller);
}

inline HpOptzLayout build_hp_optz_layout(
        const int n_dim, const int n_hp_kern_extra, const NoiseRequest noise_request, const char* caller) {
    HpOptzLayout hp_optz_layout;
    hp_optz_layout.n_dim = n_dim;
    hp_optz_layout.n_hp_kern_extra = n_hp_kern_extra;
    hp_optz_layout.noise_request = noise_request;

    check_hp_optz_layout(hp_optz_layout, caller);
    return hp_optz_layout;
}

// ── Between named hyperparameters and the flat vector ────────────────────────

// Pack four blocks into the flat vector, skipping unsearched ones. Also packs a gradient (MllOptzObjective.hpp).
// Every extent comes from the layout, so the width checks are postconditions on the caller's slicing.
inline arma::vec to_hp_optz_vec_from_blocks(
        const HpOptzLayout& hp_optz_layout,
        const arma::vec& gamma_block_vec,
        const arma::vec& kern_extra_block_vec,
        const double sig2_k,
        const double sig2_f_val) {
    const int n_gamma = hp_optz_layout.width(HpBlock::Gamma);
    const int n_kern_extra = hp_optz_layout.width(HpBlock::KernExtra);
    // The caller is named inline: a local would be unused in a build without GPFLASH_DEBUG_CHECKS.
    GPFLASH_CHECK(check_vec_len(gamma_block_vec, n_gamma, "gamma_block_vec", "to_hp_optz_vec_from_blocks"));
    GPFLASH_CHECK(
            check_vec_len(kern_extra_block_vec, n_kern_extra, "kern_extra_block_vec", "to_hp_optz_vec_from_blocks"));

    arma::vec hp_optz_vec(hp_optz_layout.n_hp_optz());
    hp_optz_vec.head(n_gamma) = gamma_block_vec;
    if (hp_optz_layout.is_searched(HpBlock::KernExtra)) {
        const arma::uword i_first = static_cast<arma::uword>(hp_optz_layout.offset(HpBlock::KernExtra));
        hp_optz_vec.subvec(i_first, i_first + static_cast<arma::uword>(n_kern_extra) - 1) = kern_extra_block_vec;
    }
    if (hp_optz_layout.is_searched(HpBlock::Sig2K)) hp_optz_vec(hp_optz_layout.offset(HpBlock::Sig2K)) = sig2_k;
    if (hp_optz_layout.is_searched(HpBlock::Sig2FVal))
        hp_optz_vec(hp_optz_layout.offset(HpBlock::Sig2FVal)) = sig2_f_val;

    return hp_optz_vec;
}

inline arma::vec to_hp_optz_vec(const HpOptzLayout& hp_optz_layout, const HpVal& hp_val_struct) {
    return to_hp_optz_vec_from_blocks(
            hp_optz_layout,
            hp_val_struct.hp_gamma_vec,
            hp_val_struct.hp_kern_extra_vec,
            hp_val_struct.hp_sig2_k,
            hp_val_struct.hp_sig2_f_val);
}

// One block of a flat vector; empty when not searched. Works for vec and uvec.
template <class VecType>
inline VecType to_hp_block_vec(const HpOptzLayout& hp_optz_layout, const VecType& hp_optz_vec, const HpBlock hp_block) {
    if (!hp_optz_layout.is_searched(hp_block)) return VecType();

    const arma::uword i_first = static_cast<arma::uword>(hp_optz_layout.offset(hp_block));

    return hp_optz_vec.subvec(i_first, i_first + static_cast<arma::uword>(hp_optz_layout.width(hp_block)) - 1);
}

// Inverse of to_hp_optz_vec. An unsearched scalar keeps HpVal's negative sentinel.
inline HpVal to_hp_val(const HpOptzLayout& hp_optz_layout, const arma::vec& hp_optz_vec) {
    HpVal hp_val_struct;
    hp_val_struct.hp_gamma_vec = to_hp_block_vec(hp_optz_layout, hp_optz_vec, HpBlock::Gamma);
    hp_val_struct.hp_kern_extra_vec = to_hp_block_vec(hp_optz_layout, hp_optz_vec, HpBlock::KernExtra);
    if (hp_optz_layout.is_searched(HpBlock::Sig2K))
        hp_val_struct.hp_sig2_k = hp_optz_vec(hp_optz_layout.offset(HpBlock::Sig2K));
    if (hp_optz_layout.is_searched(HpBlock::Sig2FVal))
        hp_val_struct.hp_sig2_f_val = hp_optz_vec(hp_optz_layout.offset(HpBlock::Sig2FVal));

    return hp_val_struct;
}

}  // namespace detail
}  // namespace gpflash
