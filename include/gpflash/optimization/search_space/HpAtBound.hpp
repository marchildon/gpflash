/**
 * HpAtBound.hpp -- which hyperparameters the fit left on a box bound. One field per HpVal field.
 */
#pragma once

#include <algorithm>  // std::max
#include <armadillo>  // arma::uvec, the per-block flag vectors
#include <cstddef>    // std::size_t
#include <string>     // the summary below

#include "gpflash/core/Errors.hpp"                             // the field checks below compose
#include "gpflash/core/Platform.hpp"                           // GPFLASH_CHECK
#include "gpflash/optimization/search_space/HpOptzLayout.hpp"  // HpOptzLayout, whose offsets to_hp_at_bound slices against

namespace gpflash {

struct HpAtBound {
    arma::uvec hp_gamma_at_bound_vec;       // (n_dim), every kernel
    arma::uvec hp_kern_extra_at_bound_vec;  // (n_hp_kern_extra), empty for a kernel with no extras

    bool hp_sig2_k_at_bound = false;      // false where the block was not searched
    bool hp_sig2_f_val_at_bound = false;  // likewise

    // arma::any is false on an empty vector, which is what a kernel with no extras carries.
    bool has_any() const {
        return hp_sig2_k_at_bound || hp_sig2_f_val_at_bound || arma::any(hp_gamma_at_bound_vec)
                || arma::any(hp_kern_extra_at_bound_vec);
    }
};

namespace detail {

// Postcondition. n_hp_kern_extra comes from the kernel, which this tier cannot reach.
inline void check_hp_at_bound(
        const HpAtBound& hp_at_bound_struct, const int n_dim, const int n_hp_kern_extra, const char* caller) {
    check_vec_len(hp_at_bound_struct.hp_gamma_at_bound_vec, n_dim, "HpAtBound::hp_gamma_at_bound_vec", caller);
    check_vec_len(
            hp_at_bound_struct.hp_kern_extra_at_bound_vec,
            n_hp_kern_extra,
            "HpAtBound::hp_kern_extra_at_bound_vec",
            caller);
}

// Slice a flat flag vector into the named blocks.
inline HpAtBound to_hp_at_bound(const HpOptzLayout& hp_optz_layout, const arma::uvec& flags_vec) {
    check_vec_len(flags_vec, hp_optz_layout.n_hp_optz(), "flags_vec", "to_hp_at_bound", "One flag per searched entry.");

    HpAtBound hp_at_bound_struct;
    hp_at_bound_struct.hp_gamma_at_bound_vec = to_hp_block_vec(hp_optz_layout, flags_vec, HpBlock::Gamma);
    hp_at_bound_struct.hp_kern_extra_at_bound_vec = to_hp_block_vec(hp_optz_layout, flags_vec, HpBlock::KernExtra);
    if (hp_optz_layout.is_searched(HpBlock::Sig2K))
        hp_at_bound_struct.hp_sig2_k_at_bound = flags_vec(hp_optz_layout.offset(HpBlock::Sig2K)) != 0;
    if (hp_optz_layout.is_searched(HpBlock::Sig2FVal))
        hp_at_bound_struct.hp_sig2_f_val_at_bound = flags_vec(hp_optz_layout.offset(HpBlock::Sig2FVal)) != 0;

    GPFLASH_CHECK(check_hp_at_bound(
            hp_at_bound_struct, hp_optz_layout.n_dim, hp_optz_layout.n_hp_kern_extra, "to_hp_at_bound"));
    return hp_at_bound_struct;
}

// Which positions in a 0/1 flag vector are set: {0, 1, 0, 1} gives "1, 3", all-clear gives "none", empty gives
// "n/a" -- for the extras block, that the kernel has none rather than that none is pinned.
inline std::string to_at_bound_index_string(const arma::uvec& at_bound_flags_vec) {
    if (at_bound_flags_vec.is_empty()) return "n/a";

    std::string index_str;
    for (arma::uword i_flag = 0; i_flag < at_bound_flags_vec.n_elem; ++i_flag) {
        if (at_bound_flags_vec(i_flag) == 0) continue;
        if (!index_str.empty()) index_str += ", ";
        index_str += std::to_string(i_flag);
    }

    return index_str.empty() ? "none" : index_str;
}

}  // namespace detail

// Block names are padded to this width so the answers line up.
constexpr int SUMMARY_BLOCK_NAME_WIDTH = 14;

// One line per block, named as in HpOptzLayout. A vector block lists the pinned indices; a scalar says yes/no.
// The lines are built as strings, so an index list of any length reaches the report whole.
inline std::string to_summary_string(const HpAtBound& hp_at_bound_struct) {
    const std::string hp_gamma_str = detail::to_at_bound_index_string(hp_at_bound_struct.hp_gamma_at_bound_vec);
    const std::string hp_kern_extra_str =
            detail::to_at_bound_index_string(hp_at_bound_struct.hp_kern_extra_at_bound_vec);
    const std::string hp_sig2_k_str = hp_at_bound_struct.hp_sig2_k_at_bound ? "yes" : "no";
    const std::string hp_sig2_f_val_str = hp_at_bound_struct.hp_sig2_f_val_at_bound ? "yes" : "no";

    std::string summary_str;
    for (int i_block = 0; i_block < N_HP_BLOCK; ++i_block) {
        const HpBlock hp_block = static_cast<HpBlock>(i_block);
        const std::string& answer_str = (hp_block == HpBlock::Gamma) ? hp_gamma_str
                : (hp_block == HpBlock::KernExtra)                   ? hp_kern_extra_str
                : (hp_block == HpBlock::Sig2K)                       ? hp_sig2_k_str
                                                                     : hp_sig2_f_val_str;
        const std::string name_str = to_hp_block_name(hp_block);
        const int n_pad = std::max(0, SUMMARY_BLOCK_NAME_WIDTH - static_cast<int>(name_str.size()));
        summary_str +=
                "  " + name_str + std::string(static_cast<std::size_t>(n_pad), ' ') + " at bound  " + answer_str + "\n";
    }

    return summary_str;
}

}  // namespace gpflash
