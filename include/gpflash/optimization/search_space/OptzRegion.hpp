/**
 * OptzRegion.hpp -- where the search looks, in log space: the constraint box, the box restarts are sampled from, and
 * the start point. One entry per searched hyperparameter, in HpOptzLayout order.
 *
 * build_optz_region resolves everything in natural space from OptzOptions and the kernel's OptzSeeding, then takes
 * the logarithm. An infinite constraint bound means "no bound" (use_bounds = false).
 */
#pragma once

#include <armadillo>  // arma::vec, arma::stddev, arma::log
#include <limits>     // std::numeric_limits, the infinities an unconstrained fit resolves to
#include <string>     // std::string

#include "gpflash/core/Errors.hpp"                             // check_vec_len, check_all_finite_vals, is_finite_val
#include "gpflash/inputs/HpVal.hpp"                            // HpVal, the named form each corner is read into
#include "gpflash/inputs/TrainingData.hpp"                     // TrainingData
#include "gpflash/kernels/OptzSeeding.hpp"                     // OptzSeeding, check_optz_seeding
#include "gpflash/optimization/search_space/HpOptzLayout.hpp"  // HpOptzLayout, to_hp_optz_vec, to_hp_entry_name
#include "gpflash/optimization/search_space/OptzOptions.hpp"   // OptzOptions

namespace gpflash {

// All in log space. Public because MllOptzResult carries the region a fit was held to.
struct OptzRegion {
    arma::vec constraint_lb_vec;  // the optimizer may not leave this box; +/-inf when unconstrained
    arma::vec constraint_ub_vec;

    arma::vec sample_lb_vec;  // restart points are drawn from this box; inside the constraint box, always finite
    arma::vec sample_ub_vec;

    arma::vec start_vec;  // the first optimization's start; inside the constraint box, always finite
};

namespace detail {

// ── Checks ───────────────────────────────────────────────────────────────────

inline void check_ln_finite_vec(
        const arma::vec& hp_vec, const int n_hp_optz, const char* field_name, const char* caller) {
    check_vec_len(hp_vec, n_hp_optz, field_name, caller, "One entry per searched hyperparameter.");
    check_all_finite_vals(hp_vec, field_name, caller);
}

// Infinite means "no bound", so only NaN is rejected.
inline void check_ln_constraint_vec(
        const arma::vec& bound_vec, const int n_hp_optz, const char* field_name, const char* caller) {
    check_vec_len(bound_vec, n_hp_optz, field_name, caller, "One entry per searched hyperparameter.");
    if (bound_vec.has_nan()) throw ErrorInvalidInput(error_prefix(caller, field_name) + " contains NaN.");
}

// Precondition: every number traces back to OptzOptions, the data or the kernel's seeding. Widths first, since the
// loop below indexes into all five vectors.
inline void check_optz_region(const OptzRegion& optz_region, const HpOptzLayout& hp_optz_layout, const char* caller) {
    const int n_hp_optz = hp_optz_layout.n_hp_optz();
    check_ln_constraint_vec(optz_region.constraint_lb_vec, n_hp_optz, "OptzRegion::constraint_lb_vec", caller);
    check_ln_constraint_vec(optz_region.constraint_ub_vec, n_hp_optz, "OptzRegion::constraint_ub_vec", caller);
    check_ln_finite_vec(optz_region.sample_lb_vec, n_hp_optz, "OptzRegion::sample_lb_vec", caller);
    check_ln_finite_vec(optz_region.sample_ub_vec, n_hp_optz, "OptzRegion::sample_ub_vec", caller);
    check_ln_finite_vec(optz_region.start_vec, n_hp_optz, "OptzRegion::start_vec", caller);

    // Sampling corners and start must lie inside the constraint box (closed: clamping lands on a bound).
    const char* clamp_hint =
            "Everything the region carries is clamped into it, so a value outside means a skipped "
            "clamp.";
    for (arma::uword i_hp_optz = 0; i_hp_optz < optz_region.constraint_lb_vec.n_elem; ++i_hp_optz) {
        const double constraint_lb = optz_region.constraint_lb_vec(i_hp_optz);
        const double constraint_ub = optz_region.constraint_ub_vec(i_hp_optz);
        check_finite_in_closed_range_val(
                optz_region.sample_lb_vec(i_hp_optz),
                constraint_lb,
                constraint_ub,
                "OptzRegion::sample_lb_vec",
                caller,
                clamp_hint);
        check_finite_in_closed_range_val(
                optz_region.sample_ub_vec(i_hp_optz),
                constraint_lb,
                constraint_ub,
                "OptzRegion::sample_ub_vec",
                caller,
                clamp_hint);
        check_finite_in_closed_range_val(
                optz_region.start_vec(i_hp_optz),
                constraint_lb,
                constraint_ub,
                "OptzRegion::start_vec",
                caller,
                clamp_hint);
    }
}

// ── Resolving the corners in natural space ───────────────────────────────────

// Clamp every entry into [constraint_lb, constraint_ub].
inline void fill_within_constraint_inplace(
        const arma::vec& constraint_lb_vec, const arma::vec& constraint_ub_vec, arma::vec& hp_optz_vec) {
    hp_optz_vec = arma::min(arma::max(hp_optz_vec, constraint_lb_vec), constraint_ub_vec);
}

// The variance options are multiples of f_train_residual_var, the variance the mean model leaves in f_train
// (calc_least_squares_residual_var), since a variance carries the units of f_train^2.
inline double calc_variance_scale(const double f_train_residual_var, const OptzOptions& optz_opts) {
    if (!optz_opts.scale_variance_bounds_by_residual_var) return 1.0;
    // No residual: use the options as absolute.
    return (f_train_residual_var > 0.0 && is_finite_val(f_train_residual_var)) ? f_train_residual_var : 1.0;
}

// A single entry broadcasts over the block; one per component passes through. check_hp_bound_vec established the
// length is 1 or n_hp_in_block.
inline arma::vec calc_hp_bound_block_vec(const arma::vec& bound_vec, const int n_hp_in_block) {
    if (n_hp_in_block == 0) return arma::vec();  // a kernel with no extras
    if (static_cast<int>(bound_vec.n_elem) == n_hp_in_block) return bound_vec;

    arma::vec bound_block_vec(n_hp_in_block);
    bound_block_vec.fill(bound_vec(0));

    return bound_block_vec;
}

// The constraint box in natural space. Both ends in one function so a field read into the wrong corner shows as a
// mismatched pair. An unsearched bound is set and never packed.
inline void fill_constraint_bound_natural_vecs(
        const OptzOptions& optz_opts,
        const HpOptzLayout& hp_optz_layout,
        const double var_scale,
        arma::vec& constraint_lb_natural_vec_out,
        arma::vec& constraint_ub_natural_vec_out) {
    const int n_dim = hp_optz_layout.n_dim;
    const int n_hp_kern_extra = hp_optz_layout.n_hp_kern_extra;

    HpVal lb_hp_val;
    HpVal ub_hp_val;
    lb_hp_val.hp_gamma_vec = calc_hp_bound_block_vec(optz_opts.hp_gamma_lb_vec, n_dim);
    ub_hp_val.hp_gamma_vec = calc_hp_bound_block_vec(optz_opts.hp_gamma_ub_vec, n_dim);
    lb_hp_val.hp_kern_extra_vec = calc_hp_bound_block_vec(optz_opts.hp_kern_extra_lb_vec, n_hp_kern_extra);
    ub_hp_val.hp_kern_extra_vec = calc_hp_bound_block_vec(optz_opts.hp_kern_extra_ub_vec, n_hp_kern_extra);
    lb_hp_val.hp_sig2_k = optz_opts.hp_sig2_k_lb * var_scale;
    ub_hp_val.hp_sig2_k = optz_opts.hp_sig2_k_ub * var_scale;
    lb_hp_val.hp_sig2_f_val = optz_opts.hp_sig2_f_val_lb * var_scale;
    ub_hp_val.hp_sig2_f_val = optz_opts.hp_sig2_f_val_ub * var_scale;

    constraint_lb_natural_vec_out = to_hp_optz_vec(hp_optz_layout, lb_hp_val);
    constraint_ub_natural_vec_out = to_hp_optz_vec(hp_optz_layout, ub_hp_val);
}

// hp_gamma_d = 1 / stddev(X_d); a constant dimension falls back to 1. Measured from the kernel's center, so the start
// is the same wherever the data sits.
inline arma::vec calc_hp_gamma_start_vec(const arma::mat& X_ndim_ntrain_mat, const arma::vec& x_center_vec) {
    const int n_dim = static_cast<int>(X_ndim_ntrain_mat.n_rows);
    arma::vec hp_gamma_start_vec(n_dim);
    for (int i_dim = 0; i_dim < n_dim; ++i_dim) {
        const double spread = arma::stddev(X_ndim_ntrain_mat.row(i_dim) - x_center_vec(i_dim));
        hp_gamma_start_vec(i_dim) = (spread > 0.0) ? 1.0 / spread : 1.0;
    }

    return hp_gamma_start_vec;
}

// The heuristic start: hp_gamma from the data, every other block from its _init option. Unclamped.
inline arma::vec calc_start_natural_vec(
        const TrainingData& train_data,
        const arma::vec& x_center_vec,
        const OptzOptions& optz_opts,
        const HpOptzLayout& hp_optz_layout,
        const double var_scale) {
    HpVal start_hp_val;
    start_hp_val.hp_gamma_vec = calc_hp_gamma_start_vec(train_data.X_ndim_ntrain_mat, x_center_vec);
    start_hp_val.hp_kern_extra_vec.set_size(hp_optz_layout.n_hp_kern_extra);
    start_hp_val.hp_kern_extra_vec.fill(optz_opts.hp_kern_extra_init);
    start_hp_val.hp_sig2_k = optz_opts.hp_sig2_k_init * var_scale;
    start_hp_val.hp_sig2_f_val = optz_opts.hp_sig2_f_val_init * var_scale;

    return to_hp_optz_vec(hp_optz_layout, start_hp_val);
}

// Write the kernel's seeding over the shared start and sampling corners. Only the blocks it names are touched.
inline void fill_seeded_natural_inplace(
        const OptzSeeding& optz_seeding,
        const OptzOptions& optz_opts,
        const HpOptzLayout& hp_optz_layout,
        arma::vec& start_natural_vec,
        arma::vec& sample_lb_natural_vec,
        arma::vec& sample_ub_natural_vec,
        const char* caller) {
    const int n_dim = hp_optz_layout.n_dim;
    const char* len_hint = "A kernel seeds the whole hp_gamma block, or none of it.";

    if (!optz_seeding.hp_gamma_start_vec.is_empty()) {
        check_vec_len(optz_seeding.hp_gamma_start_vec, n_dim, "OptzSeeding::hp_gamma_start_vec", caller, len_hint);
        start_natural_vec.head(n_dim) = optz_seeding.hp_gamma_start_vec;
    }

    if (!optz_seeding.hp_gamma_sample_lb_vec.is_empty()) {
        check_vec_len(
                optz_seeding.hp_gamma_sample_lb_vec, n_dim, "OptzSeeding::hp_gamma_sample_lb_vec", caller, len_hint);
        check_vec_len(
                optz_seeding.hp_gamma_sample_ub_vec, n_dim, "OptzSeeding::hp_gamma_sample_ub_vec", caller, len_hint);
        sample_lb_natural_vec.head(n_dim) = optz_seeding.hp_gamma_sample_lb_vec;
        sample_ub_natural_vec.head(n_dim) = optz_seeding.hp_gamma_sample_ub_vec;
    }

    // The extras band is [init / factor, init * factor].
    if (optz_seeding.hp_kern_extra_sample_factor != 0.0 && hp_optz_layout.is_searched(HpBlock::KernExtra)) {
        const arma::uword i_first = static_cast<arma::uword>(hp_optz_layout.offset(HpBlock::KernExtra));
        const arma::uword i_last = i_first + static_cast<arma::uword>(hp_optz_layout.width(HpBlock::KernExtra)) - 1;
        sample_lb_natural_vec.subvec(i_first, i_last)
                .fill(optz_opts.hp_kern_extra_init / optz_seeding.hp_kern_extra_sample_factor);
        sample_ub_natural_vec.subvec(i_first, i_last)
                .fill(optz_opts.hp_kern_extra_init * optz_seeding.hp_kern_extra_sample_factor);
    }
}

// The first ordering check on the hp_gamma and hp_kern_extra bounds, entry by entry once broadcast. In natural
// space, so the message names the caller's numbers.
inline void check_constraint_bound_natural_vecs(
        const arma::vec& constraint_lb_natural_vec,
        const arma::vec& constraint_ub_natural_vec,
        const HpOptzLayout& hp_optz_layout,
        const char* caller) {
    const char* hint = "A single-entry bound is broadcast over its block before the two corners are compared.";
    for (arma::uword i_hp_optz = 0; i_hp_optz < constraint_lb_natural_vec.n_elem; ++i_hp_optz) {
        const std::string entry_name = to_hp_entry_name(hp_optz_layout, static_cast<int>(i_hp_optz));
        const std::string lb_field_name = entry_name + " lower bound";
        const std::string ub_field_name = entry_name + " upper bound";
        check_bound_pair_val(
                constraint_lb_natural_vec(i_hp_optz),
                constraint_ub_natural_vec(i_hp_optz),
                lb_field_name.c_str(),
                ub_field_name.c_str(),
                caller,
                hint);
    }
}

// ── Natural space to log space ───────────────────────────────────────────────

// Constraint box, then sampling box and start, then the kernel's seeding, each clamped into the box, then the log.
// use_bounds is applied after the log, so an unconstrained fit still starts and samples where the options say.
inline OptzRegion build_optz_region(
        const TrainingData& train_data,
        const arma::vec& x_center_vec,
        const double f_train_residual_var,
        const OptzOptions& optz_opts,
        const OptzSeeding& optz_seeding,
        const HpOptzLayout& hp_optz_layout,
        const char* caller) {
    check_optz_seeding(optz_seeding, caller);
    const double var_scale = calc_variance_scale(f_train_residual_var, optz_opts);

    arma::vec constraint_lb_natural_vec;
    arma::vec constraint_ub_natural_vec;
    fill_constraint_bound_natural_vecs(
            optz_opts, hp_optz_layout, var_scale, constraint_lb_natural_vec, constraint_ub_natural_vec);
    check_constraint_bound_natural_vecs(constraint_lb_natural_vec, constraint_ub_natural_vec, hp_optz_layout, caller);

    // Sample the whole constraint box unless the kernel narrows it.
    arma::vec sample_lb_natural_vec = constraint_lb_natural_vec;
    arma::vec sample_ub_natural_vec = constraint_ub_natural_vec;
    arma::vec start_natural_vec =
            calc_start_natural_vec(train_data, x_center_vec, optz_opts, hp_optz_layout, var_scale);
    fill_seeded_natural_inplace(
            optz_seeding,
            optz_opts,
            hp_optz_layout,
            start_natural_vec,
            sample_lb_natural_vec,
            sample_ub_natural_vec,
            caller);

    fill_within_constraint_inplace(constraint_lb_natural_vec, constraint_ub_natural_vec, sample_lb_natural_vec);
    fill_within_constraint_inplace(constraint_lb_natural_vec, constraint_ub_natural_vec, sample_ub_natural_vec);
    fill_within_constraint_inplace(constraint_lb_natural_vec, constraint_ub_natural_vec, start_natural_vec);

    OptzRegion optz_region;
    optz_region.sample_lb_vec = arma::log(sample_lb_natural_vec);
    optz_region.sample_ub_vec = arma::log(sample_ub_natural_vec);
    optz_region.start_vec = arma::log(start_natural_vec);
    if (optz_opts.use_bounds) {
        optz_region.constraint_lb_vec = arma::log(constraint_lb_natural_vec);
        optz_region.constraint_ub_vec = arma::log(constraint_ub_natural_vec);
    } else {
        optz_region.constraint_lb_vec =
                arma::vec(hp_optz_layout.n_hp_optz()).fill(-std::numeric_limits<double>::infinity());
        optz_region.constraint_ub_vec =
                arma::vec(hp_optz_layout.n_hp_optz()).fill(std::numeric_limits<double>::infinity());
    }

    check_optz_region(optz_region, hp_optz_layout, caller);
    return optz_region;
}

}  // namespace detail
}  // namespace gpflash
