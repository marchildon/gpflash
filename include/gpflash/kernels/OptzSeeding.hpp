/**
 * OptzSeeding.hpp -- where a kernel wants its hyperparameter fit to start and sample, in natural space.
 *
 * Every field unset (the default) means the shared heuristic start and the whole constraint region.
 * build_optz_region clamps whatever a kernel supplies into the constraint region.
 */
#pragma once

#include <armadillo>  // arma::vec
#include <string>     // std::to_string

#include "gpflash/core/Errors.hpp"  // the field checks

namespace gpflash {

struct OptzSeeding {
    arma::vec hp_gamma_start_vec;  // (n_dim), or empty for the default

    // Restart sampling band for hp_gamma. Both or neither; empty samples the whole constraint region.
    arma::vec hp_gamma_sample_lb_vec;  // (n_dim)
    arma::vec hp_gamma_sample_ub_vec;  // (n_dim)

    // Extras sampled in [init / factor, init * factor] around OptzOptions::hp_kern_extra_init. 0 = whole region.
    double hp_kern_extra_sample_factor = 0.0;
};

namespace detail {

// Precondition. Widths need n_dim and are checked by build_optz_region.
inline void check_optz_seeding(const OptzSeeding& optz_seeding, const char* caller) {
    // Positive: natural-space hyperparameters, searched in log space. Empty passes.
    check_all_positive_vals(optz_seeding.hp_gamma_start_vec, "OptzSeeding::hp_gamma_start_vec", caller);
    check_all_positive_vals(optz_seeding.hp_gamma_sample_lb_vec, "OptzSeeding::hp_gamma_sample_lb_vec", caller);
    check_all_positive_vals(optz_seeding.hp_gamma_sample_ub_vec, "OptzSeeding::hp_gamma_sample_ub_vec", caller);

    if (optz_seeding.hp_gamma_sample_lb_vec.n_elem != optz_seeding.hp_gamma_sample_ub_vec.n_elem)
        throw ErrorInvalidInput(
                error_prefix(caller, "OptzSeeding::hp_gamma_sample_lb_vec")
                + " and hp_gamma_sample_ub_vec must have the same length: give both corners of the sampling band, or "
                  "neither.");
    // Equal corners are allowed: the band collapses to a point.
    for (arma::uword i_dim = 0; i_dim < optz_seeding.hp_gamma_sample_lb_vec.n_elem; ++i_dim)
        if (optz_seeding.hp_gamma_sample_ub_vec(i_dim) < optz_seeding.hp_gamma_sample_lb_vec(i_dim))
            throw ErrorInvalidInput(
                    error_prefix(caller, "OptzSeeding::hp_gamma_sample_lb_vec") + " exceeds hp_gamma_sample_ub_vec at ["
                    + std::to_string(i_dim) + "]: the lower corner comes first.");

    // 1 collapses the band to a point, below 1 inverts it.
    if (optz_seeding.hp_kern_extra_sample_factor != 0.0)
        check_finite_above_val(
                optz_seeding.hp_kern_extra_sample_factor, 1.0, "OptzSeeding::hp_kern_extra_sample_factor", caller);
}

}  // namespace detail
}  // namespace gpflash
