/**
 * What a posterior query returns. A single-query call has n_post == 1.
 */
#pragma once

#include <armadillo>

#include "gpflash/core/Errors.hpp"  // the field checks below compose

namespace gpflash {

struct PosteriorResult {
    // The values. Each is empty unless its PosteriorOptions flag asked for it.
    arma::vec mean_post_vec;  // (n_post)
    arma::vec var_post_vec;   // (n_post)

    // Their derivatives with respect to the query point, one gradient per column.
    arma::mat d_mean_post_d_x_post_ndim_npost_mat;
    arma::mat d_var_post_d_x_post_ndim_npost_mat;
};

namespace detail {

// Postcondition. Each field is filled per its PosteriorOptions flag, so an empty
// one means it was not requested and every check below passes it.
inline void check_posterior_result(const PosteriorResult& posterior_result, const char* caller) {
    check_all_finite_vals(posterior_result.mean_post_vec, "PosteriorResult::mean_post_vec", caller);

    // A variance, clamped at zero by PosteriorEval.hpp, so this catches a NaN, which both clamps keep.
    check_all_non_negative_vals(posterior_result.var_post_vec, "PosteriorResult::var_post_vec", caller);

    check_all_finite_vals(
            posterior_result.d_mean_post_d_x_post_ndim_npost_mat,
            "PosteriorResult::d_mean_post_d_x_post_ndim_npost_mat",
            caller);
    check_all_finite_vals(
            posterior_result.d_var_post_d_x_post_ndim_npost_mat,
            "PosteriorResult::d_var_post_d_x_post_ndim_npost_mat",
            caller);
}

}  // namespace detail

}  // namespace gpflash
