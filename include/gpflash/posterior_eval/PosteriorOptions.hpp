/**
 * Which parts of a posterior query to compute, and whether to check its query points.
 */
#pragma once

namespace gpflash {

// One flag per PosteriorResult field, independently selectable, then one for the input checks. Either variance
// quantity pays the triangular solve and the two gradients share one pass; unselected work is skipped, not discarded.
struct PosteriorOptions {
    // The values, both on by default.
    bool return_mean = true;
    bool return_var = true;

    // Their derivatives with respect to the query point, off by default. The variance's costs a further solve.
    bool return_mean_gradient = false;
    bool return_var_gradient = false;

    // Whether to check the query points' row count and finiteness, on by default. Clearing it saves a scan of
    // n_dim values per point.
    bool check_query_points = true;
};

namespace detail {

// Whether no quantity was asked for, whatever check_query_points says. Not an error: a query then returns an empty
// PosteriorResult, which is what every field already means when its flag is clear.
inline bool is_empty_request(const PosteriorOptions& post_opts) {
    return !post_opts.return_mean && !post_opts.return_var && !post_opts.return_mean_gradient
            && !post_opts.return_var_gradient;
}

}  // namespace detail
}  // namespace gpflash
