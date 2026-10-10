/**
 * LatinHypercube.hpp -- a Latin-hypercube design over the unit cube, which is where multistart gets its initial
 * points.
 *
 * Each axis is cut into n_sample equal strata, one point lands in each, and the strata are paired across axes by an
 * independent random permutation per axis, so every one-dimensional projection is uniform by construction. Nothing
 * constrains the joint spacing.
 *
 * Reproducible given the seed, on every platform: the design is built from the generator's raw output, which the
 * standard fixes, rather than from std::shuffle or a std:: distribution, which it does not. Every MPI process builds
 * the whole design from the same seed, so they agree without communicating, on a heterogeneous cluster too.
 *
 * Nothing here names a hyperparameter: the caller maps the cube onto whatever box it is sampling.
 */
#pragma once

#include <cstddef>  // std::size_t
#include <cstdint>  // std::uint64_t
#include <limits>   // std::numeric_limits
#include <numeric>  // std::iota
#include <random>   // std::mt19937_64
#include <utility>  // std::swap
#include <vector>   // one permutation, reused across axes

#include <armadillo>  // arma::mat, the design

namespace gpflash {
namespace detail {

// Uniform on [0, 1) from the top 53 bits of one raw draw. std::mt19937_64's output sequence is fixed by the
// standard and dividing a 53-bit integer by 2^53 is exact, so the value is the same in every standard library.
inline double draw_unit_uniform_portable(std::mt19937_64& rng) {
    return static_cast<double>(static_cast<std::uint64_t>(rng()) >> 11) / 9007199254740992.0;
}

// Uniform integer in [0, n_bound), n_bound >= 1, by rejection: draws at or above the largest multiple of n_bound
// that fits are redrawn, so there is no modulo bias. How many draws a call consumes varies with the values drawn,
// but the sequence decides it, so every platform consumes the same draws.
inline std::uint64_t draw_index_below_portable(std::mt19937_64& rng, const std::uint64_t n_bound) {
    const std::uint64_t draw_max = std::numeric_limits<std::uint64_t>::max();
    const std::uint64_t accept_limit = draw_max - draw_max % n_bound;  // a multiple of n_bound
    std::uint64_t draw = static_cast<std::uint64_t>(rng());
    while (draw >= accept_limit) draw = static_cast<std::uint64_t>(rng());
    return draw % n_bound;
}

// Fisher-Yates from the back: position i swaps with a uniform index in [0, i]. Unlike std::shuffle, the draws it
// consumes are decided here, not by the standard library.
inline void shuffle_portable(std::vector<int>& permutation_vec, std::mt19937_64& rng) {
    for (std::size_t i_pos = permutation_vec.size(); i_pos > 1; --i_pos) {
        const std::size_t i_swap = draw_index_below_portable(rng, i_pos);
        std::swap(permutation_vec[i_pos - 1], permutation_vec[i_swap]);
    }
}

// n_sample points in [0,1]^n_dim_sample, one point per row. Both counts must be >= 1, which the caller establishes:
// the sample count comes from validated options and the axis count from a checked layout.
//
// One pass over each column, which is the whole cost: O(n_sample n_dim_sample) with one permutation buffer reused.
// The axis loop is the outer one so the writes run down a column, Armadillo being column-major.
inline arma::mat build_latin_hypercube_unit_cube(const int n_sample, const int n_dim_sample, std::mt19937_64& rng) {
    arma::mat unit_cube_mat(n_sample, n_dim_sample);

    std::vector<int> permutation_vec(static_cast<std::size_t>(n_sample));
    for (int i_dim_sample = 0; i_dim_sample < n_dim_sample; ++i_dim_sample) {
        // Which stratum each point takes on this axis. Permuted independently per axis, which is what decouples
        // the axes; one shared permutation would place every point on a diagonal.
        std::iota(permutation_vec.begin(), permutation_vec.end(), 0);
        shuffle_portable(permutation_vec, rng);

        // One add and one divide, both correctly rounded, so the value is the same on every target.
        for (int i_sample = 0; i_sample < n_sample; ++i_sample)
            unit_cube_mat(i_sample, i_dim_sample) =
                    (permutation_vec[static_cast<std::size_t>(i_sample)] + draw_unit_uniform_portable(rng)) / n_sample;
    }

    return unit_cube_mat;
}

}  // namespace detail
}  // namespace gpflash
