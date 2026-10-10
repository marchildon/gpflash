/**
 * MeanModel.hpp -- the mean model: the mean model basis, basis(x), in the prior mean basis(x)^T hp_beta_vec, the
 * checks on it, what a query reads off it, its basis at the training points, and the variance its least-squares fit
 * leaves in f_train.
 *
 * The mean model basis is additive in the input dimensions, with at most one column per dimension and no cross terms:
 *
 *   basis(x) = [1, x_1^degree_1, x_2^degree_2, ...],   n_basis = has_constant + count(degree_d > 0)
 *
 * The constant is present only when has_constant is set, and a dimension of degree 0 has no column. With neither, it
 * is the zero mean model, which has no weights to fit. It reads raw points, not hp_gamma-scaled ones: a basis that
 * moved with a searched hyperparameter would add a term to the MLL gradient.
 *
 * Under a symmetry model every column must have the model's parity about the mirror plane, so that the prior mean has
 * it too. The constant is even, and x_d^p is odd only where the mirror flips x_d and p is odd.
 */
#pragma once

#include <armadillo>
#include <string>

#include "gpflash/core/Errors.hpp"          // ErrorInvalidInput, ErrorNumericalFailure, the field checks
#include "gpflash/inputs/ModelOptions.hpp"  // ModelOptions, MEAN_MODEL_DEGREE_MAX
#include "gpflash/inputs/SymmetrySpec.hpp"  // SymmetrySpec

namespace gpflash {
namespace mean {

struct MeanModel {
    bool has_constant = true;  // whether basis(x) starts with the constant column
    arma::uvec degree_vec;     // (n_dim) the power in each input dimension's one column; 0 for no column

    int n_basis() const {
        const int n_power_col = static_cast<int>(arma::accu(degree_vec > 0));
        return (has_constant ? 1 : 0) + n_power_col;
    }

    // The mean model basis, (n_point, n_basis), one row per point with columns in basis(x)'s order. n_point is n_train
    // or n_post depending on the caller.
    arma::mat calc_basis_mat(const arma::mat& X_ndim_npoint_mat) const {
        arma::mat Basis_npoint_nbasis_mat(X_ndim_npoint_mat.n_cols, static_cast<arma::uword>(n_basis()));

        arma::uword i_basis = 0;
        if (has_constant) {
            Basis_npoint_nbasis_mat.col(0).ones();
            i_basis = 1;
        }

        for (arma::uword i_dim = 0; i_dim < degree_vec.n_elem; ++i_dim) {
            if (degree_vec(i_dim) == 0) continue;

            // x_d^degree_d by repeated products, exact at the degrees MEAN_MODEL_DEGREE_MAX admits.
            const arma::vec x_dim_vec = X_ndim_npoint_mat.row(i_dim).t();
            arma::vec x_power_vec = x_dim_vec;
            for (arma::uword i_power = 2; i_power <= degree_vec(i_dim); ++i_power) x_power_vec %= x_dim_vec;
            Basis_npoint_nbasis_mat.col(i_basis) = x_power_vec;
            ++i_basis;
        }

        return Basis_npoint_nbasis_mat;
    }

    // The prior mean basis(x_post_vec)^T hp_beta_vec at one query point.
    double calc_prior_mean_val(const arma::vec& x_post_vec, const arma::vec& hp_beta_vec) const {
        const int n_basis_val = n_basis();
        const bool is_constant_only = has_constant && n_basis_val == 1;

        if (n_basis_val == 0) {
            return 0.0;
        } else if (is_constant_only) {
            // The constant mean model basis is exactly 1, so the prior mean is its one weight, with no basis to build.
            return hp_beta_vec(0);
        } else {
            const arma::vec basis_post_vec = calc_basis_mat(x_post_vec).t();
            return arma::dot(basis_post_vec, hp_beta_vec);
        }
    }

    // Batch counterpart, one value per column of X_ndim_npost_mat.
    arma::vec calc_prior_mean_val_vec(const arma::mat& X_ndim_npost_mat, const arma::vec& hp_beta_vec) const {
        const arma::uword n_post = X_ndim_npost_mat.n_cols;
        const int n_basis_val = n_basis();
        const bool is_constant_only = has_constant && n_basis_val == 1;

        if (n_basis_val == 0) {
            return arma::zeros<arma::vec>(n_post);
        } else if (is_constant_only) {
            return arma::vec(n_post).fill(hp_beta_vec(0));
        } else {
            const arma::mat Basis_npost_nbasis_mat = calc_basis_mat(X_ndim_npost_mat);
            return Basis_npost_nbasis_mat * hp_beta_vec;
        }
    }

    // Adds d(basis(x)^T hp_beta_vec)/dx at each query point into d_mean_post_d_x_post_ndim_npost_mat.
    // The constant column contributes nothing, so neither does a mean model with every degree zero.
    void add_d_prior_mean_d_x_post_inplace(
            const arma::mat& X_ndim_npost_mat,
            const arma::vec& hp_beta_vec,
            arma::mat& d_mean_post_d_x_post_ndim_npost_mat) const {
        // Walks the columns in calc_basis_mat's order, so i_basis indexes the weight of the column it differentiates.
        arma::uword i_basis = has_constant ? 1 : 0;  // past the constant column, if any
        for (arma::uword i_dim = 0; i_dim < degree_vec.n_elem; ++i_dim) {
            const arma::uword degree = degree_vec(i_dim);
            if (degree == 0) continue;

            const arma::rowvec x_dim_row = X_ndim_npost_mat.row(i_dim);
            arma::rowvec x_power_row(x_dim_row.n_elem, arma::fill::ones);  // x_d^(degree - 1)
            for (arma::uword i_power = 2; i_power <= degree; ++i_power) x_power_row %= x_dim_row;

            // d(x^degree)/dx = degree x^(degree - 1)
            const double derivative_coefficient = static_cast<double>(degree) * hp_beta_vec(i_basis);
            d_mean_post_d_x_post_ndim_npost_mat.row(i_dim) += derivative_coefficient * x_power_row;
            ++i_basis;
        }
    }
};

// Rejects a degree above MEAN_MODEL_DEGREE_MAX.
inline void check_mean_model_degree_vec(
        const arma::uvec& mean_model_degree_vec, const char* field_name, const char* caller) {
    if (mean_model_degree_vec.n_elem > 0 && mean_model_degree_vec.max() > MEAN_MODEL_DEGREE_MAX)
        throw ErrorInvalidInput(
                gpflash::detail::error_prefix(caller, field_name) + " holds a degree of "
                + std::to_string(mean_model_degree_vec.max()) + ", above the maximum of "
                + std::to_string(MEAN_MODEL_DEGREE_MAX) + ".");
}

// Rejects a degree_vec whose length is not n_dim or whose degrees the mean model cannot take.
inline void check_mean_model(const MeanModel& mean_model, const int n_dim, const char* caller) {
    gpflash::detail::check_vec_len(mean_model.degree_vec, n_dim, "MeanModel::degree_vec", caller);
    check_mean_model_degree_vec(mean_model.degree_vec, "MeanModel::degree_vec", caller);
}

// Rejects a mean model basis column without the symmetry model's parity about the mirror plane.
inline void check_mean_model_fits_symmetry(
        const MeanModel& mean_model, const SymmetrySpec& symmetry_spec, const char* caller) {
    const bool is_odd_symmetry = symmetry_spec.is_odd_symmetry();
    const char* model_parity_name = is_odd_symmetry ? "odd" : "even";

    if (mean_model.has_constant && is_odd_symmetry)
        throw ErrorInvalidInput(
                gpflash::detail::error_prefix(caller, "ModelOptions::has_mean_model_constant")
                + " is true, but the constant is even about the mirror plane and the odd symmetry model's prior mean "
                  "is odd. Set has_mean_model_constant to false.");

    for (arma::uword i_dim = 0; i_dim < mean_model.degree_vec.n_elem; ++i_dim) {
        const arma::uword degree = mean_model.degree_vec(i_dim);
        const bool is_flipped = symmetry_spec.mirror_sign_vec(i_dim) < 0.0;
        const bool is_odd_col = is_flipped && degree % 2 == 1;
        const bool is_parity_mismatch = degree > 0 && is_odd_col != is_odd_symmetry;
        if (is_parity_mismatch) {
            std::string fix_hint;
            if (is_flipped) {
                fix_hint = std::string("Use an ") + model_parity_name + " degree, or 0.";
            } else {
                fix_hint = "The mirror preserves that dimension, so its every power is even; set its degree to 0.";
            }
            const char* col_parity_name = is_odd_col ? "odd" : "even";
            throw ErrorInvalidInput(
                    gpflash::detail::error_prefix(caller, "ModelOptions::mean_model_degree_vec") + " gives dimension "
                    + std::to_string(i_dim) + " a degree of " + std::to_string(degree) + ", but x_"
                    + std::to_string(i_dim) + "^" + std::to_string(degree) + " is " + col_parity_name
                    + " about the mirror plane and the " + model_parity_name + " symmetry model's prior mean is "
                    + model_parity_name + ". " + fix_hint);
        }
    }
}

// The mean model model_opts asks for, with one degree per input dimension. The checks on the fields alone run here;
// build_train_basis_mat checks them against the training points.
inline MeanModel make_mean_model(const ModelOptions& model_opts, const int n_dim, const char* caller) {
    const arma::uvec& mean_model_degree_vec = model_opts.mean_model_degree_vec;

    // Reported as ModelOptions::mean_model_degree_vec, the field the caller set. The length check also rejects an empty
    // one.
    if (mean_model_degree_vec.n_elem != 1 && static_cast<int>(mean_model_degree_vec.n_elem) != n_dim)
        throw ErrorInvalidInput(
                gpflash::detail::error_prefix(caller, "ModelOptions::mean_model_degree_vec") + " has "
                + std::to_string(mean_model_degree_vec.n_elem)
                + " entries; give 1 to apply to every input dimension, or " + std::to_string(n_dim)
                + ", one per dimension.");
    check_mean_model_degree_vec(mean_model_degree_vec, "ModelOptions::mean_model_degree_vec", caller);

    // A single degree applies to every dimension.
    MeanModel mean_model;
    mean_model.has_constant = model_opts.has_mean_model_constant;
    if (mean_model_degree_vec.n_elem == 1) {
        mean_model.degree_vec = arma::uvec(n_dim).fill(mean_model_degree_vec(0));
    } else {
        mean_model.degree_vec = mean_model_degree_vec;
    }

    // Matched against the mirror only once both have one entry per input dimension.
    if (model_opts.symmetry_spec.is_on()) {
        gpflash::detail::check_vec_len(
                model_opts.symmetry_spec.mirror_sign_vec,
                static_cast<arma::uword>(n_dim),
                "SymmetrySpec::mirror_sign_vec",
                caller,
                "One sign per input dimension, matching hp_gamma.");
        check_mean_model_fits_symmetry(mean_model, model_opts.symmetry_spec, caller);
    }

    return mean_model;
}

// The mean model basis at the training points, (n_train, n_basis), rejected where the mean model weights cannot be
// solved for: more weights than training points, a power that overflows, or a column that is zero at every training
// point. Nearly dependent columns are left to the mean model nugget.
inline arma::mat build_train_basis_mat(
        const MeanModel& mean_model, const arma::mat& X_ndim_ntrain_mat, const char* caller) {
    arma::mat Basis_ntrain_nbasis_mat = mean_model.calc_basis_mat(X_ndim_ntrain_mat);
    const arma::uword n_train = X_ndim_ntrain_mat.n_cols;
    const arma::uword n_basis = Basis_ntrain_nbasis_mat.n_cols;
    const bool has_zero_col = n_basis > 0 && arma::any(arma::all(Basis_ntrain_nbasis_mat == 0.0, 0));
    // The training points are finite, so only a power can overflow.
    const bool has_non_finite_val = !Basis_ntrain_nbasis_mat.is_finite();
    if (n_basis > n_train) {
        throw ErrorInvalidInput(
                gpflash::detail::error_prefix(caller, "ModelOptions::mean_model_degree_vec") + " gives "
                + std::to_string(n_basis) + " mean model basis functions, more than the " + std::to_string(n_train)
                + " training points. Set more degrees to 0, or add training points.");
    } else if (has_non_finite_val) {
        throw ErrorInvalidInput(
                gpflash::detail::error_prefix(caller, "ModelOptions::mean_model_degree_vec")
                + " gives a mean model basis function that overflows at a training point. Scale the inputs, or lower "
                  "that dimension's degree.");
    } else if (has_zero_col) {
        // A zero column carries no information about its weight, and would zero the preconditioner.
        throw ErrorInvalidInput(
                gpflash::detail::error_prefix(caller, "ModelOptions::mean_model_degree_vec")
                + " gives a mean model basis function that is zero at every training point. Set that dimension's "
                  "degree to 0.");
    } else {
        return Basis_ntrain_nbasis_mat;
    }
}

namespace detail {

// f_train's variance about the mean model basis's least-squares fit, over n_train - rank(Basis) degrees of freedom:
// the part of f_train left for the kernel and the noise. Zero where the basis spans every training point.
//
// Basis_ntrain_nbasis_mat is build_train_basis_mat's: finite, no zero column, and no more columns than points.
inline double calc_least_squares_residual_var(
        const arma::mat& Basis_ntrain_nbasis_mat, const arma::vec& f_train_vec, const bool has_mean_model_constant) {
    const arma::uword n_train = f_train_vec.n_elem;
    const arma::uword n_basis = Basis_ntrain_nbasis_mat.n_cols;
    const bool is_constant_only = has_mean_model_constant && n_basis == 1;
    double residual_var = 0.0;

    if (n_basis == 0) {
        // The zero mean model fits nothing, so the residual is f_train, over n_train degrees of freedom.
        residual_var = arma::dot(f_train_vec, f_train_vec) / static_cast<double>(n_train);
    } else if (is_constant_only) {
        // The least-squares constant is the sample mean, over n_train - 1 degrees of freedom; zero for one point.
        residual_var = arma::var(f_train_vec);
    } else {
        // Through the SVD, so dependent columns, which build_train_basis_mat accepts, count once toward the rank.
        arma::mat U_ntrain_nbasis_mat;
        arma::vec singular_val_vec;
        arma::mat V_nbasis_nbasis_mat;
        if (!arma::svd_econ(U_ntrain_nbasis_mat, singular_val_vec, V_nbasis_nbasis_mat, Basis_ntrain_nbasis_mat))
            throw ErrorNumericalFailure(
                    gpflash::detail::error_prefix("calc_least_squares_residual_var", "the mean model basis")
                    + " has no singular value decomposition.");

        // arma::pinv's default tolerance; the singular values are in decreasing order.
        const double rank_tol = static_cast<double>(n_train) * singular_val_vec(0) * arma::datum::eps;
        const arma::uword basis_rank = arma::accu(singular_val_vec > rank_tol);
        const arma::uword n_dof = n_train - basis_rank;

        // The fit projects f_train onto the basis's column space, spanned by the leading columns of U.
        if (n_dof > 0) {
            const arma::mat U_range_mat = U_ntrain_nbasis_mat.head_cols(basis_rank);
            const arma::vec U_range_f_train_inner_prod_vec = U_range_mat.t() * f_train_vec;
            const arma::vec least_squares_fit_vec = U_range_mat * U_range_f_train_inner_prod_vec;
            const arma::vec residual_vec = f_train_vec - least_squares_fit_vec;
            residual_var = arma::dot(residual_vec, residual_vec) / static_cast<double>(n_dof);
        }
    }

    return residual_var;
}

}  // namespace detail
}  // namespace mean
}  // namespace gpflash
