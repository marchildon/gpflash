/**
 * ModelOptions.hpp -- which model is being evaluated.
 */
#pragma once

#include <armadillo>

#include "gpflash/core/Errors.hpp"            // the field checks check_model_options composes
#include "gpflash/inputs/EigMaxUbMethod.hpp"  // EigMaxUbMethod
#include "gpflash/inputs/KernelId.hpp"        // gpflash::kernel::KernelId
#include "gpflash/inputs/SymmetrySpec.hpp"    // SymmetrySpec, check_symmetry_spec

namespace gpflash {

// ── The choices that define which model is being evaluated. ──────────────────
// The upper bound for the condition number that sizes the nugget eta.
constexpr double COND_MAX_DEFAULT = 1.0e10;

// Upper limit for the condition number that can be selected before an error is raised.
constexpr double COND_MAX_LIMIT = 1.0e13;

// The highest power of an input dimension the mean model basis may carry: a supported limit, not one the basis code
// needs. The mean model is for a constant, linear or quadratic trend, and the kernel models what remains.
constexpr arma::uword MEAN_MODEL_DEGREE_MAX = 2;

struct ModelOptions {
    gpflash::kernel::KernelId kernel_id = gpflash::kernel::KernelId::Matern52;  // no "unset" value exists
    double cond_max = COND_MAX_DEFAULT;  // sizes the nuggets eta and mean_model_eta
    // The bound on lambda_max(K_corr) eta is sized from. Gershgorin's is tighter; the trace is constant, so
    // d(eta)/d(hp) is zero.
    EigMaxUbMethod kmat_eig_max_ub_method = EigMaxUbMethod::Gershgorin;

    // The mean model; see mean/MeanModel.hpp. Its weights are solved in closed form, never searched. With no constant
    // and every degree 0 it is the zero mean model.
    bool has_mean_model_constant = true;
    // degree_d, 0 to MEAN_MODEL_DEGREE_MAX: the one power of x_d in the basis, or no column for 0. One entry applies to
    // every dimension.
    arma::uvec mean_model_degree_vec = {0};
    // The bound on lambda_max of the preconditioned mean model Gram matrix that mean_model_eta is sized from. The trace
    // by default: that matrix is small, so the looser bound costs little, and it is constant, so mean_model_eta is
    // fixed through a fit. Gershgorin's is tighter and moves with the hyperparameters.
    EigMaxUbMethod mean_model_eig_max_ub_method = EigMaxUbMethod::Trace;

    SymmetrySpec symmetry_spec;  // gives kernel_id a mirror symmetry; off by default
};

namespace detail {

// make_mean_model checks the mean model fields together, since their length check needs n_dim.
inline void check_model_options(const ModelOptions& model_opts, const char* caller) {
    check_enum_val(model_opts.kernel_id, "KernelId", "ModelOptions::kernel_id", caller);
    check_finite_in_range_val(
            model_opts.cond_max,
            1.0,
            COND_MAX_LIMIT,
            "ModelOptions::cond_max",
            caller,
            "Past the limit the nugget it sizes changes the model rather than regularizing it.");
    check_enum_val(model_opts.kmat_eig_max_ub_method, "EigMaxUbMethod", "ModelOptions::kmat_eig_max_ub_method", caller);
    check_enum_val(
            model_opts.mean_model_eig_max_ub_method,
            "EigMaxUbMethod",
            "ModelOptions::mean_model_eig_max_ub_method",
            caller);
    check_symmetry_spec(model_opts.symmetry_spec, caller);
}
}  // namespace detail
}  // namespace gpflash
