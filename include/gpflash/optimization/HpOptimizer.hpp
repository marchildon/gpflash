/**
 * HpOptimizer.hpp -- select GpFlash hyperparameters by maximizing the marginal log likelihood (MLL).
 *
 * A fit is three steps: the objective (and the layout of the vector it takes), the region to search, and the
 * search. A kernel that seeds itself differently overrides KernelBase::calc_optz_seeding; there is no per-kernel fit.
 *
 * The posterior entry points at the bottom read a fit result. They are not make_posterior overloads because the
 * posterior tier must not depend on the optimizer.
 */
#pragma once

#include <armadillo>  // arma::vec
#include <cstring>    // std::memcpy
#include <exception>  // std::exception_ptr, a setup error held until the processes agree
#include <memory>     // std::unique_ptr, std::make_unique, for the objective built inside that hold
#include <string>     // the agreed error message
#include <utility>    // std::move

#include "gpflash/core/Errors.hpp"                               // ErrorPeerFailure
#include "gpflash/core/ParallelContext.hpp"                      // ParallelContext, the processes a fit runs on
#include "gpflash/core/Platform.hpp"                             // GPFLASH_CHECK
#include "gpflash/inputs/ModelOptions.hpp"                       // ModelOptions
#include "gpflash/inputs/SymmetrySpec.hpp"                       // SymmetrySpec, check_symmetry_applies
#include "gpflash/inputs/TrainingData.hpp"                       // TrainingData, check_training_data
#include "gpflash/kernels/KernelBase.hpp"                        // KernelBase
#include "gpflash/mean/MeanModel.hpp"                            // calc_least_squares_residual_var
#include "gpflash/optimization/MllOptzObjective.hpp"             // MllOptzObjective
#include "gpflash/optimization/MllOptzResult.hpp"                // MllOptzResult, check_mll_optz_result
#include "gpflash/optimization/multistart/MultistartResult.hpp"  // MultistartResult
#include "gpflash/optimization/multistart/RunMultistart.hpp"     // run_multistart, agree_fit_error_inplace
#include "gpflash/optimization/search_space/ActiveBounds.hpp"    // calc_projected_gradient_norm
#include "gpflash/optimization/search_space/OptzOptions.hpp"     // OptzOptions, check_optz_options
#include "gpflash/optimization/search_space/OptzRegion.hpp"      // OptzRegion, build_optz_region
#include "gpflash/posterior_build/PosteriorBuild.hpp"            // make_posterior
#include "gpflash/posterior_cache/CacheInputHash.hpp"            // cache::detail::fnv1a_bytes, calc_symmetry_spec_hash
#include "gpflash/posterior_eval/GpPosterior.hpp"                // GpPosterior

namespace gpflash {

namespace detail {

// Every field, shape included, folded into one value the processes sharing a fit can compare. Not a cache key; it
// only borrows the cache's hash.
inline arma::u64 calc_training_data_hash(const TrainingData& train_data) {
    const arma::mat& X_mat = train_data.X_ndim_ntrain_mat;
    const arma::u64 n_dim = X_mat.n_rows;
    const arma::u64 n_train = X_mat.n_cols;
    arma::u64 hash_val = cache::detail::fnv1a_bytes(&n_dim, sizeof(n_dim));
    hash_val = cache::detail::fnv1a_bytes(&n_train, sizeof(n_train), hash_val);
    hash_val = cache::detail::fnv1a_bytes(X_mat.memptr(), X_mat.n_elem * sizeof(double), hash_val);
    hash_val = cache::detail::fnv1a_bytes(
            train_data.f_train_vec.memptr(), train_data.f_train_vec.n_elem * sizeof(double), hash_val);

    return cache::detail::fnv1a_bytes(&train_data.sig2_f_val, sizeof(train_data.sig2_f_val), hash_val);
}

// The mean model degrees and their count in one value the processes sharing a fit can compare, as given rather than
// broadcast: {2} and {2, 2} are the same model but different inputs, and every process must be given the same inputs.
inline arma::u64 calc_mean_model_degree_hash(const arma::uvec& mean_model_degree_vec) {
    const arma::u64_vec mean_model_degree_u64_vec = arma::conv_to<arma::u64_vec>::from(mean_model_degree_vec);
    const arma::u64 n_degree = mean_model_degree_u64_vec.n_elem;
    const arma::u64 hash_val = cache::detail::fnv1a_bytes(&n_degree, sizeof(n_degree));

    return cache::detail::fnv1a_bytes(mean_model_degree_u64_vec.memptr(), n_degree * sizeof(arma::u64), hash_val);
}

// A double's bits as an integer, so it can pass through an agreement check that takes only integers.
inline arma::u64 to_bit_pattern(const double double_val) {
    static_assert(sizeof(double) == sizeof(arma::u64), "A double must fill the integer exactly.");

    arma::u64 bits_val = 0;
    std::memcpy(&bits_val, &double_val, sizeof(bits_val));

    return bits_val;
}

// The fit from a finished search. One MLL evaluation at the winner gives the closed-form hyperparameters and the
// gradient for the optimality norm.
inline MllOptzResult build_mll_optz_result(
        const MllOptzObjective& optz_objective, const OptzRegion& optz_region, MultistartResult multistart_result) {
    const arma::vec ln_hp_best_vec = multistart_result.best_ln_hp_vec();

    MllOptzObjective::NegMllEval neg_mll_eval = optz_objective.calc_neg_mll(ln_hp_best_vec, /*eval_grad=*/true);

    MllOptzResult mll_optz_result;
    mll_optz_result.hp_val_struct = std::move(neg_mll_eval.hp_val_struct);
    mll_optz_result.model_options = optz_objective.model_options();
    mll_optz_result.optimality = calc_projected_gradient_norm(
            ln_hp_best_vec, neg_mll_eval.grad_vec, optz_region.constraint_lb_vec, optz_region.constraint_ub_vec);
    mll_optz_result.optz_region = optz_region;
    mll_optz_result.multistart_result = std::move(multistart_result);

    return mll_optz_result;
}

}  // namespace detail

// ── Public interface ─────────────────────────────────────────────────────────
// The kernel comes from optz_opts.model_options.kernel_id: the objective resolves it once and holds it for
// every evaluation, so a fit costs one switch rather than one per evaluation.
//
// parallel_context is the processes the independent optimizations are spread over, and defaults to every process
// the program has. A fit is a collective over them: all must call it, none may skip it, and all must pass the same
// train_data and optz_opts, since each solves part of one problem and the parts are summed.
inline MllOptzResult select_hp_with_max_mll(
        const TrainingData& train_data,
        const OptzOptions& optz_opts = {},
        const ParallelContext& parallel_context = {}) {
    const char* caller = "select_hp_with_max_mll";

    // 0. The same problem everywhere: the data, and the model it is fit with. Before the local checks, so a
    // process that fails one does not leave its peers waiting in a collective.
    parallel_context.check_same_on_every_process(
            detail::calc_training_data_hash(train_data), "the training data", caller);
    // A process that differs on one of these model options need not change any collective's size, so the search's
    // count comparisons cannot be relied on to see it.
    parallel_context.check_same_on_every_process(
            static_cast<arma::u64>(optz_opts.model_options.kernel_id), "the kernel id", caller);
    parallel_context.check_same_on_every_process(
            detail::to_bit_pattern(optz_opts.model_options.cond_max), "ModelOptions::cond_max", caller);
    parallel_context.check_same_on_every_process(
            static_cast<arma::u64>(optz_opts.model_options.kmat_eig_max_ub_method),
            "ModelOptions::kmat_eig_max_ub_method",
            caller);
    parallel_context.check_same_on_every_process(
            cache::detail::calc_symmetry_spec_hash(optz_opts.model_options.symmetry_spec),
            "ModelOptions::symmetry_spec",
            caller);
    parallel_context.check_same_on_every_process(
            static_cast<arma::u64>(optz_opts.model_options.has_mean_model_constant),
            "ModelOptions::has_mean_model_constant",
            caller);
    parallel_context.check_same_on_every_process(
            detail::calc_mean_model_degree_hash(optz_opts.model_options.mean_model_degree_vec),
            "ModelOptions::mean_model_degree_vec",
            caller);
    parallel_context.check_same_on_every_process(
            static_cast<arma::u64>(optz_opts.model_options.mean_model_eig_max_ub_method),
            "ModelOptions::mean_model_eig_max_ub_method",
            caller);

    // Steps 1 and 2 are local, so a process can raise in them alone. Held until every process has finished them,
    // since its peers would otherwise wait for it in the search's first collective.
    std::unique_ptr<const detail::MllOptzObjective> optz_objective_ptr;
    OptzRegion optz_region;
    std::exception_ptr setup_error_ptr;
    try {
        // 1. The objective and the layout. Data before layout, so malformed data reports itself; the objective checks
        // the mean model options as it builds the basis, and the other options follow, since the layout says which
        // options this fit reads. The objective evaluates unchecked, so the mirror is matched against the kernel and
        // the data first: the objective's kernel origin reads it.
        detail::check_training_data(train_data, caller);
        const kernel::KernelBase& kern_fun = kernel::kernel_by_id(optz_opts.model_options.kernel_id);
        detail::check_symmetry_applies(
                optz_opts.model_options.symmetry_spec,
                train_data.X_ndim_ntrain_mat,
                kern_fun.is_symmetrizable(),
                kern_fun.kernel_name(),
                caller);
        optz_objective_ptr = std::make_unique<const detail::MllOptzObjective>(train_data, optz_opts, caller);
        detail::check_optz_options(optz_opts, optz_objective_ptr->layout(), caller);

        // 2. The region, with this kernel's seeding, its variances scaled to what the mean model leaves in f_train.
        // Fitted only when the bounds are scaled by it: the fit can raise, and calc_variance_scale ignores it
        // otherwise.
        double f_train_residual_var = 0.0;
        if (optz_opts.scale_variance_bounds_by_residual_var)
            f_train_residual_var = mean::detail::calc_least_squares_residual_var(
                    optz_objective_ptr->Basis_ntrain_nbasis_mat(),
                    train_data.f_train_vec,
                    optz_opts.model_options.has_mean_model_constant);
        optz_region = detail::build_optz_region(
                train_data,
                optz_objective_ptr->x_center_vec(),
                f_train_residual_var,
                optz_opts,
                kern_fun.calc_optz_seeding(train_data.X_ndim_ntrain_mat),
                optz_objective_ptr->layout(),
                caller);
    } catch (...) {
        setup_error_ptr = std::current_exception();
    }

    // The process that raised rethrows its own error; every other one learns of it here.
    detail::FitErrorKind setup_error_kind = detail::FitErrorKind::None;
    std::string setup_error_message;
    if (setup_error_ptr) {
        setup_error_kind = detail::FitErrorKind::SetupFailed;
        setup_error_message = detail::to_error_message(setup_error_ptr);
    }
    const int i_failed_process =
            detail::agree_fit_error_inplace(setup_error_kind, setup_error_message, parallel_context);
    if (setup_error_ptr) std::rethrow_exception(setup_error_ptr);
    if (setup_error_kind != detail::FitErrorKind::None)
        throw ErrorPeerFailure(
                std::string(caller) + ": process " + std::to_string(i_failed_process) + " of "
                + std::to_string(parallel_context.n_process()) + " failed before the search: " + setup_error_message);
    const detail::MllOptzObjective& optz_objective = *optz_objective_ptr;

    // 3. The search, then the fit.
    MultistartResult multistart_result = detail::run_multistart(
            optz_region,
            optz_objective.layout(),
            optz_objective.local_objective(),
            optz_opts,
            parallel_context,
            caller);
    MllOptzResult mll_optz_result =
            detail::build_mll_optz_result(optz_objective, optz_region, std::move(multistart_result));

    GPFLASH_CHECK(detail::check_mll_optz_result(mll_optz_result, caller));
    return mll_optz_result;
}

// ── Posterior from a fit ─────────────────────────────────────────────────────

// train_data must be the data the fit read. The fit's ModelOptions come from the result.
inline GpPosterior make_posterior_from_fit(const TrainingData& train_data, const MllOptzResult& mll_optz_result) {
    return make_posterior(train_data, mll_optz_result.hp_val_struct, mll_optz_result.model_options);
}

// Train then predict in one call. Not cached; call save_precompute on the result to keep it.
inline GpPosterior fit_and_make_posterior(
        const TrainingData& train_data,
        const OptzOptions& optz_opts = {},
        const ParallelContext& parallel_context = {}) {
    return make_posterior_from_fit(train_data, select_hp_with_max_mll(train_data, optz_opts, parallel_context));
}

}  // namespace gpflash
