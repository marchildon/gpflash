/**
 * OptzOptions.hpp -- every tunable of the hyperparameter fit, in one place.
 */
#pragma once

#include <armadillo>  // arma::vec, the per-block bound overrides
#include <string>     // std::to_string, in the messages below

#include "gpflash/core/Errors.hpp"                             // ErrorInvalidInput, the field checks below
#include "gpflash/inputs/ModelOptions.hpp"                     // ModelOptions, check_model_options
#include "gpflash/optimization/search_space/HpOptzLayout.hpp"  // HpOptzLayout, which says what a run will read

namespace gpflash {

// Which algorithm the numerical optimizer drives the search with.
enum class NumericalOptzAlgorithm {
    Sqp,    // sequential quadratic programming, dense BFGS model plus a line search (default)
    Ccsaq,  // conservative convex separable approximation, quadratic form (Svanberg)
    End,    // one past the last, so the range check below needs no per-value maintenance
};

// What the fit searches over is not set here: TrainingData::sig2_f_val decides it.
struct OptzOptions {
    // Which model to fit: the same struct eval_mll and make_posterior take, so one object flows from fit to
    // evaluation to posterior.
    ModelOptions model_options;

    // One solve from the heuristic start by default. With use_multistart, n_ind_optz (>= 1) solves: the first from
    // the heuristic start, the rest from Latin-hypercube samples.
    bool use_multistart = false;
    int n_ind_optz = 5;
    unsigned hypercube_sampling_seed = 5;  // fixed, so a multistart fit is deterministic

    // Independent optimizations: nothing couples them until their results are combined, so the processes of the
    // ParallelContext the fit is given run them at once. Each one running holds its own n_train by n_train factor,
    // so max_parallel_ind_optz caps peak memory; lower it when memory is tight.
    int max_parallel_ind_optz = 5;

    // Which algorithm drives each local solve; optimization/README.md compares them.
    NumericalOptzAlgorithm numerical_optz_algorithm = NumericalOptzAlgorithm::Sqp;

    // Stopping criteria, as generic concepts the backend translates, so this struct names no optimizer.
    double rel_step_tol = 1e-8;  // stop when the step shrinks below this fraction of ||x||
    double rel_obj_tol = 1e-10;  // stop when the objective improves by less than this fraction of |f|
    int max_evals = 500;         // cap on MLL evaluations per local solve (a safety net)

    // The box, in natural space, log-transformed internally. Clearing use_bounds is an unconstrained search, not a
    // wider box. Multistart draws its starts from the box, which that search does not enforce, so the pair is rejected.
    bool use_bounds = true;

    // One bound per hyperparameter block. A single entry is broadcast over the whole block; otherwise give one
    // entry per component.
    arma::vec hp_gamma_lb_vec = arma::vec{1e-3};  // inverse length scales, n_dim components
    arma::vec hp_gamma_ub_vec = arma::vec{1e3};
    arma::vec hp_kern_extra_lb_vec = arma::vec{1e-2};  // the kernel's extras, n_hp_kern_extra components
    arma::vec hp_kern_extra_ub_vec = arma::vec{1e2};

    // Variance bounds, as multiples of the residual variance: f_train's variance about the mean model's least-squares
    // fit, which is var(f_train) for the constant mean model. Clear the flag to read them as absolute variances.
    bool scale_variance_bounds_by_residual_var = true;
    double hp_sig2_k_lb = 1e-4;      // x the residual variance
    double hp_sig2_k_ub = 1e4;       // x the residual variance
    double hp_sig2_f_val_lb = 1e-8;  // x the residual variance
    double hp_sig2_f_val_ub = 1e1;   // x the residual variance

    // Starting values. hp_gamma needs none: its start is 1 / stddev(X_d), from the data.
    double hp_sig2_k_init = 1.0;       // x the residual variance (or absolute; see the flag above)
    double hp_sig2_f_val_init = 1e-2;  // x the residual variance (or absolute); read only when the noise is estimated
    double hp_kern_extra_init = 1.0;   // broadcast over every component of the extras block
};

namespace detail {

// Both ends of one bound pair. Positive because the search runs in log space, and ordered because an empty interval
// has no feasible point.
//
// Above 0 rather than POSITIVE_MIN: a bound scaled by a small residual variance may sit below it and still has a
// logarithm.
inline void check_bound_pair_val(
        const double lb,
        const double ub,
        const char* lb_field_name,
        const char* ub_field_name,
        const char* caller,
        const char* hint = nullptr) {
    check_finite_above_val(lb, 0.0, lb_field_name, caller);
    check_finite_above_val(ub, 0.0, ub_field_name, caller);
    if (!(ub > lb))
        throw ErrorInvalidInput(with_hint(
                error_prefix(caller, ub_field_name) + " of " + std::to_string(ub) + " must exceed " + lb_field_name
                        + " of " + std::to_string(lb) + ".",
                hint));
}

// One block bound: a single entry to broadcast, or one per component. Length and positivity are all one corner can
// be judged on -- ordering needs both corners, so it waits for the resolved region.
inline void check_hp_bound_vec(
        const arma::vec& bound_vec, const int n_hp_in_block, const char* field_name, const char* caller) {
    if (bound_vec.n_elem != 1 && static_cast<int>(bound_vec.n_elem) != n_hp_in_block)
        throw ErrorInvalidInput(
                error_prefix(caller, field_name) + " has " + std::to_string(bound_vec.n_elem)
                + " entries; give 1 to broadcast over the block, or " + std::to_string(n_hp_in_block)
                + ", one per component.");
    check_all_positive_vals(bound_vec, field_name, caller, "The search runs in log space.");
}

// One pass over every option this fit will consume, the layout being what says which those are. A parameter the
// layout leaves out is not searched at all, so its settings are ignored rather than validated.
inline void check_optz_options(const OptzOptions& optz_opts, const HpOptzLayout& hp_optz_layout, const char* caller) {
    // The model the fit runs against.
    check_model_options(optz_opts.model_options, caller);

    // The multistart. Both counts are checked whether or not the flags reading them are set, so an invalid value
    // cannot come alive on a flag flip; the two flags below are a genuine conflict, so both must be read together.
    check_positive_count(optz_opts.n_ind_optz, "OptzOptions::n_ind_optz", caller);
    check_positive_count(optz_opts.max_parallel_ind_optz, "OptzOptions::max_parallel_ind_optz", caller);
    if (optz_opts.use_multistart && !optz_opts.use_bounds)
        throw ErrorInvalidInput(
                error_prefix(caller, "OptzOptions::use_multistart")
                + " samples inside the bounds, but use_bounds is false so they are not enforced. Set use_bounds = "
                + "true, or use_multistart = false to run one optimization from the heuristic start.");

    // The local solve: which algorithm, and when it stops.
    check_enum_val(
            optz_opts.numerical_optz_algorithm,
            "NumericalOptzAlgorithm",
            "OptzOptions::numerical_optz_algorithm",
            caller);
    check_finite_positive_val(optz_opts.rel_step_tol, "OptzOptions::rel_step_tol", caller);
    check_finite_positive_val(optz_opts.rel_obj_tol, "OptzOptions::rel_obj_tol", caller);
    check_positive_count(optz_opts.max_evals, "OptzOptions::max_evals", caller);

    // Each block bound on its own. Their ordering waits for the resolved region, since a lower corner given as one
    // broadcast entry and an upper corner given per component cannot be compared in this form.
    check_hp_bound_vec(optz_opts.hp_gamma_lb_vec, hp_optz_layout.n_dim, "OptzOptions::hp_gamma_lb_vec", caller);
    check_hp_bound_vec(optz_opts.hp_gamma_ub_vec, hp_optz_layout.n_dim, "OptzOptions::hp_gamma_ub_vec", caller);
    if (hp_optz_layout.n_hp_kern_extra > 0) {
        check_hp_bound_vec(
                optz_opts.hp_kern_extra_lb_vec,
                hp_optz_layout.n_hp_kern_extra,
                "OptzOptions::hp_kern_extra_lb_vec",
                caller);
        check_hp_bound_vec(
                optz_opts.hp_kern_extra_ub_vec,
                hp_optz_layout.n_hp_kern_extra,
                "OptzOptions::hp_kern_extra_ub_vec",
                caller);
    }
    // The layout alone says which variances are searched, so nothing here can disagree with it.
    if (hp_optz_layout.is_searched(HpBlock::Sig2K))
        check_bound_pair_val(
                optz_opts.hp_sig2_k_lb,
                optz_opts.hp_sig2_k_ub,
                "OptzOptions::hp_sig2_k_lb",
                "OptzOptions::hp_sig2_k_ub",
                caller);
    if (hp_optz_layout.is_searched(HpBlock::Sig2FVal))
        check_bound_pair_val(
                optz_opts.hp_sig2_f_val_lb,
                optz_opts.hp_sig2_f_val_ub,
                "OptzOptions::hp_sig2_f_val_lb",
                "OptzOptions::hp_sig2_f_val_ub",
                caller);

    // The starting values, each read only where its own parameter is searched.
    if (hp_optz_layout.is_searched(HpBlock::Sig2K))
        check_finite_positive_val(optz_opts.hp_sig2_k_init, "OptzOptions::hp_sig2_k_init", caller);
    if (hp_optz_layout.is_searched(HpBlock::Sig2FVal))
        check_finite_positive_val(optz_opts.hp_sig2_f_val_init, "OptzOptions::hp_sig2_f_val_init", caller);
    if (hp_optz_layout.n_hp_kern_extra > 0)
        check_finite_positive_val(optz_opts.hp_kern_extra_init, "OptzOptions::hp_kern_extra_init", caller);
}

}  // namespace detail

}  // namespace gpflash
