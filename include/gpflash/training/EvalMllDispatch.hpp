/**
 * EvalMllDispatch.hpp -- every entry point onto EvalMll.hpp's evaluation, and the input validation they run:
 *
 *   check_eval_mll_inputs   every check the evaluation depends on except the mean model's
 *   detail::eval_mll        checked, kernel given as an instance
 *   gpflash::eval_mll       the public one, which resolves model_opts.kernel_id to that instance
 *
 * No arithmetic here: each validates, names the kernel and forwards. The public form resolves the kernel once
 * per call, outside every loop, so the switch costs one branch per evaluation.
 */
#pragma once

#include "gpflash/core/NoiseModel.hpp"         // NoiseRequest, classify_noise_request
#include "gpflash/inputs/HpVal.hpp"            // HpVal, check_supplied_hp_sig2_k
#include "gpflash/inputs/ModelOptions.hpp"     // ModelOptions
#include "gpflash/inputs/TrainingData.hpp"     // TrainingData, select_sig2_f_val
#include "gpflash/kernels/KernelBase.hpp"      // kernel::KernelBase
#include "gpflash/kernels/KernelDispatch.hpp"  // kernel_by_id
#include "gpflash/mean/MeanModel.hpp"          // mean::MeanModel, make_mean_model, build_train_basis_mat
#include "gpflash/training/EvalMll.hpp"        // eval_mll_inputs_unchecked
#include "gpflash/training/EvalVariances.hpp"  // EvalVariances, check_eval_variances
#include "gpflash/training/MllResult.hpp"      // MllResult

namespace gpflash {
namespace detail {

// ── Input validation ─────────────────────────────────────────────────────────

// Every input the evaluation reads except the mean model, which make_mean_model and build_train_basis_mat check.
// Load-bearing rather than defensive: ARMA_NO_DEBUG removes Armadillo's bounds checks, so a mispacked hyperparameter
// block reads past its allocation.
inline void check_eval_mll_inputs(
        const kernel::KernelBase& kern_fun,
        const TrainingData& train_data,
        const HpVal& hp_val_struct,
        const ModelOptions& model_opts,
        const EvalVariances& eval_variances) {
    const int n_dim = static_cast<int>(train_data.X_ndim_ntrain_mat.n_rows);

    check_training_data(train_data, "eval_mll");
    check_hp_val(hp_val_struct, n_dim, kern_fun.n_hp_kern_extra(n_dim), "eval_mll");
    check_model_options(model_opts, "eval_mll");
    check_symmetry_applies(
            model_opts.symmetry_spec,
            train_data.X_ndim_ntrain_mat,
            kern_fun.is_symmetrizable(),
            kern_fun.kernel_name(),
            "eval_mll");
    check_eval_variances(eval_variances, "eval_mll");
    // The noise-free evaluation supplies hp_sig2_k in closed form; every other one reads the caller's.
    if (classify_noise_request(eval_variances.sig2_f_val) != NoiseRequest::NoiseFree)
        check_supplied_hp_sig2_k(hp_val_struct, "eval_mll");
}

// ── Checked evaluation ───────────────────────────────────────────────────────

// Checks every input, then runs the unchecked evaluation. The form to call unless you are the
// optimizer's inner loop.
inline MllResult eval_mll(
        const kernel::KernelBase& kern_fun,
        const TrainingData& train_data,
        const HpVal& hp_val_struct,
        const ModelOptions& model_opts,
        const bool eval_grad) {
    EvalVariances eval_variances;
    eval_variances.sig2_f_val = select_sig2_f_val(train_data, hp_val_struct.hp_sig2_f_val, "eval_mll");
    eval_variances.hp_sig2_k = hp_val_struct.hp_sig2_k;

    check_eval_mll_inputs(kern_fun, train_data, hp_val_struct, model_opts, eval_variances);

    // The kernel's center, then the mean model and its basis at the training points.
    const arma::vec x_center_vec = kern_fun.calc_x_center_vec(train_data.X_ndim_ntrain_mat, model_opts.symmetry_spec);
    const int n_dim = static_cast<int>(train_data.X_ndim_ntrain_mat.n_rows);
    const mean::MeanModel mean_model = mean::make_mean_model(model_opts, n_dim, "eval_mll");
    const arma::mat Basis_ntrain_nbasis_mat =
            mean::build_train_basis_mat(mean_model, train_data.X_ndim_ntrain_mat, "eval_mll");

    return eval_mll_inputs_unchecked(
            kern_fun,
            x_center_vec,
            Basis_ntrain_nbasis_mat,
            train_data,
            hp_val_struct,
            eval_variances,
            model_opts,
            eval_grad);
}

}  // namespace detail

// ── Public interface: the named-hyperparameter entry point ───────────────────
// The hyperparameters arrive named and reach the kernel as its two blocks.
//
// return_mll_gradient is an argument and not a model_opts field: it says what is asked of the model, not which
// model it is. False leaves MllResult::mll_hp_grad_struct unset and never forms Sigma^-1.
inline MllResult eval_mll(
        const TrainingData& train_data,
        const HpVal& hp_val_struct,
        const ModelOptions& model_opts,
        const bool return_mll_gradient = false) {
    return detail::eval_mll(
            kernel::kernel_by_id(model_opts.kernel_id), train_data, hp_val_struct, model_opts, return_mll_gradient);
}

}  // namespace gpflash
