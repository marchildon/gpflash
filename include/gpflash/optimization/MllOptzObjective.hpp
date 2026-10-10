/**
 * MllOptzObjective.hpp -- the negative MLL as a function of the log-hyperparameter vector the optimizer searches.
 *
 * Exponentiates the searched vector once per evaluation.
 */
#pragma once

#include <armadillo>  // arma::vec
#include <utility>    // std::move

#include "gpflash/core/NoiseModel.hpp"                          // classify_noise_request
#include "gpflash/inputs/HpVal.hpp"                             // HpVal
#include "gpflash/inputs/ModelOptions.hpp"                      // ModelOptions
#include "gpflash/inputs/TrainingData.hpp"                      // TrainingData, select_sig2_f_val
#include "gpflash/kernels/KernelBase.hpp"                       // kernel::KernelBase, the kernel it holds
#include "gpflash/kernels/KernelDispatch.hpp"                   // kernel_by_id
#include "gpflash/mean/MeanModel.hpp"                           // make_mean_model, build_train_basis_mat
#include "gpflash/optimization/local_optz/LocalOptzResult.hpp"  // LocalObjective
#include "gpflash/optimization/search_space/HpOptzLayout.hpp"   // HpOptzLayout, build_hp_optz_layout, the codec
#include "gpflash/optimization/search_space/OptzOptions.hpp"    // OptzOptions
#include "gpflash/training/EvalMll.hpp"                         // eval_mll_inputs_unchecked
#include "gpflash/training/EvalVariances.hpp"                   // EvalVariances
#include "gpflash/training/MllHpGrad.hpp"                       // MllHpGrad, read off NegMllEval's mll_result
#include "gpflash/training/MllResult.hpp"                       // MllResult, fill_closed_form_hp_inplace

namespace gpflash {
namespace detail {

class MllOptzObjective {
   public:
    // The kernel, its center and the mean model basis at the training points, resolved once here from
    // optz_opts.model_options and read by every call below, so the objective cannot disagree with the options it
    // reports. train_data must outlive this object; optz_opts.model_options is the only part read, and it is copied.
    // caller names the entry point in any error raised here.
    MllOptzObjective(const TrainingData& train_data, const OptzOptions& optz_opts, const char* caller)
            : kern_fun_(kernel::kernel_by_id(optz_opts.model_options.kernel_id)),
              x_center_vec_(
                      kern_fun_.calc_x_center_vec(train_data.X_ndim_ntrain_mat, optz_opts.model_options.symmetry_spec)),
              Basis_ntrain_nbasis_mat_(mean::build_train_basis_mat(
                      mean::make_mean_model(
                              optz_opts.model_options, static_cast<int>(train_data.X_ndim_ntrain_mat.n_rows), caller),
                      train_data.X_ndim_ntrain_mat,
                      caller)),
              train_data_(train_data),
              model_options_(optz_opts.model_options),
              hp_optz_layout_(build_hp_optz_layout(
                      static_cast<int>(train_data.X_ndim_ntrain_mat.n_rows),
                      kern_fun_.n_hp_kern_extra(static_cast<int>(train_data.X_ndim_ntrain_mat.n_rows)),
                      classify_noise_request(train_data.sig2_f_val),
                      caller)) {}

    const HpOptzLayout& layout() const { return hp_optz_layout_; }

    // The kernel every evaluation runs with.
    const kernel::KernelBase& kern_fun() const { return kern_fun_; }

    // The options every evaluation runs with.
    const ModelOptions& model_options() const { return model_options_; }

    const arma::mat& Basis_ntrain_nbasis_mat() const { return Basis_ntrain_nbasis_mat_; }
    const arma::vec& x_center_vec() const { return x_center_vec_; }
    // Negative MLL at ln_hp_optz_vec, with its gradient when grad_vec_out != nullptr. The hot path.
    double calc_objective(const arma::vec& ln_hp_optz_vec, arma::vec* grad_vec_out) const {
        NegMllEval neg_mll_eval = calc_neg_mll(ln_hp_optz_vec, /*eval_grad=*/grad_vec_out != nullptr);
        if (grad_vec_out) *grad_vec_out = std::move(neg_mll_eval.grad_vec);

        return neg_mll_eval.neg_mll;
    }

    // calc_objective as the backend seam takes it. Captures this.
    LocalObjective local_objective() const {
        return [this](const arma::vec& ln_hp_optz_vec, arma::vec* grad_vec_out) {
            return calc_objective(ln_hp_optz_vec, grad_vec_out);
        };
    }

    // Everything one MLL evaluation produces.
    struct NegMllEval {
        double neg_mll = 0.0;
        arma::vec grad_vec;   // in log space; empty unless requested
        HpVal hp_val_struct;  // every hyperparameter: searched ones from the vector, closed-form from the evaluation
        MllResult mll_result;
    };

    // One evaluation at ln_hp_optz_vec, with the log-space gradient when eval_grad.
    NegMllEval calc_neg_mll(const arma::vec& ln_hp_optz_vec, const bool eval_grad) const {
        const arma::vec hp_optz_vec = arma::exp(ln_hp_optz_vec);

        NegMllEval neg_mll_eval;
        neg_mll_eval.hp_val_struct = to_hp_val(hp_optz_layout_, hp_optz_vec);

        // An unsearched sigma_f^2 is at HpVal's sentinel, so select_sig2_f_val takes the data's value.
        EvalVariances eval_variances;
        eval_variances.sig2_f_val = select_sig2_f_val(
                train_data_, neg_mll_eval.hp_val_struct.hp_sig2_f_val, "MllOptzObjective::calc_neg_mll");
        eval_variances.hp_sig2_k = neg_mll_eval.hp_val_struct.hp_sig2_k;

        neg_mll_eval.mll_result = eval_mll_inputs_unchecked(
                kern_fun_,
                x_center_vec_,
                Basis_ntrain_nbasis_mat_,
                train_data_,
                neg_mll_eval.hp_val_struct,
                eval_variances,
                model_options_,
                eval_grad);
        neg_mll_eval.neg_mll = -neg_mll_eval.mll_result.mll;
        // After the evaluation, which reads only the searched hyperparameters.
        fill_closed_form_hp_inplace(neg_mll_eval.mll_result, neg_mll_eval.hp_val_struct);

        // Chain rule into log space: d(-MLL)/d(ln hp) = -hp * dMLL/d(hp). MllHpGrad has HpVal's four blocks, so
        // the layout packs it the same way.
        if (eval_grad) {
            const MllHpGrad& mll_hp_grad_struct = neg_mll_eval.mll_result.mll_hp_grad_struct;
            neg_mll_eval.grad_vec = -hp_optz_vec
                    % to_hp_optz_vec_from_blocks(
                            hp_optz_layout_,
                            mll_hp_grad_struct.d_mll_d_hp_gamma_vec,
                            mll_hp_grad_struct.d_mll_d_hp_kern_extra_vec,
                            mll_hp_grad_struct.d_mll_d_hp_sig2_k,
                            mll_hp_grad_struct.d_mll_d_hp_sig2_f_val);
        }

        return neg_mll_eval;
    }

   private:
    const kernel::KernelBase& kern_fun_;
    // The point the kernel measures the training points from, fixed through a fit.
    const arma::vec x_center_vec_;
    // The mean model basis at the training points, fixed through a fit.
    const arma::mat Basis_ntrain_nbasis_mat_;
    const TrainingData& train_data_;
    const ModelOptions model_options_;
    const HpOptzLayout hp_optz_layout_;
};

}  // namespace detail
}  // namespace gpflash
