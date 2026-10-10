/**
 * PosteriorBuild.hpp -- build a queryable posterior from training data and hyperparameters.
 *
 * make_posterior near the bottom is the entry point, followed by load_posterior_verified, which checks a file
 * against these same inputs. Above both is the O(n_train^3) work a build runs once:
 *
 *   1) Validate the inputs.
 *   2) The kernel's center, then Ydot, its per-point squared norms, K_corr and the conditioning data.
 *   3) Open the precompute: the shapes, the hyperparameters, mirror and mean model a query re-reads, and Ydot.
 *   4) Complete Sigma_corr = hp_sig2_k (K_corr + eta I) + sigma_f^2 P^-2 in the buffer step 2 returned.
 *   5) Factorize it, then take the factor back to absolute coordinates.
 *   6) Read the mean model weights and the likelihood off that factor, and record the fit and its input hash.
 *   7) One back solve finishing lin_sol_sigma_residual = Sigma^-1 (f_train - Basis hp_beta_vec), with Basis the mean
 *      model basis at the training points.
 *
 * Everything a query needs lands in a PosteriorPrecompute, and nothing a query does not.
 */
#pragma once

#include <armadillo>
#include <string>
#include <utility>  // std::move

#include "gpflash/assembly/CholTerms.hpp"                   // CholTerms, calc_chol_terms
#include "gpflash/assembly/KernelAssembly.hpp"              // KernelAssembly, build_kern_assembly
#include "gpflash/assembly/SigmaMatrix.hpp"                 // build_sigma_corr_inplace
#include "gpflash/core/Platform.hpp"                        // GPFLASH_CHECK
#include "gpflash/inputs/HpVal.hpp"                         // HpVal
#include "gpflash/inputs/ModelOptions.hpp"                  // ModelOptions
#include "gpflash/inputs/TrainingData.hpp"                  // TrainingData, select_sig2_f_val
#include "gpflash/kernels/KernelDispatch.hpp"               // kernel_by_id
#include "gpflash/linear_algebra/Cholesky.hpp"              // the triangular solves
#include "gpflash/mean/MeanModel.hpp"                       // mean::MeanModel, make_mean_model, build_train_basis_mat
#include "gpflash/posterior_cache/CacheInputHash.hpp"       // cache::calc_posterior_input_hash
#include "gpflash/posterior_cache/PosteriorCache.hpp"       // cache::load_precompute_for_kernel
#include "gpflash/posterior_cache/PosteriorContents.hpp"    // cache::PosteriorContents, check_posterior_contents
#include "gpflash/posterior_cache/PosteriorModelInfo.hpp"   // gpflash::cache::PosteriorModelInfo
#include "gpflash/posterior_cache/PosteriorPrecompute.hpp"  // gpflash::detail::PosteriorPrecompute
#include "gpflash/posterior_eval/GpPosterior.hpp"           // GpPosterior
#include "gpflash/preconditioning/ConditionedCholesky.hpp"  // factorize_M_precon_to_L_M_inplace
#include "gpflash/training/EvalMll.hpp"                     // calc_mll_from_L_Sigma

namespace gpflash {
namespace detail {

inline cache::PosteriorContents build_posterior_precompute(
        const kernel::KernelBase& kern_fun,
        const TrainingData& train_data,
        const HpVal& hp_val_struct,
        const ModelOptions& model_opts) {
    // Step 1: validated before the kernel assembly.
    const int n_dim = static_cast<int>(train_data.X_ndim_ntrain_mat.n_rows);
    check_training_data(train_data, "build_posterior_precompute");
    check_hp_val(hp_val_struct, n_dim, kern_fun.n_hp_kern_extra(n_dim), "build_posterior_precompute");
    check_model_options(model_opts, "build_posterior_precompute");
    check_symmetry_applies(
            model_opts.symmetry_spec,
            train_data.X_ndim_ntrain_mat,
            kern_fun.is_symmetrizable(),
            kern_fun.kernel_name(),
            "build_posterior_precompute");
    // Also a check; see select_sig2_f_val.
    const double sig2_f_val = select_sig2_f_val(train_data, hp_val_struct.hp_sig2_f_val, "build_posterior_precompute");
    // Not in check_hp_val: a noise-free fit may leave hp_sig2_k unset, a posterior always carries one.
    check_supplied_hp_sig2_k(hp_val_struct, "build_posterior_precompute");
    // The mean model and its training basis.
    mean::MeanModel mean_model = mean::make_mean_model(model_opts, n_dim, "build_posterior_precompute");
    const arma::mat Basis_ntrain_nbasis_mat =
            mean::build_train_basis_mat(mean_model, train_data.X_ndim_ntrain_mat, "build_posterior_precompute");

    // Step 2: Ydot, K_corr and the conditioning data. eval_grad is false: a posterior
    // never differentiates by a hyperparameter, so neither the c^(rdot) factors nor the eta active row is kept.
    arma::vec x_center_vec = kern_fun.calc_x_center_vec(train_data.X_ndim_ntrain_mat, model_opts.symmetry_spec);
    KernelAssembly kern_assembly = build_kern_assembly(
            kern_fun,
            train_data.X_ndim_ntrain_mat,
            x_center_vec,
            hp_val_struct.hp_gamma_vec,
            hp_val_struct.hp_kern_extra_vec,
            model_opts.symmetry_spec,
            model_opts.cond_max,
            model_opts.kmat_eig_max_ub_method,
            /*eval_grad=*/false);
    const EtaTerms& kmat_eta_terms = kern_assembly.kmat_eta_terms;

    // Step 3: everything a query reads back that the factorization below does not produce. Ydot and its norms move
    // out of the assembly, which is done with them, and the precompute owns both from here.
    PosteriorPrecompute post_precompute;
    post_precompute.n_train = static_cast<int>(train_data.X_ndim_ntrain_mat.n_cols);
    post_precompute.n_dim = n_dim;
    post_precompute.hp_sig2_k = hp_val_struct.hp_sig2_k;
    // The center and hyperparameters a query needs to map its own point the same way.
    post_precompute.x_center_vec = std::move(x_center_vec);
    post_precompute.hp_gamma_vec = hp_val_struct.hp_gamma_vec;
    post_precompute.hp_kern_extra_vec = hp_val_struct.hp_kern_extra_vec;
    // The same mirror the two terms of K were built from.
    post_precompute.symmetry_spec = model_opts.symmetry_spec;
    // The mean model a query evaluates its prior mean with.
    post_precompute.mean_model = std::move(mean_model);

    post_precompute.Ydot_nfeature_ntrain_mat = std::move(kern_assembly.Ydot_nfeature_ntrain_mat);
    post_precompute.ydot_train_sq_norm_vec = std::move(kern_assembly.ydot_train_sq_norm_vec);
    post_precompute.n_feature = static_cast<int>(post_precompute.Ydot_nfeature_ntrain_mat.n_rows);
    fill_ydot_transpose(post_precompute);

    // Step 4: the buffer arrives holding K_corr and leaves holding L_Sigma, so it becomes L_Sigma_mat_lower.
    post_precompute.L_Sigma_mat_lower = std::move(kern_assembly.Buffer_mat);
    build_sigma_corr_inplace(
            post_precompute.L_Sigma_mat_lower,
            kmat_eta_terms.eta,
            sig2_f_val,
            post_precompute.hp_sig2_k,
            kmat_eta_terms.precon_scales.inv_precon_scale_vec);

    // Step 5: hp_sig2_k is always supplied here, so Sigma_corr carries both
    // variances and the factor is L_Sigma itself, with no variance factor to restore.
    factorize_M_precon_to_L_M_inplace(
            post_precompute.L_Sigma_mat_lower, kmat_eta_terms.precon_scales, "build_posterior_precompute/dpotrf");

    // Step 6: the mean model weights, the half solve L_Sigma^-1 r, the MLL's quadratic form and ln|Sigma|, all off
    // L_Sigma.
    CholTerms chol_terms = calc_chol_terms(
            post_precompute.L_Sigma_mat_lower,
            train_data.f_train_vec,
            Basis_ntrain_nbasis_mat,
            model_opts.cond_max,
            model_opts.mean_model_eig_max_ub_method);
    post_precompute.hp_beta_vec = std::move(chol_terms.hp_beta_vec);
    cache::PosteriorModelInfo post_model_info;
    post_model_info.kernel_id = kern_fun.kernel_id();
    // The factor is L_Sigma, so this form holds at any sig2_f_val, zero included.
    post_model_info.mll = calc_mll_from_L_Sigma(chol_terms, post_precompute.n_train);
    post_model_info.sig2_f_val = sig2_f_val;
    post_model_info.cond_max = model_opts.cond_max;
    post_model_info.kmat_eig_max_ub_method = model_opts.kmat_eig_max_ub_method;
    post_model_info.mean_model_eig_max_ub_method = model_opts.mean_model_eig_max_ub_method;
    post_model_info.input_hash = cache::calc_posterior_input_hash(
            post_model_info.kernel_id, post_precompute.mean_model, train_data, hp_val_struct, model_opts, sig2_f_val);

    // Step 7: the half solve L_Sigma^-1 r, finished by the back solve below to Sigma^-1 r, the field's name.
    post_precompute.lin_sol_sigma_residual_vec = std::move(chol_terms.lin_sol_L_residual_vec);
    solve_chol_transpose_vec_inplace(post_precompute.L_Sigma_mat_lower, post_precompute.lin_sol_sigma_residual_vec);

    cache::PosteriorContents post_contents{std::move(post_precompute), post_model_info};
    GPFLASH_CHECK(cache::check_posterior_contents(kern_fun, post_contents, "build_posterior_precompute"));

    return post_contents;
}

}  // namespace detail

// ── The entry points ─────────────────────────────────────────────────────────
// The kernel comes from ModelOptions::kernel_id, resolved once here rather than per evaluation.
//
// Points crossing this boundary are raw input points; a kernel's feature map is applied inside.

inline GpPosterior make_posterior(
        const TrainingData& train_data, const HpVal& hp_val_struct, const ModelOptions& model_opts) {
    cache::PosteriorContents post_contents = detail::build_posterior_precompute(
            kernel::kernel_by_id(model_opts.kernel_id), train_data, hp_val_struct, model_opts);

    return GpPosterior(std::move(post_contents));
}

// ── Loading a file these inputs produced ─────────────────────────────────────
// load_posterior in GpPosterior.hpp checks the file; this also recomputes the input hash and throws unless
// it matches, so a file fitted to other data cannot serve predictions. It needs the training data, which is why
// it lives here and not beside load_posterior. The hash covers the kernel id.
inline GpPosterior load_posterior_verified(
        const std::string& path,
        const TrainingData& train_data,
        const HpVal& hp_val_struct,
        const ModelOptions& model_opts) {
    const kernel::KernelBase& kern_fun = kernel::kernel_by_id(model_opts.kernel_id);
    const mean::MeanModel mean_model = mean::make_mean_model(
            model_opts, static_cast<int>(train_data.X_ndim_ntrain_mat.n_rows), "load_posterior_verified");
    const arma::u64 expected_input_hash = cache::calc_posterior_input_hash(
            kern_fun.kernel_id(),
            mean_model,
            train_data,
            hp_val_struct,
            model_opts,
            detail::select_sig2_f_val(train_data, hp_val_struct.hp_sig2_f_val, "load_posterior_verified"));
    cache::PosteriorContents post_contents =
            cache::load_precompute_for_kernel(kern_fun, path, "load_posterior_verified", &expected_input_hash);

    return GpPosterior(std::move(post_contents));
}

}  // namespace gpflash
