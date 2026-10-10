/**
 * GpPosterior.hpp -- hold a fitted posterior, serve queries against it, and load one from a cache file.
 *
 * It owns the precompute and reads its kernel from the fit info it carries, so the queried kernel and the
 * fitted one cannot disagree: there is no second place to name one.
 *
 * Building one from training data is posterior_build/; what is here needs a precompute that already exists. The
 * caller works in raw input space even for a feature-map kernel, since the map is applied inside.
 *
 * load_posterior checks the file -- format version, kernel, header values, array shapes -- but not which fit
 * wrote it. To check that too, a caller still holding the training data calls load_posterior_verified in
 * posterior_build/.
 */
#pragma once

#include <armadillo>
#include <string>
#include <utility>  // std::move

#include "gpflash/kernels/KernelDispatch.hpp"               // kernel_by_id
#include "gpflash/posterior_cache/PosteriorCache.hpp"       // cache::save_precompute, load_precompute_for_file_kernel
#include "gpflash/posterior_cache/PosteriorContents.hpp"    // cache::PosteriorContents
#include "gpflash/posterior_cache/PosteriorModelInfo.hpp"   // cache::PosteriorModelInfo
#include "gpflash/posterior_cache/PosteriorPrecompute.hpp"  // PosteriorPrecompute
#include "gpflash/posterior_eval/PosteriorOptions.hpp"      // PosteriorOptions
#include "gpflash/posterior_eval/PosteriorQuery.hpp"        // detail::eval_precompute
#include "gpflash/posterior_eval/PosteriorResult.hpp"       // PosteriorResult

namespace gpflash {

class GpPosterior {
   public:
    // Adopt a precompute that already exists, whether it came off a file or out of a build. Its model info
    // travels with it so a later save_precompute records what produced it.
    explicit GpPosterior(gpflash::cache::PosteriorContents post_contents)
            : post_precompute_(std::move(post_contents.post_precompute)),
              post_model_info_(post_contents.post_model_info) {}

    // ── Queries ──────────────────────────────────────────────────────────────
    // One kernel lookup per call, outside the per-point loops.
    PosteriorResult eval(const arma::vec& x_post_vec, const PosteriorOptions& post_opts = {}) const {
        return gpflash::detail::eval_precompute(kern_fun(), post_precompute_, x_post_vec, post_opts);
    }

    PosteriorResult eval(const arma::mat& X_ndim_npost_mat, const PosteriorOptions& post_opts = {}) const {
        return gpflash::detail::eval_precompute(kern_fun(), post_precompute_, X_ndim_npost_mat, post_opts);
    }

    // ── Shapes ───────────────────────────────────────────────────────────────
    int n_train() const { return post_precompute_.n_train; }
    int n_dim() const { return post_precompute_.n_dim; }

    // Rows the kernel evaluates on. Equal to n_dim except under a feature map, where Periodic doubles it.
    int n_feature() const { return post_precompute_.n_feature; }

    // ── Hyperparameters for the model ────────────────────────────────────────
    const arma::vec& hp_gamma_vec() const { return post_precompute_.hp_gamma_vec; }

    // Empty for a kernel with no extra hyperparameters.
    const arma::vec& hp_kern_extra_vec() const { return post_precompute_.hp_kern_extra_vec; }

    // The mean model weights, one per basis function, solved in closed form at fit time rather than supplied.
    const arma::vec& hp_beta_vec() const { return post_precompute_.hp_beta_vec; }

    // The mean model, whose basis hp_beta_vec multiplies in basis(x)'s column order.
    const mean::MeanModel& mean_model() const { return post_precompute_.mean_model; }

    double hp_sig2_k() const { return post_precompute_.hp_sig2_k; }

    // ── What this posterior can answer ───────────────────────────────────────
    // False for one loaded from a mean-only cache file, which stores no Cholesky factor.
    bool has_variance_data() const { return post_precompute_.has_variance_data(); }

    // ── Provenance ───────────────────────────────────────────────────────────
    // Which fit produced this. A query reads none of it; it is held because a re-save must stamp the new file
    // with the same values.
    const cache::PosteriorModelInfo& post_model_info() const { return post_model_info_; }

    // The kernel the fit used, and the one every query above evaluates.
    kernel::KernelId kernel_id() const { return post_model_info_.kernel_id; }

    // The hash of the inputs this was built from, or the one recorded in the file it was loaded from.
    arma::u64 input_hash() const { return post_model_info_.input_hash; }

    // The marginal log likelihood at these hyperparameters. Ranks two fits of the same inputs.
    double mll() const { return post_model_info_.mll; }

    // ── Persist ──────────────────────────────────────────────────────────────
    // note is free text stored verbatim. The kernel written is post_model_info_'s, so a re-saved posterior
    // records the fit that produced it.
    void save_precompute(
            const std::string& path, const std::string& note = "", const bool include_variance_data = true) const {
        cache::save_precompute(path, post_model_info_, post_precompute_, note, include_variance_data);
    }

   private:
    const kernel::KernelBase& kern_fun() const { return kernel::kernel_by_id(post_model_info_.kernel_id); }

    gpflash::detail::PosteriorPrecompute post_precompute_;
    cache::PosteriorModelInfo post_model_info_;
};

// Reload a posterior saved with save_precompute. The kernel comes from the file's own header, so the caller names
// none and a file written by one kernel cannot be queried as another.
inline GpPosterior load_posterior(const std::string& path) {
    return GpPosterior(cache::detail::load_precompute_for_file_kernel(path, "load_posterior"));
}

}  // namespace gpflash
