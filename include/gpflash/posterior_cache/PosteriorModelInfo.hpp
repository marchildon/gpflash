/**
 * PosteriorModelInfo.hpp -- the model a posterior was fitted to, and what that fit scored. A query reads none of
 * it; it is here so a saved file can say what produced its arrays.
 *
 * CacheInfo holds one of these beside the file's other header scalars.
 */
#pragma once

#include <armadillo>
#include <string>

#include "gpflash/core/Errors.hpp"             // the field checks check_posterior_model_info composes
#include "gpflash/core/StringFormat.hpp"       // to_formatted_string
#include "gpflash/inputs/EigMaxUbMethod.hpp"   // EigMaxUbMethod, to_eig_max_ub_method_name
#include "gpflash/kernels/KernelDispatch.hpp"  // kernel::KernelId, to_kernel_name

namespace gpflash {
namespace cache {

struct PosteriorModelInfo {
    // Which kernel used for the surrogate.
    kernel::KernelId kernel_id = kernel::KernelId::End;  // no "unset" value exists, so End marks one here

    // Which inputs it was fitted to. Compared on a verified load, so a file from other inputs is refused rather than
    // served.
    arma::u64 input_hash = 0;

    // The marginal log likelihood at these hyperparameters. Ranks several fits, and only where input_hash matches.
    double mll = 0.0;

    // How the fit was configured.
    double sig2_f_val = 0.0;  // the resolved noise variance, whichever of the three sign rules supplied it
    double cond_max = 0.0;    // the conditioning limit that sized the nugget
    EigMaxUbMethod kmat_eig_max_ub_method = EigMaxUbMethod::End;        // the bound it was sized from; End marks unset
    EigMaxUbMethod mean_model_eig_max_ub_method = EigMaxUbMethod::End;  // the mean model nugget's; End marks unset
};

// The ranges the header requires of these seven. check_cache_info delegates here, so a value this rejects is one
// no file can carry, on the way in or out.
inline void check_posterior_model_info(const PosteriorModelInfo& post_model_info, const char* caller) {
    if (post_model_info.kernel_id == kernel::KernelId::End)
        throw ErrorInvalidInput(
                gpflash::detail::error_prefix(caller, "PosteriorModelInfo::kernel_id")
                + " is End, which names no kernel.");

    // Fits are ranked on this, so it must be a real number.
    gpflash::detail::check_finite_val(post_model_info.mll, "PosteriorModelInfo::mll", caller);
    // Zero is the noise-free case, so non-negative rather than positive.
    gpflash::detail::check_finite_non_negative_val(
            post_model_info.sig2_f_val, "PosteriorModelInfo::sig2_f_val", caller);
    gpflash::detail::check_finite_above_val(post_model_info.cond_max, 1.0, "PosteriorModelInfo::cond_max", caller);
    gpflash::detail::check_enum_val(
            post_model_info.kmat_eig_max_ub_method,
            "EigMaxUbMethod",
            "PosteriorModelInfo::kmat_eig_max_ub_method",
            caller);
    gpflash::detail::check_enum_val(
            post_model_info.mean_model_eig_max_ub_method,
            "EigMaxUbMethod",
            "PosteriorModelInfo::mean_model_eig_max_ub_method",
            caller);
}

// The model block of a cache summary.
inline std::string to_summary_string(const PosteriorModelInfo& post_model_info) {
    return gpflash::detail::to_formatted_string(
            "  kernel             %s\n"
            "  marginal log lik   %.6g\n"
            "  noise variance     %.6g\n"
            "  conditioning limit %.6g\n"
            "  eig_max bound      %s\n"
            "  mean model bound   %s\n"
            "  input hash         0x%016llx\n",
            kernel::to_kernel_name(post_model_info.kernel_id),
            post_model_info.mll,
            post_model_info.sig2_f_val,
            post_model_info.cond_max,
            to_eig_max_ub_method_name(post_model_info.kmat_eig_max_ub_method),
            to_eig_max_ub_method_name(post_model_info.mean_model_eig_max_ub_method),
            static_cast<unsigned long long>(post_model_info.input_hash));
}

}  // namespace cache
}  // namespace gpflash
