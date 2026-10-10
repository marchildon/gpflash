/**
 * PosteriorContents.hpp -- what a posterior is made of: the arrays a query reads, and the record of which fit
 * produced them. The build path and the load path both produce one.
 */
#pragma once

#include "gpflash/kernels/KernelBase.hpp"                   // kernel::KernelBase, passed to the checker
#include "gpflash/posterior_cache/PosteriorModelInfo.hpp"   // PosteriorModelInfo, check_posterior_model_info
#include "gpflash/posterior_cache/PosteriorPrecompute.hpp"  // PosteriorPrecompute, check_posterior_precompute

namespace gpflash {
namespace cache {

struct PosteriorContents {
    gpflash::detail::PosteriorPrecompute post_precompute;  // what a query evaluates
    PosteriorModelInfo post_model_info;                    // what produced it; no query reads this
};

// Composes both members' checkers, so one call validates the whole struct. The kernel is what the struct cannot
// know: its array widths are only checkable against the kernel's declared ones.
inline void check_posterior_contents(
        const kernel::KernelBase& kern_fun, const PosteriorContents& post_contents, const char* caller) {
    gpflash::detail::check_posterior_precompute(kern_fun, post_contents.post_precompute, caller);
    check_posterior_model_info(post_contents.post_model_info, caller);
}

}  // namespace cache
}  // namespace gpflash
