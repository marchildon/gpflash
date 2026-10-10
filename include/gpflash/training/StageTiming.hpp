/**
 * StageTiming.hpp -- optional per-stage timing for eval_mll (compile-time; zero overhead when off).
 *
 * Define GPFLASH_STAGE_TIMING before the first gpflash include to record each stage's wall time, in seconds, into
 * gpflash::detail::time_stages(). Without it the markers compile to nothing.
 */
#pragma once
#ifdef GPFLASH_STAGE_TIMING
#include <chrono>
namespace gpflash {
namespace detail {
// One field per eval_mll stage, in the order the stages run. Each is the wall time of the code between the
// two markers that bracket it, so the name says which call it covers:
//
//   kern_assembly_s  build_kern_assembly: the feature map, the K_corr build, the preconditioning and
//                    the eta terms. The whole assembly stage, not just the kernel matrix.
//   chol_s           the nugget or Sigma build, then the Cholesky factorization.
//   lin_sol_s        calc_chol_terms: the mean model weight and residual solves off the factor, and the noise-free
//                    closed-form hp_sig2_k that reads their norm.
//   sigma_inv_s      the back solve and recursive_dpotri_lower that turn the factor into Sigma^-1, then the
//                    gradient inputs and tr(Psi). Gradient only.
//   grad_s           the rest of calc_mll_hp_grad: the variance and kernel derivatives. Gradient only.
//
// Seconds, this package's default time unit. A consumer that wants another unit converts on its way out.
//
// Nothing resets these between evaluations, so a field the latest evaluation did not write still holds what
// an earlier one left there. Read them straight after the evaluation you mean, and read the two gradient
// fields only when that evaluation asked for a gradient -- otherwise they are a previous gradient's numbers,
// not zeros.
struct TimeStages {
    double kern_assembly_s = 0;
    double chol_s = 0;
    double lin_sol_s = 0;
    double sigma_inv_s = 0;
    double grad_s = 0;
    // The running mark, carried here so BEGIN and MARK need not share a scope: the stages live in
    // separate functions.
    std::chrono::steady_clock::time_point stage_start;
};
inline TimeStages& time_stages() {
    static thread_local TimeStages time_stages_singleton;
    return time_stages_singleton;
}
}  // namespace detail
}  // namespace gpflash
// BEGIN starts the clock; each MARK closes one stage and starts the next. MARK's argument is the name of a
// TimeStages field above, pasted straight into a member access -- which is why a call site reads
// GPFLASH_STAGE_MARK(chol_s) with no such variable in scope, and why the names stay lower_snake_case like
// the members they are rather than taking the macro's own uppercase.
//
// duration<double> is a duration in seconds -- its default ratio is 1 -- so .count() lands in the unit the
// fields are named for.
#define GPFLASH_STAGE_BEGIN() (gpflash::detail::time_stages().stage_start = std::chrono::steady_clock::now())
#define GPFLASH_STAGE_MARK(field)                                                                                  \
    do {                                                                                                           \
        const auto gpf_stage_now = std::chrono::steady_clock::now();                                               \
        gpflash::detail::time_stages().field =                                                                     \
                std::chrono::duration<double>(gpf_stage_now - gpflash::detail::time_stages().stage_start).count(); \
        gpflash::detail::time_stages().stage_start = gpf_stage_now;                                                \
    } while (0)
#else
#define GPFLASH_STAGE_BEGIN() ((void)0)
#define GPFLASH_STAGE_MARK(field) ((void)0)
#endif
