// StageTiming.hpp's instrumented branch. What this file asserts:
//
//   1) stage_timing_records  every stage of eval_mll writes a finite, non-negative wall time, on both the
//                            noise-free and the noisy driver, and the whole pipeline accounts for more than zero
//
// The only translation unit compiled with GPFLASH_STAGE_TIMING, so it is what keeps that branch building; the
// markers expand to nothing elsewhere. The define must precede every gpflash include, hence its own file.
#define GPFLASH_STAGE_TIMING 1

#include <armadillo>
#include <cmath>
#include <cstdio>

#include "GpFlashTestSupport.hpp"
#include "gpflash/training/EvalMllDispatch.hpp"

using gpflash_test::kDetailLen;
using gpflash_test::make_points;
using gpflash_test::report;

namespace {

// Wall times, so only usability can be asserted, not values. n_train is large enough not to round every stage to 0.
bool test_stage_timing_records(const char* name, const double sig2_f_val) {
    const int n_dim = 3, n_train = 120;
    const arma::mat X_ndim_ntrain_mat = make_points(n_dim, n_train);
    const arma::vec f_train_vec = make_points(n_train, 1);  // the (n_train, 1) case of the generator, read as a vector
    arma::vec hp_gamma_vec(n_dim);
    const arma::vec hp_kern_extra_vec;  // Matern52 declares none
    for (int i_dim = 0; i_dim < n_dim; ++i_dim) hp_gamma_vec(i_dim) = 0.8 + 0.2 * i_dim;

    // The noise-free driver solves hp_sig2_k in closed form, so it takes the neutral 1.
    const bool is_noise_free = gpflash::detail::classify_noise_request(sig2_f_val) == gpflash::NoiseRequest::NoiseFree;
    gpflash::detail::EvalVariances eval_variances;
    eval_variances.hp_sig2_k = is_noise_free ? 1.0 : 1.3;
    eval_variances.sig2_f_val = sig2_f_val;

    // Nothing resets the fields, so a stage this evaluation does not time keeps the negative value set here.
    const double unwritten_stage_s = -1.0;
    gpflash::detail::TimeStages& time_stages = gpflash::detail::time_stages();
    time_stages.kern_assembly_s = unwritten_stage_s;
    time_stages.chol_s = unwritten_stage_s;
    time_stages.lin_sol_s = unwritten_stage_s;
    time_stages.sigma_inv_s = unwritten_stage_s;
    time_stages.grad_s = unwritten_stage_s;

    gpflash::ModelOptions model_opts;
    model_opts.cond_max = gpflash::COND_MAX_DEFAULT;
    const gpflash::MllResult mll_result = gpflash::detail::eval_mll(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Matern52),
            gpflash::TrainingData{X_ndim_ntrain_mat, f_train_vec, eval_variances.sig2_f_val},
            gpflash::detail::build_hp_val(hp_gamma_vec, hp_kern_extra_vec, eval_variances.hp_sig2_k),
            model_opts,
            /*eval_grad=*/true);
    const double mll = mll_result.mll;
    const arma::vec& d_mll_d_hp_gamma_vec = mll_result.mll_hp_grad_struct.d_mll_d_hp_gamma_vec;

    const double stage_s_arr[] = {
            time_stages.kern_assembly_s,
            time_stages.chol_s,
            time_stages.lin_sol_s,
            time_stages.sigma_inv_s,
            time_stages.grad_s};
    bool every_stage_usable = true;
    double total_s = 0.0;
    for (const double one_stage_s : stage_s_arr) {
        every_stage_usable = every_stage_usable && std::isfinite(one_stage_s) && one_stage_s >= 0.0;
        total_s += one_stage_s;
    }
    // The gradient path runs all five stages, so the total cannot be zero at this size.
    const bool pass = every_stage_usable && total_s > 0.0 && std::isfinite(mll)
            && d_mll_d_hp_gamma_vec.n_elem == static_cast<arma::uword>(n_dim);

    // Seconds, so six decimals: every stage at this size is well under a millisecond.
    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "kern_assembly=%.6f chol=%.6f lin_sol=%.6f sigma_inv=%.6f grad=%.6f s",
            time_stages.kern_assembly_s,
            time_stages.chol_s,
            time_stages.lin_sol_s,
            time_stages.sigma_inv_s,
            time_stages.grad_s);

    return report(name, pass, detail);
}

}  // namespace

int main() {
    std::printf("== GpFlash stage timing ==\n");
    bool ok = true;
    ok &= test_stage_timing_records("stage_timing_records/noise_free", 0.0);
    ok &= test_stage_timing_records("stage_timing_records/noisy", 1.0e-2);
    std::printf("\n%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");

    return ok ? 0 : 1;
}
