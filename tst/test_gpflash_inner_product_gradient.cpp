// The inner-product family's marginal-log-likelihood hyperparameter gradient against central finite differences
// of the MLL.
//
// This is the only end-to-end MLL check of the varying-diagonal and preconditioned gradient branches: for a
// correlation-matrix kernel they are identically zero, so a bug in the diagonal derivative
// (calc_d_kern_diag_d_hp_gamma_mat) or the preconditioned terms passes every stationary-kernel test. n_train >
// n_dim makes K singular, so the nugget eta is load-bearing and its gradient carries real weight; cond_max
// stays well below the default, where a near-singular Sigma loses more precision than the tolerance allows.
// Linear fixes dk/d(inner_prod) = 1; Quadratic makes it vary with the pair. Quadratic also runs with the trace
// bound, whose eta must not move with the hyperparameters: a trace taken on K rather than K_corr fails.
#include <armadillo>
#include <cstdio>
#include <random>

#include "GpFlashTestSupport.hpp"
#include "gpflash/inputs/EigMaxUbMethod.hpp"  // EigMaxUbMethod
#include "gpflash/kernels/KernelDispatch.hpp"
#include "gpflash/kernels/inner_product/Linear.hpp"
#include "gpflash/kernels/inner_product/Quadratic.hpp"
#include "gpflash/training/EvalMllDispatch.hpp"

using gpflash_test::draw_uniform_vec;
using gpflash_test::kDetailLen;
using gpflash_test::kFdScaleFloor;
using gpflash_test::kFdStepObj;
using gpflash_test::kTolFdObj;
using gpflash_test::make_points;
using gpflash_test::report;

namespace {

// Asserts eval_mll's analytic d(MLL)/d(hp_gamma_l) equals a central finite difference of the MLL value. The
// value and the gradient are separate code paths, so differencing the value validates the gradient.
bool test_mll_hp_gamma_grad_vs_fd(
        const gpflash::kernel::KernelBase& kern_fun,
        const char* name,
        const double sig2_f_val,
        const gpflash::EigMaxUbMethod kmat_eig_max_ub_method = gpflash::EigMaxUbMethod::Gershgorin) {
    const int n_dim = 3, n_train = 14;
    const arma::mat X_ndim_ntrain_mat = make_points(n_dim, n_train);
    std::mt19937_64 rng(43);
    const arma::vec f_train_vec = draw_uniform_vec(rng, n_train, 2.0, 3.0);
    arma::vec hp_gamma_vec(n_dim);
    for (int i_dim = 0; i_dim < n_dim; ++i_dim) hp_gamma_vec(i_dim) = 0.7 + 0.4 * i_dim;
    arma::vec hp_kern_extra_vec(kern_fun.n_hp_kern_extra(n_dim));
    if (!hp_kern_extra_vec.is_empty()) hp_kern_extra_vec.fill(1.0);

    const bool is_noise_free = gpflash::detail::classify_noise_request(sig2_f_val) == gpflash::NoiseRequest::NoiseFree;
    const double hp_sig2_k = is_noise_free ? 1.0 : 2.3;

    // The checked entry point takes structs, and only hp_gamma varies below, so gather the rest once.
    const gpflash::TrainingData train_data = gpflash::TrainingData{X_ndim_ntrain_mat, f_train_vec, sig2_f_val};
    gpflash::ModelOptions model_opts;
    model_opts.cond_max = 1.0e4;
    model_opts.kmat_eig_max_ub_method = kmat_eig_max_ub_method;
    const auto mll_at_hp_gamma = [&](const arma::vec& hp_gamma_at, const bool eval_grad) {
        return gpflash::detail::eval_mll(
                kern_fun,
                train_data,
                gpflash::detail::build_hp_val(hp_gamma_at, hp_kern_extra_vec, hp_sig2_k),
                model_opts,
                eval_grad);
    };

    const arma::vec d_mll_d_hp_gamma_vec =
            mll_at_hp_gamma(hp_gamma_vec, /*eval_grad=*/true).mll_hp_grad_struct.d_mll_d_hp_gamma_vec;

    double worst = 0.0;
    for (int i_dim = 0; i_dim < n_dim; ++i_dim) {
        arma::vec hp_gamma_plus_vec = hp_gamma_vec, hp_gamma_minus_vec = hp_gamma_vec;
        hp_gamma_plus_vec(i_dim) += kFdStepObj;
        hp_gamma_minus_vec(i_dim) -= kFdStepObj;
        const double grad_fd = (mll_at_hp_gamma(hp_gamma_plus_vec, /*eval_grad=*/false).mll
                                - mll_at_hp_gamma(hp_gamma_minus_vec, /*eval_grad=*/false).mll)
                / (2.0 * kFdStepObj);
        worst = std::max(
                worst, std::fabs(d_mll_d_hp_gamma_vec(i_dim) - grad_fd) / std::max(std::fabs(grad_fd), kFdScaleFloor));
    }
    char detail[kDetailLen];
    std::snprintf(detail, sizeof(detail), "sigma_f^2 %.3f  worst rel %.2e", sig2_f_val, worst);

    return report(name, worst < kTolFdObj, detail);
}

}  // namespace

int main() {
    std::printf("== GpFlash inner-product kernel gradient ==\n");
    bool ok = true;
    // Linear: dk/d(inner_prod) = 1, so the diagonal derivative is a bare 2 hp_gamma_l x_il^2.
    ok &= test_mll_hp_gamma_grad_vs_fd(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Linear), "mll_hp_gamma_grad_vs_fd/Linear", 0.0);
    ok &= test_mll_hp_gamma_grad_vs_fd(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Linear),
            "mll_hp_gamma_grad_vs_fd/Linear_noisy",
            0.031);
    // Quadratic: dk/d(inner_prod) varies with the pair, so the diagonal derivative exercises its own factor.
    ok &= test_mll_hp_gamma_grad_vs_fd(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Quadratic),
            "mll_hp_gamma_grad_vs_fd/Quadratic",
            0.0);
    ok &= test_mll_hp_gamma_grad_vs_fd(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Quadratic),
            "mll_hp_gamma_grad_vs_fd/Quadratic_noisy",
            0.031);
    for (const double sig2_f_val : {0.0, 0.031})
        ok &= test_mll_hp_gamma_grad_vs_fd(
                gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Quadratic),
                "mll_hp_gamma_grad_vs_fd/Quadratic_trace",
                sig2_f_val,
                gpflash::EigMaxUbMethod::Trace);
    std::printf("\n%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");

    return ok ? 0 : 1;
}
