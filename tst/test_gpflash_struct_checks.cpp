/**
 * test_gpflash_struct_checks.cpp -- the struct checkers reject what they promise to reject.
 *
 * GPFLASH_CHECK runs every postcondition on every test, which catches a checker that wrongly rejects valid
 * data. Nothing there catches the opposite: a checker whose condition is inverted, or whose helper call has
 * two arguments swapped, accepts everything and the suite still passes. So each case below builds a valid
 * instance, asserts it is accepted, then breaks one guarded field at a time and asserts each break throws.
 *
 *   1) field_checks           the core/Errors.hpp primitives, called with literals
 *   2) assembly_postconditions   PreconScales, EtaTerms, KernelAssembly
 *   3) training_postconditions   CholTerms, MllHpGrad, MllResult
 *   4) gradient_postconditions   KernTermHpGrad, EtaTermHpGrad, HpGradInputs
 *   5) cache_checks            PosteriorPrecompute, CacheInfo
 *   6) query_checks            PosteriorOptions, PosteriorResult
 *   optimization/'s checks are tested beside that tier, in test_gpflash_local_optz, _multistart, _search_space
 *   and _optimizer.
 */
#include <armadillo>
#include <cstddef>
#include <cstdio>
#include <limits>

#include "GpFlashTestSupport.hpp"
#include "gpflash/assembly/CholTerms.hpp"
#include "gpflash/assembly/EtaTermHpGrad.hpp"
#include "gpflash/assembly/KernTermHpGrad.hpp"
#include "gpflash/assembly/KernelAssembly.hpp"
#include "gpflash/core/Errors.hpp"
#include "gpflash/kernels/HpGradInputs.hpp"
#include "gpflash/kernels/KernelDispatch.hpp"  // kernel_by_id
#include "gpflash/posterior_cache/CacheInfo.hpp"
#include "gpflash/posterior_cache/PosteriorPrecompute.hpp"
#include "gpflash/posterior_eval/PosteriorOptions.hpp"
#include "gpflash/posterior_eval/PosteriorResult.hpp"
#include "gpflash/preconditioning/EtaTerms.hpp"
#include "gpflash/preconditioning/PreconScales.hpp"
#include "gpflash/training/MllHpGrad.hpp"
#include "gpflash/training/MllResult.hpp"

using gpflash_test::accepts;
using gpflash_test::kDetailLen;
using gpflash_test::rejects;
using gpflash_test::report;

namespace {

// constexpr, not arma::datum: Armadillo 9.870's datum is dynamically initialized, so a namespace-scope copy of it
// can run first and read 0.0 (static initialization order).
constexpr double kNan = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

// Index of the first case that did not hold, or -1 when every case held.
template <std::size_t N_CASE>
int find_first_failed_case(const bool (&is_case_held)[N_CASE]) {
    for (std::size_t i_case = 0; i_case < N_CASE; ++i_case)
        if (!is_case_held[i_case]) return static_cast<int>(i_case);

    return -1;
}

// ── 1) The field checks in core/Errors.hpp ───────────────────────────────────

bool test_field_checks(const char* name) {
    using namespace gpflash::detail;
    const arma::vec unit_vec = arma::ones<arma::vec>(3);
    int sink = 0;

    const bool accepted = accepts([&] {
        check_not_nan_val(kInf, "f", "test");  // an infinity is a value here, only NaN is not
        check_non_negative_or_inf_val(kInf, "f", "test");
        check_ptr_not_null(&sink, "f", "test");
        check_idx_or_inactive(-1, 4, "f", "test");  // -1 marks an absent position
        check_idx_or_inactive(3, 4, "f", "test");
        check_non_negative_count(0, "f", "test");
        check_all_non_negative_vals(arma::zeros<arma::vec>(3), "f", "test");  // zero is valid for a variance
        check_mat_is_square(arma::eye<arma::mat>(3, 3), "f", "test");
    });

    const bool field_case_rejected[] = {
            rejects([&] { check_not_nan_val(kNan, "f", "test"); }),
            rejects([&] { check_non_negative_or_inf_val(-1.0, "f", "test"); }),
            rejects([&] { check_non_negative_or_inf_val(kNan, "f", "test"); }),
            rejects([&] { check_ptr_not_null(nullptr, "f", "test"); }),
            rejects([&] { check_idx_or_inactive(-2, 4, "f", "test"); }),
            rejects([&] { check_idx_or_inactive(4, 4, "f", "test"); }),  // one past the last entry
            rejects([&] { check_non_negative_count(-1, "f", "test"); }),
            rejects([&] { check_all_non_negative_vals(-unit_vec, "f", "test"); }),
            rejects([&] { check_mat_is_square(arma::ones<arma::mat>(3, 4), "f", "test"); })};
    const int idx_field_unrejected = find_first_failed_case(field_case_rejected);

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "accepted %d  first unrejected: field %d",
            static_cast<int>(accepted),
            idx_field_unrejected);
    return report(name, accepted && idx_field_unrejected == -1, detail);
}

// ── 2) assembly/ and preconditioning/ ────────────────────────────────────────

// All three vectors positive, or all empty where P = I.
gpflash::detail::PreconScales valid_precon_scales() {
    gpflash::detail::PreconScales precon_scales;
    precon_scales.M_diag_vec = arma::ones<arma::vec>(3);
    precon_scales.precon_scale_vec = arma::ones<arma::vec>(3);
    precon_scales.inv_precon_scale_vec = arma::ones<arma::vec>(3);
    return precon_scales;
}

gpflash::detail::EtaTerms valid_eta_terms() {
    gpflash::detail::EtaTerms eta_terms;
    eta_terms.eig_max_ub_M_precon = 2.0;
    eta_terms.cond_max = 1.0e10;
    eta_terms.eta = 1.0e-6;
    eta_terms.precon_scales = valid_precon_scales();
    return eta_terms;
}

bool test_assembly_postconditions(const char* name) {
    using gpflash::detail::check_eta_terms;
    using gpflash::detail::check_kernel_assembly;
    using gpflash::detail::check_precon_scales;

    // Ydot is what K was built from and there is one norm per training point, so both size against the buffer.
    gpflash::detail::KernelAssembly kernel_assembly;
    kernel_assembly.Buffer_mat = arma::eye<arma::mat>(3, 3);
    kernel_assembly.kmat_eta_terms = valid_eta_terms();
    kernel_assembly.Ydot_nfeature_ntrain_mat = arma::ones<arma::mat>(2, 3);
    kernel_assembly.ydot_train_sq_norm_vec = arma::ones<arma::vec>(3);

    // The lower triangle with the diagonal of the 3 by 3 buffer.
    const arma::uword n_mirror_packed_ref = 6;

    const bool accepted = accepts([&] {
        check_precon_scales(valid_precon_scales(), "test");
        check_precon_scales(gpflash::detail::PreconScales{}, "test");  // all empty is P = I
        check_eta_terms(valid_eta_terms(), "test");
        check_kernel_assembly(kernel_assembly, "test");
        gpflash::detail::KernelAssembly mirrored = kernel_assembly;
        mirrored.kern_common_rdot_mirror_packed_vec = arma::ones<arma::vec>(n_mirror_packed_ref);
        check_kernel_assembly(mirrored, "test");
    });

    const bool scales_case_rejected[] = {
            rejects([&] {
                gpflash::detail::PreconScales precon_scales = valid_precon_scales();
                precon_scales.precon_scale_vec(1) = -1.0;  // a scale is a square root of M's diagonal
                check_precon_scales(precon_scales, "test");
            }),
            rejects([&] {
                gpflash::detail::PreconScales precon_scales = valid_precon_scales();
                precon_scales.inv_precon_scale_vec(2) = -1.0;
                check_precon_scales(precon_scales, "test");
            })};

    // calc_eta divides by cond_max - 1, so 1.0 is not a usable target.
    const bool eta_case_rejected[] = {
            rejects([&] {
                gpflash::detail::EtaTerms eta_terms = valid_eta_terms();
                eta_terms.cond_max = 1.0;
                check_eta_terms(eta_terms, "test");
            }),
            rejects([&] {
                gpflash::detail::EtaTerms eta_terms = valid_eta_terms();
                eta_terms.eta = -1.0e-6;
                check_eta_terms(eta_terms, "test");
            }),
            rejects([&] {
                gpflash::detail::EtaTerms eta_terms = valid_eta_terms();
                eta_terms.eig_max_ub_M_precon = kNan;
                check_eta_terms(eta_terms, "test");
            }),
            rejects([&] {
                // A gathered row is indexed by the bound's row, so the index must address it.
                gpflash::detail::EtaTerms eta_terms = valid_eta_terms();
                eta_terms.active_row_M_precon_vec = arma::ones<arma::vec>(4);
                eta_terms.idx_M_precon_max_abs_row_sum = 4;
                check_eta_terms(eta_terms, "test");
            }),
            rejects([&] {
                // The index stays inactive, so only the row's entry is wrong.
                gpflash::detail::EtaTerms eta_terms = valid_eta_terms();
                eta_terms.active_row_M_precon_vec = arma::ones<arma::vec>(4);
                eta_terms.active_row_M_precon_vec(1) = kNan;
                check_eta_terms(eta_terms, "test");
            }),
            rejects([&] {
                // The composed PreconScales check must fire through EtaTerms.
                gpflash::detail::EtaTerms eta_terms = valid_eta_terms();
                eta_terms.precon_scales.M_diag_vec(0) = -1.0;
                check_eta_terms(eta_terms, "test");
            })};

    const bool assembly_case_rejected[] = {
            rejects([&] {
                gpflash::detail::KernelAssembly broken = kernel_assembly;
                broken.Buffer_mat =
                        arma::ones<arma::mat>(3, 4);  // later stages overwrite it in place, so it stays square
                check_kernel_assembly(broken, "test");
            }),
            rejects([&] {
                // Ydot's column count is sized against the buffer and fails too, so the empty check cannot be
                // isolated.
                gpflash::detail::KernelAssembly broken = kernel_assembly;
                broken.Buffer_mat.reset();
                check_kernel_assembly(broken, "test");
            }),
            rejects([&] {
                // The composed EtaTerms check must fire through KernelAssembly.
                gpflash::detail::KernelAssembly broken = kernel_assembly;
                broken.kmat_eta_terms.cond_max = 1.0;
                check_kernel_assembly(broken, "test");
            }),
            rejects([&] {
                // One column per training point, or the gradient's map derivative reads
                // past the points it was built from.
                gpflash::detail::KernelAssembly broken = kernel_assembly;
                broken.Y_nfeature_ntrain_mat = arma::ones<arma::mat>(2, 4);
                check_kernel_assembly(broken, "test");
            }),
            rejects([&] {
                gpflash::detail::KernelAssembly broken = kernel_assembly;
                broken.Ydot_nfeature_ntrain_mat = arma::ones<arma::mat>(2, 4);
                check_kernel_assembly(broken, "test");
            }),
            rejects([&] {
                // The posterior moves these out and stores them, so a missing Ydot is
                // a precompute with no points to query against.
                gpflash::detail::KernelAssembly broken = kernel_assembly;
                broken.Ydot_nfeature_ntrain_mat.set_size(0, 3);  // one column per point, so only the emptiness is wrong
                check_kernel_assembly(broken, "test");
            }),
            rejects([&] {
                gpflash::detail::KernelAssembly broken = kernel_assembly;
                broken.ydot_train_sq_norm_vec = arma::ones<arma::vec>(4);  // not n_train
                check_kernel_assembly(broken, "test");
            }),
            rejects([&] {
                // One entry short, which the symmetry gradient would read past.
                gpflash::detail::KernelAssembly broken = kernel_assembly;
                broken.kern_common_rdot_mirror_packed_vec = arma::ones<arma::vec>(n_mirror_packed_ref - 1);
                check_kernel_assembly(broken, "test");
            }),
            rejects([&] {
                gpflash::detail::KernelAssembly broken = kernel_assembly;
                broken.kern_common_rdot_mirror_packed_vec = arma::ones<arma::vec>(n_mirror_packed_ref);
                broken.kern_common_rdot_mirror_packed_vec(3) = kNan;
                check_kernel_assembly(broken, "test");
            })};

    const int idx_scales_unrejected = find_first_failed_case(scales_case_rejected);
    const int idx_eta_unrejected = find_first_failed_case(eta_case_rejected);
    const int idx_assembly_unrejected = find_first_failed_case(assembly_case_rejected);

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "accepted %d  first unrejected: scales %d  eta %d  assembly %d",
            static_cast<int>(accepted),
            idx_scales_unrejected,
            idx_eta_unrejected,
            idx_assembly_unrejected);
    return report(
            name,
            accepted && idx_scales_unrejected == -1 && idx_eta_unrejected == -1 && idx_assembly_unrejected == -1,
            detail);
}

// ── 3) the training postconditions ───────────────────────────────────────────

gpflash::detail::CholTerms valid_chol_terms() {
    gpflash::detail::CholTerms chol_terms;
    chol_terms.hp_beta_vec = {0.5};
    chol_terms.lin_sol_L_residual_vec = arma::ones<arma::vec>(4);
    chol_terms.lin_sol_L_residual_sq_norm = 4.0;
    chol_terms.ln_det = -1.25;  // ln|Sigma| is any finite number
    return chol_terms;
}

bool test_training_postconditions(const char* name) {
    using gpflash::detail::check_chol_terms;
    using gpflash::detail::check_mll_hp_grad;
    using gpflash::detail::check_mll_result;

    gpflash::MllHpGrad mll_hp_grad_struct;
    mll_hp_grad_struct.d_mll_d_hp_gamma_vec = arma::ones<arma::vec>(2);

    gpflash::MllResult mll_result;
    mll_result.mll = -12.5;
    mll_result.hp_beta_vec = {0.5};
    mll_result.mll_hp_grad_struct = mll_hp_grad_struct;

    const bool accepted = accepts([&] {
        check_chol_terms(valid_chol_terms(), "test");
        check_mll_hp_grad(mll_hp_grad_struct, "test");
        check_mll_hp_grad(gpflash::MllHpGrad{}, "test");  // no gradient requested
        check_mll_result(mll_result, "test");
    });

    const bool chol_case_rejected[] = {
            rejects([&] {
                gpflash::detail::CholTerms chol_terms = valid_chol_terms();
                chol_terms.ln_det = kInf;  // a Cholesky that succeeded has a finite log determinant
                check_chol_terms(chol_terms, "test");
            }),
            rejects([&] {
                gpflash::detail::CholTerms chol_terms = valid_chol_terms();
                chol_terms.lin_sol_L_residual_sq_norm = -1.0;
                check_chol_terms(chol_terms, "test");
            }),
            rejects([&] {
                gpflash::detail::CholTerms chol_terms = valid_chol_terms();
                chol_terms.lin_sol_L_residual_vec.reset();
                check_chol_terms(chol_terms, "test");
            }),
            rejects([&] {
                gpflash::detail::CholTerms chol_terms = valid_chol_terms();
                chol_terms.lin_sol_L_residual_vec(2) = kNan;
                check_chol_terms(chol_terms, "test");
            }),
            rejects([&] {
                gpflash::detail::CholTerms chol_terms = valid_chol_terms();
                chol_terms.hp_beta_vec = {kNan};
                check_chol_terms(chol_terms, "test");
            })};

    const bool grad_case_rejected[] = {
            rejects([&] {
                gpflash::MllHpGrad broken = mll_hp_grad_struct;
                broken.d_mll_d_hp_gamma_vec(0) = kNan;
                check_mll_hp_grad(broken, "test");
            }),
            rejects([&] {
                gpflash::MllHpGrad broken = mll_hp_grad_struct;
                broken.d_mll_d_hp_kern_extra_vec = {kNan};
                check_mll_hp_grad(broken, "test");
            }),
            rejects([&] {
                gpflash::MllHpGrad broken = mll_hp_grad_struct;
                broken.d_mll_d_hp_sig2_k = kInf;  // a derivative is finite; either sign is legal
                check_mll_hp_grad(broken, "test");
            }),
            rejects([&] {
                gpflash::MllHpGrad broken = mll_hp_grad_struct;
                broken.d_mll_d_hp_sig2_f_val = kInf;
                check_mll_hp_grad(broken, "test");
            })};

    const bool result_case_rejected[] = {
            rejects([&] {
                gpflash::MllResult broken = mll_result;
                broken.mll = kNan;
                check_mll_result(broken, "test");
            }),
            rejects([&] {
                gpflash::MllResult broken = mll_result;
                broken.hp_beta_vec = {kNan};
                check_mll_result(broken, "test");
            }),
            rejects([&] {
                gpflash::MllResult broken = mll_result;
                broken.hp_sig2_k = -1.0;  // a variance
                check_mll_result(broken, "test");
            }),
            rejects([&] {
                // The composed MllHpGrad check must fire through MllResult.
                gpflash::MllResult broken = mll_result;
                broken.mll_hp_grad_struct.d_mll_d_hp_gamma_vec(1) = kNan;
                check_mll_result(broken, "test");
            })};

    const int idx_chol_unrejected = find_first_failed_case(chol_case_rejected);
    const int idx_grad_unrejected = find_first_failed_case(grad_case_rejected);
    const int idx_result_unrejected = find_first_failed_case(result_case_rejected);

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "accepted %d  first unrejected: chol %d  grad %d  result %d",
            static_cast<int>(accepted),
            idx_chol_unrejected,
            idx_grad_unrejected,
            idx_result_unrejected);
    return report(
            name,
            accepted && idx_chol_unrejected == -1 && idx_grad_unrejected == -1 && idx_result_unrejected == -1,
            detail);
}

// ── 4) the gradient inputs and terms ─────────────────────────────────────────

bool test_gradient_postconditions(const char* name) {
    using gpflash::detail::check_eta_term_hp_grad;
    using gpflash::detail::check_kern_term_hp_grad;
    using gpflash::kernel::check_hp_grad_inputs;

    const int n_dim = 2, n_train = 5;

    gpflash::detail::KernTermHpGrad kern_term_hp_grad;
    kern_term_hp_grad.d_mll_d_hp_gamma_vec = arma::ones<arma::vec>(n_dim);
    kern_term_hp_grad.active_row_kern_common_rdot_vec = arma::ones<arma::vec>(n_train);

    gpflash::detail::EtaTermHpGrad eta_term_hp_grad;
    eta_term_hp_grad.d_mll_d_hp_gamma_vec = arma::ones<arma::vec>(n_dim);

    gpflash::kernel::HpGradInputs hp_grad_inputs;
    hp_grad_inputs.hp_gamma_vec = arma::ones<arma::vec>(n_dim);
    const arma::mat X_ndim_ntrain_mat = arma::ones<arma::mat>(n_dim, n_train);
    hp_grad_inputs.X_ndim_ntrain_mat_ptr = &X_ndim_ntrain_mat;
    hp_grad_inputs.X_ntrain_ndim_mat = arma::ones<arma::mat>(n_train, n_dim);

    const bool accepted = accepts([&] {
        check_kern_term_hp_grad(kern_term_hp_grad, "test");
        check_eta_term_hp_grad(eta_term_hp_grad, "test");
        check_hp_grad_inputs(hp_grad_inputs, "test");
    });

    const bool terms_case_rejected[] = {
            rejects([&] {
                gpflash::detail::KernTermHpGrad broken = kern_term_hp_grad;
                broken.active_row_kern_common_rdot_vec(2) = kNan;
                check_kern_term_hp_grad(broken, "test");
            }),
            rejects([&] {
                gpflash::detail::KernTermHpGrad broken = kern_term_hp_grad;
                broken.d_mll_d_hp_gamma_vec(0) = kInf;
                check_kern_term_hp_grad(broken, "test");
            }),
            rejects([&] {
                gpflash::detail::KernTermHpGrad broken = kern_term_hp_grad;
                broken.d_mll_d_hp_kern_extra_vec = {kNan};
                check_kern_term_hp_grad(broken, "test");
            }),
            rejects([&] {
                gpflash::detail::KernTermHpGrad broken = kern_term_hp_grad;
                broken.active_row_kern_common_rdot_mirror_vec = arma::ones<arma::vec>(n_train);
                broken.active_row_kern_common_rdot_mirror_vec(3) = kNan;
                check_kern_term_hp_grad(broken, "test");
            }),
            rejects([&] {
                gpflash::detail::EtaTermHpGrad broken = eta_term_hp_grad;
                broken.d_mll_d_hp_gamma_vec(1) = kNan;
                check_eta_term_hp_grad(broken, "test");
            }),
            rejects([&] {
                gpflash::detail::EtaTermHpGrad broken = eta_term_hp_grad;
                broken.d_mll_d_hp_kern_extra_vec = {kNan};
                check_eta_term_hp_grad(broken, "test");
            })};

    // One length scale per column of X, or the gradient loops read past a block.
    const bool inputs_case_rejected[] = {
            rejects([&] {
                gpflash::kernel::HpGradInputs broken = hp_grad_inputs;
                broken.hp_gamma_vec = arma::ones<arma::vec>(n_dim + 1);
                check_hp_grad_inputs(broken, "test");
            }),
            rejects([&] {
                gpflash::kernel::HpGradInputs broken = hp_grad_inputs;
                // No columns and no length scales, so only the emptiness is wrong.
                broken.X_ntrain_ndim_mat.set_size(n_train, 0);
                broken.hp_gamma_vec.reset();
                check_hp_grad_inputs(broken, "test");
            }),
            rejects([&] {
                gpflash::kernel::HpGradInputs broken = hp_grad_inputs;
                broken.X_ndim_ntrain_mat_ptr = nullptr;
                check_hp_grad_inputs(broken, "test");
            }),
            rejects([&] {
                gpflash::kernel::HpGradInputs broken = hp_grad_inputs;
                const arma::mat X_one_extra_mat = arma::ones<arma::mat>(n_dim, n_train + 1);  // one point too many
                broken.X_ndim_ntrain_mat_ptr = &X_one_extra_mat;
                check_hp_grad_inputs(broken, "test");
            }),
            rejects([&] {
                gpflash::kernel::HpGradInputs broken = hp_grad_inputs;
                broken.hp_kern_extra_vec = {0.0};  // natural space, so positive
                check_hp_grad_inputs(broken, "test");
            }),
            rejects([&] {
                gpflash::kernel::HpGradInputs broken = hp_grad_inputs;
                broken.Y_ntrain_nfeature_mat = arma::ones<arma::mat>(n_train, 2 * n_dim);  // Periodic's width
                broken.Y_ntrain_nfeature_mat(1, 0) = kNan;
                check_hp_grad_inputs(broken, "test");
            })};

    const int idx_terms_unrejected = find_first_failed_case(terms_case_rejected);
    const int idx_inputs_unrejected = find_first_failed_case(inputs_case_rejected);

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "accepted %d  first unrejected: terms %d  inputs %d",
            static_cast<int>(accepted),
            idx_terms_unrejected,
            idx_inputs_unrejected);
    return report(name, accepted && idx_terms_unrejected == -1 && idx_inputs_unrejected == -1, detail);
}

// ── 5) posterior_cache/ ───────────────────────────────────────────────────────────

// n_feature must map back to n_dim through the kernel, which for a radial kernel means they are equal.
gpflash::detail::PosteriorPrecompute valid_precompute(const int n_dim) {
    gpflash::detail::PosteriorPrecompute post_precompute;
    post_precompute.n_dim = n_dim;
    post_precompute.n_feature = n_dim;
    post_precompute.x_center_vec = arma::zeros<arma::vec>(n_dim);
    post_precompute.hp_gamma_vec = arma::ones<arma::vec>(n_dim);
    post_precompute.mean_model.degree_vec = arma::zeros<arma::uvec>(n_dim);  // the constant mean model
    post_precompute.hp_beta_vec = {0.25};                                    // its one weight
    return post_precompute;
}

gpflash::cache::CacheInfo valid_cache_info() {
    gpflash::cache::CacheInfo cache_info;
    cache_info.format_version = 1;
    cache_info.post_model_info.kernel_id = gpflash::kernel::KernelId::SqExp;
    cache_info.n_train = 7;
    cache_info.n_dim = 3;
    cache_info.n_feature = 3;
    cache_info.post_model_info.mll = -8.5;
    cache_info.hp_sig2_k = 1.5;
    cache_info.post_model_info.sig2_f_val = 0.013;
    cache_info.post_model_info.cond_max = 1e10;
    cache_info.post_model_info.kmat_eig_max_ub_method = gpflash::EigMaxUbMethod::Gershgorin;
    cache_info.post_model_info.mean_model_eig_max_ub_method = gpflash::EigMaxUbMethod::Trace;
    return cache_info;
}

bool test_cache_checks(const char* name) {
    using gpflash::cache::check_cache_info;
    using gpflash::detail::check_posterior_precompute;

    const int n_dim = 3;

    const bool accepted = accepts([&] {
        check_posterior_precompute(
                gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::SqExp), valid_precompute(n_dim), "test");
        check_cache_info(valid_cache_info(), "test");
        gpflash::cache::CacheInfo noise_free = valid_cache_info();  // sig2_f_val == 0 is the noise-free fit
        noise_free.post_model_info.sig2_f_val = 0.0;
        check_cache_info(noise_free, "test");
        // An even mirror on a kernel that admits one, with the constant mean model, so the mirror's checks pass too.
        gpflash::detail::PosteriorPrecompute mirrored = valid_precompute(n_dim);
        mirrored.symmetry_spec.mirror_sign_vec = {1.0, -1.0, 1.0};
        check_posterior_precompute(gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::SqExp), mirrored, "test");
    });

    // A file saved for another kernel is what these catch, so each shape mismatch must fire.
    const bool precompute_case_rejected[] = {
            rejects([&] {
                gpflash::detail::PosteriorPrecompute broken = valid_precompute(n_dim);
                broken.hp_gamma_vec = arma::ones<arma::vec>(n_dim + 1);
                check_posterior_precompute(
                        gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::SqExp), broken, "test");
            }),
            rejects([&] {
                gpflash::detail::PosteriorPrecompute broken = valid_precompute(n_dim);
                broken.x_center_vec = arma::zeros<arma::vec>(n_dim + 1);
                check_posterior_precompute(
                        gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::SqExp), broken, "test");
            }),
            rejects([&] {
                gpflash::detail::PosteriorPrecompute broken = valid_precompute(n_dim);
                broken.hp_kern_extra_vec = {1.0};  // SqExp has no extras
                check_posterior_precompute(
                        gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::SqExp), broken, "test");
            }),
            rejects([&] {
                gpflash::detail::PosteriorPrecompute broken = valid_precompute(n_dim);
                broken.n_feature = n_dim + 1;  // no radial kernel maps a wider feature space back to n_dim
                check_posterior_precompute(
                        gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::SqExp), broken, "test");
            }),
            rejects([&] {
                // Every width consistent with n_dim = 0, so only the count is wrong.
                gpflash::detail::PosteriorPrecompute broken = valid_precompute(n_dim);
                broken.n_dim = 0;
                broken.n_feature = 0;
                broken.hp_gamma_vec.reset();
                broken.mean_model.degree_vec.reset();
                check_posterior_precompute(
                        gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::SqExp), broken, "test");
            }),
            rejects([&] {
                gpflash::detail::PosteriorPrecompute broken = valid_precompute(n_dim);
                broken.hp_beta_vec = {0.25, 0.5};  // one weight per mean model basis function
                check_posterior_precompute(
                        gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::SqExp), broken, "test");
            }),
            rejects([&] {
                gpflash::detail::PosteriorPrecompute broken = valid_precompute(n_dim);
                broken.hp_beta_vec = {kNan};  // read off disk, so it must be finite to predict with
                check_posterior_precompute(
                        gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::SqExp), broken, "test");
            }),
            rejects([&] {
                gpflash::detail::PosteriorPrecompute broken = valid_precompute(n_dim);
                broken.mean_model.degree_vec = arma::zeros<arma::uvec>(n_dim + 1);  // one degree per input dimension
                check_posterior_precompute(
                        gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::SqExp), broken, "test");
            }),
            rejects([&] {
                gpflash::detail::PosteriorPrecompute broken = valid_precompute(n_dim);
                const arma::uword mean_model_degree_past_max_ref = 3;  // one past the highest degree accepted
                broken.mean_model.degree_vec(0) = mean_model_degree_past_max_ref;
                broken.hp_beta_vec = arma::zeros<arma::vec>(broken.mean_model.n_basis());  // widths agree
                check_posterior_precompute(
                        gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::SqExp), broken, "test");
            }),
            rejects([&] {
                // A valid mirror one dimension short, which a query would mirror its point with.
                gpflash::detail::PosteriorPrecompute broken = valid_precompute(n_dim);
                broken.symmetry_spec.mirror_sign_vec = {1.0, -1.0};
                check_posterior_precompute(
                        gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::SqExp), broken, "test");
            }),
            rejects([&] {
                // The composed SymmetrySpec check must fire: a mirror that flips nothing.
                gpflash::detail::PosteriorPrecompute broken = valid_precompute(n_dim);
                broken.symmetry_spec.mirror_sign_vec = {1.0, 1.0, 1.0};
                check_posterior_precompute(
                        gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::SqExp), broken, "test");
            }),
            rejects([&] {
                // A mirror on a kernel that admits none. Linear takes the base widths, so only the mirror is wrong.
                gpflash::detail::PosteriorPrecompute broken = valid_precompute(n_dim);
                broken.symmetry_spec.mirror_sign_vec = {1.0, -1.0, 1.0};
                check_posterior_precompute(
                        gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Linear), broken, "test");
            }),
            rejects([&] {
                // An odd mirror with the constant mean model, which the odd parity does not admit.
                gpflash::detail::PosteriorPrecompute broken = valid_precompute(n_dim);
                broken.symmetry_spec.mirror_sign_vec = {1.0, -1.0, 1.0};
                broken.symmetry_spec.is_even_symmetry = false;
                check_posterior_precompute(
                        gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::SqExp), broken, "test");
            })};

    const bool cache_case_rejected[] = {
            rejects([&] {
                gpflash::cache::CacheInfo broken = valid_cache_info();
                broken.format_version = 0;  // no file carries version 0
                check_cache_info(broken, "test");
            }),
            rejects([&] {
                gpflash::cache::CacheInfo broken = valid_cache_info();
                broken.post_model_info.kernel_id =
                        gpflash::kernel::KernelId::End;  // the list terminator names no kernel
                check_cache_info(broken, "test");
            }),
            rejects([&] {
                gpflash::cache::CacheInfo broken = valid_cache_info();
                broken.post_model_info.mll = kNan;  // fits are ranked on this
                check_cache_info(broken, "test");
            }),
            rejects([&] {
                gpflash::cache::CacheInfo broken = valid_cache_info();
                broken.version_major = -1;
                check_cache_info(broken, "test");
            }),
            rejects([&] {
                gpflash::cache::CacheInfo broken = valid_cache_info();
                broken.version_minor = -1;
                check_cache_info(broken, "test");
            }),
            rejects([&] {
                gpflash::cache::CacheInfo broken = valid_cache_info();
                broken.version_patch = -1;
                check_cache_info(broken, "test");
            }),
            rejects([&] {
                gpflash::cache::CacheInfo broken = valid_cache_info();
                broken.n_train = 0;  // the shape a reader sizes a refit against
                check_cache_info(broken, "test");
            }),
            rejects([&] {
                gpflash::cache::CacheInfo broken = valid_cache_info();
                broken.n_dim = 0;
                check_cache_info(broken, "test");
            }),
            rejects([&] {
                gpflash::cache::CacheInfo broken = valid_cache_info();
                broken.n_feature = 0;
                check_cache_info(broken, "test");
            }),
            rejects([&] {
                gpflash::cache::CacheInfo broken = valid_cache_info();
                broken.hp_sig2_k = -1.0;  // the unset sentinel
                check_cache_info(broken, "test");
            }),
            rejects([&] {
                // Zero is legal here -- noise-free data -- so only a negative one is wrong.
                gpflash::cache::CacheInfo broken = valid_cache_info();
                broken.post_model_info.sig2_f_val = -1.0;
                check_cache_info(broken, "test");
            }),
            rejects([&] {
                gpflash::cache::CacheInfo broken = valid_cache_info();
                broken.post_model_info.cond_max = 1.0;  // a conditioning limit must exceed one
                check_cache_info(broken, "test");
            }),
            rejects([&] {
                gpflash::cache::CacheInfo broken = valid_cache_info();
                broken.post_model_info.kmat_eig_max_ub_method = gpflash::EigMaxUbMethod::End;
                check_cache_info(broken, "test");
            }),
            rejects([&] {
                gpflash::cache::CacheInfo broken = valid_cache_info();
                broken.post_model_info.mean_model_eig_max_ub_method = gpflash::EigMaxUbMethod::End;
                check_cache_info(broken, "test");
            }),
            rejects([&] {
                gpflash::cache::CacheInfo broken = valid_cache_info();
                broken.symmetry_kind = gpflash::cache::SymmetryKind::End;  // names no symmetry
                check_cache_info(broken, "test");
            })};

    const int idx_precompute_unrejected = find_first_failed_case(precompute_case_rejected);
    const int idx_cache_unrejected = find_first_failed_case(cache_case_rejected);

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "accepted %d  first unrejected: precompute %d  cache %d",
            static_cast<int>(accepted),
            idx_precompute_unrejected,
            idx_cache_unrejected);
    return report(name, accepted && idx_precompute_unrejected == -1 && idx_cache_unrejected == -1, detail);
}

// ── 6) posterior_eval/ ────────────────────────────────────────────────────────────

bool test_query_checks(const char* name) {
    using gpflash::detail::check_posterior_result;
    using gpflash::detail::is_empty_request;

    const int n_dim = 2, n_post = 4;

    gpflash::PosteriorResult posterior_result;
    posterior_result.mean_post_vec = arma::ones<arma::vec>(n_post);
    posterior_result.var_post_vec = arma::ones<arma::vec>(n_post);
    posterior_result.d_mean_post_d_x_post_ndim_npost_mat = arma::ones<arma::mat>(n_dim, n_post);
    posterior_result.d_var_post_d_x_post_ndim_npost_mat = arma::ones<arma::mat>(n_dim, n_post);

    const bool accepted = accepts([&] {
        check_posterior_result(posterior_result, "test");
        check_posterior_result(gpflash::PosteriorResult{}, "test");  // nothing requested, so nothing to relate
    });

    // An empty request is legal, so what must hold is that it is recognized -- eval returns early on it, and
    // neither a default request nor a gradient-only one may be mistaken for one.
    gpflash::PosteriorOptions empty_opts;
    empty_opts.return_mean = false;
    empty_opts.return_var = false;
    empty_opts.return_mean_gradient = false;
    empty_opts.return_var_gradient = false;
    gpflash::PosteriorOptions mean_gradient_only_opts = empty_opts;
    mean_gradient_only_opts.return_mean_gradient = true;
    gpflash::PosteriorOptions var_gradient_only_opts = empty_opts;
    var_gradient_only_opts.return_var_gradient = true;
    const bool is_empty_recognised = is_empty_request(empty_opts);
    const bool is_default_nonempty = !is_empty_request(gpflash::PosteriorOptions{});
    const bool is_gradient_only_nonempty =
            !is_empty_request(mean_gradient_only_opts) && !is_empty_request(var_gradient_only_opts);

    const bool result_case_rejected[] = {
            rejects([&] {
                gpflash::PosteriorResult broken = posterior_result;
                broken.var_post_vec(1) = -1.0;  // a posterior variance; zero is reachable at a training point
                check_posterior_result(broken, "test");
            }),
            rejects([&] {
                gpflash::PosteriorResult broken = posterior_result;
                broken.mean_post_vec(0) = kNan;
                check_posterior_result(broken, "test");
            }),
            rejects([&] {
                gpflash::PosteriorResult broken = posterior_result;
                broken.d_mean_post_d_x_post_ndim_npost_mat(1, 2) = kNan;
                check_posterior_result(broken, "test");
            }),
            rejects([&] {
                gpflash::PosteriorResult broken = posterior_result;
                broken.d_var_post_d_x_post_ndim_npost_mat(0, 3) = kInf;
                check_posterior_result(broken, "test");
            })};
    const int idx_result_unrejected = find_first_failed_case(result_case_rejected);

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "accepted %d  empty %d  default %d  gradient_only %d  first unrejected: result %d",
            static_cast<int>(accepted),
            static_cast<int>(is_empty_recognised),
            static_cast<int>(is_default_nonempty),
            static_cast<int>(is_gradient_only_nonempty),
            idx_result_unrejected);
    return report(
            name,
            accepted && is_empty_recognised && is_default_nonempty && is_gradient_only_nonempty
                    && idx_result_unrejected == -1,
            detail);
}

}  // namespace

int main() {
    std::printf("== GpFlash struct checks ==\n");
    bool ok = true;
    ok &= test_field_checks("field_checks");
    ok &= test_assembly_postconditions("assembly_postconditions");
    ok &= test_training_postconditions("training_postconditions");
    ok &= test_gradient_postconditions("gradient_postconditions");
    ok &= test_cache_checks("cache_checks");
    ok &= test_query_checks("query_checks");
    std::printf("\n%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return ok ? 0 : 1;
}
