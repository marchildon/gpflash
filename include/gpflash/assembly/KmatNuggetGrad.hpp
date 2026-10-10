/**
 * KmatNuggetGrad.hpp -- what the kernel matrix's nugget contributes to the hyperparameter gradient.
 *
 *   Sigma = sigma_k^2 (K + eta W) + sigma_f^2 I,   W = P P = diag(K)
 *
 * eta and W both depend on the kernel hyperparameters, so d(eta)/d(hp) enters the gradient weighted by tr(Psi W).
 * Neither variance hyperparameter has an eta term, P being built from diag(K) alone.
 *
 * d(diag K) arrives as an argument rather than a call back into the kernel, so the caller evaluates it once.
 */
#pragma once

#include <armadillo>
#include <string>

#include "gpflash/assembly/EtaTermHpGrad.hpp"    // EtaTermHpGrad, check_eta_term_hp_grad
#include "gpflash/core/Errors.hpp"               // ErrorNotImplemented
#include "gpflash/core/Platform.hpp"             // GPFLASH_CHECK
#include "gpflash/inputs/SymmetrySpec.hpp"       // SymmetrySpec
#include "gpflash/kernels/HpGradInputs.hpp"      // kernel::HpGradInputs
#include "gpflash/kernels/KernelBase.hpp"        // kernel::KernelBase
#include "gpflash/preconditioning/EtaTerms.hpp"  // EtaTerms

namespace gpflash {
namespace detail {

// The d(P)/d(hp) part of d(eta)/d(hp), which exists because K_corr = P^-1 K P^-1 carries P on both sides.
// With p_i = sqrt(K_ii) the entries of diag(P), and dln_p(i, l) the log-scale derivative held in
// d_ln_precon_scale_d_hp_ntrain_nhp_mat:
//
//   dln_p(i, l)          = d(ln p_i)/d(hp_l) = d(K_ii)/d(hp_l) / (2 K_ii)   <- 2 from the square root
//   d(K_corr_ij)/d(hp_l) = d(K_ij)/d(hp_l) / (p_i p_j) - K_corr_ij [dln_p(i, l) + dln_p(j, l)]
//
// The first term is the pair contraction the caller already runs, leaving the second. Over the active row i*,
// whose off-diagonal |K_corr| sum is the bound minus one:
//
//   -sum_{j != i*} |K_corr_{i* j}| dln_p(j, l) - (bound - 1) dln_p(i*, l)
//
// P = I (is_identity) makes the term zero: W is the identity and d(ln p)/d(hp) vanishes. An active row of -1, set
// by a single point or by the trace bound, makes it zero too: the bound does not move, and there is no row to read.
inline arma::vec calc_eta_precon_scale_grad(
        const arma::mat& d_kern_diag_d_hp_ntrain_nhp_mat, const EtaTerms& kmat_eta_terms, const int n_hp) {
    if (kmat_eta_terms.precon_scales.is_identity() || kmat_eta_terms.idx_M_precon_max_abs_row_sum < 0)
        return arma::vec(n_hp, arma::fill::zeros);

    arma::mat d_ln_precon_scale_d_hp_ntrain_nhp_mat = d_kern_diag_d_hp_ntrain_nhp_mat;
    d_ln_precon_scale_d_hp_ntrain_nhp_mat.each_col() /= 2.0 * kmat_eta_terms.precon_scales.M_diag_vec;

    // Already zero at the active row, where the bound's sum does not run.
    const arma::vec abs_K_corr_vec = arma::abs(kmat_eta_terms.active_row_M_precon_vec);

    return -(
            d_ln_precon_scale_d_hp_ntrain_nhp_mat.t() * abs_K_corr_vec
            + (kmat_eta_terms.eig_max_ub_M_precon - 1.0)
                    * d_ln_precon_scale_d_hp_ntrain_nhp_mat.row(kmat_eta_terms.idx_M_precon_max_abs_row_sum).t());
}

// Applicability: the d(P) term needs d(diag K) for every block it differentiates, so the check below rejects a
// varying diagonal with extras whose derivative the caller did not supply. It throws rather than failing to
// compile, the kernel being a value.
//
// eta sums |K_corr| over the active row, so its derivative is that row's sign-weighted entry derivatives,
// scaled by trace_psi_w / (kmat_eta_terms.cond_max - 1). active_row_kern_common_rdot_vec is that row's c^(rdot),
// gathered by the pair term, so no kernel is evaluated here. A kernel with no extras gets no extras term; the
// outer sigma_k^2 is the caller's.
//
// Requires kmat_eta_terms built with want_active_row = true: its active row is gathered only then, and an empty one
// is a silent size mismatch under ARMA_NO_DEBUG.
inline EtaTermHpGrad calc_eta_term_hp_grad(
        const kernel::KernelBase& kern_fun,
        const kernel::HpGradInputs& hp_grad_inputs,
        const EtaTerms& kmat_eta_terms,
        const arma::mat& d_kern_diag_d_hp_gamma_ntrain_ndim_mat,
        const arma::mat& d_kern_diag_d_hp_kern_extra_ntrain_nextra_mat,
        const arma::vec& active_row_kern_common_rdot_vec,
        const double trace_psi_w,
        const SymmetrySpec& symmetry_spec,
        const arma::vec& active_row_kern_common_rdot_mirror_vec) {
    const int n_dim = static_cast<int>(hp_grad_inputs.hp_gamma_vec.n_elem);
    const int n_hp_kern_extra = kern_fun.n_hp_kern_extra(n_dim);
    const bool has_symmetry = symmetry_spec.is_on();
    // Whether P was the identity, answered by the scales assembly produced rather than by the kernel: a symmetry
    // sum varies the diagonal whatever its base kernel reports.
    const bool is_precon_identity = kmat_eta_terms.precon_scales.is_identity();
    if (!is_precon_identity && n_hp_kern_extra > 0 && d_kern_diag_d_hp_kern_extra_ntrain_nextra_mat.is_empty())
        throw ErrorNotImplemented(
                std::string("gpflash::detail::calc_eta_term_hp_grad: ") + kern_fun.kernel_name()
                + " has a varying diagonal and extra hyperparameters, and d(diag K)/d(hp_kern_extra) is "
                  "implemented only for symmetry models, so the d(P) part of its eta gradient cannot be formed.");
    const int idx_K_corr_max_abs_row_sum = kmat_eta_terms.idx_M_precon_max_abs_row_sum;
    const int n_train = static_cast<int>(active_row_kern_common_rdot_vec.n_elem);

    EtaTermHpGrad eta_term_hp_grad;
    eta_term_hp_grad.d_mll_d_hp_gamma_vec.zeros(n_dim);
    eta_term_hp_grad.d_mll_d_hp_kern_extra_vec.zeros(n_hp_kern_extra);
    // No active row, from a single point or the trace bound: eta does not move with the hyperparameters.
    if (idx_K_corr_max_abs_row_sum == -1) return eta_term_hp_grad;

    // d|K_corr|/d(alpha) = sign(K_corr) d(K_corr)/d(alpha). |x| has no derivative at an exact zero, where +1 is taken.
    arma::vec active_row_sign_vec(n_train, arma::fill::ones);
    active_row_sign_vec.elem(arma::find(kmat_eta_terms.active_row_M_precon_vec < 0.0)).fill(-1.0);

    arma::vec active_row_A_vec = active_row_sign_vec % active_row_kern_common_rdot_vec;
    // The mirrored term's own adjoint row, carrying the parity so it accumulates into the same running sum.
    arma::vec active_row_A_mirror_vec;
    if (has_symmetry)
        active_row_A_mirror_vec =
                (symmetry_spec.parity_sign() * active_row_sign_vec) % active_row_kern_common_rdot_mirror_vec;
    // d(K_corr)/d(hp) carries P^-1 on both sides, so the pair weights pick up 1 / (p_i* p_j).
    if (!is_precon_identity) {
        const arma::vec inv_precon_scale_pair_vec = kmat_eta_terms.precon_scales.inv_precon_scale_vec
                * kmat_eta_terms.precon_scales.inv_precon_scale_vec(idx_K_corr_max_abs_row_sum);
        active_row_A_vec %= inv_precon_scale_pair_vec;
        if (has_symmetry) active_row_A_mirror_vec %= inv_precon_scale_pair_vec;
    }

    arma::vec d_eta_d_hp_gamma_vec(n_dim, arma::fill::zeros);
    arma::vec d_eta_d_hp_kern_extra_vec(n_hp_kern_extra, arma::fill::zeros);
    double* const d_eta_d_hp_kern_extra_ptr = (n_hp_kern_extra > 0) ? d_eta_d_hp_kern_extra_vec.memptr() : nullptr;

    // Two ranges, split at the diagonal the bound excludes: its unset factor would otherwise reach a
    // kernel-extra derivative that may be singular at zero.
    const auto add_eta_gradient_for_idx_b_range = [&](const int idx_b_begin, const int idx_b_end) {
        if (idx_b_begin >= idx_b_end) return;
        kern_fun.add_hp_gradient_contribution(
                hp_grad_inputs,
                idx_K_corr_max_abs_row_sum,
                idx_b_begin,
                idx_b_end,
                active_row_A_vec.memptr() + idx_b_begin,
                d_eta_d_hp_gamma_vec.memptr(),
                d_eta_d_hp_kern_extra_ptr);
        if (has_symmetry)
            kern_fun.add_hp_gradient_contribution_mirrored(
                    hp_grad_inputs,
                    idx_K_corr_max_abs_row_sum,
                    idx_b_begin,
                    idx_b_end,
                    active_row_A_mirror_vec.memptr() + idx_b_begin,
                    symmetry_spec.mirror_sign_vec,
                    d_eta_d_hp_gamma_vec.memptr(),
                    d_eta_d_hp_kern_extra_ptr);
    };
    add_eta_gradient_for_idx_b_range(0, idx_K_corr_max_abs_row_sum);
    add_eta_gradient_for_idx_b_range(idx_K_corr_max_abs_row_sum + 1, n_train);
    d_eta_d_hp_gamma_vec += calc_eta_precon_scale_grad(d_kern_diag_d_hp_gamma_ntrain_ndim_mat, kmat_eta_terms, n_dim);
    if (!d_kern_diag_d_hp_kern_extra_ntrain_nextra_mat.is_empty())
        d_eta_d_hp_kern_extra_vec += calc_eta_precon_scale_grad(
                d_kern_diag_d_hp_kern_extra_ntrain_nextra_mat, kmat_eta_terms, n_hp_kern_extra);

    const double eta_grad_scale = trace_psi_w / (kmat_eta_terms.cond_max - 1.0);
    eta_term_hp_grad.d_mll_d_hp_gamma_vec = eta_grad_scale * d_eta_d_hp_gamma_vec;
    eta_term_hp_grad.d_mll_d_hp_kern_extra_vec = eta_grad_scale * d_eta_d_hp_kern_extra_vec;

    GPFLASH_CHECK(check_eta_term_hp_grad(eta_term_hp_grad, "calc_eta_term_hp_grad"));
    return eta_term_hp_grad;
}

}  // namespace detail
}  // namespace gpflash
