/**
 * EvalMllGrad.hpp -- d(MLL)/d(alpha), assembled from the matrix-gradient terms. Alpha is any one
 * hyperparameter the kernel exposes -- a length scale gamma_l, or one of its extras -- since every term below
 * takes the same form whichever it is. It is the same alpha as KernelMatrixGrad.hpp's h^(alpha,l).
 *
 *   d(MLL)/d(alpha)   = <Psi, d(Sigma)/d(alpha)>
 *   d(Sigma)/d(alpha) = sigma_k^2 [d(K)/d(alpha) + d(eta)/d(alpha) W + eta d(W)/d(alpha)]
 *
 * KernelMatrixGrad.hpp supplies d(K)/d(alpha) and d(W)/d(alpha), KmatNuggetGrad.hpp supplies
 * d(eta)/d(alpha), and SigmaGrad.hpp supplies Psi and the two variance derivatives. This file combines them.
 */
#pragma once

#include <armadillo>

#include "gpflash/assembly/EtaTermHpGrad.hpp"     // EtaTermHpGrad
#include "gpflash/assembly/KernTermHpGrad.hpp"    // KernTermHpGrad
#include "gpflash/assembly/KernelMatrixGrad.hpp"  // calc_kern_term_hp_grad, calc_kern_diag_hp_grad
#include "gpflash/assembly/KmatNuggetGrad.hpp"    // calc_eta_term_hp_grad
#include "gpflash/assembly/SigmaGrad.hpp"         // Psi and variance gradients
#include "gpflash/assembly/SymmetryMatrix.hpp"    // calc_symmetry_kern_diag_hp_grad_mat
#include "gpflash/inputs/SymmetrySpec.hpp"        // SymmetrySpec
#include "gpflash/kernels/HpGradInputs.hpp"       // kernel::HpGradInputs
#include "gpflash/kernels/KernelBase.hpp"         // kernel::KernelBase
#include "gpflash/preconditioning/EtaTerms.hpp"   // EtaTerms
#include "gpflash/training/MllHpGrad.hpp"         // MllHpGrad
#include "gpflash/training/StageTiming.hpp"       // GPFLASH_STAGE_MARK

namespace gpflash {
namespace detail {

// Sigma_inv_mat_lower is read only: Psi's Sigma^-1 (SigmaGrad.hpp) is in its lower triangle and diagonal, c^(rdot) in
// its strict upper.
inline void fill_mll_hp_kern_grad(
        const kernel::KernelBase& kern_fun,
        const kernel::HpGradInputs& hp_grad_inputs,
        const arma::vec& lin_sol_sigma_residual_vec,
        const arma::mat& Sigma_inv_mat_lower,
        const EtaTerms& kmat_eta_terms,
        const double trace_psi,
        const double hp_sig2_k,
        const SymmetrySpec& symmetry_spec,
        const arma::vec& kern_common_rdot_mirror_packed_vec,
        MllHpGrad& mll_hp_grad_out) {
    const int n_dim = static_cast<int>(hp_grad_inputs.hp_gamma_vec.n_elem);

    // Term 1: d(K)/d(alpha), plus each term's active row of c^(rdot) for d(eta)/d(alpha).
    const KernTermHpGrad kern_term_hp_grad = calc_kern_term_hp_grad(
            kern_fun,
            hp_grad_inputs,
            lin_sol_sigma_residual_vec,
            Sigma_inv_mat_lower,
            kmat_eta_terms.idx_M_precon_max_abs_row_sum,
            symmetry_spec,
            kern_common_rdot_mirror_packed_vec);
    mll_hp_grad_out.d_mll_d_hp_gamma_vec = kern_term_hp_grad.d_mll_d_hp_gamma_vec;
    mll_hp_grad_out.d_mll_d_hp_kern_extra_vec = kern_term_hp_grad.d_mll_d_hp_kern_extra_vec;

    // Terms 2, 3 and 4 all read W = diag(K), so one branch covers all three. Where W = I they collapse:
    // tr(Psi W) is tr(Psi), and these two stay empty, which is how Terms 3 and 4 below come out zero.
    arma::mat d_kern_diag_d_hp_gamma_ntrain_ndim_mat;
    arma::mat d_kern_diag_d_hp_kern_extra_ntrain_nextra_mat;
    arma::vec psi_diag_vec;
    double trace_psi_w;  // tr(Psi W), the GpFlash paper's zeta
    // Whether the matrix needed preconditioning, which a symmetry sum forces whatever its base kernel's own
    // diagonal is. Read off the scales assembly produced rather than asked of the kernel.
    if (!kmat_eta_terms.precon_scales.is_identity()) {
        // A symmetry model's varying diagonal is the mirrored term's, which the kernel cannot report: it is
        // built from the mirrored factor assembly packed away.
        if (symmetry_spec.is_on()) {
            const arma::mat d_kern_diag_d_hp_ntrain_nhp_mat = calc_symmetry_kern_diag_hp_grad_mat(
                    kern_fun, hp_grad_inputs, symmetry_spec, kern_common_rdot_mirror_packed_vec);
            d_kern_diag_d_hp_gamma_ntrain_ndim_mat = d_kern_diag_d_hp_ntrain_nhp_mat.head_cols(n_dim);
            d_kern_diag_d_hp_kern_extra_ntrain_nextra_mat =
                    d_kern_diag_d_hp_ntrain_nhp_mat.tail_cols(d_kern_diag_d_hp_ntrain_nhp_mat.n_cols - n_dim);
        } else {
            d_kern_diag_d_hp_gamma_ntrain_ndim_mat = kern_fun.calc_d_kern_diag_d_hp_gamma_mat(hp_grad_inputs);
        }

        psi_diag_vec = calc_psi_diag_vec(lin_sol_sigma_residual_vec, Sigma_inv_mat_lower);
        trace_psi_w = arma::dot(psi_diag_vec, kmat_eta_terms.precon_scales.M_diag_vec);
    } else {
        trace_psi_w = trace_psi;
    }

    // Term 2: d(eta)/d(alpha)
    const EtaTermHpGrad eta_term_hp_grad = calc_eta_term_hp_grad(
            kern_fun,
            hp_grad_inputs,
            kmat_eta_terms,
            d_kern_diag_d_hp_gamma_ntrain_ndim_mat,
            d_kern_diag_d_hp_kern_extra_ntrain_nextra_mat,
            kern_term_hp_grad.active_row_kern_common_rdot_vec,
            trace_psi_w,
            symmetry_spec,
            kern_term_hp_grad.active_row_kern_common_rdot_mirror_vec);
    mll_hp_grad_out.d_mll_d_hp_gamma_vec += eta_term_hp_grad.d_mll_d_hp_gamma_vec;
    mll_hp_grad_out.d_mll_d_hp_kern_extra_vec += eta_term_hp_grad.d_mll_d_hp_kern_extra_vec;

    // Terms 3 and 4: d(diag K)/d(alpha) + eta d(W)/d(alpha), one (1 + eta) diagonal pass. Unbranched because
    // calc_kern_diag_hp_grad reads the empty matrix above as zeros, so W = I already costs nothing here.
    const arma::vec kern_diag_hp_grad_vec =
            calc_kern_diag_hp_grad(d_kern_diag_d_hp_gamma_ntrain_ndim_mat, psi_diag_vec, n_dim);
    mll_hp_grad_out.d_mll_d_hp_gamma_vec += (1.0 + kmat_eta_terms.eta) * kern_diag_hp_grad_vec;
    if (!d_kern_diag_d_hp_kern_extra_ntrain_nextra_mat.is_empty())
        mll_hp_grad_out.d_mll_d_hp_kern_extra_vec +=
                (1.0 + kmat_eta_terms.eta) * (d_kern_diag_d_hp_kern_extra_ntrain_nextra_mat.t() * psi_diag_vec);

    mll_hp_grad_out.d_mll_d_hp_gamma_vec *= hp_sig2_k;
    mll_hp_grad_out.d_mll_d_hp_kern_extra_vec *= hp_sig2_k;
}

// Calculate the complete gradient: the two variance derivatives, then every kernel term. The caller has
// already converted the residual solve and Sigma^-1 to the real Sigma's coordinates.
inline MllHpGrad calc_mll_hp_grad(
        const kernel::KernelBase& kern_fun,
        const kernel::HpGradInputs& hp_grad_inputs,
        const EtaTerms& kmat_eta_terms,
        const arma::vec& lin_sol_sigma_residual_vec,
        const arma::mat& Sigma_inv_mat_lower,
        const double trace_psi_sigma,
        const double sig2_f_val,
        const double hp_sig2_k,
        const SymmetrySpec& symmetry_spec,
        const arma::vec& kern_common_rdot_mirror_packed_vec) {
    MllHpGrad mll_hp_grad;
    const double trace_psi = calc_trace_psi(lin_sol_sigma_residual_vec, Sigma_inv_mat_lower);
    GPFLASH_STAGE_MARK(sigma_inv_s);
    mll_hp_grad.d_mll_d_hp_sig2_f_val = calc_d_mll_d_sig2_f(trace_psi);
    mll_hp_grad.d_mll_d_hp_sig2_k = calc_d_mll_d_sig2_k(trace_psi_sigma, trace_psi, sig2_f_val, hp_sig2_k);

    fill_mll_hp_kern_grad(
            kern_fun,
            hp_grad_inputs,
            lin_sol_sigma_residual_vec,
            Sigma_inv_mat_lower,
            kmat_eta_terms,
            trace_psi,
            hp_sig2_k,
            symmetry_spec,
            kern_common_rdot_mirror_packed_vec,
            mll_hp_grad);
    GPFLASH_STAGE_MARK(grad_s);

    return mll_hp_grad;
}

}  // namespace detail
}  // namespace gpflash
