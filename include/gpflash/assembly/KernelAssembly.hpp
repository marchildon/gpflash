/**
 * KernelAssembly.hpp -- K_corr and the conditioning data measured on it, from one call.
 *
 * build_eta_terms preconditions K -> K_corr in place, so the matrix and the terms measured on it only exist
 * together: hence one struct rather than two calls. X_train is an input, not a product, so it stays with the
 * caller. The matrix comes back in Buffer_mat, named for the reuse every later stage makes of that memory.
 *
 * K_corr is preconditioning/ConditionedCholesky.hpp's M_precon with M = K, and a _corr suffix marks those
 * coordinates, so Sigma_corr is Sigma's counterpart. eta bounds kappa(K_corr + eta I), which kappa(Sigma) can exceed
 * by up to a factor of kappa(P)^2.
 */
#pragma once

#include <armadillo>

#include "gpflash/assembly/KernelMatrix.hpp"                // build_gram_lower_tri, calc_gram_to_kmat_inplace
#include "gpflash/assembly/SymmetryMatrix.hpp"              // build_symmetry_kmat_lower_tri, calc_packed_lower_n_elem
#include "gpflash/core/Errors.hpp"                          // the field checks below compose
#include "gpflash/core/Platform.hpp"                        // GPFLASH_CHECK
#include "gpflash/inputs/EigMaxUbMethod.hpp"                // EigMaxUbMethod
#include "gpflash/inputs/HpVal.hpp"                         // calc_hp_kern_extra_ptr
#include "gpflash/inputs/SymmetrySpec.hpp"                  // SymmetrySpec
#include "gpflash/kernels/KernelBase.hpp"                   // the kernel contract
#include "gpflash/preconditioning/ConditionedCholesky.hpp"  // build_eta_terms
#include "gpflash/preconditioning/EtaTerms.hpp"             // EtaTerms

namespace gpflash {
namespace detail {

// Everything assembly produces, returned by value rather than through out-params.
struct KernelAssembly {
    // The one n_train by n_train buffer: K_corr on return, then overwritten in place by the later stages, which
    // differ by path. No _lower suffix: both triangles are live. Those stages write below the diagonal, while the
    // strict upper holds the kernel's c^(rdot) when a gradient follows, so lower-triangle work must leave it intact.
    arma::mat Buffer_mat;
    EtaTerms kmat_eta_terms;
    // The unscaled feature points, which only the gradient stage reads, to differentiate the map. Empty where
    // the kernel runs no map, and where eval_grad was false.
    arma::mat Y_nfeature_ntrain_mat;

    // The scaled points K was built from and their per-point squared norms; the posterior stores both.
    arma::mat Ydot_nfeature_ntrain_mat;
    arma::vec ydot_train_sq_norm_vec;

    // The mirrored term's c^(rdot), packed over the lower triangle with the diagonal, the base term's having
    // taken the buffer's strict upper. Empty without symmetry, or without a gradient to read it.
    arma::vec kern_common_rdot_mirror_packed_vec;
};

// Postcondition. Buffer_mat's contents are not scanned: without a gradient its strict upper is never written.
inline void check_kernel_assembly(const KernelAssembly& kernel_assembly, const char* caller) {
    check_mat_not_empty(kernel_assembly.Buffer_mat, "KernelAssembly::Buffer_mat", caller);
    check_mat_is_square(kernel_assembly.Buffer_mat, "KernelAssembly::Buffer_mat", caller);

    const arma::uword n_train = kernel_assembly.Buffer_mat.n_rows;

    // One column per training point, or the gradient's map derivative reads past the points it was built from.
    // Y is empty where the kernel runs no map, or where no gradient follows.
    if (!kernel_assembly.Y_nfeature_ntrain_mat.is_empty())
        check_mat_n_cols(
                kernel_assembly.Y_nfeature_ntrain_mat, n_train, "KernelAssembly::Y_nfeature_ntrain_mat", caller);
    check_mat_not_empty(kernel_assembly.Ydot_nfeature_ntrain_mat, "KernelAssembly::Ydot_nfeature_ntrain_mat", caller);
    check_mat_n_cols(
            kernel_assembly.Ydot_nfeature_ntrain_mat, n_train, "KernelAssembly::Ydot_nfeature_ntrain_mat", caller);

    // One norm per training point, which is Buffer_mat's square side.
    check_vec_len(kernel_assembly.ydot_train_sq_norm_vec, n_train, "KernelAssembly::ydot_train_sq_norm_vec", caller);

    // Empty without symmetry, or without a gradient to read it; present, it spans the lower triangle exactly.
    if (!kernel_assembly.kern_common_rdot_mirror_packed_vec.is_empty())
        check_vec_len(
                kernel_assembly.kern_common_rdot_mirror_packed_vec,
                static_cast<arma::uword>(calc_packed_lower_n_elem(static_cast<int>(n_train))),
                "KernelAssembly::kern_common_rdot_mirror_packed_vec",
                caller);
    check_all_finite_vals(
            kernel_assembly.kern_common_rdot_mirror_packed_vec,
            "KernelAssembly::kern_common_rdot_mirror_packed_vec",
            caller);

    check_eta_terms(kernel_assembly.kmat_eta_terms, caller);
}

// Build K_corr and measure the conditioning data on it. eval_grad keeps the extras only the gradient reads:
// the unscaled feature points, the kernel's c^(rdot) factors, and the eta data's active row.
inline KernelAssembly build_kern_assembly(
        const kernel::KernelBase& kern_fun,
        const arma::mat& X_ndim_ntrain_mat,
        const arma::vec& x_center_vec,
        const arma::vec& hp_gamma_vec,
        const arma::vec& hp_kern_extra_vec,
        const SymmetrySpec& symmetry_spec,
        const double cond_max,
        const EigMaxUbMethod kmat_eig_max_ub_method,
        const bool eval_grad) {
    KernelAssembly kern_assembly;

    // Same center and hyperparameters to both calls, so Ydot is the scaled form of this Y.
    if (eval_grad)
        kern_assembly.Y_nfeature_ntrain_mat =
                kern_fun.calc_feature_points(X_ndim_ntrain_mat, x_center_vec, hp_gamma_vec, hp_kern_extra_vec);
    kern_assembly.Ydot_nfeature_ntrain_mat =
            kern_fun.calc_scaled_feature_points(X_ndim_ntrain_mat, x_center_vec, hp_gamma_vec, hp_kern_extra_vec);

    const double* hp_kern_extra_ptr = calc_hp_kern_extra_ptr(hp_kern_extra_vec);
    if (symmetry_spec.is_on()) {
        kern_assembly.Buffer_mat = build_symmetry_kmat_lower_tri(
                kern_fun,
                kern_assembly.Ydot_nfeature_ntrain_mat,
                symmetry_spec,
                hp_kern_extra_ptr,
                kern_assembly.ydot_train_sq_norm_vec,
                eval_grad ? &kern_assembly.kern_common_rdot_mirror_packed_vec : nullptr);
    } else {
        // The norms are copied off the Gram diagonal before K's own diagonal replaces it, so a later query's k_*
        // reads the norms K was built from.
        kern_assembly.Buffer_mat = build_gram_lower_tri(kern_assembly.Ydot_nfeature_ntrain_mat);
        kern_assembly.ydot_train_sq_norm_vec = kern_assembly.Buffer_mat.diag();
        calc_gram_to_kmat_inplace(
                kern_fun,
                kern_assembly.Buffer_mat,
                kern_assembly.ydot_train_sq_norm_vec,
                hp_kern_extra_ptr,
                /*store_hp_grad_factors=*/eval_grad);
    }

    // Preconditions the buffer K -> K_corr in place, which is the state it is returned in. Summing the two
    // symmetry terms varies the diagonal whatever the base kernel's own diagonal is, so the matrix is asked, not
    // the kernel. A true answer skips the scaling; test_gpflash_kernel_contract and build_eta_terms check it against
    // the diagonal.
    const bool is_kern_diag_unity = kern_fun.is_kern_diag_unity() && !symmetry_spec.is_on();
    kern_assembly.kmat_eta_terms = build_eta_terms(
            is_kern_diag_unity,
            kern_assembly.Buffer_mat,
            cond_max,
            kmat_eig_max_ub_method,
            /*want_active_row=*/eval_grad);

    GPFLASH_CHECK(check_kernel_assembly(kern_assembly, "build_kern_assembly"));
    return kern_assembly;
}

}  // namespace detail
}  // namespace gpflash
