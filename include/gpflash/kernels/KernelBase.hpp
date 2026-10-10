/**
 * KernelBase.hpp -- the kernel interface every tier above kernels/ names.
 *
 * A kernel is an object: every tier takes `const KernelBase&`, and kernel_by_id in KernelDispatch.hpp turns a
 * KernelId into one. A concrete kernel derives a family (KernelRadial, ...), which supplies every method but
 * the kernel's own.
 *
 * Every method takes a whole column, run or matrix of points, so one virtual call covers it and the loop over
 * points stays inside the kernel, beside the per-element math it calls and open to vectorization. Some
 * quantities also have a single-point form, which the one-query posterior path reaches once per call.
 *
 * Past the methods that build the kernel's own space from raw X, points arrive in that space, so the declarations
 * spell them Ydot and n_feature. A family that runs no feature map spells the same arguments Xdot and n_dim, where
 * the two coincide.
 *
 * Instances hold no mutable state: one is shared across every OpenMP thread. Hyperparameters stay in HpVal and
 * arrive as arguments.
 */
#pragma once

#include <armadillo>  // arma::mat, arma::vec, arma::rowvec
#include <cstddef>    // std::ptrdiff_t
#include <string>     // std::string, for the not-implemented message

#include "gpflash/core/Errors.hpp"           // ErrorNotImplemented, for the defaults that throw
#include "gpflash/inputs/KernelId.hpp"       // KernelId
#include "gpflash/inputs/SymmetrySpec.hpp"   // SymmetrySpec, which calc_x_center_vec reads
#include "gpflash/kernels/HpGradInputs.hpp"  // HpGradInputs
#include "gpflash/kernels/OptzSeeding.hpp"   // OptzSeeding

namespace gpflash {
namespace kernel {

class KernelBase {
   public:
    virtual ~KernelBase() = default;

    // ── Identity ──────────────────────────────────────────────────────────────
    // The one name accepted for this kernel, and its cache-file integer.
    virtual const char* kernel_name() const = 0;
    virtual KernelId kernel_id() const = 0;

    // ── What the kernel declares ──────────────────────────────────────────────
    // Extra hyperparameters beyond hp_gamma's n_dim entries, in the order the kernel unpacks them. The count is
    // constant or grows with the dimension, so n_hp_kern_extra(1) == 0 means "none at all".
    virtual int n_hp_kern_extra(int /*n_dim_input*/) const { return 0; }

    // Whether k(x, x) == 1 for every point, which makes K a correlation matrix and lets assembly's Gershgorin
    // bound hold. A varying diagonal returns false and overrides both diagonal methods below.
    virtual bool is_kern_diag_unity() const = 0;
    // Whether k(x, y) depends on x - y alone, so the points can be measured from any origin.
    virtual bool is_stationary() const { return false; }
    // Whether a symmetry model may wrap this kernel: it is stationary, and it defines the mirror in its feature space
    // and the mirrored gradients below.
    virtual bool is_symmetrizable() const { return false; }

    // The input dimension behind a feature count. The identity unless the kernel maps its points.
    virtual int calc_n_dim_input(const int n_feature) const { return n_feature; }

    // Where a fit of this kernel should start and sample; an unset OptzSeeding asks for the shared heuristic.
    virtual OptzSeeding calc_optz_seeding(const arma::mat& /*X_ndim_ntrain_mat*/) const { return OptzSeeding(); }

    // ── The points the kernel is evaluated on ─────────────────────────────────
    // Every point is measured from x_center_vec before the map: a stationary kernel's squared distances lose digits
    // to cancellation where the data sits far from 0 (KernelStationary.hpp), and its model is the same from any
    // origin. Any other kernel's model depends on the origin, so its center is 0.
    //
    // The training points' per-dimension median, so one extreme point does not move the center. A dimension the
    // mirror flips keeps 0, on the mirror plane; the mirror is checked against the data before this.
    arma::vec calc_x_center_vec(const arma::mat& X_ndim_ntrain_mat, const SymmetrySpec& symmetry_spec) const {
        if (!is_stationary()) return arma::vec(X_ndim_ntrain_mat.n_rows, arma::fill::zeros);

        arma::vec x_center_vec = arma::median(X_ndim_ntrain_mat, 1);
        if (symmetry_spec.is_on()) x_center_vec.elem(arma::find(symmetry_spec.mirror_sign_vec < 0.0)).zeros();
        return x_center_vec;
    }

    // Y: the centered points in the kernel's own space, unscaled, or empty where the kernel runs no map -- which is
    // what callers test. These defaults run no map, so Ydot = Xdot = hp_gamma .* (X - x_center).
    virtual arma::mat calc_feature_points(
            const arma::mat& /*X_ndim_npoint_mat*/,
            const arma::vec& /*x_center_vec*/,
            const arma::vec& /*hp_gamma_vec*/,
            const arma::vec& /*hp_kern_extra_vec*/) const {
        return arma::mat{};
    }

    // Ydot: those points scaled by their own space's length scales, which every kernel value is computed from.
    virtual arma::mat calc_scaled_feature_points(
            const arma::mat& X_ndim_npoint_mat,
            const arma::vec& x_center_vec,
            const arma::vec& hp_gamma_vec,
            const arma::vec& /*hp_kern_extra_vec*/) const {
        // Centered into the one result matrix, then scaled in place, so no second matrix is formed.
        arma::mat Ydot_ndim_npoint_mat = X_ndim_npoint_mat.each_col() - x_center_vec;
        Ydot_ndim_npoint_mat.each_col() %= hp_gamma_vec;
        return Ydot_ndim_npoint_mat;
    }

    // The mirror S of a symmetry model as it acts on Ydot, one sign per feature: Ydot(S x) = feature signs .* Ydot(x).
    // mirror_sign_vec is S in input dimensions. A kernel that does not define it cannot be symmetrized.
    virtual arma::vec calc_feature_mirror_sign_vec(const arma::vec& /*mirror_sign_vec*/) const {
        throw ErrorNotImplemented(
                std::string("gpflash::kernel::") + kernel_name()
                + " does not define calc_feature_mirror_sign_vec, so it cannot be the base kernel of a symmetry "
                  "model.");
    }

    // Single-point counterpart, for the one-query posterior path.
    virtual arma::vec calc_scaled_feature_point_vec(
            const arma::vec& x_ndim_vec,
            const arma::vec& x_center_vec,
            const arma::vec& hp_gamma_vec,
            const arma::vec& /*hp_kern_extra_vec*/) const {
        return (x_ndim_vec - x_center_vec) % hp_gamma_vec;
    }

    // ── From points to the scalar the kernel reads ────────────────────────────
    // The family decides what that scalar is: the scaled squared distance for a stationary kernel, the Gram
    // cross term for a non-stationary one.

    // Training against query points, an (n_train, n_post) buffer in one call. Sized here, so a caller passes an
    // empty matrix.
    virtual void fill_kern_input_post_mat(
            const arma::mat& Ydot_nfeature_ntrain_mat,
            const arma::vec& ydot_train_sq_norm_vec,
            const arma::mat& Ydot_nfeature_npost_mat,
            const arma::vec& ydot_post_sq_norm_vec,
            arma::mat& kern_input_ntrain_npost_mat_out) const = 0;

    // A contiguous run of pairs sharing one of their two points: the training points against one query.
    // sq_norm_shared is that one point's norm, the same for every entry.
    //
    // kern_input_ptr memory: IN: ydot_inner_prod, OUT: the kernel's input scalar
    virtual void calc_kern_input_inplace(
            double* kern_input_ptr, int n_elem, const double* sq_norm_ptr, double sq_norm_shared) const = 0;

    // ── From that scalar to the kernel value ──────────────────────────────────
    // Two methods, one per caller: an evaluation without a gradient wants the values alone and writes one array,
    // the gradient path wants c^(rdot) with them and writes two. Both count in std::ptrdiff_t, not int: the batch
    // posterior hands them a whole (n_train, n_post) block at once, and that product leaves int within reach.

    // kern_input_ptr memory: IN: the kernel's input scalar, OUT: kernel values
    virtual void calc_kern_val_inplace(
            double* kern_input_ptr, std::ptrdiff_t n_elem, const double* hp_kern_extra_ptr) const = 0;

    // The same, also writing c^(rdot), the per-pair factor every derivative of k carries. It falls out of the
    // value, so a gradient costs no second kernel evaluation.
    //
    // kern_input_ptr memory: IN: the kernel's input scalar, OUT: kernel values
    virtual void calc_kern_val_and_common_rdot_inplace(
            double* kern_input_ptr,
            std::ptrdiff_t n_elem,
            const double* hp_kern_extra_ptr,
            double* kern_common_rdot_ptr_out) const = 0;

    // One column of K's strict lower triangle: n_below kernel inputs, overwritten with n_below kernel values.
    // kern_common_rdot_col_ptr_out receives c^(rdot) for the same pairs; null asks for the values alone.
    //
    // ydot_inner_prod_col_ptr memory: IN: ydot_inner_prod, OUT: kernel values
    virtual void calc_kmat_column_inplace(
            double* ydot_inner_prod_col_ptr,
            int n_below,
            double sq_norm_col,
            const double* sq_norm_below_ptr,
            const double* hp_kern_extra_ptr,
            double* kern_common_rdot_col_ptr_out) const = 0;

    // ── The diagonal ──────────────────────────────────────────────────────────
    // k(x, x): K's diagonal, and the posterior's prior variance, at one point or at many. The two must agree, which
    // test_gpflash_kernel_contract.cpp checks.
    virtual double calc_kern_diag_val(double ydot_sq_norm, const double* hp_kern_extra_ptr) const = 0;
    virtual arma::vec calc_kern_diag_val_vec(
            const arma::vec& ydot_sq_norm_vec, const double* hp_kern_extra_ptr) const = 0;

    // ── The hyperparameter gradient ───────────────────────────────────────────
    // Accumulates sum_row_a into the running sums, for point idx_a against the contiguous range
    // [idx_b_begin, idx_b_end). A_ptr holds the caller's adjoint with c^(rdot) already folded in, entry zero at
    // idx_b_begin. d_mll_d_hp_kern_extra_ptr is null to skip the extras.
    virtual void add_hp_gradient_contribution(
            const HpGradInputs& hp_grad_inputs,
            int idx_a,
            int idx_b_begin,
            int idx_b_end,
            const double* A_ptr,
            double* d_mll_d_hp_gamma_ptr,
            double* d_mll_d_hp_kern_extra_ptr) const = 0;

    // The same sums with the second point mirrored, which a symmetry model's mirrored term needs: every h^(alpha,l)
    // is taken at the pair (x_a, S x_b). With idx_b_begin = idx_a it is the pair (x_a, S x_a), whose value is the
    // mirrored diagonal. A kernel that does not define it cannot be symmetrized.
    virtual void add_hp_gradient_contribution_mirrored(
            const HpGradInputs& /*hp_grad_inputs*/,
            int /*idx_a*/,
            int /*idx_b_begin*/,
            int /*idx_b_end*/,
            const double* /*A_ptr*/,
            const arma::vec& /*mirror_sign_vec*/,
            double* /*d_mll_d_hp_gamma_ptr*/,
            double* /*d_mll_d_hp_kern_extra_ptr*/) const {
        throw ErrorNotImplemented(
                std::string("gpflash::kernel::") + kernel_name()
                + " does not define add_hp_gradient_contribution_mirrored, so it cannot be the base kernel "
                  "of a symmetry model.");
    }

    // The same for the kernel's own extras, whose derivative is not the family's chain rule. A kernel declaring
    // extras and not overriding this throws on its first gradient evaluation.
    virtual void add_hp_kern_extra_gradient_contribution(
            const HpGradInputs& /*hp_grad_inputs*/,
            int /*idx_a*/,
            int /*idx_b_begin*/,
            int /*idx_b_end*/,
            const double* /*A_ptr*/,
            double* /*d_mll_d_hp_kern_extra_ptr*/) const {
        throw ErrorNotImplemented(
                std::string("gpflash::kernel::") + kernel_name()
                + " declares extra hyperparameters but does not define add_hp_kern_extra_gradient_contribution: "
                  "their derivative is not the family's chain rule, so the kernel must supply it.");
    }

    // Its mirrored counterpart, at the pair (x_a, S x_b), for a kernel whose family defines the mirrored hp_gamma sum.
    virtual void add_hp_kern_extra_gradient_contribution_mirrored(
            const HpGradInputs& /*hp_grad_inputs*/,
            int /*idx_a*/,
            int /*idx_b_begin*/,
            int /*idx_b_end*/,
            const double* /*A_ptr*/,
            const arma::vec& /*mirror_sign_vec*/,
            double* /*d_mll_d_hp_kern_extra_ptr*/) const {
        throw ErrorNotImplemented(
                std::string("gpflash::kernel::") + kernel_name()
                + " declares extra hyperparameters but does not define "
                  "add_hp_kern_extra_gradient_contribution_mirrored, so it cannot be the base kernel of a symmetry "
                  "model.");
    }

    // d(K_ii)/d(hp_gamma_l), (n_train, n_dim), empty where k(x, x) is constant -- which callers read as zero. A
    // kernel whose diagonal varies must override it, so the throw is what keeps that zero from being assumed.
    virtual arma::mat calc_d_kern_diag_d_hp_gamma_mat(const HpGradInputs& /*hp_grad_inputs*/) const {
        if (!is_kern_diag_unity()) throw_varying_diag_not_implemented("calc_d_kern_diag_d_hp_gamma_mat");

        return arma::mat{};
    }

    // ── The posterior input-space gradient ────────────────────────────────────
    // q^T H^(r), the factor the posterior mean and variance gradients share. The caller passes its own q, so one
    // method serves both; the batch form also applies the model_scale it is given.
    virtual arma::vec calc_d_kern_sum_d_x_post(
            const arma::mat& Ydot_ntrain_nfeature_mat,
            const arma::vec& ydot_post_vec,
            const arma::vec& hp_gamma_vec,
            const arma::vec& q_vec) const = 0;

    virtual arma::mat calc_d_kern_sum_d_x_post_batch(
            const arma::mat& Ydot_nfeature_npost_mat,
            const arma::mat& q_ydot_sum_nfeature_npost_mat,
            const arma::rowvec& q_sum_row,
            const arma::vec& hp_gamma_vec,
            double model_scale) const = 0;

    // ── The prior variance term of the posterior variance gradient ────────────
    //     variance(x')       = sigma_k^2 ( k(x', x') - sigma_k^2 ||L_Sigma^-1 k_*||^2 )
    //     d(variance)/d(x')  = sigma_k^2 d(k(x', x'))/d(x') - 2 sigma_k^4 (Sigma^-1 k_*)^T J_k
    //
    // These two return d(k(x', x'))/d(x'), the first term without its sigma_k^2, empty where the diagonal is
    // constant so the caller skips it.
    virtual arma::vec calc_d_kern_diag_val_d_x_post(
            const arma::vec& /*ydot_post_vec*/,
            const arma::vec& /*hp_gamma_vec*/,
            const double* /*hp_kern_extra_ptr*/) const {
        if (!is_kern_diag_unity()) throw_varying_diag_not_implemented("calc_d_kern_diag_val_d_x_post");

        return arma::vec{};
    }

    virtual arma::mat calc_d_kern_diag_val_d_x_post_batch(
            const arma::mat& /*Ydot_nfeature_npost_mat*/,
            const arma::vec& /*hp_gamma_vec*/,
            const double* /*hp_kern_extra_ptr*/) const {
        if (!is_kern_diag_unity()) throw_varying_diag_not_implemented("calc_d_kern_diag_val_d_x_post_batch");

        return arma::mat{};
    }

   protected:
    // Only a family derives this; a kernel deriving it directly reimplements a family by accident.
    KernelBase() = default;

    // Consumers read the three empty defaults above as zero, which holds only for a constant diagonal. A kernel
    // that inherits one while declaring a varying diagonal gets this instead of a wrong gradient.
    void throw_varying_diag_not_implemented(const char* method_name) const {
        throw ErrorNotImplemented(
                std::string("gpflash::kernel::") + kernel_name() + " declares a varying k(x, x) but does not define "
                + method_name + ": the empty default it would inherit means zero, which is true only of a constant "
                                "diagonal.");
    }
};

}  // namespace kernel
}  // namespace gpflash
