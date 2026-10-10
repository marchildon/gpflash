/**
 * KernelInnerProduct.hpp -- what every inner-product kernel shares, as the KernelBase methods it does not write.
 *
 * An inner-product kernel is a function of the scaled inner product alone,
 *
 *   xdot_inner_prod = dot(xdot_i_vec, xdot_j_vec) = sum_d hp_gamma_d^2 x_id x_jd,   k_ij = k(xdot_inner_prod),
 *
 * and one DSYRK produces every xdot_inner_prod at once, so assembly hands the kernel that scalar rather than the
 * two points. The kernel's input scalar is that inner product, so the input step is the identity.
 *
 * This family runs no feature map: Y is X, n_feature is n_dim, Ydot is hp_gamma .* X. It inherits KernelBase's
 * feature-point defaults, which is why only X appears below.
 *
 * A derived kernel writes its value loops, its varying diagonal, and its identity:
 *
 *     is_kern_diag_unity                                             returns false: k(x, x) varies with the point
 *     calc_kern_val_inplace, calc_kern_val_and_common_rdot_inplace   k and c^(rdot) over a run
 *     calc_kern_diag_val, calc_kern_diag_val_vec                     k(x, x) at each point
 *     kernel_name, kernel_id
 *
 * The per-pair gradient factor c^(rdot) is dk/d(xdot_inner_prod), which the family reads back from the value-and-rdot
 * loop wherever it needs that derivative, so the derivative has one source.
 *
 * These are a subset of the non-stationary kernels: one that also needs the two self-norms, not their inner
 * product alone, needs a family of its own.
 */
#pragma once

#include <armadillo>  // arma::mat, arma::vec, arma::rowvec
#include <cstddef>    // std::ptrdiff_t

#include "gpflash/core/Platform.hpp"                 // GPFLASH_RESTRICT, GPFLASH_OMP_SIMD_REDUCTION
#include "gpflash/inputs/HpVal.hpp"                  // calc_hp_kern_extra_ptr
#include "gpflash/kernels/HpGradInputs.hpp"          // HpGradInputs
#include "gpflash/kernels/KernelBase.hpp"            // KernelBase
#include "gpflash/linear_algebra/InnerProducts.hpp"  // fill_inner_prod_mat

namespace gpflash {
namespace kernel {

class KernelInnerProduct : public KernelBase {
   public:
    // ── The scalar every inner-product kernel reads ───────────────────────────
    // The Gram cross term dot(ydot_train_i, ydot_post_j) for every train/post pair; the norm arguments a
    // stationary kernel needs are unused.
    void fill_kern_input_post_mat(
            const arma::mat& Ydot_nfeature_ntrain_mat,
            const arma::vec& /*ydot_train_sq_norm_vec*/,
            const arma::mat& Ydot_nfeature_npost_mat,
            const arma::vec& /*ydot_post_sq_norm_vec*/,
            arma::mat& kern_input_ntrain_npost_mat_out) const final {
        detail::fill_inner_prod_mat(
                Ydot_nfeature_ntrain_mat,
                Ydot_nfeature_npost_mat,
                /*scale_factor=*/1.0,
                kern_input_ntrain_npost_mat_out);
    }

    // The single-query run form, a no-op: the input is already the inner product.
    void calc_kern_input_inplace(
            double* /*kern_input_ptr*/,
            const int /*n_elem*/,
            const double* /*sq_norm_ptr*/,
            const double /*sq_norm_shared*/) const final {}

    // One column of K: the inner products this kernel reads directly, then its value loop. The norm arguments a
    // stationary kernel needs are unused.
    void calc_kmat_column_inplace(
            double* ydot_inner_prod_col_ptr,
            const int n_below,
            const double /*sq_norm_col*/,
            const double* /*sq_norm_below_ptr*/,
            const double* hp_kern_extra_ptr,
            double* kern_common_rdot_col_ptr_out) const final {
        if (kern_common_rdot_col_ptr_out)
            calc_kern_val_and_common_rdot_inplace(
                    ydot_inner_prod_col_ptr, n_below, hp_kern_extra_ptr, kern_common_rdot_col_ptr_out);
        else
            calc_kern_val_inplace(ydot_inner_prod_col_ptr, n_below, hp_kern_extra_ptr);
    }

    // ── The hyperparameter gradient ───────────────────────────────────────────
    // Accumulates sum_b A_ab h^(gamma,l)_ab into d_mll_d_hp_gamma_ptr over [idx_b_begin, idx_b_end), with A_ptr
    // already carrying c^(rdot) and the inner-product h term
    //
    //     h^(gamma,l)_ab = 2 hp_gamma_l x_al x_bl
    //
    // so each dimension is one inner product over the range times a per-a constant.
    void add_hp_gradient_contribution(
            const HpGradInputs& hp_grad_inputs,
            const int idx_a,
            const int idx_b_begin,
            const int idx_b_end,
            const double* GPFLASH_RESTRICT A_ptr,
            double* GPFLASH_RESTRICT d_mll_d_hp_gamma_ptr,
            double* GPFLASH_RESTRICT d_mll_d_hp_kern_extra_ptr) const final {
        const int n_dim = static_cast<int>(hp_grad_inputs.X_ntrain_ndim_mat.n_cols);
        const int n_pair = idx_b_end - idx_b_begin;

        for (int i_dim = 0; i_dim < n_dim; ++i_dim) {
            const double* GPFLASH_RESTRICT x_dim_ptr = hp_grad_inputs.X_ntrain_ndim_mat.colptr(i_dim);

            double a_x_sum = 0.0;
            GPFLASH_OMP_SIMD_REDUCTION(+ : a_x_sum)
            for (int i_pair = 0; i_pair < n_pair; ++i_pair) a_x_sum += A_ptr[i_pair] * x_dim_ptr[idx_b_begin + i_pair];

            d_mll_d_hp_gamma_ptr[i_dim] += 2.0 * hp_grad_inputs.hp_gamma_vec(i_dim) * x_dim_ptr[idx_a] * a_x_sum;
        }

        // Reached only by a kernel that has hp_kern_extra, and only when the caller asks for its gradient. The
        // KernelBase default throws for a kernel that declares extras without defining it.
        if (d_mll_d_hp_kern_extra_ptr)
            add_hp_kern_extra_gradient_contribution(
                    hp_grad_inputs, idx_a, idx_b_begin, idx_b_end, A_ptr, d_mll_d_hp_kern_extra_ptr);
    }

    // d(K_ii)/d(hp_gamma_l) = dk/d(xdot_inner_prod) 2 hp_gamma_l x_il^2, (n_train, n_dim), with dk/d(xdot_inner_prod)
    // taken at each point's own inner product g_ii = ||xdot_i_vec||^2.
    arma::mat calc_d_kern_diag_d_hp_gamma_mat(const HpGradInputs& hp_grad_inputs) const final {
        const arma::vec& hp_gamma_vec = hp_grad_inputs.hp_gamma_vec;
        const arma::mat X_sq_ntrain_ndim_mat = arma::square(hp_grad_inputs.X_ntrain_ndim_mat);

        // g_ii = sum_d hp_gamma_d^2 x_id^2, one matrix-vector product for every point.
        const arma::vec g_self_vec = X_sq_ntrain_ndim_mat * arma::square(hp_gamma_vec);
        const double* hp_kern_extra_ptr = detail::calc_hp_kern_extra_ptr(hp_grad_inputs.hp_kern_extra_vec);
        const arma::vec d_kern_val_d_inner_prod_vec = calc_d_kern_val_d_inner_prod_vec(g_self_vec, hp_kern_extra_ptr);

        arma::mat d_kern_diag_d_hp_gamma_ntrain_ndim_mat = X_sq_ntrain_ndim_mat;
        d_kern_diag_d_hp_gamma_ntrain_ndim_mat.each_row() %= 2.0 * hp_gamma_vec.t();
        d_kern_diag_d_hp_gamma_ntrain_ndim_mat.each_col() %= d_kern_val_d_inner_prod_vec;

        return d_kern_diag_d_hp_gamma_ntrain_ndim_mat;
    }

    // ── The prior variance term of the posterior variance gradient ────────────
    // d(k(x', x'))/d(x'_l) = dk/d(xdot_inner_prod) 2 hp_gamma_l xdot'_l, needed because k(x', x') moves with the
    // query point, unlike a stationary kernel's constant diagonal.
    arma::vec calc_d_kern_diag_val_d_x_post(
            const arma::vec& xdot_post_vec,
            const arma::vec& hp_gamma_vec,
            const double* hp_kern_extra_ptr) const final {
        const double g_post = arma::dot(xdot_post_vec, xdot_post_vec);
        const double d_kern_val_d_inner_prod = calc_d_kern_val_d_inner_prod_scalar(g_post, hp_kern_extra_ptr);

        return (2.0 * d_kern_val_d_inner_prod) * (hp_gamma_vec % xdot_post_vec);
    }

    // The same for a batch of query points, (n_dim, n_post).
    arma::mat calc_d_kern_diag_val_d_x_post_batch(
            const arma::mat& Xdot_ndim_npost_mat,
            const arma::vec& hp_gamma_vec,
            const double* hp_kern_extra_ptr) const final {
        // g_tt for every query point, the column sums of the squared scaled points.
        const arma::vec g_self_vec = arma::vectorise(arma::sum(arma::square(Xdot_ndim_npost_mat), 0));
        const arma::vec d_kern_val_d_inner_prod_vec = calc_d_kern_val_d_inner_prod_vec(g_self_vec, hp_kern_extra_ptr);

        arma::mat d_kern_diag_val_ndim_npost_mat = Xdot_ndim_npost_mat;
        d_kern_diag_val_ndim_npost_mat.each_row() %= 2.0 * d_kern_val_d_inner_prod_vec.t();
        d_kern_diag_val_ndim_npost_mat.each_col() %= hp_gamma_vec;

        return d_kern_diag_val_ndim_npost_mat;
    }

    // ── The posterior input-space gradient ────────────────────────────────────
    // out[l] = d/d(x'_l) sum_i v_i k(x', x_i) = sum_i v_i dk/d(xdot_inner_prod) hp_gamma_l xdot_il. The caller folds
    // dk/d(xdot_inner_prod) into q_i, so the sum over i is one matrix-vector product; the query point does not enter,
    // g being linear in it.
    arma::vec calc_d_kern_sum_d_x_post(
            const arma::mat& Xdot_ntrain_ndim_mat,
            const arma::vec& /*xdot_post_vec*/,
            const arma::vec& hp_gamma_vec,
            const arma::vec& q_vec) const final {
        return hp_gamma_vec % (Xdot_ntrain_ndim_mat.t() * q_vec);
    }

    // The same for a batch, (n_dim, n_post). Only q_xdot_sum_ndim_npost_mat = Xdot_train Q is read.
    arma::mat calc_d_kern_sum_d_x_post_batch(
            const arma::mat& /*Xdot_ndim_npost_mat*/,
            const arma::mat& q_xdot_sum_ndim_npost_mat,
            const arma::rowvec& /*q_sum_row*/,
            const arma::vec& hp_gamma_vec,
            const double model_scale) const final {
        arma::mat d_kern_sum_d_x_post_ndim_npost_mat = q_xdot_sum_ndim_npost_mat;
        d_kern_sum_d_x_post_ndim_npost_mat.each_col() %= hp_gamma_vec;
        d_kern_sum_d_x_post_ndim_npost_mat *= model_scale;

        return d_kern_sum_d_x_post_ndim_npost_mat;
    }

   protected:
    KernelInnerProduct() = default;

    // Added to xdot_inner_prod by an inhomogeneous kernel, k = (xdot_inner_prod + XDOT_INNER_PROD_OFFSET)^p. Fixed,
    // not fitted: (g + c)^p = c^p (g/c + 1)^p, which hp_sig2_k and hp_gamma absorb, so 1 loses no generality.
    static constexpr double XDOT_INNER_PROD_OFFSET = 1.0;

    // dk/d(xdot_inner_prod) over a run, read from the value-and-rdot loop's c^(rdot) output so the derivative has
    // one source. The value the loop also writes is scratch here.
    arma::vec calc_d_kern_val_d_inner_prod_vec(const arma::vec& g_vec, const double* hp_kern_extra_ptr) const {
        arma::vec kern_val_scratch_vec = g_vec;
        arma::vec d_kern_val_d_inner_prod_vec(g_vec.n_elem, arma::fill::none);
        calc_kern_val_and_common_rdot_inplace(
                kern_val_scratch_vec.memptr(),
                static_cast<std::ptrdiff_t>(g_vec.n_elem),
                hp_kern_extra_ptr,
                d_kern_val_d_inner_prod_vec.memptr());

        return d_kern_val_d_inner_prod_vec;
    }

    // The single-point counterpart, for the one-query prior variance gradient.
    double calc_d_kern_val_d_inner_prod_scalar(const double g, const double* hp_kern_extra_ptr) const {
        double kern_val_scratch = g;
        double d_kern_val_d_inner_prod = 0.0;
        calc_kern_val_and_common_rdot_inplace(&kern_val_scratch, 1, hp_kern_extra_ptr, &d_kern_val_d_inner_prod);

        return d_kern_val_d_inner_prod;
    }
};

}  // namespace kernel
}  // namespace gpflash
