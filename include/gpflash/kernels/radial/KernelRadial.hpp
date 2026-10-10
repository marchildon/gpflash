/**
 * KernelRadial.hpp -- what every radial kernel shares beyond the squared-distance machinery, as the KernelBase
 * methods a radial kernel does not write.
 *
 * A radial kernel reads its inputs only through the scaled distance ||rdot_vec||, where
 *
 *     rdot_vec = hp_gamma_vec .* (x_i_vec - x_j_vec)
 *
 * Radial here does not mean isotropic: hp_gamma_vec carries one inverse length scale per dimension, so these
 * kernels are anisotropic, as is every kernel in the package unless its own header says otherwise. Depending on
 * nothing else lets the GpFlash paper factor the kernel and both of its derivatives into a per-pair part and a
 * per-dimension part:
 *
 *     k               = s^(k) c^(k)
 *     c^(rdot)        = s^(rdot) c^(k)
 *     d(k)/d(r_l)     = h^(r,l) c^(rdot)
 *     d(k)/d(gamma_l) = h^(gamma,l) c^(rdot)
 *
 * The h terms are shared by every radial kernel, so the chain rules that use them are written once here. The s
 * and c terms are each kernel's own, so a derived kernel writes:
 *
 *     calc_kern_val_inplace                    values alone, over a run
 *     calc_kern_val_and_common_rdot_inplace    values and c^(rdot), over a run
 *     kernel_name, kernel_id
 *
 * and, where it carries extras, n_hp_kern_extra and add_hp_kern_extra_gradient_contribution with its _mirrored
 * counterpart -- only RatQuad does. The squared-distance machinery is on the KernelStationary base; the gradient and
 * posterior chain rules specific to a radial kernel are here and `final`.
 *
 * This family runs no feature map, so Y is X, n_feature is n_dim, and Ydot is Xdot = hp_gamma .* X. It inherits
 * KernelBase's feature-point defaults, which is why only X appears below.
 */
#pragma once

#include <armadillo>  // arma::mat, arma::vec, arma::rowvec

#include "gpflash/core/Platform.hpp"             // GPFLASH_RESTRICT, GPFLASH_OMP_SIMD, PERF_GRAD_MIN_PAIR_PER_DIM
#include "gpflash/kernels/HpGradInputs.hpp"      // HpGradInputs
#include "gpflash/kernels/KernelStationary.hpp"  // KernelStationary

namespace gpflash {
namespace kernel {

class KernelRadial : public KernelStationary {
   public:
    bool is_symmetrizable() const final { return true; }

    // A radial kernel runs no feature map, so the mirror acts on Ydot as it does on the inputs.
    arma::vec calc_feature_mirror_sign_vec(const arma::vec& mirror_sign_vec) const final { return mirror_sign_vec; }

    // ── The hyperparameter gradient ───────────────────────────────────────────
    // For an arbitrary hyperparameter alpha_l:
    //
    //     d(ln L)/d(alpha_l)  = sigma_k^2 ( <A, H^(alpha,l)>_F + zeta d(eta)/d(alpha_l) )
    //     A                   = Psi .* C^(rdot)
    //     Psi                 = (v_res v_res^T - Sigma^-1) / 2
    //     <A, H^(alpha,l)>_F  = sum_a sum_row_a
    //     sum_row_a           = sum_b A_ab h^(alpha,l)_ab
    //
    // <A, B>_F is the Frobenius inner product. One call adds sum_row_a for every hyperparameter at once. A and
    // H^(alpha,l) are both symmetric, so the caller walks only the strict lower triangle and owns the resulting
    // factor of two, along with sigma_k^2 and the zeta term.
    //
    // Neither matrix is formed: H^(alpha,l) never exists, A arrives one column at a time, and A carries
    // c^(rdot), so no kernel value, exponential or square root is evaluated here.
    //
    // alpha = gamma, all n_dim at once, with the h term every radial kernel shares:
    //
    //     h^(gamma,l)_ab = gamma_l (x_al - x_bl)^2
    //
    // The eta gradient reuses this with A_ptr built from sign(K_corr) over its active row rather than from Psi.
    void add_hp_gradient_contribution(
            const HpGradInputs& hp_grad_inputs,
            const int idx_a,
            const int idx_b_begin,
            const int idx_b_end,
            const double* GPFLASH_RESTRICT A_ptr,
            double* GPFLASH_RESTRICT d_mll_d_hp_gamma_ptr,
            double* GPFLASH_RESTRICT d_mll_d_hp_kern_extra_ptr) const final {
        add_hp_gamma_gradient_contribution</*kHasMirror=*/false>(
                hp_grad_inputs, idx_a, idx_b_begin, idx_b_end, A_ptr, nullptr, d_mll_d_hp_gamma_ptr);

        // Reached only by a kernel that has hp_kern_extra, and only when the caller asks for its gradient.
        if (d_mll_d_hp_kern_extra_ptr)
            add_hp_kern_extra_gradient_contribution(
                    hp_grad_inputs, idx_a, idx_b_begin, idx_b_end, A_ptr, d_mll_d_hp_kern_extra_ptr);
    }

    // The mirrored term of a symmetry model, which differs from the sum above only in the sign the second
    // point's coordinate carries.
    void add_hp_gradient_contribution_mirrored(
            const HpGradInputs& hp_grad_inputs,
            const int idx_a,
            const int idx_b_begin,
            const int idx_b_end,
            const double* GPFLASH_RESTRICT A_ptr,
            const arma::vec& mirror_sign_vec,
            double* GPFLASH_RESTRICT d_mll_d_hp_gamma_ptr,
            double* GPFLASH_RESTRICT d_mll_d_hp_kern_extra_ptr) const final {
        add_hp_gamma_gradient_contribution</*kHasMirror=*/true>(
                hp_grad_inputs, idx_a, idx_b_begin, idx_b_end, A_ptr, mirror_sign_vec.memptr(), d_mll_d_hp_gamma_ptr);

        if (d_mll_d_hp_kern_extra_ptr)
            add_hp_kern_extra_gradient_contribution_mirrored(
                    hp_grad_inputs, idx_a, idx_b_begin, idx_b_end, A_ptr, mirror_sign_vec, d_mll_d_hp_kern_extra_ptr);
    }

    // ── The posterior input-space gradient ────────────────────────────────────
    // Both methods below compute q^T H^(r), the factor the posterior mean and variance gradients share, and
    // differ only in how the sum over training points is taken. The paper gives, for radial kernels:
    //
    //     d(mean)/d(x')      = sigma_k^2 q_mu^T H^(r)
    //     d(variance)/d(x')  = -2 sigma_k^4 q_sigma^T H^(r)
    //     (q^T H^(r))_l      = sum_i q_i h^(r,l)_i
    //     h^(r,l)_i          = gamma_l^2 (x'_l - x_il)
    //
    // The caller passes q_mu or q_sigma, so one method serves both. H^(r) is never formed.
    //
    // One query point, one dot product per dimension. The difference is taken inside the sum, which stays
    // accurate where the query sits among the training points.
    arma::vec calc_d_kern_sum_d_x_post(
            const arma::mat& Xdot_ntrain_ndim_mat,  // so each dimension is unit-stride
            const arma::vec& xdot_post_vec,
            const arma::vec& hp_gamma_vec,
            const arma::vec& q_vec) const final {
        const int n_train = static_cast<int>(Xdot_ntrain_ndim_mat.n_rows);
        const int n_dim = static_cast<int>(Xdot_ntrain_ndim_mat.n_cols);
        const double* GPFLASH_RESTRICT q_ptr = q_vec.memptr();
        const double* GPFLASH_RESTRICT xdot_post_ptr = xdot_post_vec.memptr();
        arma::vec d_kern_sum_d_x_post_vec(n_dim);

        // Per dimension, the q-weighted sum of rdot = query minus training point.
        for (int i_dim = 0; i_dim < n_dim; ++i_dim) {
            const double* GPFLASH_RESTRICT xdot_train_dim_ptr = Xdot_ntrain_ndim_mat.colptr(i_dim);
            const double xdot_post_dim = xdot_post_ptr[i_dim];
            double q_rdot_sum = 0.0;
            GPFLASH_OMP_SIMD_REDUCTION(+ : q_rdot_sum)
            for (int i_train = 0; i_train < n_train; ++i_train)
                q_rdot_sum += q_ptr[i_train] * (xdot_post_dim - xdot_train_dim_ptr[i_train]);
            d_kern_sum_d_x_post_vec(i_dim) = q_rdot_sum;
        }

        // The remaining hp_gamma of h^(r,l); the points carry the other one already.
        return hp_gamma_vec % d_kern_sum_d_x_post_vec;
    }

    // Many query points, returned as (n_dim, n_post). Splitting the difference lets the sum over training points
    // become one DGEMM, which the caller has already done. For query point p and dimension l:
    //
    //     sum_i q_ip (x'_pl - x_il) = x'_pl sum_i q_ip - sum_i q_ip x_il
    //     returned(l, p)            = model_scale gamma_l ( xdot'_pl q_sum_row_p - q_xdot_sum_mat_lp )
    //
    // so each entry is a difference of two precomputed aggregates rather than a dot product. model_scale is the
    // caller's leading factor: sigma_k^2 for the mean, -2 sigma_k^4 for the variance.
    arma::mat calc_d_kern_sum_d_x_post_batch(
            const arma::mat& Xdot_ndim_npost_mat,
            const arma::mat& q_xdot_sum_ndim_npost_mat,
            const arma::rowvec& q_sum_row,
            const arma::vec& hp_gamma_vec,
            const double model_scale) const final {
        arma::mat d_kern_sum_d_x_post_ndim_npost_mat = Xdot_ndim_npost_mat;

        // Each query point times its own q sum, less the training aggregate: sum_i q_i rdot.
        d_kern_sum_d_x_post_ndim_npost_mat.each_row() %= q_sum_row;
        d_kern_sum_d_x_post_ndim_npost_mat -= q_xdot_sum_ndim_npost_mat;

        // The remaining hp_gamma of h^(r,l), then the caller's model factor.
        d_kern_sum_d_x_post_ndim_npost_mat.each_col() %= hp_gamma_vec;

        d_kern_sum_d_x_post_ndim_npost_mat *= model_scale;
        return d_kern_sum_d_x_post_ndim_npost_mat;
    }

   protected:
    KernelRadial() = default;

   private:
    // h^(gamma,l) = hp_gamma_l r_l^2, with r_l the separation in dimension l: x_bl - x_al, or x_bl s_l - x_al
    // for the mirrored term of a symmetry model. The mirror is a template parameter so the unmirrored
    // path, which every kernel takes, carries none of its arithmetic.
    template <bool kHasMirror>
    static void add_hp_gamma_gradient_contribution(
            const HpGradInputs& hp_grad_inputs,
            const int idx_a,
            const int idx_b_begin,
            const int idx_b_end,
            const double* GPFLASH_RESTRICT A_ptr,
            const double* GPFLASH_RESTRICT mirror_sign_ptr,
            double* GPFLASH_RESTRICT d_mll_d_hp_gamma_ptr) {
        const int n_dim = static_cast<int>(hp_grad_inputs.X_ntrain_ndim_mat.n_cols);
        const int n_pair = idx_b_end - idx_b_begin;
        const double* GPFLASH_RESTRICT hp_gamma_ptr = hp_grad_inputs.hp_gamma_vec.memptr();

        // The inner loop runs over pairs where there are enough of them, and otherwise over dimensions.
        if (n_pair >= PERF_GRAD_MIN_PAIR_PER_DIM * n_dim) {
            // Each dimension a unit-stride column of the transposed points; hp_gamma_l factors out of the sum.
            for (int i_dim = 0; i_dim < n_dim; ++i_dim) {
                const double* GPFLASH_RESTRICT x_dim_ptr = hp_grad_inputs.X_ntrain_ndim_mat.colptr(i_dim);
                const double x_a = x_dim_ptr[idx_a];

                double sum_row_a = 0.0;
                GPFLASH_OMP_SIMD_REDUCTION(+ : sum_row_a)
                for (int i_pair = 0; i_pair < n_pair; ++i_pair) {
                    const double x_b = x_dim_ptr[idx_b_begin + i_pair];
                    const double r_dim = kHasMirror ? (mirror_sign_ptr[i_dim] * x_b - x_a) : (x_b - x_a);

                    // Scaled by A before squaring: A's C^(rdot) vanishes for a far pair, which keeps r_dim^2 finite.
                    sum_row_a += (A_ptr[i_pair] * r_dim) * r_dim;
                }
                d_mll_d_hp_gamma_ptr[i_dim] += hp_gamma_ptr[i_dim] * sum_row_a;
            }
        } else {
            // Each point a unit-stride column of the untransposed points.
            const arma::mat& X_ndim_ntrain_mat = *hp_grad_inputs.X_ndim_ntrain_mat_ptr;
            const double* GPFLASH_RESTRICT x_a_ptr = X_ndim_ntrain_mat.colptr(idx_a);
            for (int i_pair = 0; i_pair < n_pair; ++i_pair) {
                const double* GPFLASH_RESTRICT x_b_ptr = X_ndim_ntrain_mat.colptr(idx_b_begin + i_pair);
                const double A_val = A_ptr[i_pair];

                GPFLASH_OMP_SIMD
                for (int i_dim = 0; i_dim < n_dim; ++i_dim) {
                    const double x_b = x_b_ptr[i_dim];
                    const double r_dim =
                            kHasMirror ? (mirror_sign_ptr[i_dim] * x_b - x_a_ptr[i_dim]) : (x_b - x_a_ptr[i_dim]);

                    // Scaled by A before squaring, as above.
                    d_mll_d_hp_gamma_ptr[i_dim] += (A_val * hp_gamma_ptr[i_dim] * r_dim) * r_dim;
                }
            }
        }
    }
};

}  // namespace kernel
}  // namespace gpflash
