/** RatQuad.hpp -- rational quadratic kernel and its shape derivative. */
#pragma once

#include <armadillo>  // arma::vec
#include <cmath>      // std::log, std::pow
#include <cstddef>    // std::ptrdiff_t, std::size_t

#include "gpflash/core/Platform.hpp"                // GPFLASH_RESTRICT
#include "gpflash/kernels/HpGradInputs.hpp"         // HpGradInputs
#include "gpflash/kernels/radial/KernelRadial.hpp"  // KernelRadial

namespace gpflash {
namespace kernel {

// Every loop below overwrites the buffer it reads: IN: the scaled squared distance, OUT: the kernel value. Each
// forms this kernel's factors (see private) and multiplies them as KernelRadial.hpp writes them:
//
//     k = s^(k) c^(k),   c^(rdot) = s^(rdot) c^(k)
class RatQuad final : public KernelRadial {
   public:
    const char* kernel_name() const override { return "rat_quad"; }
    KernelId kernel_id() const override { return KernelId::RatQuad; }

    // hp_rho > 0, the shape.
    int n_hp_kern_extra(const int /*n_dim_input*/) const override { return 1; }

    void calc_kern_val_inplace(
            double* GPFLASH_RESTRICT rdot_sq_norm_ptr,
            const std::ptrdiff_t n_elem,
            const double* hp_kern_extra_ptr) const override {
        const double hp_rho = hp_kern_extra_ptr[0];
        for (std::ptrdiff_t i_elem = 0; i_elem < n_elem; ++i_elem) {
            const double kern_static_k = calc_kern_static_k(rdot_sq_norm_ptr[i_elem], hp_rho);
            rdot_sq_norm_ptr[i_elem] = kern_static_k * calc_kern_common_k(kern_static_k, hp_rho);
        }
    }

    void calc_kern_val_and_common_rdot_inplace(
            double* GPFLASH_RESTRICT rdot_sq_norm_ptr,
            const std::ptrdiff_t n_elem,
            const double* hp_kern_extra_ptr,
            double* GPFLASH_RESTRICT kern_common_rdot_ptr_out) const override {
        const double hp_rho = hp_kern_extra_ptr[0];
        for (std::ptrdiff_t i_elem = 0; i_elem < n_elem; ++i_elem) {
            const double kern_static_k = calc_kern_static_k(rdot_sq_norm_ptr[i_elem], hp_rho);
            const double kern_common_k = calc_kern_common_k(kern_static_k, hp_rho);
            rdot_sq_norm_ptr[i_elem] = kern_static_k * kern_common_k;
            kern_common_rdot_ptr_out[i_elem] = kern_static_rdot * kern_common_k;
        }
    }

    // alpha = hp_rho, whose h term is
    //
    //     h^(rho)_ab = s^(k) ln s^(k) - (s^(k) - 1)
    //
    // KernelRadial documents A, the index range and what the caller owns. s^(k) is rebuilt from the points
    // rather than cached: arithmetic, but no extra memory.
    void add_hp_kern_extra_gradient_contribution(
            const HpGradInputs& hp_grad_inputs,
            const int idx_a,
            const int idx_b_begin,
            const int idx_b_end,
            const double* GPFLASH_RESTRICT A_ptr,
            double* GPFLASH_RESTRICT d_mll_d_hp_kern_extra_ptr) const override {
        add_hp_rho_gradient_contribution</*kHasMirror=*/false>(
                hp_grad_inputs, idx_a, idx_b_begin, idx_b_end, A_ptr, nullptr, d_mll_d_hp_kern_extra_ptr);
    }

    // The same at the pair (x_a, S x_b), for the mirrored term of a symmetry model.
    void add_hp_kern_extra_gradient_contribution_mirrored(
            const HpGradInputs& hp_grad_inputs,
            const int idx_a,
            const int idx_b_begin,
            const int idx_b_end,
            const double* GPFLASH_RESTRICT A_ptr,
            const arma::vec& mirror_sign_vec,
            double* GPFLASH_RESTRICT d_mll_d_hp_kern_extra_ptr) const override {
        add_hp_rho_gradient_contribution</*kHasMirror=*/true>(
                hp_grad_inputs,
                idx_a,
                idx_b_begin,
                idx_b_end,
                A_ptr,
                mirror_sign_vec.memptr(),
                d_mll_d_hp_kern_extra_ptr);
    }

    // The two functions below have no production caller; the unit tests use them to check the products above.
    // kern_val = s^(k) c^(k)
    static double calc_kern_val_unfactored(const double rdot_sq_norm, const double* hp_kern_extra_ptr) {
        const double hp_rho = hp_kern_extra_ptr[0];
        return std::pow(1.0 + rdot_sq_norm / (2.0 * hp_rho), -hp_rho);
    }

    // kern_common_rdot = s^(rdot) c^(k)
    static double calc_kern_common_rdot_unfactored(const double rdot_sq_norm, const double* hp_kern_extra_ptr) {
        const double hp_rho = hp_kern_extra_ptr[0];
        return -std::pow(1.0 + rdot_sq_norm / (2.0 * hp_rho), -hp_rho - 1.0);
    }

   private:
    // The separation in dimension l is x_bl - x_al, or s_l x_bl - x_al for the mirrored term. The mirror is a
    // template parameter so the unmirrored path carries none of its arithmetic.
    template <bool kHasMirror>
    static void add_hp_rho_gradient_contribution(
            const HpGradInputs& hp_grad_inputs,
            const int idx_a,
            const int idx_b_begin,
            const int idx_b_end,
            const double* GPFLASH_RESTRICT A_ptr,
            const double* GPFLASH_RESTRICT mirror_sign_ptr,
            double* GPFLASH_RESTRICT d_mll_d_hp_kern_extra_ptr) {
        const double inv_2_hp_rho = 0.5 / hp_grad_inputs.hp_kern_extra_vec(0);
        const int n_dim = static_cast<int>(hp_grad_inputs.X_ntrain_ndim_mat.n_cols);
        // Column i_dim starts at X + i_dim * n_train.
        const double* GPFLASH_RESTRICT X_ptr = hp_grad_inputs.X_ntrain_ndim_mat.memptr();
        const std::size_t n_train_stride = hp_grad_inputs.X_ntrain_ndim_mat.n_rows;
        const double* GPFLASH_RESTRICT hp_gamma_ptr = hp_grad_inputs.hp_gamma_vec.memptr();
        const int n_pair = idx_b_end - idx_b_begin;

        double sum_row_a = 0.0;
        for (int i_pair = 0; i_pair < n_pair; ++i_pair) {
            const int idx_b = idx_b_begin + i_pair;

            double rdot_sq_norm = 0.0;
            for (int i_dim = 0; i_dim < n_dim; ++i_dim) {
                const double* GPFLASH_RESTRICT x_dim_ptr = X_ptr + static_cast<std::size_t>(i_dim) * n_train_stride;
                const double x_b = kHasMirror ? mirror_sign_ptr[i_dim] * x_dim_ptr[idx_b] : x_dim_ptr[idx_b];
                const double rdot_dim = hp_gamma_ptr[i_dim] * (x_b - x_dim_ptr[idx_a]);
                rdot_sq_norm += rdot_dim * rdot_dim;
            }
            const double kern_static_k = 1.0 + rdot_sq_norm * inv_2_hp_rho;

            const double kern_h_hp_rho = kern_static_k * std::log(kern_static_k) - (kern_static_k - 1.0);
            sum_row_a += A_ptr[i_pair] * kern_h_hp_rho;
        }
        d_mll_d_hp_kern_extra_ptr[0] += sum_row_a;
    }

    // k = s^(k)^(-hp_rho), as the factors the loops above read: s^(k) comes first, one pow over it gives c^(k),
    // and the value is then one more multiply.
    static double calc_kern_static_k(const double rdot_sq_norm, const double hp_rho) {
        return 1.0 + rdot_sq_norm / (2.0 * hp_rho);
    }

    static double calc_kern_common_k(const double kern_static_k, const double hp_rho) {
        return std::pow(kern_static_k, -hp_rho - 1.0);
    }

    static constexpr double kern_static_rdot = -1.0;  // s^(rdot), the same for every pair
};

}  // namespace kernel
}  // namespace gpflash
