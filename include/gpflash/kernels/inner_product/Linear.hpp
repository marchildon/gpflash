/**
 * Linear.hpp -- the inhomogeneous ARD linear kernel.
 *
 *   k_lin(x_i_vec, x_j_vec) = xdot_inner_prod + XDOT_INNER_PROD_OFFSET,   xdot_inner_prod = sum_d hp_gamma_d^2 x_id
 * x_jd
 *
 * so dk/d(xdot_inner_prod) is 1. KernelInnerProduct derives every gradient from those two facts. The offset gives
 * the prior line a random intercept of variance hp_sig2_k XDOT_INNER_PROD_OFFSET alongside a slope of variance
 * hp_sig2_k hp_gamma_d^2.
 *
 * K has rank at most n_dim + 1, so it is singular once n_train > n_dim + 1 and the nugget is load-bearing for the
 * Cholesky. k(x, x) = ||xdot_vec||^2 + XDOT_INNER_PROD_OFFSET varies with the point, which puts this kernel on the
 * preconditioned path, and is positive everywhere, which that path needs.
 *
 * The constant mean model's weight already carries the intercept, so the MLL drives the offset's intercept variance
 * toward zero: a fit ends with hp_gamma at its upper bound on noise-free data, or hp_sig2_k at its lower bound on
 * noisy data. The prediction is unaffected; the bound report flags it.
 */
#pragma once

#include <cstddef>  // std::ptrdiff_t

#include "gpflash/core/Platform.hpp"                             // GPFLASH_RESTRICT
#include "gpflash/kernels/inner_product/KernelInnerProduct.hpp"  // KernelInnerProduct

namespace gpflash {
namespace kernel {

class Linear final : public KernelInnerProduct {
   public:
    const char* kernel_name() const override { return "linear"; }
    KernelId kernel_id() const override { return KernelId::Linear; }
    bool is_kern_diag_unity() const override { return false; }  // k(x, x) = ||xdot_vec||^2 + XDOT_INNER_PROD_OFFSET

    void calc_kern_val_inplace(
            double* GPFLASH_RESTRICT xdot_inner_prod_ptr,
            const std::ptrdiff_t n_elem,
            const double* /*hp_kern_extra_ptr*/) const override {
        for (std::ptrdiff_t i_elem = 0; i_elem < n_elem; ++i_elem)
            xdot_inner_prod_ptr[i_elem] += XDOT_INNER_PROD_OFFSET;
    }

    void calc_kern_val_and_common_rdot_inplace(
            double* GPFLASH_RESTRICT xdot_inner_prod_ptr,
            const std::ptrdiff_t n_elem,
            const double* /*hp_kern_extra_ptr*/,
            double* GPFLASH_RESTRICT kern_common_rdot_ptr_out) const override {
        for (std::ptrdiff_t i_elem = 0; i_elem < n_elem; ++i_elem) {
            xdot_inner_prod_ptr[i_elem] += XDOT_INNER_PROD_OFFSET;
            kern_common_rdot_ptr_out[i_elem] = 1.0;
        }
    }

    double calc_kern_diag_val(const double xdot_sq_norm, const double* /*hp_kern_extra_ptr*/) const override {
        return xdot_sq_norm + XDOT_INNER_PROD_OFFSET;
    }

    arma::vec calc_kern_diag_val_vec(
            const arma::vec& xdot_sq_norm_vec, const double* /*hp_kern_extra_ptr*/) const override {
        return xdot_sq_norm_vec + XDOT_INNER_PROD_OFFSET;
    }
};

}  // namespace kernel
}  // namespace gpflash
