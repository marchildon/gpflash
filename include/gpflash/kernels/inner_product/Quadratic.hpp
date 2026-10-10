/**
 * Quadratic.hpp -- the inhomogeneous quadratic kernel.
 *
 *   k(x_i_vec, x_j_vec) = (xdot_inner_prod + XDOT_INNER_PROD_OFFSET)^2,   xdot_inner_prod = dot(xdot_i_vec, xdot_j_vec)
 *
 * so dk/d(xdot_inner_prod) = 2 (xdot_inner_prod + XDOT_INNER_PROD_OFFSET). The offset makes it inhomogeneous,
 * giving the constant and linear terms as well as the quadratic one; hp_gamma sets the relative weight of the inner
 * product against that offset.
 *
 * Positive semi-definite as the square of the positive semi-definite xdot_inner_prod + XDOT_INNER_PROD_OFFSET, and of
 * rank at most (n_dim + 1)(n_dim + 2)/2, so K is singular once n_train exceeds that and the nugget is load-bearing for
 * the Cholesky. k(x, x) = (||xdot_vec||^2 + XDOT_INNER_PROD_OFFSET)^2 varies with the point, which puts this kernel
 * on the preconditioned path.
 */
#pragma once

#include <cstddef>  // std::ptrdiff_t

#include "gpflash/core/Platform.hpp"                             // GPFLASH_RESTRICT
#include "gpflash/kernels/inner_product/KernelInnerProduct.hpp"  // KernelInnerProduct

namespace gpflash {
namespace kernel {

class Quadratic final : public KernelInnerProduct {
   public:
    const char* kernel_name() const override { return "quadratic"; }
    KernelId kernel_id() const override { return KernelId::Quadratic; }
    bool is_kern_diag_unity() const override { return false; }  // k(x, x) = (||xdot_vec||^2 + XDOT_INNER_PROD_OFFSET)^2

    void calc_kern_val_inplace(
            double* GPFLASH_RESTRICT xdot_inner_prod_ptr,
            const std::ptrdiff_t n_elem,
            const double* /*hp_kern_extra_ptr*/) const override {
        for (std::ptrdiff_t i_elem = 0; i_elem < n_elem; ++i_elem) {
            const double base = xdot_inner_prod_ptr[i_elem] + XDOT_INNER_PROD_OFFSET;
            xdot_inner_prod_ptr[i_elem] = base * base;
        }
    }

    void calc_kern_val_and_common_rdot_inplace(
            double* GPFLASH_RESTRICT xdot_inner_prod_ptr,
            const std::ptrdiff_t n_elem,
            const double* /*hp_kern_extra_ptr*/,
            double* GPFLASH_RESTRICT kern_common_rdot_ptr_out) const override {
        for (std::ptrdiff_t i_elem = 0; i_elem < n_elem; ++i_elem) {
            const double base = xdot_inner_prod_ptr[i_elem] + XDOT_INNER_PROD_OFFSET;
            kern_common_rdot_ptr_out[i_elem] = 2.0 * base;
            xdot_inner_prod_ptr[i_elem] = base * base;
        }
    }

    double calc_kern_diag_val(const double xdot_sq_norm, const double* /*hp_kern_extra_ptr*/) const override {
        const double base = xdot_sq_norm + XDOT_INNER_PROD_OFFSET;
        return base * base;
    }

    arma::vec calc_kern_diag_val_vec(
            const arma::vec& xdot_sq_norm_vec, const double* /*hp_kern_extra_ptr*/) const override {
        return arma::square(xdot_sq_norm_vec + XDOT_INNER_PROD_OFFSET);
    }
};

}  // namespace kernel
}  // namespace gpflash
