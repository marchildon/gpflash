/** SqExp.hpp -- squared exponential kernel. */
#pragma once

#include <cmath>    // std::exp
#include <cstddef>  // std::ptrdiff_t

#include "gpflash/core/Platform.hpp"                // GPFLASH_RESTRICT
#include "gpflash/kernels/radial/KernelRadial.hpp"  // KernelRadial

namespace gpflash {
namespace kernel {

// Every loop below overwrites the buffer it reads: IN: the scaled squared distance, OUT: the kernel value. This
// kernel's s^(k) is 1, so the value is c^(k) itself and only c^(rdot) = s^(rdot) c^(k) needs a product.
class SqExp final : public KernelRadial {
   public:
    const char* kernel_name() const override { return "sq_exp"; }
    KernelId kernel_id() const override { return KernelId::SqExp; }

    void calc_kern_val_inplace(
            double* GPFLASH_RESTRICT rdot_sq_norm_ptr,
            const std::ptrdiff_t n_elem,
            const double* /*hp_kern_extra_ptr*/) const override {
        for (std::ptrdiff_t i_elem = 0; i_elem < n_elem; ++i_elem)
            rdot_sq_norm_ptr[i_elem] = calc_kern_common_k(rdot_sq_norm_ptr[i_elem]);
    }

    void calc_kern_val_and_common_rdot_inplace(
            double* GPFLASH_RESTRICT rdot_sq_norm_ptr,
            const std::ptrdiff_t n_elem,
            const double* /*hp_kern_extra_ptr*/,
            double* GPFLASH_RESTRICT kern_common_rdot_ptr_out) const override {
        for (std::ptrdiff_t i_elem = 0; i_elem < n_elem; ++i_elem) {
            const double kern_common_k = calc_kern_common_k(rdot_sq_norm_ptr[i_elem]);
            rdot_sq_norm_ptr[i_elem] = kern_common_k;
            kern_common_rdot_ptr_out[i_elem] = kern_static_rdot * kern_common_k;
        }
    }

    // The two functions below have no production caller; the unit tests use them to check the products above.
    // kern_val = s^(k) c^(k)
    static double calc_kern_val_unfactored(const double rdot_sq_norm, const double* /*hp_kern_extra_ptr*/) {
        return std::exp(-0.5 * rdot_sq_norm);
    }

    // kern_common_rdot = s^(rdot) c^(k)
    static double calc_kern_common_rdot_unfactored(const double rdot_sq_norm, const double* /*hp_kern_extra_ptr*/) {
        return -std::exp(-0.5 * rdot_sq_norm);
    }

   private:
    // k = exp(-0.5 ||rdot_vec||^2), which is c^(k).
    static double calc_kern_common_k(const double rdot_sq_norm) { return std::exp(-0.5 * rdot_sq_norm); }

    static constexpr double kern_static_rdot = -1.0;  // s^(rdot), the same for every pair
};

}  // namespace kernel
}  // namespace gpflash
