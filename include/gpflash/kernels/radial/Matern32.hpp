/** Matern32.hpp -- Matérn 3/2 kernel. */
#pragma once

#include <cmath>    // std::exp, std::sqrt
#include <cstddef>  // std::ptrdiff_t

#include "gpflash/core/Platform.hpp"                // GPFLASH_RESTRICT
#include "gpflash/kernels/radial/KernelRadial.hpp"  // KernelRadial

namespace gpflash {
namespace kernel {

// Every loop below overwrites the buffer it reads: IN: the scaled squared distance, OUT: the kernel value. Each
// forms this kernel's factors (see private) and multiplies them as KernelRadial.hpp writes them:
//
//     k = s^(k) c^(k),   c^(rdot) = s^(rdot) c^(k)
class Matern32 final : public KernelRadial {
   public:
    const char* kernel_name() const override { return "matern_3_2"; }
    KernelId kernel_id() const override { return KernelId::Matern32; }

    void calc_kern_val_inplace(
            double* GPFLASH_RESTRICT rdot_sq_norm_ptr,
            const std::ptrdiff_t n_elem,
            const double* /*hp_kern_extra_ptr*/) const override {
        for (std::ptrdiff_t i_elem = 0; i_elem < n_elem; ++i_elem) {
            const double scaled_rdot_norm = calc_scaled_rdot_norm(rdot_sq_norm_ptr[i_elem]);
            rdot_sq_norm_ptr[i_elem] = calc_kern_static_k(scaled_rdot_norm) * calc_kern_common_k(scaled_rdot_norm);
        }
    }

    void calc_kern_val_and_common_rdot_inplace(
            double* GPFLASH_RESTRICT rdot_sq_norm_ptr,
            const std::ptrdiff_t n_elem,
            const double* /*hp_kern_extra_ptr*/,
            double* GPFLASH_RESTRICT kern_common_rdot_ptr_out) const override {
        for (std::ptrdiff_t i_elem = 0; i_elem < n_elem; ++i_elem) {
            const double scaled_rdot_norm = calc_scaled_rdot_norm(rdot_sq_norm_ptr[i_elem]);
            const double kern_common_k = calc_kern_common_k(scaled_rdot_norm);
            rdot_sq_norm_ptr[i_elem] = calc_kern_static_k(scaled_rdot_norm) * kern_common_k;
            kern_common_rdot_ptr_out[i_elem] = kern_static_rdot * kern_common_k;
        }
    }

    // The two functions below have no production caller; the unit tests use them to check the products above.
    // kern_val = s^(k) c^(k)
    static double calc_kern_val_unfactored(const double rdot_sq_norm, const double* /*hp_kern_extra_ptr*/) {
        const double rdot_norm = std::sqrt(rdot_sq_norm);
        return (1.0 + std::sqrt(3.0) * rdot_norm) * std::exp(-std::sqrt(3.0) * rdot_norm);
    }

    // kern_common_rdot = s^(rdot) c^(k)
    static double calc_kern_common_rdot_unfactored(const double rdot_sq_norm, const double* /*hp_kern_extra_ptr*/) {
        return -3.0 * std::exp(-std::sqrt(3.0) * std::sqrt(rdot_sq_norm));
    }

   private:
    // k = (1 + sqrt(3) ||rdot_vec||) exp(-sqrt(3) ||rdot_vec||), as the three factors the loops above read. The
    // first is the scaled distance the other two are written in terms of, so a loop forms it once per pair.
    static double calc_scaled_rdot_norm(const double rdot_sq_norm) { return std::sqrt(3.0 * rdot_sq_norm); }

    static double calc_kern_common_k(const double scaled_rdot_norm) { return std::exp(-scaled_rdot_norm); }

    static double calc_kern_static_k(const double scaled_rdot_norm) { return 1.0 + scaled_rdot_norm; }

    static constexpr double kern_static_rdot = -3.0;  // s^(rdot), the same for every pair
};

}  // namespace kernel
}  // namespace gpflash
