/**
 * KernelDispatch.hpp -- the mapping between gpflash::kernel::KernelId and the kernel it names.
 *
 * kernel_by_id turns an id into the one KernelBase instance for that kernel, and both name lookups go through
 * it. The switch is generated from GPFLASH_KERNEL_LIST, so registering a kernel means adding it there and
 * including it below.
 *
 * Kernels hold no state (KernelBase.hpp), so one function-local static per id suffices: no allocation, no
 * lifetime question, thread-safe by the language. A caller wanting one kernel in its binary constructs it
 * directly instead of including this header.
 */
#pragma once

#include <string>  // std::string, for parse_kernel_id's argument

#include "gpflash/core/Errors.hpp"         // gpflash::ErrorInvalidInput
#include "gpflash/inputs/KernelId.hpp"     // KernelId, N_KERNEL_ID, GPFLASH_KERNEL_LIST
#include "gpflash/kernels/KernelBase.hpp"  // KernelBase

// One include per kernel in GPFLASH_KERNEL_LIST; the switch below needs each type complete.
#include "gpflash/kernels/inner_product/Linear.hpp"
#include "gpflash/kernels/inner_product/Quadratic.hpp"
#include "gpflash/kernels/radial/Matern32.hpp"
#include "gpflash/kernels/radial/Matern52.hpp"
#include "gpflash/kernels/radial/RatQuad.hpp"
#include "gpflash/kernels/radial/SqExp.hpp"
#include "gpflash/kernels/stationary/Periodic.hpp"

namespace gpflash {
namespace kernel {

#define GPFLASH_KERNEL_DISPATCH_CASE(kernel_type_name) \
    case KernelId::kernel_type_name: {                 \
        static const kernel_type_name kernel_instance; \
        return kernel_instance;                        \
    }

inline const KernelBase& kernel_by_id(const KernelId kernel_id) {
    switch (kernel_id) {
        GPFLASH_KERNEL_LIST(GPFLASH_KERNEL_DISPATCH_CASE)
        case KernelId::End:
            break;  // not a kernel; the bound N_KERNEL_ID is derived from it
    }

    throw gpflash::ErrorInvalidInput("gpflash::kernel::kernel_by_id: invalid KernelId");
}

#undef GPFLASH_KERNEL_DISPATCH_CASE

// Id -> the one name accepted for that kernel.
inline const char* to_kernel_name(const KernelId kernel_id) { return kernel_by_id(kernel_id).kernel_name(); }

// Name -> id, the inverse of to_kernel_name. Exactly one name is accepted per kernel; any other throws.
inline KernelId parse_kernel_id(const std::string& kernel_name) {
    for (int i_kernel = 0; i_kernel < N_KERNEL_ID; ++i_kernel) {
        const KernelId kernel_id = static_cast<KernelId>(i_kernel);
        if (kernel_name == to_kernel_name(kernel_id)) return kernel_id;
    }

    throw gpflash::ErrorInvalidInput("gpflash::kernel: unknown kernel '" + kernel_name + "'");
}

}  // namespace kernel
}  // namespace gpflash
