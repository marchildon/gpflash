/**
 * KernelId.hpp -- the kernel list, and the runtime identifier generated from it.
 *
 * Registering a kernel means adding it to GPFLASH_KERNEL_LIST; KernelId and the id-to-instance dispatch in
 * KernelDispatch.hpp are both generated from it.
 *
 * Append, never insert: the posterior cache stores a kernel as its KernelId integer, so inserting renames
 * every existing cache file's kernel.
 */
#pragma once

#define GPFLASH_KERNEL_LIST(GPFLASH_KERNEL) \
    GPFLASH_KERNEL(SqExp)                   \
    GPFLASH_KERNEL(Matern52)                \
    GPFLASH_KERNEL(Matern32)                \
    GPFLASH_KERNEL(RatQuad)                 \
    GPFLASH_KERNEL(Periodic)                \
    GPFLASH_KERNEL(Linear)                  \
    GPFLASH_KERNEL(Quadratic)

namespace gpflash {
namespace kernel {

// End is one past the last kernel, so N_KERNEL_ID needs no hand count.
#define GPFLASH_KERNEL_ID_ENTRY(kernel_type_name) kernel_type_name,
enum class KernelId { GPFLASH_KERNEL_LIST(GPFLASH_KERNEL_ID_ENTRY) End };
#undef GPFLASH_KERNEL_ID_ENTRY

// How many kernels there are, for a caller walking the ids as a range.
constexpr int N_KERNEL_ID = static_cast<int>(KernelId::End);

}  // namespace kernel
}  // namespace gpflash
