/**
 * PosteriorCache.hpp -- the cache file API: peek, save, load. These free functions are the whole of it; a
 * posterior class only marshals its members through them.
 *
 * Reading never creates, modifies, or removes a file -- every failure is a throw. Only save_precompute writes,
 * and it writes one file, removing nothing but its own scratch file.
 *
 * Not safe to call concurrently, reads included, even on different files; see Hdf5Store.hpp.
 */
#pragma once

#include <armadillo>
#include <cstdio>  // std::remove
#include <string>
#include <utility>  // std::move

#include "gpflash/kernels/KernelDispatch.hpp"               // kernel::KernelId, to_kernel_name, kernel_by_id
#include "gpflash/posterior_cache/CacheInfo.hpp"            // CacheInfo
#include "gpflash/posterior_cache/CacheLayout.hpp"          // the header packing and the named-array list
#include "gpflash/posterior_cache/Hdf5Store.hpp"            // the file and named-array primitives, cache_error
#include "gpflash/posterior_cache/PosteriorContents.hpp"    // PosteriorContents, check_posterior_contents
#include "gpflash/posterior_cache/PosteriorModelInfo.hpp"   // PosteriorModelInfo, check_posterior_model_info
#include "gpflash/posterior_cache/PosteriorPrecompute.hpp"  // PosteriorPrecompute

namespace gpflash {
namespace cache {

// What the file says about itself, without loading the arrays. Raises on an unreadable file, one another format
// version wrote, or an impossible stored value; the kernel and hash checks need what only the load path is given.
inline CacheInfo peek_info(const std::string& path) { return detail::load_cache_info(path, "peek_info"); }

// The kernel a file was written with, so it can be loaded without naming that kernel in advance.
inline kernel::KernelId peek_kernel_id(const std::string& path) { return peek_info(path).post_model_info.kernel_id; }

namespace detail {

// The gate on the file a save would replace: one that is not a GpFlash cache makes the save throw with nothing
// touched, so a mistyped path cannot destroy unrelated data.
inline void check_destination_replaceable(const std::string& dest_path, const char* caller) {
    if (has_file(dest_path) && !is_gpflash_cache_file(dest_path))
        throw cache_error(caller, "refusing to replace " + dest_path + ": it exists but is not a GpFlash cache file");
}

// file_has_variance_data sets both the stored flag and whether the packed L_Sigma is written, so a file cannot
// claim the factor without holding it. Written to a sibling temporary and renamed over the destination, so an
// interrupted save leaves the previous file intact -- a sibling because rename is atomic only within a filesystem.
inline void write_precompute_file(
        const std::string& path,
        const PosteriorModelInfo& post_model_info,
        const gpflash::detail::PosteriorPrecompute& post_precompute,
        const std::string& note,
        const bool file_has_variance_data) {
    check_destination_replaceable(path, "save_precompute");
    CacheInfo cache_info = make_cache_info(post_model_info, post_precompute, note, file_has_variance_data);
    // The check the load path makes, so a file that would not load is never written.
    check_cache_info(cache_info, "save_precompute");

    const std::string temp_path = path + ".tmp";
    std::remove(temp_path.c_str());  // a temp left by an earlier crash would otherwise be appended to

    // Headers, then arrays. Armadillo closes the file after every array, so it is flushed before the rename.
    // A throw part way through removes the partial scratch file, so a failed save leaves nothing behind.
    try {
        save_cache_info(temp_path, cache_info, "save_precompute");
        apply_to_named_arrays(post_precompute, [&temp_path](const char* array_name, const auto& data_array) {
            save_array(temp_path, array_name, data_array, "save_precompute");
        });
        if (file_has_variance_data) {
            const arma::vec L_Sigma_packed_vec = to_lower_packed_vec(post_precompute.L_Sigma_mat_lower);
            save_array(temp_path, L_Sigma_packed_name, L_Sigma_packed_vec, "save_precompute");
        }
    } catch (...) {
        std::remove(temp_path.c_str());
        throw;
    }

    if (!replace_file(temp_path, path)) {
        std::remove(temp_path.c_str());
        throw cache_error("save_precompute", "cannot replace " + path);
    }
}

}  // namespace detail

// Writes one file at path, carrying L_Sigma only when include_variance_data is set and the precompute holds the
// factor. Removes nothing but the sibling .tmp scratch file.
inline void save_precompute(
        const std::string& path,
        const PosteriorModelInfo& post_model_info,
        const gpflash::detail::PosteriorPrecompute& post_precompute,
        const std::string& note = "",
        const bool include_variance_data = true) {
    detail::write_precompute_file(
            path,
            post_model_info,
            post_precompute,
            note,
            /*file_has_variance_data=*/include_variance_data && post_precompute.has_variance_data());
}

namespace detail {

// load_precompute after its header read, for a caller that already holds the validated header.
inline PosteriorContents load_precompute_after_header(
        const std::string& path,
        const CacheInfo& cache_info,
        const kernel::KernelId expected_kernel_id,
        const arma::u64* expected_input_hash_ptr,
        const char* caller) {
    // The refusals, all off the header and before any array is read.
    if (cache_info.post_model_info.kernel_id != expected_kernel_id)
        throw detail::cache_error(
                caller,
                std::string("kernel mismatch (file is ") + kernel::to_kernel_name(cache_info.post_model_info.kernel_id)
                        + ", expected " + kernel::to_kernel_name(expected_kernel_id) + ")");
    if (expected_input_hash_ptr != nullptr && cache_info.post_model_info.input_hash != *expected_input_hash_ptr)
        throw detail::cache_error(caller, path + " was built from different inputs; refit and save it again");

    gpflash::detail::PosteriorPrecompute post_precompute;
    post_precompute.n_dim = cache_info.n_dim;
    post_precompute.hp_sig2_k = cache_info.hp_sig2_k;
    post_precompute.mean_model.has_constant = cache_info.has_mean_model_constant;
    // Read only where the mirror is set, which the arrays below decide.
    post_precompute.symmetry_spec.is_even_symmetry = cache_info.symmetry_kind != SymmetryKind::Odd;

    // The arrays, by the same list the writer walked.
    detail::apply_to_named_arrays(post_precompute, [&path, caller](const char* array_name, auto& data_array) {
        detail::load_array(path, array_name, data_array, caller);
    });

    // Ydot_train defines the shape that every other array, and the file's own stored counts, are checked against.
    post_precompute.n_train = static_cast<int>(post_precompute.Ydot_nfeature_ntrain_mat.n_cols);
    post_precompute.n_feature = static_cast<int>(post_precompute.Ydot_nfeature_ntrain_mat.n_rows);
    detail::check_loaded_array_shapes(
            post_precompute, cache_info.n_train, cache_info.n_feature, cache_info.symmetry_kind, caller);

    // The factor, at the n_train just checked, so a packed length the file disagrees with is caught before unpacking.
    if (cache_info.has_variance_data) {
        const arma::uword n_train_uword = post_precompute.n_train;
        // Read into the matrix's own memory and unpacked there, so the load never holds a second copy of the factor.
        arma::mat L_Sigma_mat_lower(n_train_uword, n_train_uword, arma::fill::none);
        detail::load_doubles_into(
                path,
                detail::L_Sigma_packed_name,
                L_Sigma_mat_lower.memptr(),
                detail::calc_n_lower_packed(n_train_uword),
                caller);
        detail::unpack_lower_inplace(L_Sigma_mat_lower);
        post_precompute.L_Sigma_mat_lower = std::move(L_Sigma_mat_lower);
    }

    // Derived rather than stored, and rebuilt because a reloaded posterior serves gradients too.
    gpflash::detail::fill_ydot_transpose(post_precompute);

    return PosteriorContents{std::move(post_precompute), cache_info.post_model_info};
}

}  // namespace detail

// Read a file back, refusing a foreign format version, a kernel other than expected_kernel_id, and arrays that
// do not agree on one shape. Passing expected_input_hash_ptr also refuses a file built from other inputs; pass
// it whenever the caller still holds them. Null skips only that check, for a reload where they are gone.
inline PosteriorContents load_precompute(
        const std::string& path,
        const kernel::KernelId expected_kernel_id,
        const arma::u64* expected_input_hash_ptr = nullptr) {
    // The same validated header read peek_info uses, so neither can accept a file the other rejects. A NaN
    // mll or a negative hp_sig2_k throws here, in release as well as debug.
    const CacheInfo cache_info = detail::load_cache_info(path, "load_precompute");

    return detail::load_precompute_after_header(
            path, cache_info, expected_kernel_id, expected_input_hash_ptr, "load_precompute");
}

// load_precompute validates what the format can check; it cannot check the arrays against the kernel's declared
// widths, not knowing the kernel. This does both, in that order, so a rejected file never yields a half-built
// posterior. One form, not two: the kernel reports the id to read the file at.
inline PosteriorContents load_precompute_for_kernel(
        const kernel::KernelBase& kern_fun,
        const std::string& path,
        const char* caller,
        const arma::u64* expected_input_hash_ptr = nullptr) {
    const CacheInfo cache_info = detail::load_cache_info(path, caller);
    PosteriorContents post_contents = detail::load_precompute_after_header(
            path, cache_info, kern_fun.kernel_id(), expected_input_hash_ptr, caller);
    check_posterior_contents(kern_fun, post_contents, caller);

    return post_contents;
}

namespace detail {

// load_precompute_for_kernel at the kernel the file names, its header read once.
inline PosteriorContents load_precompute_for_file_kernel(const std::string& path, const char* caller) {
    const CacheInfo cache_info = load_cache_info(path, caller);
    const kernel::KernelBase& kern_fun = kernel::kernel_by_id(cache_info.post_model_info.kernel_id);
    PosteriorContents post_contents = load_precompute_after_header(
            path, cache_info, kern_fun.kernel_id(), /*expected_input_hash_ptr=*/nullptr, caller);
    check_posterior_contents(kern_fun, post_contents, caller);

    return post_contents;
}

}  // namespace detail

}  // namespace cache
}  // namespace gpflash
