/**
 * Hdf5Store.hpp -- HDF5 storage: named arrays, read and written through Armadillo's HDF5 support. Specific to
 * HDF5, and the only file in the cache that names a storage format.
 *
 * Armadillo does the I/O, and prints nothing on failure, so a bad file surfaces as an exception rather than an
 * HDF5 error dump. Added on top: exceptions naming the array and file, an existence probe, a read straight into
 * caller memory, an atomic replace, and strings as byte arrays.
 *
 * Not safe to call concurrently: HDF5 is linked without --enable-threadsafe, so its global state makes two
 * simultaneous calls unsafe even to different files.
 */
#pragma once

// Armadillo includes <hdf5.h> itself, after selecting the HDF5 API version it was written against; an earlier
// include of <hdf5.h> would fix the version before it could. has_array and load_doubles_into call HDF5 through it.
#include <armadillo>
#if !defined(ARMA_USE_HDF5)
#error "The posterior cache requires ARMA_USE_HDF5 (linking GpFlash::GpFlash defines it)"
#endif

#include <cstdio>   // std::rename, std::remove
#include <fstream>  // has_file
#include <string>

#include "gpflash/core/Errors.hpp"    // ErrorInvalidInput, which cache_error returns
#include "gpflash/core/Platform.hpp"  // GPFLASH_RENAME_REPLACES_EXISTING

namespace gpflash {
namespace cache {

// The default extension for a cache file. Here because it names this store's format.
constexpr const char* store_file_extension = ".h5";

// Whether anything at all is at path. It does not ask whether the file is a cache: telling an absent cache from
// an unusable one is the reuse decision, and reporting mere existence is what leaves that decision to a caller.
inline bool has_file(const std::string& path) { return std::ifstream(path.c_str(), std::ios::binary).good(); }

namespace detail {

// Every failure in the cache, named for the function that raised it. One type for all of them, so a caller can
// tell a rejected file from an unrelated exception such as a failed allocation.
inline ErrorInvalidInput cache_error(const char* caller, const std::string& what_failed) {
    return ErrorInvalidInput("gpflash::cache::" + std::string(caller) + ": " + what_failed);
}

// A missing or unreadable file, reported as that rather than as a failed read of whichever array came first.
inline void check_file_readable(const std::string& path, const char* caller) {
    if (!has_file(path)) throw cache_error(caller, "cannot open " + path);
}

// Replace dest_path with src_path; false on failure. Atomic on POSIX, so a reader never sees a partial file.
// Where rename cannot replace, the destination is unlinked first and that atomicity is lost.
inline bool replace_file(const std::string& src_path, const std::string& dest_path) {
#if !GPFLASH_RENAME_REPLACES_EXISTING
    std::remove(dest_path.c_str());
#endif
    return std::rename(src_path.c_str(), dest_path.c_str()) == 0;
}

// ── Named arrays ─────────────────────────────────────────────────────────────

// Appended, so a name added later disturbs none of the others. Armadillo creates the file if it does not exist.
template <class ArrayT>
inline void save_array(const std::string& path, const char* array_name, const ArrayT& array, const char* caller) {
    if (!array.save(arma::hdf5_name(path, array_name, arma::hdf5_opts::append), arma::hdf5_binary))
        throw cache_error(caller, std::string("cannot write ") + array_name + " to " + path);
}

template <class ArrayT>
inline void load_array(const std::string& path, const char* array_name, ArrayT& array_out, const char* caller) {
    if (!array_out.load(arma::hdf5_name(path, array_name), arma::hdf5_binary))
        throw cache_error(caller, std::string("cannot read ") + array_name + " from " + path);
}

// Armadillo offers no existence query, so this asks HDF5 directly: one file open whatever the array's size,
// and false for a file that will not open at all, which is what a caller vetting a path wants. The guard is
// Armadillo's own -- without it HDF5 prints its error stack, and a foreign file is an answer here, not a fault.
inline bool has_array(const std::string& path, const char* array_name) {
    const arma::hdf5_misc::hdf5_suspend_printing_errors quiet_hdf5;
    const hid_t file_id = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) return false;
    const htri_t array_exists = H5Lexists(file_id, array_name, H5P_DEFAULT);
    H5Fclose(file_id);
    return array_exists > 0;
}

// A dataset of doubles read straight into caller memory, which Armadillo cannot do without an array of its own.
// n_expected entries in all, whatever the dataset's shape; any other count is refused before anything is written.
inline void load_doubles_into(
        const std::string& path,
        const char* array_name,
        double* dest_ptr,
        const arma::uword n_expected,
        const char* caller) {
    const arma::hdf5_misc::hdf5_suspend_printing_errors quiet_hdf5;
    // Each handle opens only if the one before it did, and every opened handle is closed before anything throws.
    const hid_t file_id = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    const hid_t dataset_id = (file_id >= 0) ? H5Dopen2(file_id, array_name, H5P_DEFAULT) : -1;
    const hid_t dataspace_id = (dataset_id >= 0) ? H5Dget_space(dataset_id) : -1;
    const hssize_t n_stored = (dataspace_id >= 0) ? H5Sget_simple_extent_npoints(dataspace_id) : -1;
    const bool is_length_expected = n_stored == static_cast<hssize_t>(n_expected);
    herr_t read_status = -1;
    if (is_length_expected)
        read_status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dest_ptr);
    if (dataspace_id >= 0) H5Sclose(dataspace_id);
    if (dataset_id >= 0) H5Dclose(dataset_id);
    if (file_id >= 0) H5Fclose(file_id);

    if (n_stored < 0) {
        throw cache_error(caller, std::string("cannot read ") + array_name + " from " + path);
    } else if (!is_length_expected) {
        throw cache_error(
                caller,
                std::string(array_name) + " in " + path + " holds " + std::to_string(n_stored) + " values, not the "
                        + std::to_string(n_expected) + " the header implies");
    } else if (read_status < 0) {
        throw cache_error(caller, std::string("cannot read ") + array_name + " from " + path);
    }
}

// A string, stored as bytes. Absent and empty are the same state: an empty string writes nothing, and a
// missing array reads back empty rather than failing.
inline void save_string(const std::string& path, const char* array_name, const std::string& text, const char* caller) {
    if (text.empty()) return;
    const arma::uchar_vec text_vec(reinterpret_cast<const unsigned char*>(text.data()), text.size());
    save_array(path, array_name, text_vec, caller);
}

inline std::string load_string(const std::string& path, const char* array_name) {
    arma::uchar_vec text_vec;
    // is_empty covers a hand-written zero-length array, whose memptr() may be null.
    if (!text_vec.load(arma::hdf5_name(path, array_name), arma::hdf5_binary) || text_vec.is_empty())
        return std::string();
    return std::string(reinterpret_cast<const char*>(text_vec.memptr()), text_vec.n_elem);
}

}  // namespace detail
}  // namespace cache
}  // namespace gpflash
