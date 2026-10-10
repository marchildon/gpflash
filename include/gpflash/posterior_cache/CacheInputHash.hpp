/**
 * CacheInputHash.hpp -- the staleness key: one 64-bit hash over everything a factorization depends on.
 *
 * Also the default cache filename, which is not built from the hash. Nothing here reads or writes a
 * file. README.md covers how the hash is checked and what it does not guard against.
 */
#pragma once

#include <armadillo>
#include <cstddef>
#include <string>

#include "gpflash/inputs/EigMaxUbMethod.hpp"      // EigMaxUbMethod
#include "gpflash/inputs/HpVal.hpp"               // HpVal
#include "gpflash/inputs/ModelOptions.hpp"        // ModelOptions
#include "gpflash/inputs/SymmetrySpec.hpp"        // SymmetrySpec
#include "gpflash/inputs/TrainingData.hpp"        // TrainingData
#include "gpflash/kernels/KernelDispatch.hpp"     // kernel::KernelId, to_kernel_name
#include "gpflash/mean/MeanModel.hpp"             // mean::MeanModel, whose resolved degrees are hashed
#include "gpflash/posterior_cache/Hdf5Store.hpp"  // store_file_extension

namespace gpflash {
namespace cache {
namespace detail {

// FNV-1a's 64-bit offset basis and prime: the algorithm's specification, not tuning parameters. Changing either
// rewrites every hash ever written.
constexpr arma::u64 fnv1a_offset_basis = 0xcbf29ce484222325ULL;
constexpr arma::u64 fnv1a_prime = 0x100000001b3ULL;

// One pass over n_bytes. Chained by passing the running value back in, so a sequence of fields hashes as one
// stream.
inline arma::u64 fnv1a_bytes(const void* data_ptr, const std::size_t n_bytes, arma::u64 hash_val = fnv1a_offset_basis) {
    const auto* byte_ptr = static_cast<const unsigned char*>(data_ptr);
    for (std::size_t i_byte = 0; i_byte < n_bytes; ++i_byte) {
        hash_val ^= byte_ptr[i_byte];
        hash_val *= fnv1a_prime;
    }

    return hash_val;
}

// The mirror and its parity, folded into one value. The parity is read only where a mirror is set, so without one
// it hashes the same whatever the field holds.
inline arma::u64 calc_symmetry_spec_hash(const SymmetrySpec& symmetry_spec) {
    const arma::vec& mirror_sign_vec = symmetry_spec.mirror_sign_vec;
    const arma::u64 n_mirror_sign = mirror_sign_vec.n_elem;
    const arma::u64 is_odd_symmetry_u64 = symmetry_spec.is_odd_symmetry() ? 1u : 0u;
    arma::u64 hash_val = fnv1a_bytes(&n_mirror_sign, sizeof(n_mirror_sign));
    hash_val = fnv1a_bytes(mirror_sign_vec.memptr(), mirror_sign_vec.n_elem * sizeof(double), hash_val);

    return fnv1a_bytes(&is_odd_symmetry_u64, sizeof(is_odd_symmetry_u64), hash_val);
}

}  // namespace detail

// The feed order is part of the hash. Adding, removing or reordering an input changes every stored hash, so a
// verified load then refuses every existing file.
inline arma::u64 calc_input_hash(
        const kernel::KernelId kernel_id,
        const arma::mat& X_ndim_ntrain_mat,
        const arma::vec& f_train_vec,
        const arma::vec& hp_gamma_vec,
        const arma::vec& hp_kern_extra_vec,
        const double sig2_f_val,
        const double cond_max,
        const double hp_sig2_k,
        const EigMaxUbMethod kmat_eig_max_ub_method,
        const bool has_mean_model_constant,
        const arma::uvec& mean_model_degree_vec,
        const EigMaxUbMethod mean_model_eig_max_ub_method,
        const SymmetrySpec& symmetry_spec) {
    // Fixed widths, so the digest cannot change with ARMA_64BIT_WORD or an enum's underlying type.
    const arma::u64 kernel_id_u64 = static_cast<arma::u64>(kernel_id);
    const arma::u64 kmat_eig_max_ub_method_u64 = static_cast<arma::u64>(kmat_eig_max_ub_method);
    const arma::u64 has_mean_model_constant_u64 = has_mean_model_constant ? 1u : 0u;
    const arma::u64_vec mean_model_degree_u64_vec = arma::conv_to<arma::u64_vec>::from(mean_model_degree_vec);
    const arma::u64 mean_model_eig_max_ub_method_u64 = static_cast<arma::u64>(mean_model_eig_max_ub_method);
    const arma::u64 n_dim = X_ndim_ntrain_mat.n_rows, n_train = X_ndim_ntrain_mat.n_cols;

    arma::u64 hash_val = detail::fnv1a_bytes(&kernel_id_u64, sizeof(kernel_id_u64));
    // Shape as well as contents, so the same values in a different layout do not collide.
    hash_val = detail::fnv1a_bytes(&n_dim, sizeof(n_dim), hash_val);
    hash_val = detail::fnv1a_bytes(&n_train, sizeof(n_train), hash_val);
    hash_val = detail::fnv1a_bytes(X_ndim_ntrain_mat.memptr(), X_ndim_ntrain_mat.n_elem * sizeof(double), hash_val);
    hash_val = detail::fnv1a_bytes(f_train_vec.memptr(), f_train_vec.n_elem * sizeof(double), hash_val);
    hash_val = detail::fnv1a_bytes(hp_gamma_vec.memptr(), hp_gamma_vec.n_elem * sizeof(double), hash_val);
    hash_val = detail::fnv1a_bytes(hp_kern_extra_vec.memptr(), hp_kern_extra_vec.n_elem * sizeof(double), hash_val);
    hash_val = detail::fnv1a_bytes(&sig2_f_val, sizeof(sig2_f_val), hash_val);
    hash_val = detail::fnv1a_bytes(&cond_max, sizeof(cond_max), hash_val);
    hash_val = detail::fnv1a_bytes(&hp_sig2_k, sizeof(hp_sig2_k), hash_val);
    hash_val = detail::fnv1a_bytes(&kmat_eig_max_ub_method_u64, sizeof(kmat_eig_max_ub_method_u64), hash_val);
    hash_val = detail::fnv1a_bytes(&has_mean_model_constant_u64, sizeof(has_mean_model_constant_u64), hash_val);
    hash_val = detail::fnv1a_bytes(
            mean_model_degree_u64_vec.memptr(), mean_model_degree_u64_vec.n_elem * sizeof(arma::u64), hash_val);
    hash_val =
            detail::fnv1a_bytes(&mean_model_eig_max_ub_method_u64, sizeof(mean_model_eig_max_ub_method_u64), hash_val);
    const arma::u64 symmetry_spec_hash = detail::calc_symmetry_spec_hash(symmetry_spec);
    hash_val = detail::fnv1a_bytes(&symmetry_spec_hash, sizeof(symmetry_spec_hash), hash_val);

    return hash_val;
}

// What a posterior is stamped with, and what a caller still holding the fit inputs recomputes to check a file.
// The only place the argument list above is spelled out: the feed order is part of the hash, so a
// second site that drifted would make every lookup miss, or collide.
inline arma::u64 calc_posterior_input_hash(
        const kernel::KernelId kernel_id,
        const mean::MeanModel& mean_model,
        const TrainingData& train_data,
        const HpVal& hp_val_struct,
        const ModelOptions& model_opts,
        const double sig2_f_val) {
    return calc_input_hash(
            kernel_id,
            train_data.X_ndim_ntrain_mat,
            train_data.f_train_vec,
            hp_val_struct.hp_gamma_vec,
            hp_val_struct.hp_kern_extra_vec,
            sig2_f_val,
            model_opts.cond_max,
            hp_val_struct.hp_sig2_k,
            model_opts.kmat_eig_max_ub_method,
            mean_model.has_constant,
            mean_model.degree_vec,
            model_opts.mean_model_eig_max_ub_method,
            model_opts.symmetry_spec);
}

// A readable default cache filename. The staleness check is the hash stored inside the file, not the name, so
// two fits that differ only in their values share this name and the second overwrites the first.
inline std::string calc_default_cache_path(
        const kernel::KernelId kernel_id, const TrainingData& train_data, const std::string& directory = "") {
    // A separator only where the caller left one off, so "out" and "out/" both work.
    const std::string prefix =
            (directory.empty() || directory.back() == '/' || directory.back() == '\\') ? directory : directory + "/";

    return prefix + "gpflash_" + kernel::to_kernel_name(kernel_id) + "_ntrain"
            + std::to_string(train_data.X_ndim_ntrain_mat.n_cols) + "_ndim"
            + std::to_string(train_data.X_ndim_ntrain_mat.n_rows) + store_file_extension;
}

}  // namespace cache
}  // namespace gpflash
