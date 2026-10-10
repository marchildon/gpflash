/**
 * CacheLayout.hpp -- what a cache file holds: the format version, the stored names, the header packing, and the
 * checks a loaded file must pass. Names no storage format, and decides nothing about what to do with a file.
 *
 * The scalars share three header arrays -- an integer vector, a real vector, the note -- since every stored
 * array costs a file open and its own metadata. Each remaining array gets its own name, which makes the data
 * self-describing.
 */
#pragma once

#include <armadillo>
#include <ctime>  // std::time, for the save stamp
#include <string>
#include <type_traits>  // std::enable_if, std::is_enum, for the integer slot overloads

#include "gpflash/Version.hpp"                              // GPFLASH_VERSION_* (macros only; header-safe)
#include "gpflash/inputs/EigMaxUbMethod.hpp"                // EigMaxUbMethod
#include "gpflash/inputs/KernelId.hpp"                      // kernel::KernelId
#include "gpflash/posterior_cache/CacheInfo.hpp"            // CacheInfo, check_cache_info
#include "gpflash/posterior_cache/Hdf5Store.hpp"            // cache_error, the array and string helpers
#include "gpflash/posterior_cache/PosteriorModelInfo.hpp"   // PosteriorModelInfo
#include "gpflash/posterior_cache/PosteriorPrecompute.hpp"  // the arrays named below

namespace gpflash {
namespace cache {

// Bump on any change to what the file stores or what a stored value means: a slot or array added, renamed or
// reoriented, or numerics that change L_Sigma at the same inputs. Checked for equality, so a mismatch either way
// is a refusal.
//
// Two things never change: the header_u64 array name and slot 0 holding the version. They are how a reader of
// any version recognizes a GpFlash cache and reports a foreign one.
constexpr int cache_format_version = 1;

// A "slot" is a scalar's fixed position in a header vector: slot_x indexes the element carrying x. The trailing
// entry counts the slots, so no length is hand-written. Every value stores as u64 or double whatever its C++
// type, so neither ARMA_64BIT_WORD nor an enum's underlying type can change the file.
enum CacheIntegerIndex : arma::uword {
    // What the file is.
    slot_format_version = 0,
    slot_kernel_id,

    // How big the model is.
    slot_n_dim,
    slot_n_train,
    slot_n_feature,

    // Where it came from.
    slot_version_major,
    slot_version_minor,
    slot_version_patch,
    slot_saved_unix_time_s,

    // What it was fitted to, and whether L_Sigma is in the file.
    slot_input_hash,
    slot_has_variance_data,
    // How the nugget was sized, beside slot_cond_max in the real header.
    slot_kmat_eig_max_ub_method,

    // The mean model, and how its nugget was sized.
    slot_has_mean_model_constant,
    slot_mean_model_eig_max_ub_method,

    // The symmetry's kind; its mirror is an array.
    slot_symmetry_kind,

    n_cache_integer_slots
};

enum CacheRealIndex : arma::uword {
    // Fitted.
    slot_hp_sig2_k = 0,
    slot_mll,

    // What the fit was configured with.
    slot_sig2_f_val,
    slot_cond_max,

    n_cache_real_slots
};

// ════════════════ implementation ═════════════════════════════════════════════
namespace detail {

// Named because save and load both reach for them, and the recognition probe for header_u64.
constexpr const char* header_u64_name = "header_u64";
constexpr const char* header_real_name = "header_real";
constexpr const char* note_name = "note";

// ── CacheInfo <-> the file ───────────────────────────────────────────────────
// Gather it from a fit, write it, read it back.

// file_has_variance_data is the caller's, since it depends on what that caller is about to write.
inline CacheInfo make_cache_info(
        const PosteriorModelInfo& post_model_info,
        const gpflash::detail::PosteriorPrecompute& post_precompute,
        const std::string& note,
        const bool file_has_variance_data) {
    CacheInfo cache_info;
    cache_info.post_model_info = post_model_info;
    cache_info.format_version = cache_format_version;
    cache_info.n_dim = post_precompute.n_dim;
    cache_info.n_train = post_precompute.n_train;
    cache_info.n_feature = post_precompute.n_feature;
    cache_info.version_major = GPFLASH_VERSION_MAJOR;
    cache_info.version_minor = GPFLASH_VERSION_MINOR;
    cache_info.version_patch = GPFLASH_VERSION_PATCH;
    // Stamped here rather than by the caller, so every file carries one.
    cache_info.saved_unix_time_s = static_cast<arma::u64>(std::time(nullptr));
    cache_info.has_variance_data = file_has_variance_data;
    cache_info.hp_sig2_k = post_precompute.hp_sig2_k;
    cache_info.has_mean_model_constant = post_precompute.mean_model.has_constant;
    cache_info.symmetry_kind = to_symmetry_kind(post_precompute.symmetry_spec);
    cache_info.note = note;

    return cache_info;
}

// Every header slot beside the CacheInfo field it carries, so the writer and the reader walk one list.
// integer_slot_operation(slot, field) takes each integer slot, whatever the field's C++ type, and
// real_slot_operation(slot, field) each real one.
template <class CacheInfoT, class IntegerSlotOperation, class RealSlotOperation>
inline void apply_to_header_slots(
        CacheInfoT& cache_info,
        IntegerSlotOperation&& integer_slot_operation,
        RealSlotOperation&& real_slot_operation) {
    integer_slot_operation(slot_format_version, cache_info.format_version);
    integer_slot_operation(slot_kernel_id, cache_info.post_model_info.kernel_id);
    integer_slot_operation(slot_n_dim, cache_info.n_dim);
    integer_slot_operation(slot_n_train, cache_info.n_train);
    integer_slot_operation(slot_n_feature, cache_info.n_feature);
    integer_slot_operation(slot_version_major, cache_info.version_major);
    integer_slot_operation(slot_version_minor, cache_info.version_minor);
    integer_slot_operation(slot_version_patch, cache_info.version_patch);
    integer_slot_operation(slot_saved_unix_time_s, cache_info.saved_unix_time_s);
    integer_slot_operation(slot_input_hash, cache_info.post_model_info.input_hash);
    integer_slot_operation(slot_has_variance_data, cache_info.has_variance_data);
    integer_slot_operation(slot_kmat_eig_max_ub_method, cache_info.post_model_info.kmat_eig_max_ub_method);
    integer_slot_operation(slot_has_mean_model_constant, cache_info.has_mean_model_constant);
    integer_slot_operation(slot_mean_model_eig_max_ub_method, cache_info.post_model_info.mean_model_eig_max_ub_method);
    integer_slot_operation(slot_symmetry_kind, cache_info.symmetry_kind);

    real_slot_operation(slot_hp_sig2_k, cache_info.hp_sig2_k);
    real_slot_operation(slot_mll, cache_info.post_model_info.mll);
    real_slot_operation(slot_sig2_f_val, cache_info.post_model_info.sig2_f_val);
    real_slot_operation(slot_cond_max, cache_info.post_model_info.cond_max);
}

// An integer slot's value back as its field's type, one overload per type a slot may carry, so a field of any other
// type fails to compile until its conversion is written here. A count or an enum was stored from an int, so it
// reads back through one; a hash or a time needs all 64 bits.
inline void fill_from_integer_slot(const arma::u64 slot_val, int& field_out) { field_out = static_cast<int>(slot_val); }
template <class EnumT, typename std::enable_if<std::is_enum<EnumT>::value, int>::type = 0>
inline void fill_from_integer_slot(const arma::u64 slot_val, EnumT& field_out) {
    field_out = static_cast<EnumT>(static_cast<int>(slot_val));
}
inline void fill_from_integer_slot(const arma::u64 slot_val, arma::u64& field_out) { field_out = slot_val; }
inline void fill_from_integer_slot(const arma::u64 slot_val, bool& field_out) { field_out = slot_val != 0; }

// True when path holds a file this format wrote, at any version -- the integer header is there under its
// never-renamed name. Version-blind because a caller about to replace a file needs that, not a version match.
inline bool is_gpflash_cache_file(const std::string& path) { return has_array(path, header_u64_name); }

// The inverse of load_cache_info, so the two stay in step. Validating the CacheInfo first is the caller's job.
inline void save_cache_info(const std::string& path, const CacheInfo& cache_info, const char* caller) {
    arma::u64_vec integer_header_vec(n_cache_integer_slots, arma::fill::zeros);
    arma::vec real_header_vec(n_cache_real_slots, arma::fill::zeros);
    apply_to_header_slots(
            cache_info,
            [&integer_header_vec](const CacheIntegerIndex slot, const auto& field) {
                integer_header_vec(slot) = static_cast<arma::u64>(field);
            },
            [&real_header_vec](const CacheRealIndex slot, const double field) { real_header_vec(slot) = field; });

    save_array(path, header_u64_name, integer_header_vec, caller);
    save_array(path, header_real_name, real_header_vec, caller);
    save_string(path, note_name, cache_info.note, caller);
}

// Every scalar the file carries, read and validated. The only place they are read, so all readers agree.
inline CacheInfo load_cache_info(const std::string& path, const char* caller) {
    check_file_readable(path, caller);

    arma::u64_vec integer_header_vec;
    load_array(path, header_u64_name, integer_header_vec, caller);
    // The version first, before either length: slot 0 has held it in every version, so a file another version
    // wrote is refused by name whatever else that version changed.
    if (integer_header_vec.is_empty()) throw cache_error(caller, "the integer header of " + path + " is empty");
    if (integer_header_vec(slot_format_version) != static_cast<arma::u64>(cache_format_version))
        throw cache_error(
                caller,
                path + " was written by cache format version " + std::to_string(integer_header_vec(slot_format_version))
                        + "; this build reads version " + std::to_string(cache_format_version));
    if (integer_header_vec.n_elem != n_cache_integer_slots)
        throw cache_error(
                caller,
                "the integer header of " + path + " holds " + std::to_string(integer_header_vec.n_elem)
                        + " slots, not the " + std::to_string(static_cast<int>(n_cache_integer_slots))
                        + " this build reads");
    arma::vec real_header_vec;
    load_array(path, header_real_name, real_header_vec, caller);
    if (real_header_vec.n_elem != n_cache_real_slots)
        throw cache_error(
                caller,
                "the real header of " + path + " holds " + std::to_string(real_header_vec.n_elem) + " slots, not the "
                        + std::to_string(static_cast<int>(n_cache_real_slots)) + " this build reads");

    CacheInfo cache_info;
    apply_to_header_slots(
            cache_info,
            [&integer_header_vec](const CacheIntegerIndex slot, auto& field) {
                fill_from_integer_slot(integer_header_vec(slot), field);
            },
            [&real_header_vec](const CacheRealIndex slot, double& field) { field = real_header_vec(slot); });
    cache_info.note = load_string(path, note_name);

    // Unwrapped: these came off disk, so this is input validation and must run in release too.
    check_cache_info(cache_info, caller);
    return cache_info;
}

// ── The array list ───────────────────────────────────────────────────────────
// The record of which arrays a file holds and what each is called. array_operation(array_name, data_array) is
// applied to every array, save and load passing their own, so neither can carry a name the other does not, and
// each stored name sits beside the member it carries. L_Sigma is stored packed, beside the list rather than in it.
template <class PrecomputeT, class ArrayOperation>
inline void apply_to_named_arrays(PrecomputeT& post_precompute, ArrayOperation&& array_operation) {
    array_operation("x_center_vec", post_precompute.x_center_vec);
    array_operation("hp_gamma_vec", post_precompute.hp_gamma_vec);
    array_operation("hp_kern_extra_vec", post_precompute.hp_kern_extra_vec);
    array_operation("hp_beta_vec", post_precompute.hp_beta_vec);
    array_operation("mean_model_degree_vec", post_precompute.mean_model.degree_vec);
    array_operation("ydot_train_sq_norm_vec", post_precompute.ydot_train_sq_norm_vec);
    array_operation("lin_sol_sigma_residual_vec", post_precompute.lin_sol_sigma_residual_vec);
    array_operation("Ydot_nfeature_ntrain_mat", post_precompute.Ydot_nfeature_ntrain_mat);
    array_operation("mirror_sign_vec", post_precompute.symmetry_spec.mirror_sign_vec);
}

// ── The factor, packed ───────────────────────────────────────────────────────
// L_Sigma's lower triangle, column by column: (n_train (n_train + 1) / 2) entries where the full matrix has
// n_train^2. No solve reads the strict upper triangle, so it is neither stored nor restored. Stored only where the
// file holds the variance data.
constexpr const char* L_Sigma_packed_name = "L_Sigma_lower_packed_vec";

inline arma::uword calc_n_lower_packed(const arma::uword n_row) { return n_row * (n_row + 1) / 2; }

inline arma::vec to_lower_packed_vec(const arma::mat& mat_lower) {
    const arma::uword n_row = mat_lower.n_rows;
    arma::vec packed_vec(calc_n_lower_packed(n_row));
    arma::uword i_packed = 0;
    for (arma::uword i_col = 0; i_col < n_row; ++i_col)
        for (arma::uword i_row = i_col; i_row < n_row; ++i_row) packed_vec(i_packed++) = mat_lower(i_row, i_col);

    return packed_vec;
}

// The inverse of to_lower_packed_vec, in place: mat_lower's first calc_n_lower_packed entries hold the packed
// triangle on entry, and each moves to its column-major position. The strict upper triangle is left unset, as the
// factorization leaves it.
//
// Backward, from the last entry: (i_row, i_col) sits i_col (i_col + 1) / 2 entries after its packed position, so a
// destination is never an entry still to be read.
inline void unpack_lower_inplace(arma::mat& mat_lower) {
    const arma::uword n_row = mat_lower.n_rows;
    double* const mat_ptr = mat_lower.memptr();
    arma::uword i_packed = calc_n_lower_packed(n_row);
    for (arma::uword i_col = n_row; i_col-- > 0;)
        for (arma::uword i_row = n_row; i_row-- > i_col;) mat_ptr[i_col * n_row + i_row] = mat_ptr[--i_packed];
}

// ── Cross-checks ─────────────────────────────────────────────────────────────

// The arrays load one at a time, so these confirm they describe one model: a file whose parts disagree would
// otherwise index out of bounds under ARMA_NO_DEBUG. One check per rule, so a failure names what is wrong.
inline void check_loaded_array_shapes(
        const gpflash::detail::PosteriorPrecompute& post_precompute,
        const int n_train_stored,
        const int n_feature_stored,
        const SymmetryKind symmetry_kind_stored,
        const char* caller) {
    const arma::uword n_train_uword = post_precompute.n_train;  // the declared type does the conversion

    // The counts read off the arrays.
    gpflash::detail::check_positive_count(post_precompute.n_train, "PosteriorPrecompute::n_train", caller);
    gpflash::detail::check_positive_count(post_precompute.n_feature, "PosteriorPrecompute::n_feature", caller);

    // The file's own copy of the shape against what the arrays actually are.
    if (post_precompute.n_train != n_train_stored)
        throw cache_error(
                caller,
                "stored n_train of " + std::to_string(n_train_stored) + " but the arrays hold "
                        + std::to_string(post_precompute.n_train));
    if (post_precompute.n_feature != n_feature_stored)
        throw cache_error(
                caller,
                "stored n_feature of " + std::to_string(n_feature_stored) + " but the arrays hold "
                        + std::to_string(post_precompute.n_feature));

    // Every vector against the count that sets its length.
    const auto check_len = [caller](const arma::vec& data_vec, const arma::uword n_expected, const char* vec_name) {
        constexpr const char* stale_hint = "The file's parts disagree, so it was probably truncated or hand-edited.";
        gpflash::detail::check_vec_len(data_vec, n_expected, vec_name, caller, stale_hint);
    };
    check_len(post_precompute.x_center_vec, post_precompute.n_dim, "PosteriorPrecompute::x_center_vec");
    check_len(post_precompute.hp_gamma_vec, post_precompute.n_dim, "PosteriorPrecompute::hp_gamma_vec");
    // The mean model degrees set the mean model weights' length, so they are validated first.
    mean::check_mean_model(post_precompute.mean_model, post_precompute.n_dim, caller);
    check_len(post_precompute.hp_beta_vec, post_precompute.mean_model.n_basis(), "PosteriorPrecompute::hp_beta_vec");
    check_len(post_precompute.ydot_train_sq_norm_vec, n_train_uword, "PosteriorPrecompute::ydot_train_sq_norm_vec");
    check_len(
            post_precompute.lin_sol_sigma_residual_vec,
            n_train_uword,
            "PosteriorPrecompute::lin_sol_sigma_residual_vec");
    // A mirror exactly where the header records a symmetry, one sign per input dimension.
    const bool has_mirror_stored = symmetry_kind_stored != SymmetryKind::None;
    const arma::uword n_mirror_sign_expected = has_mirror_stored ? post_precompute.n_dim : 0;
    check_len(
            post_precompute.symmetry_spec.mirror_sign_vec,
            n_mirror_sign_expected,
            "PosteriorPrecompute::symmetry_spec.mirror_sign_vec");
}
}  // namespace detail

}  // namespace cache
}  // namespace gpflash
