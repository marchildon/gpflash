/**
 * CacheInfo.hpp -- what a cache file says about itself, readable without loading the arrays.
 */
#pragma once

#include <armadillo>
#include <cstddef>  // std::size_t
#include <cstdio>   // std::fputs
#include <ctime>    // std::tm, std::strftime
#include <string>

#include "gpflash/core/Errors.hpp"                         // the field checks check_cache_info composes
#include "gpflash/core/Platform.hpp"                       // GPFLASH_HAVE_GMTIME_S
#include "gpflash/core/StringFormat.hpp"                   // to_formatted_string
#include "gpflash/inputs/SymmetrySpec.hpp"                 // SymmetrySpec, which to_symmetry_kind reads
#include "gpflash/posterior_cache/PosteriorModelInfo.hpp"  // PosteriorModelInfo, its checker and its summary

namespace gpflash {
namespace cache {

// The symmetry a file records, as an integer header slot rather than as the parity sign.
// End is one past the last, for check_enum_val.
enum class SymmetryKind { None, Even, Odd, End };

// A reader overwrites every field, so the initializers below show only on a default-constructed instance.
struct CacheInfo {
    // Which model produced the file. All of its fields are header slots, so it is stored whole rather than
    // copied field by field.
    PosteriorModelInfo post_model_info;

    // What the file is, and what it can serve.
    int format_version = 0;
    // False for a mean-only cache file.
    bool has_variance_data = false;

    // The model's size, so the cost of a refit is readable without loading the arrays.
    int n_train = 0;
    int n_dim = 0;      // a query point's dimension
    int n_feature = 0;  // rows the kernel evaluates on, which a feature map raises above n_dim

    // The writing GpFlash release, for diagnostics only: never compared on load.
    int version_major = 0;
    int version_minor = 0;
    int version_patch = 0;
    // Seconds since the Unix epoch. In the file because a filesystem timestamp does not survive a copy.
    arma::u64 saved_unix_time_s = 0;

    // The fitted signal variance, which the load path reads from here rather than recomputing.
    double hp_sig2_k = 0.0;

    // The mean model's constant column; its degrees are stored with the arrays.
    bool has_mean_model_constant = true;

    SymmetryKind symmetry_kind = SymmetryKind::None;  // the mirror itself is stored with the arrays

    std::string note;  // whatever the writer passed to save_precompute; empty when none
};

inline void check_cache_info(const CacheInfo& cache_info, const char* caller) {
    check_posterior_model_info(cache_info.post_model_info, caller);

    gpflash::detail::check_positive_count(cache_info.format_version, "CacheInfo::format_version", caller);
    gpflash::detail::check_positive_count(cache_info.n_train, "CacheInfo::n_train", caller);
    gpflash::detail::check_positive_count(cache_info.n_dim, "CacheInfo::n_dim", caller);
    gpflash::detail::check_positive_count(cache_info.n_feature, "CacheInfo::n_feature", caller);

    // Recorded for diagnostics, so any released triple is admissible.
    gpflash::detail::check_non_negative_count(cache_info.version_major, "CacheInfo::version_major", caller);
    gpflash::detail::check_non_negative_count(cache_info.version_minor, "CacheInfo::version_minor", caller);
    gpflash::detail::check_non_negative_count(cache_info.version_patch, "CacheInfo::version_patch", caller);

    gpflash::detail::check_finite_positive_val(
            cache_info.hp_sig2_k, "CacheInfo::hp_sig2_k", caller, "A negative one means it was never set.");
    gpflash::detail::check_enum_val(cache_info.symmetry_kind, "SymmetryKind", "CacheInfo::symmetry_kind", caller);
}

// The kind a file records for symmetry_spec.
inline SymmetryKind to_symmetry_kind(const SymmetrySpec& symmetry_spec) {
    if (!symmetry_spec.is_on()) {
        return SymmetryKind::None;
    } else if (symmetry_spec.is_even_symmetry) {
        return SymmetryKind::Even;
    } else {
        return SymmetryKind::Odd;
    }
}

namespace detail {
// "YYYY-MM-DD HH:MM:SS UTC", or "unknown" for 0, the unset value, and for a time gmtime cannot convert.
inline std::string to_utc_time_string(const arma::u64 unix_time_s) {
    if (unix_time_s == 0) return "unknown";

    const std::time_t time_s = static_cast<std::time_t>(unix_time_s);
    std::tm utc_tm{};
#if GPFLASH_HAVE_GMTIME_S
    const bool is_converted = ::gmtime_s(&utc_tm, &time_s) == 0;
#else
    const bool is_converted = ::gmtime_r(&time_s, &utc_tm) != nullptr;
#endif
    if (!is_converted) return "unknown";

    char utc_time_str[32];  // the format's 23 characters, with room for a longer year
    const std::size_t n_char = std::strftime(utc_time_str, sizeof(utc_time_str), "%Y-%m-%d %H:%M:%S UTC", &utc_tm);
    if (n_char == 0) return "unknown";

    return utc_time_str;
}

// The summary's word for a stored kind.
inline const char* to_symmetry_name(const SymmetryKind symmetry_kind) {
    switch (symmetry_kind) {
        case SymmetryKind::None:
            return "none";
        case SymmetryKind::Even:
            return "even";
        case SymmetryKind::Odd:
            return "odd";
        case SymmetryKind::End:
            break;
    }
    // Only End, or a value cast in from outside the enum, which check_cache_info rejects first.
    throw ErrorInvalidInput("to_symmetry_name: unknown SymmetryKind.");
}
}  // namespace detail

// A human-readable overview of a cache file's scalars.
inline std::string to_summary_string(const CacheInfo& cache_info, const std::string& label = "") {
    const std::string utc_time_str = detail::to_utc_time_string(cache_info.saved_unix_time_s);
    const std::string body_str = gpflash::detail::to_formatted_string(
            "  training points     %d\n"
            "  input dimension     %d\n"
            "  feature dimension   %d\n"
            "  variance data       %s\n"
            "  mean model constant %s\n"
            "  symmetry            %s\n"
            "  signal variance     %.6g\n"
            "  written by GpFlash  %d.%d.%d\n"
            "  written at          %s\n"
            "  note                %s\n",
            cache_info.n_train,
            cache_info.n_dim,
            cache_info.n_feature,
            cache_info.has_variance_data ? "yes (mean, variance, both gradients)" : "no (mean and its gradient only)",
            cache_info.has_mean_model_constant ? "yes" : "no",
            detail::to_symmetry_name(cache_info.symmetry_kind),
            cache_info.hp_sig2_k,
            cache_info.version_major,
            cache_info.version_minor,
            cache_info.version_patch,
            utc_time_str.c_str(),
            cache_info.note.empty() ? "(none)" : cache_info.note.c_str());

    // post_model_info prints its own block, so a field added there needs no change here.
    const std::string summary_str = "GpFlash cache" + (label.empty() ? std::string() : ": " + label) + "\n"
            + to_summary_string(cache_info.post_model_info) + body_str;

    return summary_str;
}

inline void print_cache_info(const CacheInfo& cache_info, const std::string& label = "") {
    std::fputs(to_summary_string(cache_info, label).c_str(), stdout);
}

}  // namespace cache
}  // namespace gpflash
