/**
 * Exception types, LAPACK status checking, and input validation.
 */
#pragma once

#include <armadillo>
#include <cmath>
#include <stdexcept>
#include <string>

namespace gpflash {

namespace detail {

// The finiteness test the field checks below share.
inline bool is_finite_val(const double val) { return std::isfinite(val); }

// Whether every element of ptr[0, n_elem) is finite.
inline bool is_all_finite(const double* ptr, const std::size_t n_elem) {
    for (std::size_t i_elem = 0; i_elem < n_elem; ++i_elem)
        if (!is_finite_val(ptr[i_elem])) return false;
    return true;
}

// Smallest positive value for every quantity it later inverts or takes the log of.
constexpr double POSITIVE_MIN = 1.0e-20;

// Inclusive, because clamping to POSITIVE_MIN must produce a value this same test accepts.
inline bool is_positive_val(const double val) { return is_finite_val(val) && val >= POSITIVE_MIN; }

}  // namespace detail

// A matrix operation failed on well-formed input: the inputs are usable but this point is not, so a
// hyperparameter search can reject it and carry on.
class ErrorNumericalFailure : public std::runtime_error {
   public:
    explicit ErrorNumericalFailure(const std::string& what_arg) : std::runtime_error(what_arg) {}
};

// Arguments that are inconsistent or out of range. Not recoverable: retrying cannot help.
class ErrorInvalidInput : public std::invalid_argument {
   public:
    explicit ErrorInvalidInput(const std::string& what_arg) : std::invalid_argument(what_arg) {}
};

// Another process sharing a collective computation raised, so this one stops with it. The process that raised
// throws its own error; its peers throw this.
class ErrorPeerFailure : public std::runtime_error {
   public:
    explicit ErrorPeerFailure(const std::string& what_arg) : std::runtime_error(what_arg) {}
};

// An unimplemented path reached with valid arguments. A logic_error, so catching ErrorNumericalFailure does
// not also catch this.
class ErrorNotImplemented : public std::logic_error {
   public:
    explicit ErrorNotImplemented(const std::string& what_arg) : std::logic_error(what_arg) {}
};

namespace detail {

// Maps a LAPACK info to an exception: negative throws logic_error, positive throws ErrorNumericalFailure
// carrying failure_description.
inline void check_lapack_info(
        const arma::blas_int blas_info, const char* blas_call_site, const int n, const char* failure_description) {
    if (blas_info == 0) return;

    const std::string where = std::string(blas_call_site) + " (n=" + std::to_string(n) + ")";

    if (blas_info < 0)
        throw std::logic_error(
                "gpflash: " + where + " rejected argument " + std::to_string(-blas_info)
                + " -- a dimension was computed incorrectly. This is a GpFlash bug.");

    throw ErrorNumericalFailure(
            "gpflash: " + where + " failed (info=" + std::to_string(blas_info) + "): " + failure_description);
}

// ── Field checks ─────────────────────────────────────────────────────────────
// The building blocks every struct checker composes, one field per call. Load-bearing rather than defensive:
// ARMA_NO_DEBUG removes Armadillo's bounds checks, so a mismatched length can write past an allocation.
//
// field_name is the field as the caller spells it, so a message points at the member to fix. hint carries the
// usual cause where there is one. The checks name no model quantity of their own.

inline std::string error_prefix(const char* caller, const char* field_name) {
    return std::string("gpflash::") + caller + ": " + field_name;
}

inline std::string with_hint(const std::string& message, const char* hint) {
    return hint ? message + " " + hint : message;
}

inline void check_vec_not_empty(
        const arma::vec& vec, const char* field_name, const char* caller, const char* hint = nullptr) {
    if (vec.n_elem == 0) throw ErrorInvalidInput(with_hint(error_prefix(caller, field_name) + " is empty.", hint));
}

inline void check_mat_not_empty(
        const arma::mat& mat, const char* field_name, const char* caller, const char* hint = nullptr) {
    if (mat.n_rows == 0 || mat.n_cols == 0)
        throw ErrorInvalidInput(with_hint(
                error_prefix(caller, field_name) + " is empty (" + std::to_string(mat.n_rows) + " x "
                        + std::to_string(mat.n_cols) + ").",
                hint));
}

inline void check_n_elem(
        const arma::uword n_elem,
        const arma::uword n_elem_expected,
        const char* field_name,
        const char* caller,
        const char* hint) {
    if (n_elem != n_elem_expected)
        throw ErrorInvalidInput(with_hint(
                error_prefix(caller, field_name) + " has " + std::to_string(n_elem) + " elements, expected "
                        + std::to_string(n_elem_expected) + ".",
                hint));
}

inline void check_vec_len(
        const arma::vec& vec,
        const arma::uword n_elem_expected,
        const char* field_name,
        const char* caller,
        const char* hint = nullptr) {
    check_n_elem(vec.n_elem, n_elem_expected, field_name, caller, hint);
}

// Same check for an index or flag vector. An overload rather than a second name, so callers need not know
// which element type the field happens to use.
inline void check_vec_len(
        const arma::uvec& vec,
        const arma::uword n_elem_expected,
        const char* field_name,
        const char* caller,
        const char* hint = nullptr) {
    check_n_elem(vec.n_elem, n_elem_expected, field_name, caller, hint);
}

inline void check_all_finite_vals(
        const double* ptr, const std::size_t n_elem, const char* field_name, const char* caller) {
    if (!is_all_finite(ptr, n_elem))
        throw ErrorInvalidInput(error_prefix(caller, field_name) + " contains a non-finite value (NaN or Inf).");
}

inline void check_all_finite_vals(const arma::mat& mat, const char* field_name, const char* caller) {
    check_all_finite_vals(mat.memptr(), mat.n_elem, field_name, caller);
}

// Finite and >= POSITIVE_MIN, entry by entry. An empty vector passes, which is what a kernel with no extras
// hands in; min() on an empty one is undefined under ARMA_NO_DEBUG.
inline void check_all_positive_vals(
        const arma::vec& vec, const char* field_name, const char* caller, const char* hint = nullptr) {
    if (vec.is_empty()) return;
    check_all_finite_vals(vec.memptr(), vec.n_elem, field_name, caller);
    if (vec.min() < POSITIVE_MIN)
        throw ErrorInvalidInput(
                with_hint(error_prefix(caller, field_name) + " has an entry below POSITIVE_MIN.", hint));
}

// The same where zero is valid, as a posterior variance is at a training point.
inline void check_all_non_negative_vals(
        const arma::vec& vec, const char* field_name, const char* caller, const char* hint = nullptr) {
    if (vec.is_empty()) return;
    check_all_finite_vals(vec.memptr(), vec.n_elem, field_name, caller);
    if (vec.min() < 0.0)
        throw ErrorInvalidInput(with_hint(error_prefix(caller, field_name) + " has a negative entry.", hint));
}

// One column per item, where another quantity fixes how many there are.
inline void check_mat_n_cols(
        const arma::mat& mat,
        const arma::uword n_cols_expected,
        const char* field_name,
        const char* caller,
        const char* hint = nullptr) {
    if (mat.n_cols != n_cols_expected)
        throw ErrorInvalidInput(with_hint(
                error_prefix(caller, field_name) + " has " + std::to_string(mat.n_cols) + " columns, expected "
                        + std::to_string(n_cols_expected) + ".",
                hint));
}

// One row per component, where another quantity fixes how many there are.
inline void check_mat_n_rows(
        const arma::mat& mat,
        const arma::uword n_rows_expected,
        const char* field_name,
        const char* caller,
        const char* hint = nullptr) {
    if (mat.n_rows != n_rows_expected)
        throw ErrorInvalidInput(with_hint(
                error_prefix(caller, field_name) + " has " + std::to_string(mat.n_rows) + " rows, expected "
                        + std::to_string(n_rows_expected) + ".",
                hint));
}

inline void check_mat_is_square(const arma::mat& mat, const char* field_name, const char* caller) {
    if (mat.n_rows != mat.n_cols)
        throw ErrorInvalidInput(
                error_prefix(caller, field_name) + " is " + std::to_string(mat.n_rows) + " by "
                + std::to_string(mat.n_cols) + ", expected square.");
}

inline void check_finite_val(const double val, const char* field_name, const char* caller) {
    if (!is_finite_val(val))
        throw ErrorInvalidInput(error_prefix(caller, field_name) + " is not finite (got " + std::to_string(val) + ").");
}

inline void check_finite_positive_val(
        const double val, const char* field_name, const char* caller, const char* hint = nullptr) {
    if (!is_positive_val(val))
        throw ErrorInvalidInput(with_hint(
                error_prefix(caller, field_name) + " must be finite and >= POSITIVE_MIN, got " + std::to_string(val)
                        + ".",
                hint));
}

inline void check_finite_non_negative_val(const double val, const char* field_name, const char* caller) {
    if (!is_finite_val(val) || val < 0.0)
        throw ErrorInvalidInput(
                error_prefix(caller, field_name) + " must be finite and >= 0 (got " + std::to_string(val) + ").");
}

// For a non-owning member that the holder is useless without.
inline void check_ptr_not_null(const void* ptr, const char* field_name, const char* caller) {
    if (ptr == nullptr) throw ErrorInvalidInput(error_prefix(caller, field_name) + " is null.");
}

// For a position into a vector of n_idx entries, where -1 marks one that is absent.
inline void check_idx_or_inactive(
        const int idx, const int n_idx, const char* field_name, const char* caller, const char* hint = nullptr) {
    if (idx < -1 || idx >= n_idx)
        throw ErrorInvalidInput(with_hint(
                error_prefix(caller, field_name) + " is " + std::to_string(idx) + ", outside -1 (inactive) and "
                        + std::to_string(n_idx - 1) + ".",
                hint));
}

// For an enum class whose last enumerator is End: rejects End and any value cast in from outside the enum.
template <class EnumType>
inline void check_enum_val(const EnumType val, const char* enum_name, const char* field_name, const char* caller) {
    const int enum_int_val = static_cast<int>(val);
    if (enum_int_val < 0 || enum_int_val >= static_cast<int>(EnumType::End))
        throw ErrorInvalidInput(
                error_prefix(caller, field_name) + " is not a " + enum_name + " value (got "
                + std::to_string(enum_int_val) + ").");
}

// For a count that must be present: a dimension, a point total, a block width.
inline void check_positive_count(
        const int count, const char* field_name, const char* caller, const char* hint = nullptr) {
    if (count <= 0)
        throw ErrorInvalidInput(with_hint(
                error_prefix(caller, field_name) + " must be > 0 (got " + std::to_string(count) + ").", hint));
}

// For a tally that may legitimately be zero: evaluations spent, independent optimizations run.
inline void check_non_negative_count(const int count, const char* field_name, const char* caller) {
    if (count < 0)
        throw ErrorInvalidInput(
                error_prefix(caller, field_name) + " must be >= 0 (got " + std::to_string(count) + ").");
}

// For a field where an infinity is a meaningful value, as an objective is for a failed start. NaN is still ruled
// out: it would silently win a minimum comparison.
inline void check_not_nan_val(const double val, const char* field_name, const char* caller) {
    if (std::isnan(val)) throw ErrorInvalidInput(error_prefix(caller, field_name) + " is NaN.");
}

// The same, for a magnitude: a norm stays non-negative whether or not it is finite.
inline void check_non_negative_or_inf_val(const double val, const char* field_name, const char* caller) {
    check_not_nan_val(val, field_name, caller);
    if (val < 0.0)
        throw ErrorInvalidInput(error_prefix(caller, field_name) + " must be >= 0 (got " + std::to_string(val) + ").");
}

inline void check_finite_above_val(const double val, const double bound, const char* field_name, const char* caller) {
    if (!is_finite_val(val) || !(val > bound))
        throw ErrorInvalidInput(
                error_prefix(caller, field_name) + " must be finite and > " + std::to_string(bound) + " (got "
                + std::to_string(val) + ").");
}

// For a value that may sit on either end, as a value clamped into an interval does.
inline void check_finite_in_closed_range_val(
        const double val,
        const double lower_bound,
        const double upper_bound,
        const char* field_name,
        const char* caller,
        const char* hint = nullptr) {
    if (!is_finite_val(val) || val < lower_bound || val > upper_bound)
        throw ErrorInvalidInput(with_hint(
                error_prefix(caller, field_name) + " must be finite and within " + std::to_string(lower_bound) + " and "
                        + std::to_string(upper_bound) + " (got " + std::to_string(val) + ").",
                hint));
}

// For a value with a ceiling as well as a floor, half-open as lb < val <= ub; the closed twin above allows either
// end. Both state the span in one message, so a caller sees both ends.
inline void check_finite_in_range_val(
        const double val,
        const double lower_bound,
        const double upper_bound,
        const char* field_name,
        const char* caller,
        const char* hint = nullptr) {
    if (!is_finite_val(val) || !(val > lower_bound) || val > upper_bound)
        throw ErrorInvalidInput(with_hint(
                error_prefix(caller, field_name) + " must be finite, > " + std::to_string(lower_bound)
                        + " and <= " + std::to_string(upper_bound) + " (got " + std::to_string(val) + ").",
                hint));
}

}  // namespace detail
}  // namespace gpflash
