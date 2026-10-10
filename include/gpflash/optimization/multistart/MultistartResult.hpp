/**
 * MultistartResult.hpp -- every independent optimization a multistart fit ran, and which one won.
 *
 * Three arrays with one column per optimization, so a distributed fit gathers each with one sum. The winner and
 * the totals are derived, never stored.
 */
#pragma once

#include <armadillo>  // arma::mat
#include <string>     // std::to_string

#include "gpflash/core/Errors.hpp"                              // ErrorInvalidInput, the field checks
#include "gpflash/core/ParallelContext.hpp"                     // ParallelContext, for the gather
#include "gpflash/optimization/local_optz/LocalOptzResult.hpp"  // LocalOptzResult, OptzStopReason
#include "gpflash/optimization/search_space/HpOptzLayout.hpp"   // HpOptzLayout

namespace gpflash {

// Rows of start_field_mat. Append, never insert.
enum class StartField {
    ObjectiveVal,  // at the minimizer; +inf where the solve failed
    NEval,         // objective evaluations: the backend's count, or the driver's tally for a failed start
    NMllEval,      // value-only MLL evaluations
    NMllGradEval,  // MLL evaluations that also formed the gradient
    StopReason,    // an OptzStopReason, as its integer
    End,           // one past the last: the row count
};

struct MultistartResult {
    HpOptzLayout hp_optz_layout;  // how to slice the columns below into named hyperparameters

    // One column per optimization, in launch order.
    arma::mat ln_hp_mat;        // (n_hp_optz, n_ind_optz) where each solve ended
    arma::mat ln_hp_init_mat;   // (n_hp_optz, n_ind_optz) where each began
    arma::mat start_field_mat;  // (StartField::End, n_ind_optz)

    int n_ind_optz_run() const { return static_cast<int>(start_field_mat.n_cols); }

    // ── One optimization ─────────────────────────────────────────────────────

    double start_field_val(const StartField start_field, const int i_ind_optz) const {
        return start_field_mat(static_cast<arma::uword>(start_field), static_cast<arma::uword>(i_ind_optz));
    }

    // Counts are stored as doubles so one array and one MPI type serve every field.
    int start_field_count(const StartField start_field, const int i_ind_optz) const {
        return static_cast<int>(start_field_val(start_field, i_ind_optz));
    }

    OptzStopReason stop_reason(const int i_ind_optz) const {
        return static_cast<OptzStopReason>(start_field_count(StartField::StopReason, i_ind_optz));
    }

    arma::vec ln_hp_vec(const int i_ind_optz) const { return ln_hp_mat.col(static_cast<arma::uword>(i_ind_optz)); }
    arma::vec ln_hp_init_vec(const int i_ind_optz) const {
        return ln_hp_init_mat.col(static_cast<arma::uword>(i_ind_optz));
    }

    // One optimization as the backend returned it.
    LocalOptzResult build_local_optz_result(const int i_ind_optz) const {
        LocalOptzResult local_optz_result;
        local_optz_result.ln_hp_vec = ln_hp_vec(i_ind_optz);
        local_optz_result.objective_val = start_field_val(StartField::ObjectiveVal, i_ind_optz);
        local_optz_result.n_eval = start_field_count(StartField::NEval, i_ind_optz);
        local_optz_result.stop_reason = stop_reason(i_ind_optz);

        return local_optz_result;
    }

    // ── The fit over all of them ─────────────────────────────────────────────

    // Lowest objective, the first on a tie. Throws on an empty record: this is public and the postcondition is
    // compiled out unless GPFLASH_DEBUG_CHECKS is defined.
    int idx_best() const {
        detail::check_positive_count(
                n_ind_optz_run(),
                "MultistartResult::start_field_mat",
                "MultistartResult::idx_best",
                "A search reports a winner only once it has run an optimization.");

        return static_cast<int>(start_field_mat.row(static_cast<arma::uword>(StartField::ObjectiveVal)).index_min());
    }

    arma::vec best_ln_hp_vec() const { return ln_hp_vec(idx_best()); }
    LocalOptzResult build_best_local_optz_result() const { return build_local_optz_result(idx_best()); }

    // The objective is the negative MLL.
    double mll() const { return -start_field_val(StartField::ObjectiveVal, idx_best()); }

    int total_start_field_count(const StartField start_field) const {
        return static_cast<int>(arma::accu(start_field_mat.row(static_cast<arma::uword>(start_field))));
    }

    int calc_n_failed_start() const {
        int n_failed = 0;
        for (int i_ind_optz = 0; i_ind_optz < n_ind_optz_run(); ++i_ind_optz)
            if (stop_reason(i_ind_optz) == OptzStopReason::Failed) ++n_failed;

        return n_failed;
    }
};

namespace detail {

inline void check_start_mat_dims(
        const arma::mat& start_mat, const int n_row, const int n_ind_optz, const char* field_name, const char* caller) {
    if (static_cast<int>(start_mat.n_rows) != n_row || static_cast<int>(start_mat.n_cols) != n_ind_optz)
        throw ErrorInvalidInput(
                error_prefix(caller, field_name) + " is " + std::to_string(start_mat.n_rows) + " by "
                + std::to_string(start_mat.n_cols) + "; expected " + std::to_string(n_row) + " by "
                + std::to_string(n_ind_optz) + ", one column per independent optimization.");
}

// Postcondition.
inline void check_multistart_result(const MultistartResult& multistart_result, const char* caller) {
    check_hp_optz_layout(multistart_result.hp_optz_layout, caller);

    // Extents first: everything below indexes into the arrays.
    const int n_ind_optz = multistart_result.n_ind_optz_run();
    check_positive_count(n_ind_optz, "MultistartResult::start_field_mat", caller, "A fit runs at least one.");
    const int n_hp_optz = multistart_result.hp_optz_layout.n_hp_optz();
    check_start_mat_dims(
            multistart_result.start_field_mat,
            static_cast<int>(StartField::End),
            n_ind_optz,
            "MultistartResult::start_field_mat",
            caller);
    check_start_mat_dims(multistart_result.ln_hp_mat, n_hp_optz, n_ind_optz, "MultistartResult::ln_hp_mat", caller);
    check_start_mat_dims(
            multistart_result.ln_hp_init_mat, n_hp_optz, n_ind_optz, "MultistartResult::ln_hp_init_mat", caller);

    check_all_finite_vals(multistart_result.ln_hp_mat, "MultistartResult::ln_hp_mat", caller);
    check_all_finite_vals(multistart_result.ln_hp_init_mat, "MultistartResult::ln_hp_init_mat", caller);

    // Every optimization, since the report tabulates all of them.
    for (int i_ind_optz = 0; i_ind_optz < n_ind_optz; ++i_ind_optz) {
        check_local_optz_result(multistart_result.build_local_optz_result(i_ind_optz), caller);
        check_non_negative_count(
                multistart_result.start_field_count(StartField::NMllEval, i_ind_optz),
                "MultistartResult::NMllEval",
                caller);
        check_non_negative_count(
                multistart_result.start_field_count(StartField::NMllGradEval, i_ind_optz),
                "MultistartResult::NMllGradEval",
                caller);
    }

    // run_multistart raises when every start failed, so the winner has a finite objective.
    check_finite_val(
            multistart_result.start_field_val(StartField::ObjectiveVal, multistart_result.idx_best()),
            "MultistartResult::ObjectiveVal at idx_best",
            caller);
}

// ── The gather ───────────────────────────────────────────────────────────────

// Bring every process's columns to every process. Each column is written by one process and left zero on the
// others, so summing reconstructs all of them: a process the worker cap excluded contributes a zeroed record and
// receives a complete one.
inline void fill_all_ind_optz_inplace(MultistartResult& multistart_result, const ParallelContext& parallel_context) {
    parallel_context.calc_sum_inplace(multistart_result.ln_hp_mat, "MultistartResult::ln_hp_mat");
    parallel_context.calc_sum_inplace(multistart_result.ln_hp_init_mat, "MultistartResult::ln_hp_init_mat");
    parallel_context.calc_sum_inplace(multistart_result.start_field_mat, "MultistartResult::start_field_mat");
}

}  // namespace detail
}  // namespace gpflash
