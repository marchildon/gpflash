/**
 * MultistartReport.hpp -- the per-optimization report for a multistart fit, as text.
 *
 * Four tables: the fit overall, a row per independent optimization, the point each one reached, and the point
 * each one started from; then a glossary of their terms. The tables read off MllOptzResult, so no caller
 * assembles anything.
 *
 * The hyperparameter tables take their columns from HpOptzLayout's blocks.
 *
 * Aligned plain text rather than Markdown, since it is read in a terminal and diffed between runs.
 */
#pragma once

#include <armadillo>  // arma::exp, arma::median, the natural-space cells
#include <cstdio>     // std::fputs
#include <fstream>    // std::ofstream, for the report file
#include <string>     // the report itself
#include <vector>     // one row of cells at a time

#include "gpflash/core/Errors.hpp"                              // ErrorInvalidInput, for a path that will not open
#include "gpflash/core/StringFormat.hpp"                        // to_formatted_string
#include "gpflash/optimization/MllOptzResult.hpp"               // MllOptzResult
#include "gpflash/optimization/multistart/OptimumClusters.hpp"  // calc_optimum_label_vec, calc_n_optimum
#include "gpflash/optimization/search_space/HpOptzLayout.hpp"   // HpOptzLayout, to_hp_entry_name, the column headings

namespace gpflash {

// Above this many length scales the hyperparameter tables print min/median/max of hp_gamma instead of every
// component, so a row stays readable in a terminal.
constexpr int REPORT_MAX_N_GAMMA_COLUMN = 5;

namespace detail {

// ── Layout ───────────────────────────────────────────────────────────────────
// A column is as wide as the wider of its heading and its values. Where the heading is the wider, the width is
// taken from the heading, so renaming it moves the column with it.

// Every table row's left margin.
constexpr char REPORT_ROW_INDENT[] = "    ";

// The optimization index every table leads with, up to 999.
constexpr char REPORT_IDX_HEADING[] = "Idx";
constexpr int REPORT_IDX_WIDTH = 3;

// One hyperparameter cell, sized for "hp_gamma_median". A longer heading, such as "hp_kern_extra[10]", overruns
// its cell without losing anything.
constexpr int REPORT_HP_CELL_WIDTH = 15;

// The per-optimization table.
constexpr char REPORT_MLL_HEADING[] = "MLL";
constexpr int REPORT_MLL_WIDTH = 13;  // a signed %.6g value with a three-digit exponent
constexpr char REPORT_OPTIMUM_HEADING[] = "Optimum";
constexpr int REPORT_OPTIMUM_WIDTH = sizeof(REPORT_OPTIMUM_HEADING) - 1;
constexpr char REPORT_STOP_REASON_HEADING[] = "Stopped on";
constexpr int REPORT_STOP_REASON_WIDTH = 12;  // room for the longest stop reason, "eval_budget"
constexpr char REPORT_N_EVAL_HEADING[] = "N eval";
constexpr int REPORT_N_EVAL_WIDTH = sizeof(REPORT_N_EVAL_HEADING) - 1;
constexpr char REPORT_N_AT_BOUND_HEADING[] = "N at bound";
constexpr int REPORT_N_AT_BOUND_WIDTH = sizeof(REPORT_N_AT_BOUND_HEADING) - 1;

// The initial-point table's last column.
constexpr char REPORT_LN_DIST_HEADING[] = "Ln dist init to final";
constexpr int REPORT_LN_DIST_WIDTH = sizeof(REPORT_LN_DIST_HEADING) - 1;

// One aligned row of values under a matching header, so the two cannot drift apart. Unterminated, so a caller can
// append a further column.
inline std::string to_report_row_string(const std::string& label_str, const std::vector<double>& value_vec) {
    std::string row_str = label_str;
    for (const double cell_value : value_vec)
        row_str += to_formatted_string(" %*.5g", REPORT_HP_CELL_WIDTH, cell_value);
    return row_str;
}

// The row label every table below shares.
inline std::string to_report_idx_string(const int i_ind_optz) {
    return to_formatted_string("%s%*d", REPORT_ROW_INDENT, REPORT_IDX_WIDTH, i_ind_optz);
}

// Above REPORT_MAX_N_GAMMA_COLUMN length scales, hp_gamma prints as min/median/max. One predicate for the row and
// its header, so they cannot disagree.
inline bool is_hp_gamma_collapsed(const HpOptzLayout& hp_optz_layout) {
    return hp_optz_layout.width(HpBlock::Gamma) > REPORT_MAX_N_GAMMA_COLUMN;
}

// One row in natural space: hp_gamma (or its min/median/max), then the other blocks.
inline std::vector<double> to_hp_cell_vec(const arma::vec& ln_hp_vec, const HpOptzLayout& hp_optz_layout) {
    const int n_dim = hp_optz_layout.width(HpBlock::Gamma);
    const arma::vec hp_vec = arma::exp(ln_hp_vec);
    std::vector<double> cell_vec;
    if (is_hp_gamma_collapsed(hp_optz_layout)) {
        const arma::vec hp_gamma_vec = hp_vec.head(n_dim);
        cell_vec = {hp_gamma_vec.min(), arma::median(hp_gamma_vec), hp_gamma_vec.max()};
    } else {
        for (int i_dim = 0; i_dim < n_dim; ++i_dim) cell_vec.push_back(hp_vec(i_dim));
    }
    for (arma::uword i_hp_optz = static_cast<arma::uword>(n_dim); i_hp_optz < hp_vec.n_elem; ++i_hp_optz)
        cell_vec.push_back(hp_vec(i_hp_optz));

    return cell_vec;
}

// The header matching to_hp_cell_vec. Per-entry columns carry to_hp_entry_name, the name an error message uses, so
// a column and a message about the same hyperparameter agree.
inline std::string to_hp_header_string(const HpOptzLayout& hp_optz_layout) {
    std::string header_str = to_formatted_string("%s%*s", REPORT_ROW_INDENT, REPORT_IDX_WIDTH, REPORT_IDX_HEADING);
    const auto append_heading = [&header_str](const std::string& heading_str) {
        header_str += to_formatted_string(" %*s", REPORT_HP_CELL_WIDTH, heading_str.c_str());
    };

    int i_hp_optz_first = 0;
    if (is_hp_gamma_collapsed(hp_optz_layout)) {
        for (const char* stat_name : {"min", "median", "max"})
            append_heading(std::string(to_hp_block_name(HpBlock::Gamma)) + "_" + stat_name);
        i_hp_optz_first = hp_optz_layout.width(HpBlock::Gamma);
    }
    for (int i_hp_optz = i_hp_optz_first; i_hp_optz < hp_optz_layout.n_hp_optz(); ++i_hp_optz)
        append_heading(to_hp_entry_name(hp_optz_layout, i_hp_optz));

    return header_str;
}

// The fit overall.
inline std::string to_fit_head_string(
        const MllOptzResult& mll_optz_result, const int n_optimum, const std::string& label) {
    const std::string head_str = to_formatted_string(
            "Independent optimizations    %d\n"
            "Failed optimizations         %d\n"
            "Distinct optima found        %d\n"
            "\nBest optimization idx        %d\n"
            "Best marginal ln likelihood  %.8g\n"
            "Best optimality              %.6g\n"
            "\nMLL evaluations, value only  %d\n"
            "MLL evaluations, with grad   %d\n"
            "Total MLL evaluations        %d\n",
            mll_optz_result.n_ind_optz_run(),
            mll_optz_result.calc_n_failed_start(),
            n_optimum,
            mll_optz_result.idx_best(),
            mll_optz_result.mll(),
            mll_optz_result.optimality,
            mll_optz_result.n_total_mll_eval(),
            mll_optz_result.n_total_mll_grad_eval(),
            mll_optz_result.n_total_mll_eval() + mll_optz_result.n_total_mll_grad_eval());

    return "GpFlash multistart fit" + (label.empty() ? std::string() : ": " + label) + "\n\n" + head_str;
}

// One row per optimization in launch order, the winner marked. A failed one has no optimum.
inline std::string to_per_optz_table_string(const MllOptzResult& mll_optz_result, const arma::uvec& optimum_label_vec) {
    std::string table_str = "\n  Per optimization\n";
    table_str += to_formatted_string(
            "%s%*s %*s  %*s  %*s  %*s  %*s\n",
            REPORT_ROW_INDENT,
            REPORT_IDX_WIDTH,
            REPORT_IDX_HEADING,
            REPORT_MLL_WIDTH,
            REPORT_MLL_HEADING,
            REPORT_OPTIMUM_WIDTH,
            REPORT_OPTIMUM_HEADING,
            REPORT_STOP_REASON_WIDTH,
            REPORT_STOP_REASON_HEADING,
            REPORT_N_EVAL_WIDTH,
            REPORT_N_EVAL_HEADING,
            REPORT_N_AT_BOUND_WIDTH,
            REPORT_N_AT_BOUND_HEADING);

    const int idx_best = mll_optz_result.idx_best();
    for (int i_ind_optz = 0; i_ind_optz < mll_optz_result.n_ind_optz_run(); ++i_ind_optz) {
        const LocalOptzResult local_optz_result = mll_optz_result.multistart_result.build_local_optz_result(i_ind_optz);
        const arma::uword optimum_label = optimum_label_vec(static_cast<arma::uword>(i_ind_optz));
        std::string optimum_str = "n/a";
        if (optimum_label != OPTIMUM_LABEL_NONE) optimum_str = std::to_string(optimum_label);

        table_str += to_formatted_string(
                "%s%*d %*.6g  %*s  %*s  %*d  %*d%s\n",
                REPORT_ROW_INDENT,
                REPORT_IDX_WIDTH,
                i_ind_optz,
                REPORT_MLL_WIDTH,
                -local_optz_result.objective_val,  // the objective is the negative MLL
                REPORT_OPTIMUM_WIDTH,
                optimum_str.c_str(),
                REPORT_STOP_REASON_WIDTH,
                to_stop_reason_name(local_optz_result.stop_reason),
                REPORT_N_EVAL_WIDTH,
                local_optz_result.n_eval,
                REPORT_N_AT_BOUND_WIDTH,
                mll_optz_result.calc_n_at_bound(i_ind_optz),
                i_ind_optz == idx_best ? "  <- best" : "");
    }

    return table_str;
}

// The point each optimization reached.
inline std::string to_final_hp_table_string(const MllOptzResult& mll_optz_result, const std::string& hp_header_str) {
    const HpOptzLayout& hp_optz_layout = mll_optz_result.hp_optz_layout();
    std::string table_str = "\n  Final hyperparameters, natural space\n" + hp_header_str + "\n";
    for (int i_ind_optz = 0; i_ind_optz < mll_optz_result.n_ind_optz_run(); ++i_ind_optz)
        table_str += to_report_row_string(
                             to_report_idx_string(i_ind_optz),
                             to_hp_cell_vec(mll_optz_result.multistart_result.ln_hp_vec(i_ind_optz), hp_optz_layout))
                + "\n";

    return table_str;
}

// The point each optimization started from, in the same columns as the table above so the two read against each
// other, plus how far it travelled.
inline std::string to_init_hp_table_string(const MllOptzResult& mll_optz_result, const std::string& hp_header_str) {
    const HpOptzLayout& hp_optz_layout = mll_optz_result.hp_optz_layout();
    std::string table_str = "\n  Initial hyperparameters, natural space  (optimization 0 is the heuristic start)\n";
    table_str += hp_header_str + " " + REPORT_LN_DIST_HEADING + "\n";

    for (int i_ind_optz = 0; i_ind_optz < mll_optz_result.n_ind_optz_run(); ++i_ind_optz) {
        const arma::vec ln_hp_init_vec = mll_optz_result.multistart_result.ln_hp_init_vec(i_ind_optz);
        // Log-space distance, so it reads as a relative change.
        const double ln_dist_init_to_final =
                arma::norm(mll_optz_result.multistart_result.ln_hp_vec(i_ind_optz) - ln_hp_init_vec, 2);
        const std::string ln_dist_str = to_formatted_string(" %*.5g\n", REPORT_LN_DIST_WIDTH, ln_dist_init_to_final);
        table_str +=
                to_report_row_string(to_report_idx_string(i_ind_optz), to_hp_cell_vec(ln_hp_init_vec, hp_optz_layout))
                + ln_dist_str;
    }

    return table_str;
}

// The report's terms, for a reader who has not seen one of these before.
inline std::string to_report_terms_string() {
    return "\n  Terms\n"
           "    MLL                       Marginal log likelihood, maximized by the fit. Larger is better.\n"
           "    Stopped on                Why the solve ended: tolerance, eval_budget, roundoff, or failed.\n"
           "    Optimum                   Which distinct optimum this run reached, numbered by first appearance.\n"
           "    Distinct optima found     Runs grouped by where they landed and the MLL they reached. A lower bound.\n"
           "    Failed optimizations      Runs that raised a numerical failure. The fit stands on the rest.\n"
           "    Optimality                L2 norm of the gradient, dropping components held by a bound.\n"
           "    N at bound                Hyperparameters left pinned to a bound.\n"
           "    Ln dist init to final     Start-to-solution distance in log space, so a relative change.\n"
           "    hp_gamma_i                Inverse length scale for input i. Larger varies faster along it.\n"
           "    hp_sig2_k, hp_sig2_f_val  Kernel signal variance, and noise variance when the fit estimates it.\n";
}

}  // namespace detail

// The report: the fit, then a row per optimization, then where each landed and where each started, then the terms.
inline std::string to_multistart_report_string(const MllOptzResult& mll_optz_result, const std::string& label = "") {
    if (mll_optz_result.n_ind_optz_run() == 0)
        throw ErrorInvalidInput(
                detail::error_prefix("to_multistart_report_string", "MllOptzResult::multistart_result")
                + " records no optimization, so there is no fit to report.");

    const arma::uvec optimum_label_vec = detail::calc_optimum_label_vec(mll_optz_result.multistart_result);
    const int n_optimum = detail::calc_n_optimum(optimum_label_vec);
    const std::string hp_header_str = detail::to_hp_header_string(mll_optz_result.hp_optz_layout());

    return detail::to_fit_head_string(mll_optz_result, n_optimum, label)
            + detail::to_per_optz_table_string(mll_optz_result, optimum_label_vec)
            + detail::to_final_hp_table_string(mll_optz_result, hp_header_str)
            + detail::to_init_hp_table_string(mll_optz_result, hp_header_str) + detail::to_report_terms_string();
}

// The report on stdout. Wraps to_multistart_report_string, so a test asserts on the string instead.
inline void print_multistart_report(const MllOptzResult& mll_optz_result, const std::string& label = "") {
    std::fputs(to_multistart_report_string(mll_optz_result, label).c_str(), stdout);
}

// The same text, to a file. Overwrites, since a fit's report describes that fit and not the one before it.
//
// The report is formatted before the file is opened, so a failure while formatting leaves no handle and no partial
// file. Binary, so the file holds exactly the string's bytes; text mode writes each newline as \r\n on Windows.
// The close is explicit and checked, so a write that fails late is still seen.
inline void save_multistart_report(
        const std::string& report_path, const MllOptzResult& mll_optz_result, const std::string& label = "") {
    const std::string report_str = to_multistart_report_string(mll_optz_result, label);

    std::ofstream report_file(report_path, std::ios::out | std::ios::trunc | std::ios::binary);
    if (!report_file)
        throw ErrorInvalidInput(
                detail::error_prefix("save_multistart_report", "report_path") + " could not be opened for writing ("
                + report_path + ").");
    report_file << report_str;
    report_file.close();
    if (!report_file)
        throw ErrorInvalidInput(
                detail::error_prefix("save_multistart_report", "report_path") + " could not be written (" + report_path
                + ").");
}

}  // namespace gpflash
