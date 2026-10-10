// The multistart report: the optimum clustering, the column layout, and the string/stdout/file contract.
//
// Built on hand-made MllOptzResult values rather than real fits: the report is formatting over a struct, so
// supplying one directly keeps every case deterministic and independent of optimizer numerics.
#include <armadillo>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

#include "GpFlashTestSupport.hpp"
#include "gpflash/optimization/MultistartReport.hpp"
#include "gpflash/optimization/multistart/OptimumClusters.hpp"  // SAME_OPTIMUM_LN_DIST
#include "gpflash/optimization/multistart/RunMultistart.hpp"    // fill_ind_optz_column_inplace, IndOptzOutcome
#include "gpflash/optimization/search_space/HpOptzLayout.hpp"   // build_hp_optz_layout, for the layout a report reads

using gpflash::LocalOptzResult;
using gpflash::MllOptzResult;
using gpflash_test::kDetailLen;
using gpflash_test::rejects;
using gpflash_test::report;

namespace {

// A result whose only meaningful content is the layout and the points reached, which is all the report reads.
//
// The layout comes from the width of the points supplied: n_dim gives the noise-free layout, n_dim + 1 the
// fixed-noise one, n_dim + 2 the estimated-noise one, so one builder drives all three.
//
// The constraint corners are wide, so at-bound counts are zero unless a case puts a point on a bound.
MllOptzResult build_report_input(
        const std::vector<arma::vec>& ln_hp_vec_vec,
        const int n_dim,
        const int n_hp_kern_extra = 0,
        // One objective per solve. Empty means all equal, so the grouping below is decided by position alone; a case
        // that wants position and objective to disagree supplies them.
        const std::vector<double>& objective_val_vec = {}) {
    MllOptzResult mll_optz_result;
    mll_optz_result.hp_val_struct.hp_gamma_vec = arma::vec(static_cast<arma::uword>(n_dim), arma::fill::ones);
    mll_optz_result.hp_val_struct.hp_kern_extra_vec =
            arma::vec(static_cast<arma::uword>(n_hp_kern_extra), arma::fill::ones);

    const int n_hp_optz =
            ln_hp_vec_vec.empty() ? n_dim + n_hp_kern_extra : static_cast<int>(ln_hp_vec_vec.front().n_elem);
    const int n_variance = n_hp_optz - n_dim - n_hp_kern_extra;
    const gpflash::NoiseRequest noise_request = (n_variance >= 2)
            ? gpflash::NoiseRequest::Estimate
            : ((n_variance == 1) ? gpflash::NoiseRequest::Fixed : gpflash::NoiseRequest::NoiseFree);

    gpflash::MultistartResult& multistart_result = mll_optz_result.multistart_result;
    multistart_result.hp_optz_layout =
            gpflash::detail::build_hp_optz_layout(n_dim, n_hp_kern_extra, noise_request, "build_report_input");
    const int n_ind_optz = static_cast<int>(ln_hp_vec_vec.size());
    multistart_result.ln_hp_mat.zeros(n_hp_optz, n_ind_optz);
    multistart_result.ln_hp_init_mat.zeros(n_hp_optz, n_ind_optz);
    multistart_result.start_field_mat.zeros(static_cast<int>(gpflash::StartField::End), n_ind_optz);
    for (int i_ind_optz = 0; i_ind_optz < n_ind_optz; ++i_ind_optz) {
        const std::size_t i_start = static_cast<std::size_t>(i_ind_optz);
        gpflash::detail::IndOptzOutcome ind_optz_outcome;
        ind_optz_outcome.ln_hp_init_vec = ln_hp_vec_vec[i_start] + 1.0;  // a start one unit away
        ind_optz_outcome.local_optz_result.ln_hp_vec = ln_hp_vec_vec[i_start];
        ind_optz_outcome.local_optz_result.objective_val =
                objective_val_vec.empty() ? -1.0 : objective_val_vec[i_start];
        ind_optz_outcome.local_optz_result.n_eval = 10 + i_ind_optz;
        ind_optz_outcome.local_optz_result.stop_reason = gpflash::OptzStopReason::ToleranceMet;
        ind_optz_outcome.n_mll_eval = 7;
        ind_optz_outcome.n_mll_grad_eval = 11;
        gpflash::detail::fill_ind_optz_column_inplace(multistart_result, i_ind_optz, ind_optz_outcome);
    }

    mll_optz_result.optimality = 1e-6;
    gpflash::OptzRegion& optz_region = mll_optz_result.optz_region;
    optz_region.constraint_lb_vec = arma::vec(static_cast<arma::uword>(n_hp_optz), arma::fill::ones) * -1.0e3;
    optz_region.constraint_ub_vec = arma::vec(static_cast<arma::uword>(n_hp_optz), arma::fill::ones) * 1.0e3;
    optz_region.sample_lb_vec = optz_region.constraint_lb_vec;
    optz_region.sample_ub_vec = optz_region.constraint_ub_vec;
    optz_region.start_vec = arma::vec(static_cast<arma::uword>(n_hp_optz), arma::fill::zeros);
    return mll_optz_result;
}

// Where each whitespace-separated field ends. Right-aligned columns share these positions with their header, so
// comparing them catches a heading whose width does not match its cells.
std::vector<std::size_t> to_field_end_vec(const std::string& line_str) {
    std::vector<std::size_t> end_vec;
    for (std::size_t i_char = 0; i_char < line_str.size(); ++i_char) {
        const bool is_field = line_str[i_char] != ' ';
        const bool is_last = i_char + 1 == line_str.size() || line_str[i_char + 1] == ' ';
        if (is_field && is_last) end_vec.push_back(i_char);
    }
    return end_vec;
}

// The whitespace-separated fields of one line, so a cell can be asserted by its column position.
std::vector<std::string> to_field_vec(const std::string& line_str) {
    std::vector<std::string> field_vec;
    std::istringstream line_stream(line_str);
    std::string field_str;
    while (line_stream >> field_str) field_vec.push_back(field_str);
    return field_vec;
}

std::vector<std::string> to_line_vec(const std::string& text_str) {
    std::vector<std::string> line_vec;
    std::size_t i_begin = 0;
    while (i_begin <= text_str.size()) {
        const std::size_t i_end = text_str.find('\n', i_begin);
        if (i_end == std::string::npos) break;
        line_vec.push_back(text_str.substr(i_begin, i_end - i_begin));
        i_begin = i_end + 1;
    }
    return line_vec;
}

// The rows under a section title: its header line, then every row until the blank line.
std::vector<std::string> to_section_row_vec(const std::string& report_str, const std::string& title_str) {
    const std::vector<std::string> line_vec = to_line_vec(report_str);
    std::vector<std::string> row_vec;
    bool in_section = false;
    for (const std::string& line_str : line_vec) {
        if (line_str.find(title_str) != std::string::npos) {
            in_section = true;
            continue;
        }
        if (!in_section) continue;
        if (line_str.empty()) break;
        row_vec.push_back(line_str);
    }
    return row_vec;
}

arma::vec to_ln_hp_vec(const double value, const int n_hp_optz) {
    return arma::vec(static_cast<arma::uword>(n_hp_optz), arma::fill::ones) * value;
}

// ── 1. The optimum clustering ────────────────────────────────────────────────

bool test_optimum_label_vec(const char* name) {
    static constexpr int n_hp_optz = 3;
    // Identical points are one optimum.
    const arma::uvec label_same_vec = gpflash::detail::calc_optimum_label_vec(
            build_report_input({to_ln_hp_vec(1.0, n_hp_optz), to_ln_hp_vec(1.0, n_hp_optz)}, n_hp_optz)
                    .multistart_result);
    const bool same_ok = label_same_vec.max() == 0;

    // Well-separated points are separate optima, labelled by first appearance.
    const MllOptzResult far_input = build_report_input(
            {to_ln_hp_vec(0.0, n_hp_optz), to_ln_hp_vec(5.0, n_hp_optz), to_ln_hp_vec(0.0, n_hp_optz)}, n_hp_optz);
    const arma::uvec label_far_vec = gpflash::detail::calc_optimum_label_vec(far_input.multistart_result);
    const bool far_ok = label_far_vec(0) == 0 && label_far_vec(1) == 1 && label_far_vec(2) == 0;

    // Single linkage: A and C are further apart than the tolerance, but each is within it of B, so all three are
    // one optimum.
    const double step = 0.6 * gpflash::SAME_OPTIMUM_LN_DIST / std::sqrt(static_cast<double>(n_hp_optz));
    const arma::uvec label_chain_vec = gpflash::detail::calc_optimum_label_vec(
            build_report_input(
                    {to_ln_hp_vec(0.0, n_hp_optz), to_ln_hp_vec(step, n_hp_optz), to_ln_hp_vec(2.0 * step, n_hp_optz)},
                    n_hp_optz)
                    .multistart_result);
    const bool chain_ok = label_chain_vec.max() == 0;

    // The tolerance boundary is where it is documented: half the tolerance joins, twice the tolerance does not.
    const double inside = 0.5 * gpflash::SAME_OPTIMUM_LN_DIST / std::sqrt(static_cast<double>(n_hp_optz));
    const bool inside_ok =
            gpflash::detail::calc_optimum_label_vec(
                    build_report_input({to_ln_hp_vec(0.0, n_hp_optz), to_ln_hp_vec(inside, n_hp_optz)}, n_hp_optz)
                            .multistart_result)
                    .max()
            == 0;
    const double outside = 2.0 * gpflash::SAME_OPTIMUM_LN_DIST / std::sqrt(static_cast<double>(n_hp_optz));
    const bool outside_ok =
            gpflash::detail::calc_optimum_label_vec(
                    build_report_input({to_ln_hp_vec(0.0, n_hp_optz), to_ln_hp_vec(outside, n_hp_optz)}, n_hp_optz)
                            .multistart_result)
                    .max()
            == 1;

    // Order independence: the same points in a different order give the same grouping.
    const arma::uvec label_shuffled_vec = gpflash::detail::calc_optimum_label_vec(
            build_report_input(
                    {to_ln_hp_vec(5.0, n_hp_optz), to_ln_hp_vec(0.0, n_hp_optz), to_ln_hp_vec(5.0, n_hp_optz)},
                    n_hp_optz)
                    .multistart_result);
    const bool order_ok = label_shuffled_vec(0) == label_shuffled_vec(2)
            && label_shuffled_vec(1) != label_shuffled_vec(0) && label_shuffled_vec.max() == 1;

    // Position is not sufficient: two solves at the same point whose objectives disagree are not the same optimum,
    // because the objective is flat at an optimum and cannot differ there. Without this the count would merge them.
    const arma::uvec label_objective_vec = gpflash::detail::calc_optimum_label_vec(
            build_report_input(
                    {to_ln_hp_vec(1.0, n_hp_optz), to_ln_hp_vec(1.0, n_hp_optz)},
                    n_hp_optz,
                    /*n_hp_kern_extra=*/0,
                    /*objective_val_vec=*/{-1.0, -2.0})
                    .multistart_result);
    const bool objective_ok = label_objective_vec.max() == 1;

    // And an objective difference far below the tolerance does not split a genuine pair.
    const arma::uvec label_objective_near_vec = gpflash::detail::calc_optimum_label_vec(
            build_report_input(
                    {to_ln_hp_vec(1.0, n_hp_optz), to_ln_hp_vec(1.0, n_hp_optz)},
                    n_hp_optz,
                    /*n_hp_kern_extra=*/0,
                    /*objective_val_vec=*/{-1.0, -1.0 - 1e-12})
                    .multistart_result);
    const bool objective_near_ok = label_objective_near_vec.max() == 0;

    // The objective rule away from magnitude 1, where its relative scale and its absolute floor stop coinciding.
    const auto calc_n_label = [](const double objective_val_a, const double objective_val_b) {
        return gpflash::detail::calc_optimum_label_vec(
                       build_report_input(
                               {to_ln_hp_vec(1.0, n_hp_optz), to_ln_hp_vec(1.0, n_hp_optz)},
                               n_hp_optz,
                               /*n_hp_kern_extra=*/0,
                               /*objective_val_vec=*/{objective_val_a, objective_val_b})
                               .multistart_result)
                       .max()
                + 1;
    };
    const bool objective_scale_ok = calc_n_label(-1.0e4, -1.0e4 * (1.0 + 1.0e-8)) == 1  // relative: joins
            && calc_n_label(-1.0e4, -1.0e4 * (1.0 + 1.0e-5)) == 2                       // relative: splits
            && calc_n_label(-1.0e-3, -1.0e-3 - 1.0e-7) == 1;                            // absolute floor below 1: joins

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "same=%d far=%d chain=%d inside=%d outside=%d order=%d obj=%d obj_near=%d obj_scale=%d",
            static_cast<int>(same_ok),
            static_cast<int>(far_ok),
            static_cast<int>(chain_ok),
            static_cast<int>(inside_ok),
            static_cast<int>(outside_ok),
            static_cast<int>(order_ok),
            static_cast<int>(objective_ok),
            static_cast<int>(objective_near_ok),
            static_cast<int>(objective_scale_ok));
    return report(
            name,
            same_ok && far_ok && chain_ok && inside_ok && outside_ok && order_ok && objective_ok && objective_near_ok
                    && objective_scale_ok,
            detail);
}

// ── 2. Column layout ────────────────────────────────────────────────────────

bool test_column_alignment(const char* name) {
    const int n_dim = 3;
    const MllOptzResult mll_optz_result =
            build_report_input({to_ln_hp_vec(0.1, n_dim + 1), to_ln_hp_vec(0.2, n_dim + 1)}, n_dim);
    const std::string report_str = gpflash::to_multistart_report_string(mll_optz_result);

    // Only the final-hyperparameter table can be checked this way: its headings are one word each, so a field end
    // belongs to exactly one column. The other two have multi-word headings. It also carries the variance
    // headings, which is where a heading sized differently from its cells would show.
    bool aligned = true;
    int n_row_checked = 0;
    const std::vector<std::string> row_vec = to_section_row_vec(report_str, "Final hyperparameters");
    if (row_vec.size() < 2) {
        aligned = false;
    } else {
        const std::vector<std::size_t> header_end_vec = to_field_end_vec(row_vec.front());
        for (std::size_t i_row = 1; i_row < row_vec.size(); ++i_row) {
            const std::vector<std::size_t> row_end_vec = to_field_end_vec(row_vec[i_row]);
            if (row_end_vec != header_end_vec) aligned = false;
            ++n_row_checked;
        }
    }

    char detail[kDetailLen];
    std::snprintf(detail, sizeof(detail), "rows_checked=%d aligned=%d", n_row_checked, static_cast<int>(aligned));
    return report(name, aligned && n_row_checked == 2, detail);
}

// ── 3. Which variance columns appear ────────────────────────────────────────

bool test_variance_columns(const char* name) {
    const int n_dim = 2;
    // The layout puts hp_sig2_k first and hp_sig2_f_val second, so the count of trailing parameters decides which
    // are present. All three widths a NoiseRequest can produce are pinned here.
    const std::string noise_free_str =
            gpflash::to_multistart_report_string(build_report_input({to_ln_hp_vec(0.5, n_dim)}, n_dim));
    const std::string known_var_str =
            gpflash::to_multistart_report_string(build_report_input({to_ln_hp_vec(0.5, n_dim + 1)}, n_dim));
    const std::string estimated_str =
            gpflash::to_multistart_report_string(build_report_input({to_ln_hp_vec(0.5, n_dim + 2)}, n_dim));

    const std::string noise_free_head = to_section_row_vec(noise_free_str, "Final hyperparameters").front();
    const std::string known_var_head = to_section_row_vec(known_var_str, "Final hyperparameters").front();
    const std::string estimated_head = to_section_row_vec(estimated_str, "Final hyperparameters").front();

    const bool noise_free_ok = noise_free_head.find("hp_sig2_k") == std::string::npos
            && noise_free_head.find("hp_sig2_f_val") == std::string::npos;
    const bool known_var_ok = known_var_head.find("hp_sig2_k") != std::string::npos
            && known_var_head.find("hp_sig2_f_val") == std::string::npos;
    const bool estimated_ok = estimated_head.find("hp_sig2_k") != std::string::npos
            && estimated_head.find("hp_sig2_f_val") != std::string::npos;

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "noise_free=%d known_var=%d estimated=%d",
            static_cast<int>(noise_free_ok),
            static_cast<int>(known_var_ok),
            static_cast<int>(estimated_ok));
    return report(name, noise_free_ok && known_var_ok && estimated_ok, detail);
}

// ── 4. The hp_gamma collapse ────────────────────────────────────────────────

bool test_gamma_collapse(const char* name) {
    // At the threshold every component prints; one past it, the spread replaces them.
    const int n_at = gpflash::REPORT_MAX_N_GAMMA_COLUMN;
    const std::string at_str =
            gpflash::to_multistart_report_string(build_report_input({to_ln_hp_vec(0.0, n_at)}, n_at));
    const bool at_ok = at_str.find("hp_gamma[" + std::to_string(n_at - 1) + "]") != std::string::npos
            && at_str.find("hp_gamma_min") == std::string::npos;

    // Known components: ln values -2, then zeros, then 2, so the median is 1 between exp(-2) and exp(2).
    const int n_over = gpflash::REPORT_MAX_N_GAMMA_COLUMN + 1;
    arma::vec ln_hp_vec(static_cast<arma::uword>(n_over), arma::fill::zeros);
    ln_hp_vec(0) = -2.0;
    ln_hp_vec(n_over - 1) = 2.0;
    const std::string over_str = gpflash::to_multistart_report_string(build_report_input({ln_hp_vec}, n_over));
    const std::vector<std::string> over_row_vec = to_section_row_vec(over_str, "Final hyperparameters");
    // Idx, then exp(-2), exp(0) and exp(+2) under min, median and max.
    const bool over_cells_ok = over_row_vec.size() == 2
            && to_field_vec(over_row_vec[1]) == std::vector<std::string>{"0", "0.13534", "1", "7.3891"};
    const bool over_ok = over_str.find("hp_gamma_min") != std::string::npos
            && over_str.find("hp_gamma_median") != std::string::npos
            && over_str.find("hp_gamma_max") != std::string::npos && over_str.find("hp_gamma[0]") == std::string::npos
            && over_cells_ok;

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "at_threshold=%d over_threshold=%d",
            static_cast<int>(at_ok),
            static_cast<int>(over_ok));
    return report(name, at_ok && over_ok, detail);
}

// ── 5. MllOptzResult's own short summary ────────────────────────────────────
// The other text form a fit can produce. Asserted here rather than in test_gpflash_optimizer.cpp because it is
// formatting over a struct, which is this file's remit.
bool test_summary_string(const char* name) {
    const int n_dim = 2;
    const MllOptzResult mll_optz_result = build_report_input({to_ln_hp_vec(0.3, n_dim + 1)}, n_dim);
    const std::string summary_str = gpflash::to_summary_string(mll_optz_result, "label text");

    // The label it was given.
    const bool label_ok = summary_str.find("label text") != std::string::npos;

    // The HpAtBound block is appended whole, so a block added there appears here without this file changing.
    const bool bound_block_ok = summary_str.find("hp_gamma       at bound") != std::string::npos
            && summary_str.find(gpflash::to_summary_string(mll_optz_result.calc_hp_at_bound_struct()))
                    != std::string::npos;

    // Every line the summary promises, each value pinned to its own label, so a rotation among the adjacent counts
    // shows. The fixture's one optimization has n_eval 10, 7 MLL evaluations, 11 with the gradient, and optimality
    // 1e-6.
    const std::string mll_line =
            gpflash::detail::to_formatted_string("  marginal log lik   %.6g\n", mll_optz_result.mll());
    const std::string stop_reason_line = std::string("  stopped on         ")
            + gpflash::to_stop_reason_name(gpflash::OptzStopReason::ToleranceMet) + "\n";
    const bool values_ok = summary_str.find(mll_line) != std::string::npos
            && summary_str.find("  optimality         1e-06\n") != std::string::npos
            && summary_str.find(stop_reason_line) != std::string::npos
            && summary_str.find("  ind optz run       1\n") != std::string::npos
            && summary_str.find("  failed starts      0\n") != std::string::npos
            && summary_str.find("  evals (best)       10\n") != std::string::npos
            && summary_str.find("  MLL evals          7\n") != std::string::npos
            && summary_str.find("  MLL+grad evals     11\n") != std::string::npos;

    // No optimizations means no winner, so the accessors it reads must throw rather than index on -1.
    const bool empty_rejected = rejects([&] { gpflash::to_summary_string(MllOptzResult{}); });

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "label=%d bound_block=%d values=%d empty=%d",
            static_cast<int>(label_ok),
            static_cast<int>(bound_block_ok),
            static_cast<int>(values_ok),
            static_cast<int>(empty_rejected));
    return report(name, label_ok && bound_block_ok && values_ok && empty_rejected, detail);
}

// ── 6. The string, stdout and file contract ─────────────────────────────────

bool test_output_contract(const char* name) {
    const int n_dim = 2;
    const MllOptzResult mll_optz_result = build_report_input({to_ln_hp_vec(0.3, n_dim + 1)}, n_dim);
    const std::string report_str = gpflash::to_multistart_report_string(mll_optz_result, "label text");

    // The label reaches the report, and the terms block is present for a reader who needs it.
    const bool label_ok = report_str.find("label text") != std::string::npos;
    const bool terms_ok = report_str.find("Terms") != std::string::npos;

    // Optimization i has n_eval = 10 + i and i entries on the lower bound, so the two columns never agree and a swap
    // would show.
    const double ln_hp_bound = -1.0;
    std::vector<arma::vec> ln_hp_at_bound_vec_vec;
    for (int i_ind_optz = 0; i_ind_optz < 3; ++i_ind_optz) {
        arma::vec ln_hp_vec(static_cast<arma::uword>(n_dim), arma::fill::zeros);
        ln_hp_vec.head(i_ind_optz).fill(ln_hp_bound);
        ln_hp_at_bound_vec_vec.push_back(ln_hp_vec);
    }
    MllOptzResult mll_optz_result_rows = build_report_input(
            ln_hp_at_bound_vec_vec, n_dim, /*n_hp_kern_extra=*/0, /*objective_val_vec=*/{-1.0, -0.5, -0.25});
    const std::vector<std::string> mll_cell_vec = {"1", "0.5", "0.25"};  // the negated objectives
    mll_optz_result_rows.optz_region.constraint_lb_vec.fill(ln_hp_bound);
    mll_optz_result_rows.optz_region.constraint_ub_vec.fill(1.0);
    const std::vector<std::string> per_optz_row_vec =
            to_section_row_vec(gpflash::to_multistart_report_string(mll_optz_result_rows), "Per optimization");
    const bool has_per_optz_rows = per_optz_row_vec.size() == 4;  // the header plus one row per optimization
    int idx_first_bad_optz = -1;
    for (std::size_t i_row = 1; has_per_optz_rows && i_row < per_optz_row_vec.size(); ++i_row) {
        // Idx, MLL, Optimum, Stopped on, N eval, N at bound, then the winner's two-word "<- best" marker. The MLLs
        // are 1, 0.5 and 0.25, so optimization 0 wins; the three points are distinct, so each is its own optimum
        // in launch order.
        const std::vector<std::string> field_vec = to_field_vec(per_optz_row_vec[i_row]);
        const int i_ind_optz = static_cast<int>(i_row) - 1;
        const bool is_winner = i_ind_optz == 0;
        const bool is_row_ok = field_vec.size() >= 6 && field_vec[0] == std::to_string(i_ind_optz)
                && field_vec[1] == mll_cell_vec[static_cast<std::size_t>(i_ind_optz)]
                && field_vec[2] == std::to_string(i_ind_optz)
                && field_vec[3] == gpflash::to_stop_reason_name(gpflash::OptzStopReason::ToleranceMet)
                && field_vec[4] == std::to_string(10 + i_ind_optz) && field_vec[5] == std::to_string(i_ind_optz)
                && (field_vec.size() == 8) == is_winner;
        if (!is_row_ok && idx_first_bad_optz < 0) idx_first_bad_optz = i_ind_optz;
    }
    const bool columns_ok = has_per_optz_rows && idx_first_bad_optz < 0;

    // The start is one unit above the final 0.3 in each of the 3 entries: exp(1.3) per cell, and sqrt(3) travelled.
    const std::vector<std::string> init_row_vec = to_section_row_vec(report_str, "Initial hyperparameters");
    const bool init_ok = init_row_vec.size() == 2
            && to_field_vec(init_row_vec[1]) == std::vector<std::string>{"0", "3.6693", "3.6693", "3.6693", "1.7321"};

    // save_ writes exactly what to_*_string returns: one report, three ways out.
    const std::string report_path = "test_gpflash_multistart_report.out.txt";
    gpflash::save_multistart_report(report_path, mll_optz_result, "label text");
    std::FILE* report_file = std::fopen(report_path.c_str(), "rb");
    std::string file_str;
    if (report_file != nullptr) {
        char buf[4096];
        std::size_t n_read = 0;
        while ((n_read = std::fread(buf, 1, sizeof(buf), report_file)) > 0) file_str.append(buf, n_read);
        std::fclose(report_file);
        std::remove(report_path.c_str());
    }
    const bool file_ok = file_str == report_str;

    // A directory that cannot be opened for writing is an input error, not a crash.
    const bool bad_path_rejected = rejects([&] { gpflash::save_multistart_report("/", mll_optz_result); });

    // No optimizations is an input error too: there is no fit to describe.
    const bool empty_rejected = rejects([&] { gpflash::to_multistart_report_string(MllOptzResult{}); });

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "label=%d terms=%d columns=%d first_bad_optz=%d init=%d file=%d bad_path=%d empty=%d",
            static_cast<int>(label_ok),
            static_cast<int>(terms_ok),
            static_cast<int>(columns_ok),
            idx_first_bad_optz,
            static_cast<int>(init_ok),
            static_cast<int>(file_ok),
            static_cast<int>(bad_path_rejected),
            static_cast<int>(empty_rejected));
    return report(
            name,
            label_ok && terms_ok && columns_ok && init_ok && file_ok && bad_path_rejected && empty_rejected,
            detail);
}

// ── 7. A failed start in the report ─────────────────────────────────────────

bool test_failed_start_rows(const char* name) {
    const int n_dim = 2;
    // Three starts; the middle one failed. Its ln_hp_vec is its start point and its objective +inf.
    MllOptzResult mll_optz_result =
            build_report_input({to_ln_hp_vec(0.1, n_dim), to_ln_hp_vec(0.5, n_dim), to_ln_hp_vec(0.1, n_dim)}, n_dim);
    gpflash::detail::IndOptzOutcome ind_optz_outcome_failed;
    ind_optz_outcome_failed.ln_hp_init_vec = to_ln_hp_vec(0.5, n_dim);
    ind_optz_outcome_failed.local_optz_result.ln_hp_vec = to_ln_hp_vec(0.5, n_dim);
    ind_optz_outcome_failed.local_optz_result.stop_reason = gpflash::OptzStopReason::Failed;
    gpflash::detail::fill_ind_optz_column_inplace(mll_optz_result.multistart_result, 1, ind_optz_outcome_failed);

    // The clustering leaves it unlabelled and does not count it as an optimum.
    const arma::uvec optimum_label_vec = gpflash::detail::calc_optimum_label_vec(mll_optz_result.multistart_result);
    const bool cluster_ok = optimum_label_vec(1) == gpflash::OPTIMUM_LABEL_NONE && optimum_label_vec(0) == 0
            && optimum_label_vec(2) == 0 && gpflash::detail::calc_n_optimum(optimum_label_vec) == 1;

    // The report counts it, and its row shows the reason and no optimum.
    const std::string report_str = gpflash::to_multistart_report_string(mll_optz_result);
    const bool head_ok = report_str.find("Failed optimizations         1") != std::string::npos
            && report_str.find("Distinct optima found        1") != std::string::npos;
    const std::vector<std::string> per_optz_row_vec = to_section_row_vec(report_str, "Per optimization");
    bool row_ok = per_optz_row_vec.size() == 4;
    if (row_ok) {
        const std::vector<std::string> field_vec = to_field_vec(per_optz_row_vec[2]);  // the failed start
        row_ok = field_vec.size() >= 6 && field_vec[2] == "n/a"
                && field_vec[3] == gpflash::to_stop_reason_name(gpflash::OptzStopReason::Failed);
        // Start 2 reached start 0's point, so it carries optimum label 0 rather than its own index.
        const std::vector<std::string> rejoined_field_vec = to_field_vec(per_optz_row_vec[3]);
        row_ok = row_ok && rejoined_field_vec.size() >= 6 && rejoined_field_vec[2] == "0";
    }

    // The winner is one of the two that ran.
    const bool winner_ok = mll_optz_result.idx_best() != 1 && mll_optz_result.calc_n_failed_start() == 1;

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "cluster=%d head=%d row=%d winner=%d",
            static_cast<int>(cluster_ok),
            static_cast<int>(head_ok),
            static_cast<int>(row_ok),
            static_cast<int>(winner_ok));
    return report(name, cluster_ok && head_ok && row_ok && winner_ok, detail);
}

// REPORT_STOP_REASON_WIDTH is a fixed width, while the names it must hold live in LocalOptzResult.hpp, so a name
// added there is checked against the column here: a longer one would shift its row against the header.
bool test_stop_reason_names_fit_column(const char* name) {
    std::string longest_name;
    for (int i_reason = 0; i_reason < static_cast<int>(gpflash::OptzStopReason::End); ++i_reason) {
        const std::string reason_str = gpflash::to_stop_reason_name(static_cast<gpflash::OptzStopReason>(i_reason));
        if (reason_str.size() > longest_name.size()) longest_name = reason_str;
    }
    const bool fits = longest_name.size() <= static_cast<std::size_t>(gpflash::detail::REPORT_STOP_REASON_WIDTH);

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "longest=%.48s (%d of %d)",
            longest_name.c_str(),
            static_cast<int>(longest_name.size()),
            gpflash::detail::REPORT_STOP_REASON_WIDTH);
    return report(name, fits, detail);
}

// ── 8. The fit head: every value pinned to its heading ──────────────────────
// to_fit_head_string formats nine values through one format string, so a rotation among its %d conversions, or
// between its two %g ones, still compiles and prints the wrong number under the right heading. The nine values
// here are distinct, so any rotation shows.
bool test_fit_head_values(const char* name) {
    const int n_dim = 2;
    // Four starts; the second and fourth failed, and the first and third reached the same point and objective.
    const double objective_best_val = -1.2345678901;
    MllOptzResult mll_optz_result = build_report_input(
            {to_ln_hp_vec(0.1, n_dim), to_ln_hp_vec(0.5, n_dim), to_ln_hp_vec(0.1, n_dim), to_ln_hp_vec(0.7, n_dim)},
            n_dim,
            /*n_hp_kern_extra=*/0,
            /*objective_val_vec=*/{objective_best_val, 0.0, objective_best_val, 0.0});
    for (const int idx_failed : {1, 3}) {
        gpflash::detail::IndOptzOutcome ind_optz_outcome_failed;  // +inf objective, no evaluations
        ind_optz_outcome_failed.ln_hp_init_vec = to_ln_hp_vec(0.5, n_dim);
        ind_optz_outcome_failed.local_optz_result.ln_hp_vec = to_ln_hp_vec(0.5, n_dim);
        ind_optz_outcome_failed.local_optz_result.stop_reason = gpflash::OptzStopReason::Failed;
        gpflash::detail::fill_ind_optz_column_inplace(
                mll_optz_result.multistart_result, idx_failed, ind_optz_outcome_failed);
    }

    // One optimum, won by start 0, the first on the tie; the MLL is the negative of its objective. The two that ran
    // spent 7 value-only and 11 gradient evaluations each.
    const std::string report_str = gpflash::to_multistart_report_string(mll_optz_result);
    const char* const expected_line_vec[] = {
            "Independent optimizations    4\n",
            "Failed optimizations         2\n",
            "Distinct optima found        1\n",
            "Best optimization idx        0\n",
            "Best marginal ln likelihood  1.2345679\n",
            "Best optimality              1e-06\n",
            "MLL evaluations, value only  14\n",
            "MLL evaluations, with grad   22\n",
            "Total MLL evaluations        36\n",
    };
    const int n_line = static_cast<int>(sizeof(expected_line_vec) / sizeof(expected_line_vec[0]));
    int idx_first_missing = -1;
    for (int idx_line = 0; idx_line < n_line && idx_first_missing < 0; ++idx_line) {
        const bool is_line_found = report_str.find(expected_line_vec[idx_line]) != std::string::npos;
        if (!is_line_found) idx_first_missing = idx_line;
    }

    char detail[kDetailLen];
    std::snprintf(detail, sizeof(detail), "n_line=%d first_missing=%d", n_line, idx_first_missing);
    return report(name, idx_first_missing < 0, detail);
}

}  // namespace

int main() {
    std::printf("== GpFlash multistart report ==\n");
    bool ok = true;
    ok &= test_optimum_label_vec("optimum_label_vec");
    ok &= test_column_alignment("column_alignment");
    ok &= test_variance_columns("variance_columns");
    ok &= test_gamma_collapse("gamma_collapse");
    ok &= test_summary_string("summary_string");
    ok &= test_output_contract("output_contract");
    ok &= test_failed_start_rows("failed_start_rows");
    ok &= test_stop_reason_names_fit_column("stop_reason_names_fit_column");
    ok &= test_fit_head_values("fit_head_values");
    std::printf("\n%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return ok ? 0 : 1;
}
