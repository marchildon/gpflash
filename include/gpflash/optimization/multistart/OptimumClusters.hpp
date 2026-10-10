/**
 * OptimumClusters.hpp -- how many distinct optima a multistart fit found, by grouping the solves that agree.
 *
 * Several solves may converge to the same place, so their count says nothing on its own. Grouping them gives a lower
 * bound on how many optima the MLL has over the search region, which says whether more restarts would pay.
 */
#pragma once

#include <algorithm>  // std::max
#include <armadillo>  // arma::uvec, arma::norm
#include <cmath>      // std::fabs
#include <limits>     // std::numeric_limits, for OPTIMUM_LABEL_NONE
#include <vector>     // the frontier of one group

#include "gpflash/optimization/multistart/MultistartResult.hpp"  // MultistartResult, OptzStopReason

namespace gpflash {

// How close two solves must land to count as the same optimum. Euclidean distance in log space, so it is relative and
// one number serves every hyperparameter. Converged solves that agree land far closer than this.
constexpr double SAME_OPTIMUM_LN_DIST = 1e-3;

// And how closely their objectives must agree: relative to the larger magnitude, or absolute below 1. Position alone
// is not enough: the objective is flat at an optimum, so two solves this close to the same one agree far more tightly
// than this. A larger gap means two nearby optima, or a solve that did not converge.
constexpr double SAME_OPTIMUM_REL_OBJECTIVE = 1e-6;

// Label of a failed start. Distinct from every group label.
constexpr arma::uword OPTIMUM_LABEL_NONE = std::numeric_limits<arma::uword>::max();

namespace detail {

// Same place, and the same objective there.
inline bool is_same_optimum(const MultistartResult& multistart_result, const int i_ind_optz_a, const int i_ind_optz_b) {
    const arma::uword i_col_a = static_cast<arma::uword>(i_ind_optz_a);
    const arma::uword i_col_b = static_cast<arma::uword>(i_ind_optz_b);
    const double ln_dist =
            arma::norm(multistart_result.ln_hp_mat.col(i_col_a) - multistart_result.ln_hp_mat.col(i_col_b), 2);
    if (ln_dist > SAME_OPTIMUM_LN_DIST) return false;

    const double objective_val_a = multistart_result.start_field_val(StartField::ObjectiveVal, i_ind_optz_a);
    const double objective_val_b = multistart_result.start_field_val(StartField::ObjectiveVal, i_ind_optz_b);
    const double objective_diff = std::fabs(objective_val_a - objective_val_b);
    const double objective_scale = std::max(std::fabs(objective_val_a), std::fabs(objective_val_b));

    return objective_diff <= SAME_OPTIMUM_REL_OBJECTIVE * std::max(1.0, objective_scale);
}

// Which optimum each solve reached, labelled in order of first appearance, and OPTIMUM_LABEL_NONE for one that
// failed. The label count is a lower bound on the objective's optima.
//
// Single linkage: a chain of same-optimum neighbours joins two solves even where the ends are further apart than the
// tolerance, so the grouping does not depend on which solve seeds it.
inline arma::uvec calc_optimum_label_vec(const MultistartResult& multistart_result) {
    const int n_ind_optz = multistart_result.n_ind_optz_run();
    const arma::uword optimum_unlabelled = static_cast<arma::uword>(n_ind_optz);  // one past the largest label

    arma::uvec optimum_label_vec(static_cast<arma::uword>(n_ind_optz));
    optimum_label_vec.fill(optimum_unlabelled);
    // Failed starts neither seed a group nor join one.
    for (int i_ind_optz = 0; i_ind_optz < n_ind_optz; ++i_ind_optz)
        if (multistart_result.stop_reason(i_ind_optz) == OptzStopReason::Failed)
            optimum_label_vec(static_cast<arma::uword>(i_ind_optz)) = OPTIMUM_LABEL_NONE;

    arma::uword n_optimum = 0;
    // Each unlabelled solve seeds an optimum that grows outward: a solve pulled in has its own neighbours scanned in
    // turn, which is what makes the linkage transitive. A solve leaves pending_member_vec once scanned, which bounds
    // the work.
    std::vector<int> pending_member_vec;
    for (int i_ind_optz_seed = 0; i_ind_optz_seed < n_ind_optz; ++i_ind_optz_seed) {
        if (optimum_label_vec(static_cast<arma::uword>(i_ind_optz_seed)) != optimum_unlabelled) continue;

        optimum_label_vec(static_cast<arma::uword>(i_ind_optz_seed)) = n_optimum;
        pending_member_vec.assign(1, i_ind_optz_seed);
        while (!pending_member_vec.empty()) {
            const int i_ind_optz_member = pending_member_vec.back();
            pending_member_vec.pop_back();

            for (int i_ind_optz_other = 0; i_ind_optz_other < n_ind_optz; ++i_ind_optz_other) {
                if (optimum_label_vec(static_cast<arma::uword>(i_ind_optz_other)) != optimum_unlabelled) continue;
                if (!is_same_optimum(multistart_result, i_ind_optz_member, i_ind_optz_other)) continue;

                optimum_label_vec(static_cast<arma::uword>(i_ind_optz_other)) = n_optimum;
                pending_member_vec.push_back(i_ind_optz_other);
            }
        }
        ++n_optimum;
    }

    return optimum_label_vec;
}

// One past the largest label, failures excluded.
inline int calc_n_optimum(const arma::uvec& optimum_label_vec) {
    int n_optimum = 0;
    for (arma::uword i_ind_optz = 0; i_ind_optz < optimum_label_vec.n_elem; ++i_ind_optz) {
        const arma::uword optimum_label = optimum_label_vec(i_ind_optz);
        if (optimum_label == OPTIMUM_LABEL_NONE) continue;
        n_optimum = std::max(n_optimum, static_cast<int>(optimum_label) + 1);
    }

    return n_optimum;
}

}  // namespace detail
}  // namespace gpflash
