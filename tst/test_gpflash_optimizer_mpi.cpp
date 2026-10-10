// The independent-optimization split over MPI processes: the only test compiled with GPFLASH_HAVE_MPI=1 and
// launched on several processes, so it is what keeps that path from rotting. Four properties, none checkable
// from one process: every process reaches the same answer, that answer equals the one a single process reaches
// alone, a failure in one surfaces on all of them rather than stranding the others in a collective, and
// processes handed different problems all refuse rather than combining them.
//
// Every assertion is itself a collective, so its verdict is the same everywhere; one process prints. The
// assertions call MPI directly rather than through ParallelContext, the wrapper the library gathers through: an
// instrument sharing a defect with its subject would report agreement either way.
#include <mpi.h>
#include <algorithm>  // std::max
#include <armadillo>
#include <cmath>  // std::sin
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "GpFlashTestSupport.hpp"
#include "gpflash/inputs/KernelId.hpp"  // kernel::KernelId
#include "gpflash/optimization/HpOptimizer.hpp"
#include "gpflash/optimization/multistart/RunMultistart.hpp"  // detail::run_multistart
#include "gpflash/optimization/search_space/OptzRegion.hpp"   // OptzRegion, build_optz_region

using gpflash_test::draw_uniform_mat;
using gpflash_test::kDetailLen;
using gpflash_test::report;

namespace {

// n_train stays below PERF_OMP_MIN_N_POINT so the assembly's own OpenMP regions never open. Above it their
// reduction order varies run to run, which would mask the cross-process agreement being tested here.
constexpr int kNTrain = 60;
constexpr int kNDim = 2;
// Coprime with GPFLASH_MPI_TEST_N_PROC in CMakeLists.txt, so the round-robin split is uneven.
constexpr int kNIndOptz = 7;

gpflash::TrainingData build_train_data() {
    std::mt19937_64 rng(7);
    gpflash::TrainingData train_data;
    train_data.X_ndim_ntrain_mat = draw_uniform_mat(rng, kNDim, kNTrain);
    train_data.f_train_vec = arma::vec(kNTrain);
    for (int i_train = 0; i_train < kNTrain; ++i_train)
        train_data.f_train_vec(i_train) = std::sin(3.0 * train_data.X_ndim_ntrain_mat(0, i_train));
    train_data.sig2_f_val = 1e-4;
    return train_data;
}

gpflash::OptzOptions build_optz_options() {
    gpflash::OptzOptions optz_opts;
    optz_opts.model_options.kernel_id = gpflash::kernel::KernelId::SqExp;
    optz_opts.use_multistart = true;
    optz_opts.n_ind_optz = kNIndOptz;
    return optz_opts;
}

// Every process must reach the same fit, checked by reducing each answer to its spread rather than trusting one
// process's print: a gather that dropped a column still looks self-consistent on the process that wrote it.
bool test_all_process_agree(const char* name, const int n_rank, const bool is_reporting_process) {
    const gpflash::MllOptzResult mll_optz_result =
            gpflash::select_hp_with_max_mll(build_train_data(), build_optz_options());

    const double mll = mll_optz_result.mll();
    double mll_min = 0.0;
    double mll_max = 0.0;
    MPI_Allreduce(&mll, &mll_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&mll, &mll_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    // The counters sum over every optimization, so a dropped column shows even when the reporting process holds
    // the winner.
    const int n_eval = mll_optz_result.n_total_mll_eval() + mll_optz_result.n_total_mll_grad_eval();
    int n_eval_min = 0;
    int n_eval_max = 0;
    MPI_Allreduce(&n_eval, &n_eval_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&n_eval, &n_eval_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    // The hyperparameters too, not just the scalar summary: two processes could agree on the MLL while
    // disagreeing on where it was attained, and the gather transports these element by element. Held by value, the
    // record assembling its columns on demand rather than storing points to hand out by reference.
    const arma::vec ln_hp_vec = mll_optz_result.multistart_result.best_ln_hp_vec();
    const int n_hp_optz = static_cast<int>(ln_hp_vec.n_elem);
    arma::vec ln_hp_min_vec(ln_hp_vec.n_elem);
    arma::vec ln_hp_max_vec(ln_hp_vec.n_elem);
    MPI_Allreduce(ln_hp_vec.memptr(), ln_hp_min_vec.memptr(), n_hp_optz, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(ln_hp_vec.memptr(), ln_hp_max_vec.memptr(), n_hp_optz, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    const double ln_hp_spread = arma::max(ln_hp_max_vec - ln_hp_min_vec);

    // Bitwise, not to a tolerance: below PERF_OMP_MIN_N_POINT each optimization is deterministic and the gather
    // adds only exact zeros, so any difference is a defect rather than rounding.
    const bool pass = mll_min == mll_max && n_eval_min == n_eval_max && ln_hp_spread == 0.0
            && mll_optz_result.n_ind_optz_run() == kNIndOptz;
    if (!is_reporting_process) return pass;

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "n_rank=%d  mll=%.14f spread=%.1e  ln_hp_spread=%.1e  ev=%d  n_ind_optz_run=%d",
            n_rank,
            mll_min,
            mll_max - mll_min,
            ln_hp_spread,
            n_eval_min,
            mll_optz_result.n_ind_optz_run());
    return report(name, pass, detail);
}

// Consensus is not correctness: every process can agree on a record the gather dropped a column from. The same
// deterministic fit runs twice, over MPI_COMM_WORLD and then over MPI_COMM_SELF, where each process runs every
// optimization alone and no collective is entered, and each process compares the two records entry by entry
// before the verdicts are reduced.
bool test_distributed_matches_serial(const char* name, const int n_rank, const bool is_reporting_process) {
    const gpflash::MllOptzResult distributed_result =
            gpflash::select_hp_with_max_mll(build_train_data(), build_optz_options());
    const gpflash::MllOptzResult serial_result = gpflash::select_hp_with_max_mll(
            build_train_data(), build_optz_options(), gpflash::ParallelContext(MPI_COMM_SELF));
    const gpflash::MultistartResult& distributed_multistart = distributed_result.multistart_result;
    const gpflash::MultistartResult& serial_multistart = serial_result.multistart_result;

    // Bitwise, for the reason test_all_process_agree gives.
    const bool shapes_match = distributed_multistart.n_ind_optz_run() == serial_multistart.n_ind_optz_run()
            && distributed_multistart.ln_hp_mat.n_rows == serial_multistart.ln_hp_mat.n_rows;
    double worst_diff = 0.0;
    if (shapes_match)
        worst_diff = std::max(
                {arma::abs(distributed_multistart.ln_hp_mat - serial_multistart.ln_hp_mat).max(),
                 arma::abs(distributed_multistart.ln_hp_init_mat - serial_multistart.ln_hp_init_mat).max(),
                 arma::abs(distributed_multistart.start_field_mat - serial_multistart.start_field_mat).max()});
    // start_field_mat holds each start's objective, evaluation counts and stop reason; the winner and the fitted
    // hyperparameters checked next are what a caller reads.
    const bool winner_matches = distributed_multistart.idx_best() == serial_multistart.idx_best()
            && distributed_result.mll() == serial_result.mll()
            && arma::abs(distributed_result.hp_val_struct.hp_gamma_vec - serial_result.hp_val_struct.hp_gamma_vec).max()
                    == 0.0;

    const int pass_local = (shapes_match && worst_diff == 0.0 && winner_matches) ? 1 : 0;
    int pass_every_process = 0;
    MPI_Allreduce(&pass_local, &pass_every_process, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!is_reporting_process) return pass_every_process == 1;

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "n_rank=%d  shapes=%d  worst_entry_diff=%.1e  winner+mll+hp_gamma match=%d",
            n_rank,
            static_cast<int>(shapes_match),
            worst_diff,
            static_cast<int>(winner_matches));
    return report(name, pass_every_process == 1, detail);
}

// One process handed a different problem must make every process raise, rather than the gather summing columns
// from two searches. The last process dissents in turn on a count that sizes the collectives, on a training value
// that does not, and on each model field, which changes no count at all. With one process there is no peer to
// disagree with, so nothing raises.
bool test_input_disagreement_reaches_all_process(const char* name, const int n_rank, const bool is_reporting_process) {
    int i_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &i_rank);
    const bool is_dissenting_process = i_rank == n_rank - 1;
    const bool expects_raise = n_rank > 1;

    enum class DissentOn {
        CollectiveCount,
        TrainingValue,
        Kernel,
        CondMax,
        EigMaxUbMethod,
        Symmetry,
        MeanModelConstant,
        MeanModelDegree,
        MeanModelEigMaxUbMethod
    };

    // field_name is what the comparison that catches the dissent names, on every process.
    struct DissentCase {
        DissentOn dissent_on;
        const char* field_name;
    };
    const DissentCase dissent_case_list[] = {
            {DissentOn::CollectiveCount, "the number of independent optimizations"},
            {DissentOn::TrainingValue, "the training data"},
            {DissentOn::Kernel, "the kernel id"},
            {DissentOn::CondMax, "ModelOptions::cond_max"},
            {DissentOn::EigMaxUbMethod, "ModelOptions::kmat_eig_max_ub_method"},
            {DissentOn::Symmetry, "ModelOptions::symmetry_spec"},
            {DissentOn::MeanModelConstant, "ModelOptions::has_mean_model_constant"},
            {DissentOn::MeanModelDegree, "ModelOptions::mean_model_degree_vec"},
            {DissentOn::MeanModelEigMaxUbMethod, "ModelOptions::mean_model_eig_max_ub_method"}};

    bool pass = true;
    std::string failed_case_list;  // DissentOn indices, in declaration order
    for (const DissentCase& dissent_case : dissent_case_list) {
        const DissentOn dissent_on = dissent_case.dissent_on;
        gpflash::OptzOptions optz_opts = build_optz_options();
        gpflash::TrainingData train_data = build_train_data();
        if (is_dissenting_process) {
            switch (dissent_on) {
                case DissentOn::CollectiveCount:
                    optz_opts.n_ind_optz = kNIndOptz + 1;
                    break;
                case DissentOn::TrainingValue:
                    train_data.f_train_vec(0) += 1.0;
                    break;
                // Matern52 declares as many hyperparameters as the SqExp the other processes hold, so every
                // count the search compares still agrees and only the model comparison can catch this.
                case DissentOn::Kernel:
                    optz_opts.model_options.kernel_id = gpflash::kernel::KernelId::Matern52;
                    break;
                // Inside the accepted range, so the field checks pass on every process and leave the
                // disagreement itself as the only thing to catch.
                case DissentOn::CondMax:
                    optz_opts.model_options.cond_max = 10.0 * gpflash::COND_MAX_DEFAULT;
                    break;
                case DissentOn::EigMaxUbMethod:
                    optz_opts.model_options.kmat_eig_max_ub_method = gpflash::EigMaxUbMethod::Trace;
                    break;
                // A valid even mirror on the last dimension, so it too passes every local check.
                case DissentOn::Symmetry:
                    optz_opts.model_options.symmetry_spec.mirror_sign_vec = arma::ones<arma::vec>(kNDim);
                    optz_opts.model_options.symmetry_spec.mirror_sign_vec(kNDim - 1) = -1.0;
                    break;
                // The mean model weights are solved, not searched, so neither changes a collective's size either.
                case DissentOn::MeanModelConstant:
                    optz_opts.model_options.has_mean_model_constant = false;
                    break;
                case DissentOn::MeanModelDegree:
                    optz_opts.model_options.mean_model_degree_vec = {1};
                    break;
                case DissentOn::MeanModelEigMaxUbMethod:
                    optz_opts.model_options.mean_model_eig_max_ub_method = gpflash::EigMaxUbMethod::Gershgorin;
                    break;
            }
        }

        bool threw = false;
        bool threw_naming_field = false;
        bool threw_other_type = false;
        try {
            gpflash::select_hp_with_max_mll(train_data, optz_opts);
        } catch (const gpflash::ErrorInvalidInput& caught) {
            threw = true;
            threw_naming_field = std::string(caught.what()).find(dissent_case.field_name) != std::string::npos;
        } catch (...) {
            // Any other type is the regression, so it fails the reduction below rather than leaving the rest waiting in
            // it.
            threw_other_type = true;
        }

        // MPI_MIN over a 0/1 flag: every process must have seen the expected outcome.
        const bool is_as_expected = !threw_other_type && threw == expects_raise && threw_naming_field == expects_raise;
        const int as_expected_flag = is_as_expected ? 1 : 0;
        int n_least_as_expected = 0;
        MPI_Allreduce(&as_expected_flag, &n_least_as_expected, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (n_least_as_expected != 1) failed_case_list += " " + std::to_string(static_cast<int>(dissent_on));
        pass = pass && n_least_as_expected == 1;
    }
    if (!is_reporting_process) return pass;

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "n_rank=%d  expects_raise=%d  failed_dissent_on=[%s ]",
            n_rank,
            static_cast<int>(expects_raise),
            failed_case_list.c_str());
    return report(name, pass, detail);
}

// A rejected configuration must raise on every process, never a subset. Input validation catches this one before
// the split, so it covers the symmetric case; test_local_rejection_reaches_all_process below covers one process alone.
bool test_rejected_config_reaches_all_process(const char* name, const int n_rank, const bool is_reporting_process) {
    gpflash::OptzOptions bad_optz_opts = build_optz_options();
    // Rejected because cond_max must be finite and above 1: 1 + 1e-300 rounds to exactly 1 in double.
    bad_optz_opts.model_options.cond_max = 1.0 + 1e-300;

    std::string message;
    bool raised_cond_max_rejection = false;
    try {
        gpflash::select_hp_with_max_mll(build_train_data(), bad_optz_opts);
    } catch (const gpflash::ErrorInvalidInput& caught) {
        raised_cond_max_rejection = std::string(caught.what()).find("ModelOptions::cond_max") != std::string::npos;
        message = caught.what();
    } catch (...) {
        // Any other type is the regression, so it fails the reduction below rather than leaving the rest waiting in it.
    }

    // MPI_MIN over a 0/1 flag, so the result is 1 only if every process raised the cond_max rejection.
    const int raised_flag = raised_cond_max_rejection ? 1 : 0;
    int n_least_raised = 0;
    MPI_Allreduce(&raised_flag, &n_least_raised, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    const bool pass = n_least_raised == 1;
    if (!is_reporting_process) return pass;

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "n_rank=%d  every_process_raised=%d  msg=%.48s",
            n_rank,
            n_least_raised,
            message.c_str());
    return report(name, pass, detail);
}

// One process whose own options fail a local check must not strand its peers in the search's first collective. The
// last process holds a negative tolerance, which no cross-process comparison covers: it raises ErrorInvalidInput and
// every other process ErrorPeerFailure naming it. A hang is the failure, so the ctest timeout is the assertion that
// catches it.
bool test_local_rejection_reaches_all_process(const char* name, const int n_rank, const bool is_reporting_process) {
    int i_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &i_rank);
    const bool is_rejecting_process = i_rank == n_rank - 1;
    const std::string rejecting_process_name =
            "process " + std::to_string(n_rank - 1) + " of " + std::to_string(n_rank);

    gpflash::OptzOptions optz_opts = build_optz_options();
    if (is_rejecting_process) optz_opts.rel_obj_tol = -1.0;

    bool raised_expected_type = false;
    try {
        gpflash::select_hp_with_max_mll(build_train_data(), optz_opts);
    } catch (const gpflash::ErrorInvalidInput&) {
        raised_expected_type = is_rejecting_process;
    } catch (const gpflash::ErrorPeerFailure& caught) {
        raised_expected_type =
                !is_rejecting_process && std::string(caught.what()).find(rejecting_process_name) != std::string::npos;
    } catch (...) {
        // Any other type is the regression, so it fails the reduction below rather than leaving the rest waiting in it.
        raised_expected_type = false;
    }

    // MPI_MIN over a 0/1 flag, so the result is 1 only if every process raised the type expected of it.
    const int as_expected_flag = raised_expected_type ? 1 : 0;
    int n_least_as_expected = 0;
    MPI_Allreduce(&as_expected_flag, &n_least_as_expected, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    const bool pass = n_least_as_expected == 1;
    if (!is_reporting_process) return pass;

    char detail[kDetailLen];
    std::snprintf(detail, sizeof(detail), "n_rank=%d  every_process_as_expected=%d", n_rank, n_least_as_expected);
    return report(name, pass, detail);
}

// run_multistart's failure paths, over several processes:
//   1. a start that fails numerically is recorded, not rethrown;
//   2. every start failing ends the fit on every process;
//   3. a defect on one process ends the fit on all of them, the raiser rethrowing its own type and the rest throwing
//      ErrorPeerFailure naming it.
// Driven with synthetic objectives, since nothing in OptzOptions makes a start fail. None may unwind alone (a timeout
// here is the diagnosis).
bool test_start_failure_paths(const char* name, const int n_rank, const bool is_reporting_process) {
    const gpflash::TrainingData train_data = build_train_data();
    const gpflash::OptzOptions optz_opts = build_optz_options();
    const gpflash::ParallelContext parallel_context;
    const gpflash::detail::MllOptzObjective optz_objective(train_data, optz_opts, "test");
    const gpflash::HpOptzLayout& hp_optz_layout = optz_objective.layout();
    const double f_train_residual_var = arma::var(train_data.f_train_vec);  // the constant mean model's
    const arma::vec x_center_vec = arma::zeros<arma::vec>(train_data.X_ndim_ntrain_mat.n_rows);
    const gpflash::OptzRegion optz_region = gpflash::detail::build_optz_region(
            train_data, x_center_vec, f_train_residual_var, optz_opts, gpflash::OptzSeeding(), hp_optz_layout, "test");

    // A convex quadratic that raises at exactly optz_region.start_vec, which only optimization 0 evaluates. A failing
    // half-space would also catch every solve that steps through it.
    const arma::vec ln_hp_start_vec = optz_region.start_vec;
    const gpflash::detail::LocalObjective one_failing_objective =
            [ln_hp_start_vec](const arma::vec& ln_hp_vec, arma::vec* grad_vec_out) -> double {
        if (arma::approx_equal(ln_hp_vec, ln_hp_start_vec, "absdiff", 0.0))
            throw gpflash::ErrorNumericalFailure("seeded failure at the heuristic start");
        if (grad_vec_out) *grad_vec_out = 2.0 * ln_hp_vec;
        return arma::dot(ln_hp_vec, ln_hp_vec);
    };

    gpflash::MultistartResult multistart_result;
    bool survived = true;
    try {
        multistart_result = gpflash::detail::run_multistart(
                optz_region, hp_optz_layout, one_failing_objective, optz_opts, parallel_context, "test");
    } catch (...) {
        survived = false;
    }

    // Every process must see the same count, or the failed column did not cross the gather.
    const int n_failed = survived ? multistart_result.calc_n_failed_start() : -1;
    int n_failed_min = 0;
    int n_failed_max = 0;
    MPI_Allreduce(&n_failed, &n_failed_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&n_failed, &n_failed_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    // Exactly one fails; a failed column is +inf and a surviving one finite.
    bool columns_ok = survived && n_failed == 1 && multistart_result.stop_reason(0) == gpflash::OptzStopReason::Failed;
    if (survived)
        for (int i_ind_optz = 0; i_ind_optz < multistart_result.n_ind_optz_run(); ++i_ind_optz) {
            const bool is_failed = multistart_result.stop_reason(i_ind_optz) == gpflash::OptzStopReason::Failed;
            const double objective_val =
                    multistart_result.start_field_val(gpflash::StartField::ObjectiveVal, i_ind_optz);
            columns_ok = columns_ok
                    && (is_failed ? !gpflash::detail::is_finite_val(objective_val)
                                  : gpflash::detail::is_finite_val(objective_val));
        }
    // The winner ran.
    const bool winner_survived = survived
            && multistart_result.stop_reason(multistart_result.idx_best()) != gpflash::OptzStopReason::Failed
            && gpflash::detail::is_finite_val(multistart_result.mll());

    // All failing ends the fit, on every process.
    const gpflash::detail::LocalObjective always_failing_objective = [](const arma::vec&, arma::vec*) -> double {
        throw gpflash::ErrorNumericalFailure("seeded failure at every start");
    };
    bool all_failed_raised = false;
    try {
        gpflash::detail::run_multistart(
                optz_region, hp_optz_layout, always_failing_objective, optz_opts, parallel_context, "test");
    } catch (const gpflash::ErrorNumericalFailure&) {
        all_failed_raised = true;
    } catch (...) {
        // Any other type is the regression, so it fails the reduction below rather than leaving the rest waiting in it.
    }
    const int all_failed_flag = all_failed_raised ? 1 : 0;
    int n_least_all_failed = 0;
    MPI_Allreduce(&all_failed_flag, &n_least_all_failed, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    // A defect on one process ends the fit on every process: the one that raised rethrows its own type, and the rest
    // throw ErrorPeerFailure naming it. Only optimization 0 evaluates the heuristic start, and process 0 runs it.
    const gpflash::detail::LocalObjective one_defective_objective =
            [ln_hp_start_vec](const arma::vec& ln_hp_vec, arma::vec* grad_vec_out) -> double {
        if (arma::approx_equal(ln_hp_vec, ln_hp_start_vec, "absdiff", 0.0))
            throw gpflash::ErrorNotImplemented("seeded defect at the heuristic start");
        if (grad_vec_out) *grad_vec_out = 2.0 * ln_hp_vec;
        return arma::dot(ln_hp_vec, ln_hp_vec);
    };
    int i_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &i_rank);
    const bool is_defective_process = i_rank == 0;
    const std::string defective_process_name = "process 0 of " + std::to_string(n_rank);
    bool defect_as_expected = false;
    try {
        gpflash::detail::run_multistart(
                optz_region, hp_optz_layout, one_defective_objective, optz_opts, parallel_context, "test");
    } catch (const gpflash::ErrorNotImplemented&) {
        defect_as_expected = is_defective_process;
    } catch (const gpflash::ErrorPeerFailure& caught) {
        defect_as_expected =
                !is_defective_process && std::string(caught.what()).find(defective_process_name) != std::string::npos;
    } catch (...) {
        // Any other type is the regression, so it fails the reduction below rather than leaving the rest waiting in it.
        defect_as_expected = false;
    }
    const int defect_flag = defect_as_expected ? 1 : 0;
    int n_least_defect_as_expected = 0;
    MPI_Allreduce(&defect_flag, &n_least_defect_as_expected, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    // Reduced, so the verdict covers every process rather than the one that prints.
    const int local_pass = (columns_ok && winner_survived) ? 1 : 0;
    int all_pass = 0;
    MPI_Allreduce(&local_pass, &all_pass, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    const bool pass = all_pass == 1 && n_least_all_failed == 1 && n_least_defect_as_expected == 1;
    if (!is_reporting_process) return pass;
    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "n_rank=%d  n_failed=%d (spread=%d)  winner_survived=%d  all_failed_raised_everywhere=%d  "
            "defect_as_expected_everywhere=%d",
            n_rank,
            n_failed_min,
            n_failed_max - n_failed_min,
            static_cast<int>(winner_survived),
            n_least_all_failed,
            n_least_defect_as_expected);
    return report(name, pass, detail);
}

}  // namespace

int main(int argc, char** argv) {
    // FUNNELED, the level the README asks of a program: GpFlash calls MPI only from the thread that initialized it,
    // while its OpenMP regions run other threads. Plain MPI_Init promises a single thread, which no GpFlash program
    // keeps.
    int provided_thread_level = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided_thread_level);
    int i_rank = 0;
    int n_rank = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &i_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_rank);
    const bool is_reporting_process = i_rank == 0;

    if (is_reporting_process) std::printf("== GpFlash optimizer over MPI ==\n");

    // Reduced, so every process stops together if any was granted less; the rest would test outside MPI's contract.
    int lowest_thread_level = provided_thread_level;
    MPI_Allreduce(MPI_IN_PLACE, &lowest_thread_level, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    const bool has_thread_level = lowest_thread_level >= MPI_THREAD_FUNNELED;
    if (is_reporting_process) {
        char detail[kDetailLen];
        std::snprintf(
                detail,
                sizeof(detail),
                "n_rank=%d  requested=%d (MPI_THREAD_FUNNELED)  lowest_provided=%d",
                n_rank,
                MPI_THREAD_FUNNELED,
                lowest_thread_level);
        report("thread_level_funneled", has_thread_level, detail);
    }
    if (!has_thread_level) {
        if (is_reporting_process) std::printf("\nSOME TESTS FAILED\n");
        MPI_Finalize();
        return 1;
    }

    bool ok = true;
    ok &= test_all_process_agree("all_process_agree", n_rank, is_reporting_process);
    ok &= test_distributed_matches_serial("distributed_matches_serial", n_rank, is_reporting_process);
    ok &= test_input_disagreement_reaches_all_process(
            "input_disagreement_reaches_all_process", n_rank, is_reporting_process);
    ok &= test_rejected_config_reaches_all_process("rejected_config_reaches_all_process", n_rank, is_reporting_process);
    ok &= test_local_rejection_reaches_all_process("local_rejection_reaches_all_process", n_rank, is_reporting_process);
    ok &= test_start_failure_paths("start_failure_paths", n_rank, is_reporting_process);
    if (is_reporting_process) std::printf("\n%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");

    MPI_Finalize();
    // Every process returns the verdict, so ctest fails if any of them disagrees.
    return ok ? 0 : 1;
}
