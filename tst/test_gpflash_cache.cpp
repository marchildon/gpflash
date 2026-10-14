// PosteriorCache: the on-disk format. What this file asserts:
//
//   1) roundtrip_predictions_match  save then load reproduces the original's predictions, single and batch, and
//                                   its mean model, for the constant, zero and {2, 1, 0} mean models
//   2) peek_kernel_id               the file's kernel is readable from the header, before the arrays
//   3) kernel_mismatch_is_rejected  naming a kernel other than the file's throws, through
//                                   load_precompute_for_kernel and load_posterior_verified
//   4) bad_format_version           a foreign format_version is refused by name, by peek_info too, before any length
//   5) truncated_file               a half-written file throws rather than reading a partial array
//   6) missing_file                 a path that does not exist is reported as missing
//   7) input_hash_detects_stale     load_posterior_verified accepts the fit's own inputs, including the same mean
//                                   model degrees written per dimension, and rejects changed ones, the mean model
//                                   and either eig_max bound included
//   8) version_and_note_round_trip  peek_info returns the scalars without loading the arrays
//   9) feature_dim_differs          n_dim and n_feature stay separate through a save/load
//  10) mll_matches_eval_mll         the stored mll equals eval_mll's, and survives a save/load
//  11) mean_only_roundtrip          a file saved without variance data reproduces the mean and its gradient
//  12) mean_only_refuses_variance   both variance quantities are refused, the mean is not
//  13) load_mean_only_file         a load serves a file saved without variance data, refusing the variance
//  14) save_refuses_foreign_destination  a non-cache file at the destination survives; the save throws
//  15) verified_raises_on_stale_file  a file built from other inputs is reported, left intact, replaceable
//  16) model_info_survives_resave  a loaded posterior re-saves with the original fit's identity, not zeros
//  17) bound_method_round_trip     non-default eig_max bound methods survive a save and a resave
//  18) summary_reports_mean_model_constant  to_summary_string puts every field on its own line, the mean model
//                                  constant, the variance data and the symmetry kind in each state
//  19) failed_save_removes_scratch_file  a save that fails part way through raises and leaves neither the .tmp nor
//                                  the destination behind
//  20) default_cache_path_names_kernel_and_shape  the default file name, bare and after a directory either way
//  21) header_slot_layout_is_pinned  every slot sits at its literal position, a header written slot by slot reads
//                                  back with every field in its own slot, and every declared slot is visited
//                                  exactly once by the one slot list
//  22) summary_carries_whole_note  a note of any length reaches the summary whole, and no note at all reads as
//                                  "(none)"
//  23) summary_written_at_is_utc_or_unknown  the written-at line is the saved time in UTC, and "unknown" when unset
//                                  or unconvertible
//  24) factor_stored_packed        the file holds the factor's lower triangle packed, and nothing above it
//  25) disagreeing_parts_are_rejected  a mirror and the header's symmetry, the header's counts and the arrays', or a
//                                  per-point vector or the packed factor and n_train, that disagree are refused, as
//                                  is a packed factor the header promises and the file lacks
//
// Every stored array is derived from the fit, so the file carries an input hash. load_posterior skips that
// check, for when the raw dataset is gone; load_posterior_verified makes it.
#include <armadillo>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#if !defined(_WIN32)
#include <sys/resource.h>  // setrlimit, the file-size limit that fails a save part way through
#include <csignal>         // SIGXFSZ, which the limit would otherwise raise
#endif

#include "GpFlashTestSupport.hpp"
#include "gpflash/posterior_build/PosteriorBuild.hpp"     // make_posterior, load_posterior_verified
#include "gpflash/posterior_cache/CacheInfo.hpp"          // CacheInfo, to_summary_string
#include "gpflash/posterior_cache/CacheInputHash.hpp"     // calc_default_cache_path
#include "gpflash/posterior_cache/CacheLayout.hpp"        // the header packing, the array list
#include "gpflash/posterior_cache/Hdf5Store.hpp"          // save_array, for the hand-written file below
#include "gpflash/posterior_cache/PosteriorCache.hpp"     // peek_info, save_precompute, load_precompute
#include "gpflash/posterior_cache/PosteriorContents.hpp"  // cache::PosteriorContents
#include "gpflash/posterior_eval/GpPosterior.hpp"         // load_posterior
#include "gpflash/training/EvalMllDispatch.hpp"           // eval_mll, for the cross-tier MLL check

// C++14 stand-ins for the three <filesystem> facilities this test needs: <filesystem> is C++17, and an
// <experimental/filesystem> shim would drag in the -lstdc++fs link on GCC 7/8.
namespace gpf_fs {

inline void remove(const std::string& path) { std::remove(path.c_str()); }

inline std::streamsize file_size(const std::string& path) {
    std::ifstream input_stream(path, std::ios::binary | std::ios::ate);
    return input_stream.tellg();
}

// Truncate by rewriting the head. Portable, and these cache files are kilobytes.
inline void resize_file(const std::string& path, std::streamsize n_bytes) {
    std::vector<char> head(static_cast<std::size_t>(n_bytes));
    {
        std::ifstream input_stream(path, std::ios::binary);
        input_stream.read(head.data(), n_bytes);
    }
    std::ofstream output_stream(path, std::ios::binary | std::ios::trunc);
    output_stream.write(head.data(), n_bytes);
}

}  // namespace gpf_fs

using gpflash_test::draw_normal;
using gpflash_test::draw_uniform_mat;
using gpflash_test::draw_uniform_vec;
using gpflash_test::kTolAltPath;
using gpflash_test::report;

using namespace gpflash;

namespace {

// The fixed-seed inputs the tests fit against.  Returned separately as well as
// fitted, because the input-hash check needs to be handed the same inputs.
struct FitInputs {
    TrainingData train_data;
    HpVal hp_val_struct;
    ModelOptions model_opts;  // the kernel, cond_max, mean model and bound methods
};

// The cache is checked through predictions; where the point is that two posteriors agree, the mean suffices.
constexpr PosteriorOptions kOptsMeanOnly{
        /*return_mean=*/true, /*return_var=*/false, /*return_mean_gradient=*/false, /*return_var_gradient=*/false};

FitInputs make_fit_inputs() {
    std::mt19937_64 rng(42);
    const int n_train = 40, n_dim = 3;
    FitInputs fit_inputs;
    fit_inputs.train_data.X_ndim_ntrain_mat.set_size(n_dim, n_train);
    for (double& x_val : fit_inputs.train_data.X_ndim_ntrain_mat) x_val = draw_normal(rng);
    fit_inputs.train_data.f_train_vec.set_size(n_train);
    for (double& f_val : fit_inputs.train_data.f_train_vec) f_val = draw_normal(rng);
    // Non-unit, mutually unequal variances so the round-trip actually validates the stored
    // signal variance (hp_sig2_k = 1 would hide a dropped factor).
    fit_inputs.train_data.sig2_f_val = 0.013;
    fit_inputs.hp_val_struct.hp_gamma_vec = arma::ones<arma::vec>(n_dim);
    fit_inputs.hp_val_struct.hp_sig2_k = 2.7;
    // The kernel these tests fit with: make_posterior resolves it from this field, and every file below records
    // it.
    fit_inputs.model_opts.kernel_id = gpflash::kernel::KernelId::SqExp;

    return fit_inputs;
}

// A small, fixed-seed fitted posterior to exercise save/load against.
GpPosterior make_fitted_posterior() {
    const FitInputs fit_inputs = make_fit_inputs();
    return make_posterior(fit_inputs.train_data, fit_inputs.hp_val_struct, fit_inputs.model_opts);
}

// A reloaded posterior evaluates the same arithmetic on the same arrays, but the compiler may associate or contract
// it differently at each call site, so comparisons use kTolAltPath rather than zero.

bool close(const arma::vec& a_vec, const arma::vec& b_vec) { return arma::norm(a_vec - b_vec, "inf") <= kTolAltPath; }
bool close(const arma::mat& a_mat, const arma::mat& b_mat) {
    return arma::norm(arma::vectorise(a_mat - b_mat), "inf") <= kTolAltPath;
}
bool close(const double a, const double b) { return std::abs(a - b) <= kTolAltPath; }

// What function throws, or empty when it returns.
template <class Function>
std::string calc_error_message(Function&& function) {
    try {
        function();
    } catch (const std::exception& caught) {
        return caught.what();
    }
    return std::string();
}

// True if function throws.
template <class Function>
bool throws(Function&& function) {
    try {
        function();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

// True if function throws a cache error raised under expected_caller, the entry point rather than a helper below it.
template <class Function>
bool throws_naming(const char* expected_caller, Function&& function) {
    const std::string expected_prefix = std::string("gpflash::cache::") + expected_caller + ":";
    try {
        function();
    } catch (const std::exception& caught) {
        return std::string(caught.what()).find(expected_prefix) != std::string::npos;
    }
    return false;
}

// True if function throws load_posterior_verified's refusal of a file built from other inputs, and not another error.
template <class Function>
bool is_stale_refusal(Function&& function) {
    const std::string error_message = calc_error_message(function);
    return error_message.find("gpflash::cache::load_posterior_verified: ") != std::string::npos
            && error_message.find("was built from different inputs") != std::string::npos;
}

// -----------------------------------------------------------------------
// Test 1: save + load reproduces the predictions, at a single query and a batch, for the given mean model.
bool test_roundtrip_predictions_match(
        const char* name, const bool has_mean_model_constant, const arma::uvec& mean_model_degree_vec) {
    FitInputs fit_inputs = make_fit_inputs();
    fit_inputs.model_opts.has_mean_model_constant = has_mean_model_constant;
    fit_inputs.model_opts.mean_model_degree_vec = mean_model_degree_vec;
    GpPosterior posterior = make_posterior(fit_inputs.train_data, fit_inputs.hp_val_struct, fit_inputs.model_opts);
    const std::string path = "test_cache_roundtrip.h5";
    posterior.save_precompute(path);
    GpPosterior posterior_loaded = load_posterior(path);
    gpf_fs::remove(path);

    std::mt19937_64 rng(7);
    arma::vec x_post_vec(posterior.n_dim());
    for (auto& v : x_post_vec) v = draw_normal(rng);

    const gpflash::PosteriorOptions opts_all{true, true, true, true};  // every field
    const gpflash::PosteriorResult post_result_saved = posterior.eval(x_post_vec, opts_all);
    const gpflash::PosteriorResult post_result_loaded = posterior_loaded.eval(x_post_vec, opts_all);

    arma::mat X_ndim_npost_mat(posterior.n_dim(), 5);
    for (auto& v : X_ndim_npost_mat) v = draw_normal(rng);
    const gpflash::PosteriorResult batch_result_saved = posterior.eval(X_ndim_npost_mat, opts_all);
    const gpflash::PosteriorResult batch_result_loaded = posterior_loaded.eval(X_ndim_npost_mat, opts_all);

    const bool single_ok = close(post_result_saved.mean_post_vec, post_result_loaded.mean_post_vec)
            && close(post_result_saved.var_post_vec, post_result_loaded.var_post_vec)
            && close(post_result_saved.d_mean_post_d_x_post_ndim_npost_mat,
                     post_result_loaded.d_mean_post_d_x_post_ndim_npost_mat)
            && close(post_result_saved.d_var_post_d_x_post_ndim_npost_mat,
                     post_result_loaded.d_var_post_d_x_post_ndim_npost_mat);
    const bool batch_ok = close(batch_result_saved.mean_post_vec, batch_result_loaded.mean_post_vec)
            && close(batch_result_saved.var_post_vec, batch_result_loaded.var_post_vec)
            && close(batch_result_saved.d_mean_post_d_x_post_ndim_npost_mat,
                     batch_result_loaded.d_mean_post_d_x_post_ndim_npost_mat)
            && close(batch_result_saved.d_var_post_d_x_post_ndim_npost_mat,
                     batch_result_loaded.d_var_post_d_x_post_ndim_npost_mat);
    const bool shape_ok =
            posterior_loaded.n_train() == posterior.n_train() && posterior_loaded.n_dim() == posterior.n_dim();
    const bool mean_model_ok = posterior_loaded.mean_model().has_constant == posterior.mean_model().has_constant
            && arma::all(posterior_loaded.mean_model().degree_vec == posterior.mean_model().degree_vec)
            && arma::approx_equal(posterior_loaded.hp_beta_vec(), posterior.hp_beta_vec(), "absdiff", kTolAltPath);

    const bool pass = single_ok && batch_ok && shape_ok && mean_model_ok;
    char detail[gpflash_test::kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "single %d  batch %d  shape %d  mean_model %d",
            single_ok ? 1 : 0,
            batch_ok ? 1 : 0,
            shape_ok ? 1 : 0,
            mean_model_ok ? 1 : 0);
    report(name, pass, detail);

    return pass;
}

// -----------------------------------------------------------------------
// Test 2: peek_kernel_id reports the file's kernel from the header, without loading the arrays.
bool test_peek_kernel_id_reports_correct_kernel() {
    FitInputs fit_inputs = make_fit_inputs();
    const std::string path = "test_cache_peek.h5";
    make_posterior(fit_inputs.train_data, fit_inputs.hp_val_struct, fit_inputs.model_opts).save_precompute(path);
    const gpflash::kernel::KernelId sq_exp_peeked_id = cache::peek_kernel_id(path);

    // SqExp is KernelId 0, which a zero-filled kernel slot also reads as, so a non-zero id shows the slot is read.
    fit_inputs.model_opts.kernel_id = gpflash::kernel::KernelId::Matern52;
    make_posterior(fit_inputs.train_data, fit_inputs.hp_val_struct, fit_inputs.model_opts).save_precompute(path);
    const gpflash::kernel::KernelId matern52_peeked_id = cache::peek_kernel_id(path);
    gpf_fs::remove(path);

    const bool pass = sq_exp_peeked_id == gpflash::kernel::KernelId::SqExp
            && matern52_peeked_id == gpflash::kernel::KernelId::Matern52;
    char detail[gpflash_test::kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "sq_exp read as %d  matern52 read as %d",
            static_cast<int>(sq_exp_peeked_id),
            static_cast<int>(matern52_peeked_id));
    report("peek_kernel_id_reports_correct_kernel", pass, detail);
    return pass;
}

// -----------------------------------------------------------------------
// Test 3: the file records which kernel wrote it, so a load at any other kernel must refuse rather than
// reinterpreting SqExp's L / lin_sol_sigma_residual as Matern52's. load_posterior reads the kernel off the file
// and cannot be asked for another, so this drives load_precompute_for_kernel and load_posterior_verified.
bool test_kernel_mismatch_is_rejected() {
    const FitInputs fit_inputs = make_fit_inputs();
    const std::string path = "test_cache_kernel_mismatch.h5";
    make_posterior(fit_inputs.train_data, fit_inputs.hp_val_struct, fit_inputs.model_opts).save_precompute(path);

    // No input hash is passed, so the kernel is the only thing that can throw.
    const bool loader_refuses = throws_naming("kernel_mismatch", [&] {
        cache::load_precompute_for_kernel(
                gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Matern52), path, "kernel_mismatch");
    });

    ModelOptions model_opts_other_kernel = fit_inputs.model_opts;
    model_opts_other_kernel.kernel_id = gpflash::kernel::KernelId::Matern52;
    const bool verified_refuses = throws_naming("load_posterior_verified", [&] {
        load_posterior_verified(path, fit_inputs.train_data, fit_inputs.hp_val_struct, model_opts_other_kernel);
    });

    gpf_fs::remove(path);
    const bool pass = loader_refuses && verified_refuses;
    report("kernel_mismatch_is_rejected", pass);

    return pass;
}

// Hand-write a file bypassing cache::save_precompute, valid except for what tamper changes, so each check on a
// loaded file can be tested in isolation. tamper(cache_info, precompute, L_Sigma_packed_vec) runs before anything
// is written.
template <class Tamper>
void write_hand_made_file(const std::string& path, Tamper&& tamper, const bool is_packed_factor_written = true) {
    namespace cd = gpflash::cache::detail;
    const int n_dim = 3, n_train = 5;
    gpf_fs::remove(path);  // every array below is written once, so the file has to start out absent

    // The headers, packed the way the writer packs them.
    gpflash::cache::CacheInfo cache_info;
    cache_info.format_version = cache::cache_format_version;
    cache_info.post_model_info.kernel_id = gpflash::kernel::KernelId::SqExp;
    cache_info.n_dim = n_dim;
    cache_info.n_train = n_train;
    cache_info.n_feature = n_dim;
    cache_info.version_minor = 1;
    cache_info.saved_unix_time_s = 1;
    cache_info.has_variance_data = true;
    cache_info.hp_sig2_k = 2.7;
    cache_info.post_model_info.mll = -1.0;
    cache_info.post_model_info.sig2_f_val = 0.013;
    cache_info.post_model_info.cond_max = 1e10;
    cache_info.post_model_info.kmat_eig_max_ub_method = gpflash::EigMaxUbMethod::Gershgorin;
    cache_info.post_model_info.mean_model_eig_max_ub_method = gpflash::EigMaxUbMethod::Trace;

    // Walked rather than listed, so an array added to the format lands here too instead of leaving this file
    // one short.
    gpflash::detail::PosteriorPrecompute precompute;
    precompute.x_center_vec = arma::zeros<arma::vec>(n_dim);
    precompute.hp_gamma_vec = arma::ones<arma::vec>(n_dim);
    precompute.mean_model.degree_vec = arma::zeros<arma::uvec>(n_dim);
    precompute.hp_beta_vec = arma::ones<arma::vec>(1);
    precompute.ydot_train_sq_norm_vec = arma::ones<arma::vec>(n_train);
    precompute.lin_sol_sigma_residual_vec = arma::ones<arma::vec>(n_train);
    precompute.Ydot_nfeature_ntrain_mat = arma::ones<arma::mat>(n_dim, n_train);
    arma::vec L_Sigma_packed_vec = cd::to_lower_packed_vec(arma::eye<arma::mat>(n_train, n_train));

    tamper(cache_info, precompute, L_Sigma_packed_vec);
    cd::save_cache_info(path, cache_info, "test");
    cd::apply_to_named_arrays(precompute, [&path](const char* array_name, const auto& array) {
        cd::save_array(path, array_name, array, "test");
    });
    // The header says the factor is in the file, so leaving it out is a file whose parts disagree.
    if (is_packed_factor_written) cd::save_array(path, cd::L_Sigma_packed_name, L_Sigma_packed_vec, "test");
}

// A hand-made file with a foreign format_version, the version save_precompute never writes.
void write_file_with_version(const std::string& path, const int version) {
    write_hand_made_file(
            path, [version](cache::CacheInfo& cache_info, gpflash::detail::PosteriorPrecompute&, arma::vec&) {
                cache_info.format_version = version;
            });
}

// -----------------------------------------------------------------------
// Test 4: a foreign format_version is rejected, not silently reinterpreted.
bool test_bad_format_version_is_rejected() {
    const std::string path = "test_cache_bad_version.h5";
    write_file_with_version(path, /*version=*/999);

    const bool threw = throws_naming("load_posterior", [&] { load_posterior(path); });
    // peek_info refuses it too, and names the version that wrote it.
    const bool peek_names_version =
            calc_error_message([&] { cache::peek_info(path); }).find("version 999") != std::string::npos;
    gpf_fs::remove(path);

    // A later version that also changed the real header's length is still refused by its version, since the version
    // is read before either length.
    const std::string longer_path = "test_cache_bad_version_longer_header.h5";
    gpf_fs::remove(longer_path);
    arma::u64_vec integer_header_vec(cache::n_cache_integer_slots, arma::fill::zeros);
    integer_header_vec(cache::slot_format_version) = 999;
    const arma::vec longer_real_header_vec(cache::n_cache_real_slots + 1, arma::fill::ones);
    cache::detail::save_array(longer_path, cache::detail::header_u64_name, integer_header_vec, "test");
    cache::detail::save_array(longer_path, cache::detail::header_real_name, longer_real_header_vec, "test");
    const bool longer_names_version =
            calc_error_message([&] { cache::peek_info(longer_path); }).find("version 999") != std::string::npos;
    gpf_fs::remove(longer_path);

    // The same for a longer integer header, the one a new slot lengthens.
    const std::string longer_integer_path = "test_cache_bad_version_longer_integer_header.h5";
    gpf_fs::remove(longer_integer_path);
    arma::u64_vec longer_integer_header_vec(cache::n_cache_integer_slots + 1, arma::fill::zeros);
    longer_integer_header_vec(cache::slot_format_version) = 999;
    const arma::vec real_header_vec(cache::n_cache_real_slots, arma::fill::ones);
    cache::detail::save_array(longer_integer_path, cache::detail::header_u64_name, longer_integer_header_vec, "test");
    cache::detail::save_array(longer_integer_path, cache::detail::header_real_name, real_header_vec, "test");
    const bool longer_integer_names_version =
            calc_error_message([&] { cache::peek_info(longer_integer_path); }).find("version 999") != std::string::npos;
    gpf_fs::remove(longer_integer_path);

    // An integer header with no slot 0 at all is refused as empty rather than indexed.
    const std::string empty_path = "test_cache_empty_integer_header.h5";
    gpf_fs::remove(empty_path);
    cache::detail::save_array(empty_path, cache::detail::header_u64_name, arma::u64_vec(), "test");
    cache::detail::save_array(empty_path, cache::detail::header_real_name, real_header_vec, "test");
    const bool empty_refused =
            calc_error_message([&] { cache::peek_info(empty_path); }).find("is empty") != std::string::npos;
    gpf_fs::remove(empty_path);

    const bool pass =
            threw && peek_names_version && longer_names_version && longer_integer_names_version && empty_refused;
    report("bad_format_version_is_rejected", pass);
    return pass;
}

// -----------------------------------------------------------------------
// Test 5: a truncated file (a crash mid-write, a copy cut short) throws cleanly, with no partial read.
bool test_truncated_file_is_rejected() {
    GpPosterior posterior = make_fitted_posterior();
    const std::string path = "test_cache_truncated.h5";
    posterior.save_precompute(path);

    const auto full_size = gpf_fs::file_size(path);
    gpf_fs::resize_file(path, full_size / 2);  // chop off the back half

    const bool threw = throws_naming("load_posterior", [&] { load_posterior(path); });
    gpf_fs::remove(path);
    report("truncated_file_is_rejected", threw);

    return threw;
}

// -----------------------------------------------------------------------
// Test 6: loading a path that does not exist is reported as a missing file, not as a failed read of its first array.
bool test_missing_file_is_rejected() {
    const std::string error_message = calc_error_message([&] { load_posterior("test_cache_does_not_exist_12345.h5"); });
    const bool is_reported_missing =
            error_message.find("gpflash::cache::load_posterior: cannot open") != std::string::npos;
    report("missing_file_is_rejected", is_reported_missing);

    return is_reported_missing;
}

// -----------------------------------------------------------------------
// Test 7: the verifying load accepts the inputs the file was built from and rejects inputs that differ, the
// check load_posterior skips.
bool test_input_hash_detects_stale_cache() {
    const std::string path = "test_cache_hash.h5";
    const FitInputs fit_inputs = make_fit_inputs();
    make_posterior(fit_inputs.train_data, fit_inputs.hp_val_struct, fit_inputs.model_opts).save_precompute(path);

    bool pass = true;

    // Matching inputs: accepted, and equivalent to the unchecked load.
    bool threw = false;
    try {
        GpPosterior verified_posterior =
                load_posterior_verified(path, fit_inputs.train_data, fit_inputs.hp_val_struct, fit_inputs.model_opts);
        const arma::vec x_post_vec =
                fit_inputs.train_data.X_ndim_ntrain_mat.col(0);  // materialized: eval() is overloaded on vec vs mat
        pass &=
                close(verified_posterior.eval(x_post_vec, kOptsMeanOnly).mean_post_vec,
                      load_posterior(path).eval(x_post_vec, kOptsMeanOnly).mean_post_vec);
    } catch (const std::exception&) {
        threw = true;
    }
    pass &= !threw;
    report("input_hash/matching_accepted", !threw);

    // One perturbed training point: rejected.
    TrainingData train_data_changed_x = fit_inputs.train_data;
    train_data_changed_x.X_ndim_ntrain_mat(0, 0) += 1e-9;
    const bool changed_x_rejected = is_stale_refusal([&] {
        load_posterior_verified(path, train_data_changed_x, fit_inputs.hp_val_struct, fit_inputs.model_opts);
    });
    pass &= changed_x_rejected;
    report("input_hash/changed_point_rejected", changed_x_rejected);

    // Changed values at the same points: rejected, since they change the weights and the residual.
    TrainingData train_data_changed_f = fit_inputs.train_data;
    train_data_changed_f.f_train_vec(0) += 1e-9;
    const bool changed_f_train_rejected = is_stale_refusal([&] {
        load_posterior_verified(path, train_data_changed_f, fit_inputs.hp_val_struct, fit_inputs.model_opts);
    });
    pass &= changed_f_train_rejected;
    report("input_hash/changed_f_train_rejected", changed_f_train_rejected);

    // Changed hyperparameters: rejected.
    HpVal hp_changed_struct = fit_inputs.hp_val_struct;
    hp_changed_struct.hp_gamma_vec(0) *= 2.0;
    const bool changed_hp_rejected = is_stale_refusal(
            [&] { load_posterior_verified(path, fit_inputs.train_data, hp_changed_struct, fit_inputs.model_opts); });
    pass &= changed_hp_rejected;
    report("input_hash/changed_hp_rejected", changed_hp_rejected);

    // A changed signal variance: rejected, because it scales K.
    HpVal hp_changed_sig2_k_struct = fit_inputs.hp_val_struct;
    hp_changed_sig2_k_struct.hp_sig2_k *= 2.0;
    const bool changed_sig2_k_rejected = is_stale_refusal([&] {
        load_posterior_verified(path, fit_inputs.train_data, hp_changed_sig2_k_struct, fit_inputs.model_opts);
    });
    pass &= changed_sig2_k_rejected;
    report("input_hash/changed_sig2_k_rejected", changed_sig2_k_rejected);

    // Changed noise: rejected, because it changes Sigma.
    TrainingData train_data_changed_noise = fit_inputs.train_data;
    train_data_changed_noise.sig2_f_val *= 2.0;
    const bool changed_noise_rejected = is_stale_refusal([&] {
        load_posterior_verified(path, train_data_changed_noise, fit_inputs.hp_val_struct, fit_inputs.model_opts);
    });
    pass &= changed_noise_rejected;
    report("input_hash/changed_noise_rejected", changed_noise_rejected);

    // A changed conditioning limit: rejected, because it sizes both nuggets.
    ModelOptions model_opts_changed_cond_max = fit_inputs.model_opts;
    model_opts_changed_cond_max.cond_max *= 0.5;
    const bool changed_cond_max_rejected = is_stale_refusal([&] {
        load_posterior_verified(path, fit_inputs.train_data, fit_inputs.hp_val_struct, model_opts_changed_cond_max);
    });
    pass &= changed_cond_max_rejected;
    report("input_hash/changed_cond_max_rejected", changed_cond_max_rejected);

    // A changed eig_max bound, on either matrix: rejected, because the hash covers both, even the mean model's,
    // which sizes no nugget for this one-weight mean model.
    ModelOptions model_opts_changed_bound = fit_inputs.model_opts;
    model_opts_changed_bound.kmat_eig_max_ub_method = gpflash::EigMaxUbMethod::Trace;
    ModelOptions model_opts_changed_mean_model_bound = fit_inputs.model_opts;
    model_opts_changed_mean_model_bound.mean_model_eig_max_ub_method = gpflash::EigMaxUbMethod::Gershgorin;
    const bool changed_bound_rejected =
            is_stale_refusal([&] {
                load_posterior_verified(
                        path, fit_inputs.train_data, fit_inputs.hp_val_struct, model_opts_changed_bound);
            })
            && is_stale_refusal([&] {
                   load_posterior_verified(
                           path, fit_inputs.train_data, fit_inputs.hp_val_struct, model_opts_changed_mean_model_bound);
               });
    pass &= changed_bound_rejected;
    report("input_hash/changed_bound_rejected", changed_bound_rejected);

    // A changed mean model: rejected, since it changes the weights and the residual.
    ModelOptions model_opts_zero_mean_model = fit_inputs.model_opts;
    model_opts_zero_mean_model.has_mean_model_constant = false;
    ModelOptions model_opts_degree_1_mean_model = fit_inputs.model_opts;
    model_opts_degree_1_mean_model.mean_model_degree_vec = {1};
    const bool changed_mean_model_rejected =
            is_stale_refusal([&] {
                load_posterior_verified(
                        path, fit_inputs.train_data, fit_inputs.hp_val_struct, model_opts_zero_mean_model);
            })
            && is_stale_refusal([&] {
                   load_posterior_verified(
                           path, fit_inputs.train_data, fit_inputs.hp_val_struct, model_opts_degree_1_mean_model);
               });
    pass &= changed_mean_model_rejected;
    report("input_hash/changed_mean_model_rejected", changed_mean_model_rejected);

    // A changed kernel extra hyperparameter: rejected. SqExp has none, so this fits RatQuad, whose own inputs must
    // load first so the refusal is the extra's.
    const std::string rat_quad_path = "test_cache_hash_rat_quad.h5";
    HpVal hp_rat_quad_struct = fit_inputs.hp_val_struct;
    hp_rat_quad_struct.hp_kern_extra_vec = {gpflash_test::kRatQuadHpRho};
    ModelOptions model_opts_rat_quad = fit_inputs.model_opts;
    model_opts_rat_quad.kernel_id = gpflash::kernel::KernelId::RatQuad;
    make_posterior(fit_inputs.train_data, hp_rat_quad_struct, model_opts_rat_quad).save_precompute(rat_quad_path);
    HpVal hp_changed_kern_extra_struct = hp_rat_quad_struct;
    hp_changed_kern_extra_struct.hp_kern_extra_vec(0) *= 2.0;
    const bool rat_quad_matching_accepted = gpflash_test::accepts([&] {
        load_posterior_verified(rat_quad_path, fit_inputs.train_data, hp_rat_quad_struct, model_opts_rat_quad);
    });
    const bool changed_kern_extra_rejected =
            rat_quad_matching_accepted && is_stale_refusal([&] {
                load_posterior_verified(
                        rat_quad_path, fit_inputs.train_data, hp_changed_kern_extra_struct, model_opts_rat_quad);
            });
    gpf_fs::remove(rat_quad_path);
    pass &= changed_kern_extra_rejected;
    report("input_hash/changed_kern_extra_rejected", changed_kern_extra_rejected);

    // The same degrees written per dimension rather than broadcast: accepted, since it is the same model.
    ModelOptions model_opts_per_dim = fit_inputs.model_opts;
    model_opts_per_dim.mean_model_degree_vec = arma::zeros<arma::uvec>(fit_inputs.train_data.X_ndim_ntrain_mat.n_rows);
    const bool per_dimension_degrees_accepted = !throws([&] {
        load_posterior_verified(path, fit_inputs.train_data, fit_inputs.hp_val_struct, model_opts_per_dim);
    });
    pass &= per_dimension_degrees_accepted;
    report("input_hash/per_dimension_degrees_accepted", per_dimension_degrees_accepted);

    gpf_fs::remove(path);
    return pass;
}

// -----------------------------------------------------------------------
// Test 8: version and note come back through peek_info without loading the arrays. An absent note reads back as
// an empty string.
bool test_version_and_note_round_trip() {
    GpPosterior posterior = make_fitted_posterior();
    const std::string path = "test_cache_info.h5";
    const std::string note = "fitted from wt_run_42.h5 on 2026-08-11";

    const arma::vec x_post_vec(posterior.n_dim(), arma::fill::ones);
    posterior.save_precompute(path, note);
    cache::CacheInfo with_note = cache::peek_info(path);
    const double mean_with_note = load_posterior(path).eval(x_post_vec, kOptsMeanOnly).mean_post_vec(0);
    gpf_fs::remove(path);

    posterior.save_precompute(path);  // no note
    cache::CacheInfo no_note = cache::peek_info(path);
    const double mean_no_note = load_posterior(path).eval(x_post_vec, kOptsMeanOnly).mean_post_vec(0);
    gpf_fs::remove(path);

    const bool pass = with_note.note == note && no_note.note.empty() && with_note.version_major == GPFLASH_VERSION_MAJOR
            && with_note.version_minor == GPFLASH_VERSION_MINOR && with_note.version_patch == GPFLASH_VERSION_PATCH
            && with_note.format_version == cache::cache_format_version
            && with_note.post_model_info.kernel_id == gpflash::kernel::KernelId::SqExp
            // A note leaves the loaded arrays unchanged.
            && close(mean_with_note, mean_no_note);
    report("version_and_note_round_trip", pass);
    return pass;
}

// -----------------------------------------------------------------------
// Test 9: n_dim (the query point's dimension) and n_feature (the rows the kernel evaluates on) stay
// separate through a save/load, driven through the format directly with the two distinct; a load that
// conflated them would fail here.
bool test_feature_dim_differs_from_input_dim() {
    const std::string path = "test_cache_feature_dim.h5";
    const int n_dim = 3, n_feature = 6, n_train = 5;  // 6 != 3, as a cosine/sine map would give
    std::mt19937_64 rng(13);

    gpflash::detail::PosteriorPrecompute written;
    written.n_dim = n_dim;
    written.n_feature = n_feature;
    written.n_train = n_train;
    written.Ydot_nfeature_ntrain_mat = draw_uniform_mat(rng, n_feature, n_train);
    written.Ydot_ntrain_nfeature_mat = written.Ydot_nfeature_ntrain_mat.t();
    written.ydot_train_sq_norm_vec = arma::sum(arma::square(written.Ydot_nfeature_ntrain_mat), 0).t();
    written.lin_sol_sigma_residual_vec = draw_uniform_vec(rng, n_train);
    written.L_Sigma_mat_lower = arma::eye<arma::mat>(n_train, n_train);
    written.x_center_vec = arma::zeros<arma::vec>(n_dim);  // one entry per input dimension
    written.hp_gamma_vec = arma::ones<arma::vec>(n_dim);
    written.hp_kern_extra_vec = arma::ones<arma::vec>(n_dim);
    written.mean_model.degree_vec = arma::zeros<arma::uvec>(n_dim);  // the constant mean model, one weight
    written.hp_beta_vec = {0.25};
    written.hp_sig2_k = 1.3;

    gpflash::cache::PosteriorModelInfo written_model_info;
    written_model_info.kernel_id = gpflash::kernel::KernelId::SqExp;
    written_model_info.input_hash = 12345;
    written_model_info.mll = -42.5;
    written_model_info.sig2_f_val = 0.013;
    written_model_info.cond_max = 1e10;
    written_model_info.kmat_eig_max_ub_method = gpflash::EigMaxUbMethod::Gershgorin;
    written_model_info.mean_model_eig_max_ub_method = gpflash::EigMaxUbMethod::Trace;

    cache::save_precompute(path, written_model_info, written, "feature_dim");
    const cache::PosteriorContents post_contents = cache::load_precompute(path, gpflash::kernel::KernelId::SqExp);
    const gpflash::detail::PosteriorPrecompute& loaded_precompute = post_contents.post_precompute;

    // The shape check accepted the file, so hp_gamma was measured against n_dim and not n_feature.
    const bool n_dim_kept = loaded_precompute.n_dim == n_dim;
    const bool n_feature_kept = loaded_precompute.n_feature == n_feature;
    const bool transpose_rebuilt =
            loaded_precompute.Ydot_ntrain_nfeature_mat.n_rows == static_cast<arma::uword>(n_train)
            && loaded_precompute.Ydot_ntrain_nfeature_mat.n_cols == static_cast<arma::uword>(n_feature);
    // The MLL is stored as a double, so it must survive a round trip bit for bit.
    const bool mll_kept = post_contents.post_model_info.mll == written_model_info.mll
            && cache::peek_info(path).post_model_info.mll == written_model_info.mll;
    const bool pass = n_dim_kept && n_feature_kept && transpose_rebuilt && mll_kept;

    char detail[gpflash_test::kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "n_dim %d->%d  n_feature %d->%d",
            n_dim,
            loaded_precompute.n_dim,
            n_feature,
            loaded_precompute.n_feature);
    report("feature_dim_differs_from_input_dim", pass, detail);
    gpf_fs::remove(path);
    return pass;
}

// -----------------------------------------------------------------------
// Test 10: the MLL the posterior stores equals the one eval_mll reports at the same inputs, and survives a
// save/load. This is the cross-tier check: the two tiers factorize Sigma independently and read the
// likelihood off their own factor, so agreement means both the formula and the 2 pi convention match.
bool test_mll_matches_eval_mll() {
    const FitInputs fit_inputs = make_fit_inputs();
    const gpflash::MllResult mll_result =
            gpflash::eval_mll(fit_inputs.train_data, fit_inputs.hp_val_struct, fit_inputs.model_opts);

    auto posterior = make_posterior(fit_inputs.train_data, fit_inputs.hp_val_struct, fit_inputs.model_opts);
    const std::string path = "test_cache_mll.h5";
    posterior.save_precompute(path);
    const double mll_peeked = cache::peek_info(path).post_model_info.mll;
    const double mll_loaded = load_posterior(path).mll();
    gpf_fs::remove(path);

    const double scale = std::max(1.0, std::fabs(mll_result.mll));
    const bool matches_eval_mll = std::fabs(posterior.mll() - mll_result.mll) / scale < kTolAltPath;
    const bool round_trips = mll_peeked == posterior.mll() && mll_loaded == posterior.mll();

    char detail[gpflash_test::kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "posterior %.6f  eval_mll %.6f  round trip %d",
            posterior.mll(),
            mll_result.mll,
            round_trips ? 1 : 0);
    const bool pass = matches_eval_mll && round_trips;
    report("mll_matches_eval_mll", pass, detail);

    return pass;
}

// -----------------------------------------------------------------------
// Test 11: a file saved with include_variance_data cleared omits L_Sigma, so it must still reproduce the mean
// and the mean gradient, be smaller by at least the packed factor's bytes, and report through peek_info what it holds.
bool test_mean_only_roundtrip_matches_mean() {
    GpPosterior posterior = make_fitted_posterior();
    const std::string path = "test_cache_full.h5";
    const std::string mean_path = "test_cache_mean_only.h5";
    posterior.save_precompute(path);
    posterior.save_precompute(mean_path, "", /*include_variance_data=*/false);

    // Both kinds say what they hold, readable without touching the arrays.
    const bool contents_reported =
            cache::peek_info(path).has_variance_data && !cache::peek_info(mean_path).has_variance_data;
    // What the mean-only file drops is exactly L_Sigma's packed lower triangle, n_train (n_train + 1) / 2 doubles,
    // so the sizes must differ by at least those bytes. A difference rather than a ratio: HDF5 spends order a
    // kilobyte per dataset, which at this n_train is the larger part of either file, so a ratio would measure that
    // floor instead.
    const std::streamsize n_train = static_cast<std::streamsize>(posterior.n_train());
    const std::streamsize factor_bytes = n_train * (n_train + 1) / 2 * static_cast<std::streamsize>(sizeof(double));
    const bool factor_accounts_for_difference = gpf_fs::file_size(path) - gpf_fs::file_size(mean_path) >= factor_bytes;

    GpPosterior mean_only_posterior = load_posterior(mean_path);

    std::mt19937_64 rng(11);
    arma::vec x_post_vec(posterior.n_dim());
    for (auto& v : x_post_vec) v = draw_normal(rng);
    arma::mat X_ndim_npost_mat(posterior.n_dim(), 5);
    for (auto& v : X_ndim_npost_mat) v = draw_normal(rng);

    // Everything a mean-only file can serve: the mean and its gradient, at both shapes.
    const gpflash::PosteriorOptions opts_mean_grad{
            /*return_mean=*/true, /*return_var=*/false, /*return_mean_gradient=*/true, /*return_var_gradient=*/false};
    const gpflash::PosteriorResult full_single = posterior.eval(x_post_vec, opts_mean_grad);
    const gpflash::PosteriorResult mean_single = mean_only_posterior.eval(x_post_vec, opts_mean_grad);
    const gpflash::PosteriorResult full_batch = posterior.eval(X_ndim_npost_mat, opts_mean_grad);
    const gpflash::PosteriorResult mean_batch = mean_only_posterior.eval(X_ndim_npost_mat, opts_mean_grad);
    gpf_fs::remove(path);
    gpf_fs::remove(mean_path);

    const bool predictions_match = close(full_single.mean_post_vec, mean_single.mean_post_vec)
            && close(full_single.d_mean_post_d_x_post_ndim_npost_mat, mean_single.d_mean_post_d_x_post_ndim_npost_mat)
            && close(full_batch.mean_post_vec, mean_batch.mean_post_vec)
            && close(full_batch.d_mean_post_d_x_post_ndim_npost_mat, mean_batch.d_mean_post_d_x_post_ndim_npost_mat);

    const bool pass = contents_reported && factor_accounts_for_difference && predictions_match;
    char detail[gpflash_test::kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "contents %d  factor_dropped %d  predictions %d",
            contents_reported ? 1 : 0,
            factor_accounts_for_difference ? 1 : 0,
            predictions_match ? 1 : 0);
    report("mean_only_roundtrip_matches_mean", pass, detail);

    return pass;
}

// -----------------------------------------------------------------------
// Test 12: either variance quantity needs the factor a mean-only file does not store, so both must be
// refused up front rather than reaching the solve with an empty L.
bool test_mean_only_refuses_variance() {
    GpPosterior posterior = make_fitted_posterior();
    const std::string path = "test_cache_mean_only_refuses.h5";
    posterior.save_precompute(path, "", /*include_variance_data=*/false);
    GpPosterior mean_only_posterior = load_posterior(path);
    gpf_fs::remove(path);

    arma::vec x_post_vec = arma::ones<arma::vec>(posterior.n_dim());
    arma::mat X_ndim_npost_mat = arma::ones<arma::mat>(posterior.n_dim(), 3);
    const gpflash::PosteriorOptions opts_var{
            /*return_mean=*/true, /*return_var=*/true, /*return_mean_gradient=*/false, /*return_var_gradient=*/false};
    const gpflash::PosteriorOptions opts_var_grad{
            /*return_mean=*/true, /*return_var=*/false, /*return_mean_gradient=*/false, /*return_var_gradient=*/true};

    const bool var_throws = gpflash_test::rejects([&] { mean_only_posterior.eval(x_post_vec, opts_var); })
            && gpflash_test::rejects([&] { mean_only_posterior.eval(X_ndim_npost_mat, opts_var); });
    const bool var_grad_throws = gpflash_test::rejects([&] { mean_only_posterior.eval(x_post_vec, opts_var_grad); })
            && gpflash_test::rejects([&] { mean_only_posterior.eval(X_ndim_npost_mat, opts_var_grad); });
    // The mean still works, so the refusal is scoped to the variance rather than to the file.
    const bool mean_still_works = !throws([&] { mean_only_posterior.eval(x_post_vec, kOptsMeanOnly); });

    const bool pass = var_throws && var_grad_throws && mean_still_works;
    char detail[gpflash_test::kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "var %d  var_grad %d  mean_ok %d",
            var_throws ? 1 : 0,
            var_grad_throws ? 1 : 0,
            mean_still_works ? 1 : 0);
    report("mean_only_refuses_variance", pass, detail);

    return pass;
}

// -----------------------------------------------------------------------
// Test 13: with include_variance_data cleared the file stores no factor. A fresh fit would carry one, so a
// variance request throwing is what proves the second posterior came from the file rather than a refit.
bool test_load_mean_only_file() {
    const FitInputs fit_inputs = make_fit_inputs();
    const std::string path = "test_cache_load_mean_only.h5";

    GpPosterior saved_posterior =
            make_posterior(fit_inputs.train_data, fit_inputs.hp_val_struct, fit_inputs.model_opts);
    saved_posterior.save_precompute(path, "", /*include_variance_data=*/false);
    const bool file_written = gpf_fs::file_size(path) > 0;

    GpPosterior loaded_posterior =
            load_posterior_verified(path, fit_inputs.train_data, fit_inputs.hp_val_struct, fit_inputs.model_opts);
    gpf_fs::remove(path);

    arma::vec x_post_vec = arma::ones<arma::vec>(fit_inputs.train_data.X_ndim_ntrain_mat.n_rows);
    const bool means_match =
            close(saved_posterior.eval(x_post_vec, kOptsMeanOnly).mean_post_vec,
                  loaded_posterior.eval(x_post_vec, kOptsMeanOnly).mean_post_vec);
    // Came from the file, not a refit.
    const bool loaded_not_refitted =
            gpflash_test::rejects([&] { loaded_posterior.eval(x_post_vec, gpflash::PosteriorOptions{}); });

    const bool pass = file_written && means_match && loaded_not_refitted;
    char detail[gpflash_test::kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "written %d  means_match %d  loaded %d",
            file_written ? 1 : 0,
            means_match ? 1 : 0,
            loaded_not_refitted ? 1 : 0);
    report("load_mean_only_file", pass, detail);

    return pass;
}

// -----------------------------------------------------------------------
// Test 14: save_precompute must throw, touching nothing, when the destination holds a file that is not a
// GpFlash cache -- the guard that keeps a mistyped path from overwriting unrelated data. The foreign file
// must survive byte for byte.
bool test_save_refuses_foreign_destination() {
    GpPosterior posterior = make_fitted_posterior();
    const std::string path = "test_cache_foreign.h5";
    const std::string foreign_content = "not a cache file";

    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out << foreign_content;
    }
    const bool refused = throws_naming("save_precompute", [&] { posterior.save_precompute(path); })
            && throws_naming("save_precompute",
                             [&] { posterior.save_precompute(path, "", /*include_variance_data=*/false); });
    std::ifstream in(path, std::ios::binary);
    const std::string read_back{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    const bool intact = read_back == foreign_content;
    gpf_fs::remove(path);

    const bool pass = refused && intact;
    char detail[gpflash_test::kDetailLen];
    std::snprintf(detail, sizeof(detail), "refused %d  intact %d", refused ? 1 : 0, intact ? 1 : 0);
    report("save_refuses_foreign_destination", pass, detail);

    return pass;
}

// -----------------------------------------------------------------------
// Test 19: a save whose writes fail part way through must raise, remove its .tmp scratch file and leave no file open.
// A file-size limit of half the full file lets the header arrays through and fails a later array. Skipped on
// Windows, which has no such limit, and on HDF5 1.10 (GPFLASH_HDF5_CLOSES_AFTER_FAILED_WRITE).
#if !defined(_WIN32) && GPFLASH_HDF5_CLOSES_AFTER_FAILED_WRITE
bool test_failed_save_removes_scratch_file() {
    const GpPosterior posterior = make_fitted_posterior();
    const std::string path = "test_cache_failed_save.h5";
    const std::string temp_path = path + ".tmp";
    gpf_fs::remove(path);
    posterior.save_precompute(path);
    const std::streamsize full_file_size = gpf_fs::file_size(path);
    gpf_fs::remove(path);

    // Ignored, so a write past the limit fails with EFBIG instead of ending the process.
    const auto previous_handler = std::signal(SIGXFSZ, SIG_IGN);
    rlimit original_limit;
    getrlimit(RLIMIT_FSIZE, &original_limit);
    rlimit half_file_limit = original_limit;
    half_file_limit.rlim_cur = static_cast<rlim_t>(full_file_size / 2);
    setrlimit(RLIMIT_FSIZE, &half_file_limit);
    const bool raised = throws_naming("save_precompute", [&] { posterior.save_precompute(path); });
    setrlimit(RLIMIT_FSIZE, &original_limit);
    std::signal(SIGXFSZ, previous_handler);

    const bool scratch_removed = !std::ifstream(temp_path).good();
    const bool destination_absent = !std::ifstream(path).good();
    const bool files_closed = H5Fget_obj_count(H5F_OBJ_ALL, H5F_OBJ_FILE) == 0;
    gpf_fs::remove(temp_path);
    gpf_fs::remove(path);

    const bool pass = raised && scratch_removed && destination_absent && files_closed;
    char detail[gpflash_test::kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "raised %d  scratch_removed %d  destination_absent %d  files_closed %d",
            raised ? 1 : 0,
            scratch_removed ? 1 : 0,
            destination_absent ? 1 : 0,
            files_closed ? 1 : 0);
    report("failed_save_removes_scratch_file", pass, detail);

    return pass;
}
#endif

// -----------------------------------------------------------------------
// Test 15: a file built from other inputs must be reported, and reporting must leave it alone -- so the caller
// can look at what it complained about. Saving over it is how a caller replaces one. The accepted load must
// also carry the fit's kernel through, since queries dispatch on it.
bool test_verified_raises_on_stale_file() {
    const FitInputs fit_inputs = make_fit_inputs();
    const std::string path = "test_cache_load_stale.h5";
    gpf_fs::remove(path);

    make_posterior(fit_inputs.train_data, fit_inputs.hp_val_struct, fit_inputs.model_opts).save_precompute(path);
    const bool file_written = gpf_fs::file_size(path) > 0;

    // The fit's own inputs: accepted, carrying the kernel it was asked for.
    bool hit_is_silent = true, kernel_kept = false;
    try {
        const GpPosterior posterior_reloaded =
                load_posterior_verified(path, fit_inputs.train_data, fit_inputs.hp_val_struct, fit_inputs.model_opts);
        kernel_kept = posterior_reloaded.kernel_id() == fit_inputs.model_opts.kernel_id;
    } catch (...) {
        hit_is_silent = false;
    }

    // Different inputs: the hash no longer matches, so this must raise.
    FitInputs changed = make_fit_inputs();
    changed.train_data.X_ndim_ntrain_mat(0, 0) += 1.0;
    const bool stale_raises = throws_naming("load_posterior_verified", [&] {
        load_posterior_verified(path, changed.train_data, changed.hp_val_struct, changed.model_opts);
    });
    const bool file_intact = gpf_fs::file_size(path) > 0;

    // A save replaces it, and the replacement answers to the inputs that wrote it.
    make_posterior(changed.train_data, changed.hp_val_struct, changed.model_opts).save_precompute(path);
    const bool replacement_loads = !throws(
            [&] { load_posterior_verified(path, changed.train_data, changed.hp_val_struct, changed.model_opts); });
    gpf_fs::remove(path);

    const bool pass = file_written && hit_is_silent && kernel_kept && stale_raises && file_intact && replacement_loads;
    char detail[gpflash_test::kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "written %d  hit_silent %d  kernel_kept %d  stale_raises %d  intact %d  replaced %d",
            file_written ? 1 : 0,
            hit_is_silent ? 1 : 0,
            kernel_kept ? 1 : 0,
            stale_raises ? 1 : 0,
            file_intact ? 1 : 0,
            replacement_loads ? 1 : 0);
    report("verified_raises_on_stale_file", pass, detail);

    return pass;
}

// -----------------------------------------------------------------------
// Test 16: the model info survives load-then-save. Which fit produced a posterior is a separate struct from its
// arrays, so every construction path has to carry it; one that dropped it would write a second file whose header
// reads mll = 0 and input_hash = 0 while the arrays are perfectly good. The predictions would still match, and
// load_posterior would still load the file.
bool test_model_info_survives_resave() {
    const std::string first_path = "test_cache_model_info_first.h5";
    const std::string resaved_path = "test_cache_model_info_resaved.h5";

    const FitInputs fit_inputs = make_fit_inputs();
    make_posterior(fit_inputs.train_data, fit_inputs.hp_val_struct, fit_inputs.model_opts)
            .save_precompute(first_path, "original");
    const cache::CacheInfo first_info = cache::peek_info(first_path);

    // Load the file, save it again, and compare the two headers.
    load_posterior(first_path).save_precompute(resaved_path, "resaved");
    const cache::CacheInfo resaved_info = cache::peek_info(resaved_path);

    // Bit-for-bit: these are copied, never recomputed, so anything but equality means a path dropped them.
    const cache::PosteriorModelInfo& first = first_info.post_model_info;
    const cache::PosteriorModelInfo& resaved = resaved_info.post_model_info;
    const bool model_info_kept = resaved.kernel_id == first.kernel_id && resaved.input_hash == first.input_hash
            && resaved.mll == first.mll && resaved.sig2_f_val == first.sig2_f_val && resaved.cond_max == first.cond_max;
    // The zero a dropped fit info would leave behind, named so the test fails for the right reason.
    const bool not_zeroed = resaved.input_hash != 0 && resaved.mll != 0.0;
    // The note is per-save, so it must not be carried over from the first file.
    const bool note_is_per_save = resaved_info.note == "resaved" && first_info.note == "original";

    const bool pass = model_info_kept && not_zeroed && note_is_per_save;
    char detail[gpflash_test::kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "kept=%d nonzero=%d note=%d  mll %.6g->%.6g  hash 0x%llx->0x%llx",
            model_info_kept ? 1 : 0,
            not_zeroed ? 1 : 0,
            note_is_per_save ? 1 : 0,
            first.mll,
            resaved.mll,
            static_cast<unsigned long long>(first.input_hash),
            static_cast<unsigned long long>(resaved.input_hash));
    report("model_info_survives_resave", pass, detail);
    gpf_fs::remove(first_path);
    gpf_fs::remove(resaved_path);
    return pass;
}

// -----------------------------------------------------------------------
// Test 17: non-default bound methods reach the header and survive a resave. Non-default, because a default would
// also match a build that ignored the option.
bool test_bound_method_round_trip() {
    const std::string first_path = "test_cache_bound_first.h5";
    const std::string resaved_path = "test_cache_bound_resaved.h5";

    FitInputs fit_inputs = make_fit_inputs();
    fit_inputs.model_opts.kmat_eig_max_ub_method = gpflash::EigMaxUbMethod::Trace;
    fit_inputs.model_opts.mean_model_eig_max_ub_method = gpflash::EigMaxUbMethod::Gershgorin;
    make_posterior(fit_inputs.train_data, fit_inputs.hp_val_struct, fit_inputs.model_opts).save_precompute(first_path);
    load_posterior(first_path).save_precompute(resaved_path);

    const auto has_both_methods = [](const std::string& path) {
        const cache::PosteriorModelInfo post_model_info = cache::peek_info(path).post_model_info;
        return post_model_info.kmat_eig_max_ub_method == gpflash::EigMaxUbMethod::Trace
                && post_model_info.mean_model_eig_max_ub_method == gpflash::EigMaxUbMethod::Gershgorin;
    };
    const bool first_ok = has_both_methods(first_path);
    const bool resaved_ok = has_both_methods(resaved_path);
    bool verified_ok = true;
    try {
        load_posterior_verified(resaved_path, fit_inputs.train_data, fit_inputs.hp_val_struct, fit_inputs.model_opts);
    } catch (const std::exception&) {
        verified_ok = false;
    }

    const bool pass = first_ok && resaved_ok && verified_ok;
    char detail[gpflash_test::kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "first=%d resaved=%d verified=%d",
            static_cast<int>(first_ok),
            static_cast<int>(resaved_ok),
            static_cast<int>(verified_ok));
    report("bound_method_round_trip", pass, detail);

    gpf_fs::remove(first_path);
    gpf_fs::remove(resaved_path);
    return pass;
}

// -----------------------------------------------------------------------
// Test 18: the summary puts every field on its own labelled line, with the variance data and mean model constant
// lines in both states and the symmetry line for each kind.
bool test_summary_reports_mean_model_constant() {
    cache::CacheInfo cache_info;
    // The summary names the kernel and both bound methods.
    cache_info.post_model_info.kernel_id = gpflash::kernel::KernelId::SqExp;
    cache_info.post_model_info.kmat_eig_max_ub_method = gpflash::EigMaxUbMethod::Gershgorin;
    cache_info.post_model_info.mean_model_eig_max_ub_method = gpflash::EigMaxUbMethod::Trace;
    // Distinct values in every same-typed group, so a swap of two printf arguments of one type shows.
    cache_info.n_train = 40;
    cache_info.n_dim = 5;
    cache_info.n_feature = 6;
    cache_info.version_major = 1;
    cache_info.version_minor = 2;
    cache_info.version_patch = 3;
    cache_info.hp_sig2_k = 2.7;
    cache_info.post_model_info.mll = -1.5;
    cache_info.post_model_info.sig2_f_val = 0.25;
    cache_info.post_model_info.cond_max = 1e6;
    cache_info.post_model_info.input_hash = 0xabc;
    // The two yes/no lines in opposite states in each summary, so one flag printed from the other shows.
    cache_info.has_variance_data = false;
    cache_info.has_mean_model_constant = true;
    const std::string constant_summary_str = cache::to_summary_string(cache_info);
    cache_info.has_variance_data = true;
    cache_info.has_mean_model_constant = false;
    const std::string zero_summary_str = cache::to_summary_string(cache_info);
    // The block PosteriorModelInfo contributes: the kernel and the two bound methods, set to different values above
    // so a swap of the two method lines shows.
    const std::string kernel_line = std::string("  kernel             ")
            + gpflash::kernel::to_kernel_name(gpflash::kernel::KernelId::SqExp) + "\n";
    const std::string kmat_bound_line = std::string("  eig_max bound      ")
            + gpflash::to_eig_max_ub_method_name(gpflash::EigMaxUbMethod::Gershgorin) + "\n";
    const std::string mean_model_bound_line = std::string("  mean model bound   ")
            + gpflash::to_eig_max_ub_method_name(gpflash::EigMaxUbMethod::Trace) + "\n";
    const bool model_block_ok = constant_summary_str.find(kernel_line) != std::string::npos
            && constant_summary_str.find(kmat_bound_line) != std::string::npos
            && constant_summary_str.find(mean_model_bound_line) != std::string::npos
            && constant_summary_str.find("  marginal log lik   -1.5\n") != std::string::npos
            && constant_summary_str.find("  noise variance     0.25\n") != std::string::npos
            && constant_summary_str.find("  conditioning limit 1e+06\n") != std::string::npos
            && constant_summary_str.find("  input hash         0x0000000000000abc\n") != std::string::npos;
    const bool shape_ok = constant_summary_str.find("  training points     40\n") != std::string::npos
            && constant_summary_str.find("  input dimension     5\n") != std::string::npos
            && constant_summary_str.find("  feature dimension   6\n") != std::string::npos;
    const bool version_ok = constant_summary_str.find("  written by GpFlash  1.2.3\n") != std::string::npos;
    const bool signal_variance_ok = constant_summary_str.find("  signal variance     2.7\n") != std::string::npos;
    const bool variance_data_ok =
            zero_summary_str.find("  variance data       yes (mean, variance, both gradients)\n") != std::string::npos
            && constant_summary_str.find("  variance data       no (mean and its gradient only)\n")
                    != std::string::npos;
    const bool mean_model_constant_ok = constant_summary_str.find("  mean model constant yes\n") != std::string::npos
            && zero_summary_str.find("  mean model constant no\n") != std::string::npos;

    // The symmetry line names the kind the file records.
    cache_info.symmetry_kind = cache::SymmetryKind::Even;
    const std::string even_summary_str = cache::to_summary_string(cache_info);
    cache_info.symmetry_kind = cache::SymmetryKind::Odd;
    const std::string odd_summary_str = cache::to_summary_string(cache_info);
    cache_info.symmetry_kind = cache::SymmetryKind::None;
    const std::string none_summary_str = cache::to_summary_string(cache_info);
    const bool symmetry_named = even_summary_str.find("  symmetry            even\n") != std::string::npos
            && odd_summary_str.find("  symmetry            odd\n") != std::string::npos
            && none_summary_str.find("  symmetry            none\n") != std::string::npos;

    const bool pass = mean_model_constant_ok && model_block_ok && shape_ok && version_ok && signal_variance_ok
            && variance_data_ok && symmetry_named;
    char detail[gpflash_test::kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "constant %d  model %d  shape %d  version %d  sig2_k %d  variance_data %d  symmetry %d",
            mean_model_constant_ok ? 1 : 0,
            model_block_ok ? 1 : 0,
            shape_ok ? 1 : 0,
            version_ok ? 1 : 0,
            signal_variance_ok ? 1 : 0,
            variance_data_ok ? 1 : 0,
            symmetry_named ? 1 : 0);
    report("summary_reports_mean_model_constant", pass, detail);
    return pass;
}

// Test 20: the default path is the kernel and the training set's shape, with one separator after a directory.
bool test_default_cache_path_names_kernel_and_shape() {
    const FitInputs fit_inputs = make_fit_inputs();
    const gpflash::kernel::KernelId kernel_id = fit_inputs.model_opts.kernel_id;
    const std::string bare_path = cache::calc_default_cache_path(kernel_id, fit_inputs.train_data);
    const std::string dir_path = cache::calc_default_cache_path(kernel_id, fit_inputs.train_data, "out");
    const std::string slash_dir_path = cache::calc_default_cache_path(kernel_id, fit_inputs.train_data, "out/");

    // make_fit_inputs fits SqExp to 40 points in 3 dimensions.
    const std::string expected_file_name = "gpflash_sq_exp_ntrain40_ndim3.h5";
    const bool pass = bare_path == expected_file_name && dir_path == "out/" + expected_file_name
            && slash_dir_path == "out/" + expected_file_name;
    report("default_cache_path_names_kernel_and_shape", pass, bare_path.c_str());
    return pass;
}

// Test 21: each slot holds its own field on disk. The writer and the reader walk one slot list, so a slot paired with
// the wrong field would round-trip; writing by slot index here does not. The counts, versions, time, hash and reals
// differ from every other value, so a swap involving one shows, and the time is past 2^31, where an int round trip
// would truncate it.
//
// Five slots hold only 0 or 1 (format_version, the two flags and the two bound methods), so one header cannot give
// them distinct values. Three headers give each a distinct pattern across them, so a swap within that group changes
// at least one. The symmetry kind is Odd, 2, in all three, a value no other slot holds.
struct HeaderFlagCase {
    bool has_variance_data;
    gpflash::EigMaxUbMethod kmat_eig_max_ub_method;
    bool has_mean_model_constant;
    gpflash::EigMaxUbMethod mean_model_eig_max_ub_method;
};

bool is_header_slot_layout_pinned(const HeaderFlagCase& flag_case) {
    namespace cd = cache::detail;
    const std::string path = "test_cache_header_layout.h5";
    gpf_fs::remove(path);
    const arma::u64 saved_unix_time_s = arma::u64(1) << 40;
    const arma::u64 input_hash = 0xFEDCBA9876543210ull;

    arma::u64_vec integer_header_vec(cache::n_cache_integer_slots, arma::fill::zeros);
    integer_header_vec(cache::slot_format_version) = static_cast<arma::u64>(cache::cache_format_version);
    integer_header_vec(cache::slot_kernel_id) = static_cast<arma::u64>(gpflash::kernel::KernelId::RatQuad);
    integer_header_vec(cache::slot_n_dim) = 9;
    integer_header_vec(cache::slot_n_train) = 12;
    integer_header_vec(cache::slot_n_feature) = 10;
    integer_header_vec(cache::slot_version_major) = 13;
    integer_header_vec(cache::slot_version_minor) = 14;
    integer_header_vec(cache::slot_version_patch) = 15;
    integer_header_vec(cache::slot_saved_unix_time_s) = saved_unix_time_s;
    integer_header_vec(cache::slot_input_hash) = input_hash;
    integer_header_vec(cache::slot_has_variance_data) = flag_case.has_variance_data ? 1 : 0;
    integer_header_vec(cache::slot_kmat_eig_max_ub_method) = static_cast<arma::u64>(flag_case.kmat_eig_max_ub_method);
    integer_header_vec(cache::slot_has_mean_model_constant) = flag_case.has_mean_model_constant ? 1 : 0;
    integer_header_vec(cache::slot_mean_model_eig_max_ub_method) =
            static_cast<arma::u64>(flag_case.mean_model_eig_max_ub_method);
    integer_header_vec(cache::slot_symmetry_kind) = static_cast<arma::u64>(cache::SymmetryKind::Odd);
    arma::vec real_header_vec(cache::n_cache_real_slots, arma::fill::zeros);
    real_header_vec(cache::slot_hp_sig2_k) = 2.7;
    real_header_vec(cache::slot_mll) = -1.0;
    real_header_vec(cache::slot_sig2_f_val) = 0.013;
    real_header_vec(cache::slot_cond_max) = 1e10;
    cd::save_array(path, cd::header_u64_name, integer_header_vec, "test");
    cd::save_array(path, cd::header_real_name, real_header_vec, "test");
    cd::save_string(path, cd::note_name, "layout", "test");

    const cache::CacheInfo cache_info = cd::load_cache_info(path, "test");
    gpf_fs::remove(path);

    const cache::PosteriorModelInfo& post_model_info = cache_info.post_model_info;
    return cache_info.format_version == cache::cache_format_version
            && post_model_info.kernel_id == gpflash::kernel::KernelId::RatQuad && cache_info.n_dim == 9
            && cache_info.n_train == 12 && cache_info.n_feature == 10 && cache_info.version_major == 13
            && cache_info.version_minor == 14 && cache_info.version_patch == 15
            && cache_info.saved_unix_time_s == saved_unix_time_s && post_model_info.input_hash == input_hash
            && cache_info.has_variance_data == flag_case.has_variance_data
            && post_model_info.kmat_eig_max_ub_method == flag_case.kmat_eig_max_ub_method
            && cache_info.has_mean_model_constant == flag_case.has_mean_model_constant
            && post_model_info.mean_model_eig_max_ub_method == flag_case.mean_model_eig_max_ub_method
            && cache_info.symmetry_kind == cache::SymmetryKind::Odd && cache_info.hp_sig2_k == 2.7
            && post_model_info.mll == -1.0 && post_model_info.sig2_f_val == 0.013 && post_model_info.cond_max == 1e10
            && cache_info.note == "layout";
}

bool test_header_slot_layout_is_pinned() {
    using gpflash::EigMaxUbMethod;
    namespace cd = cache::detail;

    // Every declared slot visited exactly once by the list the writer and the reader walk, so a slot added to the
    // enum and not to apply_to_header_slots fails here instead of silently storing a default.
    arma::u64_vec integer_visit_count_vec(cache::n_cache_integer_slots, arma::fill::zeros);
    arma::u64_vec real_visit_count_vec(cache::n_cache_real_slots, arma::fill::zeros);
    cache::CacheInfo probe_info;
    cd::apply_to_header_slots(
            probe_info,
            [&integer_visit_count_vec](const cache::CacheIntegerIndex slot, const auto&) {
                ++integer_visit_count_vec(slot);
            },
            [&real_visit_count_vec](const cache::CacheRealIndex slot, const double) { ++real_visit_count_vec(slot); });
    const bool is_every_slot_visited_once = integer_visit_count_vec.min() == 1 && integer_visit_count_vec.max() == 1
            && real_visit_count_vec.min() == 1 && real_visit_count_vec.max() == 1;

    // The on-disk positions as literals, since the cases below index through the enum the reader also uses and so
    // survive a reorder. Slot 0 and the header_u64 name never change; any other position changes only with
    // cache_format_version.
    const bool is_integer_layout_pinned = cache::slot_format_version == 0 && cache::slot_kernel_id == 1
            && cache::slot_n_dim == 2 && cache::slot_n_train == 3 && cache::slot_n_feature == 4
            && cache::slot_version_major == 5 && cache::slot_version_minor == 6 && cache::slot_version_patch == 7
            && cache::slot_saved_unix_time_s == 8 && cache::slot_input_hash == 9 && cache::slot_has_variance_data == 10
            && cache::slot_kmat_eig_max_ub_method == 11 && cache::slot_has_mean_model_constant == 12
            && cache::slot_mean_model_eig_max_ub_method == 13 && cache::slot_symmetry_kind == 14
            && cache::n_cache_integer_slots == 15;
    const bool is_real_layout_pinned = cache::slot_hp_sig2_k == 0 && cache::slot_mll == 1 && cache::slot_sig2_f_val == 2
            && cache::slot_cond_max == 3 && cache::n_cache_real_slots == 4;
    const bool is_header_name_pinned = std::string(cd::header_u64_name) == "header_u64";

    // Across the three headers, with format_version 1 in each: format_version 111, has_variance_data 100,
    // kmat_eig_max_ub_method 010, has_mean_model_constant 001, mean_model_eig_max_ub_method 011 (Gershgorin is 0,
    // Trace is 1).
    const HeaderFlagCase flag_case_vec[] = {
            {/*has_variance_data=*/true,
             /*kmat_eig_max_ub_method=*/EigMaxUbMethod::Gershgorin,
             /*has_mean_model_constant=*/false,
             /*mean_model_eig_max_ub_method=*/EigMaxUbMethod::Gershgorin},
            {/*has_variance_data=*/false,
             /*kmat_eig_max_ub_method=*/EigMaxUbMethod::Trace,
             /*has_mean_model_constant=*/false,
             /*mean_model_eig_max_ub_method=*/EigMaxUbMethod::Trace},
            {/*has_variance_data=*/false,
             /*kmat_eig_max_ub_method=*/EigMaxUbMethod::Gershgorin,
             /*has_mean_model_constant=*/true,
             /*mean_model_eig_max_ub_method=*/EigMaxUbMethod::Trace},
    };
    const int n_case = static_cast<int>(sizeof(flag_case_vec) / sizeof(flag_case_vec[0]));
    // Every case runs, so the detail names the first that fails rather than stopping at it.
    int idx_first_bad_case = -1;
    for (int idx_case = 0; idx_case < n_case; ++idx_case) {
        const bool is_case_pinned = is_header_slot_layout_pinned(flag_case_vec[idx_case]);
        if (!is_case_pinned && idx_first_bad_case < 0) idx_first_bad_case = idx_case;
    }

    const bool pass = is_every_slot_visited_once && is_integer_layout_pinned && is_real_layout_pinned
            && is_header_name_pinned && idx_first_bad_case < 0;
    char detail[gpflash_test::kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "every_slot_visited_once=%d integer_layout=%d real_layout=%d header_name=%d n_case=%d first_bad_case=%d",
            static_cast<int>(is_every_slot_visited_once),
            static_cast<int>(is_integer_layout_pinned),
            static_cast<int>(is_real_layout_pinned),
            static_cast<int>(is_header_name_pinned),
            n_case,
            idx_first_bad_case);
    report("header_slot_layout_is_pinned", pass, detail);
    return pass;
}

// Test 22: a note of any length reaches the summary whole, and no note reads as "(none)".
bool test_summary_carries_whole_note() {
    cache::CacheInfo cache_info;
    cache_info.post_model_info.kernel_id = gpflash::kernel::KernelId::SqExp;
    cache_info.post_model_info.kmat_eig_max_ub_method = gpflash::EigMaxUbMethod::Gershgorin;
    cache_info.post_model_info.mean_model_eig_max_ub_method = gpflash::EigMaxUbMethod::Trace;
    const std::string no_note_summary_str = cache::to_summary_string(cache_info);
    // Longer than any fixed buffer the summary could be built in.
    cache_info.note = std::string(4096, 'n');
    const std::string summary_str = cache::to_summary_string(cache_info);

    const bool is_absent_note_none = no_note_summary_str.find("  note                (none)\n") != std::string::npos;
    const bool is_long_note_whole =
            summary_str.find("  note                " + cache_info.note + "\n") != std::string::npos;

    const bool pass = is_absent_note_none && is_long_note_whole;
    char detail[gpflash_test::kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "absent_note_none %d  long_note_whole %d",
            is_absent_note_none ? 1 : 0,
            is_long_note_whole ? 1 : 0);
    report("summary_carries_whole_note", pass, detail);
    return pass;
}

// Test 23: the written-at line is the saved time in UTC, and "unknown" for the unset value and for a time no
// calendar conversion can represent.
bool test_summary_written_at_is_utc_or_unknown() {
    cache::CacheInfo cache_info;
    cache_info.post_model_info.kernel_id = gpflash::kernel::KernelId::SqExp;
    cache_info.post_model_info.kmat_eig_max_ub_method = gpflash::EigMaxUbMethod::Gershgorin;
    cache_info.post_model_info.mean_model_eig_max_ub_method = gpflash::EigMaxUbMethod::Trace;

    cache_info.saved_unix_time_s = 0;
    const std::string unset_summary_str = cache::to_summary_string(cache_info);
    // Five days, three hours, four minutes and five seconds after the epoch: the six fields differ, so a swap of any
    // two of them, or a local-time conversion, shows.
    constexpr arma::u64 seconds_per_day = 86400;
    constexpr arma::u64 seconds_per_hour = 3600;
    constexpr arma::u64 seconds_per_minute = 60;
    cache_info.saved_unix_time_s = 5 * seconds_per_day + 3 * seconds_per_hour + 4 * seconds_per_minute + 5;
    const std::string set_summary_str = cache::to_summary_string(cache_info);
    // About year 3.2e10, past the int tm_year a conversion fills. Not the u64 maximum, which casts to time_t -1, a
    // valid time.
    cache_info.saved_unix_time_s = 1000000000000000000ull;
    const std::string unconvertible_summary_str = cache::to_summary_string(cache_info);

    const bool is_unset_unknown = unset_summary_str.find("  written at          unknown\n") != std::string::npos;
    const bool is_set_utc =
            set_summary_str.find("  written at          1970-01-06 03:04:05 UTC\n") != std::string::npos;
    const bool is_unconvertible_unknown =
            unconvertible_summary_str.find("  written at          unknown\n") != std::string::npos;

    const bool pass = is_unset_unknown && is_set_utc && is_unconvertible_unknown;
    char detail[gpflash_test::kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "unset_unknown %d  set_utc %d  unconvertible_unknown %d",
            is_unset_unknown ? 1 : 0,
            is_set_utc ? 1 : 0,
            is_unconvertible_unknown ? 1 : 0);
    report("summary_written_at_is_utc_or_unknown", pass, detail);
    return pass;
}

// Test 24: the file stores the factor's lower triangle packed column by column, and nothing above the diagonal.
bool test_factor_stored_packed() {
    const FitInputs fit_inputs = make_fit_inputs();
    const cache::PosteriorContents post_contents = detail::build_posterior_precompute(
            gpflash::kernel::kernel_by_id(fit_inputs.model_opts.kernel_id),
            fit_inputs.train_data,
            fit_inputs.hp_val_struct,
            fit_inputs.model_opts);
    const arma::mat& L_Sigma_mat_lower = post_contents.post_precompute.L_Sigma_mat_lower;
    const arma::uword n_train = L_Sigma_mat_lower.n_rows;

    const std::string path = "test_cache_packed_factor.h5";
    cache::save_precompute(path, post_contents.post_model_info, post_contents.post_precompute);
    arma::vec stored_packed_vec;
    const bool packed_read = stored_packed_vec.load(arma::hdf5_name(path, "L_Sigma_lower_packed_vec"));
    const bool has_full_matrix = cache::detail::has_array(path, "L_Sigma_mat_lower");
    gpf_fs::remove(path);

    // The column-major lower triangle, diagonal included, gathered here rather than through the library's packer.
    arma::vec expected_packed_vec(n_train * (n_train + 1) / 2);
    arma::uword i_expected = 0;
    for (arma::uword i_col = 0; i_col < n_train; ++i_col)
        for (arma::uword i_row = i_col; i_row < n_train; ++i_row)
            expected_packed_vec(i_expected++) = L_Sigma_mat_lower(i_row, i_col);
    const bool packed_matches = packed_read && stored_packed_vec.n_elem == expected_packed_vec.n_elem
            && arma::all(stored_packed_vec == expected_packed_vec);

    const bool pass = packed_matches && !has_full_matrix;
    char detail[gpflash_test::kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "packed %d of %d entries  full matrix stored %d",
            static_cast<int>(stored_packed_vec.n_elem),
            static_cast<int>(expected_packed_vec.n_elem),
            static_cast<int>(has_full_matrix));
    report("factor_stored_packed", pass, detail);
    return pass;
}
// Test 25: a file whose parts disagree is refused, by the checks that compare them. Each file differs from a valid
// hand-made one in one part, and the valid ones load, so each refusal is that part's.
bool test_disagreeing_parts_are_rejected() {
    using gpflash::detail::PosteriorPrecompute;
    const std::string path = "test_cache_disagreeing_parts.h5";
    const arma::vec mirror_sign_vec = {1.0, -1.0, 1.0};
    const auto loads = [&path](const auto& tamper) {
        write_hand_made_file(path, tamper);
        return gpflash_test::accepts([&path] { load_posterior(path); });
    };
    const auto is_refused = [&path](const auto& tamper) {
        write_hand_made_file(path, tamper);
        return gpflash_test::rejects([&path] { load_posterior(path); });
    };

    const bool plain_loads = loads([](cache::CacheInfo&, PosteriorPrecompute&, arma::vec&) {});
    const bool mirrored_loads = loads([&](cache::CacheInfo& cache_info, PosteriorPrecompute& precompute, arma::vec&) {
        cache_info.symmetry_kind = cache::SymmetryKind::Even;
        precompute.symmetry_spec.mirror_sign_vec = mirror_sign_vec;
    });
    const bool symmetry_without_mirror_refused =
            is_refused([](cache::CacheInfo& cache_info, PosteriorPrecompute&, arma::vec&) {
                cache_info.symmetry_kind = cache::SymmetryKind::Even;
            });
    const bool mirror_without_symmetry_refused =
            is_refused([&](cache::CacheInfo&, PosteriorPrecompute& precompute, arma::vec&) {
                precompute.symmetry_spec.mirror_sign_vec = mirror_sign_vec;
            });
    const bool mirror_short_refused =
            is_refused([](cache::CacheInfo& cache_info, PosteriorPrecompute& precompute, arma::vec&) {
                cache_info.symmetry_kind = cache::SymmetryKind::Even;
                precompute.symmetry_spec.mirror_sign_vec = {1.0, -1.0};
            });
    const bool packed_short_refused =
            is_refused([](cache::CacheInfo&, PosteriorPrecompute&, arma::vec& L_Sigma_packed_vec) {
                L_Sigma_packed_vec.resize(L_Sigma_packed_vec.n_elem - 1);
            });
    // The header's counts against the arrays' 5 points and 3 features, and each per-point vector one short of them.
    const bool stored_n_train_refused =
            is_refused([](cache::CacheInfo& cache_info, PosteriorPrecompute&, arma::vec&) { cache_info.n_train = 6; });
    const bool stored_n_feature_refused = is_refused(
            [](cache::CacheInfo& cache_info, PosteriorPrecompute&, arma::vec&) { cache_info.n_feature = 4; });
    const bool sq_norm_short_refused = is_refused([](cache::CacheInfo&, PosteriorPrecompute& precompute, arma::vec&) {
        precompute.ydot_train_sq_norm_vec = arma::ones<arma::vec>(4);
    });
    const bool residual_short_refused = is_refused([](cache::CacheInfo&, PosteriorPrecompute& precompute, arma::vec&) {
        precompute.lin_sol_sigma_residual_vec = arma::ones<arma::vec>(4);
    });
    // Refused by load_doubles_into's "cannot read", not its length-mismatch message; its no-dataset and failed-read
    // branches share that wording, so this pins the function, not the branch.
    write_hand_made_file(
            path, [](cache::CacheInfo&, PosteriorPrecompute&, arma::vec&) {}, /*is_packed_factor_written=*/false);
    const bool packed_missing_refused =
            calc_error_message([&path] { load_posterior(path); }).find("cannot read L_Sigma_lower_packed_vec")
            != std::string::npos;
    gpf_fs::remove(path);

    const bool pass = plain_loads && mirrored_loads && symmetry_without_mirror_refused
            && mirror_without_symmetry_refused && mirror_short_refused && packed_short_refused && packed_missing_refused
            && stored_n_train_refused && stored_n_feature_refused && sq_norm_short_refused && residual_short_refused;
    char detail[gpflash_test::kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "loads %d %d  refused %d %d %d %d %d  counts %d %d  per_point %d %d",
            plain_loads ? 1 : 0,
            mirrored_loads ? 1 : 0,
            symmetry_without_mirror_refused ? 1 : 0,
            mirror_without_symmetry_refused ? 1 : 0,
            mirror_short_refused ? 1 : 0,
            packed_short_refused ? 1 : 0,
            packed_missing_refused ? 1 : 0,
            stored_n_train_refused ? 1 : 0,
            stored_n_feature_refused ? 1 : 0,
            sq_norm_short_refused ? 1 : 0,
            residual_short_refused ? 1 : 0);
    report("disagreeing_parts_are_rejected", pass, detail);
    return pass;
}

}  // namespace

int main() {
    std::printf("== GpFlash posterior cache ==\n");
    bool ok = true;
    // The constant mean model alone cannot show that the mean model survived the round trip.
    ok &= test_roundtrip_predictions_match("roundtrip_predictions_match/constant", true, arma::uvec{0});
    ok &= test_roundtrip_predictions_match("roundtrip_predictions_match/zero", false, arma::uvec{0});
    ok &= test_roundtrip_predictions_match("roundtrip_predictions_match/{2,1,0}", true, arma::uvec{2, 1, 0});
    ok &= test_peek_kernel_id_reports_correct_kernel();
    ok &= test_kernel_mismatch_is_rejected();
    ok &= test_bad_format_version_is_rejected();
    ok &= test_truncated_file_is_rejected();
    ok &= test_missing_file_is_rejected();
    ok &= test_input_hash_detects_stale_cache();
    ok &= test_version_and_note_round_trip();
    ok &= test_feature_dim_differs_from_input_dim();
    ok &= test_mll_matches_eval_mll();
    std::printf("== Files saved without variance data ==\n");
    ok &= test_mean_only_roundtrip_matches_mean();
    ok &= test_mean_only_refuses_variance();
    ok &= test_load_mean_only_file();
    ok &= test_save_refuses_foreign_destination();
    ok &= test_verified_raises_on_stale_file();
    ok &= test_model_info_survives_resave();
    ok &= test_bound_method_round_trip();
    ok &= test_summary_reports_mean_model_constant();
#if defined(_WIN32)
    std::printf("  %-42s SKIPPED  no file-size limit on Windows\n", "failed_save_removes_scratch_file");
#elif !GPFLASH_HDF5_CLOSES_AFTER_FAILED_WRITE
    std::printf("  %-42s SKIPPED  needs HDF5 1.12 or later\n", "failed_save_removes_scratch_file");
#else
    ok &= test_failed_save_removes_scratch_file();
#endif
    ok &= test_default_cache_path_names_kernel_and_shape();
    ok &= test_header_slot_layout_is_pinned();
    ok &= test_summary_carries_whole_note();
    ok &= test_summary_written_at_is_utc_or_unknown();
    ok &= test_factor_stored_packed();
    ok &= test_disagreeing_parts_are_rejected();
    std::printf("\n%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");

    return ok ? 0 : 1;
}
