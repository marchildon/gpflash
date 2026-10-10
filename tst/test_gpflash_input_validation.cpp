// The validation tier: every check that runs before allocation or BLAS. What this file asserts:
//
//   1) training_data           empty X, a mismatched f_train length, and non-finite entries are rejected
//   2) hp_val                  a wrong length in either block, non-finite, and non-positive values are rejected
//   3) model_options           kernel_id must name a kernel; cond_max must be finite, above 1, within COND_MAX_LIMIT
//   4) mean_model_fields       a mean model degree_vec of the wrong length or past the cap is rejected, degrees
//                              with no constant are accepted, and a single degree is broadcast
//   5) mean_model_basis_columns  the basis holds the constant only when asked, then x_d^degree_d for each non-zero
//                              degree, in dimension order
//   6) mean_model_train_basis  a mean model basis wider than the training points, with a power that overflows, or
//                              with an all-zero column, is rejected; dependent columns are accepted
//   7) eval_variances          a negative or non-finite sig2_f_val is rejected
//   8) select_sig2_f_val       the noise sign rule, including the supplied-and-estimated conflict
//   9) supplied_hp_sig2_k      the unset sentinel and non-positive values are rejected
//  10) optz_options            bounds, optimization counts, tolerances, the algorithm, starting values, and
//                              per-component bound vectors; an inactive block's options are not read
//  11) resolved_hp_bound       the box a fit resolves from those options is ordered, per component
//  12) public_entry_points     eval_mll, the posterior, the fit and a verified load reject bad input rather than
//                              reading past an array
//  13) eig_max_ub_method       each bound method is accepted and named, for either matrix; End and an out-of-range
//                              value are rejected
//  14) post_points             a posterior query of the wrong dimension or with a non-finite coordinate is rejected,
//                              single-point and batch
//
// ARMA_NO_DEBUG removes Armadillo's own bounds checks, so a length that gets past this tier is an out-of-bounds
// read rather than an exception. Every case asserts the throw and its type: ErrorInvalidInput says a caller passed
// something impossible, ErrorNumericalFailure that the arithmetic failed on well-formed input.
#include <armadillo>
#include <cstdio>
#include <random>
#include <string>

#include "GpFlashTestSupport.hpp"
#include "gpflash/core/Errors.hpp"
#include "gpflash/inputs/EigMaxUbMethod.hpp"                  // EigMaxUbMethod, to_eig_max_ub_method_name
#include "gpflash/inputs/KernelId.hpp"                        // kernel::KernelId, N_KERNEL_ID
#include "gpflash/mean/MeanModel.hpp"                         // mean::make_mean_model, build_train_basis_mat
#include "gpflash/optimization/HpOptimizer.hpp"               // select_hp_with_max_mll
#include "gpflash/optimization/search_space/OptzOptions.hpp"  // OptzOptions, check_optz_options
#include "gpflash/posterior_build/PosteriorBuild.hpp"
#include "gpflash/training/EvalMllDispatch.hpp"

using gpflash_test::accepts;
using gpflash_test::draw_uniform_vec;
using gpflash_test::kDetailLen;
using gpflash_test::make_points;
using gpflash_test::rejects;
using gpflash_test::report;

namespace {

// Asserts check_training_data accepts well-formed data and rejects each way it can be malformed.
bool test_training_data(const char* name) {
    using gpflash::detail::check_training_data;
    const arma::mat X_good_mat = make_points(3, 8);
    const arma::vec f_good_vec(8, arma::fill::ones);

    // check_training_data takes the struct, so each case builds one in place.
    const auto checked =
            [](const arma::mat& X_ndim_ntrain_mat, const arma::vec& f_train_vec, const double sig2_f_val = 0.0) {
                gpflash::TrainingData train_data;
                train_data.X_ndim_ntrain_mat = X_ndim_ntrain_mat;
                train_data.f_train_vec = f_train_vec;
                train_data.sig2_f_val = sig2_f_val;
                check_training_data(train_data, "test");
            };

    const bool good_ok = accepts([&] { checked(X_good_mat, f_good_vec); });
    // An empty X would make every downstream dimension zero. The (3, 0) case pairs with an empty f_train, so only
    // the column count is wrong.
    const bool empty_rejected = rejects([&] { checked(arma::mat(), f_good_vec); })
            && rejects([&] { checked(arma::mat(3, 0), arma::vec()); })
            && rejects([&] { checked(arma::mat(0, 8), f_good_vec); });
    // A mismatched f_train is the case that reads past the allocation under ARMA_NO_DEBUG.
    const bool length_rejected = rejects([&] { checked(X_good_mat, arma::vec(7, arma::fill::ones)); })
            && rejects([&] { checked(X_good_mat, arma::vec(9, arma::fill::ones)); });
    // Non-finite input silently poisons K, the bound, and the factorization.
    arma::mat X_nan_mat = X_good_mat;
    X_nan_mat(1, 2) = arma::datum::nan;
    arma::mat X_inf_mat = X_good_mat;
    X_inf_mat(0, 0) = arma::datum::inf;
    arma::vec f_nan_vec = f_good_vec;
    f_nan_vec(3) = arma::datum::nan;
    const bool non_finite_rejected = rejects([&] { checked(X_nan_mat, f_good_vec); })
            && rejects([&] { checked(X_inf_mat, f_good_vec); }) && rejects([&] { checked(X_good_mat, f_nan_vec); })
            && rejects([&] { checked(X_good_mat, f_good_vec, /*sig2_f_val=*/arma::datum::nan); });

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "good %d  empty %d  length %d  non_finite %d",
            static_cast<int>(good_ok),
            static_cast<int>(empty_rejected),
            static_cast<int>(length_rejected),
            static_cast<int>(non_finite_rejected));
    return report(name, good_ok && empty_rejected && length_rejected && non_finite_rejected, detail);
}

// Asserts check_hp_val enforces each block's own width -- hp_gamma_vec against n_dim, hp_kern_extra_vec
// against what the kernel declares -- and positivity of every entry.
bool test_hp_val(const char* name) {
    using gpflash::detail::check_hp_val;
    const int n_dim = 3;

    // check_hp_val takes the struct; hp_sig2_k is valid throughout so only the two blocks are under test.
    const auto checked = [](const arma::vec& hp_gamma_vec,
                            const arma::vec& hp_kern_extra_vec,
                            const int n_dim_in,
                            const int n_hp_kern_extra) {
        check_hp_val(
                gpflash::detail::build_hp_val(hp_gamma_vec, hp_kern_extra_vec, 1.0), n_dim_in, n_hp_kern_extra, "test");
    };

    const bool good_ok = accepts([&] {
                             checked(arma::vec(3, arma::fill::ones),
                                     arma::vec(),
                                     /*n_dim_in=*/n_dim,
                                     /*n_hp_kern_extra=*/0);
                         })
            && accepts([&] {
                             checked(arma::vec(3, arma::fill::ones),
                                     arma::vec(1, arma::fill::ones),
                                     /*n_dim_in=*/n_dim,
                                     /*n_hp_kern_extra=*/1);
                         });
    // Exact, not "at least": either block off by one means caller and kernel disagree. The empty case is a
    // default-constructed HpVal, which is what a caller assembling a model by hand starts from.
    const bool width_rejected =
            rejects([&] { checked(arma::vec(), arma::vec(), /*n_dim_in=*/n_dim, /*n_hp_kern_extra=*/0); })
            && rejects([&] {
                   checked(arma::vec(2, arma::fill::ones), arma::vec(), /*n_dim_in=*/n_dim, /*n_hp_kern_extra=*/0);
               })
            && rejects([&] {
                   checked(arma::vec(4, arma::fill::ones), arma::vec(), /*n_dim_in=*/n_dim, /*n_hp_kern_extra=*/0);
               })
            // The extras block missing entirely, and one entry too long.
            && rejects([&] {
                   checked(arma::vec(3, arma::fill::ones), arma::vec(), /*n_dim_in=*/n_dim, /*n_hp_kern_extra=*/1);
               })
            && rejects([&] {
                   checked(arma::vec(3, arma::fill::ones),
                           arma::vec(2, arma::fill::ones),
                           /*n_dim_in=*/n_dim,
                           /*n_hp_kern_extra=*/1);
               });
    // 2 + 2 == 3 + 1, so a single combined length check would accept this mispacking; one check per block
    // rejects it.
    const bool mispacked_rejected = rejects([&] {
        checked(arma::vec(2, arma::fill::ones),
                arma::vec(2, arma::fill::ones),
                /*n_dim_in=*/n_dim,
                /*n_hp_kern_extra=*/1);
    });

    arma::vec hp_nan_vec(3, arma::fill::ones);
    hp_nan_vec(1) = arma::datum::nan;
    // A zero or negative length scale is not a bad guess but a different model: the scaled points go degenerate
    // and K loses rank.
    arma::vec hp_zero_vec(3, arma::fill::ones);
    hp_zero_vec(2) = 0.0;
    arma::vec hp_neg_vec(3, arma::fill::ones);
    hp_neg_vec(0) = -1.0;
    const bool value_rejected =
            rejects([&] { checked(hp_nan_vec, arma::vec(), /*n_dim_in=*/n_dim, /*n_hp_kern_extra=*/0); })
            && rejects([&] { checked(hp_zero_vec, arma::vec(), /*n_dim_in=*/n_dim, /*n_hp_kern_extra=*/0); })
            && rejects([&] { checked(hp_neg_vec, arma::vec(), /*n_dim_in=*/n_dim, /*n_hp_kern_extra=*/0); });

    // The extras block is checked too: an unset hp_kern_extra_vec arrives as zero.
    const arma::vec hp_extra_zero_vec(1, arma::fill::zeros);
    const bool extras_rejected = rejects([&] {
        checked(arma::vec(3, arma::fill::ones), hp_extra_zero_vec, /*n_dim_in=*/n_dim, /*n_hp_kern_extra=*/1);
    });

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "good %d  width %d  mispacked %d  value %d  extras %d",
            static_cast<int>(good_ok),
            static_cast<int>(width_rejected),
            static_cast<int>(mispacked_rejected),
            static_cast<int>(value_rejected),
            static_cast<int>(extras_rejected));
    return report(name, good_ok && width_rejected && mispacked_rejected && value_rejected && extras_rejected, detail);
}

// The mean model a caller's has_mean_model_constant and mean_model_degree_vec ask for, at n_dim.
gpflash::mean::MeanModel make_mean_model_for(
        const bool has_mean_model_constant, const arma::uvec& mean_model_degree_vec, const int n_dim) {
    gpflash::ModelOptions model_opts;
    model_opts.has_mean_model_constant = has_mean_model_constant;
    model_opts.mean_model_degree_vec = mean_model_degree_vec;
    return gpflash::mean::make_mean_model(model_opts, n_dim, "test");
}

// Asserts make_mean_model rejects a length neither 1 nor n_dim (empty included) and a degree past the cap, accepts
// degrees with no constant, and broadcasts a single degree.
bool test_mean_model_fields(const char* name) {
    static constexpr int n_dim = 3;
    // The highest degree the mean model accepts, stated here so that raising the library's cap fails the check.
    const arma::uword mean_model_degree_max_ref = 2;
    const auto make_mean_model_at_n_dim = [](const bool has_mean_model_constant,
                                             const arma::uvec& mean_model_degree_vec) {
        return make_mean_model_for(has_mean_model_constant, mean_model_degree_vec, n_dim);
    };

    const bool fields_ok = accepts([&] { make_mean_model_at_n_dim(true, arma::uvec{0}); }) && accepts([&] {
                               make_mean_model_at_n_dim(true, arma::uvec{0, 1, mean_model_degree_max_ref});
                           })
            && accepts([&] {
                               make_mean_model_at_n_dim(false, arma::uvec{0, 0, 0});
                           })
            && accepts([&] {
                               make_mean_model_at_n_dim(false, arma::uvec{0, 2, 0});
                           });
    const bool fields_rejected = rejects([&] {
                                     make_mean_model_at_n_dim(true, arma::uvec{1, 1});
                                 })
            && rejects([&] {
                                     make_mean_model_at_n_dim(true, arma::uvec{1, 1, 1, 1});
                                 })
            && rejects([&] { make_mean_model_at_n_dim(true, arma::uvec()); })
            && rejects([&] { make_mean_model_at_n_dim(true, arma::uvec{mean_model_degree_max_ref + 1}); });

    // One degree applies to every dimension, each dimension taking one column.
    const arma::uword broadcast_degree = 2;
    const gpflash::mean::MeanModel broadcast_mean_model = make_mean_model_at_n_dim(true, arma::uvec{broadcast_degree});
    const arma::uvec broadcast_degree_vec = arma::uvec(n_dim).fill(broadcast_degree);
    const bool broadcast_ok = arma::all(broadcast_mean_model.degree_vec == broadcast_degree_vec)
            && broadcast_mean_model.n_basis() == 1 + n_dim;

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "fields %d  rejected %d  broadcast %d",
            static_cast<int>(fields_ok),
            static_cast<int>(fields_rejected),
            static_cast<int>(broadcast_ok));
    return report(name, fields_ok && fields_rejected && broadcast_ok, detail);
}

// Asserts the mean model basis holds the constant only when asked, then x_d^degree_d alone for each dimension of
// non-zero degree, in dimension order, against columns written out by hand.
bool test_mean_model_basis_columns(const char* name) {
    const arma::mat X_ndim_npoint_mat = make_points(3, 6);
    const arma::vec x_0_vec = X_ndim_npoint_mat.row(0).t(), x_2_vec = X_ndim_npoint_mat.row(2).t();
    const arma::vec one_vec(X_ndim_npoint_mat.n_cols, arma::fill::ones);

    gpflash::mean::MeanModel mean_model;
    mean_model.degree_vec = arma::uvec{2, 0, 1};
    const arma::mat constant_basis_mat = mean_model.calc_basis_mat(X_ndim_npoint_mat);
    const arma::mat constant_basis_ref_mat = arma::join_horiz(one_vec, x_0_vec % x_0_vec, x_2_vec);
    mean_model.has_constant = false;
    const arma::mat no_constant_basis_mat = mean_model.calc_basis_mat(X_ndim_npoint_mat);
    const arma::mat no_constant_basis_ref_mat = arma::join_horiz(x_0_vec % x_0_vec, x_2_vec);

    const bool constant_ok = arma::size(constant_basis_mat) == arma::size(constant_basis_ref_mat)
            && arma::approx_equal(constant_basis_mat, constant_basis_ref_mat, "absdiff", gpflash_test::kTolExact);
    const bool no_constant_ok = arma::size(no_constant_basis_mat) == arma::size(no_constant_basis_ref_mat)
            && arma::approx_equal(no_constant_basis_mat, no_constant_basis_ref_mat, "absdiff", gpflash_test::kTolExact);

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "constant %d  no_constant %d",
            static_cast<int>(constant_ok),
            static_cast<int>(no_constant_ok));
    return report(name, constant_ok && no_constant_ok, detail);
}

// Asserts build_train_basis_mat rejects a mean model basis with more columns than there are training points, with a
// power that overflows, or with a column that is zero at every point, and accepts one whose columns are dependent,
// which the mean model nugget handles.
bool test_mean_model_train_basis(const char* name) {
    static constexpr int n_dim = 3;
    const auto build_train_basis_at = [](const arma::uvec& mean_model_degree_vec, const arma::mat& X_ndim_ntrain_mat) {
        return gpflash::mean::build_train_basis_mat(
                make_mean_model_for(true, mean_model_degree_vec, n_dim), X_ndim_ntrain_mat, "test");
    };

    // Dependent columns: equal x_0 and x_1 give equal columns, and on one level x_0^2 is a multiple of 1.
    const arma::mat X_ndim_ntrain_mat = make_points(n_dim, 9);
    arma::mat X_equal_dims_mat = X_ndim_ntrain_mat;
    X_equal_dims_mat.row(1) = X_equal_dims_mat.row(0);
    arma::mat X_one_level_mat = X_ndim_ntrain_mat;
    X_one_level_mat.row(0).fill(0.5);
    // An all-zero x_0, whose column carries no information about its weight.
    arma::mat X_first_dim_zero_mat = X_ndim_ntrain_mat;
    X_first_dim_zero_mat.row(0).zeros();
    // A finite x_0 whose square overflows, though its first power does not.
    arma::mat X_large_first_dim_mat = X_ndim_ntrain_mat;
    X_large_first_dim_mat(0, 0) = 1e200;
    // One degree for every dimension: accepted on as many points as mean model basis functions, rejected on one fewer.
    const arma::uword broadcast_degree = 2;
    const int n_basis_ref = 1 + n_dim;
    const arma::mat X_as_many_points_mat = make_points(n_dim, n_basis_ref);
    const arma::mat X_few_points_mat = make_points(n_dim, n_basis_ref - 1);

    const bool basis_ok = accepts([&] {
                              build_train_basis_at(arma::uvec{2, 1, 1}, X_ndim_ntrain_mat);
                          })
            && accepts([&] {
                              build_train_basis_at(arma::uvec{1, 1, 0}, X_equal_dims_mat);
                          })
            && accepts([&] {
                              build_train_basis_at(arma::uvec{2, 0, 0}, X_one_level_mat);
                          })
            && accepts([&] { build_train_basis_at(arma::uvec{broadcast_degree}, X_as_many_points_mat); })
            && accepts([&] {
                              build_train_basis_at(arma::uvec{1, 0, 0}, X_large_first_dim_mat);
                          });
    const bool basis_rejected = rejects([&] {
                                    build_train_basis_at(arma::uvec{1, 0, 0}, X_first_dim_zero_mat);
                                })
            && rejects([&] { build_train_basis_at(arma::uvec{broadcast_degree}, X_few_points_mat); }) && rejects([&] {
                                    build_train_basis_at(arma::uvec{2, 0, 0}, X_large_first_dim_mat);
                                });

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "accepted %d  rejected %d",
            static_cast<int>(basis_ok),
            static_cast<int>(basis_rejected));
    return report(name, basis_ok && basis_rejected, detail);
}

// Asserts check_model_options rejects a kernel_id outside the enum, and accepts cond_max only where the nugget
// formula is defined: finite and strictly above 1.
bool test_model_options(const char* name) {
    using gpflash::detail::check_model_options;

    const auto checked = [](const double cond_max) {
        gpflash::ModelOptions model_opts;
        model_opts.cond_max = cond_max;
        check_model_options(model_opts, "test");
    };
    const bool good_ok = accepts([&] { checked(1.0e10); }) && accepts([&] { checked(2.0); })
            && accepts([&] { checked(gpflash::COND_MAX_LIMIT); });  // the limit itself is usable
    // eta = bound / (cond_max - 1), so 1 divides by zero and below 1 flips its sign.
    const bool rejected = rejects([&] { checked(1.0); }) && rejects([&] { checked(0.5); })
            && rejects([&] { checked(-10.0); }) && rejects([&] { checked(arma::datum::inf); })
            && rejects([&] { checked(arma::datum::nan); });
    // Past the limit eta lands at the rounding floor of the sum it is added to, so the factorization every
    // caller relies on stops being guaranteed. Rejected here rather than left to fail inside LAPACK.
    const bool over_limit_rejected =
            rejects([&] { checked(gpflash::COND_MAX_LIMIT * 10.0); }) && rejects([&] { checked(1.0e16); });

    // kernel_by_id also refuses one, but check_model_options is what validates a ModelOptions on its own.
    const auto checked_kernel_id = [](const gpflash::kernel::KernelId kernel_id) {
        gpflash::ModelOptions model_opts;
        model_opts.kernel_id = kernel_id;
        check_model_options(model_opts, "test");
    };
    const auto as_kernel_id = [](const int kernel_id_int) {
        return static_cast<gpflash::kernel::KernelId>(kernel_id_int);
    };
    const bool kernel_id_ok = accepts([&] { checked_kernel_id(gpflash::kernel::KernelId::SqExp); })
            && accepts([&] { checked_kernel_id(as_kernel_id(gpflash::kernel::N_KERNEL_ID - 1)); });  // the last one
    // End is one past the last, so it is the first rejected value rather than a usable kernel.
    const bool kernel_id_rejected = rejects([&] { checked_kernel_id(gpflash::kernel::KernelId::End); })
            && rejects([&] { checked_kernel_id(as_kernel_id(gpflash::kernel::N_KERNEL_ID + 7)); })
            && rejects([&] { checked_kernel_id(as_kernel_id(-1)); });

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "cond_max good %d  rejected %d  over_limit %d  kernel_id good %d  rejected %d",
            static_cast<int>(good_ok),
            static_cast<int>(rejected),
            static_cast<int>(over_limit_rejected),
            static_cast<int>(kernel_id_ok),
            static_cast<int>(kernel_id_rejected));
    return report(name, good_ok && rejected && over_limit_rejected && kernel_id_ok && kernel_id_rejected, detail);
}

// Asserts check_eval_variances rejects a noise variance an evaluation cannot use. EvalVariances is reached after
// the sign rule is resolved, so a negative value here is a bug rather than a request to estimate.
bool test_eval_variances(const char* name) {
    using gpflash::detail::check_eval_variances;

    const auto checked = [](const double sig2_f_val) {
        gpflash::detail::EvalVariances eval_variances;
        eval_variances.sig2_f_val = sig2_f_val;
        eval_variances.hp_sig2_k = 1.0;
        check_eval_variances(eval_variances, "test");
    };

    const bool good_ok = accepts([&] { checked(0.0); }) && accepts([&] { checked(1.0e-3); });
    const bool negative_rejected = rejects([&] { checked(-1.0); });
    const bool nan_rejected = rejects([&] { checked(arma::datum::nan); });

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "good %d  negative %d  nan %d",
            static_cast<int>(good_ok),
            static_cast<int>(negative_rejected),
            static_cast<int>(nan_rejected));
    return report(name, good_ok && negative_rejected && nan_rejected, detail);
}

// Asserts select_sig2_f_val implements the noise sign rule. The sign of TrainingData::sig2_f_val says where the
// noise comes from, HpVal::hp_sig2_f_val carries an estimate; both present are two different models with no way to
// tell which was meant, so it throws rather than picking one.
bool test_select_sig2_f_val(const char* name) {
    using gpflash::detail::select_sig2_f_val;

    const auto resolved = [](const double sig2_f_val, const double hp_sig2_f_val) {
        gpflash::TrainingData train_data;
        train_data.sig2_f_val = sig2_f_val;
        gpflash::HpVal hp_val_struct;
        hp_val_struct.hp_sig2_f_val = hp_sig2_f_val;
        return select_sig2_f_val(train_data, hp_val_struct.hp_sig2_f_val, "test");
    };
    // Noise-free and known-noise pass their own value through, with no estimate present.
    const bool noise_free_ok = resolved(0.0, -1.0) == 0.0;
    const bool known_ok = resolved(0.25, -1.0) == 0.25;
    // A negative request reads the estimate instead.
    const bool estimated_ok = resolved(-1.0, 0.031) == 0.031;

    // A request for an estimate with none present: the empty-hyperparameter case again.
    const bool missing_estimate_rejected =
            rejects([&] { resolved(-1.0, -1.0); }) && rejects([&] { resolved(-1.0, 0.0); });
    // Supplied and estimated: ambiguous.
    const bool conflict_rejected = rejects([&] { resolved(0.0, 0.031); }) && rejects([&] { resolved(0.25, 0.031); });
    const bool non_finite_rejected =
            rejects([&] { resolved(arma::datum::nan, -1.0); }) && rejects([&] { resolved(0.0, arma::datum::nan); });

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "noise_free %d  known %d  estimated %d  missing %d  conflict %d  non_finite %d",
            static_cast<int>(noise_free_ok),
            static_cast<int>(known_ok),
            static_cast<int>(estimated_ok),
            static_cast<int>(missing_estimate_rejected),
            static_cast<int>(conflict_rejected),
            static_cast<int>(non_finite_rejected));
    return report(
            name,
            noise_free_ok && known_ok && estimated_ok && missing_estimate_rejected && conflict_rejected
                    && non_finite_rejected,
            detail);
}

// Asserts check_supplied_hp_sig2_k rejects the -1 sentinel HpVal defaults to, and anything non-positive.
bool test_supplied_hp_sig2_k(const char* name) {
    using gpflash::detail::check_supplied_hp_sig2_k;

    const auto checked = [](const double hp_sig2_k) {
        gpflash::HpVal hp_val_struct;
        hp_val_struct.hp_sig2_k = hp_sig2_k;
        check_supplied_hp_sig2_k(hp_val_struct, "test");
    };

    const bool good_ok = accepts([&] { checked(2.7); });
    // -1 is HpVal's "never set" default, which is what this catches.
    const bool sentinel_rejected = rejects([&] { checked(-1.0); });
    const bool zero_rejected = rejects([&] { checked(0.0); });
    const bool nan_rejected = rejects([&] { checked(arma::datum::nan); });

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "good %d  sentinel %d  zero %d  nan %d",
            static_cast<int>(good_ok),
            static_cast<int>(sentinel_rejected),
            static_cast<int>(zero_rejected),
            static_cast<int>(nan_rejected));
    return report(name, good_ok && sentinel_rejected && zero_rejected && nan_rejected, detail);
}

// Asserts check_optz_options validates every option the selected layout will read, and ignores the ones it
// will not.
bool test_optz_options(const char* name) {
    using gpflash::detail::build_hp_optz_layout;
    using gpflash::detail::check_optz_options;
    const int n_dim = 3;
    // The widest active layout, so every group below is reached: both variances searched, and an extras block.
    const gpflash::HpOptzLayout hp_optz_layout_all_active =
            build_hp_optz_layout(n_dim, /*n_hp_kern_extra=*/n_dim, gpflash::NoiseRequest::Estimate, "test");
    const auto check_all_active = [&](const gpflash::OptzOptions& optz_opts) {
        check_optz_options(optz_opts, hp_optz_layout_all_active, "test");
    };
    const bool good_ok = accepts([&] { check_all_active(gpflash::OptzOptions{}); });

    // Independent-optimization count, process cap and evaluation budget.
    gpflash::OptzOptions bad_n_ind_optz;
    bad_n_ind_optz.use_multistart = true;
    bad_n_ind_optz.n_ind_optz = 0;
    gpflash::OptzOptions bad_max_parallel;
    bad_max_parallel.max_parallel_ind_optz = 0;
    gpflash::OptzOptions bad_evals;
    bad_evals.max_evals = 0;
    const bool counts_rejected = rejects([&] { check_all_active(bad_n_ind_optz); })
            && rejects([&] { check_all_active(bad_max_parallel); }) && rejects([&] { check_all_active(bad_evals); });

    // Tolerances must be finite and positive: a zero tolerance never stops.
    gpflash::OptzOptions bad_step_tol;
    bad_step_tol.rel_step_tol = 0.0;
    gpflash::OptzOptions bad_obj_tol;
    bad_obj_tol.rel_obj_tol = -1.0;
    const bool tolerances_rejected =
            rejects([&] { check_all_active(bad_step_tol); }) && rejects([&] { check_all_active(bad_obj_tol); });

    // A value cast in from outside the enum.
    gpflash::OptzOptions bad_algorithm;
    bad_algorithm.numerical_optz_algorithm = static_cast<gpflash::NumericalOptzAlgorithm>(99);
    const bool algorithm_rejected = rejects([&] { check_all_active(bad_algorithm); });

    // Bounds are natural-space and log-transformed internally, so a non-positive one has no logarithm. Their
    // ordering is not judged here -- the two corners may be given at different lengths -- so test_resolved_hp_bound
    // covers it.
    gpflash::OptzOptions bad_gamma_lb;
    bad_gamma_lb.hp_gamma_lb_vec = arma::vec{0.0};
    gpflash::OptzOptions bad_gamma_ub;
    bad_gamma_ub.hp_gamma_ub_vec = arma::vec{-1.0};
    const bool bounds_rejected =
            rejects([&] { check_all_active(bad_gamma_lb); }) && rejects([&] { check_all_active(bad_gamma_ub); });

    // The scalar variance pairs carry both corners, so their ordering is judged here.
    gpflash::OptzOptions inverted_sig2_k;
    inverted_sig2_k.hp_sig2_k_lb = inverted_sig2_k.hp_sig2_k_ub * 2.0;
    gpflash::OptzOptions inverted_sig2_f_val;
    inverted_sig2_f_val.hp_sig2_f_val_ub = inverted_sig2_f_val.hp_sig2_f_val_lb / 2.0;
    // Each end is named on its own, so the message says which one to fix.
    gpflash::OptzOptions zero_sig2_k_lb;
    zero_sig2_k_lb.hp_sig2_k_lb = 0.0;
    const auto calc_error_message = [&](const gpflash::OptzOptions& optz_opts) {
        try {
            check_all_active(optz_opts);
        } catch (const gpflash::ErrorInvalidInput& caught) {
            return std::string(caught.what());
        }
        return std::string();
    };
    const bool zero_lb_named =
            calc_error_message(zero_sig2_k_lb).find("OptzOptions::hp_sig2_k_lb must") != std::string::npos;
    const bool inverted_ub_named =
            calc_error_message(inverted_sig2_k).find("OptzOptions::hp_sig2_k_ub of") != std::string::npos;
    const bool variance_pairs_rejected = rejects([&] { check_all_active(inverted_sig2_k); })
            && rejects([&] { check_all_active(inverted_sig2_f_val); }) && zero_lb_named && inverted_ub_named;

    // Starting values, one per block that has an option.
    gpflash::OptzOptions bad_sig2_k_init;
    bad_sig2_k_init.hp_sig2_k_init = 0.0;
    gpflash::OptzOptions bad_sig2_f_val_init;
    bad_sig2_f_val_init.hp_sig2_f_val_init = -1.0;
    gpflash::OptzOptions bad_kern_extra_init;
    bad_kern_extra_init.hp_kern_extra_init = arma::datum::inf;
    const bool inits_rejected = rejects([&] { check_all_active(bad_sig2_k_init); })
            && rejects([&] { check_all_active(bad_sig2_f_val_init); })
            && rejects([&] { check_all_active(bad_kern_extra_init); });

    // A block bound is either one entry to broadcast or one per component, and is validated entry by entry. Any
    // other length is a caller who thinks the block is a different width than it is.
    gpflash::OptzOptions optz_opts_broadcast;
    optz_opts_broadcast.hp_gamma_lb_vec = arma::vec{1.0e-3};  // the default form, stated explicitly
    gpflash::OptzOptions optz_opts_short_lb;
    optz_opts_short_lb.hp_gamma_lb_vec = arma::vec(n_dim - 1, arma::fill::ones) * 1.0e-3;
    gpflash::OptzOptions optz_opts_empty_lb;
    optz_opts_empty_lb.hp_gamma_lb_vec.reset();
    gpflash::OptzOptions optz_opts_bad_entry;
    optz_opts_bad_entry.hp_gamma_lb_vec = arma::vec(n_dim, arma::fill::ones) * 1.0e-3;
    optz_opts_bad_entry.hp_gamma_lb_vec(1) = -1.0;
    const bool per_component_rejected = accepts([&] { check_all_active(optz_opts_broadcast); })
            && rejects([&] { check_all_active(optz_opts_short_lb); })
            && rejects([&] { check_all_active(optz_opts_empty_lb); })
            && rejects([&] { check_all_active(optz_opts_bad_entry); });

    // An inactive block is not read: the extras bounds may be nonsense for a kernel that has none.
    gpflash::OptzOptions bad_hp_kern_extra;
    bad_hp_kern_extra.hp_kern_extra_lb_vec = arma::vec{-1.0};
    const gpflash::HpOptzLayout hp_optz_layout_bare =
            build_hp_optz_layout(n_dim, /*n_hp_kern_extra=*/0, gpflash::NoiseRequest::NoiseFree, "test");
    const bool inactive_ignored = accepts([&] { check_optz_options(bad_hp_kern_extra, hp_optz_layout_bare, "test"); })
            && rejects([&] { check_all_active(bad_hp_kern_extra); });

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "good %d  counts %d  tolerances %d  algorithm %d  bounds %d  variance_pairs %d  inits %d  per_component %d"
            "  inactive %d",
            static_cast<int>(good_ok),
            static_cast<int>(counts_rejected),
            static_cast<int>(tolerances_rejected),
            static_cast<int>(algorithm_rejected),
            static_cast<int>(bounds_rejected),
            static_cast<int>(variance_pairs_rejected),
            static_cast<int>(inits_rejected),
            static_cast<int>(per_component_rejected),
            static_cast<int>(inactive_ignored));
    return report(
            name,
            good_ok && counts_rejected && tolerances_rejected && algorithm_rejected && bounds_rejected
                    && variance_pairs_rejected && inits_rejected && per_component_rejected && inactive_ignored,
            detail);
}

// Asserts the resolved region is validated, which is a rule no single option carries: check_optz_options judges each
// corner's length and positivity alone, so an inverted pair only shows once both are widened to the block.
// Driven through the public entry point, since resolving the corners is part of a fit.
bool test_resolved_hp_bound(const char* name) {
    gpflash::TrainingData train_data;
    train_data.X_ndim_ntrain_mat = make_points(2, 12);
    std::mt19937_64 rng(3);
    train_data.f_train_vec = draw_uniform_vec(rng, 12);
    train_data.sig2_f_val = 1.0e-3;

    const auto fitted = [&](const gpflash::OptzOptions& optz_opts) {
        gpflash::select_hp_with_max_mll(train_data, optz_opts);
    };

    const bool good_ok = accepts([&] { fitted(gpflash::OptzOptions{}); });

    // The region's own sampling and start checks also reject a crossed box, so each case must carry the ordering
    // check's message, naming the entry whose upper bound fails.
    const auto is_order_rejected = [&](const gpflash::OptzOptions& optz_opts, const std::string& entry_name) {
        std::string error_message;
        try {
            fitted(optz_opts);
        } catch (const gpflash::ErrorInvalidInput& error) {
            error_message = error.what();
        } catch (...) {
        }
        return error_message.find(entry_name + " upper bound of") != std::string::npos;
    };

    // Each vector is valid alone -- right length, entries finite and positive -- and crosses the opposite corner's
    // default.
    gpflash::OptzOptions optz_opts_crossed_lb;
    optz_opts_crossed_lb.hp_gamma_lb_vec = arma::vec{2.0e3, 2.0e3};  // above the default hp_gamma_ub_vec of {1e3}
    gpflash::OptzOptions optz_opts_crossed_ub;
    optz_opts_crossed_ub.hp_gamma_ub_vec = arma::vec{1.0e-4, 1.0e-4};  // below the default hp_gamma_lb_vec of {1e-3}
    // Both corners overridden, so the inversion lies entirely in the per-component form.
    gpflash::OptzOptions optz_opts_inverted;
    optz_opts_inverted.hp_gamma_lb_vec = arma::vec{5.0, 5.0};
    optz_opts_inverted.hp_gamma_ub_vec = arma::vec{1.0, 1.0};

    const bool crossed_lb_rejected = is_order_rejected(optz_opts_crossed_lb, "hp_gamma[0]");
    const bool crossed_ub_rejected = is_order_rejected(optz_opts_crossed_ub, "hp_gamma[0]");
    const bool inverted_rejected = is_order_rejected(optz_opts_inverted, "hp_gamma[0]");

    // A single component inverted, the rest sound: the check is per component, not on the block's extremes.
    gpflash::OptzOptions one_component_inverted;
    one_component_inverted.hp_gamma_lb_vec = arma::vec{1.0e-3, 5.0};
    one_component_inverted.hp_gamma_ub_vec = arma::vec{1.0e3, 1.0};
    const bool one_component_rejected = is_order_rejected(one_component_inverted, "hp_gamma[1]");

    // A broadcast pair inverted against itself, which is the commonest way to get it wrong.
    gpflash::OptzOptions inverted_broadcast;
    inverted_broadcast.hp_gamma_lb_vec = arma::vec{10.0};
    inverted_broadcast.hp_gamma_ub_vec = arma::vec{1.0};
    // And the same with the search unconstrained: the region is resolved either way, because it still places and
    // clamps the start.
    gpflash::OptzOptions inverted_unbounded = inverted_broadcast;
    inverted_unbounded.use_bounds = false;
    const bool inverted_broadcast_rejected = is_order_rejected(inverted_broadcast, "hp_gamma[0]");
    const bool inverted_unbounded_rejected = is_order_rejected(inverted_unbounded, "hp_gamma[0]");

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "good %d  crossed_lb %d  crossed_ub %d  inverted %d  one_component %d  broadcast %d  unbounded %d",
            static_cast<int>(good_ok),
            static_cast<int>(crossed_lb_rejected),
            static_cast<int>(crossed_ub_rejected),
            static_cast<int>(inverted_rejected),
            static_cast<int>(one_component_rejected),
            static_cast<int>(inverted_broadcast_rejected),
            static_cast<int>(inverted_unbounded_rejected));
    return report(
            name,
            good_ok && crossed_lb_rejected && crossed_ub_rejected && inverted_rejected && one_component_rejected
                    && inverted_broadcast_rejected && inverted_unbounded_rejected,
            detail);
}

// Asserts the public entry points reject malformed input, rather than each caller having to. The empty-hp_gamma_vec
// case matters most: a default-constructed HpVal is where a caller specifying a model by hand starts, and unchecked
// it reads out of bounds.
bool test_public_entry_points(const char* name) {
    gpflash::TrainingData train_data;
    train_data.X_ndim_ntrain_mat = make_points(3, 10);
    std::mt19937_64 rng(5);
    train_data.f_train_vec = draw_uniform_vec(rng, 10);

    gpflash::HpVal hp_good_struct;
    hp_good_struct.hp_gamma_vec = arma::vec(3, arma::fill::ones);
    hp_good_struct.hp_sig2_k = 1.5;

    const gpflash::ModelOptions model_opts;  // every entry point below takes one explicitly
    const bool eval_mll_good_ok = accepts([&] { gpflash::eval_mll(train_data, hp_good_struct, model_opts); });

    // A default-constructed HpVal, whose blocks are all empty.
    const gpflash::HpVal hp_empty_struct;
    const bool eval_mll_empty_rejected = rejects([&] { gpflash::eval_mll(train_data, hp_empty_struct, model_opts); });

    // A hyperparameter vector of the wrong width for the data.
    gpflash::HpVal hp_wrong_width_struct = hp_good_struct;
    hp_wrong_width_struct.hp_gamma_vec = arma::vec(2, arma::fill::ones);
    const bool eval_mll_width_rejected =
            rejects([&] { gpflash::eval_mll(train_data, hp_wrong_width_struct, model_opts); });

    // Bad options reach the same tier.
    gpflash::ModelOptions bad_model_opts;
    bad_model_opts.cond_max = 1.0;
    const bool eval_mll_cond_rejected = rejects([&] { gpflash::eval_mll(train_data, hp_good_struct, bad_model_opts); });

    // hp_sig2_k is required exactly where no closed form supplies it: on noisy data, not on noise-free.
    gpflash::HpVal hp_no_sig2_k_struct;
    hp_no_sig2_k_struct.hp_gamma_vec = arma::vec(3, arma::fill::ones);  // hp_sig2_k left at its -1 sentinel
    gpflash::TrainingData train_data_noisy = train_data;
    train_data_noisy.sig2_f_val = 0.01;
    const bool eval_mll_noise_free_sig2_k_ok =
            accepts([&] { gpflash::eval_mll(train_data, hp_no_sig2_k_struct, model_opts); });
    const bool eval_mll_noisy_sig2_k_rejected =
            rejects([&] { gpflash::eval_mll(train_data_noisy, hp_no_sig2_k_struct, model_opts); });

    // The posterior validates the same way, and additionally requires hp_sig2_k, which it never solves for.
    const bool posterior_good_ok = accepts([&] { gpflash::make_posterior(train_data, hp_good_struct, model_opts); });
    const bool posterior_empty_rejected =
            rejects([&] { gpflash::make_posterior(train_data, hp_empty_struct, model_opts); });
    const bool posterior_sig2_k_rejected =
            rejects([&] { gpflash::make_posterior(train_data, hp_no_sig2_k_struct, model_opts); });

    // A mismatched noise pairing is rejected at the posterior too, where a fitted model is most often assembled
    // by hand.
    gpflash::TrainingData train_data_estimate_request = train_data;
    train_data_estimate_request.sig2_f_val = -1.0;  // asks for an estimate that hp_good_struct does not carry
    const bool posterior_noise_rejected =
            rejects([&] { gpflash::make_posterior(train_data_estimate_request, hp_good_struct, model_opts); });

    // A bad mean model reaches every entry point that builds one: a degree per dimension plus one extra, and degree 1
    // on a dimension that is zero at every training point. A verified load builds the mean model to hash it but never
    // its training basis, so only the first applies there. Each rejection must name the field, since a missing file is
    // rejected too.
    const auto rejects_mean_model = [](const auto& function) {
        bool is_named = false;
        try {
            function();
        } catch (const gpflash::ErrorInvalidInput& error) {
            is_named = std::string(error.what()).find("ModelOptions::mean_model_degree_vec") != std::string::npos;
        } catch (...) {
        }
        return is_named;
    };
    gpflash::ModelOptions model_opts_bad_mean_model_length;
    model_opts_bad_mean_model_length.mean_model_degree_vec = arma::uvec{1, 1, 1, 1};
    gpflash::ModelOptions model_opts_bad_mean_model_basis;
    model_opts_bad_mean_model_basis.mean_model_degree_vec = arma::uvec{1, 0, 0};
    gpflash::TrainingData train_data_first_dim_zero = train_data;
    train_data_first_dim_zero.X_ndim_ntrain_mat.row(0).zeros();
    const auto fit_with = [](const gpflash::TrainingData& train_data_fit, const gpflash::ModelOptions& model_opts_fit) {
        gpflash::OptzOptions optz_opts;
        optz_opts.model_options = model_opts_fit;
        gpflash::select_hp_with_max_mll(train_data_fit, optz_opts);
    };
    const bool mean_model_rejected =
            rejects_mean_model([&] { gpflash::eval_mll(train_data, hp_good_struct, model_opts_bad_mean_model_length); })
            && rejects_mean_model([&] {
                   gpflash::eval_mll(train_data_first_dim_zero, hp_good_struct, model_opts_bad_mean_model_basis);
               })
            && rejects_mean_model(
                    [&] { gpflash::make_posterior(train_data, hp_good_struct, model_opts_bad_mean_model_length); })
            && rejects_mean_model([&] {
                   gpflash::make_posterior(train_data_first_dim_zero, hp_good_struct, model_opts_bad_mean_model_basis);
               })
            && rejects_mean_model([&] { fit_with(train_data, model_opts_bad_mean_model_length); })
            && rejects_mean_model([&] { fit_with(train_data_first_dim_zero, model_opts_bad_mean_model_basis); })
            && rejects_mean_model([&] {
                   gpflash::load_posterior_verified(
                           "no_such_cache_file.h5", train_data, hp_good_struct, model_opts_bad_mean_model_length);
               });

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "eval_mll good %d empty %d width %d cond %d sig2_k %d/%d  posterior good %d empty %d sig2_k %d noise %d"
            "  mean_model %d",
            static_cast<int>(eval_mll_good_ok),
            static_cast<int>(eval_mll_empty_rejected),
            static_cast<int>(eval_mll_width_rejected),
            static_cast<int>(eval_mll_cond_rejected),
            static_cast<int>(eval_mll_noise_free_sig2_k_ok),
            static_cast<int>(eval_mll_noisy_sig2_k_rejected),
            static_cast<int>(posterior_good_ok),
            static_cast<int>(posterior_empty_rejected),
            static_cast<int>(posterior_sig2_k_rejected),
            static_cast<int>(posterior_noise_rejected),
            static_cast<int>(mean_model_rejected));
    return report(
            name,
            eval_mll_good_ok && eval_mll_empty_rejected && eval_mll_width_rejected && eval_mll_cond_rejected
                    && eval_mll_noise_free_sig2_k_ok && eval_mll_noisy_sig2_k_rejected && posterior_good_ok
                    && posterior_empty_rejected && posterior_sig2_k_rejected && posterior_noise_rejected
                    && mean_model_rejected,
            detail);
}

// Asserts a posterior query rejects a point of the wrong dimension or with a non-finite coordinate, naming the
// argument, in both the single-point and the batch call. Unchecked, the first reads past the training features and
// the second returns the far-field answer, which looks valid. With PosteriorOptions::check_query_points cleared, a
// valid point gives the same result and a non-finite one is not rejected by name; a wrong dimension is not tried,
// since unchecked it reads out of bounds.
bool test_post_points(const char* name) {
    gpflash::TrainingData train_data;
    train_data.X_ndim_ntrain_mat = make_points(3, 10);
    std::mt19937_64 rng(5);
    train_data.f_train_vec = draw_uniform_vec(rng, 10);
    gpflash::HpVal hp_val_struct;
    hp_val_struct.hp_gamma_vec = arma::vec(3, arma::fill::ones);
    hp_val_struct.hp_sig2_k = 1.5;
    const gpflash::GpPosterior posterior = gpflash::make_posterior(train_data, hp_val_struct, gpflash::ModelOptions());

    const auto rejects_naming = [](const char* field_name, const auto& function) {
        bool is_named = false;
        try {
            function();
        } catch (const gpflash::ErrorInvalidInput& error) {
            is_named = std::string(error.what()).find(field_name) != std::string::npos;
        } catch (...) {
        }
        return is_named;
    };
    const arma::vec x_good_vec = {0.2, 0.4, 0.6};
    const arma::mat X_good_mat = make_points(3, 4);
    arma::vec x_nan_vec = x_good_vec;
    x_nan_vec(1) = arma::datum::nan;
    arma::mat X_inf_mat = X_good_mat;
    X_inf_mat(2, 3) = -arma::datum::inf;

    const bool good_ok = accepts([&] { posterior.eval(x_good_vec); }) && accepts([&] { posterior.eval(X_good_mat); })
            && accepts([&] { posterior.eval(arma::mat(3, 0)); });
    const bool dimension_rejected = rejects_naming(
                                            "x_post_vec",
                                            [&] {
                                                posterior.eval(arma::vec{0.2, 0.4});
                                            })
            && rejects_naming("x_post_vec",
                              [&] {
                                  posterior.eval(arma::vec{0.2, 0.4, 0.6, 0.8});
                              })
            && rejects_naming("X_ndim_npost_mat", [&] { posterior.eval(make_points(2, 4)); })
            && rejects_naming("X_ndim_npost_mat", [&] { posterior.eval(make_points(4, 4)); });
    const bool non_finite_rejected = rejects_naming("x_post_vec", [&] { posterior.eval(x_nan_vec); })
            && rejects_naming("X_ndim_npost_mat", [&] { posterior.eval(X_inf_mat); });

    gpflash::PosteriorOptions post_opts_unchecked;
    post_opts_unchecked.check_query_points = false;
    const gpflash::PosteriorResult checked_vec_result = posterior.eval(x_good_vec);
    const gpflash::PosteriorResult unchecked_vec_result = posterior.eval(x_good_vec, post_opts_unchecked);
    const gpflash::PosteriorResult checked_mat_result = posterior.eval(X_good_mat);
    const gpflash::PosteriorResult unchecked_mat_result = posterior.eval(X_good_mat, post_opts_unchecked);
    // Bit for bit: the flag changes only whether the checks run, never the arithmetic.
    const auto same_result = [](const gpflash::PosteriorResult& result_a, const gpflash::PosteriorResult& result_b) {
        return arma::approx_equal(result_a.mean_post_vec, result_b.mean_post_vec, "absdiff", 0.0)
                && arma::approx_equal(result_a.var_post_vec, result_b.var_post_vec, "absdiff", 0.0);
    };
    const bool unchecked_same = same_result(checked_vec_result, unchecked_vec_result)
            && same_result(checked_mat_result, unchecked_mat_result);
    // The debug postcondition may still catch the non-finite result, but under the result's name, not the argument's.
    const bool unchecked_skipped =
            !rejects_naming("x_post_vec", [&] { posterior.eval(x_nan_vec, post_opts_unchecked); })
            && !rejects_naming("X_ndim_npost_mat", [&] { posterior.eval(X_inf_mat, post_opts_unchecked); });

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "good %d  dimension rejected %d  non-finite rejected %d  unchecked same %d  unchecked skipped %d",
            static_cast<int>(good_ok),
            static_cast<int>(dimension_rejected),
            static_cast<int>(non_finite_rejected),
            static_cast<int>(unchecked_same),
            static_cast<int>(unchecked_skipped));
    return report(
            name,
            good_ok && dimension_rejected && non_finite_rejected && unchecked_same && unchecked_skipped,
            detail);
}

// Asserts each matrix's bound method is one of the enum's methods.
bool test_eig_max_ub_method(const char* name) {
    using gpflash::detail::check_model_options;

    const auto checked = [](const gpflash::EigMaxUbMethod eig_max_ub_method, const bool is_mean_model) {
        gpflash::ModelOptions model_opts;
        if (is_mean_model) {
            model_opts.mean_model_eig_max_ub_method = eig_max_ub_method;
        } else {
            model_opts.kmat_eig_max_ub_method = eig_max_ub_method;
        }
        check_model_options(model_opts, "test");
    };
    bool good_ok = true, rejected = true;
    for (const bool is_mean_model : {false, true}) {
        good_ok = good_ok && accepts([&] { checked(gpflash::EigMaxUbMethod::Gershgorin, is_mean_model); })
                && accepts([&] { checked(gpflash::EigMaxUbMethod::Trace, is_mean_model); });
        rejected = rejected && rejects([&] { checked(gpflash::EigMaxUbMethod::End, is_mean_model); })
                && rejects([&] { checked(static_cast<gpflash::EigMaxUbMethod>(-1), is_mean_model); });
    }
    const bool named_ok =
            std::string(gpflash::to_eig_max_ub_method_name(gpflash::EigMaxUbMethod::Gershgorin)) == "Gershgorin"
            && std::string(gpflash::to_eig_max_ub_method_name(gpflash::EigMaxUbMethod::Trace)) == "trace"
            && rejects([&] { gpflash::to_eig_max_ub_method_name(gpflash::EigMaxUbMethod::End); });

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "good %d  rejected %d  named %d",
            static_cast<int>(good_ok),
            static_cast<int>(rejected),
            static_cast<int>(named_ok));
    return report(name, good_ok && rejected && named_ok, detail);
}

}  // namespace

int main() {
    std::printf("== GpFlash input validation ==\n");
    bool ok = true;
    ok &= test_training_data("training_data");
    ok &= test_hp_val("hp_val");
    ok &= test_model_options("model_options");
    ok &= test_mean_model_fields("mean_model_fields");
    ok &= test_mean_model_basis_columns("mean_model_basis_columns");
    ok &= test_mean_model_train_basis("mean_model_train_basis");
    ok &= test_eval_variances("eval_variances");
    ok &= test_select_sig2_f_val("select_sig2_f_val");
    ok &= test_supplied_hp_sig2_k("supplied_hp_sig2_k");
    ok &= test_optz_options("optz_options");
    ok &= test_resolved_hp_bound("resolved_hp_bound");
    ok &= test_public_entry_points("public_entry_points");
    ok &= test_eig_max_ub_method("eig_max_ub_method");
    ok &= test_post_points("post_points");
    std::printf("\n%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");

    return ok ? 0 : 1;
}
