// The search space (optimization/search_space): the hyperparameter layout at both of its producers, the
// pack/unpack codec, the at-bound flags mapped onto named fields, and OptzSeeding through build_optz_region.
#include <armadillo>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <utility>  // std::swap

#include "GpFlashTestSupport.hpp"
#include "gpflash/kernels/KernelDispatch.hpp"                  // kernel_by_id
#include "gpflash/optimization/MllOptzObjective.hpp"           // MllOptzObjective, the layout's second producer
#include "gpflash/optimization/search_space/HpAtBound.hpp"     // HpAtBound, check_hp_at_bound, to_hp_at_bound
#include "gpflash/optimization/search_space/HpOptzLayout.hpp"  // HpOptzLayout, HpBlock, build_hp_optz_layout
#include "gpflash/optimization/search_space/OptzOptions.hpp"   // OptzOptions
#include "gpflash/optimization/search_space/OptzRegion.hpp"    // OptzRegion, build_optz_region

using gpflash_test::accepts;
using gpflash_test::draw_uniform_mat;
using gpflash_test::draw_uniform_vec;
using gpflash_test::kDetailLen;
using gpflash_test::rejects;
using gpflash_test::report;

using namespace gpflash;

// A region entry against the log the test forms of the same natural value: only rounding separates them.
constexpr double kTolLogSpace = 1e-12;

// ── The optimization tier's struct checkers ─────────────────────────────────
// Here rather than in test_gpflash_struct_checks.cpp: each tier's struct checks sit with that tier.

// The layout, asserted at both of its producers: build_hp_optz_layout, which derives the offsets, and MllOptzObjective,
// which derives that call's arguments from the data and the kernel so no caller can describe a layout its kernel
// would not produce.
static bool test_search_space_checks(const char* name) {
    using gpflash::HpOptzLayout;
    using gpflash::NoiseRequest;
    using gpflash::detail::build_hp_optz_layout;
    using gpflash::detail::check_hp_at_bound;
    using gpflash::detail::MllOptzObjective;
    using gpflash::detail::to_hp_at_bound;
    const int n_dim = 3;
    gpflash::OptzOptions optz_opts;

    // One extra and estimated noise: [hp_gamma (n_dim)] [extra (1)] [sigma_k^2] [sigma_f^2].
    const HpOptzLayout hp_optz_layout_estimated = build_hp_optz_layout(n_dim, 1, NoiseRequest::Estimate, "test");
    const bool estimated_layout = hp_optz_layout_estimated.offset(gpflash::HpBlock::Gamma) == 0
            && hp_optz_layout_estimated.offset(gpflash::HpBlock::KernExtra) == n_dim
            && hp_optz_layout_estimated.offset(gpflash::HpBlock::Sig2K) == n_dim + 1
            && hp_optz_layout_estimated.offset(gpflash::HpBlock::Sig2FVal) == n_dim + 2
            && hp_optz_layout_estimated.width(gpflash::HpBlock::Gamma) == n_dim
            && hp_optz_layout_estimated.width(gpflash::HpBlock::Sig2K) == 1
            && hp_optz_layout_estimated.n_hp_optz() == n_dim + 3;

    // Fixed noise searches sigma_k^2 but not sigma_f^2.
    const HpOptzLayout hp_optz_layout_fixed = build_hp_optz_layout(n_dim, 1, NoiseRequest::Fixed, "test");
    const bool fixed_layout = hp_optz_layout_fixed.offset(gpflash::HpBlock::Sig2K) == n_dim + 1
            && !hp_optz_layout_fixed.is_searched(gpflash::HpBlock::Sig2FVal)
            && hp_optz_layout_fixed.width(gpflash::HpBlock::Sig2FVal) == 0
            && hp_optz_layout_fixed.offset(gpflash::HpBlock::Sig2FVal) == -1
            && hp_optz_layout_fixed.n_hp_optz() == n_dim + 2;

    // Noise-free with no extras leaves all three slots absent, not zero-width at an offset.
    const HpOptzLayout hp_optz_layout_bare = build_hp_optz_layout(n_dim, 0, NoiseRequest::NoiseFree, "test");
    const bool bare_layout = hp_optz_layout_bare.offset(gpflash::HpBlock::KernExtra) == -1
            && hp_optz_layout_bare.offset(gpflash::HpBlock::Sig2K) == -1
            && hp_optz_layout_bare.offset(gpflash::HpBlock::Sig2FVal) == -1 && hp_optz_layout_bare.n_hp_optz() == n_dim;

    // An entry in a 1-wide block is named for the block alone.
    const bool entry_named = std::string(gpflash::to_hp_block_name(gpflash::HpBlock::Gamma)) == "hp_gamma"
            && gpflash::to_hp_entry_name(hp_optz_layout_estimated, 0) == "hp_gamma[0]"
            && gpflash::to_hp_entry_name(hp_optz_layout_estimated, n_dim) == "hp_kern_extra"
            && gpflash::to_hp_entry_name(hp_optz_layout_estimated, n_dim + 1) == "hp_sig2_k"
            && gpflash::to_hp_entry_name(hp_optz_layout_estimated, n_dim + 2) == "hp_sig2_f_val";

    // Pack/unpack round trip, with distinct values so a transposition would show.
    arma::vec hp_optz_vec(hp_optz_layout_estimated.n_hp_optz());
    for (arma::uword i_hp_optz = 0; i_hp_optz < hp_optz_vec.n_elem; ++i_hp_optz)
        hp_optz_vec(i_hp_optz) = 1.0 + 0.5 * static_cast<double>(i_hp_optz);
    const gpflash::HpVal hp_val_round_trip = gpflash::detail::to_hp_val(hp_optz_layout_estimated, hp_optz_vec);
    const bool codec_round_trips = arma::approx_equal(
                                           gpflash::detail::to_hp_optz_vec(hp_optz_layout_estimated, hp_val_round_trip),
                                           hp_optz_vec,
                                           "absdiff",
                                           0.0)
            && hp_val_round_trip.hp_sig2_k == hp_optz_vec(n_dim + 1)
            && hp_val_round_trip.hp_sig2_f_val == hp_optz_vec(n_dim + 2)
            // An unsearched block keeps HpVal's sentinel.
            && gpflash::detail::to_hp_val(hp_optz_layout_bare, arma::vec(n_dim, arma::fill::ones)).hp_sig2_k < 0.0;

    // What the layout cannot derive, so still checks: its own arguments.
    const bool arguments_rejected = rejects([&] { build_hp_optz_layout(0, 0, NoiseRequest::NoiseFree, "test"); })
            && rejects([&] { build_hp_optz_layout(n_dim, -1, NoiseRequest::Estimate, "test"); })
            && rejects([&] { build_hp_optz_layout(n_dim, 0, NoiseRequest::End, "test"); });

    // The objective derives those arguments rather than accepting them, so the extras width is the kernel's and the
    // noise case is the data's. An empty X is the malformed input a caller most easily supplies.
    gpflash::TrainingData train_data;
    train_data.X_ndim_ntrain_mat = arma::ones<arma::mat>(n_dim, 5);
    train_data.f_train_vec = arma::ones<arma::vec>(5);
    train_data.sig2_f_val = -1.0;  // estimate the noise, so both variances are searched
    optz_opts.model_options.kernel_id = gpflash::kernel::KernelId::RatQuad;
    const MllOptzObjective optz_objective_derived(train_data, optz_opts, "test");
    const bool derived_layout = optz_objective_derived.layout().n_hp_kern_extra == 1            // RatQuad's shape
            && optz_objective_derived.layout().offset(gpflash::HpBlock::Sig2FVal) == n_dim + 2  // from sig2_f_val < 0
            && optz_objective_derived.layout().n_hp_optz() == hp_optz_layout_estimated.n_hp_optz()
            // The objective echoes the kernel it resolved.
            && optz_objective_derived.model_options().kernel_id == gpflash::kernel::KernelId::RatQuad;
    gpflash::TrainingData train_data_empty;
    const bool derived_rejected = rejects([&] { MllOptzObjective(train_data_empty, optz_opts, "test"); });

    // check_hp_at_bound sizes both flag vectors, which is what catches an to_hp_at_bound slice of the wrong
    // width -- the failure mode the extras block being contiguous makes possible.
    gpflash::HpAtBound hp_at_bound_struct;
    hp_at_bound_struct.hp_gamma_at_bound_vec = arma::zeros<arma::uvec>(n_dim);
    hp_at_bound_struct.hp_kern_extra_at_bound_vec = arma::zeros<arma::uvec>(1);
    const bool flags_accepted = accepts([&] { check_hp_at_bound(hp_at_bound_struct, n_dim, 1, "test"); });
    const bool flags_rejected = rejects([&] {
                                    gpflash::HpAtBound broken = hp_at_bound_struct;
                                    broken.hp_kern_extra_at_bound_vec = arma::zeros<arma::uvec>(2);
                                    check_hp_at_bound(broken, n_dim, 1, "test");
                                })
            && rejects([&] {
                                    gpflash::HpAtBound broken = hp_at_bound_struct;
                                    broken.hp_gamma_at_bound_vec = arma::zeros<arma::uvec>(n_dim + 1);
                                    check_hp_at_bound(broken, n_dim, 1, "test");
                                })
            && rejects([&] {  // a kernel with no extras must report no extras flags
                                    check_hp_at_bound(hp_at_bound_struct, n_dim, 0, "test");
                                });

    // to_hp_at_bound maps the flat flag vector onto the named HpAtBound fields, checked one index at a time
    // since an off-by-one is invisible when every flag is equal -- the case in every fit that ends interior.
    bool naming_ok = true;
    {
        // Straight off build_hp_optz_layout, so the offsets are the ones a fit would have. A 2-wide extras block,
        // because a 1-wide one cannot show a slice off by one.
        const int n_dim_named = 2;
        const int n_extra = 2;
        const HpOptzLayout hp_optz_layout_named =
                build_hp_optz_layout(n_dim_named, n_extra, NoiseRequest::Estimate, "test");
        const int idx_extra = hp_optz_layout_named.offset(HpBlock::KernExtra);
        const int n_hp_optz_named = hp_optz_layout_named.n_hp_optz();

        for (int i_hp_optz = 0; i_hp_optz < n_hp_optz_named; ++i_hp_optz) {
            arma::uvec flags_vec(n_hp_optz_named, arma::fill::zeros);
            flags_vec(i_hp_optz) = 1;
            const HpAtBound hp_at_bound_named = to_hp_at_bound(hp_optz_layout_named, flags_vec);
            const int n_gamma_flagged = static_cast<int>(arma::accu(hp_at_bound_named.hp_gamma_at_bound_vec));
            const int n_extra_flagged = static_cast<int>(arma::accu(hp_at_bound_named.hp_kern_extra_at_bound_vec));
            const int n_flagged = n_gamma_flagged + n_extra_flagged + hp_at_bound_named.hp_sig2_k_at_bound
                    + hp_at_bound_named.hp_sig2_f_val_at_bound;
            naming_ok &= hp_at_bound_named.has_any() && n_flagged == 1;
            if (i_hp_optz < n_dim_named)
                naming_ok &= hp_at_bound_named.hp_gamma_at_bound_vec(i_hp_optz) == 1;
            else if (i_hp_optz < idx_extra + n_extra)
                naming_ok &= hp_at_bound_named.hp_kern_extra_at_bound_vec(i_hp_optz - idx_extra) == 1;
            else if (i_hp_optz == hp_optz_layout_named.offset(HpBlock::Sig2K))
                naming_ok &= hp_at_bound_named.hp_sig2_k_at_bound;
            else
                naming_ok &= hp_at_bound_named.hp_sig2_f_val_at_bound;
        }
        // An all-clear flag vector must report nothing, which is what every interior fit produces.
        naming_ok &=
                !to_hp_at_bound(
                         hp_optz_layout_named, arma::uvec(static_cast<arma::uword>(n_hp_optz_named), arma::fill::zeros))
                         .has_any();
        // A flag vector of the wrong width is rejected rather than sliced past its end.
        naming_ok &= rejects([&] {
            to_hp_at_bound(hp_optz_layout_named, arma::uvec(static_cast<arma::uword>(n_hp_optz_named + 1)));
        });
    }

    // The summary's four lines, one per block, each answer distinct, so no two arms of the block-to-string chain
    // can be swapped undetected. An empty extras vector reports n/a, not none.
    gpflash::HpAtBound hp_at_bound_summary;
    hp_at_bound_summary.hp_gamma_at_bound_vec = arma::uvec{0, 1, 0};
    hp_at_bound_summary.hp_kern_extra_at_bound_vec = arma::zeros<arma::uvec>(1);
    hp_at_bound_summary.hp_sig2_k_at_bound = true;
    hp_at_bound_summary.hp_sig2_f_val_at_bound = false;
    const std::string at_bound_summary_str = gpflash::to_summary_string(hp_at_bound_summary);
    const bool summary_lines_ok = at_bound_summary_str.find("  hp_gamma       at bound  1\n") != std::string::npos
            && at_bound_summary_str.find("  hp_kern_extra  at bound  none\n") != std::string::npos
            && at_bound_summary_str.find("  hp_sig2_k      at bound  yes\n") != std::string::npos
            && at_bound_summary_str.find("  hp_sig2_f_val  at bound  no\n") != std::string::npos
            && gpflash::to_summary_string(gpflash::HpAtBound{}).find("  hp_kern_extra  at bound  n/a\n")
                    != std::string::npos;

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "estimated=%d fixed=%d bare=%d entry=%d codec=%d arguments=%d derived=%d/%d flags=%d/%d naming=%d "
            "summary=%d",
            static_cast<int>(estimated_layout),
            static_cast<int>(fixed_layout),
            static_cast<int>(bare_layout),
            static_cast<int>(entry_named),
            static_cast<int>(codec_round_trips),
            static_cast<int>(arguments_rejected),
            static_cast<int>(derived_layout),
            static_cast<int>(derived_rejected),
            static_cast<int>(flags_accepted),
            static_cast<int>(flags_rejected),
            static_cast<int>(naming_ok),
            static_cast<int>(summary_lines_ok));
    return report(
            name,
            estimated_layout && fixed_layout && bare_layout && entry_named && codec_round_trips && arguments_rejected
                    && derived_layout && derived_rejected && flags_accepted && flags_rejected && naming_ok
                    && summary_lines_ok,
            detail);
}

// OptzSeeding through build_optz_region: applied, clamped into the constraint box, and rejected when malformed.
static bool test_optz_seeding(const char* name) {
    using gpflash::detail::build_hp_optz_layout;
    using gpflash::detail::build_optz_region;

    // A kernel that seeds nothing samples the whole region.
    const int n_dim = 2;
    const int n_train = 12;
    std::mt19937_64 rng(29);
    gpflash::TrainingData train_data;
    train_data.X_ndim_ntrain_mat = draw_uniform_mat(rng, n_dim, n_train, 0.0, 4.0);
    train_data.f_train_vec = draw_uniform_vec(rng, n_train);
    train_data.sig2_f_val = 1e-3;

    const double f_train_residual_var = arma::var(train_data.f_train_vec);  // the constant mean model's
    const arma::vec x_center_vec = arma::zeros<arma::vec>(n_dim);
    gpflash::OptzOptions optz_opts;
    const gpflash::HpOptzLayout hp_optz_layout =
            build_hp_optz_layout(n_dim, 0, gpflash::NoiseRequest::Fixed, "test_optz_seeding");
    const gpflash::OptzRegion optz_region_default = build_optz_region(
            train_data, x_center_vec, f_train_residual_var, optz_opts, gpflash::OptzSeeding(), hp_optz_layout, "test");
    const bool default_samples_whole_region =
            arma::approx_equal(optz_region_default.sample_lb_vec, optz_region_default.constraint_lb_vec, "absdiff", 0.0)
            && arma::approx_equal(
                    optz_region_default.sample_ub_vec, optz_region_default.constraint_ub_vec, "absdiff", 0.0);

    // The heuristic start it takes instead: hp_gamma_d = 1 / stddev(X_d), and hp_sig2_k_init times the residual
    // variance.
    bool default_start_ok =
            std::abs(optz_region_default.start_vec(n_dim) - std::log(optz_opts.hp_sig2_k_init * f_train_residual_var))
            < kTolLogSpace;
    for (int i_dim = 0; i_dim < n_dim; ++i_dim) {
        const double ln_hp_gamma_start_ref = -std::log(arma::stddev(train_data.X_ndim_ntrain_mat.row(i_dim)));
        default_start_ok = default_start_ok
                && std::abs(optz_region_default.start_vec(i_dim) - ln_hp_gamma_start_ref) < kTolLogSpace;
    }

    // Inside the bounds: applied as given, in log space. Distinct per dimension, so a seeding written to one entry
    // only shows.
    gpflash::OptzSeeding optz_seeding;
    optz_seeding.hp_gamma_start_vec = {2.0, 3.0};
    optz_seeding.hp_gamma_sample_lb_vec = {1.0, 1.5};
    optz_seeding.hp_gamma_sample_ub_vec = {4.0, 6.0};
    const gpflash::OptzRegion optz_region_seeded = build_optz_region(
            train_data, x_center_vec, f_train_residual_var, optz_opts, optz_seeding, hp_optz_layout, "test");
    const arma::uword n_dim_u = static_cast<arma::uword>(n_dim);
    const bool seeding_applied =
            arma::norm(optz_region_seeded.start_vec.head(n_dim_u) - arma::log(optz_seeding.hp_gamma_start_vec), "inf")
                    < kTolLogSpace
            && arma::norm(
                       optz_region_seeded.sample_lb_vec.head(n_dim_u) - arma::log(optz_seeding.hp_gamma_sample_lb_vec),
                       "inf")
                    < kTolLogSpace
            && arma::norm(
                       optz_region_seeded.sample_ub_vec.head(n_dim_u) - arma::log(optz_seeding.hp_gamma_sample_ub_vec),
                       "inf")
                    < kTolLogSpace
            // Only the gamma block is the kernel's; the variance entries keep sampling their whole range.
            && optz_region_seeded.sample_ub_vec(n_dim) == optz_region_seeded.constraint_ub_vec(n_dim);

    // Outside the bounds: clamped in.
    gpflash::OptzSeeding optz_seeding_outside;
    optz_seeding_outside.hp_gamma_start_vec = arma::vec(n_dim, arma::fill::ones) * 1e9;
    optz_seeding_outside.hp_gamma_sample_lb_vec = arma::vec(n_dim, arma::fill::ones) * 1e-9;
    optz_seeding_outside.hp_gamma_sample_ub_vec = arma::vec(n_dim, arma::fill::ones) * 1e9;
    const gpflash::OptzRegion optz_region_clamped = build_optz_region(
            train_data, x_center_vec, f_train_residual_var, optz_opts, optz_seeding_outside, hp_optz_layout, "test");
    const bool seeding_clamped = arma::all(
                                         optz_region_clamped.start_vec.head(n_dim_u)
                                         == optz_region_clamped.constraint_ub_vec.head(n_dim_u))
            && arma::all(optz_region_clamped.sample_lb_vec.head(n_dim_u)
                         == optz_region_clamped.constraint_lb_vec.head(n_dim_u))
            && arma::all(optz_region_clamped.sample_ub_vec.head(n_dim_u)
                         == optz_region_clamped.constraint_ub_vec.head(n_dim_u));

    // Malformed: wrong width, one corner without the other, factor of 1, negative value, inverted corners.
    const bool width_rejected = rejects([&] {
        gpflash::OptzSeeding broken = optz_seeding;
        broken.hp_gamma_start_vec = arma::vec(n_dim + 1, arma::fill::ones);
        build_optz_region(train_data, x_center_vec, f_train_residual_var, optz_opts, broken, hp_optz_layout, "test");
    });
    const bool one_corner_rejected = rejects([&] {
        gpflash::OptzSeeding broken = optz_seeding;
        broken.hp_gamma_sample_ub_vec.reset();
        build_optz_region(train_data, x_center_vec, f_train_residual_var, optz_opts, broken, hp_optz_layout, "test");
    });
    const bool unit_factor_rejected = rejects([&] {
        gpflash::OptzSeeding broken = optz_seeding;
        broken.hp_kern_extra_sample_factor = 1.0;
        build_optz_region(train_data, x_center_vec, f_train_residual_var, optz_opts, broken, hp_optz_layout, "test");
    });
    const bool negative_rejected = rejects([&] {
        gpflash::OptzSeeding broken = optz_seeding;
        broken.hp_gamma_start_vec(0) = -1.0;
        build_optz_region(train_data, x_center_vec, f_train_residual_var, optz_opts, broken, hp_optz_layout, "test");
    });
    const bool inverted_rejected = rejects([&] {
        gpflash::OptzSeeding broken = optz_seeding;
        std::swap(broken.hp_gamma_sample_lb_vec, broken.hp_gamma_sample_ub_vec);
        build_optz_region(train_data, x_center_vec, f_train_residual_var, optz_opts, broken, hp_optz_layout, "test");
    });
    const bool seeding_rejected =
            width_rejected && one_corner_rejected && unit_factor_rejected && negative_rejected && inverted_rejected;

    // The extras band is [init / factor, init * factor] around hp_kern_extra_init, on the extras block only. A
    // non-unit init, since log(1) = 0 would hide a band that ignores it.
    const gpflash::HpOptzLayout hp_optz_layout_extra =
            build_hp_optz_layout(n_dim, 1, gpflash::NoiseRequest::Fixed, "test_optz_seeding");
    gpflash::OptzOptions optz_opts_extra = optz_opts;
    optz_opts_extra.hp_kern_extra_init = 2.5;
    gpflash::OptzSeeding optz_seeding_extra;
    optz_seeding_extra.hp_kern_extra_sample_factor = 4.0;
    const gpflash::OptzRegion optz_region_extra = build_optz_region(
            train_data,
            x_center_vec,
            f_train_residual_var,
            optz_opts_extra,
            optz_seeding_extra,
            hp_optz_layout_extra,
            "test");
    const int idx_extra = hp_optz_layout_extra.offset(gpflash::HpBlock::KernExtra);
    const bool extra_band_applied =
            std::abs(optz_region_extra.sample_lb_vec(idx_extra) - std::log(optz_opts_extra.hp_kern_extra_init / 4.0))
                    < kTolLogSpace
            && std::abs(optz_region_extra.sample_ub_vec(idx_extra) - std::log(optz_opts_extra.hp_kern_extra_init * 4.0))
                    < kTolLogSpace
            && optz_region_extra.sample_lb_vec(0) == optz_region_extra.constraint_lb_vec(0);

    // A kernel that does not override calc_optz_seeding takes the default.
    const bool registered_kernel_seeds_nothing = gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::SqExp)
                                                         .calc_optz_seeding(train_data.X_ndim_ntrain_mat)
                                                         .hp_gamma_start_vec.is_empty();

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "default_region=%d default_start=%d applied=%d clamped=%d rejected=%d%d%d%d%d extra_band=%d registered=%d",
            static_cast<int>(default_samples_whole_region),
            static_cast<int>(default_start_ok),
            static_cast<int>(seeding_applied),
            static_cast<int>(seeding_clamped),
            static_cast<int>(width_rejected),
            static_cast<int>(one_corner_rejected),
            static_cast<int>(unit_factor_rejected),
            static_cast<int>(negative_rejected),
            static_cast<int>(inverted_rejected),
            static_cast<int>(extra_band_applied),
            static_cast<int>(registered_kernel_seeds_nothing));
    return report(
            name,
            default_samples_whole_region && default_start_ok && seeding_applied && seeding_clamped && seeding_rejected
                    && extra_band_applied && registered_kernel_seeds_nothing,
            detail);
}

int main() {
    std::printf("== GpFlash search space ==\n");
    bool ok = true;
    ok &= test_search_space_checks("search_space_checks");
    ok &= test_optz_seeding("optz_seeding");
    std::printf("\n%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");

    return ok ? 0 : 1;
}
