// GpPosterior for the radial kernels. What this file asserts, each behavior in its own function:
//
//   1) mean_vs_reference      the mean matches a brute-force reference that forms Sigma^-1 explicitly
//   2) var_vs_reference       the variance matches that same reference
//   3) mean_grad_vs_fd        d(mean)/d(x') matches a finite difference of the library's own mean
//   4) var_grad_vs_fd         d(var)/d(x') matches a finite difference of the library's own variance
//   5) batch_vs_reference     the batch path's column 0 matches the reference, at two widths
//   6) batch_vs_single_grad   the batch gradients match the single-query gradients
//   7) subset_request         a partial request reproduces the full eval's fields bit-for-bit
//   8) unrequested_outputs    an unselected field comes back empty rather than zero-filled
//   9) empty_request_is_empty  asking for nothing returns an all-empty result, without throwing
//  10) train_point_var        the variance at every training point is at most the noise variance
//  11) mean_at_strong_mean_model_nugget  the mean matches the reference where the mean model nugget is large enough to
//                                  show
//  12) var_clamped_at_zero    a negative raw variance comes back as zero, single and batch
//  13) var_nan_kept           a NaN raw variance comes back as NaN, single and batch
//  14) far_query_is_prior     a query whose scaled squared distance overflows gets the far-field answer, single and
//                             batch, with zero gradients
//
// Checks 1-10 run across a range of (n_train, n_dim), and for SqExp across three mean models: the constant, the zero,
// and one whose degrees cycle from MEAN_MODEL_DEGREE_MAX down to 0 over the dimensions, whose prior mean gradient term
// the mean gradient check then covers. Checks 11 to 14 use fixed cases.
//
// The non-radial kernels are tested in test_gpflash_posterior_nonradial.cpp: the reference below has a unit diagonal
// and evaluates on raw points, which neither a varying-diagonal nor a feature-mapped kernel satisfies.
#include <armadillo>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>

#include "GpFlashTestSupport.hpp"
#include "gpflash/kernels/KernelDispatch.hpp"  // kernel_by_id
#include "gpflash/posterior_build/PosteriorBuild.hpp"
#include "gpflash/posterior_eval/GpPosterior.hpp"
#include "gpflash/posterior_eval/PosteriorEval.hpp"  // detail::calc_var_post_val, calc_var_post_val_vec

using gpflash_test::draw_normal;
using gpflash_test::kTolReferenceImpl;

// Added to a denominator so a relative error stays finite where the reference is ~0; negligible against
// the O(1)-or-larger quantities compared here.
constexpr double kRelDenomFloor = 1e-12;

// Finite-difference bounds. The variance needs the looser one: sigma^2 = k_vec(0) - ||L^-1 k_vec||^2
// subtracts two close quantities, and differencing that across a step loses further digits. The mean has no
// such subtraction, so its gradient holds a bound ten times tighter.
constexpr double kTolMeanGradFd = 1e-6;

// The posterior's step, above kFdStepKern: an ill-conditioned fit (n_train = 400) evaluates the mean with ~1e-12
// relative noise, which a 1e-6 step lifts to the tolerance; 1e-5 divides it ten times less, truncation still ~1e-10.
constexpr double kFdStepPost = 1e-5;

constexpr double kTolVarGradFd = 1e-5;

// The batch path against the reference, and its gradients against the single query's; looser than
// kTolReferenceImpl, which the single path meets against the same reference.
constexpr double kTolBatch = 1e-7;

// The variance at a training point, as a fraction of the noise variance. Observing a point cannot leave more
// uncertainty than the noise on that observation, so 1 is the ceiling; the slack absorbs roundoff only.
constexpr double kTolTrainPointVar = 1.0 + 1e-9;

// Field order: return_mean, return_var, return_mean_gradient, return_var_gradient.
constexpr gpflash::PosteriorOptions opts_mean_only{true, false, false, false};

constexpr gpflash::PosteriorOptions opts_mean_var{true, true, false, false};

constexpr gpflash::PosteriorOptions opts_mean_grad{true, false, true, false};

constexpr gpflash::PosteriorOptions opts_all{true, true, true, true};

constexpr gpflash::PosteriorOptions opts_var_grad_only{false, true, false, true};

// ── One case ────────────────────────────────────────────────────────────────

struct PostCase {
    gpflash::TrainingData train_data;
    gpflash::HpVal hp_val_struct;
    gpflash::ModelOptions model_opts;
    arma::vec x_post_vec;
    // Batch queries holding x_post_vec in column 0. Matrix batches use the Level-3 triangular solve, so two
    // widths exercise its shapes.
    arma::mat X_post_narrow_mat;  // 5 queries
    arma::mat X_post_wide_mat;    // 12 queries
    double ref_mean = 0.0, ref_var = 0.0;
};

// Relative difference, floored so a near-zero reference stays finite.
static double rel_err(const double val, const double ref) {
    return std::abs(val - ref) / (std::abs(ref) + kRelDenomFloor);
}

static double inf_diff(const arma::mat& A_mat, const arma::mat& B_mat) {
    return arma::norm(arma::vectorise(A_mat - B_mat), "inf");
}

// ── The independent reference ───────────────────────────────────────────────

// Reference kernel value between gamma-scaled points; only the scalar kernel comes from the library.
static double calc_ref_kern_val(
        const gpflash::kernel::KernelBase& kern_fun,
        const arma::vec& hp_gamma_vec,
        double hp_rho,
        const arma::vec& p_vec,
        const arma::vec& q_vec) {
    arma::vec rdot_vec = hp_gamma_vec % (p_vec - q_vec);
    // A kernel's per-element math is private, so one value comes from a one-element call.
    double kern_val = arma::dot(rdot_vec, rdot_vec);
    kern_fun.calc_kern_val_inplace(&kern_val, 1, &hp_rho);
    return kern_val;
}

struct RefPosterior {
    double mean = 0.0;
    double var = 0.0;
};

// Brute-force reference posterior mean + variance at a single query x_post_vec.
static RefPosterior calc_ref_posterior(
        const gpflash::kernel::KernelBase& kern_fun,
        const arma::mat& X_ndim_ntrain_mat,
        const arma::vec& f_train_vec,
        const arma::mat& Basis_ntrain_nbasis_mat,
        const arma::vec& basis_post_vec,
        const arma::vec& hp_gamma_vec,
        double hp_rho,
        double noise,
        double cond_max,
        double hp_sig2_k,
        const arma::vec& x_post_vec) {
    const int n_train = X_ndim_ntrain_mat.n_cols;
    arma::mat S_mat(n_train, n_train);
    for (int i_row = 0; i_row < n_train; ++i_row) {
        S_mat(i_row, i_row) = 1.0;
        for (int i_col = i_row + 1; i_col < n_train; ++i_col) {
            double kern_val = calc_ref_kern_val(
                    kern_fun, hp_gamma_vec, hp_rho, X_ndim_ntrain_mat.col(i_row), X_ndim_ntrain_mat.col(i_col));
            S_mat(i_row, i_col) = S_mat(i_col, i_row) = kern_val;
        }
    }
    // Gershgorin eta, eta = eig_max_ub_M_precon/(cond_max - 1), with the bound
    // taken as the largest full absolute row sum max_i sum_j |K_ij| (diagonal included).
    const double nugget = arma::sum(arma::abs(S_mat), 1).max() / (cond_max - 1.0);
    // This reference works in Sigma / hp_sig2_k, where the library works in Sigma itself, so agreeing across the
    // change of coordinates checks the answer rather than the arrangement.
    // S = K + (eta + noise/hp_sig2_k) I, so hp_sig2_k * S is the GpFlash paper's hp_sig2_k (K + eta I) + noise I.
    S_mat.diag() += nugget + noise / hp_sig2_k;

    arma::mat Sinv_mat = arma::inv_sympd(S_mat);
    // The reference mean model weights, then the residual solve. None for the zero mean model.
    const arma::mat lin_sol_sigma_Basis_ntrain_nbasis_mat = Sinv_mat * Basis_ntrain_nbasis_mat;
    const arma::vec lin_sol_sigma_f_train_vec = Sinv_mat * f_train_vec;
    const arma::vec hp_beta_ref_vec = gpflash_test::calc_hp_beta_ref_vec(
            Basis_ntrain_nbasis_mat, lin_sol_sigma_Basis_ntrain_nbasis_mat, lin_sol_sigma_f_train_vec, cond_max);
    const arma::vec residual_vec = f_train_vec - Basis_ntrain_nbasis_mat * hp_beta_ref_vec;
    arma::vec lin_sol_sigma_residual_vec = Sinv_mat * residual_vec;
    arma::vec k_vec(n_train);
    for (int i_train = 0; i_train < n_train; ++i_train)
        k_vec(i_train) = calc_ref_kern_val(kern_fun, hp_gamma_vec, hp_rho, x_post_vec, X_ndim_ntrain_mat.col(i_train));
    // The mean is hp_sig2_k-independent (the factor cancels through lin_sol_sigma_residual); the variance
    // carries it as an outer factor.
    RefPosterior ref;
    ref.mean = arma::dot(basis_post_vec, hp_beta_ref_vec) + arma::dot(k_vec, lin_sol_sigma_residual_vec);
    ref.var = hp_sig2_k * (1.0 - arma::dot(k_vec, Sinv_mat * k_vec));

    return ref;
}

// The mean models a case can carry.
enum class MeanModelKind { Constant, Zero, Monomial };

// The inputs for one (n_train, n_dim) case and the reference answer at the query. hp_sig2_k, the noise and
// hp_gamma are non-unit and mutually unequal, so a dropped, duplicated or swapped factor cannot cancel
// between library and reference. hp_gamma stays near unit scale: larger values underflow the kernel at
// n_dim = 10, zeroing the gradient.
static PostCase make_post_case(
        const gpflash::kernel::KernelBase& kern_fun,
        const int n_train,
        const int n_dim,
        const double hp_rho,
        const MeanModelKind mean_model_kind,
        const double cond_max,
        std::mt19937_64& rng) {
    const double noise = 0.013, hp_sig2_k = 2.7;
    PostCase test_case;

    test_case.train_data.X_ndim_ntrain_mat.set_size(n_dim, n_train);
    for (auto& v : test_case.train_data.X_ndim_ntrain_mat) v = draw_normal(rng);
    test_case.train_data.f_train_vec.set_size(n_train);
    for (auto& v : test_case.train_data.f_train_vec) v = draw_normal(rng);
    test_case.train_data.sig2_f_val = noise;

    test_case.hp_val_struct.hp_gamma_vec.set_size(n_dim);
    for (int i_dim = 0; i_dim < n_dim; ++i_dim) test_case.hp_val_struct.hp_gamma_vec(i_dim) = 0.72 + 0.051 * i_dim;
    if (kern_fun.n_hp_kern_extra(n_dim) > 0) test_case.hp_val_struct.hp_kern_extra_vec = arma::vec{hp_rho};
    test_case.hp_val_struct.hp_sig2_k = hp_sig2_k;

    // The reference reads cond_max back off the struct so it cannot drift.
    test_case.model_opts.cond_max = cond_max;
    test_case.model_opts.kernel_id = kern_fun.kernel_id();
    test_case.model_opts.has_mean_model_constant = mean_model_kind != MeanModelKind::Zero;
    // Degrees cycling from MEAN_MODEL_DEGREE_MAX down to 0 over the dimensions: every degree is present, and n_basis
    // stays below the smallest n_train at n_dim = 10.
    if (mean_model_kind == MeanModelKind::Monomial) {
        test_case.model_opts.mean_model_degree_vec.set_size(n_dim);
        for (int i_dim = 0; i_dim < n_dim; ++i_dim)
            test_case.model_opts.mean_model_degree_vec(i_dim) =
                    gpflash::MEAN_MODEL_DEGREE_MAX - i_dim % (gpflash::MEAN_MODEL_DEGREE_MAX + 1);
    }

    test_case.x_post_vec.set_size(n_dim);
    for (auto& v : test_case.x_post_vec) v = draw_normal(rng);

    for (arma::mat* X_post_mat : {&test_case.X_post_narrow_mat, &test_case.X_post_wide_mat}) {
        const int n_post = (X_post_mat == &test_case.X_post_narrow_mat) ? 5 : 12;
        X_post_mat->set_size(n_dim, n_post);
        X_post_mat->col(0) = test_case.x_post_vec;
        for (int i_post = 1; i_post < n_post; ++i_post)
            for (int i_dim = 0; i_dim < n_dim; ++i_dim) (*X_post_mat)(i_dim, i_post) = draw_normal(rng);
    }

    const bool has_mean_model_constant = test_case.model_opts.has_mean_model_constant;
    // The broadcast of a single degree is the library's, not the reference's, so the reference is always given one
    // degree per dimension.
    const bool is_broadcast = test_case.model_opts.mean_model_degree_vec.n_elem == 1;
    arma::uvec mean_model_degree_vec;
    if (is_broadcast) {
        mean_model_degree_vec = arma::uvec(n_dim).fill(test_case.model_opts.mean_model_degree_vec(0));
    } else {
        mean_model_degree_vec = test_case.model_opts.mean_model_degree_vec;
    }
    const arma::mat Basis_ntrain_nbasis_mat = gpflash_test::build_mean_model_basis_ref_mat(
            test_case.train_data.X_ndim_ntrain_mat, has_mean_model_constant, mean_model_degree_vec);
    const arma::vec basis_post_vec = gpflash_test::build_mean_model_basis_ref_mat(
                                             test_case.x_post_vec, has_mean_model_constant, mean_model_degree_vec)
                                             .t();
    const RefPosterior ref = calc_ref_posterior(
            kern_fun,
            test_case.train_data.X_ndim_ntrain_mat,
            test_case.train_data.f_train_vec,
            Basis_ntrain_nbasis_mat,
            basis_post_vec,
            test_case.hp_val_struct.hp_gamma_vec,
            hp_rho,
            noise,
            test_case.model_opts.cond_max,
            hp_sig2_k,
            test_case.x_post_vec);
    test_case.ref_mean = ref.mean;
    test_case.ref_var = ref.var;

    return test_case;
}

// ── The checks, one behavior each ───────────────────────────────────────────

// The mean matches the reference, however it was requested.
static double calc_mean_vs_reference_err(const PostCase& test_case, const gpflash::GpPosterior& gp) {
    return std::max(
            {rel_err(gp.eval(test_case.x_post_vec, opts_mean_only).mean_post_vec(0), test_case.ref_mean),
             rel_err(gp.eval(test_case.x_post_vec, opts_mean_var).mean_post_vec(0), test_case.ref_mean),
             rel_err(gp.eval(test_case.x_post_vec, opts_all).mean_post_vec(0), test_case.ref_mean)});
}

// The variance matches the reference, however it was requested. Measured against the prior variance hp_sig2_k, not
// the posterior variance: sigma^2 = hp_sig2_k (k(x,x) - ||L^-1 k_vec||^2) subtracts two O(hp_sig2_k) quantities, so
// its rounding error is cond * eps * hp_sig2_k however small the result, and a dense fit (n_train = 400) leaves a
// posterior variance hundreds of times smaller than the prior.
static double calc_var_vs_reference_err(const PostCase& test_case, const gpflash::GpPosterior& gp) {
    const double scale = test_case.hp_val_struct.hp_sig2_k;
    return std::max(
                   std::abs(gp.eval(test_case.x_post_vec, opts_mean_var).var_post_vec(0) - test_case.ref_var),
                   std::abs(gp.eval(test_case.x_post_vec, opts_all).var_post_vec(0) - test_case.ref_var))
            / scale;
}

// The mean gradient matches a finite difference of the library's own mean.
static double calc_mean_grad_vs_fd_err(
        const PostCase& test_case, const gpflash::GpPosterior& gp, const gpflash::PosteriorResult& post_result_all) {
    const int n_dim = static_cast<int>(test_case.x_post_vec.n_elem);
    arma::vec fd_mean_vec(n_dim);
    for (int i_dim = 0; i_dim < n_dim; ++i_dim) {
        arma::vec x_post_plus_vec = test_case.x_post_vec, x_post_minus_vec = test_case.x_post_vec;
        x_post_plus_vec(i_dim) += kFdStepPost;
        x_post_minus_vec(i_dim) -= kFdStepPost;
        fd_mean_vec(i_dim) = (gp.eval(x_post_plus_vec, opts_mean_only).mean_post_vec(0)
                              - gp.eval(x_post_minus_vec, opts_mean_only).mean_post_vec(0))
                / (2 * kFdStepPost);
    }
    const arma::vec d_mean_vec = post_result_all.d_mean_post_d_x_post_ndim_npost_mat.col(0);

    return arma::norm(d_mean_vec - fd_mean_vec, "inf") / (arma::norm(fd_mean_vec, "inf") + kRelDenomFloor);
}

// The variance gradient matches a finite difference of the library's own variance.
static double calc_var_grad_vs_fd_err(
        const PostCase& test_case, const gpflash::GpPosterior& gp, const gpflash::PosteriorResult& post_result_all) {
    const int n_dim = static_cast<int>(test_case.x_post_vec.n_elem);
    arma::vec fd_var_vec(n_dim);
    for (int i_dim = 0; i_dim < n_dim; ++i_dim) {
        arma::vec x_post_plus_vec = test_case.x_post_vec, x_post_minus_vec = test_case.x_post_vec;
        x_post_plus_vec(i_dim) += kFdStepPost;
        x_post_minus_vec(i_dim) -= kFdStepPost;
        fd_var_vec(i_dim) = (gp.eval(x_post_plus_vec, opts_mean_var).var_post_vec(0)
                             - gp.eval(x_post_minus_vec, opts_mean_var).var_post_vec(0))
                / (2 * kFdStepPost);
    }
    const arma::vec d_var_vec = post_result_all.d_var_post_d_x_post_ndim_npost_mat.col(0);

    return arma::norm(d_var_vec - fd_var_vec, "inf") / (arma::norm(fd_var_vec, "inf") + kRelDenomFloor);
}

// A partial request reproduces the full eval's fields bit for bit: the same expressions run, only the
// skipped ones are absent. mean+mean-gradient skips the triangular solve; variance+variance-gradient skips
// the mean dot.
static double calc_subset_request_diff(
        const PostCase& test_case, const gpflash::GpPosterior& gp, const gpflash::PosteriorResult& post_result_all) {
    const gpflash::PosteriorResult post_result_mean_grad = gp.eval(test_case.x_post_vec, opts_mean_grad);
    const gpflash::PosteriorResult post_result_var_grad = gp.eval(test_case.x_post_vec, opts_var_grad_only);

    return std::max(
            {std::abs(post_result_mean_grad.mean_post_vec(0) - post_result_all.mean_post_vec(0)),
             inf_diff(
                     post_result_mean_grad.d_mean_post_d_x_post_ndim_npost_mat,
                     post_result_all.d_mean_post_d_x_post_ndim_npost_mat),
             std::abs(post_result_var_grad.var_post_vec(0) - post_result_all.var_post_vec(0)),
             inf_diff(
                     post_result_var_grad.d_var_post_d_x_post_ndim_npost_mat,
                     post_result_all.d_var_post_d_x_post_ndim_npost_mat)});
}

// An unrequested quantity is left empty rather than zero-filled, so a caller that forgot the flag gets a
// shape error instead of a plausible wrong number.
static bool check_unrequested_outputs_empty(const PostCase& test_case, const gpflash::GpPosterior& gp) {
    const auto n_dim_u = static_cast<arma::uword>(test_case.x_post_vec.n_elem);
    const gpflash::PosteriorResult mean_only = gp.eval(test_case.x_post_vec, opts_mean_only);
    const gpflash::PosteriorResult mean_var = gp.eval(test_case.x_post_vec, opts_mean_var);
    const gpflash::PosteriorResult mean_grad = gp.eval(test_case.x_post_vec, opts_mean_grad);
    const gpflash::PosteriorResult var_grad = gp.eval(test_case.x_post_vec, opts_var_grad_only);
    const gpflash::PosteriorResult all = gp.eval(test_case.x_post_vec, opts_all);

    const bool mean_only_ok = mean_only.mean_post_vec.n_elem == 1 && mean_only.var_post_vec.is_empty()
            && mean_only.d_mean_post_d_x_post_ndim_npost_mat.is_empty()
            && mean_only.d_var_post_d_x_post_ndim_npost_mat.is_empty();
    const bool mean_var_ok =
            mean_var.var_post_vec.n_elem == 1 && mean_var.d_mean_post_d_x_post_ndim_npost_mat.is_empty();
    const bool mean_grad_ok =
            mean_grad.var_post_vec.is_empty() && mean_grad.d_var_post_d_x_post_ndim_npost_mat.is_empty();
    const bool var_grad_ok = var_grad.mean_post_vec.is_empty()
            && var_grad.d_mean_post_d_x_post_ndim_npost_mat.is_empty()
            && var_grad.d_var_post_d_x_post_ndim_npost_mat.n_rows == n_dim_u;
    const bool all_ok = all.d_mean_post_d_x_post_ndim_npost_mat.n_rows == n_dim_u
            && all.d_mean_post_d_x_post_ndim_npost_mat.n_cols == 1;

    return mean_only_ok && mean_var_ok && mean_grad_ok && var_grad_ok && all_ok;
}

// A request for nothing returns an empty result rather than throwing, and rather than filling a field nobody
// asked for. Every field empty is the same thing each field already means when its own flag is clear.
static bool check_empty_request_is_empty(const PostCase& test_case, const gpflash::GpPosterior& gp) {
    const gpflash::PosteriorResult empty =
            gp.eval(test_case.x_post_vec, gpflash::PosteriorOptions{false, false, false, false});

    return empty.mean_post_vec.is_empty() && empty.var_post_vec.is_empty()
            && empty.d_mean_post_d_x_post_ndim_npost_mat.is_empty()
            && empty.d_var_post_d_x_post_ndim_npost_mat.is_empty();
}

// The batch path's mean and variance at column 0 match the reference, at both widths and for the four requests
// that include the mean. mean-only and mean+var skip c^(rdot), which only a gradient reads.
// The last column is compared against a single query, so a batch prior mean correct only at column 0 fails.
static double calc_batch_vs_reference_err(const PostCase& test_case, const gpflash::GpPosterior& gp) {
    const gpflash::PosteriorResult narrow_all = gp.eval(test_case.X_post_narrow_mat, opts_all);
    const gpflash::PosteriorResult wide_all = gp.eval(test_case.X_post_wide_mat, opts_all);
    const gpflash::PosteriorResult wide_mean_grad = gp.eval(test_case.X_post_wide_mat, opts_mean_grad);
    const gpflash::PosteriorResult wide_mean = gp.eval(test_case.X_post_wide_mat, opts_mean_only);
    const gpflash::PosteriorResult wide_mean_var = gp.eval(test_case.X_post_wide_mat, opts_mean_var);

    const arma::uword idx_post_last = test_case.X_post_wide_mat.n_cols - 1;
    const gpflash::PosteriorResult post_result_last_single =
            gp.eval(arma::vec(test_case.X_post_wide_mat.col(idx_post_last)), opts_mean_only);

    return std::max(
            {rel_err(narrow_all.mean_post_vec(0), test_case.ref_mean),
             rel_err(narrow_all.var_post_vec(0), test_case.ref_var),
             rel_err(wide_all.mean_post_vec(0), test_case.ref_mean),
             rel_err(wide_all.var_post_vec(0), test_case.ref_var),
             rel_err(wide_mean_grad.mean_post_vec(0), test_case.ref_mean),
             rel_err(wide_mean.mean_post_vec(0), test_case.ref_mean),
             rel_err(wide_mean_var.mean_post_vec(0), test_case.ref_mean),
             rel_err(wide_mean_var.var_post_vec(0), test_case.ref_var),
             rel_err(wide_mean.mean_post_vec(idx_post_last), post_result_last_single.mean_post_vec(0))});
}

// The batch gradients match the single-query gradients at column 0 and at the last column, at both widths, and a
// batch mean+gradient request matches the full batch eval over every column. The last column is what catches a
// batch path that gets column 0 right and the rest wrong: a stale scale, or a misordered index.
static double calc_batch_vs_single_grad_err(
        const PostCase& test_case, const gpflash::GpPosterior& gp, const gpflash::PosteriorResult& post_result_all) {
    const arma::vec d_mean_vec = post_result_all.d_mean_post_d_x_post_ndim_npost_mat.col(0);
    const arma::vec d_var_vec = post_result_all.d_var_post_d_x_post_ndim_npost_mat.col(0);
    const double d_mean_scale = arma::norm(d_mean_vec, "inf") + kRelDenomFloor;
    const double d_var_scale = arma::norm(d_var_vec, "inf") + kRelDenomFloor;

    const gpflash::PosteriorResult narrow_all = gp.eval(test_case.X_post_narrow_mat, opts_all);
    const gpflash::PosteriorResult wide_all = gp.eval(test_case.X_post_wide_mat, opts_all);
    const gpflash::PosteriorResult wide_mean_grad = gp.eval(test_case.X_post_wide_mat, opts_mean_grad);

    const arma::uword i_last = test_case.X_post_wide_mat.n_cols - 1;
    const gpflash::PosteriorResult last_single = gp.eval(arma::vec(test_case.X_post_wide_mat.col(i_last)), opts_all);

    return std::max(
            {arma::norm(narrow_all.d_mean_post_d_x_post_ndim_npost_mat.col(0) - d_mean_vec, "inf") / d_mean_scale,
             arma::norm(narrow_all.d_var_post_d_x_post_ndim_npost_mat.col(0) - d_var_vec, "inf") / d_var_scale,
             arma::norm(wide_all.d_mean_post_d_x_post_ndim_npost_mat.col(0) - d_mean_vec, "inf") / d_mean_scale,
             arma::norm(wide_all.d_var_post_d_x_post_ndim_npost_mat.col(0) - d_var_vec, "inf") / d_var_scale,
             arma::norm(
                     wide_all.d_mean_post_d_x_post_ndim_npost_mat.col(i_last)
                             - last_single.d_mean_post_d_x_post_ndim_npost_mat.col(0),
                     "inf")
                     / d_mean_scale,
             arma::norm(
                     wide_all.d_var_post_d_x_post_ndim_npost_mat.col(i_last)
                             - last_single.d_var_post_d_x_post_ndim_npost_mat.col(0),
                     "inf")
                     / d_var_scale,
             inf_diff(wide_mean_grad.d_mean_post_d_x_post_ndim_npost_mat, wide_all.d_mean_post_d_x_post_ndim_npost_mat)
                     / d_mean_scale});
}

// The largest variance over the training points, as a fraction of the noise variance. A training point is where
// k(x', x') and sigma_k^2 ||L^-1 k_*||^2 come closest, so it is where the result is most constrained.
static double calc_worst_train_point_var_ratio(const PostCase& test_case, const gpflash::GpPosterior& gp) {
    const double noise_var = test_case.train_data.sig2_f_val;
    if (!(noise_var > 0.0)) return arma::datum::inf;  // every PostCase is a noisy fit, which the ratio below needs

    double worst_ratio = 0.0;
    for (arma::uword i_train = 0; i_train < test_case.train_data.X_ndim_ntrain_mat.n_cols; ++i_train) {
        const arma::vec x_train_vec(test_case.train_data.X_ndim_ntrain_mat.col(i_train));
        const double var_post_val = gp.eval(x_train_vec, opts_mean_var).var_post_vec(0);
        worst_ratio = std::max(worst_ratio, var_post_val / noise_var);
    }

    return worst_ratio;
}

// The mean matches the reference at a cond_max whose mean model nugget moves it past the reference tolerance, which
// the default cond_max cannot show, so a posterior built with a cond_max other than the caller's fails.
static bool test_mean_at_strong_mean_model_nugget(const char* name) {
    const gpflash::kernel::KernelBase& kern_fun = gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::SqExp);
    const int n_train = 80;
    std::mt19937_64 rng(7);
    bool pass = true;
    for (const int n_dim : {3, 10}) {
        const PostCase test_case = make_post_case(
                kern_fun, n_train, n_dim, 0.0, MeanModelKind::Monomial, gpflash_test::kCondMaxStrongNugget, rng);
        const gpflash::GpPosterior gp_posterior =
                gpflash::make_posterior(test_case.train_data, test_case.hp_val_struct, test_case.model_opts);
        const double mean_err = calc_mean_vs_reference_err(test_case, gp_posterior);
        const bool case_pass = mean_err < kTolReferenceImpl;
        pass = pass && case_pass;

        char detail[gpflash_test::kDetailLen];
        std::snprintf(detail, sizeof(detail), "n_dim=%d mean=%.1e", n_dim, mean_err);
        gpflash_test::report(name, case_pass, detail);
    }

    return pass;
}

// A raw variance below zero comes back as zero, from both the single and the batch form. The noisy fits above never
// produce one, so the clamp is driven directly with a quadratic form larger than the prior variance.
static bool test_var_clamped_at_zero(const char* name) {
    gpflash::detail::PosteriorPrecompute post_precompute;
    post_precompute.hp_sig2_k = 1.0;
    const double kern_diag_val = 1.0;

    // ||L^-1 k||^2 = 1.21 exceeds k(x', x') = 1, so the raw variance is -0.21.
    const arma::vec lin_sol_L_k_post_vec{1.1};
    const double var_single = gpflash::detail::calc_var_post_val(post_precompute, kern_diag_val, lin_sol_L_k_post_vec);

    // Column 0 repeats the single query; column 1 has ||L^-1 k||^2 = 0.25, so its variance stays 0.75.
    const arma::vec kern_diag_val_vec{kern_diag_val, kern_diag_val};
    const arma::mat lin_sol_L_K_ntrain_npost_mat{{1.1, 0.5}};
    const arma::vec var_batch_vec =
            gpflash::detail::calc_var_post_val_vec(post_precompute, kern_diag_val_vec, lin_sol_L_K_ntrain_npost_mat);
    const bool pass = var_single == 0.0 && var_batch_vec(0) == 0.0 && var_batch_vec(1) == 0.75;

    char detail[gpflash_test::kDetailLen];
    std::snprintf(
            detail, sizeof(detail), "single=%.2e batch=%.2e/%.2e", var_single, var_batch_vec(0), var_batch_vec(1));
    return gpflash_test::report(name, pass, detail);
}

// A NaN raw variance comes back as NaN from both forms, rather than as a zero that reads as certainty.
static bool test_var_nan_kept(const char* name) {
    gpflash::detail::PosteriorPrecompute post_precompute;
    post_precompute.hp_sig2_k = 1.0;
    const arma::vec lin_sol_L_k_post_vec{arma::datum::nan};
    const double var_single = gpflash::detail::calc_var_post_val(post_precompute, 1.0, lin_sol_L_k_post_vec);
    const arma::vec var_batch_vec = gpflash::detail::calc_var_post_val_vec(
            post_precompute, arma::vec{1.0}, arma::mat(lin_sol_L_k_post_vec.t()));
    const bool pass = std::isnan(var_single) && std::isnan(var_batch_vec(0));

    char detail[gpflash_test::kDetailLen];
    std::snprintf(detail, sizeof(detail), "single=%g batch=%g", var_single, var_batch_vec(0));
    return gpflash_test::report(name, pass, detail);
}

// Two queries far along dimension 0: one whose scaled squared distance overflows a double, and a reference at 1e60
// where every kernel here is below 1e-200 but nothing overflows. The far one must give the reference's mean and
// variance, and zero gradients.
static bool test_far_query_is_prior(const char* name) {
    const int n_dim = 2, n_train = 12;
    std::mt19937_64 rng(11);
    gpflash::TrainingData train_data;
    train_data.X_ndim_ntrain_mat = gpflash_test::draw_uniform_mat(rng, n_dim, n_train);
    train_data.f_train_vec = gpflash_test::draw_uniform_vec(rng, n_train);
    gpflash::HpVal hp_val_struct;
    hp_val_struct.hp_gamma_vec = {2.0, 1.5};
    hp_val_struct.hp_sig2_k = 1.3;
    const arma::mat X_post_ndim_npost_mat = {{1e60, 1e160}, {0.4, 0.4}};

    bool pass = true;
    for (const gpflash::kernel::KernelId kernel_id :
         {gpflash::kernel::KernelId::SqExp,
          gpflash::kernel::KernelId::Matern32,
          gpflash::kernel::KernelId::Matern52,
          gpflash::kernel::KernelId::RatQuad}) {
        const gpflash::kernel::KernelBase& kern_fun = gpflash::kernel::kernel_by_id(kernel_id);
        gpflash::ModelOptions model_opts;
        model_opts.kernel_id = kernel_id;
        hp_val_struct.hp_kern_extra_vec = arma::vec(kern_fun.n_hp_kern_extra(n_dim)).fill(2.0);
        const gpflash::GpPosterior gp_posterior = gpflash::make_posterior(train_data, hp_val_struct, model_opts);

        // A non-finite result throws from the postcondition this target compiles in.
        gpflash::PosteriorResult batch, single;
        try {
            batch = gp_posterior.eval(X_post_ndim_npost_mat, opts_all);
            single = gp_posterior.eval(arma::vec(X_post_ndim_npost_mat.col(1)), opts_all);
        } catch (const gpflash::ErrorInvalidInput& error) {
            pass = gpflash_test::report(name, false, error.what()) && pass;
            continue;
        }
        const bool batch_ok = batch.mean_post_vec(1) == batch.mean_post_vec(0)
                && batch.var_post_vec(1) == batch.var_post_vec(0)
                && arma::all(batch.d_mean_post_d_x_post_ndim_npost_mat.col(1) == 0.0)
                && arma::all(batch.d_var_post_d_x_post_ndim_npost_mat.col(1) == 0.0);
        const bool single_ok = single.mean_post_vec(0) == batch.mean_post_vec(0)
                && single.var_post_vec(0) == batch.var_post_vec(0)
                && arma::all(single.d_mean_post_d_x_post_ndim_npost_mat.col(0) == 0.0)
                && arma::all(single.d_var_post_d_x_post_ndim_npost_mat.col(0) == 0.0);
        pass = pass && batch_ok && single_ok;

        char detail[gpflash_test::kDetailLen];
        std::snprintf(
                detail,
                sizeof(detail),
                "%s mean %.6g/%.6g var %.6g/%.6g",
                kern_fun.kernel_name(),
                single.mean_post_vec(0),
                batch.mean_post_vec(0),
                single.var_post_vec(0),
                batch.var_post_vec(0));
        gpflash_test::report(name, batch_ok && single_ok, detail);
    }
    return pass;
}

// ── The driver ──────────────────────────────────────────────────────────────

static bool run_kernel(
        const gpflash::kernel::KernelBase& kern_fun,
        const std::string& name,
        double hp_rho = 0.0,
        const MeanModelKind mean_model_kind = MeanModelKind::Constant) {
    std::mt19937_64 rng(7);
    bool ok = true;
    std::printf("== Kernel: %s ==\n", name.c_str());
    for (int n_train : {15, 80, 400}) {
        for (int n_dim : {3, 10}) {
            const PostCase test_case =
                    make_post_case(kern_fun, n_train, n_dim, hp_rho, mean_model_kind, gpflash::COND_MAX_DEFAULT, rng);
            const auto gp =
                    gpflash::make_posterior(test_case.train_data, test_case.hp_val_struct, test_case.model_opts);
            const gpflash::PosteriorResult post_result_all = gp.eval(test_case.x_post_vec, opts_all);

            const double mean_err = calc_mean_vs_reference_err(test_case, gp);
            const double var_err = calc_var_vs_reference_err(test_case, gp);
            const double mean_fd_err = calc_mean_grad_vs_fd_err(test_case, gp, post_result_all);
            const double var_fd_err = calc_var_grad_vs_fd_err(test_case, gp, post_result_all);
            const double batch_err = calc_batch_vs_reference_err(test_case, gp);
            const double batch_grad_err = calc_batch_vs_single_grad_err(test_case, gp, post_result_all);
            const double subset_err = calc_subset_request_diff(test_case, gp, post_result_all);
            const bool shapes_ok = check_unrequested_outputs_empty(test_case, gp);
            const bool empty_is_empty = check_empty_request_is_empty(test_case, gp);
            const double train_var_ratio = calc_worst_train_point_var_ratio(test_case, gp);

            // A partial request runs the same expressions as the full one, only skipping fields, so the two
            // agree exactly rather than within a tolerance.
            const bool pass = mean_err < kTolReferenceImpl && var_err < kTolReferenceImpl
                    && mean_fd_err < kTolMeanGradFd && var_fd_err < kTolVarGradFd && batch_err < kTolBatch
                    && batch_grad_err < kTolBatch && subset_err == 0.0 && shapes_ok && empty_is_empty
                    && train_var_ratio < kTolTrainPointVar;
            ok &= pass;
            std::printf(
                    "  n_train=%4d n_dim=%2d  mean=%.1e var=%.1e fd_mean=%.1e fd_var=%.1e batch=%.1e "
                    "batch_grad=%.1e subset=%.1e train_var=%.1e shapes=%d empty=%d  %s\n",
                    n_train,
                    n_dim,
                    mean_err,
                    var_err,
                    mean_fd_err,
                    var_fd_err,
                    batch_err,
                    batch_grad_err,
                    subset_err,
                    train_var_ratio,
                    static_cast<int>(shapes_ok),
                    static_cast<int>(empty_is_empty),
                    pass ? "PASS" : "*** FAIL ***");
        }
    }
    return ok;
}

int main() {
    std::printf("== GpFlash posterior, radial kernels ==\n");
    bool ok = true;
    ok &= run_kernel(gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::SqExp), "SqExp");
    ok &= run_kernel(gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Matern52), "Matern52");
    ok &= run_kernel(gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::Matern32), "Matern32");
    ok &= run_kernel(gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::RatQuad), "RatQuad", /*a=*/2.0);
    ok &= run_kernel(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::SqExp),
            "SqExp, zero mean model",
            0.0,
            MeanModelKind::Zero);
    ok &= run_kernel(
            gpflash::kernel::kernel_by_id(gpflash::kernel::KernelId::SqExp),
            "SqExp, cycling-degree mean model",
            0.0,
            MeanModelKind::Monomial);
    ok &= test_mean_at_strong_mean_model_nugget("mean_at_strong_mean_model_nugget");
    ok &= test_var_clamped_at_zero("var_clamped_at_zero");
    ok &= test_var_nan_kept("var_nan_kept");
    ok &= test_far_query_is_prior("far_query_is_prior");
    std::printf("\n%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return ok ? 0 : 1;
}
