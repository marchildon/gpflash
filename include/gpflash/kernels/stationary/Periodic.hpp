/** Periodic.hpp -- periodic kernel: its feature map onto circles, and its derivatives. */
#pragma once

#include <armadillo>  // arma::mat, arma::vec, arma::rowvec
#include <cmath>      // std::cos, std::sin, std::exp
#include <cstddef>    // std::ptrdiff_t

#include "gpflash/core/Errors.hpp"               // gpflash::ErrorInvalidInput
#include "gpflash/core/Platform.hpp"             // GPFLASH_RESTRICT, GPFLASH_OMP_SIMD_REDUCTION, GPFLASH_CHECK
#include "gpflash/kernels/HpGradInputs.hpp"      // HpGradInputs
#include "gpflash/kernels/KernelStationary.hpp"  // KernelStationary

namespace gpflash {
namespace kernel {

// k = exp(-2 sum_d hp_amplitude_d^2 sin^2(theta_d)),   theta_d = pi hp_gamma_d r_d,   r_d = x_id - x_jd.
//
// In raw x that sum is no function of the scaled distance. Placing each coordinate on a circle at the per-point
// phase 2 pi hp_gamma_d x_d makes it one, with hp_gamma the phase frequency and hp_amplitude the feature length
// scales assembly reads. A pair's angle relates to the per-point phase by phase_i - phase_j = 2 theta, so the
// gradients' sin(2 theta) is the sine of the phase difference.
//
// In those feature coordinates the kernel is exp(-0.5 ||rdot_vec||^2), so it derives KernelStationary and its
// value loops below are the squared-exponential's. The feature map and the gradients are the kernel's own,
// carrying hp_gamma through the map rather than as a length scale.
class Periodic final : public KernelStationary {
   public:
    const char* kernel_name() const override { return "periodic"; }
    KernelId kernel_id() const override { return KernelId::Periodic; }

    bool is_symmetrizable() const override { return true; }

    // Negating x_d keeps its cosine axis, row d, and negates its sine axis, row n_dim + d.
    arma::vec calc_feature_mirror_sign_vec(const arma::vec& mirror_sign_vec) const override {
        return arma::join_cols(arma::ones<arma::vec>(mirror_sign_vec.n_elem), mirror_sign_vec);
    }

    // One amplitude per input dimension.
    int n_hp_kern_extra(const int n_dim_input) const override { return n_dim_input; }

    // Two feature axes per input dimension, so the input width is half the evaluation width.
    int calc_n_dim_input(const int n_feature) const override {
        if (n_feature < 0 || n_feature % 2 != 0)
            throw gpflash::ErrorInvalidInput(
                    "gpflash::kernel::Periodic::calc_n_dim_input: the evaluation dimension must be "
                    "nonnegative and even, with one cosine and one sine axis per input dimension.");
        return n_feature / 2;
    }

    // ── Seeding this kernel's fit ─────────────────────────────────────────────
    // hp_gamma is an inverse period and the MLL is multimodal in it, so the shared start 1 / stddev(X_d) is
    // meaningless and sampling the whole bound range wastes restarts.
    OptzSeeding calc_optz_seeding(const arma::mat& X_ndim_ntrain_mat) const override {
        const arma::vec data_range_vec = calc_data_range_vec(X_ndim_ntrain_mat);

        OptzSeeding optz_seeding;
        // One period across the data: a longer one cannot be identified from it.
        optz_seeding.hp_gamma_start_vec = calc_inv_period_vec(data_range_vec, 1.0);
        optz_seeding.hp_gamma_sample_lb_vec = optz_seeding.hp_gamma_start_vec;
        optz_seeding.hp_gamma_sample_ub_vec = calc_inv_period_vec(data_range_vec, SAMPLE_N_PERIOD_ACROSS_DATA_MAX);
        optz_seeding.hp_kern_extra_sample_factor = SAMPLE_HP_AMPLITUDE_FACTOR;

        return optz_seeding;
    }

    // ── The points, once each coordinate is on a circle ───────────────────────
    // Y: rows d and n_dim + d hold dimension d's cosine and sine axes at the per-point phase
    // 2 pi hp_gamma_d (x_d - x_center_d).
    arma::mat calc_feature_points(
            const arma::mat& X_ndim_npoint_mat,
            const arma::vec& x_center_vec,
            const arma::vec& hp_gamma_vec,
            const arma::vec& /*hp_amplitude_vec*/) const override {
        const int n_dim = static_cast<int>(X_ndim_npoint_mat.n_rows);
        const int n_point = static_cast<int>(X_ndim_npoint_mat.n_cols);
        arma::mat Y_nfeature_npoint_mat(2 * n_dim, n_point, arma::fill::none);  // every entry is written below

        for (int i_dim = 0; i_dim < n_dim; ++i_dim) {
            const double phase_per_x = 2.0 * arma::datum::pi * hp_gamma_vec(i_dim);
            const double x_center = x_center_vec(i_dim);
            for (int i_point = 0; i_point < n_point; ++i_point) {
                const double phase = phase_per_x * (X_ndim_npoint_mat(i_dim, i_point) - x_center);
                Y_nfeature_npoint_mat(i_dim, i_point) = std::cos(phase);
                Y_nfeature_npoint_mat(n_dim + i_dim, i_point) = std::sin(phase);
            }
        }

        return Y_nfeature_npoint_mat;
    }

    // Ydot: those features scaled by hp_amplitude in place, so the scaling costs no second matrix. Both axes of
    // dimension d take the amplitude hp_amplitude_d, so the scale vector repeats it across the cosine and sine axes.
    arma::mat calc_scaled_feature_points(
            const arma::mat& X_ndim_npoint_mat,
            const arma::vec& x_center_vec,
            const arma::vec& hp_gamma_vec,
            const arma::vec& hp_amplitude_vec) const override {
        arma::mat Ydot_nfeature_npoint_mat =
                calc_feature_points(X_ndim_npoint_mat, x_center_vec, hp_gamma_vec, hp_amplitude_vec);
        Ydot_nfeature_npoint_mat.each_col() %= arma::join_cols(hp_amplitude_vec, hp_amplitude_vec);

        return Ydot_nfeature_npoint_mat;
    }

    // Single-point counterpart to the two above, fused: same phase, same hp_amplitude scaling, written straight
    // into the (2 n_dim) result so the one-query path allocates nothing rectangular. Must produce the same
    // features as the matrix forms above, since the single- and batch-query paths must agree.
    arma::vec calc_scaled_feature_point_vec(
            const arma::vec& x_ndim_vec,
            const arma::vec& x_center_vec,
            const arma::vec& hp_gamma_vec,
            const arma::vec& hp_amplitude_vec) const override {
        const int n_dim = static_cast<int>(x_ndim_vec.n_elem);
        arma::vec ydot_nfeature_vec(2 * n_dim, arma::fill::none);  // every entry is written below
        for (int i_dim = 0; i_dim < n_dim; ++i_dim) {
            const double phase =
                    2.0 * arma::datum::pi * hp_gamma_vec(i_dim) * (x_ndim_vec(i_dim) - x_center_vec(i_dim));
            const double hp_amplitude = hp_amplitude_vec(i_dim);
            ydot_nfeature_vec(i_dim) = hp_amplitude * std::cos(phase);
            ydot_nfeature_vec(n_dim + i_dim) = hp_amplitude * std::sin(phase);
        }

        return ydot_nfeature_vec;
    }

    // ── The value, squared-exponential in feature coordinates ─────────────────
    // hp_amplitude is already in the feature length scales, so the loops ignore the extras pointer. s^(k) is 1,
    // so the value is c^(k) itself and only c^(rdot) = s^(rdot) c^(k) needs a product, with s^(rdot) = -1.
    void calc_kern_val_inplace(
            double* GPFLASH_RESTRICT rdot_sq_norm_ptr,
            const std::ptrdiff_t n_elem,
            const double* /*hp_kern_extra_ptr*/) const override {
        for (std::ptrdiff_t i_elem = 0; i_elem < n_elem; ++i_elem)
            rdot_sq_norm_ptr[i_elem] = std::exp(-0.5 * rdot_sq_norm_ptr[i_elem]);
    }

    void calc_kern_val_and_common_rdot_inplace(
            double* GPFLASH_RESTRICT rdot_sq_norm_ptr,
            const std::ptrdiff_t n_elem,
            const double* /*hp_kern_extra_ptr*/,
            double* GPFLASH_RESTRICT kern_common_rdot_ptr_out) const override {
        for (std::ptrdiff_t i_elem = 0; i_elem < n_elem; ++i_elem) {
            const double kern_common_k = std::exp(-0.5 * rdot_sq_norm_ptr[i_elem]);
            rdot_sq_norm_ptr[i_elem] = kern_common_k;
            kern_common_rdot_ptr_out[i_elem] = -kern_common_k;
        }
    }

    // ── The hyperparameter gradient ───────────────────────────────────────────
    // alpha = gamma and alpha = hp_amplitude, both accumulated in one pass over [idx_b_begin, idx_b_end), with
    // A_ptr already carrying c^(rdot):
    //
    //     sum_row_a = sum_b A_ab h^(gamma,l)_ab   into d_mll_d_hp_gamma_ptr
    //     sum_row_a = sum_b A_ab h^(a,l)_ab       into d_mll_d_hp_kern_extra_ptr
    //
    // with, from k = exp(-2 sum_d hp_amplitude_d^2 sin^2(theta_d)) and theta_d = pi hp_gamma_d r_d,
    //
    //     h^(gamma,l) = 2 pi hp_amplitude_l^2 r_l sin(2 theta_l),   h^(a,l) = 4 hp_amplitude_l sin^2(theta_l)
    //
    // Here 2 theta_l is the phase difference phase_i - phase_j, so sin(2 theta) and sin^2(theta) come from the
    // two points' stored cosines and sines by the angle-difference identities, calling no trigonometric function.
    // This kernel always declares hp_amplitude, so the caller always provides d_mll_d_hp_kern_extra_ptr and both
    // sums are written.
    void add_hp_gradient_contribution(
            const HpGradInputs& hp_grad_inputs,
            const int idx_a,
            const int idx_b_begin,
            const int idx_b_end,
            const double* GPFLASH_RESTRICT A_ptr,
            double* GPFLASH_RESTRICT d_mll_d_hp_gamma_ptr,
            double* GPFLASH_RESTRICT d_mll_d_hp_kern_extra_ptr) const override {
        add_hp_gamma_and_amplitude_gradient_contribution</*kHasMirror=*/false>(
                hp_grad_inputs,
                idx_a,
                idx_b_begin,
                idx_b_end,
                A_ptr,
                nullptr,
                d_mll_d_hp_gamma_ptr,
                d_mll_d_hp_kern_extra_ptr);
    }

    // The same at the pair (x_a, S x_b), for the mirrored term of a symmetry model. Negating x_bl keeps its cosine
    // and negates its sine, so only the sine and the separation take the sign.
    void add_hp_gradient_contribution_mirrored(
            const HpGradInputs& hp_grad_inputs,
            const int idx_a,
            const int idx_b_begin,
            const int idx_b_end,
            const double* GPFLASH_RESTRICT A_ptr,
            const arma::vec& mirror_sign_vec,
            double* GPFLASH_RESTRICT d_mll_d_hp_gamma_ptr,
            double* GPFLASH_RESTRICT d_mll_d_hp_kern_extra_ptr) const override {
        add_hp_gamma_and_amplitude_gradient_contribution</*kHasMirror=*/true>(
                hp_grad_inputs,
                idx_a,
                idx_b_begin,
                idx_b_end,
                A_ptr,
                mirror_sign_vec.memptr(),
                d_mll_d_hp_gamma_ptr,
                d_mll_d_hp_kern_extra_ptr);
    }

    // ── The posterior input-space gradient, in raw input space ────────────────
    // The query-point gradient of a weighted sum of kernel values, for one query point, with respect to the raw
    // input rather than its features:
    //
    //     out[l] = d/d(x'_l) sum_i v_i k(x', x_i) = sum_i v_i c^(rdot)_i h^(r,l)_i
    //
    // with h^(r,l) = 2 pi hp_gamma_l hp_amplitude_l^2 sin(2 theta_l). The caller folds c^(rdot) into q_i. Pairing
    // the cosine axis of one point against the sine axis of the other yields sin(2 theta) with no trigonometric
    // call, and hp_amplitude_l^2 arrives with the arguments, both axes of dimension l having been scaled by
    // hp_amplitude_l.
    arma::vec calc_d_kern_sum_d_x_post(
            const arma::mat& Ydot_ntrain_nfeature_mat,  // (n_train, 2 n_dim) so each feature axis is unit-stride
            const arma::vec& ydot_post_vec,
            const arma::vec& hp_gamma_vec,
            const arma::vec& q_vec) const override {
        const int n_train = static_cast<int>(Ydot_ntrain_nfeature_mat.n_rows);
        const int n_dim = static_cast<int>(hp_gamma_vec.n_elem);
        // A postcondition on library-computed feature arrays: a mismatch would read the wrong dimension's sine
        // axis and return a plausible gradient.
        GPFLASH_CHECK(check_feature_width(ydot_post_vec.n_elem, Ydot_ntrain_nfeature_mat.n_cols, n_dim));

        const double* GPFLASH_RESTRICT q_ptr = q_vec.memptr();
        arma::vec d_kern_sum_d_x_post_vec(n_dim);

        // Per input dimension, from its two feature axes.
        for (int i_dim = 0; i_dim < n_dim; ++i_dim) {
            const double* GPFLASH_RESTRICT ydot_cos_train_ptr = Ydot_ntrain_nfeature_mat.colptr(i_dim);
            const double* GPFLASH_RESTRICT ydot_sin_train_ptr = Ydot_ntrain_nfeature_mat.colptr(n_dim + i_dim);
            const double ydot_cos_post = ydot_post_vec(i_dim);
            const double ydot_sin_post = ydot_post_vec(n_dim + i_dim);

            // The cross product of the query and training axes is sin(2 theta).
            double q_sin_2theta_sum = 0.0;
            GPFLASH_OMP_SIMD_REDUCTION(+ : q_sin_2theta_sum)
            for (int i_train = 0; i_train < n_train; ++i_train)
                q_sin_2theta_sum += q_ptr[i_train]
                        * (ydot_sin_post * ydot_cos_train_ptr[i_train] - ydot_cos_post * ydot_sin_train_ptr[i_train]);

            d_kern_sum_d_x_post_vec(i_dim) = (2.0 * arma::datum::pi) * hp_gamma_vec(i_dim) * q_sin_2theta_sum;
        }

        return d_kern_sum_d_x_post_vec;
    }

    // The same gradient for a batch of query points, returned as (n_dim_input, n_post), half the rows of the
    // feature-space arguments it is built from.
    arma::mat calc_d_kern_sum_d_x_post_batch(
            const arma::mat& Ydot_nfeature_npost_mat,
            const arma::mat& q_ydot_sum_nfeature_npost_mat,
            const arma::rowvec& /*q_sum_row*/,
            const arma::vec& hp_gamma_vec,
            const double model_scale) const override {
        const int n_dim = static_cast<int>(hp_gamma_vec.n_elem);
        const int n_post = static_cast<int>(Ydot_nfeature_npost_mat.n_cols);
        const arma::vec dim_scale_vec = (model_scale * 2.0 * arma::datum::pi) * hp_gamma_vec;
        arma::mat d_kern_sum_d_x_post_ndim_npost_mat(n_dim, n_post);

        // The same axis cross product, the caller's q_ydot_sum_nfeature_npost_mat already holding the q-weighted
        // training sums. Query outer, so all three matrices are walked down one column at a time.
        for (int i_post = 0; i_post < n_post; ++i_post) {
            for (int i_dim = 0; i_dim < n_dim; ++i_dim)
                d_kern_sum_d_x_post_ndim_npost_mat(i_dim, i_post) = dim_scale_vec(i_dim)
                        * (Ydot_nfeature_npost_mat(n_dim + i_dim, i_post) * q_ydot_sum_nfeature_npost_mat(i_dim, i_post)
                           - Ydot_nfeature_npost_mat(i_dim, i_post)
                                   * q_ydot_sum_nfeature_npost_mat(n_dim + i_dim, i_post));
        }

        return d_kern_sum_d_x_post_ndim_npost_mat;
    }

   private:
    // Both sums over [idx_b_begin, idx_b_end), the second point mirrored where kHasMirror is set. The mirror is a
    // template parameter so the unmirrored path carries none of its arithmetic.
    template <bool kHasMirror>
    static void add_hp_gamma_and_amplitude_gradient_contribution(
            const HpGradInputs& hp_grad_inputs,
            const int idx_a,
            const int idx_b_begin,
            const int idx_b_end,
            const double* GPFLASH_RESTRICT A_ptr,
            const double* GPFLASH_RESTRICT mirror_sign_ptr,
            double* GPFLASH_RESTRICT d_mll_d_hp_gamma_ptr,
            double* GPFLASH_RESTRICT d_mll_d_hp_kern_extra_ptr) {
        const int n_dim = static_cast<int>(hp_grad_inputs.X_ntrain_ndim_mat.n_cols);
        const int n_pair = idx_b_end - idx_b_begin;

        // One hp_gamma and one hp_amplitude per dimension, from its raw column and its cosine and sine columns.
        for (int i_dim = 0; i_dim < n_dim; ++i_dim) {
            const double* GPFLASH_RESTRICT x_dim_ptr = hp_grad_inputs.X_ntrain_ndim_mat.colptr(i_dim);
            const double* GPFLASH_RESTRICT cos_ptr = hp_grad_inputs.Y_ntrain_nfeature_mat.colptr(i_dim);
            const double* GPFLASH_RESTRICT sin_ptr = hp_grad_inputs.Y_ntrain_nfeature_mat.colptr(n_dim + i_dim);
            const double x_a = x_dim_ptr[idx_a];
            const double cos_a = cos_ptr[idx_a];
            const double sin_a = sin_ptr[idx_a];
            const double mirror_sign = kHasMirror ? mirror_sign_ptr[i_dim] : 1.0;

            // The two per-dimension constants the factors carry.
            const double four_hp_amplitude = 4.0 * hp_grad_inputs.hp_kern_extra_vec(i_dim);
            const double two_pi_hp_amplitude_sq = 2.0 * arma::datum::pi * hp_grad_inputs.hp_kern_extra_vec(i_dim)
                    * hp_grad_inputs.hp_kern_extra_vec(i_dim);

            double sum_row_a_hp_amplitude = 0.0, sum_row_a_hp_gamma = 0.0;
            GPFLASH_OMP_SIMD_REDUCTION(+ : sum_row_a_hp_amplitude, sum_row_a_hp_gamma)
            for (int i_pair = 0; i_pair < n_pair; ++i_pair) {
                const int idx_b = idx_b_begin + i_pair;

                // sin(2 theta) and cos(2 theta) of the phase difference, from the two points' cosine and sine.
                const double sin_b = kHasMirror ? mirror_sign * sin_ptr[idx_b] : sin_ptr[idx_b];
                const double x_b = kHasMirror ? mirror_sign * x_dim_ptr[idx_b] : x_dim_ptr[idx_b];
                const double cos_2theta = cos_ptr[idx_b] * cos_a + sin_b * sin_a;
                const double sin_2theta = sin_b * cos_a - cos_ptr[idx_b] * sin_a;
                const double r_dim = x_b - x_a;
                const double sin_sq_theta = 0.5 * (1.0 - cos_2theta);

                sum_row_a_hp_amplitude += A_ptr[i_pair] * (four_hp_amplitude * sin_sq_theta);
                sum_row_a_hp_gamma += A_ptr[i_pair] * (two_pi_hp_amplitude_sq * r_dim * sin_2theta);
            }
            d_mll_d_hp_gamma_ptr[i_dim] += sum_row_a_hp_gamma;
            d_mll_d_hp_kern_extra_ptr[i_dim] += sum_row_a_hp_amplitude;
        }
    }

    // Restart sampling spans from one period across the data up to this many periods across it. A period longer
    // than the data range gives a near-constant kernel that cannot be identified; far shorter and it aliases.
    static constexpr double SAMPLE_N_PERIOD_ACROSS_DATA_MAX = 20.0;
    // Amplitudes are sampled within this factor either side of their start, not across the whole constraint region.
    static constexpr double SAMPLE_HP_AMPLITUDE_FACTOR = 3.0;

    // Per-dimension data range; a constant dimension falls back to 1.
    static arma::vec calc_data_range_vec(const arma::mat& X_ndim_ntrain_mat) {
        const int n_dim = static_cast<int>(X_ndim_ntrain_mat.n_rows);
        arma::vec data_range_vec(n_dim);
        for (int i_dim = 0; i_dim < n_dim; ++i_dim) {
            const double data_range = arma::range(X_ndim_ntrain_mat.row(i_dim));
            data_range_vec(i_dim) = (data_range > 0.0) ? data_range : 1.0;
        }

        return data_range_vec;
    }

    // n_period_across_data periods across each dimension's range.
    static arma::vec calc_inv_period_vec(const arma::vec& data_range_vec, const double n_period_across_data) {
        return n_period_across_data / data_range_vec;
    }

    static void check_feature_width(
            const arma::uword n_feature_post, const arma::uword n_feature_train, const int n_dim) {
        const arma::uword n_feature_expected = static_cast<arma::uword>(2 * n_dim);
        if (n_feature_post != n_feature_expected || n_feature_train != n_feature_expected)
            throw gpflash::ErrorInvalidInput(
                    "gpflash::kernel::Periodic::calc_d_kern_sum_d_x_post: the feature-space arguments must hold "
                    "twice as many features as hp_gamma_vec has elements, one cosine and one sine axis per input "
                    "dimension.");
    }
};

}  // namespace kernel
}  // namespace gpflash
