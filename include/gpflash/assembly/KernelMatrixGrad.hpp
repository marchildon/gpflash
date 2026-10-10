/**
 * KernelMatrixGrad.hpp -- the gradient counterpart of KernelMatrix.hpp:
 *
 *   <Psi, d(K)/d(gamma_l)> = 2 sum_{i>j} Psi_ij d(K_ij)/d(gamma_l) + sum_i Psi_ii d(K_ii)/d(gamma_l)
 *
 * One function per term, because a strictly-lower pair loop cannot reach the diagonal. The pair term reads
 * c^(rdot) out of the strict upper triangle build_kern_assembly left there, so it evaluates no kernel of its own
 * and forms no second n_train by n_train matrix.
 */
#pragma once

#include <armadillo>
#include <cstddef>

#include "gpflash/assembly/KernTermHpGrad.hpp"  // KernTermHpGrad, check_kern_term_hp_grad
#include "gpflash/assembly/SigmaGrad.hpp"       // calc_psi_val
#include "gpflash/assembly/SymmetryMatrix.hpp"  // calc_packed_lower_offset, gather_active_row_packed_vec
#include "gpflash/core/Platform.hpp"            // GPFLASH_RESTRICT, OMP constants
#include "gpflash/inputs/SymmetrySpec.hpp"      // SymmetrySpec
#include "gpflash/kernels/HpGradInputs.hpp"     // kernel::HpGradInputs
#include "gpflash/kernels/KernelBase.hpp"       // kernel::KernelBase

namespace gpflash {
namespace detail {

// The diagonal term, sum_i Psi_ii d(K_ii)/d(gamma_l).
//
// An empty d_kern_diag_d_hp_gamma_ntrain_ndim_mat is a constant diagonal, whose derivative is zero; the caller gets a
// zero vector of the length it asked for, so it can add the result unconditionally.
inline arma::vec calc_kern_diag_hp_grad(
        const arma::mat& d_kern_diag_d_hp_gamma_ntrain_ndim_mat, const arma::vec& psi_diag_vec, const int n_dim) {
    if (d_kern_diag_d_hp_gamma_ntrain_ndim_mat.is_empty())
        return arma::vec(n_dim, arma::fill::zeros);
    else
        return d_kern_diag_d_hp_gamma_ntrain_ndim_mat.t() * psi_diag_vec;
}

// The pair term, 2 sum_{i>j} A_ij h^(alpha,l)_ij with A = Psi .* C^(rdot), one column of A at a time in a
// thread-local buffer. The factor of two is applied here; the outer sigma_k^2 is the caller's.
//
// The active row's c^(rdot) is scattered one entry per column, so the sweep gathers it as it passes rather than
// in a second pass. idx_K_corr_max_abs_row_sum of -1 asks for none, and it comes back zero.
//
// Sigma_inv_mat_lower is read only: Psi's Sigma^-1 (SigmaGrad.hpp) lower with diag, c^(rdot) strict upper.
inline KernTermHpGrad calc_kern_term_hp_grad(
        const kernel::KernelBase& kern_fun,
        const kernel::HpGradInputs& hp_grad_inputs,
        const arma::vec& lin_sol_sigma_residual_vec,
        const arma::mat& Sigma_inv_mat_lower,
        const int idx_K_corr_max_abs_row_sum,
        const SymmetrySpec& symmetry_spec,
        const arma::vec& kern_common_rdot_mirror_packed_vec) {
    const int n_dim = static_cast<int>(hp_grad_inputs.hp_gamma_vec.n_elem);
    const int n_hp_kern_extra = kern_fun.n_hp_kern_extra(n_dim);
    const int n_train = static_cast<int>(lin_sol_sigma_residual_vec.n_elem);
    const bool has_active_row = (idx_K_corr_max_abs_row_sum != -1);
    const bool has_symmetry = symmetry_spec.is_on();
    const double parity_sign = symmetry_spec.parity_sign();

    KernTermHpGrad kern_term_hp_grad;
    arma::vec& d_mll_d_hp_gamma_vec = kern_term_hp_grad.d_mll_d_hp_gamma_vec;
    arma::vec& d_mll_d_hp_kern_extra_vec = kern_term_hp_grad.d_mll_d_hp_kern_extra_vec;
    d_mll_d_hp_gamma_vec.zeros(n_dim);
    d_mll_d_hp_kern_extra_vec.zeros(n_hp_kern_extra);

    const double* kern_common_rdot_mirror_packed_ptr =
            has_symmetry ? kern_common_rdot_mirror_packed_vec.memptr() : nullptr;

    kern_term_hp_grad.active_row_kern_common_rdot_vec.zeros(n_train);
    double* GPFLASH_RESTRICT active_row_kern_common_rdot_ptr =
            kern_term_hp_grad.active_row_kern_common_rdot_vec.memptr();
    const double* GPFLASH_RESTRICT lin_sol_ptr = lin_sol_sigma_residual_vec.memptr();
    // Sigma^-1 and c^(rdot) share this buffer, so they share one unqualified base pointer: two
    // restrict-qualified aliases of the same memory are undefined behavior as soon as either is written. The
    // per-column pointers below address disjoint elements and so keep the qualifier the inner loops need.
    const double* sigma_inv_and_rdot_ptr = Sigma_inv_mat_lower.memptr();

    // Each thread sweeps its share of the columns, then adds its sums in. Outside a parallel region the omp for and
    // critical inside run on the calling thread alone, which spares a small problem the cost of opening one.
    const auto sweep_columns = [&]() {
        arma::vec d_mll_d_hp_gamma_local_vec(n_dim, arma::fill::zeros);
        arma::vec d_mll_d_hp_kern_extra_local_vec(n_hp_kern_extra, arma::fill::zeros);
        arma::vec A_col_vec(n_train, arma::fill::none);
        double* GPFLASH_RESTRICT A_col_ptr = A_col_vec.memptr();
        // The mirrored term's adjoint column. Sized only where symmetry is on.
        arma::vec A_mirror_col_vec(has_symmetry ? n_train : 0, arma::fill::none);
        double* GPFLASH_RESTRICT A_mirror_col_ptr = A_mirror_col_vec.memptr();
        // Null asks the kernel for no extras term. Loop invariant, so it is resolved once per thread.
        double* d_mll_d_hp_kern_extra_local_ptr =
                (n_hp_kern_extra > 0) ? d_mll_d_hp_kern_extra_local_vec.memptr() : nullptr;

#pragma omp for schedule(dynamic, PERF_OMP_COL_CHUNK) nowait
        for (int i_col = 0; i_col < n_train; ++i_col) {
            if (i_col + 1 >= n_train) continue;  // the last column has nothing below the diagonal
            const double lin_sol_col = lin_sol_ptr[i_col];
            const double* GPFLASH_RESTRICT sigma_inv_col_ptr =
                    sigma_inv_and_rdot_ptr + static_cast<std::ptrdiff_t>(i_col) * n_train;
            // Column i_col's c^(rdot) is contiguous in upper column n_train - 1 - i_col. It and A_col_ptr are both
            // based at the first row below the diagonal, so both take i_below where full-length vectors take i_row.
            const double* GPFLASH_RESTRICT kern_common_rdot_col_ptr =
                    sigma_inv_and_rdot_ptr + static_cast<std::ptrdiff_t>(n_train - 1 - i_col) * n_train;
            // Columns left of the active row hold one of its entries each, its own column holds the rest, so no
            // entry is written twice.
            if (has_active_row) {
                if (i_col < idx_K_corr_max_abs_row_sum)
                    active_row_kern_common_rdot_ptr[i_col] =
                            kern_common_rdot_col_ptr[idx_K_corr_max_abs_row_sum - i_col - 1];
                else if (i_col == idx_K_corr_max_abs_row_sum)
                    for (int i_row = i_col + 1; i_row < n_train; ++i_row)
                        active_row_kern_common_rdot_ptr[i_row] = kern_common_rdot_col_ptr[i_row - i_col - 1];
            }
            // A_ij = Psi_ij c^(rdot)_ij, as the GpFlash paper writes it. Psi is the two terms' shared weight, so
            // a symmetry model forms both adjoints from one pass over it, and parity_sign is folded into the mirrored
            // one so both terms accumulate into the same running sum.
            //
            // Two loops rather than one with a test inside: the condition is invariant over the sweep, and this
            // is the gradient's hottest loop.
            if (has_symmetry) {
                const double* GPFLASH_RESTRICT kern_common_rdot_mirror_col_ptr =
                        kern_common_rdot_mirror_packed_ptr + calc_packed_lower_offset(i_col, n_train) + 1;
                for (int i_row = i_col + 1; i_row < n_train; ++i_row) {
                    const int i_below = i_row - i_col - 1;
                    const double psi_val = calc_psi_val(lin_sol_col, lin_sol_ptr[i_row], sigma_inv_col_ptr[i_row]);
                    A_col_ptr[i_below] = psi_val * kern_common_rdot_col_ptr[i_below];
                    A_mirror_col_ptr[i_below] = parity_sign * psi_val * kern_common_rdot_mirror_col_ptr[i_below];
                }
            } else {
                for (int i_row = i_col + 1; i_row < n_train; ++i_row) {
                    const int i_below = i_row - i_col - 1;
                    const double psi_val = calc_psi_val(lin_sol_col, lin_sol_ptr[i_row], sigma_inv_col_ptr[i_row]);
                    A_col_ptr[i_below] = psi_val * kern_common_rdot_col_ptr[i_below];
                }
            }
            // Point i_col against every point below it in the column, in the kernel's own int indices.
            kern_fun.add_hp_gradient_contribution(
                    hp_grad_inputs,
                    i_col,
                    i_col + 1,
                    n_train,
                    A_col_ptr,
                    d_mll_d_hp_gamma_local_vec.memptr(),
                    d_mll_d_hp_kern_extra_local_ptr);
            if (has_symmetry)
                kern_fun.add_hp_gradient_contribution_mirrored(
                        hp_grad_inputs,
                        i_col,
                        i_col + 1,
                        n_train,
                        A_mirror_col_ptr,
                        symmetry_spec.mirror_sign_vec,
                        d_mll_d_hp_gamma_local_vec.memptr(),
                        d_mll_d_hp_kern_extra_local_ptr);
        }
#pragma omp critical
        {
            d_mll_d_hp_gamma_vec += d_mll_d_hp_gamma_local_vec;
            d_mll_d_hp_kern_extra_vec += d_mll_d_hp_kern_extra_local_vec;
        }
    };

    if (n_train >= PERF_OMP_MIN_N_POINT) {
#pragma omp parallel
        sweep_columns();
    } else {
        sweep_columns();
    }

    // A and H are both symmetric, so the strict lower triangle the sweep covered stands for both halves.
    d_mll_d_hp_gamma_vec *= 2.0;
    d_mll_d_hp_kern_extra_vec *= 2.0;

    // The mirrored term's active row, which the eta gradient needs, from the packed triangle that holds it.
    if (has_symmetry && has_active_row)
        kern_term_hp_grad.active_row_kern_common_rdot_mirror_vec =
                gather_active_row_packed_vec(kern_common_rdot_mirror_packed_vec, idx_K_corr_max_abs_row_sum, n_train);

    GPFLASH_CHECK(check_kern_term_hp_grad(kern_term_hp_grad, "calc_kern_term_hp_grad"));
    return kern_term_hp_grad;
}

}  // namespace detail
}  // namespace gpflash
