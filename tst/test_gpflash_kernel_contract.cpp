// The kernel contract: what every registered kernel must answer consistently, and the KernelId <-> name mapping
// the cache reads.
//
// Most of the contract needs no test: a kernel missing a pure virtual does not compile, and a family's
// invariants are `final`. What is left is what a kernel can get wrong on its own -- agreement between the
// answers it gives -- so this sweeps every registered kernel at run time.
#include <cstdio>
#include <string>

#include "GpFlashTestSupport.hpp"
#include "gpflash/assembly/KernelMatrix.hpp"   // build_kmat_lower_tri, for the diagonal invariant
#include "gpflash/core/Errors.hpp"             // ErrorInvalidInput, thrown by the registry
#include "gpflash/inputs/HpVal.hpp"            // calc_hp_kern_extra_ptr
#include "gpflash/kernels/KernelDispatch.hpp"  // kernel_by_id, to_kernel_name, parse_kernel_id

using gpflash::kernel::KernelBase;
using gpflash::kernel::KernelId;
using gpflash_test::kTolExact;
using gpflash_test::make_points;
using gpflash_test::report;

namespace {

constexpr int kNdim = 4;
constexpr int kNtrain = 24;

// The extras block a kernel declares, filled with a usable value so the sweep can evaluate it.
arma::vec make_hp_kern_extra_vec(const KernelBase& kern_fun) {
    const int n_hp_kern_extra = kern_fun.n_hp_kern_extra(kNdim);
    arma::vec hp_kern_extra_vec(n_hp_kern_extra);
    hp_kern_extra_vec.fill(gpflash_test::kRatQuadHpRho);  // every extra in the package is positive
    return hp_kern_extra_vec;
}

// ── Each kernel's answers must agree with one another ────────────────────────
// A kernel supplies two value loops and its diagonal (KernelBase.hpp), so every way of reaching a kernel value
// must agree: otherwise one overridden loop, or a diagonal that disagrees with its own matrix, slips by.
bool test_kernel_self_consistency(const KernelBase& kern_fun) {
    const std::string label = kern_fun.kernel_name();
    const arma::vec hp_kern_extra_vec = make_hp_kern_extra_vec(kern_fun);
    const double* hp_kern_extra_ptr = gpflash::detail::calc_hp_kern_extra_ptr(hp_kern_extra_vec);
    bool pass = true;
    char detail[gpflash_test::kDetailLen];

    // The value-only loop against the value-and-gradient one, over the range assembly produces.
    arma::vec kern_input_vec(kNtrain);
    for (int i_elem = 0; i_elem < kNtrain; ++i_elem)
        kern_input_vec(i_elem) = 4.0 * static_cast<double>(i_elem) / static_cast<double>(kNtrain - 1);

    arma::vec val_only_vec = kern_input_vec, val_and_rdot_vec = kern_input_vec;
    arma::vec kern_common_rdot_vec(kNtrain);
    kern_fun.calc_kern_val_inplace(val_only_vec.memptr(), kNtrain, hp_kern_extra_ptr);
    kern_fun.calc_kern_val_and_common_rdot_inplace(
            val_and_rdot_vec.memptr(), kNtrain, hp_kern_extra_ptr, kern_common_rdot_vec.memptr());

    const double loop_err = arma::abs(val_only_vec - val_and_rdot_vec).max();
    std::snprintf(detail, sizeof(detail), "err %.3e", loop_err);
    pass &= report((label + ": value loop == value-and-rdot loop").c_str(), loop_err <= kTolExact, detail);

    // The diagonal, three ways: the scalar, the vector, and the diagonal of an assembled K.
    const arma::mat Ydot_nfeature_ntrain_mat = make_points(kNdim, kNtrain);
    const arma::vec ydot_sq_norm_vec = arma::sum(arma::square(Ydot_nfeature_ntrain_mat), 0).t();
    const arma::vec kern_diag_val_vec = kern_fun.calc_kern_diag_val_vec(ydot_sq_norm_vec, hp_kern_extra_ptr);
    const arma::mat K_mat_lower = gpflash::detail::build_kmat_lower_tri(
            kern_fun, Ydot_nfeature_ntrain_mat, hp_kern_extra_ptr, /*store_hp_grad_factors=*/false);

    double scalar_err = 0.0;
    for (int i_train = 0; i_train < kNtrain; ++i_train)
        scalar_err = std::max(
                scalar_err,
                std::fabs(
                        kern_fun.calc_kern_diag_val(ydot_sq_norm_vec(i_train), hp_kern_extra_ptr)
                        - kern_diag_val_vec(i_train)));
    const double assembled_err = arma::abs(arma::vec(K_mat_lower.diag()) - kern_diag_val_vec).max();

    std::snprintf(detail, sizeof(detail), "scalar %.3e, assembled %.3e", scalar_err, assembled_err);
    pass &=
            report((label + ": diagonal agrees scalar/vector/assembled").c_str(),
                   std::max(scalar_err, assembled_err) <= kTolExact,
                   detail);

    // The declaration and the measured diagonal must agree in both directions, so the kernel cannot disable the
    // check by answering false: assembly's Gershgorin bound and its early return both rest on this.
    const double unity_err = arma::abs(kern_diag_val_vec - 1.0).max();
    std::snprintf(detail, sizeof(detail), "declared %d, err %.3e", kern_fun.is_kern_diag_unity() ? 1 : 0, unity_err);
    pass &=
            report((label + ": declared unit diagonal agrees with the measured one").c_str(),
                   kern_fun.is_kern_diag_unity() == (unity_err <= kTolExact),
                   detail);

    // A kernel must read no more extras than it declares. One NaN past the declared width turns the output NaN if
    // it does, which a bounds check on the pointer cannot catch and an exception-based probe would miss.
    arma::vec guarded_hp_kern_extra_vec(hp_kern_extra_vec.n_elem + 1);
    if (!hp_kern_extra_vec.is_empty()) guarded_hp_kern_extra_vec.head(hp_kern_extra_vec.n_elem) = hp_kern_extra_vec;
    guarded_hp_kern_extra_vec(hp_kern_extra_vec.n_elem) = arma::datum::nan;
    arma::vec guarded_val_vec = kern_input_vec;
    kern_fun.calc_kern_val_inplace(guarded_val_vec.memptr(), kNtrain, guarded_hp_kern_extra_vec.memptr());
    // The gradient path's loop is separate code, so it is probed on the same guard.
    arma::vec guarded_val_and_rdot_vec = kern_input_vec;
    arma::vec guarded_kern_common_rdot_vec(kNtrain);
    kern_fun.calc_kern_val_and_common_rdot_inplace(
            guarded_val_and_rdot_vec.memptr(),
            kNtrain,
            guarded_hp_kern_extra_vec.memptr(),
            guarded_kern_common_rdot_vec.memptr());
    const bool is_val_only_finite = guarded_val_vec.is_finite();
    const bool is_val_and_rdot_finite =
            guarded_val_and_rdot_vec.is_finite() && guarded_kern_common_rdot_vec.is_finite();
    std::snprintf(
            detail,
            sizeof(detail),
            "NaN guard: value loop finite %d, value-and-rdot loop finite %d",
            static_cast<int>(is_val_only_finite),
            static_cast<int>(is_val_and_rdot_finite));
    pass &=
            report((label + ": reads no extras past its declared width").c_str(),
                   is_val_only_finite && is_val_and_rdot_finite,
                   detail);

    // The count is either constant or grows with the dimension; nothing in the package supports it shrinking.
    pass &=
            report((label + ": extras width does not shrink with n_dim").c_str(),
                   kern_fun.n_hp_kern_extra(2) >= kern_fun.n_hp_kern_extra(1));

    return pass;
}

// ── The registry and the name mapping ───────────────────────────────────────
bool test_kernel_id_mapping() {
    char detail[gpflash_test::kDetailLen];

    // Pinned integers: the cache stores the id, so inserting rather than appending would rename every existing
    // file's kernel, and the file would still load.
    bool pass = static_cast<int>(KernelId::SqExp) == 0 && static_cast<int>(KernelId::Matern52) == 1
            && static_cast<int>(KernelId::Matern32) == 2 && static_cast<int>(KernelId::RatQuad) == 3
            && static_cast<int>(KernelId::Periodic) == 4 && static_cast<int>(KernelId::Linear) == 5
            && static_cast<int>(KernelId::Quadratic) == 6;
    pass = report("kernel id integers are pinned", pass) && pass;

    // Pinned names, as literals: parse_kernel_id and the default cache path read them, so a rename would leave
    // existing cache files unfound. The round trip below derives every side from kernel_name() and cannot see one.
    struct PinnedKernelName {
        KernelId kernel_id;
        const char* kernel_name;
    };
    const PinnedKernelName pinned_kernel_name_arr[] = {
            {KernelId::SqExp, "sq_exp"},
            {KernelId::Matern52, "matern_5_2"},
            {KernelId::Matern32, "matern_3_2"},
            {KernelId::RatQuad, "rat_quad"},
            {KernelId::Periodic, "periodic"},
            {KernelId::Linear, "linear"},
            {KernelId::Quadratic, "quadratic"}};
    static_assert(
            sizeof(pinned_kernel_name_arr) / sizeof(pinned_kernel_name_arr[0]) == gpflash::kernel::N_KERNEL_ID,
            "every kernel needs its name pinned here");
    int first_unpinned_id = -1;
    for (const PinnedKernelName& pinned_kernel_name : pinned_kernel_name_arr) {
        KernelId parsed_kernel_id = KernelId::End;
        const bool is_name_parsed = gpflash_test::accepts(
                [&] { parsed_kernel_id = gpflash::kernel::parse_kernel_id(pinned_kernel_name.kernel_name); });
        const bool is_name_pinned = std::string(gpflash::kernel::to_kernel_name(pinned_kernel_name.kernel_id))
                        == pinned_kernel_name.kernel_name
                && is_name_parsed && parsed_kernel_id == pinned_kernel_name.kernel_id;
        if (!is_name_pinned && first_unpinned_id < 0)
            first_unpinned_id = static_cast<int>(pinned_kernel_name.kernel_id);
    }
    std::snprintf(detail, sizeof(detail), "first unpinned id %d", first_unpinned_id);
    pass = report("kernel names are pinned", first_unpinned_id < 0, detail) && pass;

    // Every id in [0, N_KERNEL_ID) resolves, names itself, agrees about its own id, and round-trips through the
    // name. N_KERNEL_ID is derived from KernelId::End, so this walks exactly the kernels the enum declares.
    int first_inconsistent_id = -1;
    for (int i_kernel = 0; i_kernel < gpflash::kernel::N_KERNEL_ID; ++i_kernel) {
        const KernelId kernel_id = static_cast<KernelId>(i_kernel);
        const KernelBase& kern_fun = gpflash::kernel::kernel_by_id(kernel_id);
        const std::string kernel_name = kern_fun.kernel_name();
        const bool is_consistent = kern_fun.kernel_id() == kernel_id
                && kernel_name == gpflash::kernel::to_kernel_name(kernel_id)
                && gpflash::kernel::parse_kernel_id(kernel_name) == kernel_id;
        if (!is_consistent && first_inconsistent_id < 0) first_inconsistent_id = i_kernel;
    }
    std::snprintf(detail, sizeof(detail), "first inconsistent id %d", first_inconsistent_id);
    pass = report("every id resolves, names itself and round-trips", first_inconsistent_id < 0, detail) && pass;

    // Exactly one instance per kernel: repeated lookups share an object, different ids do not. What
    // lets the registry hand out a reference, and safe to share across threads and MPI ranks only because a
    // kernel holds no state.
    int first_shared_instance_id = -1;
    for (int i_kernel = 0; i_kernel < gpflash::kernel::N_KERNEL_ID; ++i_kernel) {
        const KernelBase& kern_fun = gpflash::kernel::kernel_by_id(static_cast<KernelId>(i_kernel));
        bool is_one_instance = &kern_fun == &gpflash::kernel::kernel_by_id(static_cast<KernelId>(i_kernel));
        for (int i_other = 0; i_other < i_kernel; ++i_other)
            is_one_instance =
                    is_one_instance && &kern_fun != &gpflash::kernel::kernel_by_id(static_cast<KernelId>(i_other));
        if (!is_one_instance && first_shared_instance_id < 0) first_shared_instance_id = i_kernel;
    }
    std::snprintf(detail, sizeof(detail), "first failing id %d", first_shared_instance_id);
    pass = report("one instance per kernel, distinct across kernels", first_shared_instance_id < 0, detail) && pass;

    // End is a bound, not a kernel, so resolving it is an error rather than one more kernel.
    pass = report("End is not a kernel", gpflash_test::rejects([] { gpflash::kernel::kernel_by_id(KernelId::End); }))
            && pass;

    // Exactly one name per kernel: any other spelling is unknown, including plausible aliases.
    const char* first_accepted_alias = nullptr;
    for (const char* unknown_name : {"rbf", "mat52", "rq", "gaussian_process", "not_a_kernel", ""}) {
        const bool is_rejected = gpflash_test::rejects([&] { gpflash::kernel::parse_kernel_id(unknown_name); });
        if (!is_rejected && !first_accepted_alias) first_accepted_alias = unknown_name;
    }
    std::snprintf(
            detail,
            sizeof(detail),
            "first accepted alias \"%s\"",
            first_accepted_alias ? first_accepted_alias : "(none)");
    pass = report("only the one name per kernel parses", !first_accepted_alias, detail) && pass;

    return pass;
}

}  // namespace

int main() {
    std::printf("== GpFlash kernel contract ==\n");
    bool ok = true;

    // The sweep walks the registry, so a kernel added to GPFLASH_KERNEL_LIST is covered without editing this.
    for (int i_kernel = 0; i_kernel < gpflash::kernel::N_KERNEL_ID; ++i_kernel)
        ok &= test_kernel_self_consistency(gpflash::kernel::kernel_by_id(static_cast<KernelId>(i_kernel)));

    std::printf("\n");
    ok &= test_kernel_id_mapping();

    std::printf("\n%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return ok ? 0 : 1;
}
