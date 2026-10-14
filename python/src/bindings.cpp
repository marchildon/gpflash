/**
 * bindings.cpp -- the Python module gpflash._core: a thin binding of the C++ interface.
 *
 * Names and array shapes follow the C++ structs, so the C++ README describes this module too: X_ndim_ntrain_mat is
 * (n_dim, n_train), one point per column. The Python wrapper in gpflash/_gp.py takes the NumPy (n_points, n_dim)
 * layout instead.
 *
 * Array fields are converted on every get and set, so they are values, not views: assign a whole array to change
 * one. A query's points are read in place when they are already float64 and column-major, which the transpose of a
 * C-ordered (n_points, n_dim) array is. Every computation releases the GIL so other Python threads run meanwhile.
 *
 * Serial only: the module is built without GPFLASH_HAVE_MPI, so a fit runs on the calling process. MPI stays with
 * the C++ library.
 */
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>  // std::copy
#include <cstdint>    // std::int64_t
#include <string>

#include "gpflash/Version.hpp"
#include "gpflash/core/Errors.hpp"
#include "gpflash/kernels/KernelDispatch.hpp"
#include "gpflash/optimization/HpOptimizer.hpp"
#include "gpflash/optimization/MultistartReport.hpp"
#include "gpflash/posterior_build/PosteriorBuild.hpp"
#include "gpflash/posterior_cache/CacheInputHash.hpp"
#include "gpflash/posterior_cache/PosteriorCache.hpp"
#include "gpflash/posterior_eval/GpPosterior.hpp"
#include "gpflash/training/EvalMllDispatch.hpp"

namespace py = pybind11;

namespace {

using F64ArrayC = py::array_t<double, py::array::c_style | py::array::forcecast>;
using F64ArrayF = py::array_t<double, py::array::f_style | py::array::forcecast>;

// ── NumPy to Armadillo ───────────────────────────────────────────────────────

// obj as an array of Array's dtype and layout (a copy when it is not one already), with n_dim dimensions.
template <class Array>
Array ensure_array(const py::handle& obj, const int n_dim, const char* field_name) {
    const Array arr = Array::ensure(obj);
    if (!arr) throw py::type_error(std::string(field_name) + " must be convertible to a numeric array.");
    if (arr.ndim() != n_dim)
        throw py::value_error(
                std::string(field_name) + " must be " + std::to_string(n_dim) + "-D; got " + std::to_string(arr.ndim())
                + " dimensions.");
    return arr;
}

// from_numpy<T>: obj copied into an Armadillo T, one specialization per field type the structs hold.
template <class T>
T from_numpy(const py::handle& obj, const char* field_name);

template <>
arma::vec from_numpy(const py::handle& obj, const char* field_name) {
    const F64ArrayC arr = ensure_array<F64ArrayC>(obj, 1, field_name);
    return arma::vec(arr.data(), static_cast<arma::uword>(arr.size()));
}

template <>
arma::mat from_numpy(const py::handle& obj, const char* field_name) {
    const F64ArrayF arr = ensure_array<F64ArrayF>(obj, 2, field_name);
    return arma::mat(arr.data(), static_cast<arma::uword>(arr.shape(0)), static_cast<arma::uword>(arr.shape(1)));
}

template <>
arma::uvec from_numpy(const py::handle& obj, const char* field_name) {
    using I64ArrayC = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;
    const I64ArrayC arr = ensure_array<I64ArrayC>(obj, 1, field_name);
    arma::uvec uvec(static_cast<arma::uword>(arr.size()));
    for (py::ssize_t i_elem = 0; i_elem < arr.size(); ++i_elem) {
        if (arr.data()[i_elem] < 0) throw py::value_error(std::string(field_name) + " entries must be >= 0.");
        uvec(static_cast<arma::uword>(i_elem)) = static_cast<arma::uword>(arr.data()[i_elem]);
    }
    return uvec;
}

// ── Armadillo to NumPy, always a copy the caller owns ────────────────────────

py::array_t<double> to_numpy(const arma::vec& vec) {
    py::array_t<double> arr(static_cast<py::ssize_t>(vec.n_elem));
    std::copy(vec.begin(), vec.end(), arr.mutable_data());
    return arr;
}

// Column-major, matching Armadillo, so the copy is one contiguous pass.
py::array_t<double> to_numpy(const arma::mat& mat) {
    const auto n_row = static_cast<py::ssize_t>(mat.n_rows);
    const auto elem_size = static_cast<py::ssize_t>(sizeof(double));
    py::array_t<double> arr({n_row, static_cast<py::ssize_t>(mat.n_cols)}, {elem_size, elem_size * n_row});
    std::copy(mat.begin(), mat.end(), arr.mutable_data());
    return arr;
}

py::array_t<std::int64_t> to_numpy(const arma::uvec& uvec) {
    py::array_t<std::int64_t> arr(static_cast<py::ssize_t>(uvec.n_elem));
    std::copy(uvec.begin(), uvec.end(), arr.mutable_data());
    return arr;
}

// ── Array-valued struct fields as properties ─────────────────────────────────

template <class Struct, class Field>
void def_array_rw(py::class_<Struct>& cls, const char* name, Field Struct::*member, const char* doc) {
    cls.def_property(
            name,
            [member](const Struct& obj) { return to_numpy(obj.*member); },
            [member, name](Struct& obj, const py::handle& val) { obj.*member = from_numpy<Field>(val, name); },
            doc);
}

template <class Struct, class Field>
void def_array_ro(py::class_<Struct>& cls, const char* name, Field Struct::*member, const char* doc) {
    cls.def_property_readonly(
            name, [member](const Struct& obj) { return to_numpy(obj.*member); }, doc);
}

// ── Posterior queries ────────────────────────────────────────────────────────

// A 2-D array is (n_dim, n_post), one point per column; a 1-D one is a single point. Read in place when already
// float64 and column-major, so a C-ordered (n_post, n_dim) array's transpose costs no copy.
gpflash::PosteriorResult eval_posterior(
        const gpflash::GpPosterior& posterior, const py::handle& x_obj, const gpflash::PosteriorOptions& post_opts) {
    const F64ArrayF arr = F64ArrayF::ensure(x_obj);
    if (!arr) throw py::type_error("The query points must be convertible to a float64 array.");
    if (arr.ndim() == 1) {
        const arma::vec x_post_vec(const_cast<double*>(arr.data()), static_cast<arma::uword>(arr.size()), false, true);
        py::gil_scoped_release release_gil;
        return posterior.eval(x_post_vec, post_opts);
    }
    if (arr.ndim() == 2) {
        const arma::mat X_ndim_npost_mat(
                const_cast<double*>(arr.data()),
                static_cast<arma::uword>(arr.shape(0)),
                static_cast<arma::uword>(arr.shape(1)),
                false,
                true);
        py::gil_scoped_release release_gil;
        return posterior.eval(X_ndim_npost_mat, post_opts);
    }
    throw py::value_error(
            "The query points must be 1-D (one point) or 2-D (n_dim, n_post); got " + std::to_string(arr.ndim())
            + " dimensions.");
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() =
            "GpFlash C++ bindings. Names and shapes follow the C++ interface; see the gpflash package for the "
            "NumPy-style GaussianProcess wrapper.";

    // ── Version and constants ────────────────────────────────────────────────
    m.attr("__version__") = gpflash::version_string();
    m.attr("COND_MAX_DEFAULT") = gpflash::COND_MAX_DEFAULT;
    m.attr("COND_MAX_LIMIT") = gpflash::COND_MAX_LIMIT;
    m.attr("MEAN_MODEL_DEGREE_MAX") = static_cast<int>(gpflash::MEAN_MODEL_DEGREE_MAX);

    // ── Exceptions ───────────────────────────────────────────────────────────
    // Subclasses of the matching built-ins, so `except ValueError` also catches an invalid input.
    py::register_exception<gpflash::ErrorInvalidInput>(m, "InvalidInputError", PyExc_ValueError);
    py::register_exception<gpflash::ErrorNumericalFailure>(m, "NumericalFailureError", PyExc_ArithmeticError);
    py::register_exception<gpflash::ErrorNotImplemented>(m, "NotImplementedFeatureError", PyExc_NotImplementedError);

    // ── Enums ────────────────────────────────────────────────────────────────
    py::enum_<gpflash::kernel::KernelId>(m, "KernelId", "Which kernel a model uses.")
            .value("SqExp", gpflash::kernel::KernelId::SqExp)
            .value("Matern52", gpflash::kernel::KernelId::Matern52)
            .value("Matern32", gpflash::kernel::KernelId::Matern32)
            .value("RatQuad", gpflash::kernel::KernelId::RatQuad)
            .value("Periodic", gpflash::kernel::KernelId::Periodic)
            .value("Linear", gpflash::kernel::KernelId::Linear)
            .value("Quadratic", gpflash::kernel::KernelId::Quadratic);

    py::enum_<gpflash::EigMaxUbMethod>(m, "EigMaxUbMethod", "The bound on lambda_max a nugget is sized from.")
            .value("Gershgorin", gpflash::EigMaxUbMethod::Gershgorin)
            .value("Trace", gpflash::EigMaxUbMethod::Trace);

    py::enum_<gpflash::NumericalOptzAlgorithm>(m, "NumericalOptzAlgorithm", "The algorithm each local solve uses.")
            .value("Sqp", gpflash::NumericalOptzAlgorithm::Sqp)
            .value("Ccsaq", gpflash::NumericalOptzAlgorithm::Ccsaq);

    // ── Kernels ──────────────────────────────────────────────────────────────
    m.def("kernel_name",
          &gpflash::kernel::to_kernel_name,
          py::arg("kernel_id"),
          "The kernel's name, e.g. 'matern_5_2'; the inverse of parse_kernel_id.");
    m.def("parse_kernel_id",
          &gpflash::kernel::parse_kernel_id,
          py::arg("kernel_name"),
          "The KernelId for a name such as 'sq_exp'; raises InvalidInputError for an unknown one.");
    m.def(
            "n_hp_kern_extra",
            [](const gpflash::kernel::KernelId kernel_id, const int n_dim) {
                return gpflash::kernel::kernel_by_id(kernel_id).n_hp_kern_extra(n_dim);
            },
            py::arg("kernel_id"),
            py::arg("n_dim"),
            "How many entries HpVal.hp_kern_extra_vec holds for this kernel in n_dim dimensions.");

    // ── Inputs ───────────────────────────────────────────────────────────────
    py::class_<gpflash::TrainingData> training_data(m, "TrainingData", "What was measured. Fixed for a fit.");
    training_data.def(py::init<>());
    training_data.def(py::init<const gpflash::TrainingData&>(), "A copy.");
    def_array_rw(
            training_data,
            "X_ndim_ntrain_mat",
            &gpflash::TrainingData::X_ndim_ntrain_mat,
            "(n_dim, n_train) training points, one per column.");
    def_array_rw(
            training_data,
            "f_train_vec",
            &gpflash::TrainingData::f_train_vec,
            "(n_train) function evaluations at those points.");
    training_data.def_readwrite(
            "sig2_f_val",
            &gpflash::TrainingData::sig2_f_val,
            "Noise variance: 0 noise-free, > 0 known and fixed, < 0 estimated by the fit.");

    py::class_<gpflash::HpVal> hp_val(m, "HpVal", "The hyperparameters.");
    hp_val.def(py::init<>());
    hp_val.def(py::init<const gpflash::HpVal&>(), "A copy.");
    hp_val.def_readwrite("hp_sig2_k", &gpflash::HpVal::hp_sig2_k, "Signal variance; negative means unset.");
    hp_val.def_readwrite(
            "hp_sig2_f_val",
            &gpflash::HpVal::hp_sig2_f_val,
            "Estimated noise variance, set only when TrainingData.sig2_f_val < 0; negative means unset.");
    def_array_rw(
            hp_val,
            "hp_gamma_vec",
            &gpflash::HpVal::hp_gamma_vec,
            "(n_dim) inverse length scales, or inverse periods for Periodic.");
    def_array_rw(
            hp_val,
            "hp_kern_extra_vec",
            &gpflash::HpVal::hp_kern_extra_vec,
            "(n_hp_kern_extra) the kernel's extra hyperparameters, or empty.");

    py::class_<gpflash::SymmetrySpec> symmetry_spec(m, "SymmetrySpec", "An axis-aligned mirror symmetry.");
    symmetry_spec.def(py::init<>());
    symmetry_spec.def(py::init<const gpflash::SymmetrySpec&>(), "A copy.");
    def_array_rw(
            symmetry_spec,
            "mirror_sign_vec",
            &gpflash::SymmetrySpec::mirror_sign_vec,
            "(n_dim) diagonal of the mirror, each +1 or -1; empty means no symmetry.");
    symmetry_spec.def_readwrite(
            "is_even_symmetry", &gpflash::SymmetrySpec::is_even_symmetry, "Even (True) or odd (False) model.");
    symmetry_spec.def("is_on", &gpflash::SymmetrySpec::is_on);

    py::class_<gpflash::ModelOptions> model_options(m, "ModelOptions", "Which model is evaluated.");
    model_options.def(py::init<>());
    model_options.def(py::init<const gpflash::ModelOptions&>(), "A copy.");
    model_options.def_readwrite("kernel_id", &gpflash::ModelOptions::kernel_id);
    model_options.def_readwrite(
            "cond_max", &gpflash::ModelOptions::cond_max, "Condition-number bound that sizes the nuggets.");
    model_options.def_readwrite("kmat_eig_max_ub_method", &gpflash::ModelOptions::kmat_eig_max_ub_method);
    model_options.def_readwrite("has_mean_model_constant", &gpflash::ModelOptions::has_mean_model_constant);
    def_array_rw(
            model_options,
            "mean_model_degree_vec",
            &gpflash::ModelOptions::mean_model_degree_vec,
            "Power of each x_d in the mean model basis, 0 to MEAN_MODEL_DEGREE_MAX; one entry applies to all.");
    model_options.def_readwrite("mean_model_eig_max_ub_method", &gpflash::ModelOptions::mean_model_eig_max_ub_method);
    model_options.def_readwrite("symmetry_spec", &gpflash::ModelOptions::symmetry_spec);

    py::class_<gpflash::OptzOptions> optz_options(m, "OptzOptions", "How the hyperparameters are fitted.");
    optz_options.def(py::init<>());
    optz_options.def(py::init<const gpflash::OptzOptions&>(), "A copy.");
    optz_options.def_readwrite("model_options", &gpflash::OptzOptions::model_options);
    optz_options.def_readwrite("use_multistart", &gpflash::OptzOptions::use_multistart);
    optz_options.def_readwrite("n_ind_optz", &gpflash::OptzOptions::n_ind_optz);
    optz_options.def_readwrite("hypercube_sampling_seed", &gpflash::OptzOptions::hypercube_sampling_seed);
    optz_options.def_readwrite("max_parallel_ind_optz", &gpflash::OptzOptions::max_parallel_ind_optz);
    optz_options.def_readwrite("numerical_optz_algorithm", &gpflash::OptzOptions::numerical_optz_algorithm);
    optz_options.def_readwrite("rel_step_tol", &gpflash::OptzOptions::rel_step_tol);
    optz_options.def_readwrite("rel_obj_tol", &gpflash::OptzOptions::rel_obj_tol);
    optz_options.def_readwrite("max_evals", &gpflash::OptzOptions::max_evals);
    optz_options.def_readwrite("use_bounds", &gpflash::OptzOptions::use_bounds);
    def_array_rw(optz_options, "hp_gamma_lb_vec", &gpflash::OptzOptions::hp_gamma_lb_vec, "1 entry or n_dim.");
    def_array_rw(optz_options, "hp_gamma_ub_vec", &gpflash::OptzOptions::hp_gamma_ub_vec, "1 entry or n_dim.");
    def_array_rw(
            optz_options,
            "hp_kern_extra_lb_vec",
            &gpflash::OptzOptions::hp_kern_extra_lb_vec,
            "1 entry or n_hp_kern_extra.");
    def_array_rw(
            optz_options,
            "hp_kern_extra_ub_vec",
            &gpflash::OptzOptions::hp_kern_extra_ub_vec,
            "1 entry or n_hp_kern_extra.");
    optz_options.def_readwrite(
            "scale_variance_bounds_by_residual_var", &gpflash::OptzOptions::scale_variance_bounds_by_residual_var);
    optz_options.def_readwrite("hp_sig2_k_lb", &gpflash::OptzOptions::hp_sig2_k_lb);
    optz_options.def_readwrite("hp_sig2_k_ub", &gpflash::OptzOptions::hp_sig2_k_ub);
    optz_options.def_readwrite("hp_sig2_f_val_lb", &gpflash::OptzOptions::hp_sig2_f_val_lb);
    optz_options.def_readwrite("hp_sig2_f_val_ub", &gpflash::OptzOptions::hp_sig2_f_val_ub);
    optz_options.def_readwrite("hp_sig2_k_init", &gpflash::OptzOptions::hp_sig2_k_init);
    optz_options.def_readwrite("hp_sig2_f_val_init", &gpflash::OptzOptions::hp_sig2_f_val_init);
    optz_options.def_readwrite("hp_kern_extra_init", &gpflash::OptzOptions::hp_kern_extra_init);

    py::class_<gpflash::PosteriorOptions> posterior_options(
            m, "PosteriorOptions", "Which parts of a posterior query to compute.");
    posterior_options.def(py::init<>());
    posterior_options.def(py::init<const gpflash::PosteriorOptions&>(), "A copy.");
    posterior_options.def_readwrite("return_mean", &gpflash::PosteriorOptions::return_mean);
    posterior_options.def_readwrite("return_var", &gpflash::PosteriorOptions::return_var);
    posterior_options.def_readwrite("return_mean_gradient", &gpflash::PosteriorOptions::return_mean_gradient);
    posterior_options.def_readwrite("return_var_gradient", &gpflash::PosteriorOptions::return_var_gradient);
    posterior_options.def_readwrite("check_query_points", &gpflash::PosteriorOptions::check_query_points);

    // ── Results ──────────────────────────────────────────────────────────────
    py::class_<gpflash::MllHpGrad> mll_hp_grad(m, "MllHpGrad", "The MLL's derivative per hyperparameter.");
    def_array_ro(mll_hp_grad, "d_mll_d_hp_gamma_vec", &gpflash::MllHpGrad::d_mll_d_hp_gamma_vec, "(n_dim)");
    def_array_ro(
            mll_hp_grad,
            "d_mll_d_hp_kern_extra_vec",
            &gpflash::MllHpGrad::d_mll_d_hp_kern_extra_vec,
            "(n_hp_kern_extra), or empty.");
    mll_hp_grad.def_readonly("d_mll_d_hp_sig2_k", &gpflash::MllHpGrad::d_mll_d_hp_sig2_k);
    mll_hp_grad.def_readonly("d_mll_d_hp_sig2_f_val", &gpflash::MllHpGrad::d_mll_d_hp_sig2_f_val);

    py::class_<gpflash::MllResult> mll_result(m, "MllResult", "What eval_mll returns.");
    mll_result.def_readonly("mll", &gpflash::MllResult::mll);
    def_array_ro(mll_result, "hp_beta_vec", &gpflash::MllResult::hp_beta_vec, "(n_basis) mean model weights.");
    mll_result.def_readonly("hp_sig2_k_from_closed_form", &gpflash::MllResult::hp_sig2_k_from_closed_form);
    mll_result.def_readonly("hp_sig2_k", &gpflash::MllResult::hp_sig2_k);
    mll_result.def_readonly("mll_hp_grad_struct", &gpflash::MllResult::mll_hp_grad_struct);

    // ── Fit diagnostics ──────────────────────────────────────────────────────
    py::enum_<gpflash::OptzStopReason>(m, "OptzStopReason", "Why a local solve stopped.")
            .value("ToleranceMet", gpflash::OptzStopReason::ToleranceMet)
            .value("EvalBudgetSpent", gpflash::OptzStopReason::EvalBudgetSpent)
            .value("RoundoffLimited", gpflash::OptzStopReason::RoundoffLimited)
            .value("Failed", gpflash::OptzStopReason::Failed);

    py::class_<gpflash::LocalOptzResult> local_optz_result(m, "LocalOptzResult", "The outcome of one local solve.");
    def_array_ro(
            local_optz_result,
            "ln_hp_vec",
            &gpflash::LocalOptzResult::ln_hp_vec,
            "The minimizer, in log-hyperparameter space.");
    local_optz_result.def_readonly(
            "objective_val", &gpflash::LocalOptzResult::objective_val, "At the minimizer; inf when Failed.");
    local_optz_result.def_readonly("n_eval", &gpflash::LocalOptzResult::n_eval, "Objective evaluations.");
    local_optz_result.def_readonly("stop_reason", &gpflash::LocalOptzResult::stop_reason);

    py::class_<gpflash::HpAtBound> hp_at_bound(m, "HpAtBound", "Which hyperparameters a bound is holding.");
    def_array_ro(hp_at_bound, "hp_gamma_at_bound_vec", &gpflash::HpAtBound::hp_gamma_at_bound_vec, "(n_dim), 1 or 0.");
    def_array_ro(
            hp_at_bound,
            "hp_kern_extra_at_bound_vec",
            &gpflash::HpAtBound::hp_kern_extra_at_bound_vec,
            "(n_hp_kern_extra), 1 or 0; empty when the kernel has no extras.");
    hp_at_bound.def_readonly("hp_sig2_k_at_bound", &gpflash::HpAtBound::hp_sig2_k_at_bound);
    hp_at_bound.def_readonly("hp_sig2_f_val_at_bound", &gpflash::HpAtBound::hp_sig2_f_val_at_bound);
    hp_at_bound.def("has_any", &gpflash::HpAtBound::has_any);
    hp_at_bound.def("to_summary_string", [](const gpflash::HpAtBound& at_bound) {
        return gpflash::to_summary_string(at_bound);
    });

    py::class_<gpflash::MllOptzResult> mll_optz_result(m, "MllOptzResult", "What select_hp_with_max_mll returns.");
    mll_optz_result.def_readonly("hp_val_struct", &gpflash::MllOptzResult::hp_val_struct);
    mll_optz_result.def_readonly("model_options", &gpflash::MllOptzResult::model_options);
    mll_optz_result.def_readonly(
            "optimality", &gpflash::MllOptzResult::optimality, "Projected gradient norm at the winner.");
    mll_optz_result.def("mll", &gpflash::MllOptzResult::mll);
    mll_optz_result.def("n_ind_optz_run", &gpflash::MllOptzResult::n_ind_optz_run);
    mll_optz_result.def("idx_best", &gpflash::MllOptzResult::idx_best);
    mll_optz_result.def("calc_n_failed_start", &gpflash::MllOptzResult::calc_n_failed_start);
    mll_optz_result.def("n_total_mll_eval", &gpflash::MllOptzResult::n_total_mll_eval);
    mll_optz_result.def("n_total_mll_grad_eval", &gpflash::MllOptzResult::n_total_mll_grad_eval);
    mll_optz_result.def(
            "calc_n_at_bound",
            &gpflash::MllOptzResult::calc_n_at_bound,
            py::arg("i_ind_optz"),
            "How many hyperparameters optimization i_ind_optz left on a bound; idx_best() is the winner.");
    mll_optz_result.def(
            "calc_at_bound_flags_vec",
            [](const gpflash::MllOptzResult& result, const int i_ind_optz) {
                return to_numpy(result.calc_at_bound_flags_vec(i_ind_optz));
            },
            py::arg("i_ind_optz"),
            "Which entries optimization i_ind_optz left on a bound, 1 or 0 each, in the search's order.");
    mll_optz_result.def(
            "calc_hp_at_bound_struct",
            py::overload_cast<>(&gpflash::MllOptzResult::calc_hp_at_bound_struct, py::const_),
            "Which hyperparameters the winning optimization left on a bound.");
    mll_optz_result.def(
            "calc_hp_at_bound_struct",
            py::overload_cast<int>(&gpflash::MllOptzResult::calc_hp_at_bound_struct, py::const_),
            py::arg("i_ind_optz"),
            "Which hyperparameters optimization i_ind_optz left on a bound.");
    mll_optz_result.def(
            "build_best_local_optz_result",
            &gpflash::MllOptzResult::build_best_local_optz_result,
            "The winning solve; its stop_reason says why it stopped.");
    mll_optz_result.def(
            "to_summary_string",
            [](const gpflash::MllOptzResult& result, const std::string& label) {
                return gpflash::to_summary_string(result, label);
            },
            py::arg("label") = "");

    m.def("to_multistart_report_string",
          &gpflash::to_multistart_report_string,
          py::arg("mll_optz_result"),
          py::arg("label") = "",
          "The long fit report: the fit, a row per optimization, where each landed and started, and the terms.");
    // Through Python's print rather than C's stdout, so the report also appears in a notebook.
    m.def(
            "print_multistart_report",
            [](const gpflash::MllOptzResult& result, const std::string& label) {
                py::print(gpflash::to_multistart_report_string(result, label), py::arg("end") = "");
            },
            py::arg("mll_optz_result"),
            py::arg("label") = "",
            "Print the long fit report.");
    m.def("save_multistart_report",
          &gpflash::save_multistart_report,
          py::arg("report_path"),
          py::arg("mll_optz_result"),
          py::arg("label") = "",
          py::call_guard<py::gil_scoped_release>(),
          "Write the long fit report to a file, replacing any file there.");

    py::class_<gpflash::PosteriorResult> posterior_result(m, "PosteriorResult", "What a posterior query returns.");
    def_array_ro(posterior_result, "mean_post_vec", &gpflash::PosteriorResult::mean_post_vec, "(n_post), or empty.");
    def_array_ro(posterior_result, "var_post_vec", &gpflash::PosteriorResult::var_post_vec, "(n_post), or empty.");
    def_array_ro(
            posterior_result,
            "d_mean_post_d_x_post_ndim_npost_mat",
            &gpflash::PosteriorResult::d_mean_post_d_x_post_ndim_npost_mat,
            "(n_dim, n_post), or empty.");
    def_array_ro(
            posterior_result,
            "d_var_post_d_x_post_ndim_npost_mat",
            &gpflash::PosteriorResult::d_var_post_d_x_post_ndim_npost_mat,
            "(n_dim, n_post), or empty.");

    // ── Cache files ──────────────────────────────────────────────────────────
    // From C++'s gpflash::cache namespace, flattened into the module like the kernel functions.
    // None is a Python keyword, so that value is None_, after PEP 8's trailing underscore.
    py::enum_<gpflash::cache::SymmetryKind>(m, "SymmetryKind", "The symmetry a cache file records.")
            .value("None_", gpflash::cache::SymmetryKind::None)
            .value("Even", gpflash::cache::SymmetryKind::Even)
            .value("Odd", gpflash::cache::SymmetryKind::Odd);

    py::class_<gpflash::cache::PosteriorModelInfo> post_model_info(
            m, "PosteriorModelInfo", "The model a posterior was fitted to, and what that fit scored.");
    post_model_info.def_readonly("kernel_id", &gpflash::cache::PosteriorModelInfo::kernel_id);
    post_model_info.def_readonly("input_hash", &gpflash::cache::PosteriorModelInfo::input_hash);
    post_model_info.def_readonly("mll", &gpflash::cache::PosteriorModelInfo::mll);
    post_model_info.def_readonly(
            "sig2_f_val", &gpflash::cache::PosteriorModelInfo::sig2_f_val, "The noise variance the fit used.");
    post_model_info.def_readonly("cond_max", &gpflash::cache::PosteriorModelInfo::cond_max);
    post_model_info.def_readonly("kmat_eig_max_ub_method", &gpflash::cache::PosteriorModelInfo::kmat_eig_max_ub_method);
    post_model_info.def_readonly(
            "mean_model_eig_max_ub_method", &gpflash::cache::PosteriorModelInfo::mean_model_eig_max_ub_method);
    post_model_info.def("to_summary_string", [](const gpflash::cache::PosteriorModelInfo& info) {
        return gpflash::cache::to_summary_string(info);
    });

    py::class_<gpflash::cache::CacheInfo> cache_info(
            m, "CacheInfo", "What a cache file says about itself, read without loading its arrays.");
    cache_info.def_readonly("post_model_info", &gpflash::cache::CacheInfo::post_model_info);
    cache_info.def_readonly("format_version", &gpflash::cache::CacheInfo::format_version);
    cache_info.def_readonly(
            "has_variance_data", &gpflash::cache::CacheInfo::has_variance_data, "False for a mean-only file.");
    cache_info.def_readonly("n_train", &gpflash::cache::CacheInfo::n_train);
    cache_info.def_readonly("n_dim", &gpflash::cache::CacheInfo::n_dim);
    cache_info.def_readonly("n_feature", &gpflash::cache::CacheInfo::n_feature);
    cache_info.def_readonly("version_major", &gpflash::cache::CacheInfo::version_major, "The writing release.");
    cache_info.def_readonly("version_minor", &gpflash::cache::CacheInfo::version_minor);
    cache_info.def_readonly("version_patch", &gpflash::cache::CacheInfo::version_patch);
    cache_info.def_readonly(
            "saved_unix_time_s", &gpflash::cache::CacheInfo::saved_unix_time_s, "Seconds since the Unix epoch.");
    cache_info.def_readonly("hp_sig2_k", &gpflash::cache::CacheInfo::hp_sig2_k);
    cache_info.def_readonly("has_mean_model_constant", &gpflash::cache::CacheInfo::has_mean_model_constant);
    cache_info.def_readonly("symmetry_kind", &gpflash::cache::CacheInfo::symmetry_kind);
    cache_info.def_readonly("note", &gpflash::cache::CacheInfo::note, "What the writer passed to save_precompute.");
    cache_info.def(
            "to_summary_string",
            [](const gpflash::cache::CacheInfo& info, const std::string& label) {
                return gpflash::cache::to_summary_string(info, label);
            },
            py::arg("label") = "");

    m.def("peek_info",
          &gpflash::cache::peek_info,
          py::arg("path"),
          py::call_guard<py::gil_scoped_release>(),
          "Read a cache file's header without loading its arrays.");
    m.def("peek_kernel_id",
          &gpflash::cache::peek_kernel_id,
          py::arg("path"),
          py::call_guard<py::gil_scoped_release>(),
          "The kernel a cache file was written with.");
    // Through Python's print rather than C's stdout, so the summary also appears in a notebook.
    m.def(
            "print_cache_info",
            [](const gpflash::cache::CacheInfo& info, const std::string& label) {
                py::print(gpflash::cache::to_summary_string(info, label), py::arg("end") = "");
            },
            py::arg("cache_info"),
            py::arg("label") = "",
            "Print what a cache file says about itself.");
    m.def("calc_default_cache_path",
          &gpflash::cache::calc_default_cache_path,
          py::arg("kernel_id"),
          py::arg("train_data"),
          py::arg("directory") = "",
          "A readable cache filename from the kernel and the training set's shape.");

    // ── The posterior ────────────────────────────────────────────────────────
    py::class_<gpflash::GpPosterior>(m, "GpPosterior", "A fitted posterior: queries, and saving to a cache file.")
            .def("eval",
                 &eval_posterior,
                 py::arg("X_ndim_npost_mat"),
                 py::arg_v("post_opts", gpflash::PosteriorOptions(), "PosteriorOptions()"),
                 "Query at (n_dim, n_post) points, one per column, or at one 1-D point.")
            .def("n_train", &gpflash::GpPosterior::n_train)
            .def("n_dim", &gpflash::GpPosterior::n_dim)
            .def("n_feature", &gpflash::GpPosterior::n_feature)
            .def("hp_gamma_vec", [](const gpflash::GpPosterior& post) { return to_numpy(post.hp_gamma_vec()); })
            .def("hp_kern_extra_vec",
                 [](const gpflash::GpPosterior& post) { return to_numpy(post.hp_kern_extra_vec()); })
            .def("hp_beta_vec", [](const gpflash::GpPosterior& post) { return to_numpy(post.hp_beta_vec()); })
            .def("hp_sig2_k", &gpflash::GpPosterior::hp_sig2_k)
            .def("has_variance_data", &gpflash::GpPosterior::has_variance_data)
            .def("kernel_id", &gpflash::GpPosterior::kernel_id)
            .def("input_hash", &gpflash::GpPosterior::input_hash)
            .def("mll", &gpflash::GpPosterior::mll)
            .def("post_model_info",
                 &gpflash::GpPosterior::post_model_info,
                 py::return_value_policy::copy,
                 "The record of the fit this posterior came from.")
            .def("save_precompute",
                 &gpflash::GpPosterior::save_precompute,
                 py::arg("path"),
                 py::arg("note") = "",
                 py::arg("include_variance_data") = true,
                 py::call_guard<py::gil_scoped_release>(),
                 "Save to an HDF5 cache file that load_posterior, in Python or C++, reads back.");

    // ── Computations ─────────────────────────────────────────────────────────
    m.def("eval_mll",
          py::overload_cast<const gpflash::TrainingData&, const gpflash::HpVal&, const gpflash::ModelOptions&, bool>(
                  &gpflash::eval_mll),
          py::arg("train_data"),
          py::arg("hp_val_struct"),
          py::arg("model_opts"),
          py::arg("return_mll_gradient") = false,
          py::call_guard<py::gil_scoped_release>(),
          "The marginal log likelihood, and its gradient when asked.");

    m.def(
            "select_hp_with_max_mll",
            [](const gpflash::TrainingData& train_data, const gpflash::OptzOptions& optz_opts) {
                return gpflash::select_hp_with_max_mll(train_data, optz_opts);
            },
            py::arg("train_data"),
            py::arg_v("optz_opts", gpflash::OptzOptions(), "OptzOptions()"),
            py::call_guard<py::gil_scoped_release>(),
            "Fit the hyperparameters by maximizing the MLL.");

    m.def("make_posterior",
          &gpflash::make_posterior,
          py::arg("train_data"),
          py::arg("hp_val_struct"),
          py::arg("model_opts"),
          py::call_guard<py::gil_scoped_release>(),
          "Build a posterior from hyperparameters held directly.");

    m.def("make_posterior_from_fit",
          &gpflash::make_posterior_from_fit,
          py::arg("train_data"),
          py::arg("mll_optz_result"),
          py::call_guard<py::gil_scoped_release>(),
          "Build the posterior of a fit; train_data must be the data the fit read.");

    m.def(
            "fit_and_make_posterior",
            [](const gpflash::TrainingData& train_data, const gpflash::OptzOptions& optz_opts) {
                return gpflash::fit_and_make_posterior(train_data, optz_opts);
            },
            py::arg("train_data"),
            py::arg_v("optz_opts", gpflash::OptzOptions(), "OptzOptions()"),
            py::call_guard<py::gil_scoped_release>(),
            "Fit, then build the posterior, in one call.");

    m.def("load_posterior",
          &gpflash::load_posterior,
          py::arg("path"),
          py::call_guard<py::gil_scoped_release>(),
          "Load a posterior saved with GpPosterior.save_precompute. Checks the file, not which fit wrote it.");

    m.def("load_posterior_verified",
          &gpflash::load_posterior_verified,
          py::arg("path"),
          py::arg("train_data"),
          py::arg("hp_val_struct"),
          py::arg("model_opts"),
          py::call_guard<py::gil_scoped_release>(),
          "Load a posterior, refusing a file these inputs did not produce.");
}
