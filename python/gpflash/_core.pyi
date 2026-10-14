"""
GpFlash C++ bindings. Names and shapes follow the C++ interface; see the gpflash package for the NumPy-style GaussianProcess wrapper.
"""
from __future__ import annotations
import numpy
import numpy.typing
import typing
__all__: list[str] = ['COND_MAX_DEFAULT', 'COND_MAX_LIMIT', 'CacheInfo', 'EigMaxUbMethod', 'GpPosterior', 'HpAtBound', 'HpVal', 'InvalidInputError', 'KernelId', 'LocalOptzResult', 'MEAN_MODEL_DEGREE_MAX', 'MllHpGrad', 'MllOptzResult', 'MllResult', 'ModelOptions', 'NotImplementedFeatureError', 'NumericalFailureError', 'NumericalOptzAlgorithm', 'OptzOptions', 'OptzStopReason', 'PosteriorModelInfo', 'PosteriorOptions', 'PosteriorResult', 'SymmetryKind', 'SymmetrySpec', 'TrainingData', 'calc_default_cache_path', 'eval_mll', 'fit_and_make_posterior', 'kernel_name', 'load_posterior', 'load_posterior_verified', 'make_posterior', 'make_posterior_from_fit', 'n_hp_kern_extra', 'parse_kernel_id', 'peek_info', 'peek_kernel_id', 'print_cache_info', 'print_multistart_report', 'save_multistart_report', 'select_hp_with_max_mll', 'to_multistart_report_string']
class InvalidInputError(ValueError):
    pass
class NumericalFailureError(ArithmeticError):
    pass
class NotImplementedFeatureError(NotImplementedError):
    pass
class KernelId:
    """
    Which kernel a model uses.
    
    Members:
    
      SqExp
    
      Matern52
    
      Matern32
    
      RatQuad
    
      Periodic
    
      Linear
    
      Quadratic
    """
    Linear: typing.ClassVar[KernelId]
    Matern32: typing.ClassVar[KernelId]
    Matern52: typing.ClassVar[KernelId]
    Periodic: typing.ClassVar[KernelId]
    Quadratic: typing.ClassVar[KernelId]
    RatQuad: typing.ClassVar[KernelId]
    SqExp: typing.ClassVar[KernelId]
    __members__: typing.ClassVar[dict[str, KernelId]]
    @typing.overload
    def __eq__(self, other: KernelId) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    @typing.overload
    def __ne__(self, other: KernelId) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class EigMaxUbMethod:
    """
    The bound on lambda_max a nugget is sized from.
    
    Members:
    
      Gershgorin
    
      Trace
    """
    Gershgorin: typing.ClassVar[EigMaxUbMethod]
    Trace: typing.ClassVar[EigMaxUbMethod]
    __members__: typing.ClassVar[dict[str, EigMaxUbMethod]]
    @typing.overload
    def __eq__(self, other: EigMaxUbMethod) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    @typing.overload
    def __ne__(self, other: EigMaxUbMethod) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class NumericalOptzAlgorithm:
    """
    The algorithm each local solve uses.
    
    Members:
    
      Sqp
    
      Ccsaq
    """
    Ccsaq: typing.ClassVar[NumericalOptzAlgorithm]
    Sqp: typing.ClassVar[NumericalOptzAlgorithm]
    __members__: typing.ClassVar[dict[str, NumericalOptzAlgorithm]]
    @typing.overload
    def __eq__(self, other: NumericalOptzAlgorithm) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    @typing.overload
    def __ne__(self, other: NumericalOptzAlgorithm) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class TrainingData:
    """
    What was measured. Fixed for a fit.
    """
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: TrainingData) -> None:
        """
        A copy.
        """
    @property
    def X_ndim_ntrain_mat(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        (n_dim, n_train) training points, one per column.
        """
    @X_ndim_ntrain_mat.setter
    def X_ndim_ntrain_mat(self, arg1: typing.Any) -> None:
        ...
    @property
    def f_train_vec(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        (n_train) function evaluations at those points.
        """
    @f_train_vec.setter
    def f_train_vec(self, arg1: typing.Any) -> None:
        ...
    @property
    def sig2_f_val(self) -> float:
        """
        Noise variance: 0 noise-free, > 0 known and fixed, < 0 estimated by the fit.
        """
    @sig2_f_val.setter
    def sig2_f_val(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class HpVal:
    """
    The hyperparameters.
    """
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: HpVal) -> None:
        """
        A copy.
        """
    @property
    def hp_gamma_vec(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        (n_dim) inverse length scales, or inverse periods for Periodic.
        """
    @hp_gamma_vec.setter
    def hp_gamma_vec(self, arg1: typing.Any) -> None:
        ...
    @property
    def hp_kern_extra_vec(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        (n_hp_kern_extra) the kernel's extra hyperparameters, or empty.
        """
    @hp_kern_extra_vec.setter
    def hp_kern_extra_vec(self, arg1: typing.Any) -> None:
        ...
    @property
    def hp_sig2_f_val(self) -> float:
        """
        Estimated noise variance, set only when TrainingData.sig2_f_val < 0; negative means unset.
        """
    @hp_sig2_f_val.setter
    def hp_sig2_f_val(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def hp_sig2_k(self) -> float:
        """
        Signal variance; negative means unset.
        """
    @hp_sig2_k.setter
    def hp_sig2_k(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class SymmetrySpec:
    """
    An axis-aligned mirror symmetry.
    """
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: SymmetrySpec) -> None:
        """
        A copy.
        """
    def is_on(self) -> bool:
        ...
    @property
    def is_even_symmetry(self) -> bool:
        """
        Even (True) or odd (False) model.
        """
    @is_even_symmetry.setter
    def is_even_symmetry(self, arg0: bool) -> None:
        ...
    @property
    def mirror_sign_vec(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        (n_dim) diagonal of the mirror, each +1 or -1; empty means no symmetry.
        """
    @mirror_sign_vec.setter
    def mirror_sign_vec(self, arg1: typing.Any) -> None:
        ...
class ModelOptions:
    """
    Which model is evaluated.
    """
    has_mean_model_constant: bool
    kernel_id: KernelId
    kmat_eig_max_ub_method: EigMaxUbMethod
    mean_model_eig_max_ub_method: EigMaxUbMethod
    symmetry_spec: SymmetrySpec
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: ModelOptions) -> None:
        """
        A copy.
        """
    @property
    def cond_max(self) -> float:
        """
        Condition-number bound that sizes the nuggets.
        """
    @cond_max.setter
    def cond_max(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def mean_model_degree_vec(self) -> numpy.typing.NDArray[numpy.int64]:
        """
        Power of each x_d in the mean model basis, 0 to MEAN_MODEL_DEGREE_MAX; one entry applies to all.
        """
    @mean_model_degree_vec.setter
    def mean_model_degree_vec(self, arg1: typing.Any) -> None:
        ...
class OptzOptions:
    """
    How the hyperparameters are fitted.
    """
    model_options: ModelOptions
    numerical_optz_algorithm: NumericalOptzAlgorithm
    scale_variance_bounds_by_residual_var: bool
    use_bounds: bool
    use_multistart: bool
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: OptzOptions) -> None:
        """
        A copy.
        """
    @property
    def hp_gamma_lb_vec(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        1 entry or n_dim.
        """
    @hp_gamma_lb_vec.setter
    def hp_gamma_lb_vec(self, arg1: typing.Any) -> None:
        ...
    @property
    def hp_gamma_ub_vec(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        1 entry or n_dim.
        """
    @hp_gamma_ub_vec.setter
    def hp_gamma_ub_vec(self, arg1: typing.Any) -> None:
        ...
    @property
    def hp_kern_extra_init(self) -> float:
        ...
    @hp_kern_extra_init.setter
    def hp_kern_extra_init(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def hp_kern_extra_lb_vec(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        1 entry or n_hp_kern_extra.
        """
    @hp_kern_extra_lb_vec.setter
    def hp_kern_extra_lb_vec(self, arg1: typing.Any) -> None:
        ...
    @property
    def hp_kern_extra_ub_vec(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        1 entry or n_hp_kern_extra.
        """
    @hp_kern_extra_ub_vec.setter
    def hp_kern_extra_ub_vec(self, arg1: typing.Any) -> None:
        ...
    @property
    def hp_sig2_f_val_init(self) -> float:
        ...
    @hp_sig2_f_val_init.setter
    def hp_sig2_f_val_init(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def hp_sig2_f_val_lb(self) -> float:
        ...
    @hp_sig2_f_val_lb.setter
    def hp_sig2_f_val_lb(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def hp_sig2_f_val_ub(self) -> float:
        ...
    @hp_sig2_f_val_ub.setter
    def hp_sig2_f_val_ub(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def hp_sig2_k_init(self) -> float:
        ...
    @hp_sig2_k_init.setter
    def hp_sig2_k_init(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def hp_sig2_k_lb(self) -> float:
        ...
    @hp_sig2_k_lb.setter
    def hp_sig2_k_lb(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def hp_sig2_k_ub(self) -> float:
        ...
    @hp_sig2_k_ub.setter
    def hp_sig2_k_ub(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def hypercube_sampling_seed(self) -> int:
        ...
    @hypercube_sampling_seed.setter
    def hypercube_sampling_seed(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def max_evals(self) -> int:
        ...
    @max_evals.setter
    def max_evals(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def max_parallel_ind_optz(self) -> int:
        ...
    @max_parallel_ind_optz.setter
    def max_parallel_ind_optz(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def n_ind_optz(self) -> int:
        ...
    @n_ind_optz.setter
    def n_ind_optz(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def rel_obj_tol(self) -> float:
        ...
    @rel_obj_tol.setter
    def rel_obj_tol(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def rel_step_tol(self) -> float:
        ...
    @rel_step_tol.setter
    def rel_step_tol(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class PosteriorOptions:
    """
    Which parts of a posterior query to compute.
    """
    check_query_points: bool
    return_mean: bool
    return_mean_gradient: bool
    return_var: bool
    return_var_gradient: bool
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: PosteriorOptions) -> None:
        """
        A copy.
        """
class MllHpGrad:
    """
    The MLL's derivative per hyperparameter.
    """
    @property
    def d_mll_d_hp_gamma_vec(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        (n_dim)
        """
    @property
    def d_mll_d_hp_kern_extra_vec(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        (n_hp_kern_extra), or empty.
        """
    @property
    def d_mll_d_hp_sig2_f_val(self) -> float:
        ...
    @property
    def d_mll_d_hp_sig2_k(self) -> float:
        ...
class MllResult:
    """
    What eval_mll returns.
    """
    @property
    def hp_beta_vec(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        (n_basis) mean model weights.
        """
    @property
    def hp_sig2_k(self) -> float:
        ...
    @property
    def hp_sig2_k_from_closed_form(self) -> bool:
        ...
    @property
    def mll(self) -> float:
        ...
    @property
    def mll_hp_grad_struct(self) -> MllHpGrad:
        ...
class OptzStopReason:
    """
    Why a local solve stopped.
    
    Members:
    
      ToleranceMet
    
      EvalBudgetSpent
    
      RoundoffLimited
    
      Failed
    """
    EvalBudgetSpent: typing.ClassVar[OptzStopReason]
    Failed: typing.ClassVar[OptzStopReason]
    RoundoffLimited: typing.ClassVar[OptzStopReason]
    ToleranceMet: typing.ClassVar[OptzStopReason]
    __members__: typing.ClassVar[dict[str, OptzStopReason]]
    @typing.overload
    def __eq__(self, other: OptzStopReason) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    @typing.overload
    def __ne__(self, other: OptzStopReason) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class LocalOptzResult:
    """
    The outcome of one local solve.
    """
    @property
    def ln_hp_vec(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        The minimizer, in log-hyperparameter space.
        """
    @property
    def n_eval(self) -> int:
        """
        Objective evaluations.
        """
    @property
    def objective_val(self) -> float:
        """
        At the minimizer; inf when Failed.
        """
    @property
    def stop_reason(self) -> OptzStopReason:
        ...
class HpAtBound:
    """
    Which hyperparameters a bound is holding.
    """
    def has_any(self) -> bool:
        ...
    def to_summary_string(self) -> str:
        ...
    @property
    def hp_gamma_at_bound_vec(self) -> numpy.typing.NDArray[numpy.int64]:
        """
        (n_dim), 1 or 0.
        """
    @property
    def hp_kern_extra_at_bound_vec(self) -> numpy.typing.NDArray[numpy.int64]:
        """
        (n_hp_kern_extra), 1 or 0; empty when the kernel has no extras.
        """
    @property
    def hp_sig2_f_val_at_bound(self) -> bool:
        ...
    @property
    def hp_sig2_k_at_bound(self) -> bool:
        ...
class MllOptzResult:
    """
    What select_hp_with_max_mll returns.
    """
    def build_best_local_optz_result(self) -> LocalOptzResult:
        """
        The winning solve; its stop_reason says why it stopped.
        """
    def calc_at_bound_flags_vec(self, i_ind_optz: typing.SupportsInt | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.int64]:
        """
        Which entries optimization i_ind_optz left on a bound, 1 or 0 each, in the search's order.
        """
    @typing.overload
    def calc_hp_at_bound_struct(self) -> HpAtBound:
        """
        Which hyperparameters the winning optimization left on a bound.
        """
    @typing.overload
    def calc_hp_at_bound_struct(self, i_ind_optz: typing.SupportsInt | typing.SupportsIndex) -> HpAtBound:
        """
        Which hyperparameters optimization i_ind_optz left on a bound.
        """
    def calc_n_at_bound(self, i_ind_optz: typing.SupportsInt | typing.SupportsIndex) -> int:
        """
        How many hyperparameters optimization i_ind_optz left on a bound; idx_best() is the winner.
        """
    def calc_n_failed_start(self) -> int:
        ...
    def idx_best(self) -> int:
        ...
    def mll(self) -> float:
        ...
    def n_ind_optz_run(self) -> int:
        ...
    def n_total_mll_eval(self) -> int:
        ...
    def n_total_mll_grad_eval(self) -> int:
        ...
    def to_summary_string(self, label: str = '') -> str:
        ...
    @property
    def hp_val_struct(self) -> HpVal:
        ...
    @property
    def model_options(self) -> ModelOptions:
        ...
    @property
    def optimality(self) -> float:
        """
        Projected gradient norm at the winner.
        """
class PosteriorResult:
    """
    What a posterior query returns.
    """
    @property
    def d_mean_post_d_x_post_ndim_npost_mat(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        (n_dim, n_post), or empty.
        """
    @property
    def d_var_post_d_x_post_ndim_npost_mat(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        (n_dim, n_post), or empty.
        """
    @property
    def mean_post_vec(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        (n_post), or empty.
        """
    @property
    def var_post_vec(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        (n_post), or empty.
        """
class SymmetryKind:
    """
    The symmetry a cache file records.
    
    Members:
    
      None_
    
      Even
    
      Odd
    """
    Even: typing.ClassVar[SymmetryKind]
    None_: typing.ClassVar[SymmetryKind]
    Odd: typing.ClassVar[SymmetryKind]
    __members__: typing.ClassVar[dict[str, SymmetryKind]]
    @typing.overload
    def __eq__(self, other: SymmetryKind) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    @typing.overload
    def __ne__(self, other: SymmetryKind) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class PosteriorModelInfo:
    """
    The model a posterior was fitted to, and what that fit scored.
    """
    def to_summary_string(self) -> str:
        ...
    @property
    def cond_max(self) -> float:
        ...
    @property
    def input_hash(self) -> int:
        ...
    @property
    def kernel_id(self) -> KernelId:
        ...
    @property
    def kmat_eig_max_ub_method(self) -> EigMaxUbMethod:
        ...
    @property
    def mean_model_eig_max_ub_method(self) -> EigMaxUbMethod:
        ...
    @property
    def mll(self) -> float:
        ...
    @property
    def sig2_f_val(self) -> float:
        """
        The noise variance the fit used.
        """
class CacheInfo:
    """
    What a cache file says about itself, read without loading its arrays.
    """
    def to_summary_string(self, label: str = '') -> str:
        ...
    @property
    def format_version(self) -> int:
        ...
    @property
    def has_mean_model_constant(self) -> bool:
        ...
    @property
    def has_variance_data(self) -> bool:
        """
        False for a mean-only file.
        """
    @property
    def hp_sig2_k(self) -> float:
        ...
    @property
    def n_dim(self) -> int:
        ...
    @property
    def n_feature(self) -> int:
        ...
    @property
    def n_train(self) -> int:
        ...
    @property
    def note(self) -> str:
        """
        What the writer passed to save_precompute.
        """
    @property
    def post_model_info(self) -> PosteriorModelInfo:
        ...
    @property
    def saved_unix_time_s(self) -> int:
        """
        Seconds since the Unix epoch.
        """
    @property
    def symmetry_kind(self) -> SymmetryKind:
        ...
    @property
    def version_major(self) -> int:
        """
        The writing release.
        """
    @property
    def version_minor(self) -> int:
        ...
    @property
    def version_patch(self) -> int:
        ...
class GpPosterior:
    """
    A fitted posterior: queries, and saving to a cache file.
    """
    def eval(self, X_ndim_npost_mat: typing.Any, post_opts: PosteriorOptions = ...) -> PosteriorResult:
        """
        Query at (n_dim, n_post) points, one per column, or at one 1-D point.
        """
    def has_variance_data(self) -> bool:
        ...
    def hp_beta_vec(self) -> numpy.typing.NDArray[numpy.float64]:
        ...
    def hp_gamma_vec(self) -> numpy.typing.NDArray[numpy.float64]:
        ...
    def hp_kern_extra_vec(self) -> numpy.typing.NDArray[numpy.float64]:
        ...
    def hp_sig2_k(self) -> float:
        ...
    def input_hash(self) -> int:
        ...
    def kernel_id(self) -> KernelId:
        ...
    def mll(self) -> float:
        ...
    def n_dim(self) -> int:
        ...
    def n_feature(self) -> int:
        ...
    def n_train(self) -> int:
        ...
    def post_model_info(self) -> PosteriorModelInfo:
        """
        The record of the fit this posterior came from.
        """
    def save_precompute(self, path: str, note: str = '', include_variance_data: bool = True) -> None:
        """
        Save to an HDF5 cache file that load_posterior, in Python or C++, reads back.
        """
def calc_default_cache_path(kernel_id: KernelId, train_data: TrainingData, directory: str = '') -> str:
    """
    A readable cache filename from the kernel and the training set's shape.
    """
def eval_mll(train_data: TrainingData, hp_val_struct: HpVal, model_opts: ModelOptions, return_mll_gradient: bool = False) -> MllResult:
    """
    The marginal log likelihood, and its gradient when asked.
    """
def fit_and_make_posterior(train_data: TrainingData, optz_opts: OptzOptions = ...) -> GpPosterior:
    """
    Fit, then build the posterior, in one call.
    """
def kernel_name(kernel_id: KernelId) -> str:
    """
    The kernel's name, e.g. 'matern_5_2'; the inverse of parse_kernel_id.
    """
def load_posterior(path: str) -> GpPosterior:
    """
    Load a posterior saved with GpPosterior.save_precompute. Checks the file, not which fit wrote it.
    """
def load_posterior_verified(path: str, train_data: TrainingData, hp_val_struct: HpVal, model_opts: ModelOptions) -> GpPosterior:
    """
    Load a posterior, refusing a file these inputs did not produce.
    """
def make_posterior(train_data: TrainingData, hp_val_struct: HpVal, model_opts: ModelOptions) -> GpPosterior:
    """
    Build a posterior from hyperparameters held directly.
    """
def make_posterior_from_fit(train_data: TrainingData, mll_optz_result: MllOptzResult) -> GpPosterior:
    """
    Build the posterior of a fit; train_data must be the data the fit read.
    """
def n_hp_kern_extra(kernel_id: KernelId, n_dim: typing.SupportsInt | typing.SupportsIndex) -> int:
    """
    How many entries HpVal.hp_kern_extra_vec holds for this kernel in n_dim dimensions.
    """
def parse_kernel_id(kernel_name: str) -> KernelId:
    """
    The KernelId for a name such as 'sq_exp'; raises InvalidInputError for an unknown one.
    """
def peek_info(path: str) -> CacheInfo:
    """
    Read a cache file's header without loading its arrays.
    """
def peek_kernel_id(path: str) -> KernelId:
    """
    The kernel a cache file was written with.
    """
def print_cache_info(cache_info: CacheInfo, label: str = '') -> None:
    """
    Print what a cache file says about itself.
    """
def print_multistart_report(mll_optz_result: MllOptzResult, label: str = '') -> None:
    """
    Print the long fit report.
    """
def save_multistart_report(report_path: str, mll_optz_result: MllOptzResult, label: str = '') -> None:
    """
    Write the long fit report to a file, replacing any file there.
    """
def select_hp_with_max_mll(train_data: TrainingData, optz_opts: OptzOptions = ...) -> MllOptzResult:
    """
    Fit the hyperparameters by maximizing the MLL.
    """
def to_multistart_report_string(mll_optz_result: MllOptzResult, label: str = '') -> str:
    """
    The long fit report: the fit, a row per optimization, where each landed and started, and the terms.
    """
COND_MAX_DEFAULT: float = 10000000000.0
COND_MAX_LIMIT: float = 10000000000000.0
MEAN_MODEL_DEGREE_MAX: int = 2
__version__: str
