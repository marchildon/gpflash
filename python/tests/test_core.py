"""The binding layer: names, enums, struct fields, array conversion and error mapping."""

import numpy as np
import pytest

import gpflash
from gpflash import _core


def test_version_matches_package_metadata():
    from importlib.metadata import version

    assert gpflash.__version__ == version("gpflash")


def test_kernel_names_round_trip():
    for kernel_id in gpflash.KernelId.__members__.values():
        assert gpflash.parse_kernel_id(gpflash.kernel_name(kernel_id)) == kernel_id
    assert gpflash.parse_kernel_id("matern_5_2") == gpflash.KernelId.Matern52


def test_unknown_kernel_raises_invalid_input():
    with pytest.raises(gpflash.InvalidInputError):
        gpflash.parse_kernel_id("not_a_kernel")
    # InvalidInputError is a ValueError, so either except clause catches it.
    with pytest.raises(ValueError):
        gpflash.parse_kernel_id("not_a_kernel")


def test_n_hp_kern_extra():
    assert gpflash.n_hp_kern_extra(gpflash.KernelId.SqExp, 3) == 0
    assert gpflash.n_hp_kern_extra(gpflash.KernelId.RatQuad, 3) == 1


def test_defaults_match_cpp():
    model_opts = gpflash.ModelOptions()
    assert model_opts.kernel_id == gpflash.KernelId.Matern52
    assert model_opts.cond_max == gpflash.COND_MAX_DEFAULT
    assert model_opts.has_mean_model_constant
    np.testing.assert_array_equal(model_opts.mean_model_degree_vec, [0])
    optz_opts = gpflash.OptzOptions()
    assert not optz_opts.use_multistart
    assert optz_opts.numerical_optz_algorithm == gpflash.NumericalOptzAlgorithm.Sqp
    hp_val = gpflash.HpVal()
    assert hp_val.hp_sig2_k < 0.0
    assert hp_val.hp_gamma_vec.shape == (0,)


def test_array_fields_round_trip_and_are_copies():
    train_data = gpflash.TrainingData()
    X = np.arange(6.0).reshape(2, 3)
    train_data.X_ndim_ntrain_mat = X
    np.testing.assert_array_equal(train_data.X_ndim_ntrain_mat, X)
    # A value, not a view: editing what a getter returned leaves the struct unchanged.
    train_data.X_ndim_ntrain_mat[0, 0] = 99.0
    assert train_data.X_ndim_ntrain_mat[0, 0] == 0.0
    # Non-contiguous and integer inputs are converted.
    train_data.X_ndim_ntrain_mat = np.arange(12).reshape(3, 4)[::2, ::2]
    np.testing.assert_array_equal(train_data.X_ndim_ntrain_mat, [[0.0, 2.0], [8.0, 10.0]])


def test_array_field_rejects_wrong_rank():
    hp_val = gpflash.HpVal()
    with pytest.raises(ValueError):
        hp_val.hp_gamma_vec = np.ones((2, 2))
    train_data = gpflash.TrainingData()
    with pytest.raises(ValueError):
        train_data.X_ndim_ntrain_mat = np.ones(3)


def test_nested_options_are_edited_in_place():
    optz_opts = gpflash.OptzOptions()
    optz_opts.model_options.kernel_id = gpflash.KernelId.SqExp
    optz_opts.model_options.symmetry_spec.is_even_symmetry = False
    assert optz_opts.model_options.kernel_id == gpflash.KernelId.SqExp
    assert not optz_opts.model_options.symmetry_spec.is_even_symmetry


def test_copy_constructor_is_independent():
    model_opts = gpflash.ModelOptions()
    model_copy = gpflash.ModelOptions(model_opts)
    model_copy.kernel_id = gpflash.KernelId.SqExp
    assert model_opts.kernel_id == gpflash.KernelId.Matern52


def test_invalid_hyperparameters_raise_invalid_input():
    train_data = gpflash.TrainingData()
    train_data.X_ndim_ntrain_mat = np.random.default_rng(0).random((2, 5))
    train_data.f_train_vec = np.zeros(5)
    hp_val = gpflash.HpVal()
    hp_val.hp_gamma_vec = [1.0]  # n_dim is 2
    with pytest.raises(gpflash.InvalidInputError, match="hp_gamma_vec"):
        gpflash.eval_mll(train_data, hp_val, gpflash.ModelOptions())


def test_exception_hierarchy():
    assert issubclass(gpflash.InvalidInputError, ValueError)
    assert issubclass(gpflash.NumericalFailureError, ArithmeticError)
    assert issubclass(gpflash.NotImplementedFeatureError, NotImplementedError)
    assert _core.InvalidInputError is gpflash.InvalidInputError
