"""
Tests of Parametrization, BatchParametrization and the implemented parametrizations.
"""

import numpy
import pytest

import pysolvegn

from problems import numerical_jacobian

RNG = numpy.random.default_rng(0)

# (name, single builder, batch builder, kwargs, n_parameters)
CASES = [
    ("affine", pysolvegn.build_affine_parametrization, pysolvegn.build_batch_affine_parametrization,
     dict(modes=[[1.0, 2.0], [0.0, 1.0], [3.0, -1.0]], offset=[0.0, 1.0, 2.0]), 2),
    ("affine_no_offset", pysolvegn.build_affine_parametrization, pysolvegn.build_batch_affine_parametrization,
     dict(modes=[[1.0, 2.0]]), 2),
    ("fixed", pysolvegn.build_fixed_parametrization, pysolvegn.build_batch_fixed_parametrization,
     dict(n_p_outputs=4, optimized_indices=[2, 0], fixed_parameters=[1.0, 2.0, 3.0, 4.0]), 2),
    ("sigmoid", pysolvegn.build_sigmoid_parametrization, pysolvegn.build_batch_sigmoid_parametrization,
     dict(lower=[0.0, -1.0, 5.0], upper=[1.0, 1.0, 9.0]), 3),
    ("positive", pysolvegn.build_positive_parametrization, pysolvegn.build_batch_positive_parametrization,
     dict(n_parameters=3), 3),
]


@pytest.mark.parametrize("name, single_builder, batch_builder, kwargs, n", CASES)
def test_single_jacobian_finite_differences(name, single_builder, batch_builder, kwargs, n):
    parametrization = single_builder(**kwargs)
    p = RNG.normal(size=n)
    numpy.testing.assert_allclose(
        parametrization.J_func(p), numerical_jacobian(parametrization.p_func, p), atol=1e-7
    )


@pytest.mark.parametrize("name, single_builder, batch_builder, kwargs, n", CASES)
def test_batch_equals_single(name, single_builder, batch_builder, kwargs, n):
    single = single_builder(**kwargs)
    batch = batch_builder(**kwargs)
    P = 3.0 * RNG.normal(size=(6, n))
    out = batch.p_func(P)
    J = batch.J_func(P)
    assert out.ndim == 2 and out.shape[0] == 6
    assert J.shape == (6, out.shape[1], n)
    for a in range(6):
        numpy.testing.assert_allclose(out[a], single.p_func(P[a]))
        numpy.testing.assert_allclose(J[a], single.J_func(P[a]))


@pytest.mark.parametrize("name, single_builder, batch_builder, kwargs, n", CASES)
def test_batch_wrong_shape(name, single_builder, batch_builder, kwargs, n):
    batch = batch_builder(**kwargs)
    with pytest.raises(ValueError):
        batch.p_func(numpy.zeros(n))  # 1D instead of (m, n)
    with pytest.raises(ValueError):
        batch.p_func(numpy.zeros((2, n + 1)))


def test_sigmoid_bounds():
    parametrization = pysolvegn.build_sigmoid_parametrization([0.0, -1.0], [1.0, 1.0])
    out = parametrization.p_func(numpy.array([-800.0, 800.0]))  # no overflow
    assert numpy.all(numpy.isfinite(out))
    assert 0.0 <= out[0] <= 1.0 and -1.0 <= out[1] <= 1.0


def test_fixed_values():
    parametrization = pysolvegn.build_fixed_parametrization(3, [0, 2], [1.0, 2.0, 3.0])
    numpy.testing.assert_allclose(parametrization.p_func(numpy.array([7.0, 9.0])), [7.0, 2.0, 9.0])


@pytest.mark.parametrize(
    "builder, kwargs",
    [
        (pysolvegn.build_affine_parametrization, dict(modes=[1.0, 2.0])),
        (pysolvegn.build_affine_parametrization, dict(modes=[[1.0]], offset=[0.0, 1.0])),
        (pysolvegn.build_fixed_parametrization, dict(n_p_outputs=3, optimized_indices=[])),
        (pysolvegn.build_fixed_parametrization, dict(n_p_outputs=3, optimized_indices=[0, 0])),
        (pysolvegn.build_fixed_parametrization, dict(n_p_outputs=3, optimized_indices=[3])),
        (pysolvegn.build_sigmoid_parametrization, dict(lower=[1.0], upper=[1.0])),
        (pysolvegn.build_sigmoid_parametrization, dict(lower=[0.0, 0.0], upper=[1.0])),
        (pysolvegn.build_positive_parametrization, dict(n_parameters=0)),
        (pysolvegn.build_batch_sigmoid_parametrization, dict(lower=[2.0], upper=[1.0])),
        (pysolvegn.build_batch_fixed_parametrization, dict(n_p_outputs=2, optimized_indices=[5])),
    ],
)
def test_invalid_arguments(builder, kwargs):
    with pytest.raises(ValueError):
        builder(**kwargs)


def test_parametrization_finite_difference():
    p_func = lambda p: numpy.array([numpy.exp(p[0]), p[0] * p[1]])
    parametrization = pysolvegn.Parametrization(p_func, finite_difference="central")
    p = numpy.array([0.3, -0.7])
    numpy.testing.assert_allclose(parametrization.J_func(p), numerical_jacobian(p_func, p), atol=1e-6)


def test_batch_parametrization_finite_difference():
    p_func = lambda P: numpy.column_stack((numpy.exp(P[:, 0]), P[:, 0] * P[:, 1]))
    parametrization = pysolvegn.BatchParametrization(p_func, finite_difference="central")
    P = RNG.normal(size=(4, 2))
    J = parametrization.J_func(P)  # no indices for a BatchParametrization
    for a in range(4):
        numpy.testing.assert_allclose(J[a], numerical_jacobian(lambda q: p_func(q[None])[0], P[a]), atol=1e-6)


@pytest.mark.parametrize("cls", [pysolvegn.Parametrization, pysolvegn.BatchParametrization])
def test_parametrization_invalid(cls):
    with pytest.raises(ValueError):
        cls(None)
    with pytest.raises(ValueError):
        cls(lambda p: p)  # neither jacobian_func nor finite_difference
    with pytest.raises(ValueError):
        cls(lambda p: p, lambda p: p, finite_difference="central")
    with pytest.raises(ValueError):
        cls(lambda p: p, finite_difference="unknown")
