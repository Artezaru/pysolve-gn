"""
Tests of the implemented regularizations (single and batch).
"""

import numpy
import pytest

import pysolvegn
from pysolvegn.evaluation import _evaluate_system
from pysolvegn.batch_evaluation import _evaluate_batch_system

RNG = numpy.random.default_rng(0)
K, N = 4, 5
MEANS = RNG.normal(size=(K, N))
STDS = RNG.uniform(0.5, 2.0, (K, N))
THRESHOLDS = RNG.uniform(0.0, 1.0, (K, N))
P = 2.0 * RNG.normal(size=(K, N))

CASES = [
    ("squared", pysolvegn.build_squared_regularization, pysolvegn.build_batch_squared_regularization, (MEANS, STDS)),
    ("soft", pysolvegn.build_soft_squared_regularization, pysolvegn.build_batch_soft_squared_regularization, (MEANS, THRESHOLDS, STDS)),
    ("absolute", pysolvegn.build_absolute_regularization, pysolvegn.build_batch_absolute_regularization, (MEANS, STDS)),
]


def cost(term, p):
    return _evaluate_system([term], numpy.asarray(p, dtype=float), None, compute_cost=True).cost


def test_squared_cost():
    term = pysolvegn.build_squared_regularization(MEANS[0], STDS[0], weight=2.0)
    expected = 2.0 * 0.5 * numpy.sum(((P[0] - MEANS[0]) / STDS[0]) ** 2)
    numpy.testing.assert_allclose(cost(term, P[0]), expected)


def test_soft_squared_null_inside_threshold():
    term = pysolvegn.build_soft_squared_regularization([0.0, 1.0], [0.5, 0.5], [1.0, 2.0])
    assert cost(term, [0.4, 1.3]) == 0.0
    numpy.testing.assert_allclose(term.residual_func(numpy.array([1.0, 0.0])), [0.5, -0.25])
    numpy.testing.assert_allclose(term.jacobian_func(numpy.array([0.4, 0.0])), numpy.diag([0.0, 0.5]))


def test_absolute_charbonnier_cost():
    epsilon = 1e-2
    term = pysolvegn.build_absolute_regularization(MEANS[0], STDS[0], weight=2.0, epsilon=epsilon)
    z = (P[0] - MEANS[0]) / STDS[0]
    expected = 2.0 * numpy.sum(numpy.sqrt(z**2 + epsilon**2) - epsilon)
    numpy.testing.assert_allclose(cost(term, P[0]), expected)


def test_absolute_promotes_sparsity():
    # Sparse recovery: the L1 regularization sets the useless parameters to 0
    rng = numpy.random.default_rng(1)
    A = rng.normal(size=(40, 20))
    p_true = numpy.zeros(20)
    p_true[[2, 7, 13]] = [3.0, -2.0, 1.5]
    y = A @ p_true + 0.1 * rng.normal(size=40)
    data = pysolvegn.Term.from_rJ(lambda p: A @ p - y, lambda p: A)
    l1 = pysolvegn.build_absolute_regularization(numpy.zeros(20), numpy.ones(20), weight=5.0, epsilon=1e-4)
    l2 = pysolvegn.build_squared_regularization(numpy.zeros(20), numpy.ones(20), weight=5.0)
    kwargs = dict(max_iteration=300, ftol=1e-12, damping="lm-diag")
    result_l1 = pysolvegn.solve([data, l1], numpy.zeros(20), **kwargs)
    result_l2 = pysolvegn.solve([data, l2], numpy.zeros(20), **kwargs)
    assert result_l1.success and result_l2.success
    zeros = numpy.setdiff1d(numpy.arange(20), [2, 7, 13])
    assert numpy.all(numpy.abs(result_l1.parameters[zeros]) < 1e-2)
    assert numpy.sum(numpy.abs(result_l2.parameters[zeros]) < 1e-2) < 17


@pytest.mark.parametrize("name, single_builder, batch_builder, arrays", CASES)
@pytest.mark.parametrize("shared", [False, True])
def test_batch_equals_single(name, single_builder, batch_builder, arrays, shared):
    batch_arrays = tuple(a[0] if shared else a for a in arrays)
    weights = numpy.arange(1.0, K + 1.0)
    batch_term = batch_builder(*batch_arrays, weight=weights)
    indices = numpy.array([3, 0, 2])
    state = _evaluate_batch_system([batch_term], P[indices], indices, None, compute_cost=True)
    for a, k in enumerate(indices):
        single_arrays = tuple(x if shared else x[k] for x in batch_arrays)
        single = _evaluate_system([single_builder(*single_arrays, weight=weights[k])], P[k], None, compute_cost=True)
        numpy.testing.assert_allclose(state.hessian[a], single.hessian)
        numpy.testing.assert_allclose(state.second_term[a], single.second_term)
        numpy.testing.assert_allclose(state.cost[a], single.cost)


@pytest.mark.parametrize("name, single_builder, batch_builder, arrays", CASES)
def test_invalid_arrays(name, single_builder, batch_builder, arrays):
    single_arrays = tuple(a[0] for a in arrays)
    with pytest.raises(ValueError):  # negative stds
        single_builder(*single_arrays[:-1], -single_arrays[-1])
    with pytest.raises(ValueError):  # NaN
        single_builder(numpy.full(N, numpy.nan), *single_arrays[1:])
    with pytest.raises(ValueError):  # different sizes
        single_builder(single_arrays[0][:-1], *single_arrays[1:])
    with pytest.raises(ValueError):  # 2D arrays in single mode
        single_builder(*arrays)
    with pytest.raises(ValueError):  # different numbers of problems in batch mode
        batch_builder(arrays[0][:-1], *arrays[1:])


def test_soft_squared_negative_threshold():
    with pytest.raises(ValueError):
        pysolvegn.build_soft_squared_regularization([0.0], [-0.1], [1.0])


@pytest.mark.parametrize("bad", [0.0, -1e-3, numpy.nan])
def test_absolute_invalid_epsilon(bad):
    with pytest.raises(ValueError):
        pysolvegn.build_absolute_regularization([0.0], [1.0], epsilon=bad)


def test_batch_regularization_with_m_changing():
    # The callables must accept any subset of problems (m changes during the optimization)
    term = pysolvegn.build_batch_squared_regularization(MEANS, STDS)
    for indices in (numpy.arange(K), numpy.array([2]), numpy.array([3, 1])):
        r = term.residual_func(P[indices], indices)
        numpy.testing.assert_allclose(r, (P[indices] - MEANS[indices]) / STDS[indices])
