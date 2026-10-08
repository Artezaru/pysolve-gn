"""
Tests of Term and BatchTerm: construction, validation and evaluation.
"""

import numpy
import pytest

import pysolvegn
from pysolvegn.evaluation import _evaluate_system
from pysolvegn.batch_evaluation import _evaluate_batch_system


from problems import X_EXP, exp_data, exp_term, numerical_jacobian

r_func = lambda p: p - 1.0
J_func = lambda p: numpy.eye(p.size)
g_func = lambda p: p - 1.0
H_func = lambda p: numpy.eye(p.size)


# ----------------------------------------------------------------------
# Term: construction
# ----------------------------------------------------------------------


def test_term_rJ_defaults():
    term = pysolvegn.Term.from_rJ(r_func, J_func)
    assert term.type == "rJ"
    assert term.loss == "linear"
    assert term.weight == 1.0
    assert term.loss_scale == 1.0


def test_term_gH():
    term = pysolvegn.Term.from_gH(g_func, H_func, weight=2.0, cost_func=lambda p: 0.0)
    assert term.type == "gH"
    assert term.loss is None
    assert term.weight == 2.0
    assert term.cost_func is not None


def test_term_custom_loss():
    custom = lambda z: (z, numpy.ones_like(z), numpy.zeros_like(z))
    term = pysolvegn.Term.from_rJ(r_func, J_func, loss=custom)
    assert term.loss == "custom"


def test_term_loss_name_case_insensitive():
    assert pysolvegn.Term.from_rJ(r_func, J_func, loss="HUBER").loss == "huber"


def test_term_loss_scale_wraps_loss():
    term = pysolvegn.Term.from_rJ(r_func, J_func, loss="cauchy", loss_scale=2.0)
    assert term.loss_scale == 2.0
    numpy.testing.assert_allclose(term.loss_func(numpy.array([4.0]))[0], 4.0 * numpy.log(2.0))


def test_term_both_rJ_and_gH():
    with pytest.raises(ValueError):
        pysolvegn.Term(residual_func=r_func, jacobian_func=J_func, gradient_func=g_func, hessian_func=H_func)


def test_term_nothing():
    with pytest.raises(ValueError):
        pysolvegn.Term()


def test_term_missing_jacobian():
    with pytest.raises(ValueError):
        pysolvegn.Term(residual_func=r_func)


def test_term_jacobian_and_finite_difference():
    with pytest.raises(ValueError):
        pysolvegn.Term(residual_func=r_func, jacobian_func=J_func, finite_difference="central")


def test_term_gradient_without_hessian():
    with pytest.raises(ValueError):
        pysolvegn.Term(gradient_func=g_func)


def test_term_gH_with_loss():
    with pytest.raises(ValueError):
        pysolvegn.Term(gradient_func=g_func, hessian_func=H_func, loss="cauchy")


def test_term_gH_with_loss_scale():
    with pytest.raises(ValueError):
        pysolvegn.Term(gradient_func=g_func, hessian_func=H_func, loss_scale=2.0)


def test_term_invalid_loss_name():
    with pytest.raises(ValueError):
        pysolvegn.Term.from_rJ(r_func, J_func, loss="unknown")


def test_term_invalid_finite_difference():
    with pytest.raises(ValueError):
        pysolvegn.Term(residual_func=r_func, finite_difference="unknown")


def test_term_not_callable():
    with pytest.raises(TypeError):
        pysolvegn.Term(residual_func=1.0, jacobian_func=J_func)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_term_invalid_weight(bad):
    with pytest.raises(ValueError):
        pysolvegn.Term.from_rJ(r_func, J_func, weight=bad)
    term = pysolvegn.Term.from_rJ(r_func, J_func)
    with pytest.raises(ValueError):
        term.weight = bad


@pytest.mark.parametrize("bad", [0.0, -1.0, numpy.nan, numpy.inf])
def test_term_invalid_loss_scale(bad):
    with pytest.raises(ValueError):
        pysolvegn.Term.from_rJ(r_func, J_func, loss="cauchy", loss_scale=bad)


@pytest.mark.parametrize("method", ["central", "forward", "backward"])
def test_term_finite_difference_jacobian(method):
    y = exp_data()
    analytic = exp_term(y)
    numeric = pysolvegn.Term.from_rJ(analytic.residual_func, finite_difference=method)
    p = numpy.array([2.0, 0.4])
    tolerance = 1e-5 if method == "central" else 1e-3
    numpy.testing.assert_allclose(numeric.jacobian_func(p), analytic.jacobian_func(p), rtol=tolerance, atol=tolerance)


def test_term_evaluation_robust_gradient():
    # g = sum_j rho'(r_j^2) J_j^T r_j and the cost is 0.5 * sum rho(r_j^2)
    y = exp_data(n_outliers=5)
    term = exp_term(y, loss="cauchy", loss_scale=0.5, weight=3.0)
    p = numpy.array([2.0, 0.4])
    state = _evaluate_system([term], p, None, compute_cost=True)
    r = term.residual_func(p)
    J = term.jacobian_func(p)
    rho, rho_prime, _ = term.loss_func(r**2)
    numpy.testing.assert_allclose(state.second_term, 3.0 * J.T @ (rho_prime * r))
    numpy.testing.assert_allclose(state.cost, 3.0 * 0.5 * numpy.sum(rho))
    numpy.testing.assert_allclose(state.costs[0], 0.5 * numpy.sum(rho))


def test_term_gradient_is_cost_gradient():
    # The second term g is the exact gradient of the cost (for any loss)
    y = exp_data(n_outliers=5)
    p = numpy.array([2.0, 0.4])
    for loss in ["linear", "soft_l1", "huber", "cauchy", "arctan", "tukey"]:
        term = exp_term(y, loss=loss, loss_scale=2.0)
        cost = lambda q: _evaluate_system([term], q, None, compute_cost=True).cost
        g = _evaluate_system([term], p, None, compute_cost=True).second_term
        numpy.testing.assert_allclose(numerical_jacobian(cost, p), g, rtol=1e-5, atol=1e-6)


# ----------------------------------------------------------------------
# BatchTerm
# ----------------------------------------------------------------------

br_func = lambda p, I: p - 1.0
bJ_func = lambda p, I: numpy.broadcast_to(numpy.eye(p.shape[1]), (p.shape[0], p.shape[1], p.shape[1])).copy()


def test_batch_term_scalar_weight():
    term = pysolvegn.BatchTerm.from_rJ(br_func, bJ_func, weight=2.0)
    assert not term.is_vector_weight
    assert term.n_problems is None
    numpy.testing.assert_allclose(term.weight_at([0, 5, 2]), [2.0, 2.0, 2.0])


def test_batch_term_vector_weight():
    term = pysolvegn.BatchTerm.from_rJ(br_func, bJ_func, weight=[1.0, 2.0, 3.0])
    assert term.is_vector_weight
    assert term.n_problems == 3
    numpy.testing.assert_allclose(term.weight_at([2, 0]), [3.0, 1.0])
    with pytest.raises(ValueError):
        term.weight[0] = 5.0  # read-only


@pytest.mark.parametrize(
    "bad",
    [0.0, -1.0, numpy.nan, [1.0, 0.0], [1.0, -2.0], [1.0, numpy.inf], [[1.0, 2.0]], []],
)
def test_batch_term_invalid_weight(bad):
    with pytest.raises(ValueError):
        pysolvegn.BatchTerm.from_rJ(br_func, bJ_func, weight=bad)


def test_batch_term_gH_with_loss():
    with pytest.raises(ValueError):
        pysolvegn.BatchTerm(gradient_func=br_func, hessian_func=bJ_func, loss="cauchy")


@pytest.mark.parametrize("loss", ["linear", "huber", "tukey", "cauchy"])
def test_batch_term_equals_single_terms(loss):
    # Evaluating K problems at once gives the same systems as K single terms
    rng = numpy.random.default_rng(0)
    K = 5
    Y = numpy.stack([exp_data(seed=k, n_outliers=3) for k in range(K)])
    weights = rng.uniform(0.5, 2.0, K)
    P = numpy.column_stack((rng.uniform(1.5, 3.0, K), rng.uniform(0.3, 0.6, K)))

    def residual_func(p, indices):
        return p[:, :1] * numpy.exp(p[:, 1:2] * X_EXP) - Y[indices]

    def jacobian_func(p, indices):
        e = numpy.exp(p[:, 1:2] * X_EXP)
        return numpy.stack((e, p[:, :1] * X_EXP * e), axis=2)

    batch_term = pysolvegn.BatchTerm.from_rJ(residual_func, jacobian_func, weight=weights, loss=loss, loss_scale=0.7)
    indices = numpy.array([3, 0, 4])
    state = _evaluate_batch_system([batch_term], P[indices], indices, None, compute_cost=True)

    for a, k in enumerate(indices):
        term = exp_term(Y[k], weight=weights[k], loss=loss, loss_scale=0.7)
        single = _evaluate_system([term], P[k], None, compute_cost=True)
        numpy.testing.assert_allclose(state.hessian[a], single.hessian)
        numpy.testing.assert_allclose(state.second_term[a], single.second_term)
        numpy.testing.assert_allclose(state.cost[a], single.cost)


@pytest.mark.parametrize("method", ["central", "forward", "backward"])
def test_batch_numerical_jacobian(method):
    residual_func = lambda p, indices: numpy.column_stack((p[:, 0] ** 2 * (indices + 1), numpy.sin(p[:, 1])))
    jacobian = pysolvegn.build_batch_numerical_jacobian(residual_func, method=method)
    p = numpy.array([[1.0, 0.5], [2.0, -0.3]])
    indices = numpy.array([0, 3])
    expected = numpy.zeros((2, 2, 2))
    expected[:, 0, 0] = 2 * p[:, 0] * (indices + 1)
    expected[:, 1, 1] = numpy.cos(p[:, 1])
    tolerance = 1e-5 if method == "central" else 1e-3
    numpy.testing.assert_allclose(jacobian(p, indices), expected, atol=tolerance)
