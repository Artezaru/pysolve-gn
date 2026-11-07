"""
Tests of the utilities: numerical Jacobians, study_optimization and the L-curve analysis.
"""

import matplotlib

matplotlib.use("Agg")

import numpy
import pytest

import pysolvegn

from problems import exp_data, exp_term


@pytest.mark.parametrize("method, tolerance", [("central", 1e-7), ("forward", 1e-5), ("backward", 1e-5)])
def test_build_numerical_jacobian(method, tolerance):
    func = lambda p: numpy.array([p[0] ** 2, numpy.sin(p[1]) * p[0], numpy.exp(p[1])])
    jacobian = pysolvegn.build_numerical_jacobian(func, method=method)
    p = numpy.array([1.3, -0.4])
    expected = numpy.array(
        [[2 * p[0], 0.0], [numpy.sin(p[1]), numpy.cos(p[1]) * p[0]], [0.0, numpy.exp(p[1])]]
    )
    numpy.testing.assert_allclose(jacobian(p), expected, atol=tolerance)


def test_build_numerical_jacobian_invalid_method():
    with pytest.raises(ValueError):
        pysolvegn.build_numerical_jacobian(lambda p: p, method="unknown")


def test_study_optimization_runs():
    term = exp_term(exp_data())
    pysolvegn.study_optimization(term, numpy.array([2.5, 0.5]), title="test")
    pysolvegn.study_optimization([term], numpy.array([2.5, 0.5]), pysolvegn.build_positive_parametrization(2))


def test_study_optimization_nan_p0():
    with pytest.raises(RuntimeError):
        pysolvegn.study_optimization([exp_term(exp_data())], numpy.array([numpy.nan, 0.5]))


def test_Lcurve_analysis_runs_and_restores_weight():
    data_term = exp_term(exp_data(noise=1.0))
    reg_term = pysolvegn.build_squared_regularization([2.0, 0.6], [0.5, 0.2], weight=3.0)
    pysolvegn.perform_Lcurve_analysis(
        data_term,
        reg_term,
        numpy.array([2.0, 0.6]),
        numpy.logspace(-2, 3, 8),
        optimal=True,
        max_iteration=20,
        ftol=1e-8,
    )
    assert reg_term.weight == 3.0


def test_Lcurve_analysis_rejects_gH_terms():
    gH = pysolvegn.Term.from_gH(lambda p: p, lambda p: numpy.eye(2))
    with pytest.raises(ValueError):
        pysolvegn.perform_Lcurve_analysis(gH, gH, numpy.zeros(2), [1.0], max_iteration=5)


def test_public_api():
    names = [
        "solve", "SolveResult", "Term", "Parametrization",
        "solve_batch", "BatchSolveResult", "BatchTerm", "BatchParametrization",
        "build_numerical_jacobian", "build_batch_numerical_jacobian",
        "build_affine_parametrization", "build_fixed_parametrization",
        "build_sigmoid_parametrization", "build_positive_parametrization",
        "build_batch_affine_parametrization", "build_batch_fixed_parametrization",
        "build_batch_sigmoid_parametrization", "build_batch_positive_parametrization",
        "build_squared_regularization", "build_soft_squared_regularization",
        "build_absolute_regularization", "build_batch_squared_regularization",
        "build_batch_soft_squared_regularization", "build_batch_absolute_regularization",
        "linear_rho", "soft_l1_rho", "huber_rho", "cauchy_rho", "arctan_rho", "tukey_rho",
        "scale_rho_function", "get_rho_function_by_name",
        "study_optimization", "perform_Lcurve_analysis",
    ]
    missing = [name for name in names if not hasattr(pysolvegn, name)]
    assert missing == []


# ----------------------------------------------------------------------------
# Finite differences: default steps, number of calls, custom epsilon
# ----------------------------------------------------------------------------
def _fd_problem():
    func = lambda p: numpy.array([p[0] ** 2, numpy.sin(p[1]) * p[0], numpy.exp(p[1]), p[0] * p[1] ** 3])
    p = numpy.array([1.3, -0.4])
    expected = numpy.array(
        [
            [2 * p[0], 0.0],
            [numpy.sin(p[1]), numpy.cos(p[1]) * p[0]],
            [0.0, numpy.exp(p[1])],
            [p[1] ** 3, 3 * p[0] * p[1] ** 2],
        ]
    )
    return func, p, expected


def test_numerical_jacobian_central_default_step_is_accurate():
    # The default central step (eps^(1/3) ~ 6e-6) gives an error ~ 1e-10 (vs ~ 1e-8 with 1e-8)
    func, p, expected = _fd_problem()
    jacobian = pysolvegn.build_numerical_jacobian(func, method="central")
    numpy.testing.assert_allclose(jacobian(p), expected, atol=1e-9)


@pytest.mark.parametrize("method, expected_calls", [("central", 4), ("forward", 3), ("backward", 3)])
def test_numerical_jacobian_number_of_calls(method, expected_calls):
    # forward/backward: n + 1 calls (the residuals at p are computed once), central: 2 n calls
    func, p, _ = _fd_problem()
    calls = []
    jacobian = pysolvegn.build_numerical_jacobian(lambda q: calls.append(1) or func(q), method=method)
    jacobian(p)
    assert len(calls) == expected_calls


@pytest.mark.parametrize("method, expected_calls", [("central", 4), ("forward", 3), ("backward", 3)])
def test_batch_numerical_jacobian_number_of_calls(method, expected_calls):
    calls = []

    def residual_func(P, indices):
        calls.append(1)
        return numpy.stack([P[:, 0] ** 2, P[:, 0] * P[:, 1]], axis=1)

    jacobian = pysolvegn.build_batch_numerical_jacobian(residual_func, method=method)
    P = numpy.array([[1.0, 2.0], [-3.0, 0.5], [0.1, 4.0]])
    J = jacobian(P, numpy.arange(3))
    assert len(calls) == expected_calls
    expected = numpy.zeros((3, 2, 2))
    expected[:, 0, 0] = 2 * P[:, 0]
    expected[:, 1, 0] = P[:, 1]
    expected[:, 1, 1] = P[:, 0]
    numpy.testing.assert_allclose(J, expected, atol=1e-6)


def test_numerical_jacobian_does_not_modify_parameters():
    func, p, _ = _fd_problem()
    p_copy = p.copy()
    for method in ("central", "forward", "backward"):
        pysolvegn.build_numerical_jacobian(func, method=method)(p)
    numpy.testing.assert_array_equal(p, p_copy)


def test_numerical_jacobian_custom_epsilon():
    # A large step gives a large truncation error: the custom value is really used
    func, p, expected = _fd_problem()
    coarse = pysolvegn.build_numerical_jacobian(func, method="forward", epsilon=1e-1)(p)
    assert numpy.abs(coarse - expected).max() > 1e-3
    fine = pysolvegn.build_numerical_jacobian(func, method="forward", epsilon=1e-7)(p)
    numpy.testing.assert_allclose(fine, expected, atol=1e-6)


@pytest.mark.parametrize("epsilon", [0.0, -1e-6, numpy.inf, numpy.nan, True, "1e-6"])
def test_numerical_jacobian_invalid_epsilon(epsilon):
    with pytest.raises((TypeError, ValueError)):
        pysolvegn.build_numerical_jacobian(lambda p: p, epsilon=epsilon)
    with pytest.raises((TypeError, ValueError)):
        pysolvegn.build_batch_numerical_jacobian(lambda p, i: p, epsilon=epsilon)


def test_finite_difference_epsilon_is_forwarded():
    # The finite_difference_epsilon of Term / BatchTerm / Parametrization / BatchParametrization
    # is the step used by the numerical Jacobian (a coarse step gives a visible error)
    func = lambda p: p ** 2
    p = numpy.array([1.0, 2.0])
    exact = numpy.diag(2 * p)

    term = pysolvegn.Term.from_rJ(func, finite_difference="forward", finite_difference_epsilon=0.1)
    numpy.testing.assert_allclose(term.J_func(p), exact + numpy.diag(0.1 * numpy.maximum(1, p)), atol=1e-12)

    parametrization = pysolvegn.Parametrization(func, finite_difference="forward", finite_difference_epsilon=0.1)
    numpy.testing.assert_allclose(parametrization.J_func(p), term.J_func(p), atol=1e-12)

    batch_term = pysolvegn.BatchTerm.from_rJ(
        lambda P, i: P ** 2, finite_difference="forward", finite_difference_epsilon=0.1
    )
    numpy.testing.assert_allclose(batch_term.J_func(p[None], numpy.arange(1))[0], term.J_func(p), atol=1e-12)

    batch_parametrization = pysolvegn.BatchParametrization(
        lambda P: P ** 2, finite_difference="forward", finite_difference_epsilon=0.1
    )
    numpy.testing.assert_allclose(batch_parametrization.J_func(p[None])[0], term.J_func(p), atol=1e-12)


def test_finite_difference_epsilon_requires_finite_difference():
    J = lambda p: numpy.eye(2)
    with pytest.raises(ValueError):
        pysolvegn.Term.from_rJ(lambda p: p, J, finite_difference_epsilon=1e-6)
    with pytest.raises(ValueError):
        pysolvegn.Parametrization(lambda p: p, J, finite_difference_epsilon=1e-6)
    with pytest.raises(ValueError):
        pysolvegn.BatchTerm.from_rJ(lambda P, i: P, lambda P, i: P, finite_difference_epsilon=1e-6)
    with pytest.raises(ValueError):
        pysolvegn.BatchParametrization(lambda P: P, lambda P: P, finite_difference_epsilon=1e-6)