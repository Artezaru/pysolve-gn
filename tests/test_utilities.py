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
