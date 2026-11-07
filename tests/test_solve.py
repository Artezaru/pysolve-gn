"""
Tests of the single problem solver ``pysolvegn.solve``.
"""

import numpy
import pytest
import scipy.optimize
import scipy.sparse

import pysolvegn

from problems import TRUE_EXP, X_EXP, exp_data, exp_model, exp_term

P0 = numpy.array([1.5, 0.3])


def criteria(**kwargs):
    defaults = dict(max_iteration=100, ftol=1e-12, xtol=1e-12, gtol=1e-10)
    defaults.update(kwargs)
    return defaults


# ----------------------------------------------------------------------
# Accuracy
# ----------------------------------------------------------------------


def test_linear_least_squares_matches_lstsq():
    rng = numpy.random.default_rng(0)
    A = rng.normal(size=(30, 4))
    y = rng.normal(size=30)
    term = pysolvegn.Term.from_rJ(lambda p: A @ p - y, lambda p: A)
    result = pysolvegn.solve(term, numpy.zeros(4), **criteria())
    numpy.testing.assert_allclose(result.parameters, numpy.linalg.lstsq(A, y, rcond=None)[0], atol=1e-10)
    assert result.success


@pytest.mark.parametrize("damping", [None, "lm", "lm-diag"])
def test_nonlinear_matches_scipy(damping):
    y = exp_data()
    result = pysolvegn.solve(exp_term(y), P0, damping=damping, **criteria())
    reference = scipy.optimize.least_squares(lambda p: exp_model(p) - y, P0, xtol=1e-15, ftol=1e-15, gtol=1e-15)
    assert result.success
    numpy.testing.assert_allclose(result.parameters, reference.x, rtol=1e-7)


@pytest.mark.parametrize("loss", ["soft_l1", "huber", "cauchy", "arctan"])
def test_robust_loss_matches_scipy(loss):
    # As usual with robust losses, start from the linear least squares solution
    y = exp_data(n_outliers=8)
    p_start = pysolvegn.solve(exp_term(y), P0, **criteria()).parameters
    result = pysolvegn.solve(exp_term(y, loss=loss, loss_scale=0.3), p_start, damping="lm-diag", **criteria(max_iteration=300))
    reference = scipy.optimize.least_squares(
        lambda p: exp_model(p) - y, p_start, loss=loss, f_scale=0.3, xtol=1e-15, ftol=1e-15, gtol=1e-15
    )
    assert result.success
    numpy.testing.assert_allclose(result.parameters, reference.x, rtol=1e-5)


def test_robust_loss_rejects_outliers():
    y = exp_data(n_outliers=8)
    linear = pysolvegn.solve(exp_term(y), P0, **criteria())
    robust = pysolvegn.solve(exp_term(y, loss="cauchy", loss_scale=0.3), linear.parameters, damping="lm-diag", **criteria(max_iteration=300))
    assert numpy.linalg.norm(robust.parameters - TRUE_EXP) < 0.2 * numpy.linalg.norm(linear.parameters - TRUE_EXP)


@pytest.mark.parametrize("loss", ["cauchy", "huber"])
@pytest.mark.parametrize("damping", [None, "lm-diag"])
def test_loss_scale_unit_invariance(loss, damping):
    # Same solution with residuals in m (loss_scale=C) and in mm (loss_scale=1000 C)
    y = exp_data(n_outliers=8)
    solutions = []
    for unit in (1.0, 1000.0):
        term = exp_term(y * unit, loss=loss, loss_scale=0.3 * unit)
        term_scaled = pysolvegn.Term.from_rJ(
            lambda p, unit=unit: exp_model(p) * unit - y * unit,
            lambda p, unit=unit: term.jacobian_func(p) * unit,
            loss=loss,
            loss_scale=0.3 * unit,
        )
        result = pysolvegn.solve(term_scaled, P0, damping=damping, **criteria(max_iteration=300, gtol=None))
        solutions.append(result.parameters)
    numpy.testing.assert_allclose(solutions[0], solutions[1], rtol=1e-6)


def test_parametrization():
    # Optimize log(a), log(b): p_out = exp(p_in)
    y = exp_data()
    parametrization = pysolvegn.build_positive_parametrization(2)
    result = pysolvegn.solve(exp_term(y), numpy.log(P0), parametrization, **criteria())
    reference = pysolvegn.solve(exp_term(y), P0, **criteria())
    assert result.success
    numpy.testing.assert_allclose(result.term_parameters, numpy.exp(result.parameters))
    numpy.testing.assert_allclose(result.term_parameters, reference.parameters, rtol=1e-7)


def test_sparse_jacobian_equals_dense():
    rng = numpy.random.default_rng(0)
    A = rng.normal(size=(30, 4))
    A[A < 0.3] = 0.0
    y = rng.normal(size=30)
    dense = pysolvegn.Term.from_rJ(lambda p: A @ p - y, lambda p: A, loss="huber")
    sparse = pysolvegn.Term.from_rJ(lambda p: A @ p - y, lambda p: scipy.sparse.csr_matrix(A), loss="huber")
    for damping in (None, "lm-diag"):
        r_dense = pysolvegn.solve(dense, numpy.zeros(4), damping=damping, **criteria())
        r_sparse = pysolvegn.solve(sparse, numpy.zeros(4), damping=damping, **criteria())
        numpy.testing.assert_allclose(r_sparse.parameters, r_dense.parameters, atol=1e-10)


def test_gH_term_equals_rJ_term():
    rng = numpy.random.default_rng(0)
    A = rng.normal(size=(30, 4))
    y = rng.normal(size=30)
    rJ = pysolvegn.Term.from_rJ(lambda p: A @ p - y, lambda p: A)
    gH = pysolvegn.Term.from_gH(
        lambda p: A.T @ (A @ p - y),
        lambda p: A.T @ A,
        cost_func=lambda p: 0.5 * numpy.sum((A @ p - y) ** 2),
    )
    for damping in (None, "lm"):
        r_rJ = pysolvegn.solve(rJ, numpy.zeros(4), damping=damping, **criteria())
        r_gH = pysolvegn.solve(gH, numpy.zeros(4), damping=damping, **criteria())
        numpy.testing.assert_allclose(r_gH.parameters, r_rJ.parameters, atol=1e-10)
        numpy.testing.assert_allclose(r_gH.cost, r_rJ.cost)


# ----------------------------------------------------------------------
# SolveResult and stopping criteria
# ----------------------------------------------------------------------


def test_solve_result_fields():
    y = exp_data()
    result = pysolvegn.solve(exp_term(y), P0, **criteria())
    assert isinstance(result, pysolvegn.SolveResult)
    assert result.parameters.shape == (2,)
    assert isinstance(result.success, bool) and result.success
    assert result.n_iterations > 0
    assert result.cost is not None and result.optimality is not None
    numpy.testing.assert_allclose(result.term_parameters, result.parameters)
    assert isinstance(result.stop_code, int) and result.stop_code > 0
    assert result.message == "\n".join(result.reasons)
    assert result.elapsed_time >= 0.0
    assert result.n_rejected == 0
    assert result.history == []


def test_p0_not_modified():
    p0 = P0.copy()
    pysolvegn.solve(exp_term(exp_data()), p0, **criteria())
    numpy.testing.assert_array_equal(p0, P0)


@pytest.mark.parametrize(
    "kwargs, tag",
    [
        (dict(ftol=1e-3), "[ftol]"),
        (dict(xtol=1e-3), "[xtol]"),
        (dict(gtol=1e-2), "[gtol]"),
        (dict(ptol=1e-3), "[ptol]"),
        (dict(atol=1e6), "[atol]"),
    ],
)
def test_convergence_criteria(kwargs, tag):
    result = pysolvegn.solve(exp_term(exp_data()), P0, max_iteration=100, **kwargs)
    assert result.success
    assert any(reason.startswith(tag) for reason in result.reasons)
    assert result.stopped_by(tag[1:-1])


def test_max_iteration_zero():
    result = pysolvegn.solve(exp_term(exp_data()), P0, max_iteration=0)
    assert not result.success
    assert result.n_iterations == 0
    numpy.testing.assert_array_equal(result.parameters, P0)
    assert result.stopped_by("max_iteration")
    assert result.reasons == ["[max_iteration] Maximum number of iterations reached: 0."]


def test_max_time():
    result = pysolvegn.solve(exp_term(exp_data()), P0, max_time=0.0)
    assert not result.success
    assert result.stopped_by("max_time")


def test_convergence_wins_over_max_iteration():
    # A convergence criterion satisfied at the same iteration as max_iteration -> success
    result = pysolvegn.solve(exp_term(exp_data()), P0, max_iteration=0, atol=1e6)
    assert result.success


def test_nan_in_p0():
    result = pysolvegn.solve(exp_term(exp_data()), numpy.array([numpy.nan, 0.3]), max_iteration=10)
    assert not result.success
    assert result.n_iterations == 0
    assert result.term_parameters is None and result.cost is None
    assert result.stopped_by("naninf_p0") and result.stopped_by("naninf")


@pytest.mark.filterwarnings("ignore:invalid value encountered in log:RuntimeWarning")
def test_nan_in_term_parameters():
    parametrization = pysolvegn.Parametrization(lambda p: numpy.log(p), lambda p: numpy.diag(1.0 / p))
    result = pysolvegn.solve(exp_term(exp_data()), numpy.array([-1.0, 0.3]), parametrization, max_iteration=10)
    assert not result.success
    assert result.stop_code == pysolvegn.implemented_conf._STOP_CODES["naninf_term_parameters"]


def test_singular_system():
    # The second parameter has no influence on the residuals
    term = pysolvegn.Term.from_rJ(lambda p: numpy.array([p[0] - 1.0]), lambda p: numpy.array([[1.0, 0.0]]))
    result = pysolvegn.solve(term, numpy.zeros(2), max_iteration=10)
    assert not result.success
    assert result.stopped_by("singular")


def test_singular_system_sparse():
    term = pysolvegn.Term.from_rJ(
        lambda p: numpy.array([p[0] - 1.0]), lambda p: scipy.sparse.csr_matrix([[1.0, 0.0]])
    )
    result = pysolvegn.solve(term, numpy.zeros(2), max_iteration=10)
    assert not result.success
    assert result.stopped_by("singular")


def test_singular_system_solved_by_levenberg_marquardt():
    term = pysolvegn.Term.from_rJ(lambda p: numpy.array([p[0] - 1.0]), lambda p: numpy.array([[1.0, 0.0]]))
    result = pysolvegn.solve(term, numpy.zeros(2), max_iteration=50, gtol=1e-10, damping="lm")
    assert result.success
    numpy.testing.assert_allclose(result.parameters[0], 1.0, atol=1e-8)


def test_levenberg_marquardt_never_increases_cost():
    y = exp_data(n_outliers=8)
    result = pysolvegn.solve(
        exp_term(y, loss="cauchy", loss_scale=0.3),
        numpy.array([0.1, 1.5]),
        damping="lm",
        history=True,
        history_details=["cost", "damping"],
        **criteria(max_iteration=300),
    )
    costs = numpy.array([h["cost"] for h in result.history])
    assert numpy.all(numpy.diff(costs) <= 1e-12)
    assert result.history[0]["damping"] is None
    assert all(h["damping"] > 0 for h in result.history[1:])


# ----------------------------------------------------------------------
# stop_code, reasons, config and cost
# ----------------------------------------------------------------------


def test_stop_code_several_criteria():
    # Linear problem with an exact solution: reached at the first step, where the
    # gradient and the cost are both ~0, so gtol and atol are triggered together
    rng = numpy.random.default_rng(0)
    A = rng.normal(size=(30, 4))
    y = A @ numpy.arange(4.0)
    term = pysolvegn.Term.from_rJ(lambda p: A @ p - y, lambda p: A)
    result = pysolvegn.solve(term, numpy.zeros(4), max_iteration=10, gtol=1e-8, xtol=1e-8, atol=1e-20)
    codes = pysolvegn.implemented_conf._STOP_CODES
    assert result.success
    assert result.n_iterations == 1
    assert result.stop_code == codes["gtol"] | codes["atol"]
    assert len(result.reasons) == bin(result.stop_code).count("1")


def test_reasons_contain_thresholds_not_values():
    result = pysolvegn.solve(exp_term(exp_data()), P0, max_iteration=100, gtol=1e-6)
    assert result.reasons == ["[gtol] Convergence achieved: ||g||_inf < gtol with gtol = 1e-06."]


def test_stopped_by_invalid_name():
    result = pysolvegn.solve(exp_term(exp_data()), P0, max_iteration=0)
    with pytest.raises(ValueError):
        result.stopped_by("unknown")


def test_failure_wins_over_convergence():
    # Callback returning False at the iteration where gtol is satisfied: success is False
    result = pysolvegn.solve(exp_term(exp_data()), P0, max_iteration=0, atol=1e6, callback_func=lambda state: False)
    assert result.stopped_by("atol") and result.stopped_by("callback")
    assert not result.success


def test_config():
    result = pysolvegn.solve(exp_term(exp_data()), P0, max_iteration=50, xtol=1e-9, damping="LM", lm_conf={"factor": 4.0})
    assert result.config["max_iteration"] == 50
    assert result.config["xtol"] == 1e-9
    assert result.config["ftol"] is None
    assert result.config["damping"] == "lm"
    assert result.config["lm_conf"] == {"initial_scale": 1e-3, "factor": 4.0, "max_rejections": 50, "diag_floor": 1e-12, "noise": 1e-13}


def test_cost_computed_at_convergence():
    # Only gtol: the cost is not needed during the optimization, but computed at the end
    y = exp_data()
    result = pysolvegn.solve(exp_term(y), P0, max_iteration=100, gtol=1e-8)
    assert result.success
    numpy.testing.assert_allclose(result.cost, 0.5 * numpy.sum((exp_model(result.parameters) - y) ** 2))


def test_cost_not_computed_without_success():
    result = pysolvegn.solve(exp_term(exp_data()), P0, max_iteration=2)
    assert not result.success
    assert result.cost is None


def test_lm_conf_max_rejections():
    # With a huge initial damping and a factor of 2, one rejection is never enough
    y = exp_data(n_outliers=8)
    term = exp_term(y, loss="cauchy", loss_scale=0.3)
    reference = pysolvegn.solve(term, numpy.array([0.1, 1.5]), damping="lm", **criteria(max_iteration=300))
    restricted = pysolvegn.solve(
        term, numpy.array([0.1, 1.5]), damping="lm", lm_conf={"initial_scale": 1e-12, "max_rejections": 1}, **criteria(max_iteration=300)
    )
    assert reference.success
    assert restricted.stopped_by("lm") and not restricted.success
    assert "after 1 rejections" in restricted.message


def test_lm_conf_changes_initial_damping():
    result = pysolvegn.solve(
        exp_term(exp_data()), P0, damping="lm-diag", lm_conf={"initial_scale": 1.0, "factor": 2.0},
        history=True, history_details=["damping"], **criteria(),
    )
    assert result.success
    assert result.history[1]["damping"] in (1.0, 2.0, 4.0, 8.0, 16.0)  # 1.0 * 2**k


@pytest.mark.parametrize(
    "lm_conf, error",
    [
        ({"unknown": 1.0}, ValueError),
        ({"factor": 1.0}, ValueError),
        ({"initial_scale": 0.0}, ValueError),
        ({"diag_floor": -1.0}, ValueError),
        ({"max_rejections": 0}, ValueError),
        ({"max_rejections": 1.5}, TypeError),
        ({"factor": "10"}, TypeError),
        ([("factor", 10.0)], TypeError),
    ],
)
def test_lm_conf_invalid(lm_conf, error):
    with pytest.raises(error):
        pysolvegn.solve(exp_term(exp_data()), P0, max_iteration=5, damping="lm", lm_conf=lm_conf)


# ----------------------------------------------------------------------
# Callback, update_func and history
# ----------------------------------------------------------------------


def test_callback_stop():
    calls = []

    def callback(state):
        calls.append(state)
        return len(calls) < 3

    result = pysolvegn.solve(exp_term(exp_data()), P0, callback_func=callback, **criteria())
    assert not result.success
    assert len(calls) == 3
    assert result.n_iterations == 2
    assert result.stopped_by("callback")
    assert set(calls[0]) == {"parameters", "delta_parameters", "cost", "second_term", "hessian"}
    assert calls[0]["delta_parameters"] is None


def test_callback_must_return_bool():
    with pytest.raises(ValueError):
        pysolvegn.solve(exp_term(exp_data()), P0, callback_func=lambda state: 1, **criteria())


@pytest.mark.parametrize("damping", [None, "lm"])
def test_update_func(damping):
    # Clip the steps: the applied updates are stored in the history
    result = pysolvegn.solve(
        exp_term(exp_data()),
        P0,
        update_func=lambda p, dp: numpy.clip(dp, -0.05, 0.05),
        damping=damping,
        history=True,
        history_details=["delta_parameters"],
        **criteria(max_iteration=300),
    )
    assert result.success
    steps = [h["delta_parameters"] for h in result.history[1:]]
    assert max(numpy.max(numpy.abs(step)) for step in steps) <= 0.05 + 1e-15
    numpy.testing.assert_allclose(result.parameters, TRUE_EXP, atol=0.1)


def test_update_func_wrong_shape():
    with pytest.raises(ValueError):
        pysolvegn.solve(exp_term(exp_data()), P0, update_func=lambda p, dp: dp[:1], **criteria())


def test_history_all():
    result = pysolvegn.solve(exp_term(exp_data()), P0, history=True, history_details="all", **criteria())
    assert len(result.history) == result.n_iterations + 1
    entry = result.history[-1]
    assert entry["iteration"] == result.n_iterations
    numpy.testing.assert_allclose(entry["parameters"], result.parameters)
    numpy.testing.assert_allclose(entry["cost"], result.cost)
    assert entry["hessian"].shape == (2, 2)
    assert entry["jacobians"][0].shape == (X_EXP.size, 2)
    assert result.history[0]["delta_parameters"] is None
    assert result.history[0]["delta_cost"] is None


@pytest.mark.parametrize("length", [0, 2, -2])
def test_history_length(length):
    result = pysolvegn.solve(
        exp_term(exp_data()), P0, history=True, history_details=["iteration"], history_length=length, **criteria()
    )
    iterations = [h["iteration"] for h in result.history]
    if length == 0:
        assert iterations == []
    elif length > 0:
        assert iterations == [0, 1]
    else:
        assert iterations == [result.n_iterations - 1, result.n_iterations]


def test_history_entries_are_copies():
    result = pysolvegn.solve(exp_term(exp_data()), P0, history=True, history_details=["parameters"], **criteria())
    assert not numpy.shares_memory(result.history[0]["parameters"], result.history[1]["parameters"])
    numpy.testing.assert_array_equal(result.history[0]["parameters"], P0)


# ----------------------------------------------------------------------
# Invalid arguments
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, error",
    [
        (dict(), ValueError),  # no stopping criterion
        (dict(max_iteration=-1), ValueError),
        (dict(max_iteration=1.5), TypeError),
        (dict(ftol=0.0, max_iteration=5), ValueError),
        (dict(xtol=-1.0, max_iteration=5), ValueError),
        (dict(max_iteration=5, damping="unknown"), ValueError),
        (dict(max_iteration=5, verbosity=4), ValueError),
        (dict(max_iteration=5, history=True, history_details=["unknown"]), ValueError),
        (dict(max_iteration=5, callback_func=1), TypeError),
        (dict(max_iteration=5, update_func=1), TypeError),
    ],
)
def test_invalid_arguments(kwargs, error):
    with pytest.raises(error):
        pysolvegn.solve(exp_term(exp_data()), P0, **kwargs)


def test_invalid_terms_and_p0():
    with pytest.raises(ValueError):
        pysolvegn.solve([], P0, max_iteration=5)
    with pytest.raises(TypeError):
        pysolvegn.solve(["not a term"], P0, max_iteration=5)
    with pytest.raises(ValueError):
        pysolvegn.solve(exp_term(exp_data()), numpy.zeros((2, 2)), max_iteration=5)


def test_levenberg_marquardt_requires_gH_cost():
    gH = pysolvegn.Term.from_gH(lambda p: p, lambda p: numpy.eye(2))
    with pytest.warns(UserWarning, match="no cost_func"), pytest.raises(ValueError):
        pysolvegn.solve(gH, P0, max_iteration=5, damping="lm")


def test_wrong_jacobian_shape():
    term = pysolvegn.Term.from_rJ(lambda p: p - 1.0, lambda p: numpy.eye(3))
    with pytest.raises(ValueError):
        pysolvegn.solve(term, numpy.zeros(2), max_iteration=5)

# ----------------------------------------------------------------------
# 'gH' terms without cost_func: warning, and atol / ftol forbidden
# ----------------------------------------------------------------------
def _gH_without_cost():
    return pysolvegn.Term.from_gH(lambda p: p - 1.0, lambda p: numpy.eye(2))


def test_gH_without_cost_func_warns():
    with pytest.warns(UserWarning, match=r"'gH' term\(s\) \[0\] have no cost_func"):
        result = pysolvegn.solve(_gH_without_cost(), numpy.zeros(2), gtol=1e-10, max_iteration=10)
    assert result.success
    numpy.testing.assert_allclose(result.parameters, 1.0)


@pytest.mark.parametrize(
    "criteria, names",
    [
        ({"atol": 1e-8}, "atol"),
        ({"ftol": 1e-8}, "ftol"),
        ({"atol": 1e-8, "ftol": 1e-8, "max_iteration": 10}, "atol and ftol"),
        ({"ftol": 1e-8, "gtol": 1e-8}, "ftol"),
    ],
)
def test_gH_without_cost_func_forbids_cost_criteria(criteria, names):
    with pytest.warns(UserWarning), pytest.raises(ValueError, match=f"^{names} cannot be used"):
        pysolvegn.solve(_gH_without_cost(), numpy.zeros(2), **criteria)


def test_gH_without_cost_func_mixed_with_rJ_term():
    # A 'gH' regularization without cost_func next to a 'rJ' term: same rules (term index 1)
    rJ = pysolvegn.Term.from_rJ(lambda p: p - 2.0, lambda p: numpy.eye(2))
    with pytest.warns(UserWarning, match=r"\[1\]"), pytest.raises(ValueError):
        pysolvegn.solve([rJ, _gH_without_cost()], numpy.zeros(2), atol=1e-8)
    with pytest.warns(UserWarning, match=r"\[1\]"):
        result = pysolvegn.solve([rJ, _gH_without_cost()], numpy.zeros(2), xtol=1e-10, max_iteration=10)
    numpy.testing.assert_allclose(result.parameters, 1.5)


def test_gH_with_cost_func_does_not_warn():
    import warnings

    gH = pysolvegn.Term.from_gH(
        lambda p: p - 1.0, lambda p: numpy.eye(2), cost_func=lambda p: 0.5 * numpy.sum((p - 1.0) ** 2)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = pysolvegn.solve(gH, numpy.zeros(2), atol=1e-12, ftol=1e-10, max_iteration=10)
    assert result.success


# ----------------------------------------------------------------------
# Optional Armijo line search (line_search=True)
# ----------------------------------------------------------------------
def _atan_term():
    # Gauss-Newton on r(p) = arctan(p) diverges for |p0| > 1.39 (overshoot), converges to 0
    # with the line search
    return pysolvegn.Term.from_rJ(lambda p: numpy.arctan(p), lambda p: numpy.diag(1.0 / (1.0 + p**2)))


def test_line_search_disabled_by_default():
    result = pysolvegn.solve(_atan_term(), numpy.array([2.0]), max_iteration=6, history=True, history_details="all")
    assert result.config["line_search"] is False
    assert all(h["step_length"] is None for h in result.history)
    assert numpy.abs(result.parameters[0]) > 100  # pure Gauss-Newton: the full step is always applied


def test_line_search_fixes_overshoot():
    result = pysolvegn.solve(
        _atan_term(), numpy.array([2.0]), line_search=True, max_iteration=50, gtol=1e-12,
        history=True, history_details=["cost", "step_length"],
    )
    assert result.success and result.stopped_by("gtol")
    numpy.testing.assert_allclose(result.parameters, 0.0, atol=1e-10)
    costs = [h["cost"] for h in result.history]
    assert all(b <= a for a, b in zip(costs, costs[1:]))  # monotone decrease
    lengths = [h["step_length"] for h in result.history[1:]]
    assert lengths[0] < 1.0 and lengths[-1] == 1.0  # shortened far from the solution, full step near it
    assert result.n_rejected > 0


def test_line_search_full_steps_give_the_pure_gauss_newton_result():
    # Close to the solution every full step satisfies the Armijo condition: same iterates
    term = exp_term(exp_data())
    p0 = numpy.array([2.4, 0.5])
    gn = pysolvegn.solve(term, p0, **criteria())
    ls = pysolvegn.solve(term, p0, line_search=True, **criteria())
    numpy.testing.assert_array_equal(ls.parameters, gn.parameters)
    assert ls.n_iterations == gn.n_iterations and ls.n_rejected == 0


def test_line_search_at_machine_precision():
    # After convergence (no convergence criterion): the steps at the rounding noise level
    # are accepted at once (no vain backtracking) and no false "not_descent" stop
    term = exp_term(exp_data())
    result = pysolvegn.solve(term, numpy.array([2.4, 0.5]), line_search=True, max_iteration=40)
    assert result.stopped_by("max_iteration") and result.n_rejected == 0
    A = numpy.random.default_rng(0).normal(size=(50, 3))
    b = A @ numpy.array([1.0, 2.0, 3.0])
    linear = pysolvegn.Term.from_rJ(lambda p: A @ p - b, lambda p: A)
    result = pysolvegn.solve(linear, numpy.zeros(3), line_search=True, max_iteration=10)
    assert result.stopped_by("max_iteration") and not result.stopped_by("not_descent")


def test_line_search_rejects_nan_trial_steps():
    # r(p) = log(p): the full step from p = 3 gives p < 0 (NaN residual), the line search
    # rejects it and converges to 1
    term = pysolvegn.Term.from_rJ(lambda p: numpy.log(p), lambda p: numpy.diag(1.0 / p))
    with numpy.errstate(invalid="ignore"):
        gn = pysolvegn.solve(term, numpy.array([3.0]), max_iteration=10, gtol=1e-12)
        ls = pysolvegn.solve(term, numpy.array([3.0]), line_search=True, max_iteration=50, gtol=1e-12)
    assert not gn.success
    assert ls.success
    numpy.testing.assert_allclose(ls.parameters, 1.0, atol=1e-10)


def test_line_search_not_descent():
    # update_func reversing the step: g^T Δp* > 0
    result = pysolvegn.solve(
        exp_term(exp_data()), P0, line_search=True, update_func=lambda p, dp: -dp, **criteria()
    )
    assert not result.success
    assert result.stopped_by("not_descent")
    assert "[not_descent]" in result.message
    numpy.testing.assert_array_equal(result.parameters, P0)  # the update is never applied


def test_line_search_failure():
    # Linear residuals: C(p + α Δp) = C + (α - α²/2) g^T Δp, the condition needs
    # α <= 2 (1 - c1) = 0.002 -> impossible without backtracking
    A = numpy.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    term = pysolvegn.Term.from_rJ(lambda p: A @ p - 1.0, lambda p: A)
    failed = pysolvegn.solve(
        term, numpy.zeros(2), line_search=True, line_search_conf={"c1": 0.999, "max_backtracks": 0}, max_iteration=5
    )
    assert failed.stopped_by("line_search") and not failed.success
    assert "after 0 backtracks" in failed.message
    passed = pysolvegn.solve(
        term, numpy.zeros(2), line_search=True, line_search_conf={"c1": 0.999}, max_iteration=1
    )
    assert passed.n_iterations == 1 and passed.n_rejected == 9  # α = 0.5^9 ~ 0.00195
    assert passed.config["line_search_conf"] == {"c1": 0.999, "factor": 0.5, "max_backtracks": 30, "noise": 1e-13}


def test_line_search_with_gH_terms():
    A = numpy.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    gH = pysolvegn.Term.from_gH(
        lambda p: A.T @ (A @ p - 1.0), lambda p: A.T @ A, cost_func=lambda p: 0.5 * numpy.sum((A @ p - 1.0) ** 2)
    )
    result = pysolvegn.solve(gH, numpy.zeros(2), line_search=True, max_iteration=5, gtol=1e-12)
    assert result.success
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="line_search=True requires"):
        pysolvegn.solve(_gH_without_cost(), numpy.zeros(2), line_search=True, max_iteration=5)


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"line_search": True, "damping": "lm"}, ValueError),
        ({"line_search": 1}, TypeError),
        ({"line_search": True, "line_search_conf": {"c1": 0.0}}, ValueError),
        ({"line_search": True, "line_search_conf": {"factor": 1.0}}, ValueError),
        ({"line_search": True, "line_search_conf": {"max_backtracks": -1}}, ValueError),
        ({"line_search": True, "line_search_conf": {"max_backtracks": 2.5}}, TypeError),
        ({"line_search": True, "line_search_conf": {"unknown": 1}}, ValueError),
        ({"line_search": True, "line_search_conf": [1]}, TypeError),
        ({"line_search": True, "line_search_conf": {"noise": -1e-13}}, ValueError),
        ({"line_search": True, "line_search_conf": {"noise": numpy.inf}}, ValueError),
    ],
)
def test_line_search_invalid_options(kwargs, error):
    with pytest.raises(error):
        pysolvegn.solve(exp_term(exp_data()), P0, max_iteration=5, **kwargs)


def test_line_search_verbosity(capsys):
    pysolvegn.solve(_atan_term(), numpy.array([2.0]), line_search=True, max_iteration=3, verbosity=3)
    assert "α" in capsys.readouterr().out


# ----------------------------------------------------------------------
# Levenberg-Marquardt: max_time between trials, rounding noise, λ floor, gain ratio
# ----------------------------------------------------------------------
def _slow_term():
    # Each residual evaluation takes 2 ms; the minimum is at p = 0
    import time

    def residual(p):
        time.sleep(0.002)
        return numpy.array([p[0] ** 2 + 1.0])

    return pysolvegn.Term.from_rJ(residual, lambda p: numpy.array([[2.0 * p[0]]]))


@pytest.mark.parametrize(
    "options, max_trials",
    [
        ({"damping": "lm", "update_func": lambda p, dp: numpy.ones(1)}, 50),  # every trial increases the cost
        ({"line_search": True, "update_func": lambda p, dp: 1e4 * dp}, 30),  # every trial overshoots
    ],
)
def test_max_time_checked_between_trial_steps(options, max_trials):
    result = pysolvegn.solve(_slow_term(), numpy.array([1.0]), max_time=0.01, max_iteration=100, **options)
    assert result.stopped_by("max_time") and not result.success
    assert result.n_iterations == 0
    numpy.testing.assert_array_equal(result.parameters, [1.0])  # the trial step is not applied
    assert 0 < result.n_rejected < max_trials  # stopped long before exhausting the trials


def test_lm_rounding_noise():
    # Without convergence criterion, after convergence: the steps at the rounding noise
    # level are accepted (strict condition: rejections and λ drifting up)
    term = exp_term(exp_data())
    p0 = numpy.array([2.4, 0.5])
    for damping in ("lm", "lm-diag"):
        default = pysolvegn.solve(term, p0, damping=damping, max_iteration=60)
        strict = pysolvegn.solve(term, p0, damping=damping, max_iteration=60, lm_conf={"noise": 0.0})
        assert default.n_rejected == 0
        assert strict.n_rejected > 0
        numpy.testing.assert_allclose(default.parameters, strict.parameters, rtol=1e-8)


def test_lm_lambda_floor():
    # Accepted steps divide λ by 10: λ stays above its floor (never 0)
    result = pysolvegn.solve(
        exp_term(exp_data()), numpy.array([2.4, 0.5]), damping="lm", max_iteration=400,
        history=True, history_details=["damping"],
    )
    lambdas = numpy.array([h["damping"] for h in result.history[1:]])
    assert numpy.all(lambdas > 0)
    assert lambdas[-1] == lambdas[-2]  # floor reached


def test_lm_gain_ratio():
    # Linear residuals and linear loss: the Gauss-Newton model is exact, the gain ratio of
    # a step with a significant decrease is 1
    A = numpy.random.default_rng(0).normal(size=(30, 3))
    b = A @ numpy.array([1.0, 2.0, 3.0]) + 0.1 * numpy.random.default_rng(1).normal(size=30)
    term = pysolvegn.Term.from_rJ(lambda p: A @ p - b, lambda p: A)
    result = pysolvegn.solve(
        term, numpy.zeros(3), damping="lm", max_iteration=3, history=True, history_details=["gain_ratio"]
    )
    assert result.history[0]["gain_ratio"] is None
    numpy.testing.assert_allclose(result.history[1]["gain_ratio"], 1.0, rtol=1e-8)
    # Without damping: no gain ratio
    result = pysolvegn.solve(term, numpy.zeros(3), max_iteration=3, history=True, history_details="all")
    assert all(h["gain_ratio"] is None for h in result.history)


@pytest.mark.parametrize("noise, error", [(-1e-13, ValueError), (numpy.inf, ValueError), (True, TypeError), ("1", TypeError)])
def test_lm_conf_invalid_noise(noise, error):
    with pytest.raises(error):
        pysolvegn.solve(exp_term(exp_data()), P0, max_iteration=5, damping="lm", lm_conf={"noise": noise})