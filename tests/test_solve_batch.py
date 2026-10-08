"""
Tests of the batch solver ``pysolvegn.solve_batch``.

The main property: each problem of the batch gives exactly the same result as if it
was solved alone with ``pysolvegn.solve``.
"""

import warnings

import numpy
import pytest

import pysolvegn

from problems import (
    X_LINE,
    line_batch_data,
    line_batch_jacobian,
    line_batch_residual,
    line_single_term,
)

K = 12
TRUTH, Y = line_batch_data(K=K, outlier_fraction=0.15)
P0 = numpy.tile([1.5, 0.3], (K, 1))


def criteria(**kwargs):
    defaults = dict(max_iteration=60, ftol=1e-10, xtol=1e-10, gtol=1e-12)
    defaults.update(kwargs)
    return defaults


def batch_term(**kwargs):
    return pysolvegn.BatchTerm.from_rJ(line_batch_residual(Y), line_batch_jacobian, **kwargs)


# ----------------------------------------------------------------------
# Equivalence with the single solver
# ----------------------------------------------------------------------

CONFIGS = [
    dict(loss="linear", damping=None, param=False, update=False),
    dict(loss="cauchy", damping=None, param=False, update=False),
    dict(loss="huber", damping=None, param=True, update=False),
    dict(loss="cauchy", damping="lm", param=True, update=False),
    dict(loss="tukey", damping="lm-diag", param=False, update=False),
    dict(loss="soft_l1", damping="lm-diag", param=True, update=True),
    dict(loss="arctan", damping=None, param=False, update=True),
]


@pytest.mark.parametrize("config", CONFIGS)
def test_batch_equals_single(config):
    rng = numpy.random.default_rng(1)
    weights = rng.uniform(0.5, 2.0, K)
    term_kwargs = dict(loss=config["loss"], loss_scale=0.2)

    # Positive slope through p_out = [a, exp(b)] (same for all the problems)
    if config["param"]:
        single_param = pysolvegn.Parametrization(
            lambda q: numpy.array([q[0], numpy.exp(q[1])]),
            lambda q: numpy.array([[1.0, 0.0], [0.0, numpy.exp(q[1])]]),
        )
        batch_param = pysolvegn.BatchParametrization(
            lambda Q: numpy.column_stack((Q[:, 0], numpy.exp(Q[:, 1]))),
            lambda Q: numpy.stack(
                [numpy.diag([1.0, numpy.exp(b)]) for b in Q[:, 1]]
            ),
        )
        p0 = numpy.tile([1.5, numpy.log(0.3)], (K, 1))
    else:
        single_param = batch_param = None
        p0 = P0

    single_update = (lambda p, dp: numpy.clip(dp, -0.5, 0.5)) if config["update"] else None
    batch_update = (lambda p, dp, indices: numpy.clip(dp, -0.5, 0.5)) if config["update"] else None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = pysolvegn.solve_batch(
            batch_term(weight=weights, **term_kwargs),
            p0,
            batch_param,
            damping=config["damping"],
            update_func=batch_update,
            **criteria(),
        )
        for k in range(K):
            single = pysolvegn.solve(
                line_single_term(Y, k, weight=weights[k], **term_kwargs),
                p0[k],
                single_param,
                damping=config["damping"],
                update_func=single_update,
                **criteria(),
            )
            numpy.testing.assert_allclose(result.parameters[k], single.parameters, rtol=1e-9, atol=1e-12)
            assert result.success[k] == single.success
            assert result.n_iterations[k] == single.n_iterations
            assert result.n_rejected[k] == single.n_rejected
            assert result.stop_code[k] == single.stop_code
            assert result.reasons(k) == single.reasons
            numpy.testing.assert_allclose(result.cost[k], single.cost)
            numpy.testing.assert_allclose(result.term_parameters[k], single.term_parameters)


# ----------------------------------------------------------------------
# BatchSolveResult
# ----------------------------------------------------------------------


def test_batch_result_fields():
    result = pysolvegn.solve_batch(batch_term(), P0, **criteria())
    assert isinstance(result, pysolvegn.BatchSolveResult)
    assert result.parameters.shape == (K, 2)
    assert result.success.shape == (K,) and result.success.dtype == bool
    assert result.n_iterations.shape == (K,)
    assert result.n_rejected.shape == (K,)
    assert result.cost.shape == (K,) and result.optimality.shape == (K,)
    assert result.term_parameters.shape == (K, 2)
    assert result.stop_code.shape == (K,)
    assert result.message.startswith(f"{K}/{K} problems converged.")
    assert result.config["ftol"] == 1e-10 and result.config["damping"] is None
    numpy.testing.assert_allclose(result.parameters, [numpy.polyfit(X_LINE, Y[k], 1)[::-1] for k in range(K)], atol=1e-8)


def test_p0_not_modified():
    p0 = P0.copy()
    pysolvegn.solve_batch(batch_term(), p0, **criteria())
    numpy.testing.assert_array_equal(p0, P0)


def test_callables_receive_subsets():
    # m changes during the optimization: the callables receive any subset of indices
    received = []
    residual = line_batch_residual(Y)

    def residual_func(p, indices):
        received.append(indices.copy())
        assert p.shape == (indices.shape[0], 2)
        return residual(p, indices)

    term = pysolvegn.BatchTerm.from_rJ(residual_func, line_batch_jacobian, loss="cauchy", loss_scale=0.2)
    pysolvegn.solve_batch(term, P0, **criteria())
    sizes = [indices.shape[0] for indices in received]
    assert sizes[0] == K
    assert min(sizes) < K  # some problems were removed from the processing


# ----------------------------------------------------------------------
# Per-problem stops
# ----------------------------------------------------------------------


def test_nan_in_some_p0():
    p0 = P0.copy()
    p0[[2, 5]] = numpy.nan
    result = pysolvegn.solve_batch(batch_term(), p0, **criteria())
    assert not result.success[2] and not result.success[5]
    assert result.stopped_by("naninf_p0")[2] and result.stopped_by("naninf")[5]
    assert result.reasons(2) == ["[naninf] Optimization not started: NaN or Inf value in the initial parameters p0."]
    assert numpy.isnan(result.cost[2])
    others = numpy.setdiff1d(numpy.arange(K), [2, 5])
    assert numpy.all(result.success[others])


def test_nan_in_all_p0():
    result = pysolvegn.solve_batch(batch_term(), numpy.full((K, 2), numpy.nan), **criteria())
    assert not numpy.any(result.success)
    assert result.term_parameters is None
    assert result.elapsed_time == 0.0


def test_singular_problem_only():
    # The problem 3 does not depend on the slope: its system is singular
    residual = line_batch_residual(Y)

    def jacobian_func(p, indices):
        J = line_batch_jacobian(p, indices)
        J[indices == 3, :, 1] = 0.0
        return J

    term = pysolvegn.BatchTerm.from_rJ(lambda p, i: residual(p, i) - (i == 3)[:, None] * p[:, 1:2] * X_LINE, jacobian_func)
    result = pysolvegn.solve_batch(term, P0, **criteria())
    assert not result.success[3]
    assert result.stopped_by("singular")[3]
    assert numpy.count_nonzero(result.stopped_by("singular")) == 1
    assert numpy.all(numpy.delete(result.success, 3))


def test_max_iteration_zero():
    result = pysolvegn.solve_batch(batch_term(), P0, max_iteration=0)
    assert not numpy.any(result.success)
    numpy.testing.assert_array_equal(result.n_iterations, 0)
    numpy.testing.assert_array_equal(result.parameters, P0)


def test_callback_array():
    # Stop the odd problems at the first iteration
    def callback(state):
        assert set(state) == {"indices", "parameters", "delta_parameters", "cost", "second_term", "hessian"}
        return state["indices"] % 2 == 0

    result = pysolvegn.solve_batch(batch_term(), P0, callback_func=callback, **criteria())
    assert not numpy.any(result.success[1::2])
    numpy.testing.assert_array_equal(result.n_iterations[1::2], 0)
    assert numpy.all(result.success[::2])


@pytest.mark.parametrize("value", [True, numpy.bool_(True)])
def test_callback_scalar(value):
    result = pysolvegn.solve_batch(batch_term(), P0, callback_func=lambda state: value, **criteria())
    assert numpy.all(result.success)


@pytest.mark.parametrize("bad", [1, numpy.ones(K), numpy.ones(K - 1, dtype=bool)])
def test_callback_invalid(bad):
    with pytest.raises(ValueError):
        pysolvegn.solve_batch(batch_term(), P0, callback_func=lambda state: bad, **criteria())


def test_gH_term_with_vector_weight_and_lm():
    weights = numpy.arange(1.0, K + 1.0)
    target = numpy.arange(2.0 * K).reshape(K, 2)
    gH = pysolvegn.BatchTerm.from_gH(
        lambda p, i: p - target[i],
        lambda p, i: numpy.broadcast_to(numpy.eye(2), (p.shape[0], 2, 2)).copy(),
        cost_func=lambda p, i: 0.5 * numpy.sum((p - target[i]) ** 2, axis=1),
        weight=weights,
    )
    result = pysolvegn.solve_batch(gH, numpy.zeros((K, 2)), damping="lm", **criteria())
    assert numpy.all(result.success)
    numpy.testing.assert_allclose(result.parameters, target, atol=1e-8)


def test_stop_code_and_message():
    result = pysolvegn.solve_batch(batch_term(), P0, max_iteration=60, gtol=1e-12, xtol=1e-10)
    assert numpy.all(result.success)
    either = result.stopped_by("gtol") | result.stopped_by("xtol")
    assert numpy.all(either)
    message = result.message
    assert f"[gtol] {numpy.count_nonzero(result.stopped_by('gtol'))} problem(s)." in message
    with pytest.raises(ValueError):
        result.stopped_by("unknown")


def test_cost_computed_at_convergence():
    # Only gtol: the cost is not needed during the optimization, but computed at the end
    result = pysolvegn.solve_batch(batch_term(), P0, max_iteration=60, gtol=1e-10)
    assert numpy.all(result.success)
    residuals = line_batch_residual(Y)(result.parameters, numpy.arange(K))
    numpy.testing.assert_allclose(result.cost, 0.5 * numpy.sum(residuals**2, axis=1))


def test_cost_none_without_success():
    result = pysolvegn.solve_batch(batch_term(), P0, max_iteration=1)
    assert not numpy.any(result.success)
    assert result.cost is None


def test_lm_conf():
    term = batch_term(loss="cauchy", loss_scale=0.2)
    result = pysolvegn.solve_batch(term, P0, damping="lm", lm_conf={"factor": 3.0}, **criteria())
    assert numpy.all(result.success)
    assert result.config["lm_conf"]["factor"] == 3.0
    assert result.config["lm_conf"]["max_rejections"] == 50
    restricted = pysolvegn.solve_batch(
        term, P0, damping="lm", lm_conf={"initial_scale": 1e-12, "max_rejections": 1}, **criteria()
    )
    assert numpy.any(restricted.stopped_by("lm"))
    k = int(numpy.flatnonzero(restricted.stopped_by("lm"))[0])
    assert "after 1 rejections" in restricted.reasons(k)[0]
    with pytest.raises(ValueError):
        pysolvegn.solve_batch(term, P0, damping="lm", lm_conf={"unknown": 1.0}, max_iteration=5)


# ----------------------------------------------------------------------
# History
# ----------------------------------------------------------------------


def test_history_all():
    result = pysolvegn.solve_batch(batch_term(loss="cauchy", loss_scale=0.2), P0, history=True, history_details="all", **criteria())
    first, last = result.history[0], result.history[-1]
    assert first["n_processing"] == K and first["delta_parameters"] is None
    assert last["parameters"].shape == (K, 2)
    assert last["hessian"].shape == (K, 2, 2)
    assert last["residuals"][0].shape == (K, X_LINE.size)
    assert last["is_processing"].shape == (K,)
    # Frozen values: the last cost of each problem is its returned cost
    numpy.testing.assert_allclose(last["cost"], result.cost)
    # delta_cost is NaN for the problems not in processing
    assert numpy.all(numpy.isnan(last["delta_cost"][~last["is_processing"]]))


@pytest.mark.parametrize("length", [0, 2, -2])
def test_history_length(length):
    result = pysolvegn.solve_batch(batch_term(), P0, history=True, history_details=["iteration"], history_length=length, **criteria())
    iterations = [h["iteration"] for h in result.history]
    n = int(result.n_iterations.max())
    if length == 0:
        assert iterations == []
    elif length > 0:
        assert iterations == [0, 1]
    else:
        assert iterations == [n - 1, n]


# ----------------------------------------------------------------------
# Invalid arguments
# ----------------------------------------------------------------------


def test_invalid_p0():
    with pytest.raises(ValueError):
        pysolvegn.solve_batch(batch_term(), numpy.zeros(2), max_iteration=5)
    with pytest.raises(ValueError):
        pysolvegn.solve_batch(batch_term(), numpy.zeros((0, 2)), max_iteration=5)


def test_vector_weight_size_mismatch():
    with pytest.raises(ValueError):
        pysolvegn.solve_batch(batch_term(weight=numpy.ones(K + 1)), P0, max_iteration=5)


def test_single_term_rejected():
    with pytest.raises(TypeError):
        pysolvegn.solve_batch(line_single_term(Y, 0), P0, max_iteration=5)


def test_single_parametrization_rejected():
    with pytest.raises(TypeError):
        pysolvegn.solve_batch(batch_term(), P0, pysolvegn.build_positive_parametrization(2), max_iteration=5)


def test_no_stopping_criterion():
    with pytest.raises(ValueError):
        pysolvegn.solve_batch(batch_term(), P0)


def test_invalid_history_detail():
    with pytest.raises(ValueError):
        pysolvegn.solve_batch(batch_term(), P0, max_iteration=5, history=True, history_details=["unknown"])