"""
Shared test problems for the pysolve-gn test suite.
"""

import numpy

import pysolvegn


# ----------------------------------------------------------------------
# Exponential curve fitting: y = a * exp(b * x)
# ----------------------------------------------------------------------

X_EXP = numpy.linspace(0.0, 3.0, 60)
TRUE_EXP = numpy.array([2.5, 0.5])


def exp_model(params, x=X_EXP):
    a, b = params
    return a * numpy.exp(b * x)


def exp_data(noise=0.1, n_outliers=0, seed=0):
    rng = numpy.random.default_rng(seed)
    y = exp_model(TRUE_EXP) + noise * rng.normal(size=X_EXP.size)
    if n_outliers > 0:
        idx = rng.choice(X_EXP.size, n_outliers, replace=False)
        y[idx] += rng.uniform(5.0, 10.0, n_outliers)
    return y


def exp_term(y, **kwargs):
    def residual_func(p):
        return exp_model(p) - y

    def jacobian_func(p):
        a, b = p
        e = numpy.exp(b * X_EXP)
        return numpy.column_stack((e, a * X_EXP * e))

    return pysolvegn.Term.from_rJ(residual_func, jacobian_func, **kwargs)


# ----------------------------------------------------------------------
# Batch of K independent line fits: y_k = a_k + b_k * x
# ----------------------------------------------------------------------

X_LINE = numpy.linspace(0.0, 10.0, 40)


def line_batch_data(K=12, noise=0.05, outlier_fraction=0.0, seed=0):
    rng = numpy.random.default_rng(seed)
    truth = numpy.column_stack((rng.uniform(1, 3, K), rng.uniform(0.2, 0.6, K)))
    Y = truth[:, :1] + truth[:, 1:] * X_LINE + noise * rng.normal(size=(K, X_LINE.size))
    if outlier_fraction > 0:
        mask = rng.random(Y.shape) < outlier_fraction
        Y[mask] += rng.uniform(2.0, 6.0, mask.sum())
    return truth, Y


def line_batch_residual(Y):
    def residual_func(p, indices):
        return p[:, :1] + p[:, 1:2] * X_LINE - Y[indices]

    return residual_func


def line_batch_jacobian(p, indices):
    J = numpy.stack((numpy.ones_like(X_LINE), X_LINE), axis=1)
    return numpy.broadcast_to(J, (p.shape[0],) + J.shape).copy()


def line_single_term(Y, k, **kwargs):
    # Term of the problem k alone (for comparison with the batch solver)
    return pysolvegn.Term.from_rJ(
        lambda p: p[0] + p[1] * X_LINE - Y[k],
        lambda p: numpy.stack((numpy.ones_like(X_LINE), X_LINE), axis=1),
        **kwargs,
    )


def numerical_jacobian(func, p, eps=1e-6):
    # Central finite differences of func: (n,) -> (m,)
    p = numpy.asarray(p, dtype=float)
    cols = []
    for j in range(p.size):
        e = numpy.zeros_like(p)
        e[j] = eps
        cols.append((numpy.asarray(func(p + e)) - numpy.asarray(func(p - e))) / (2 * eps))
    return numpy.stack(cols, axis=-1)
