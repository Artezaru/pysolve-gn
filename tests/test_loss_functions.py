"""
Tests of the robust loss functions, the loss scaling and the robust weighting
(r~, J~) with the fallback on rho'.
"""

import numpy
import pytest
import scipy.sparse

import pysolvegn
from pysolvegn.loss_functions import (
    _build_tilde_R_and_tilde_J,
    _build_batch_tilde_R_and_tilde_J,
)

LOSSES = ["linear", "soft_l1", "huber", "cauchy", "arctan", "tukey"]

# Squared residuals away from the non-smooth point z = 1 (huber, tukey)
Z = numpy.concatenate([numpy.linspace(1e-3, 0.95, 40), numpy.linspace(1.05, 20.0, 40)])


@pytest.mark.parametrize("name", LOSSES)
def test_loss_at_zero(name):
    # All the losses behave as the linear loss for small residuals.
    rho, rho_prime, _ = pysolvegn.get_rho_function_by_name(name)(numpy.array([0.0]))
    numpy.testing.assert_allclose(rho, 0.0, atol=1e-15)
    numpy.testing.assert_allclose(rho_prime, 1.0)


@pytest.mark.parametrize("name", LOSSES)
@pytest.mark.parametrize("scale", [1.0, 0.3, 7.0])
def test_loss_derivatives_finite_differences(name, scale):
    rho_func = pysolvegn.scale_rho_function(pysolvegn.get_rho_function_by_name(name), scale)
    z = Z * scale**2
    h = 1e-6 * scale**2
    rho, rho_prime, rho_double_prime = rho_func(z)
    rho_p, rho_prime_p, _ = rho_func(z + h)
    rho_m, rho_prime_m, _ = rho_func(z - h)
    numpy.testing.assert_allclose((rho_p - rho_m) / (2 * h), rho_prime, rtol=1e-5, atol=1e-9)
    numpy.testing.assert_allclose(
        (rho_prime_p - rho_prime_m) / (2 * h), rho_double_prime, rtol=1e-4, atol=1e-7
    )


@pytest.mark.parametrize("name", LOSSES)
def test_loss_output_shapes(name):
    z = numpy.abs(numpy.random.default_rng(0).normal(size=(4, 7)))
    for output in pysolvegn.get_rho_function_by_name(name)(z):
        assert numpy.shape(output) == z.shape


def test_huber_matches_scipy():
    from scipy.optimize._lsq.least_squares import IMPLEMENTED_LOSSES

    z = numpy.linspace(0.0, 10.0, 101)
    expected = numpy.empty((3, z.size))
    IMPLEMENTED_LOSSES["huber"](z, expected, cost_only=False)
    numpy.testing.assert_allclose(numpy.array(pysolvegn.huber_rho(z)), expected)


def test_tukey_ignores_large_residuals():
    rho, rho_prime, rho_double_prime = pysolvegn.tukey_rho(numpy.array([1.0, 2.0, 1e300]))
    numpy.testing.assert_allclose(rho, 1.0 / 3.0)
    numpy.testing.assert_allclose(rho_prime, 0.0)
    numpy.testing.assert_allclose(rho_double_prime, 0.0)


@pytest.mark.parametrize("name", ["huber", "tukey"])
def test_loss_finite_for_extreme_values(name):
    for output in pysolvegn.get_rho_function_by_name(name)(numpy.array([0.0, 1.0, 1e12])):
        assert numpy.all(numpy.isfinite(output))


def test_get_rho_function_by_name_case_insensitive():
    assert pysolvegn.get_rho_function_by_name("CaUcHy") is pysolvegn.cauchy_rho


def test_get_rho_function_by_name_invalid():
    with pytest.raises(ValueError):
        pysolvegn.get_rho_function_by_name("unknown")


# ----------------------------------------------------------------------
# Loss scaling
# ----------------------------------------------------------------------


def test_scale_rho_function_identity():
    assert pysolvegn.scale_rho_function(pysolvegn.cauchy_rho, 1.0) is pysolvegn.cauchy_rho


def test_scale_rho_function_definition():
    C = 0.5
    scaled = pysolvegn.scale_rho_function(pysolvegn.cauchy_rho, C)
    rho, rho_prime, rho_double_prime = scaled(Z)
    base = pysolvegn.cauchy_rho(Z / C**2)
    numpy.testing.assert_allclose(rho, C**2 * base[0])
    numpy.testing.assert_allclose(rho_prime, base[1])
    numpy.testing.assert_allclose(rho_double_prime, base[2] / C**2)


def test_scale_rho_function_linear_invariant():
    scaled = pysolvegn.scale_rho_function(pysolvegn.linear_rho, 3.0)
    for a, b in zip(scaled(Z), pysolvegn.linear_rho(Z)):
        numpy.testing.assert_allclose(a, b)


def test_scale_rho_function_custom_loss():
    custom = lambda z: (numpy.log1p(z), 1 / (1 + z), -1 / (1 + z) ** 2)
    scaled = pysolvegn.scale_rho_function(custom, 2.0)
    numpy.testing.assert_allclose(scaled(numpy.array([8.0]))[0], 4.0 * numpy.log1p(2.0))


@pytest.mark.parametrize("bad", [0.0, -1.0, numpy.inf, numpy.nan])
def test_scale_rho_function_invalid_scale_value(bad):
    with pytest.raises(ValueError):
        pysolvegn.scale_rho_function(pysolvegn.cauchy_rho, bad)


@pytest.mark.parametrize("bad", ["1.0", True, None])
def test_scale_rho_function_invalid_scale_type(bad):
    with pytest.raises(TypeError):
        pysolvegn.scale_rho_function(pysolvegn.cauchy_rho, bad)


# ----------------------------------------------------------------------
# Robust weighting r~, J~ (with the fallback on rho')
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name", LOSSES)
def test_tilde_gradient_exact_and_hessian_psd(name):
    rng = numpy.random.default_rng(1)
    r = 2.0 * rng.normal(size=50)
    J = rng.normal(size=(50, 3))
    _, rho_prime, rho_double_prime = pysolvegn.get_rho_function_by_name(name)(r**2)
    rho_prime = numpy.broadcast_to(rho_prime, r.shape)
    r_tilde, J_tilde = _build_tilde_R_and_tilde_J(r, J, rho_prime, rho_double_prime)

    assert numpy.all(numpy.isfinite(r_tilde)) and numpy.all(numpy.isfinite(J_tilde))
    # The gradient is always exact: J~^T r~ = J^T (rho' r)
    numpy.testing.assert_allclose(J_tilde.T @ r_tilde, J.T @ (rho_prime * r), atol=1e-12)
    # The Hessian approximation stays positive semi-definite
    assert numpy.min(numpy.linalg.eigvalsh(J_tilde.T @ J_tilde)) >= -1e-12


def test_tilde_fallback_uses_rho_prime():
    # Cauchy beyond z = 1: rho' + 2 rho'' z <= 0 -> IRLS weighting sqrt(rho')
    r = numpy.array([3.0])
    J = numpy.array([[2.0]])
    _, rho_prime, rho_double_prime = pysolvegn.cauchy_rho(r**2)
    r_tilde, J_tilde = _build_tilde_R_and_tilde_J(r, J, rho_prime, rho_double_prime)
    numpy.testing.assert_allclose(r_tilde, numpy.sqrt(rho_prime) * r)
    numpy.testing.assert_allclose(J_tilde, numpy.sqrt(rho_prime)[:, None] * J)


def test_tilde_ignored_residual_tukey():
    r = numpy.array([0.1, 5.0])
    J = numpy.ones((2, 2))
    _, rho_prime, rho_double_prime = pysolvegn.tukey_rho(r**2)
    r_tilde, J_tilde = _build_tilde_R_and_tilde_J(r, J, rho_prime, rho_double_prime)
    assert r_tilde[1] == 0.0
    assert numpy.all(J_tilde[1] == 0.0)


@pytest.mark.parametrize("name", ["huber", "cauchy", "tukey"])
def test_tilde_sparse_equals_dense(name):
    rng = numpy.random.default_rng(2)
    r = 2.0 * rng.normal(size=30)
    J = rng.normal(size=(30, 4))
    _, rho_prime, rho_double_prime = pysolvegn.get_rho_function_by_name(name)(r**2)
    r_dense, J_dense = _build_tilde_R_and_tilde_J(r, J, rho_prime, rho_double_prime)
    r_sparse, J_sparse = _build_tilde_R_and_tilde_J(
        r, scipy.sparse.csr_matrix(J), rho_prime, rho_double_prime
    )
    assert scipy.sparse.issparse(J_sparse)
    numpy.testing.assert_allclose(r_sparse, r_dense)
    numpy.testing.assert_allclose(J_sparse.toarray(), J_dense)


@pytest.mark.parametrize("name", LOSSES)
def test_batch_tilde_equals_single(name):
    rng = numpy.random.default_rng(3)
    R = 2.0 * rng.normal(size=(3, 20))
    J = rng.normal(size=(3, 20, 2))
    _, rho_prime, rho_double_prime = pysolvegn.get_rho_function_by_name(name)(R**2)
    rho_prime = numpy.broadcast_to(rho_prime, R.shape)
    rho_double_prime = numpy.broadcast_to(rho_double_prime, R.shape)
    R_tilde, J_tilde = _build_batch_tilde_R_and_tilde_J(R, J, rho_prime, rho_double_prime)
    for k in range(3):
        r_k, J_k = _build_tilde_R_and_tilde_J(R[k], J[k], rho_prime[k], rho_double_prime[k])
        numpy.testing.assert_allclose(R_tilde[k], r_k)
        numpy.testing.assert_allclose(J_tilde[k], J_k)
