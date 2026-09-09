"""
pysolve-gn - Robust Gauss-Newton Least Squares Solver.
Copyright (C) 2026 Artezaru, artezaru.github@proton.me

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from typing import Tuple, Union, Callable
from numpy.typing import ArrayLike

import numpy
import scipy


def _build_tilde_R_and_tilde_J(
    residual_array: numpy.ndarray,
    jacobian_matrix: Union[numpy.ndarray, scipy.sparse.csr_matrix],
    rho_prime: numpy.ndarray,
    rho_double_prime: numpy.ndarray,
    _EPS: float = 1e-12,
) -> Tuple[numpy.ndarray, Union[numpy.ndarray, scipy.sparse.csr_matrix]]:
    r"""
    Build the modified residuals :math:`\tilde{\mathbf{R}}` and Jacobian :math:`\tilde{\mathbb{J}}`
    for the Gauss-Newton optimization with robust cost functions.

    The modified Jacobian :math:`\tilde{\mathbf{J}} = \sqrt{W_J} \mathbf{J}` and a
    modified residual :math:`\tilde{\mathbf{R}} = \frac{W_R}{\sqrt{W_J}} \mathbf{R}`
    are constructed such that the Gauss-Newton update can be written as:

    .. math::

        \tilde{\mathbf{J}}^T \tilde{\mathbf{J}} \Delta p = -\tilde{\mathbf{J}}^T \tilde{\mathbf{R}}

    where :

    .. math::

        W_J = \text{diag}\left(\rho'(|\mathbf{R}_j|^2) + 2 \rho''(|\mathbf{R}_j|^2) |\mathbf{R}_j|^2\right)

    .. math::

        W_R = \text{diag}\left(\rho'(|\mathbf{R}_j|^2)\right)


    Parameters
    ----------
    residual_array: numpy.ndarray
        The array of residuals for the least squares problem. Shape ``(n_residuals,)``.

    jacobian_matrix: Union[numpy.ndarray, scipy.sparse.csr_matrix]
        The Jacobian matrix of the residuals with respect to the parameters.
        Shape ``(n_residuals, n_parameters)``.

    rho_prime: numpy.ndarray
        The first derivative of the cost function evaluated at the squared-residual,
        with shape ``(n_residuals,)``.

    rho_double_prime: numpy.ndarray
        The second derivative of the cost function evaluated at the squared-residual,
        with shape ``(n_residuals,)``.


    Returns
    -------
    tilde_R : numpy.ndarray
        The modified residuals for the Gauss-Newton optimization. Shape ``(n_residuals,)``.

    tilde_J : Union[numpy.ndarray, scipy.sparse.csr_matrix]
        The modified Jacobian matrix for the Gauss-Newton optimization.
        Shape ``(n_residuals, n_parameters)``.

    """
    scale = rho_prime + 2 * rho_double_prime * residual_array**2
    scale = numpy.maximum(scale, _EPS)  # Avoid division by zero or negative values
    sqrt_scale = numpy.sqrt(scale)

    if scipy.sparse.issparse(jacobian_matrix):
        W_J_sqrt = scipy.sparse.diags(sqrt_scale)
        tilde_J = W_J_sqrt @ jacobian_matrix
    else:
        tilde_J = sqrt_scale[:, numpy.newaxis] * jacobian_matrix

    tilde_R = (rho_prime / sqrt_scale) * residual_array

    return tilde_R, tilde_J

def linear_rho(
    x: ArrayLike,
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    r"""
    Compute the linear robust cost function :math:`\rho(x) = x` and its derivatives.

    .. math::

        \rho'(x) = 1

    .. math::

        \rho''(x) = 0

    Parameters
    ----------
    x : ArrayLike
        Squared residuals :math:`x = \|R\|^2`.
        Shape ``(N,)``.

    Returns
    -------
    rho : numpy.ndarray
        The cost function values with shape ``(N,)``.

    rho_prime : numpy.ndarray
        The first derivatives with shape ``(N,)``.

    rho_double_prime : numpy.ndarray
        The second derivatives with shape ``(N,)``.
    """
    x = numpy.asarray(x, dtype=numpy.float64)

    return (
        x,
        numpy.ones_like(x),
        numpy.zeros_like(x),
    )


def soft_l1_rho(
    x: ArrayLike,
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    r"""
    Compute the soft L1 robust cost function
    :math:`\rho(x) = 2(\sqrt{1 + x} - 1)` and its derivatives.

    .. math::

        \rho'(x) = \frac{1}{\sqrt{1 + x}}

    .. math::

        \rho''(x) = -\frac{1}{2(1 + x)^{3/2}}

    Parameters
    ----------
    x : ArrayLike
        Squared residuals :math:`x = \|R\|^2`.
        Shape ``(N,)``.

    Returns
    -------
    rho : numpy.ndarray
        The cost function values with shape ``(N,)``.

    rho_prime : numpy.ndarray
        The first derivatives with shape ``(N,)``.

    rho_double_prime : numpy.ndarray
        The second derivatives with shape ``(N,)``.
    """
    x = numpy.asarray(x, dtype=numpy.float64)

    return (
        2 * (numpy.sqrt(1 + x) - 1),
        1 / numpy.sqrt(1 + x),
        -0.5 / (1 + x) ** (3 / 2),
    )


def cauchy_rho(
    x: ArrayLike,
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    r"""
    Compute the Cauchy robust cost function
    :math:`\rho(x) = \log(1 + x)` and its derivatives.

    .. math::

        \rho'(x) = \frac{1}{1 + x}

    .. math::

        \rho''(x) = -\frac{1}{(1 + x)^2}

    Parameters
    ----------
    x : ArrayLike
        Squared residuals :math:`x = \|R\|^2`.
        Shape ``(N,)``.

    Returns
    -------
    rho : numpy.ndarray
        The cost function values with shape ``(N,)``.

    rho_prime : numpy.ndarray
        The first derivatives with shape ``(N,)``.

    rho_double_prime : numpy.ndarray
        The second derivatives with shape ``(N,)``.
    """
    x = numpy.asarray(x, dtype=numpy.float64)

    return (
        numpy.log(1 + x),
        1 / (1 + x),
        -1 / (1 + x) ** 2,
    )


def arctan_rho(
    x: ArrayLike,
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    r"""
    Compute the arctan robust cost function
    :math:`\rho(x) = \arctan(x)` and its derivatives.

    .. math::

        \rho'(x) = \frac{1}{1 + x^2}

    .. math::

        \rho''(x) = -\frac{2x}{(1 + x^2)^2}

    Parameters
    ----------
    x : ArrayLike
        Squared residuals :math:`x = \|R\|^2`.
        Shape ``(N,)``.

    Returns
    -------
    rho : numpy.ndarray
        The cost function values with shape ``(N,)``.

    rho_prime : numpy.ndarray
        The first derivatives with shape ``(N,)``.

    rho_double_prime : numpy.ndarray
        The second derivatives with shape ``(N,)``.
    """
    x = numpy.asarray(x, dtype=numpy.float64)

    return (
        numpy.arctan(x),
        1 / (1 + x**2),
        -2 * x / (1 + x**2) ** 2,
    )


def get_rho_function_by_name(name: str) -> Callable:
    r"""
    Get a predefined rho function by name.

    Parameters
    ----------
    name : str
        The name of the rho function. Must be one of ``"linear"``,
        ``"soft_l1"``, ``"cauchy"``, or ``"arctan"``.

    Returns
    -------
    rho_func : Callable
        A function that takes the squared residuals
        :math:`x = \|R\|^2` and returns the cost function value,
        its first derivative, and its second derivative.
    """
    name = name.lower()

    if name == "linear":
        return linear_rho
    elif name == "soft_l1":
        return soft_l1_rho
    elif name == "cauchy":
        return cauchy_rho
    elif name == "arctan":
        return arctan_rho
    else:
        raise ValueError(
            f"Invalid rho function name '{name}'. "
            "Must be one of 'linear', 'soft_l1', 'cauchy', or 'arctan'."
        )