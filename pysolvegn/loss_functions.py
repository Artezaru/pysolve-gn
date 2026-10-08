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
from numbers import Real
from numpy.typing import ArrayLike

import numpy
import scipy.sparse


def _compute_tilde_factors(
    residual_array: numpy.ndarray,
    rho_prime: ArrayLike,
    rho_double_prime: ArrayLike,
    _EPS: float,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    r"""
    Compute the factors :math:`\sqrt{W_J}` and :math:`W_R / \sqrt{W_J}` (element-wise) used
    to build :math:`\tilde{\mathbf{J}}` and :math:`\tilde{\mathbf{R}}`, with the IRLS fallback:

    - :math:`W_J = \rho' + 2 \rho'' r^2` if :math:`W_J > \varepsilon`,
    - else :math:`W_J = \max(\rho', 0)` (second-order correction dropped),
    - and both factors are ``0`` where :math:`W_J = 0` (residual ignored).

    Works for any shape (``(n_residuals,)`` or ``(m, n_residuals)``).
    """
    residual_array = numpy.asarray(residual_array, dtype=numpy.float64)
    rho_prime = numpy.broadcast_to(
        numpy.asarray(rho_prime, dtype=numpy.float64), residual_array.shape
    )
    rho_double_prime = numpy.asarray(rho_double_prime, dtype=numpy.float64)

    scale = rho_prime + 2 * rho_double_prime * residual_array**2
    scale = numpy.where(scale > _EPS, scale, numpy.maximum(rho_prime, 0.0))
    sqrt_scale = numpy.sqrt(scale)

    is_active = sqrt_scale > 0.0
    ratio = numpy.zeros_like(sqrt_scale)
    numpy.divide(rho_prime, sqrt_scale, out=ratio, where=is_active)

    return sqrt_scale, ratio


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

    .. note::

        **IRLS fallback.** For a non-convex loss (``"cauchy"``, ``"arctan"``,
        ``"tukey"``, ...), the term :math:`\rho' + 2 \rho'' |\mathbf{R}_j|^2` becomes
        negative or zero for the large residuals. For these residuals
        (:math:`W_{J,j} \leq \varepsilon`), the second-order correction is dropped and
        :math:`W_{J,j} = \rho'(|\mathbf{R}_j|^2)` is used instead (as in Ceres Solver),
        which gives :math:`\tilde{\mathbf{R}}_j = \sqrt{\rho'} \mathbf{R}_j` and
        :math:`\tilde{\mathbf{J}}_j = \sqrt{\rho'} \mathbf{J}_j` (standard IRLS weighting).
        The gradient :math:`\tilde{\mathbf{J}}^T \tilde{\mathbf{R}} = \mathbf{J}^T \rho' \mathbf{R}`
        is exact in both cases. If :math:`\rho' \leq 0` too (e.g. ``"tukey"`` beyond its
        threshold), the residual is simply ignored (:math:`\tilde{\mathbf{R}}_j = 0` and
        :math:`\tilde{\mathbf{J}}_j = 0`).


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
    sqrt_scale, ratio = _compute_tilde_factors(
        residual_array, rho_prime, rho_double_prime, _EPS
    )

    if scipy.sparse.issparse(jacobian_matrix):
        W_J_sqrt = scipy.sparse.diags(sqrt_scale)
        tilde_J = W_J_sqrt @ jacobian_matrix
    else:
        tilde_J = sqrt_scale[:, numpy.newaxis] * jacobian_matrix

    tilde_R = ratio * residual_array

    return tilde_R, tilde_J


def _build_batch_tilde_R_and_tilde_J(
    residual_array: numpy.ndarray,
    jacobian_array: numpy.ndarray,
    rho_prime: numpy.ndarray,
    rho_double_prime: numpy.ndarray,
    _EPS: float = 1e-12,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    r"""
    Build the modified residuals :math:`\tilde{\mathbf{R}}_k` and Jacobians :math:`\tilde{\mathbf{J}}_k`
    for each problem :math:`k` of a batch of independent Gauss-Newton optimizations with robust
    cost functions.

    This is the batched counterpart of :func:`_build_tilde_R_and_tilde_J`. For each problem
    :math:`k`, the modified Jacobian :math:`\tilde{\mathbf{J}}_k = \sqrt{W_{J,k}} \mathbf{J}_k` and
    the modified residual :math:`\tilde{\mathbf{R}}_k = \frac{W_{R,k}}{\sqrt{W_{J,k}}} \mathbf{R}_k`
    are constructed such that the Gauss-Newton update can be written as:

    .. math::

        \tilde{\mathbf{J}}_k^T \tilde{\mathbf{J}}_k \Delta p_k = -\tilde{\mathbf{J}}_k^T \tilde{\mathbf{R}}_k

    where :

    .. math::

        W_{J,k} = \text{diag}\left(\rho'(|\mathbf{R}_{k,j}|^2) + 2 \rho''(|\mathbf{R}_{k,j}|^2) |\mathbf{R}_{k,j}|^2\right)

    .. math::

        W_{R,k} = \text{diag}\left(\rho'(|\mathbf{R}_{k,j}|^2)\right)

    The same IRLS fallback as :func:`_build_tilde_R_and_tilde_J` is used when
    :math:`W_{J,k,j} \leq \varepsilon`.

    Parameters
    ----------
    residual_array: numpy.ndarray
        The residuals of each problem of the batch. Shape ``(m, n_residuals)``.

    jacobian_array: numpy.ndarray
        The Jacobian matrices of the residuals with respect to the parameters of each
        problem of the batch. Shape ``(m, n_residuals, n_parameters)``.

    rho_prime: numpy.ndarray
        The first derivative of the cost function evaluated at the squared-residuals,
        with shape ``(m, n_residuals)``.

    rho_double_prime: numpy.ndarray
        The second derivative of the cost function evaluated at the squared-residuals,
        with shape ``(m, n_residuals)``.


    Returns
    -------
    tilde_R : numpy.ndarray
        The modified residuals for the batched Gauss-Newton optimization.
        Shape ``(m, n_residuals)``.

    tilde_J : numpy.ndarray
        The modified Jacobian matrices for the batched Gauss-Newton optimization.
        Shape ``(m, n_residuals, n_parameters)``.

    """
    sqrt_scale, ratio = _compute_tilde_factors(
        residual_array, rho_prime, rho_double_prime, _EPS
    )

    tilde_J = sqrt_scale[..., numpy.newaxis] * jacobian_array
    tilde_R = ratio * residual_array

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


def huber_rho(
    x: ArrayLike,
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    r"""
    Compute the Huber robust cost function and its derivatives
    (same definition as ``scipy.optimize.least_squares``).

    .. math::

        \rho(x) = \begin{cases} x & \text{if } x \leq 1 \\ 2\sqrt{x} - 1 & \text{otherwise} \end{cases}

    .. math::

        \rho'(x) = \begin{cases} 1 & \text{if } x \leq 1 \\ x^{-1/2} & \text{otherwise} \end{cases}

    .. math::

        \rho''(x) = \begin{cases} 0 & \text{if } x \leq 1 \\ -\frac{1}{2} x^{-3/2} & \text{otherwise} \end{cases}

    Quadratic for the small residuals (:math:`|r| \leq 1`) and linear for the large ones.
    The loss is convex: :math:`\rho' + 2 \rho'' x \geq 0` everywhere (equal to ``0``
    beyond the threshold, where the IRLS fallback of :func:`_build_tilde_R_and_tilde_J` applies).

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
    is_inlier = x <= 1.0
    sqrt_x = numpy.sqrt(numpy.where(is_inlier, 1.0, x))  # avoid 0 ** (-1/2) on the inliers

    return (
        numpy.where(is_inlier, x, 2 * sqrt_x - 1),
        numpy.where(is_inlier, 1.0, 1 / sqrt_x),
        numpy.where(is_inlier, 0.0, -0.5 / sqrt_x**3),
    )


def tukey_rho(
    x: ArrayLike,
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    r"""
    Compute the Tukey biweight (bisquare) robust cost function and its derivatives.

    .. math::

        \rho(x) = \begin{cases} \frac{1 - (1 - x)^3}{3} & \text{if } x \leq 1 \\ \frac{1}{3} & \text{otherwise} \end{cases}

    .. math::

        \rho'(x) = \begin{cases} (1 - x)^2 & \text{if } x \leq 1 \\ 0 & \text{otherwise} \end{cases}

    .. math::

        \rho''(x) = \begin{cases} -2(1 - x) & \text{if } x \leq 1 \\ 0 & \text{otherwise} \end{cases}

    The residuals beyond the threshold (:math:`|r| > 1`) are completely ignored
    (:math:`\rho' = 0`). The loss is strongly non-convex: a good initial guess is
    required, since a point whose residuals are all beyond the threshold has a zero
    gradient (and a singular Hessian).

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
    one_minus_x = numpy.maximum(1.0 - x, 0.0)  # = 0 beyond the threshold

    return (
        (1 - one_minus_x**3) / 3,
        one_minus_x**2,
        -2 * one_minus_x,
    )


def scale_rho_function(rho_func: Callable, loss_scale: Real) -> Callable:
    r"""
    Wrap any loss function :math:`\rho` (predefined or custom) into its scaled version
    :math:`\rho_C` with the soft threshold :math:`C` (``loss_scale``):

    .. math::

        \rho_C(x) = C^2 \rho\left(\frac{x}{C^2}\right)

    .. math::

        \rho_C'(x) = \rho'\left(\frac{x}{C^2}\right)
        \qquad
        \rho_C''(x) = \frac{1}{C^2} \rho''\left(\frac{x}{C^2}\right)

    :math:`C` is the residual value separating the inliers from the outliers, expressed
    in the unit of the residuals (same as ``f_scale`` in ``scipy.optimize.least_squares``).
    For small residuals (:math:`|r| \ll C`), :math:`\rho_C(r^2) \approx r^2` whatever
    :math:`C`. The linear loss is invariant: :math:`\rho_C(x) = x`.

    Parameters
    ----------
    rho_func : Callable
        The loss function taking the squared residuals and returning
        :math:`(\rho, \rho', \rho'')`.

    loss_scale : Real
        The soft threshold :math:`C`. Must be finite and strictly positive.

    Returns
    -------
    Callable
        The scaled loss function with the same signature. If ``loss_scale == 1``,
        ``rho_func`` is returned unchanged.
    """
    if not callable(rho_func):
        raise TypeError(f"rho_func must be a callable function, got {type(rho_func)}.")
    loss_scale = _validate_loss_scale(loss_scale)
    if loss_scale == 1.0:
        return rho_func

    C2 = loss_scale**2

    def scaled_rho(x: ArrayLike) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        x = numpy.asarray(x, dtype=numpy.float64)
        rho, rho_prime, rho_double_prime = rho_func(x / C2)
        return (
            C2 * numpy.asarray(rho, dtype=numpy.float64),
            numpy.asarray(rho_prime, dtype=numpy.float64),
            numpy.asarray(rho_double_prime, dtype=numpy.float64) / C2,
        )

    scaled_rho.__name__ = f"scaled_{getattr(rho_func, '__name__', 'rho')}"
    scaled_rho.__doc__ = f"{getattr(rho_func, '__name__', 'rho')} scaled with loss_scale={loss_scale}."
    return scaled_rho


def _validate_loss_scale(loss_scale: Real) -> float:
    r"""
    Check that ``loss_scale`` is a finite and strictly positive real number and return it as a float.
    """
    if isinstance(loss_scale, bool) or not isinstance(loss_scale, Real):
        raise TypeError(f"loss_scale must be a real number, got {type(loss_scale)}.")
    loss_scale = float(loss_scale)
    if not numpy.isfinite(loss_scale) or loss_scale <= 0.0:
        raise ValueError(f"loss_scale must be a finite strictly positive number, got {loss_scale}.")
    return loss_scale


_RHO_FUNCTIONS = {
    "linear": linear_rho,
    "soft_l1": soft_l1_rho,
    "cauchy": cauchy_rho,
    "arctan": arctan_rho,
    "huber": huber_rho,
    "tukey": tukey_rho,
}


def get_rho_function_by_name(name: str) -> Callable:
    r"""
    Get a predefined rho function by name.

    Parameters
    ----------
    name : str
        The name of the rho function. Must be one of ``"linear"``,
        ``"soft_l1"``, ``"cauchy"``, ``"arctan"``, ``"huber"`` or ``"tukey"``.

    Returns
    -------
    rho_func : Callable
        A function that takes the squared residuals
        :math:`x = \|R\|^2` and returns the cost function value,
        its first derivative, and its second derivative.
    """
    key = name.lower()
    if key not in _RHO_FUNCTIONS:
        raise ValueError(
            f"Invalid rho function name '{name}'. Must be one of {tuple(_RHO_FUNCTIONS)}."
        )
    return _RHO_FUNCTIONS[key]