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

from typing import Callable, Optional, Tuple
from numbers import Real

import numpy

from .implemented_conf import _IMPLEMENTED_FINITE_DIFFERENCE_METHODS


# Default relative steps of the finite differences (optimal order of magnitude for
# float64 when the residuals are computed to machine precision):
# - one-sided differences: truncation error O(h), rounding error O(eps / h) -> h ~ sqrt(eps)
# - central differences:   truncation error O(h^2), rounding error O(eps / h) -> h ~ eps^(1/3)
_MACHINE_EPSILON = float(numpy.finfo(numpy.float64).eps)
_DEFAULT_FINITE_DIFFERENCE_EPSILON = {
    "forward": _MACHINE_EPSILON ** (1.0 / 2.0),  # ~1.49e-8
    "backward": _MACHINE_EPSILON ** (1.0 / 2.0),  # ~1.49e-8
    "central": _MACHINE_EPSILON ** (1.0 / 3.0),  # ~6.06e-6
}


def _validate_finite_difference(
    method: Optional[str],
    epsilon: Optional[Real],
    *,
    allow_none: bool = False,
) -> Tuple[Optional[str], Optional[float]]:
    r"""
    Validate a finite difference method and its relative step.

    Parameters
    ----------
    method : Optional[str]
        ``"central"``, ``"forward"`` or ``"backward"`` (case insensitive). None is
        accepted only if ``allow_none`` is True (no finite differences: ``epsilon`` must
        then be None too).

    epsilon : Optional[Real]
        The relative step, finite and strictly positive, or None for the default step of
        the method (see ``_DEFAULT_FINITE_DIFFERENCE_EPSILON``).

    Returns
    -------
    Tuple[Optional[str], Optional[float]]
        The lower-case method and the step actually used (None, None if ``method`` is None).
    """
    if method is None:
        if not allow_none:
            raise ValueError(
                f"The finite difference method must be one of {_IMPLEMENTED_FINITE_DIFFERENCE_METHODS}."
            )
        if epsilon is not None:
            raise ValueError(
                "finite_difference_epsilon can only be given with finite_difference."
            )
        return None, None

    if not isinstance(method, str):
        raise TypeError("The finite difference method must be a string.")
    method = method.lower()
    if method not in _IMPLEMENTED_FINITE_DIFFERENCE_METHODS:
        raise ValueError(
            f"The finite difference method must be one of {_IMPLEMENTED_FINITE_DIFFERENCE_METHODS}, got '{method}'."
        )

    if epsilon is None:
        return method, _DEFAULT_FINITE_DIFFERENCE_EPSILON[method]
    if isinstance(epsilon, bool) or not isinstance(epsilon, Real):
        raise TypeError("The finite difference epsilon must be a real number.")
    epsilon = float(epsilon)
    if not numpy.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("The finite difference epsilon must be a finite strictly positive number.")
    return method, epsilon


def build_numerical_jacobian(
    residual_func: Callable,
    method: str = "central",
    epsilon: Optional[Real] = None,
) -> Callable:
    r"""
    Build a Jacobian function for the Gauss-Newton optimization by computing the numerical
    derivatives of the residual function with respect to the parameters using finite differences.

    The step of the parameter :math:`j` is relative to its magnitude:
    :math:`h_j = \epsilon \max(1, |p_j|)`. The step actually used is the representable
    one, :math:`h_j = (p_j + h_j) - p_j`, which removes the rounding error on the step.

    With the ``"central"`` method (``2 n_parameters`` calls of ``residual_func``),
    the Jacobian is computed as:

    .. math::

        J_{i,j} = \frac{R_i(p + h_j e_j) - R_i(p - h_j e_j)}{2 h_j}

    With the ``"forward"`` method (``n_parameters + 1`` calls), the Jacobian is computed as:

    .. math::

        J_{i,j} = \frac{R_i(p + h_j e_j) - R_i(p)}{h_j}

    With the ``"backward"`` method (``n_parameters + 1`` calls), the Jacobian is computed as:

    .. math::

        J_{i,j} = \frac{R_i(p) - R_i(p - h_j e_j)}{h_j}

    Parameters
    ----------
    residual_func : Callable
        A function that computes the residuals for a given set of parameters.
        The function should take a 1D array of parameters as input and
        return a 1D array of residuals.

    method : str, optional (default="central")
        The finite difference method to use for computing the numerical Jacobian.
        Must be one of "central", "forward", or "backward".

    epsilon : Optional[Real], optional (default=None)
        The relative step :math:`\epsilon` of the finite differences (finite and strictly
        positive). If None, the optimal order of magnitude for residuals computed to the
        machine precision :math:`\varepsilon_m \approx 2.2 \times 10^{-16}` is used:
        :math:`\sqrt{\varepsilon_m} \approx 1.5 \times 10^{-8}` for ``"forward"`` and
        ``"backward"``, :math:`\varepsilon_m^{1/3} \approx 6.1 \times 10^{-6}` for
        ``"central"``. Use a larger value if the residuals are noisy or computed with
        a lower precision.

    Returns
    -------
    jacobian_func : Callable
        A function that computes the Jacobian matrix for a given set of parameters.
        The function takes a 1D array of parameters as input and returns
        a 2D array representing the Jacobian matrix.

    Version
    -------
    - 0.0.1: Initial version.
    - 0.0.2: Renamed from build_jacobian to build_numerical_jacobian for clarity.
    - 1.0.0: Added support for forward, backward, and central finite difference methods for numerical Jacobian computation.
    - 1.1.0: Default step depending on the method (``epsilon=None``), representable steps,
      the residuals at ``p`` are computed only once (``"forward"`` / ``"backward"``) and
      not at all (``"central"``).
    """
    if not callable(residual_func):
        raise ValueError("The residual function must be callable.")
    method, epsilon = _validate_finite_difference(method, epsilon)

    def evaluate(parameters: numpy.ndarray) -> numpy.ndarray:
        residual = numpy.asarray(residual_func(parameters), dtype=numpy.float64)
        if residual.ndim != 1:
            raise ValueError(
                f"The residual function must return a 1D array, got shape {residual.shape}."
            )
        return residual

    def jacobian_func(parameters: numpy.ndarray) -> numpy.ndarray:
        parameters = numpy.array(parameters, dtype=numpy.float64)  # (copy)
        if parameters.ndim != 1:
            raise ValueError(
                f"The parameters must be a 1D array, got shape {parameters.shape}."
            )
        n_parameters = parameters.shape[0]
        steps = epsilon * numpy.maximum(1.0, numpy.abs(parameters))

        # Residuals at p: only needed by the one-sided differences (or to get the size)
        base = None
        if method != "central" or n_parameters == 0:
            base = evaluate(parameters)
        jacobian = None if base is None else numpy.empty((base.shape[0], n_parameters))

        for index in range(n_parameters):
            if method == "central":
                plus = parameters.copy()
                plus[index] += steps[index]
                minus = parameters.copy()
                minus[index] -= steps[index]
                difference = evaluate(plus) - evaluate(minus)
                step = plus[index] - minus[index]  # representable 2h
            elif method == "forward":
                plus = parameters.copy()
                plus[index] += steps[index]
                difference = evaluate(plus) - base
                step = plus[index] - parameters[index]
            else:  # backward
                minus = parameters.copy()
                minus[index] -= steps[index]
                difference = base - evaluate(minus)
                step = parameters[index] - minus[index]

            if jacobian is None:
                jacobian = numpy.empty((difference.shape[0], n_parameters))
            elif difference.shape[0] != jacobian.shape[0]:
                raise ValueError("The residual function must always return the same number of residuals.")
            jacobian[:, index] = difference / step

        return jacobian

    return jacobian_func


def build_batch_numerical_jacobian(
    residual_func: Callable,
    method: str = "central",
    epsilon: Optional[Real] = None,
) -> Callable:
    r"""
    Build a batched Jacobian function for the batched Gauss-Newton optimization by computing
    the numerical derivatives of a batched residual function with respect to the parameters
    using finite differences.

    This is the batched counterpart of :func:`pysolvegn.build_numerical_jacobian`: the ``m``
    problems of the batch are perturbed simultaneously, so the residual function is called
    ``n_parameters + 1`` times (``2 * n_parameters`` times for the ``"central"`` method)
    whatever the number of problems.

    For the problem :math:`k`, with the relative step :math:`h_{k,j} = \epsilon \max(1, |p_{k,j}|)`
    (the representable step :math:`(p_{k,j} + h_{k,j}) - p_{k,j}` is used),
    the ``"central"`` method computes:

    .. math::

        J_{k,i,j} = \frac{R_{k,i}(p_k + h_{k,j} e_j) - R_{k,i}(p_k - h_{k,j} e_j)}{2 h_{k,j}}

    With the ``"forward"`` method:

    .. math::

        J_{k,i,j} = \frac{R_{k,i}(p_k + h_{k,j} e_j) - R_{k,i}(p_k)}{h_{k,j}}

    With the ``"backward"`` method:

    .. math::

        J_{k,i,j} = \frac{R_{k,i}(p_k) - R_{k,i}(p_k - h_{k,j} e_j)}{h_{k,j}}


    Parameters
    ----------
    residual_func : Callable
        A batched function ``residual_func(parameters, indices)`` that computes the residuals.
        It takes the parameters of ``m`` independent problems as a 2D array with shape
        ``(m, n_parameters)`` (``m`` is any positive integer) and the indices of these
        problems in the full batch as a 1D array with shape ``(m,)``, and returns a 2D
        array of residuals with shape ``(m, n_residuals)``, where the row ``a`` of the
        residuals only depends on the row ``a`` of the parameters.

    method : str, optional (default="central")
        The finite difference method to use for computing the numerical Jacobian.
        Must be one of "central", "forward", or "backward".

    epsilon : Optional[Real], optional (default=None)
        The relative step :math:`\epsilon` of the finite differences. If None, the default
        step of the method is used (see :func:`pysolvegn.build_numerical_jacobian`).


    Returns
    -------
    jacobian_func : Callable
        A batched function ``jacobian_func(parameters, indices)`` that computes the Jacobian
        matrices of the ``m`` problems. It takes the parameters with shape
        ``(m, n_parameters)`` and the indices with shape ``(m,)``, and returns a 3D array with shape ``(m, n_residuals, n_parameters)``.

    """
    if not callable(residual_func):
        raise ValueError("The residual function must be callable.")
    method, epsilon = _validate_finite_difference(method, epsilon)

    def jacobian_func(parameters: numpy.ndarray, indices: numpy.ndarray) -> numpy.ndarray:
        parameters = numpy.array(parameters, dtype=numpy.float64)  # (copy)
        if parameters.ndim != 2:
            raise ValueError("The parameters must be a 2D array with shape (m, n_parameters).")
        n_problems, n_parameters = parameters.shape

        def evaluate(p: numpy.ndarray) -> numpy.ndarray:
            residual = numpy.asarray(residual_func(p, indices), dtype=numpy.float64)
            if residual.ndim != 2 or residual.shape[0] != n_problems:
                raise ValueError(
                    "The residual function must return a 2D array with shape (m, n_residuals), "
                    f"got shape {residual.shape}."
                )
            return residual

        steps = epsilon * numpy.maximum(1.0, numpy.abs(parameters))  # (m, n_parameters)

        base = None
        if method != "central" or n_parameters == 0:
            base = evaluate(parameters)
        jacobian = None if base is None else numpy.empty((n_problems, base.shape[1], n_parameters))

        for index in range(n_parameters):
            if method == "central":
                plus = parameters.copy()
                plus[:, index] += steps[:, index]
                minus = parameters.copy()
                minus[:, index] -= steps[:, index]
                difference = evaluate(plus) - evaluate(minus)
                step = plus[:, index] - minus[:, index]  # representable 2h, (m,)
            elif method == "forward":
                plus = parameters.copy()
                plus[:, index] += steps[:, index]
                difference = evaluate(plus) - base
                step = plus[:, index] - parameters[:, index]
            else:  # backward
                minus = parameters.copy()
                minus[:, index] -= steps[:, index]
                difference = base - evaluate(minus)
                step = parameters[:, index] - minus[:, index]

            if jacobian is None:
                jacobian = numpy.empty((n_problems, difference.shape[1], n_parameters))
            elif difference.shape[1] != jacobian.shape[1]:
                raise ValueError("The residual function must always return the same number of residuals.")
            jacobian[:, :, index] = difference / step[:, None]

        return jacobian

    return jacobian_func