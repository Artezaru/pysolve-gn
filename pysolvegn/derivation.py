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

from typing import Callable
from numbers import Real

import numpy


def build_numerical_jacobian(
    residual_func: Callable,
    method: str = "central",
    epsilon: Real = 1e-8,
) -> Callable:
    r"""
    Build a Jacobian function for the Gauss-Newton optimization by computing the numerical
    derivatives of the residual function with respect to the parameters using finite differences.

    With the ``"central"`` method, the Jacobian is computed as:

    .. math::

        J_{i,j} = \frac{R_i(p + \epsilon e_j) - R_i(p - \epsilon e_j)}{2\epsilon}

    With the ``"forward"`` method, the Jacobian is computed as:

    .. math::

        J_{i,j} = \frac{R_i(p + \epsilon e_j) - R_i(p)}{\epsilon}

    With the ``"backward"`` method, the Jacobian is computed as:

    .. math::

        J_{i,j} = \frac{R_i(p) - R_i(p - \epsilon e_j)}{\epsilon}


    Parameters
    ----------
    residual_func : Callable
        A function that computes the residuals for a given set of parameters.
        The function should take a 1D array of parameters as input and
        return a 1D array of residuals.

    method : str, optional (default="central")
        The finite difference method to use for computing the numerical Jacobian.
        Must be one of "central", "forward", or "backward".

    epsilon : Real, optional (default=1e-8)
        A small perturbation value used for finite difference approximation of the Jacobian.


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

    """
    if not callable(residual_func):
        raise ValueError("The residual function must be callable.")
    if not isinstance(epsilon, Real):
        raise ValueError("Epsilon must be a real number.")
    if epsilon <= 0:
        raise ValueError("Epsilon must be a positive number.")
    epsilon = float(epsilon)
    if not isinstance(method, str):
        raise ValueError("Method must be a string.")
    method = method.lower()
    if method not in ["central", "forward", "backward"]:
        raise ValueError("Method must be one of 'central', 'forward', or 'backward'.")

    def jacobian_func(parameters: numpy.ndarray) -> numpy.ndarray:
        parameters = numpy.asarray(parameters, dtype=numpy.float64)
        n_parameters = len(parameters)
        residual = residual_func(parameters)

        if not isinstance(residual, numpy.ndarray):
            raise ValueError("The residual function must return a numpy array.")
        if residual.ndim != 1:
            raise ValueError("The residual function must return a 1D array.")

        n_residual = len(residual)
        jacobian = numpy.zeros((n_residual, n_parameters), dtype=numpy.float64)
        perturbation = numpy.zeros((n_parameters,), dtype=numpy.float64)

        for index in range(n_parameters):
            perturbation[index] = epsilon * max(1.0, abs(parameters[index]))

            if method == "central":
                jacobian[:, index] = (
                    residual_func(parameters + perturbation)
                    - residual_func(parameters - perturbation)
                ) / (2 * perturbation[index])
            elif method == "forward":
                jacobian[:, index] = (
                    residual_func(parameters + perturbation) - residual_func(parameters)
                ) / perturbation[index]
            elif method == "backward":
                jacobian[:, index] = (
                    residual_func(parameters) - residual_func(parameters - perturbation)
                ) / perturbation[index]

            perturbation[index] = 0.0

        return jacobian

    return jacobian_func
