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


def build_jacobian(
    residual_func: Callable,
    epsilon: Real = 1e-8,
) -> Callable:
    r"""
    Build a Jacobian function for the Gauss-Newton optimization by computing the numerical
    derivatives of the residual function with respect to the parameters using finite differences.

    Parameters
    ----------
    residual_func : Callable
        A function that computes the residuals for a given set of parameters. The function should
        take a 1D array of parameters as input and return a 1D array of residuals.

    epsilon : Real, optional
        A small perturbation value used for finite difference approximation of the Jacobian.
        Default is 1e-8.


    Returns
    -------
    jacobian_func : Callable
        A function that computes the Jacobian matrix for a given set of parameters. The function should
        take a 1D array of parameters as input and return a 2D array representing the Jacobian matrix.


    Version
    -------
    1.0.0: Initial version.

    """
    if not callable(residual_func):
        raise ValueError("The residual function must be callable.")
    if not isinstance(epsilon, Real):
        raise ValueError("Epsilon must be a real number.")
    if epsilon <= 0:
        raise ValueError("Epsilon must be a positive number.")
    epsilon = float(epsilon)

    def jacobian_func(params: numpy.ndarray) -> numpy.ndarray:
        params = numpy.asarray(params, dtype=numpy.float64)
        n_params = len(params)
        residuals = residual_func(params)
        n_residuals = len(residuals)
        jacobian = numpy.zeros((n_residuals, n_params), dtype=numpy.float64)

        for i in range(n_params):
            perturbation = numpy.zeros_like(params)
            perturbation[i] = epsilon
            jacobian[:, i] = (
                residual_func(params + perturbation) - residuals
            ) / epsilon

        return jacobian

    return jacobian_func
