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
from numpy.typing import ArrayLike

import numpy


def build_squared_regularization(
    means: ArrayLike,
    stds: ArrayLike,
) -> Tuple[Callable, Callable]:
    r"""
    Build a simple squared regularization of the parameters according to a Gaussian
    prior on the parameters, where the residuals and jacobian of the regularization are defined as:

    .. math::

        R_{reg}(\lambda) = \frac{\lambda - \mu}{\sigma}

    .. math::

        J_{reg}(\lambda) = \frac{1}{\sigma}

    Then the callable functions returned by this function can be used as the
    residual and jacobian of a regularization term in the Gauss-Newton optimization,
    where the regularization term will be added to the residuals of the transformations,
    and the jacobian of the regularization will be added to the jacobian of the
    transformations.


    Parameters
    ----------
    means : ArrayLike
        A 1D array of mean values for each parameter. The length of the means array
        must be equal to the total number of parameters of all transformations.

    stds : ArrayLike
        A 1D array of standard deviation values for each parameter. The length of the std
        array must be equal to the total number of parameters of all transformations.


    Returns
    -------
    residual_func : Callable
        A function that computes the residuals of the regularization for a given set of parameters.

    jacobian_func : Callable
        A function that computes the Jacobian of the regularization for a given set of parameters.


    Version
    -------
    - 0.0.1: Initial version.

    """
    means = numpy.asarray(means, dtype=numpy.float64)
    stds = numpy.asarray(stds, dtype=numpy.float64)
    if means.ndim != 1:
        raise ValueError(f"means must be a 1D array, got {means.ndim} dimensions.")
    if stds.ndim != 1:
        raise ValueError(f"stds must be a 1D array, got {stds.ndim} dimensions.")
    if means.size != stds.size:
        raise ValueError(
            f"means and stds must have the same length, got {means.size} and {stds.size} respectively."
        )

    def residual_func(params: numpy.ndarray) -> numpy.ndarray:
        return (params - means) / stds

    def jacobian_func(params: numpy.ndarray) -> numpy.ndarray:
        jacobian = numpy.zeros((len(params), len(params)), dtype=numpy.float64)
        jacobian[:, :] = numpy.diag(1 / stds)
        return jacobian

    return residual_func, jacobian_func


def build_soft_squared_regularization(
    means: ArrayLike,
    thresholds: ArrayLike,
    stds: ArrayLike,
) -> Tuple[Callable, Callable]:
    r"""
    Build the residual and jacobian functions of a soft squared regularization of 
    the parameters according to a Gaussian prior on the parameters with a soft threshold, 
    where the regularization is null for parameters that are within a certain 
    threshold of the mean value, and increases as the squared distance of the 
    parameters to a bounds value relative to a standard deviation, which can be 
    interpreted as a Gaussian prior on the parameters with a soft threshold.

    This regularisation is independant for each parameter and is null for parameters that
    are within a certain threshold of the mean value, and increases as the squared distance
    of the parameters to a bounds value relative to a standard deviation,
    which can be interpreted as a Gaussian prior on the parameters with a soft threshold.

    .. math::

        \text{Reg}(\lambda) = \left(\frac{\lambda - \tau_{-}}{\sigma}\right)^2 \quad \text{if} \quad \lambda < \mu - \tau = \tau_{-}

    .. math::

        \text{Reg}(\lambda) = 0 \quad \text{if} \quad |\lambda - \mu| \leq \tau

    .. math::

        \text{Reg}(\lambda) = \left(\frac{\lambda - \tau_{+}}{\sigma}\right)^2 \quad \text{if} \quad \lambda > \mu + \tau = \tau_{+}

    Thus the residuals and jacobian of the regularization are defined as:
    
    .. math::

        R_{reg}(\lambda) = \begin{cases}
            \frac{\lambda - \tau_{-}}{\sigma} & \text{if} \quad \lambda < \mu - \tau = \tau_{-} \\
            0 & \text{if} \quad |\lambda - \mu| \leq \tau \\
            \frac{\lambda - \tau_{+}}{\sigma} & \text{if} \quad \lambda > \mu + \tau = \tau_{+}
        \end{cases}
        
    .. math::
    
        J_{reg}(\lambda) = \begin{cases}
            \frac{1}{\sigma} & \text{if} \quad \lambda < \mu - \tau = \tau_{-} \\
            0 & \text{if} \quad |\lambda - \mu| \leq \tau \\
            \frac{1}{\sigma} & \text{if} \quad \lambda > \mu + \tau = \tau_{+}
        \end{cases}
        
    Then the callable functions returned by this function can be used as the 
    residual and jacobian of a regularization term in the Gauss-Newton optimization,
    where the regularization term will be added to the residuals of the transformations,
    and the jacobian of the regularization will be added to the jacobian of the 
    transformations.
        
        
    Parameters
    ----------
    means : ArrayLike
        A 1D array of mean values for each parameter. The length of the means array
        must be equal to the total number of parameters of all transformations.
        
    thresholds : ArrayLike
        A 1D array of threshold values for each parameter. The length of the thresholds array
        must be equal to the total number of parameters of all transformations.
        
    stds : ArrayLike
        A 1D array of standard deviation values for each parameter. The length of the std
        array must be equal to the total number of parameters of all transformations.

        
    Returns
    -------
    residual_func : Callable
        A function that computes the residuals of the regularization for a given set of parameters.
        
    jacobian_func : Callable
        A function that computes the Jacobian of the regularization for a given set of parameters.
           
           
    Version
    -------
    - 0.0.1: Initial version.

    """
    means = numpy.asarray(means, dtype=numpy.float64)
    thresholds = numpy.asarray(thresholds, dtype=numpy.float64)
    stds = numpy.asarray(stds, dtype=numpy.float64)
    if means.ndim != 1:
        raise ValueError(f"means must be a 1D array, got {means.ndim} dimensions.")
    if thresholds.ndim != 1:
        raise ValueError(
            f"thresholds must be a 1D array, got {thresholds.ndim} dimensions."
        )
    if stds.ndim != 1:
        raise ValueError(f"stds must be a 1D array, got {stds.ndim} dimensions.")
    if means.size != thresholds.size or means.size != stds.size:
        raise ValueError(
            f"means, thresholds and stds must have the same length, got {means.size}, {thresholds.size} and {stds.size} respectively."
        )

    def residual_func(params: numpy.ndarray) -> numpy.ndarray:
        residuals = numpy.zeros(len(params), dtype=numpy.float64)
        current_index = 0
        for i in range(len(params)):
            if params[i] < means[i] - thresholds[i]:
                residuals[current_index] = (
                    params[i] - (means[i] - thresholds[i])
                ) / stds[i]
            elif params[i] > means[i] + thresholds[i]:
                residuals[current_index] = (
                    params[i] - (means[i] + thresholds[i])
                ) / stds[i]
            else:
                residuals[current_index] = 0.0
            current_index += 1
        return residuals

    def jacobian_func(params: numpy.ndarray) -> numpy.ndarray:
        jacobian = numpy.zeros((len(params), len(params)), dtype=numpy.float64)
        current_index = 0
        for i in range(len(params)):
            if (
                params[i] < means[i] - thresholds[i]
                or params[i] > means[i] + thresholds[i]
            ):
                jacobian[current_index, i] = 1 / stds[i]
            else:
                jacobian[current_index, i] = 0.0
            current_index += 1
        return jacobian

    return residual_func, jacobian_func
