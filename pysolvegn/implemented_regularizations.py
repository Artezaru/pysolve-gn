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

from numbers import Real

import numpy
from numpy.typing import ArrayLike

from .term import Term


def build_squared_regularization(
    means: ArrayLike,
    stds: ArrayLike,
    *,
    weight: Real = 1.0,
    loss: str = "linear",
    finite_difference: str = "central",
) -> Term:
    r"""
    Build a squared regularization term based on a Gaussian prior.

    The regularization residuals are defined as:

    .. math::

        R_{\mathrm{reg},i}(\mathbf{p})
        =
        \frac{p_i - \mu_i}{\sigma_i}

    with Jacobian:

    .. math::

        J_{\mathrm{reg},ij}
        =
        \begin{cases}
            \frac{1}{\sigma_i} & \text{if } i=j \\
            0 & \text{otherwise}.
        \end{cases}

    The resulting :class:`Term` can be directly passed to
    :func:`pysolve_gn.solve`.

    Parameters
    ----------
    means : ArrayLike
        Mean values of the Gaussian prior for each parameter.

    stds : ArrayLike
        Standard deviation values of the Gaussian prior for each parameter.
        All values must be strictly positive.

    weight : Real, optional (default=1.0)
        Weight of the regularization term.

    loss : str, optional (default="linear")
        Loss function applied to the regularization residuals.

    finite_difference : str, optional (default="central")
        Finite difference method used by :class:`Term` if the Jacobian
        is not explicitly available.

    Returns
    -------
    Term
        A :class:`Term` representing the squared Gaussian regularization.
    """
    means = numpy.asarray(means, dtype=numpy.float64)
    stds = numpy.asarray(stds, dtype=numpy.float64)

    if means.ndim != 1:
        raise ValueError(f"means must be a 1D array, got {means.ndim} dimensions.")

    if stds.ndim != 1:
        raise ValueError(f"stds must be a 1D array, got {stds.ndim} dimensions.")

    if means.size != stds.size:
        raise ValueError(
            f"means and stds must have the same length, "
            f"got {means.size} and {stds.size} respectively."
        )

    if numpy.any(~numpy.isfinite(means)):
        raise ValueError("means must contain only finite values.")

    if numpy.any(~numpy.isfinite(stds)):
        raise ValueError("stds must contain only finite values.")

    if numpy.any(stds <= 0):
        raise ValueError("stds must contain only strictly positive values.")

    def residual_func(params: numpy.ndarray) -> numpy.ndarray:
        params = numpy.asarray(params, dtype=numpy.float64)

        if params.ndim != 1:
            raise ValueError(
                f"params must be a 1D array, got {params.ndim} dimensions."
            )

        if params.size != means.size:
            raise ValueError(f"params must have size {means.size}, got {params.size}.")

        return (params - means) / stds

    def jacobian_func(params: numpy.ndarray) -> numpy.ndarray:
        params = numpy.asarray(params, dtype=numpy.float64)

        if params.ndim != 1:
            raise ValueError(
                f"params must be a 1D array, got {params.ndim} dimensions."
            )

        if params.size != means.size:
            raise ValueError(f"params must have size {means.size}, got {params.size}.")

        return numpy.diag(1.0 / stds)

    return Term(
        residual_func=residual_func,
        jacobian_func=jacobian_func,
        weight=weight,
        loss=loss,
        finite_difference=finite_difference,
    )


def build_soft_squared_regularization(
    means: ArrayLike,
    thresholds: ArrayLike,
    stds: ArrayLike,
    *,
    weight: Real = 1.0,
    loss: str = "linear",
    finite_difference: str = "central",
) -> Term:
    r"""
    Build a soft squared regularization term based on a Gaussian prior.

    The regularization is null inside a symmetric threshold around the
    mean and increases quadratically outside this interval.

    For each parameter:

    .. math::

        R_{\mathrm{reg},i}(\mathbf{p}) =
        \begin{cases}
            \dfrac{p_i - (\mu_i-\tau_i)}{\sigma_i}
            & \text{if } p_i < \mu_i-\tau_i \\[6pt]
            0
            & \text{if } |p_i-\mu_i| \leq \tau_i \\[6pt]
            \dfrac{p_i - (\mu_i+\tau_i)}{\sigma_i}
            & \text{if } p_i > \mu_i+\tau_i.
        \end{cases}

    Its Jacobian is:

    .. math::

        J_{\mathrm{reg},ij} =
        \begin{cases}
            \dfrac{1}{\sigma_i}
            & \text{if } i=j \text{ and } |p_i-\mu_i|>\tau_i \\[6pt]
            0
            & \text{otherwise}.
        \end{cases}

    The resulting :class:`Term` can be directly passed to
    :func:`pysolve_gn.solve`.

    Parameters
    ----------
    means : ArrayLike
        Mean values for each parameter.

    thresholds : ArrayLike
        Threshold values around the corresponding means.
        All values must be non-negative.

    stds : ArrayLike
        Standard deviation values controlling the strength of the
        regularization outside the threshold.
        All values must be strictly positive.

    weight : Real, optional (default=1.0)
        Weight of the regularization term.

    loss : str, optional (default="linear")
        Loss function applied to the regularization residuals.

    finite_difference : str, optional (default="central")
        Finite difference method used by :class:`Term` if the Jacobian
        is not explicitly available.

    Returns
    -------
    Term
        A :class:`Term` representing the soft squared regularization.
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

    if not (means.size == thresholds.size == stds.size):
        raise ValueError(
            f"means, thresholds and stds must have the same length, "
            f"got {means.size}, {thresholds.size} and {stds.size} respectively."
        )

    if numpy.any(~numpy.isfinite(means)):
        raise ValueError("means must contain only finite values.")

    if numpy.any(~numpy.isfinite(thresholds)):
        raise ValueError("thresholds must contain only finite values.")

    if numpy.any(~numpy.isfinite(stds)):
        raise ValueError("stds must contain only finite values.")

    if numpy.any(thresholds < 0):
        raise ValueError("thresholds must contain only non-negative values.")

    if numpy.any(stds <= 0):
        raise ValueError("stds must contain only strictly positive values.")

    lower_bounds = means - thresholds
    upper_bounds = means + thresholds

    def residual_func(params: numpy.ndarray) -> numpy.ndarray:
        params = numpy.asarray(params, dtype=numpy.float64)

        if params.ndim != 1:
            raise ValueError(
                f"params must be a 1D array, got {params.ndim} dimensions."
            )

        if params.size != means.size:
            raise ValueError(f"params must have size {means.size}, got {params.size}.")

        residuals = numpy.zeros_like(params)

        lower_mask = params < lower_bounds
        upper_mask = params > upper_bounds

        residuals[lower_mask] = (params[lower_mask] - lower_bounds[lower_mask]) / stds[
            lower_mask
        ]

        residuals[upper_mask] = (params[upper_mask] - upper_bounds[upper_mask]) / stds[
            upper_mask
        ]

        return residuals

    def jacobian_func(params: numpy.ndarray) -> numpy.ndarray:
        params = numpy.asarray(params, dtype=numpy.float64)

        if params.ndim != 1:
            raise ValueError(
                f"params must be a 1D array, got {params.ndim} dimensions."
            )

        if params.size != means.size:
            raise ValueError(f"params must have size {means.size}, got {params.size}.")

        active = (params < lower_bounds) | (params > upper_bounds)

        return numpy.diag(numpy.where(active, 1.0 / stds, 0.0))

    return Term(
        residual_func=residual_func,
        jacobian_func=jacobian_func,
        weight=weight,
        loss=loss,
        finite_difference=finite_difference,
    )
