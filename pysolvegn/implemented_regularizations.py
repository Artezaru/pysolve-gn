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

from typing import Callable, Dict, Optional, Union
from numbers import Real

import numpy
from numpy.typing import ArrayLike

from .term import Term
from .batch_term import BatchTerm


# ======================================================================
# Validation helpers
# ======================================================================


def _validate_arrays(
    arrays: Dict[str, ArrayLike],
    *,
    batch: bool,
) -> Dict[str, numpy.ndarray]:
    r"""
    Convert the regularization arrays to float arrays and check their shapes.

    - single mode (``batch=False``): every array must be 1D with shape ``(n,)``.
    - batch mode (``batch=True``): every array must be 1D with shape ``(n,)`` (shared by
      all the problems) or 2D with shape ``(k, n)`` (one row per problem). All the 2D
      arrays must have the same number of problems ``k``.

    All the arrays must have the same number of parameters ``n`` and finite values.
    """
    converted = {name: numpy.asarray(array, dtype=numpy.float64) for name, array in arrays.items()}

    allowed_ndim = (1, 2) if batch else (1,)
    for name, array in converted.items():
        if array.ndim not in allowed_ndim:
            expected = "a 1D array (n,) or a 2D array (k, n)" if batch else "a 1D array"
            raise ValueError(f"{name} must be {expected}, got {array.ndim} dimensions.")
        if numpy.any(~numpy.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values.")

    sizes = {name: array.shape[-1] for name, array in converted.items()}
    if len(set(sizes.values())) != 1:
        raise ValueError(
            "All the arrays must have the same number of parameters, got "
            + ", ".join(f"{name}: {size}" for name, size in sizes.items())
            + "."
        )

    n_problems = {name: array.shape[0] for name, array in converted.items() if array.ndim == 2}
    if len(set(n_problems.values())) > 1:
        raise ValueError(
            "All the 2D arrays must have the same number of problems, got "
            + ", ".join(f"{name}: {k}" for name, k in n_problems.items())
            + "."
        )

    return converted


def _validate_positive_real(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    value = float(value)
    if not numpy.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite strictly positive number.")
    return value


def _check_params(params: numpy.ndarray, n: int) -> numpy.ndarray:
    # Single mode: params with shape (n,)
    params = numpy.asarray(params, dtype=numpy.float64)
    if params.ndim != 1:
        raise ValueError(f"params must be a 1D array, got {params.ndim} dimensions.")
    if params.size != n:
        raise ValueError(f"params must have size {n}, got {params.size}.")
    return params


def _check_batch_params(params: numpy.ndarray, indices: numpy.ndarray, n: int) -> numpy.ndarray:
    # Batch mode: params with shape (m, n) and indices with shape (m,)
    params = numpy.asarray(params, dtype=numpy.float64)
    if params.ndim != 2 or params.shape[1] != n:
        raise ValueError(f"params must be a 2D array with shape (m, {n}), got shape {params.shape}.")
    if numpy.shape(indices) != (params.shape[0],):
        raise ValueError(f"indices must have shape ({params.shape[0]},), got {numpy.shape(indices)}.")
    return params


def _select(array: numpy.ndarray, indices: numpy.ndarray) -> numpy.ndarray:
    # Values of the problems `indices`: shared (n,) array or per-problem (k, n) array
    return array[None, :] if array.ndim == 1 else array[indices]


def _batch_diagonal(diagonal: numpy.ndarray) -> numpy.ndarray:
    # Stack of diagonal matrices (m, n, n) from the diagonals (m, n)
    m, n = diagonal.shape
    jacobian = numpy.zeros((m, n, n), dtype=numpy.float64)
    index = numpy.arange(n)
    jacobian[:, index, index] = diagonal
    return jacobian


# ======================================================================
# Squared regularization
# ======================================================================


def build_squared_regularization(
    means: ArrayLike,
    stds: ArrayLike,
    *,
    weight: Real = 1.0,
    loss: Optional[Union[str, Callable]] = None,
    loss_scale: Real = 1.0,
) -> Term:
    r"""
    Build a squared regularization term based on a Gaussian prior.

    The regularization residuals are defined as:

    .. math::

        r_{\mathrm{reg},i}(\mathbf{p})
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
    :func:`pysolvegn.solve`.

    .. seealso::

        :func:`build_batch_squared_regularization` for the batched version.

    Parameters
    ----------
    means : ArrayLike
        Mean values of the Gaussian prior for each parameter.

    stds : ArrayLike
        Standard deviation values of the Gaussian prior for each parameter.
        All values must be strictly positive.

    weight : Real (default=1.0)
        Weight of the regularization term.

    loss : Optional[Union[str, Callable]] (default=None)
        Loss function applied to the regularization residuals.

    loss_scale : Real (default=1.0)
        Soft threshold of the loss function, in the unit of the residuals
        (number of standard deviations), see :class:`pysolvegn.Term`.

    Returns
    -------
    Term
        A :class:`Term` representing the squared Gaussian regularization.
    """
    arrays = _validate_arrays({"means": means, "stds": stds}, batch=False)
    means, stds = arrays["means"], arrays["stds"]

    if numpy.any(stds <= 0):
        raise ValueError("stds must contain only strictly positive values.")

    n = means.size

    def residual_func(params: numpy.ndarray) -> numpy.ndarray:
        params = _check_params(params, n)
        return (params - means) / stds

    def jacobian_func(params: numpy.ndarray) -> numpy.ndarray:
        _check_params(params, n)
        return numpy.diag(1.0 / stds)

    return Term(
        residual_func=residual_func,
        jacobian_func=jacobian_func,
        weight=weight,
        loss=loss,
        loss_scale=loss_scale,
    )


def build_batch_squared_regularization(
    means: ArrayLike,
    stds: ArrayLike,
    *,
    weight: Union[Real, ArrayLike] = 1.0,
    loss: Optional[Union[str, Callable]] = None,
    loss_scale: Real = 1.0,
) -> BatchTerm:
    r"""
    Build a batched squared regularization term based on a Gaussian prior
    (batched counterpart of :func:`build_squared_regularization`).

    For each problem :math:`k` of the batch:

    .. math::

        r_{\mathrm{reg},k,i}(\mathbf{p}_k)
        =
        \frac{p_{k,i} - \mu_{k,i}}{\sigma_{k,i}}

    The resulting :class:`BatchTerm` can be directly passed to
    :func:`pysolvegn.solve_batch`.

    Parameters
    ----------
    means : ArrayLike
        Mean values of the Gaussian prior, with shape ``(n,)`` (same prior for all the
        problems) or ``(k, n)`` (one prior per problem).

    stds : ArrayLike
        Standard deviation values of the Gaussian prior, with shape ``(n,)`` or
        ``(k, n)``. All values must be strictly positive.

    weight : Union[Real, ArrayLike] (default=1.0)
        Weight of the regularization term: a scalar or one weight per problem with
        shape ``(k,)`` (see :class:`pysolvegn.BatchTerm`).

    loss : Optional[Union[str, Callable]] (default=None)
        Loss function applied to the regularization residuals.

    loss_scale : Real (default=1.0)
        Soft threshold of the loss function, in the unit of the residuals
        (number of standard deviations), see :class:`pysolvegn.BatchTerm`.

    Returns
    -------
    BatchTerm
        A :class:`BatchTerm` representing the squared Gaussian regularization.
    """
    arrays = _validate_arrays({"means": means, "stds": stds}, batch=True)
    means, stds = arrays["means"], arrays["stds"]

    if numpy.any(stds <= 0):
        raise ValueError("stds must contain only strictly positive values.")

    n = means.shape[-1]

    def residual_func(params: numpy.ndarray, indices: numpy.ndarray) -> numpy.ndarray:
        params = _check_batch_params(params, indices, n)
        return (params - _select(means, indices)) / _select(stds, indices)

    def jacobian_func(params: numpy.ndarray, indices: numpy.ndarray) -> numpy.ndarray:
        params = _check_batch_params(params, indices, n)
        inverse_stds = numpy.broadcast_to(1.0 / _select(stds, indices), params.shape)
        return _batch_diagonal(inverse_stds)

    return BatchTerm(
        residual_func=residual_func,
        jacobian_func=jacobian_func,
        weight=weight,
        loss=loss,
        loss_scale=loss_scale,
    )


# ======================================================================
# Soft squared regularization
# ======================================================================


def build_soft_squared_regularization(
    means: ArrayLike,
    thresholds: ArrayLike,
    stds: ArrayLike,
    *,
    weight: Real = 1.0,
    loss: Optional[Union[str, Callable]] = None,
    loss_scale: Real = 1.0,
) -> Term:
    r"""
    Build a soft squared regularization term based on a Gaussian prior.

    The regularization is null inside a symmetric threshold around the
    mean and increases quadratically outside this interval.

    For each parameter:

    .. math::

        r_{\mathrm{reg},i}(\mathbf{p}) =
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
    :func:`pysolvegn.solve`.

    .. seealso::

        :func:`build_batch_soft_squared_regularization` for the batched version.

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

    weight : Real (default=1.0)
        Weight of the regularization term.

    loss : Optional[Union[str, Callable]] (default=None)
        Loss function applied to the regularization residuals.

    loss_scale : Real (default=1.0)
        Soft threshold of the loss function, in the unit of the residuals
        (number of standard deviations), see :class:`pysolvegn.Term`.

    Returns
    -------
    Term
        A :class:`Term` representing the soft squared regularization.
    """
    arrays = _validate_arrays(
        {"means": means, "thresholds": thresholds, "stds": stds}, batch=False
    )
    means, thresholds, stds = arrays["means"], arrays["thresholds"], arrays["stds"]

    if numpy.any(thresholds < 0):
        raise ValueError("thresholds must contain only non-negative values.")

    if numpy.any(stds <= 0):
        raise ValueError("stds must contain only strictly positive values.")

    n = means.size
    lower_bounds = means - thresholds
    upper_bounds = means + thresholds

    def residual_func(params: numpy.ndarray) -> numpy.ndarray:
        params = _check_params(params, n)
        below = numpy.minimum(params - lower_bounds, 0.0)  # < 0 if p < mu - tau
        above = numpy.maximum(params - upper_bounds, 0.0)  # > 0 if p > mu + tau
        return (below + above) / stds

    def jacobian_func(params: numpy.ndarray) -> numpy.ndarray:
        params = _check_params(params, n)
        active = (params < lower_bounds) | (params > upper_bounds)
        return numpy.diag(numpy.where(active, 1.0 / stds, 0.0))

    return Term(
        residual_func=residual_func,
        jacobian_func=jacobian_func,
        weight=weight,
        loss=loss,
        loss_scale=loss_scale,
    )


def build_batch_soft_squared_regularization(
    means: ArrayLike,
    thresholds: ArrayLike,
    stds: ArrayLike,
    *,
    weight: Union[Real, ArrayLike] = 1.0,
    loss: Optional[Union[str, Callable]] = None,
    loss_scale: Real = 1.0,
) -> BatchTerm:
    r"""
    Build a batched soft squared regularization term (batched counterpart of
    :func:`build_soft_squared_regularization`).

    For each problem :math:`k` of the batch, the regularization is null inside the
    interval :math:`[\mu_{k,i} - \tau_{k,i}, \mu_{k,i} + \tau_{k,i}]` and increases
    quadratically outside this interval, with the strength :math:`1 / \sigma_{k,i}`.

    The resulting :class:`BatchTerm` can be directly passed to
    :func:`pysolvegn.solve_batch`.

    Parameters
    ----------
    means : ArrayLike
        Mean values, with shape ``(n,)`` (shared by all the problems) or ``(k, n)``.

    thresholds : ArrayLike
        Threshold values around the corresponding means, with shape ``(n,)`` or
        ``(k, n)``. All values must be non-negative.

    stds : ArrayLike
        Standard deviation values controlling the strength of the regularization
        outside the threshold, with shape ``(n,)`` or ``(k, n)``. All values must be
        strictly positive.

    weight : Union[Real, ArrayLike] (default=1.0)
        Weight of the regularization term: a scalar or one weight per problem with
        shape ``(k,)`` (see :class:`pysolvegn.BatchTerm`).

    loss : Optional[Union[str, Callable]] (default=None)
        Loss function applied to the regularization residuals.

    loss_scale : Real (default=1.0)
        Soft threshold of the loss function, in the unit of the residuals
        (number of standard deviations), see :class:`pysolvegn.BatchTerm`.

    Returns
    -------
    BatchTerm
        A :class:`BatchTerm` representing the soft squared regularization.
    """
    arrays = _validate_arrays(
        {"means": means, "thresholds": thresholds, "stds": stds}, batch=True
    )
    means, thresholds, stds = arrays["means"], arrays["thresholds"], arrays["stds"]

    if numpy.any(thresholds < 0):
        raise ValueError("thresholds must contain only non-negative values.")

    if numpy.any(stds <= 0):
        raise ValueError("stds must contain only strictly positive values.")

    n = means.shape[-1]
    lower_bounds = means - thresholds  # (n,) or (k, n) by broadcasting
    upper_bounds = means + thresholds

    def residual_func(params: numpy.ndarray, indices: numpy.ndarray) -> numpy.ndarray:
        params = _check_batch_params(params, indices, n)
        below = numpy.minimum(params - _select(lower_bounds, indices), 0.0)
        above = numpy.maximum(params - _select(upper_bounds, indices), 0.0)
        return (below + above) / _select(stds, indices)

    def jacobian_func(params: numpy.ndarray, indices: numpy.ndarray) -> numpy.ndarray:
        params = _check_batch_params(params, indices, n)
        active = (params < _select(lower_bounds, indices)) | (params > _select(upper_bounds, indices))
        inverse_stds = numpy.broadcast_to(1.0 / _select(stds, indices), params.shape)
        return _batch_diagonal(numpy.where(active, inverse_stds, 0.0))

    return BatchTerm(
        residual_func=residual_func,
        jacobian_func=jacobian_func,
        weight=weight,
        loss=loss,
        loss_scale=loss_scale,
    )


# ======================================================================
# Absolute (L1) regularization
# ======================================================================


def build_absolute_regularization(
    means: ArrayLike,
    stds: ArrayLike,
    *,
    weight: Real = 1.0,
    epsilon: Real = 1e-3,
) -> Term:
    r"""
    Build an absolute value (L1) regularization term, approximated by a smooth
    Charbonnier (pseudo-Huber) function.

    The cost of the term is, for each parameter:

    .. math::

        C_{\mathrm{reg},i}(\mathbf{p})
        =
        w \left( \sqrt{z_i^2 + \varepsilon^2} - \varepsilon \right)
        \quad \text{with} \quad
        z_i = \frac{p_i - \mu_i}{\sigma_i}

    which behaves as:

    .. math::

        C_{\mathrm{reg},i} \approx w \left( |z_i| - \varepsilon \right)
        \quad \text{if } |z_i| \gg \varepsilon
        \qquad
        C_{\mathrm{reg},i} \approx \frac{w}{2 \varepsilon} z_i^2
        \quad \text{if } |z_i| \ll \varepsilon

    The L1 penalty :math:`w |z_i|` favors sparse solutions (many parameters exactly at
    their mean), unlike the squared regularization which only shrinks them. The smooth
    quadratic zone of width :math:`\varepsilon` around :math:`\mu_i` keeps the
    Gauss-Newton system well defined.

    It is implemented as a :class:`Term` with the residuals :math:`z_i`, the
    ``"soft_l1"`` loss function with ``loss_scale=epsilon`` and the weight
    :math:`w / \varepsilon`:

    .. math::

        \frac{1}{2} \frac{w}{\varepsilon} \rho_{\varepsilon}(z_i^2)
        = \frac{w}{\varepsilon} \varepsilon^2 \left( \sqrt{1 + z_i^2 / \varepsilon^2} - 1 \right)
        = w \left( \sqrt{z_i^2 + \varepsilon^2} - \varepsilon \right)

    .. note::

        Using the residual :math:`\sqrt{|z_i|}` (such that :math:`\frac{1}{2} r_i^2 = \frac{1}{2}|z_i|`)
        does not work with the Gauss-Newton method: its Jacobian is infinite at
        :math:`z_i = 0`, and the Gauss-Newton step is :math:`\Delta z_i = -2 z_i`, so the
        parameters oscillate between :math:`z_i` and :math:`-z_i` without converging.

    .. tip::

        A small ``epsilon`` gives a sharper L1 penalty but a stiffer problem: use
        ``damping="lm-diag"`` in :func:`pysolvegn.solve`, and a tolerance on the
        parameters larger than ``epsilon * stds``.

    The resulting :class:`Term` can be directly passed to
    :func:`pysolvegn.solve`.

    .. seealso::

        :func:`build_batch_absolute_regularization` for the batched version.

    Parameters
    ----------
    means : ArrayLike
        Center of the regularization for each parameter.

    stds : ArrayLike
        Scale of the regularization for each parameter (the penalty is
        :math:`w |p_i - \mu_i| / \sigma_i`). All values must be strictly positive.

    weight : Real (default=1.0)
        Weight :math:`w` of the L1 penalty.

    epsilon : Real (default=1e-3)
        Width :math:`\varepsilon` of the smooth quadratic zone, in the unit of
        :math:`z_i` (number of ``stds``). Must be strictly positive.

    Returns
    -------
    Term
        A :class:`Term` representing the absolute value regularization.

    """
    arrays = _validate_arrays({"means": means, "stds": stds}, batch=False)
    means, stds = arrays["means"], arrays["stds"]

    if numpy.any(stds <= 0):
        raise ValueError("stds must contain only strictly positive values.")

    weight = _validate_positive_real("weight", weight)
    epsilon = _validate_positive_real("epsilon", epsilon)

    term = build_squared_regularization(means, stds, weight=1.0)
    return Term(
        residual_func=term.residual_func,
        jacobian_func=term.jacobian_func,
        weight=weight / epsilon,
        loss="soft_l1",
        loss_scale=epsilon,
    )


def build_batch_absolute_regularization(
    means: ArrayLike,
    stds: ArrayLike,
    *,
    weight: Union[Real, ArrayLike] = 1.0,
    epsilon: Real = 1e-3,
) -> BatchTerm:
    r"""
    Build a batched absolute value (L1) regularization term (batched counterpart of
    :func:`build_absolute_regularization`).

    For each problem :math:`k` of the batch, the cost of the term is:

    .. math::

        C_{\mathrm{reg},k} = w_k \sum_i \left( \sqrt{z_{k,i}^2 + \varepsilon^2} - \varepsilon \right)
        \quad \text{with} \quad
        z_{k,i} = \frac{p_{k,i} - \mu_{k,i}}{\sigma_{k,i}}

    The resulting :class:`BatchTerm` can be directly passed to
    :func:`pysolvegn.solve_batch`.

    Parameters
    ----------
    means : ArrayLike
        Center of the regularization, with shape ``(n,)`` (shared by all the problems)
        or ``(k, n)``.

    stds : ArrayLike
        Scale of the regularization, with shape ``(n,)`` or ``(k, n)``.
        All values must be strictly positive.

    weight : Union[Real, ArrayLike] (default=1.0)
        Weight :math:`w` of the L1 penalty: a scalar or one weight per problem with
        shape ``(k,)``.

    epsilon : Real (default=1e-3)
        Width :math:`\varepsilon` of the smooth quadratic zone, in the unit of
        :math:`z_{k,i}` (number of ``stds``). Must be strictly positive.

    Returns
    -------
    BatchTerm
        A :class:`BatchTerm` representing the absolute value regularization.
    """
    epsilon = _validate_positive_real("epsilon", epsilon)
    if isinstance(weight, Real):
        weight = _validate_positive_real("weight", weight)
    else:
        weight = numpy.asarray(weight, dtype=numpy.float64)  # (k,) checked by BatchTerm

    term = build_batch_squared_regularization(means, stds, weight=1.0)
    return BatchTerm(
        residual_func=term.residual_func,
        jacobian_func=term.jacobian_func,
        weight=weight / epsilon,
        loss="soft_l1",
        loss_scale=epsilon,
    )