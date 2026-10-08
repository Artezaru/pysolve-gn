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

from typing import Optional, Sequence
from numpy.typing import ArrayLike

import numpy

from .parametrization import Parametrization
from .batch_parametrization import BatchParametrization


def _sigmoid(x: numpy.ndarray) -> numpy.ndarray:
    # Numerically stable sigmoid (any shape).
    result = numpy.empty_like(x)

    positive = x >= 0.0
    result[positive] = 1.0 / (1.0 + numpy.exp(-x[positive]))

    exp_x = numpy.exp(x[~positive])
    result[~positive] = exp_x / (1.0 + exp_x)

    return result


def _check_batch_p_in(p_in: ArrayLike, n_parameters: int) -> numpy.ndarray:
    # Batch mode: input parameters of m problems with shape (m, n_parameters)
    p_in = numpy.asarray(p_in, dtype=numpy.float64)

    if p_in.ndim != 2:
        raise ValueError(f"p_in must be a 2D array, got {p_in.ndim} dimensions.")

    if p_in.shape[1] != n_parameters:
        raise ValueError(
            f"p_in must have shape (m, {n_parameters}), " f"got {p_in.shape}."
        )

    return p_in


def _batch_diagonal(diagonal: numpy.ndarray) -> numpy.ndarray:
    # Stack of diagonal matrices (m, n, n) from the diagonals (m, n)
    m, n = diagonal.shape
    jacobian = numpy.zeros((m, n, n), dtype=numpy.float64)
    index = numpy.arange(n)
    jacobian[:, index, index] = diagonal
    return jacobian


def build_affine_parametrization(
    modes: ArrayLike,
    offset: Optional[ArrayLike] = None,
) -> Parametrization:
    r"""
    Build a :class:`Parametrization` based on an affine transformation.

    The parametrization maps the optimized input parameters
    :math:`\mathbf{p}_{in}` to the output parameters passed to the terms
    according to:

    .. math::

        \mathbf{p}_{out}
        =
        M \mathbf{p}_{in}
        +
        \mathbf{p}_0

    where :math:`M` is the transformation matrix and
    :math:`\mathbf{p}_0` is an optional offset.

    The corresponding Jacobian is constant:

    .. math::

        \mathbf{J}_P
        =
        \frac{\partial \mathbf{p}_{out}}
             {\partial \mathbf{p}_{in}}
        =
        M.

    Parameters
    ----------
    modes : ArrayLike
        The matrix of the affine transformation with shape
        ``(n_p_outputs, n_parameters)``.

    offset : Optional[ArrayLike], optional
        The offset of the affine transformation with shape
        ``(n_p_outputs,)``.
        If ``None``, a zero offset is used.

    Returns
    -------
    Parametrization
        A :class:`Parametrization` implementing the affine transformation.

    Examples
    --------
    A two-dimensional parameterization from one optimized parameter:

    >>> parametrization = build_affine_parametrization(
    ...     modes=[[1.0], [2.0]],
    ...     offset=[0.0, 1.0],
    ... )

    This defines:

    .. math::

        \begin{bmatrix}
        p_{out,0} \\
        p_{out,1}
        \end{bmatrix}
        =
        \begin{bmatrix}
        1 \\
        2
        \end{bmatrix}
        p_{in}
        +
        \begin{bmatrix}
        0 \\
        1
        \end{bmatrix}.

    """

    modes = numpy.asarray(modes, dtype=numpy.float64)

    if modes.ndim != 2:
        raise ValueError("modes must be a 2D array.")

    if modes.shape[0] == 0 or modes.shape[1] == 0:
        raise ValueError("modes must have a non-zero shape.")

    n_p_outputs, n_parameters = modes.shape

    if offset is None:
        offset = numpy.zeros(n_p_outputs, dtype=numpy.float64)
    else:
        offset = numpy.asarray(offset, dtype=numpy.float64)

        if offset.ndim != 1:
            raise ValueError("offset must be a 1D array.")

        if offset.shape[0] != n_p_outputs:
            raise ValueError(
                "offset must have shape " f"({n_p_outputs},), got {offset.shape}."
            )

    def parametric_func(
        p_in: ArrayLike,
    ) -> numpy.ndarray:
        p_in = numpy.asarray(p_in, dtype=numpy.float64)

        if p_in.ndim != 1:
            raise ValueError(f"p_in must be a 1D array, got {p_in.ndim} dimensions.")

        if p_in.shape[0] != n_parameters:
            raise ValueError(
                f"p_in must have shape ({n_parameters},), " f"got {p_in.shape}."
            )

        return modes @ p_in + offset

    def jacobian_func(
        p_in: ArrayLike,
    ) -> numpy.ndarray:
        p_in = numpy.asarray(p_in, dtype=numpy.float64)

        if p_in.ndim != 1:
            raise ValueError(f"p_in must be a 1D array, got {p_in.ndim} dimensions.")

        if p_in.shape[0] != n_parameters:
            raise ValueError(
                f"p_in must have shape ({n_parameters},), " f"got {p_in.shape}."
            )

        return modes.copy()

    return Parametrization(
        parametric_func=parametric_func,
        jacobian_func=jacobian_func,
    )


def build_fixed_parametrization(
    n_p_outputs: int,
    optimized_indices: Sequence[int],
    fixed_parameters: Optional[ArrayLike] = None,
) -> Parametrization:
    r"""
    Build a parametrization that fixes some output parameters while optimizing
    only a selected subset of them.

    The transformation is defined as:

    .. math::

        \mathbf{p}_{out} = P(\mathbf{p}_{in})

    where the values at ``optimized_indices`` are taken from
    :math:`\mathbf{p}_{in}` and all other values are fixed.

    Parameters
    ----------
    n_p_outputs : int
        Number of output parameters.

    optimized_indices : Sequence[int]
        Indices of the output parameters that are optimized.
        The order defines the order of the input parameters.

    fixed_parameters : Optional[ArrayLike], optional
        Initial/fixed values of all output parameters, with shape
        ``(n_p_outputs,)``.
        If ``None``, all fixed parameters are initialized to zero.

    Returns
    -------
    Parametrization
        The resulting parametrization.

    Examples
    --------
    For three output parameters, with only parameters 0 and 2 optimized:

    .. code-block:: python

        parametrization = build_fixed_parametrization(
            n_p_outputs=3,
            optimized_indices=[0, 2],
            fixed_parameters=[1.0, 2.0, 3.0],
        )

    Then:

    .. math::

        \mathbf{p}_{out}
        =
        \begin{bmatrix}
        p_{in,0} \\
        2 \\
        p_{in,1}
        \end{bmatrix}

    and the Jacobian has shape ``(3, 2)``.
    """

    if not isinstance(n_p_outputs, (int, numpy.integer)):
        raise ValueError("n_p_outputs must be an integer.")

    n_p_outputs = int(n_p_outputs)

    if n_p_outputs <= 0:
        raise ValueError("n_p_outputs must be strictly positive.")

    optimized_indices = list(optimized_indices)

    if len(optimized_indices) == 0:
        raise ValueError("optimized_indices cannot be empty.")

    if len(set(optimized_indices)) != len(optimized_indices):
        raise ValueError("optimized_indices must contain unique indices.")

    if any(not isinstance(index, (int, numpy.integer)) for index in optimized_indices):
        raise ValueError("optimized_indices must contain integers.")

    optimized_indices = [int(index) for index in optimized_indices]

    if any(index < 0 or index >= n_p_outputs for index in optimized_indices):
        raise ValueError(
            "All optimized_indices must be in the range " f"[0, {n_p_outputs - 1}]."
        )

    if fixed_parameters is None:
        fixed_parameters = numpy.zeros(n_p_outputs, dtype=numpy.float64)
    else:
        fixed_parameters = numpy.asarray(
            fixed_parameters,
            dtype=numpy.float64,
        ).copy()

        if fixed_parameters.ndim != 1:
            raise ValueError("fixed_parameters must be a 1D array.")

        if fixed_parameters.shape[0] != n_p_outputs:
            raise ValueError(
                "fixed_parameters must have shape "
                f"({n_p_outputs},), got {fixed_parameters.shape}."
            )

    n_parameters = len(optimized_indices)

    selection_matrix = numpy.zeros(
        (n_p_outputs, n_parameters),
        dtype=numpy.float64,
    )

    for j, index in enumerate(optimized_indices):
        selection_matrix[index, j] = 1.0

    def parametric_func(p_in: numpy.ndarray) -> numpy.ndarray:
        p_in = numpy.asarray(p_in, dtype=numpy.float64)

        if p_in.ndim != 1:
            raise ValueError("p_in must be a 1D array.")

        if p_in.shape[0] != n_parameters:
            raise ValueError(
                f"p_in must have shape ({n_parameters},), " f"got {p_in.shape}."
            )

        p_out = fixed_parameters.copy()
        p_out[optimized_indices] = p_in

        return p_out

    def jacobian_func(p_in: numpy.ndarray) -> numpy.ndarray:
        p_in = numpy.asarray(p_in, dtype=numpy.float64)

        if p_in.ndim != 1:
            raise ValueError("p_in must be a 1D array.")

        if p_in.shape[0] != n_parameters:
            raise ValueError(
                f"p_in must have shape ({n_parameters},), " f"got {p_in.shape}."
            )

        return selection_matrix

    return Parametrization(
        parametric_func=parametric_func,
        jacobian_func=jacobian_func,
    )


def build_sigmoid_parametrization(
    lower: ArrayLike,
    upper: ArrayLike,
) -> Parametrization:
    r"""
    Build a parametrization that constrains parameters to a finite interval
    using a sigmoid transformation.

    The transformation is defined component-wise as:

    .. math::

        p_{out}
        =
        l + (u-l)\sigma(p_{in})

    where:

    .. math::

        \sigma(x) = \frac{1}{1 + \exp(-x)}

    and :math:`l` and :math:`u` are respectively the lower and upper bounds.

    Consequently:

    .. math::

        l < p_{out} < u.

    The Jacobian is diagonal:

    .. math::

        \frac{\partial p_{out}}{\partial p_{in}}
        =
        (u-l)\sigma(p_{in})(1-\sigma(p_{in})).

    Parameters
    ----------
    lower : ArrayLike
        Lower bounds with shape ``(n_parameters,)``.

    upper : ArrayLike
        Upper bounds with shape ``(n_parameters,)``.

    Returns
    -------
    Parametrization
        The resulting bounded parametrization.
    """

    lower = numpy.asarray(lower, dtype=numpy.float64).copy()
    upper = numpy.asarray(upper, dtype=numpy.float64).copy()

    if lower.ndim != 1:
        raise ValueError("lower must be a 1D array.")

    if upper.ndim != 1:
        raise ValueError("upper must be a 1D array.")

    if lower.shape != upper.shape:
        raise ValueError("lower and upper must have the same shape.")

    if lower.shape[0] == 0:
        raise ValueError("lower and upper cannot be empty.")

    if numpy.any(lower >= upper):
        raise ValueError("Each lower bound must be strictly smaller than upper.")

    n_parameters = lower.shape[0]
    sigmoid = _sigmoid

    def parametric_func(p_in: numpy.ndarray) -> numpy.ndarray:
        p_in = numpy.asarray(p_in, dtype=numpy.float64)

        if p_in.ndim != 1:
            raise ValueError("p_in must be a 1D array.")

        if p_in.shape[0] != n_parameters:
            raise ValueError(
                f"p_in must have shape ({n_parameters},), " f"got {p_in.shape}."
            )

        s = sigmoid(p_in)

        return lower + (upper - lower) * s

    def jacobian_func(p_in: numpy.ndarray) -> numpy.ndarray:
        p_in = numpy.asarray(p_in, dtype=numpy.float64)

        if p_in.ndim != 1:
            raise ValueError("p_in must be a 1D array.")

        if p_in.shape[0] != n_parameters:
            raise ValueError(
                f"p_in must have shape ({n_parameters},), " f"got {p_in.shape}."
            )

        s = sigmoid(p_in)

        return numpy.diag((upper - lower) * s * (1.0 - s))

    return Parametrization(
        parametric_func=parametric_func,
        jacobian_func=jacobian_func,
    )


def build_positive_parametrization(
    n_parameters: int,
) -> Parametrization:
    r"""
    Build a parametrization that constrains parameters to be strictly positive.

    The transformation is defined component-wise as:

    .. math::

        p_{out} = \exp(p_{in}).

    Therefore:

    .. math::

        p_{out} > 0.

    Its Jacobian is:

    .. math::

        \frac{\partial p_{out}}
             {\partial p_{in}}
        =
        \operatorname{diag}(\exp(p_{in})).

    Parameters
    ----------
    n_parameters : int
        Number of positive output parameters and input parameters.

    Returns
    -------
    Parametrization
        The resulting positive parametrization.
    """

    if not isinstance(n_parameters, (int, numpy.integer)):
        raise ValueError("n_parameters must be an integer.")

    n_parameters = int(n_parameters)

    if n_parameters <= 0:
        raise ValueError("n_parameters must be strictly positive.")

    def parametric_func(p_in: numpy.ndarray) -> numpy.ndarray:
        p_in = numpy.asarray(p_in, dtype=numpy.float64)

        if p_in.ndim != 1:
            raise ValueError("p_in must be a 1D array.")

        if p_in.shape[0] != n_parameters:
            raise ValueError(
                f"p_in must have shape ({n_parameters},), " f"got {p_in.shape}."
            )

        return numpy.exp(p_in)

    def jacobian_func(p_in: numpy.ndarray) -> numpy.ndarray:
        p_in = numpy.asarray(p_in, dtype=numpy.float64)

        if p_in.ndim != 1:
            raise ValueError("p_in must be a 1D array.")

        if p_in.shape[0] != n_parameters:
            raise ValueError(
                f"p_in must have shape ({n_parameters},), " f"got {p_in.shape}."
            )

        return numpy.diag(numpy.exp(p_in))

    return Parametrization(
        parametric_func=parametric_func,
        jacobian_func=jacobian_func,
    )


# ======================================================================
# Batch versions (same transformation for all the problems of the batch)
# ======================================================================


def build_batch_affine_parametrization(
    modes: ArrayLike,
    offset: Optional[ArrayLike] = None,
) -> BatchParametrization:
    r"""
    Build a :class:`BatchParametrization` based on an affine transformation
    (batched counterpart of :func:`build_affine_parametrization`).

    The same transformation is applied to each problem :math:`k` of the batch:

    .. math::

        \mathbf{p}_{out,k}
        =
        M \mathbf{p}_{in,k}
        +
        \mathbf{p}_0

    with the constant Jacobian :math:`\mathbf{J}_{P,k} = M`.

    Parameters
    ----------
    modes : ArrayLike
        The matrix of the affine transformation with shape
        ``(n_p_outputs, n_parameters)``.

    offset : Optional[ArrayLike], optional
        The offset of the affine transformation with shape
        ``(n_p_outputs,)``.
        If ``None``, a zero offset is used.

    Returns
    -------
    BatchParametrization
        A :class:`BatchParametrization` implementing the affine transformation:
        ``(m, n_parameters) -> (m, n_p_outputs)``.
    """
    single = build_affine_parametrization(modes, offset)  # validation
    modes = numpy.asarray(modes, dtype=numpy.float64)
    offset = single.p_func(numpy.zeros(modes.shape[1]))  # M @ 0 + offset
    n_p_outputs, n_parameters = modes.shape

    def parametric_func(p_in: ArrayLike) -> numpy.ndarray:
        p_in = _check_batch_p_in(p_in, n_parameters)
        return p_in @ modes.T + offset

    def jacobian_func(p_in: ArrayLike) -> numpy.ndarray:
        p_in = _check_batch_p_in(p_in, n_parameters)
        return numpy.broadcast_to(modes, (p_in.shape[0], n_p_outputs, n_parameters)).copy()

    return BatchParametrization(
        parametric_func=parametric_func,
        jacobian_func=jacobian_func,
    )


def build_batch_fixed_parametrization(
    n_p_outputs: int,
    optimized_indices: Sequence[int],
    fixed_parameters: Optional[ArrayLike] = None,
) -> BatchParametrization:
    r"""
    Build a :class:`BatchParametrization` that fixes some output parameters while
    optimizing only a selected subset of them (batched counterpart of
    :func:`build_fixed_parametrization`).

    The same parameters are fixed, at the same values, for all the problems of the
    batch.

    Parameters
    ----------
    n_p_outputs : int
        Number of output parameters.

    optimized_indices : Sequence[int]
        Indices of the output parameters that are optimized.
        The order defines the order of the input parameters.

    fixed_parameters : Optional[ArrayLike], optional
        Initial/fixed values of all output parameters, with shape
        ``(n_p_outputs,)``.
        If ``None``, all fixed parameters are initialized to zero.

    Returns
    -------
    BatchParametrization
        The resulting parametrization: ``(m, len(optimized_indices)) -> (m, n_p_outputs)``.
    """
    single = build_fixed_parametrization(n_p_outputs, optimized_indices, fixed_parameters)  # validation
    optimized_indices = [int(index) for index in optimized_indices]
    n_parameters = len(optimized_indices)
    selection_matrix = single.J_func(numpy.zeros(n_parameters))
    fixed_parameters = (
        numpy.zeros(int(n_p_outputs), dtype=numpy.float64)
        if fixed_parameters is None
        else numpy.asarray(fixed_parameters, dtype=numpy.float64).copy()
    )

    def parametric_func(p_in: numpy.ndarray) -> numpy.ndarray:
        p_in = _check_batch_p_in(p_in, n_parameters)
        p_out = numpy.tile(fixed_parameters, (p_in.shape[0], 1))
        p_out[:, optimized_indices] = p_in
        return p_out

    def jacobian_func(p_in: numpy.ndarray) -> numpy.ndarray:
        p_in = _check_batch_p_in(p_in, n_parameters)
        return numpy.broadcast_to(
            selection_matrix, (p_in.shape[0],) + selection_matrix.shape
        ).copy()

    return BatchParametrization(
        parametric_func=parametric_func,
        jacobian_func=jacobian_func,
    )


def build_batch_sigmoid_parametrization(
    lower: ArrayLike,
    upper: ArrayLike,
) -> BatchParametrization:
    r"""
    Build a :class:`BatchParametrization` that constrains parameters to a finite
    interval using a sigmoid transformation (batched counterpart of
    :func:`build_sigmoid_parametrization`).

    The same bounds are used for all the problems of the batch:

    .. math::

        p_{out,k,i} = l_i + (u_i - l_i) \sigma(p_{in,k,i})

    Parameters
    ----------
    lower : ArrayLike
        Lower bounds with shape ``(n_parameters,)``.

    upper : ArrayLike
        Upper bounds with shape ``(n_parameters,)``.

    Returns
    -------
    BatchParametrization
        The resulting bounded parametrization: ``(m, n_parameters) -> (m, n_parameters)``.
    """
    build_sigmoid_parametrization(lower, upper)  # validation
    lower = numpy.asarray(lower, dtype=numpy.float64).copy()
    upper = numpy.asarray(upper, dtype=numpy.float64).copy()
    n_parameters = lower.shape[0]

    def parametric_func(p_in: numpy.ndarray) -> numpy.ndarray:
        p_in = _check_batch_p_in(p_in, n_parameters)
        s = _sigmoid(p_in)
        return lower + (upper - lower) * s

    def jacobian_func(p_in: numpy.ndarray) -> numpy.ndarray:
        p_in = _check_batch_p_in(p_in, n_parameters)
        s = _sigmoid(p_in)
        return _batch_diagonal((upper - lower) * s * (1.0 - s))

    return BatchParametrization(
        parametric_func=parametric_func,
        jacobian_func=jacobian_func,
    )


def build_batch_positive_parametrization(
    n_parameters: int,
) -> BatchParametrization:
    r"""
    Build a :class:`BatchParametrization` that constrains parameters to be strictly
    positive (batched counterpart of :func:`build_positive_parametrization`):

    .. math::

        p_{out,k,i} = \exp(p_{in,k,i}).

    Parameters
    ----------
    n_parameters : int
        Number of positive output parameters and input parameters.

    Returns
    -------
    BatchParametrization
        The resulting positive parametrization: ``(m, n_parameters) -> (m, n_parameters)``.
    """
    build_positive_parametrization(n_parameters)  # validation
    n_parameters = int(n_parameters)

    def parametric_func(p_in: numpy.ndarray) -> numpy.ndarray:
        p_in = _check_batch_p_in(p_in, n_parameters)
        return numpy.exp(p_in)

    def jacobian_func(p_in: numpy.ndarray) -> numpy.ndarray:
        p_in = _check_batch_p_in(p_in, n_parameters)
        return _batch_diagonal(numpy.exp(p_in))

    return BatchParametrization(
        parametric_func=parametric_func,
        jacobian_func=jacobian_func,
    )