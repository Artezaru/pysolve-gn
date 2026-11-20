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

from typing import Optional, Sequence, Union
from numpy.typing import ArrayLike

import numpy
import scipy.sparse

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


def _validate_sparse(sparse: bool, allow_none: bool = False) -> Optional[bool]:
    # The sparse option of the parametrizations
    if sparse is None and allow_none:
        return None
    if not isinstance(sparse, bool):
        raise ValueError("sparse must be a boolean" + (" or None." if allow_none else "."))
    return sparse


def _diagonal_jacobian(diagonal: numpy.ndarray, sparse: bool) -> Union[numpy.ndarray, scipy.sparse.csr_matrix]:
    # Diagonal Jacobian (n, n): dense array or sparse csr matrix (n entries)
    if sparse:
        return scipy.sparse.diags(diagonal, format="csr")
    return numpy.diag(diagonal)


def _batch_diagonal(diagonal: numpy.ndarray) -> numpy.ndarray:
    # Stack of diagonal matrices (m, n, n) from the diagonals (m, n)
    m, n = diagonal.shape
    jacobian = numpy.zeros((m, n, n), dtype=numpy.float64)
    index = numpy.arange(n)
    jacobian[:, index, index] = diagonal
    return jacobian


def build_affine_parametrization(
    modes: Union[ArrayLike, scipy.sparse.spmatrix],
    offset: Optional[ArrayLike] = None,
    *,
    sparse: Optional[bool] = None,
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
    modes : Union[ArrayLike, scipy.sparse.spmatrix]
        The matrix of the affine transformation with shape
        ``(n_p_outputs, n_parameters)``, a dense array or a ``scipy.sparse`` matrix.

    offset : Optional[ArrayLike], optional
        The offset of the affine transformation with shape
        ``(n_p_outputs,)``.
        If ``None``, a zero offset is used.

    sparse : Optional[bool], optional (default: ``None``)
        If True, the Jacobian is a ``scipy.sparse.csr_matrix``; if False, a dense
        ``numpy.ndarray``. If None, the Jacobian has the format of ``modes`` (sparse for a
        ``scipy.sparse`` matrix, dense otherwise).

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

    sparse = _validate_sparse(sparse, allow_none=True)
    if sparse is None:
        sparse = scipy.sparse.issparse(modes)
    if sparse:
        modes = scipy.sparse.csr_matrix(modes, dtype=numpy.float64)
    elif scipy.sparse.issparse(modes):
        modes = modes.toarray().astype(numpy.float64)
    else:
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
    *,
    sparse: bool = False,
) -> Parametrization:
    r"""
    Build a parametrization that fixes some output parameters while optimizing
    only a selected subset of them.

    The transformation is defined as:

    .. math::

        \mathbf{p}_{out} = P(\mathbf{p}_{in})

    where the values at ``optimized_indices`` are taken from
    :math:`\mathbf{p}_{in}` and all other values are fixed.

    The Jacobian is the constant selection matrix :math:`S` of shape
    ``(n_p_outputs, len(optimized_indices))``, with :math:`S_{i, j} = 1` if
    ``optimized_indices[j] == i``.

    .. warning::

        The dense selection matrix has ``n_p_outputs * len(optimized_indices)``
        entries. For large problems (e.g. displacements of the vertices of a mesh),
        use ``sparse=True``: the matrix has then one entry per optimized parameter.

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

    sparse : bool, optional (default: ``False``)
        If True, the Jacobian is a ``scipy.sparse.csr_matrix``, otherwise a dense
        ``numpy.ndarray``.

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

    if not isinstance(n_p_outputs, (int, numpy.integer)) or isinstance(n_p_outputs, bool):
        raise ValueError("n_p_outputs must be an integer.")

    n_p_outputs = int(n_p_outputs)

    if n_p_outputs <= 0:
        raise ValueError("n_p_outputs must be strictly positive.")

    sparse = _validate_sparse(sparse)

    optimized_indices = numpy.asarray(optimized_indices)

    if optimized_indices.ndim != 1:
        raise ValueError("optimized_indices must be a 1D sequence of integers.")

    if optimized_indices.size == 0:
        raise ValueError("optimized_indices cannot be empty.")

    if not numpy.issubdtype(optimized_indices.dtype, numpy.integer):
        raise ValueError("optimized_indices must contain integers.")

    optimized_indices = optimized_indices.astype(numpy.int64)

    if numpy.unique(optimized_indices).size != optimized_indices.size:
        raise ValueError("optimized_indices must contain unique indices.")

    if numpy.any((optimized_indices < 0) | (optimized_indices >= n_p_outputs)):
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

    n_parameters = optimized_indices.size

    selection_matrix = scipy.sparse.csr_matrix(
        (
            numpy.ones(n_parameters, dtype=numpy.float64),
            (optimized_indices, numpy.arange(n_parameters)),
        ),
        shape=(n_p_outputs, n_parameters),
    )
    if not sparse:
        selection_matrix = selection_matrix.toarray()

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

    def jacobian_func(p_in: numpy.ndarray) -> Union[numpy.ndarray, scipy.sparse.csr_matrix]:
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
    *,
    sparse: bool = False,
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

    sparse : bool, optional (default: ``False``)
        If True, the diagonal Jacobian is a ``scipy.sparse.csr_matrix`` (``n`` entries),
        otherwise a dense ``numpy.ndarray`` of shape ``(n, n)``. Use ``sparse=True`` for
        a large number of parameters.

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

    sparse = _validate_sparse(sparse)
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

    def jacobian_func(p_in: numpy.ndarray) -> Union[numpy.ndarray, scipy.sparse.csr_matrix]:
        p_in = numpy.asarray(p_in, dtype=numpy.float64)

        if p_in.ndim != 1:
            raise ValueError("p_in must be a 1D array.")

        if p_in.shape[0] != n_parameters:
            raise ValueError(
                f"p_in must have shape ({n_parameters},), " f"got {p_in.shape}."
            )

        s = sigmoid(p_in)

        return _diagonal_jacobian((upper - lower) * s * (1.0 - s), sparse)

    return Parametrization(
        parametric_func=parametric_func,
        jacobian_func=jacobian_func,
    )


def build_positive_parametrization(
    n_parameters: int,
    *,
    sparse: bool = False,
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

    sparse : bool, optional (default: ``False``)
        If True, the diagonal Jacobian is a ``scipy.sparse.csr_matrix`` (``n`` entries),
        otherwise a dense ``numpy.ndarray`` of shape ``(n, n)``. Use ``sparse=True`` for
        a large number of parameters.

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

    sparse = _validate_sparse(sparse)

    def parametric_func(p_in: numpy.ndarray) -> numpy.ndarray:
        p_in = numpy.asarray(p_in, dtype=numpy.float64)

        if p_in.ndim != 1:
            raise ValueError("p_in must be a 1D array.")

        if p_in.shape[0] != n_parameters:
            raise ValueError(
                f"p_in must have shape ({n_parameters},), " f"got {p_in.shape}."
            )

        return numpy.exp(p_in)

    def jacobian_func(p_in: numpy.ndarray) -> Union[numpy.ndarray, scipy.sparse.csr_matrix]:
        p_in = numpy.asarray(p_in, dtype=numpy.float64)

        if p_in.ndim != 1:
            raise ValueError("p_in must be a 1D array.")

        if p_in.shape[0] != n_parameters:
            raise ValueError(
                f"p_in must have shape ({n_parameters},), " f"got {p_in.shape}."
            )

        return _diagonal_jacobian(numpy.exp(p_in), sparse)

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
    modes : Union[ArrayLike, scipy.sparse.spmatrix]
        The matrix of the affine transformation with shape
        ``(n_p_outputs, n_parameters)`` (a sparse matrix is converted to a dense
        array: the batched Jacobians are dense).

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
    if scipy.sparse.issparse(modes):
        modes = modes.toarray()  # The batched Jacobians (m, n_p_outputs, n_parameters) are dense
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
    single = build_fixed_parametrization(n_p_outputs, optimized_indices, fixed_parameters)  # validation (dense)
    optimized_indices = numpy.asarray(optimized_indices, dtype=numpy.int64)
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