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

from __future__ import annotations
from typing import Callable, Optional

import numpy

from .derivation import build_batch_numerical_jacobian
from .implemented_conf import (
    _IMPLEMENTED_FINITE_DIFFERENCE_METHODS,
)


class BatchParametrization(object):
    r"""
    A parametric transformation :math:`P` applied to each problem of a batch of
    independent least squares problems. It maps the input parameters
    :math:`\vec{p_{in,k}}` of each problem :math:`k` from an input parametric space
    :math:`\mathbb{R}^{n_{\mathrm{parameters}}}` to output parameters
    :math:`\vec{p_{out,k}}` in an output parametric space
    :math:`\mathbb{R}^{n_{\mathrm{p\_outputs}}}`.

    This is the batched counterpart of :class:`pysolvegn.Parametrization`, used by
    :func:`pysolvegn.solve_batch`.

    .. math::

        \vec{p_{out,k}} = P(\vec{p_{in,k}})

    .. important::

        **The same transformation** :math:`P` **is used for all the problems of the batch.**

        Unlike :class:`pysolvegn.BatchTerm`, the callables of a ``BatchParametrization``
        do **not** receive the indices of the problems: the transformation can not depend
        on the problem :math:`k`. Only the input parameters change from one problem to
        another. The callables are simply vectorized over the first dimension: the row
        ``a`` of every output only depends on the row ``a`` of the input parameters.

        **The number of evaluated problems** ``m`` **is NOT fixed**: the solver removes
        the stopped problems at each iteration, so the callables are called with
        parameters with shape ``(m, n_parameters)`` for any ``1 <= m <= k``.

    Considering the following batch of least-square problems:

    .. math::

        \min_{\vec{p_{out,k}}} \frac{1}{2} \sum_{i} w_{i,k} \sum_j
        \rho_i\left(\| \mathbf{r}_{i,j}(\vec{p_{out,k}}) \|^2\right)

    each problem can be rewritten using the parametric transformation :math:`P` as:

    .. math::

        \min_{\vec{p_{in,k}}} \frac{1}{2} \sum_{i} w_{i,k} \sum_j
        \rho_i\left(
            \| \mathbf{r}_{i,j}(P(\vec{p_{in,k}})) \|^2
        \right)

    The parameters :math:`\vec{p_{in,k}}` are the parameters actually optimized
    by the batched Gauss-Newton algorithm, while :math:`\vec{p_{out,k}}` are the
    parameters passed to the residual functions of the :class:`pysolvegn.BatchTerm`.

    This class is used to store the parametric transformation :math:`P` and
    its first derivative, evaluated for each problem:

    .. math::

        \mathbf{J}_{P,k} =
        \frac{\partial \vec{p_{out}}}
             {\partial \vec{p_{in}}}\left(\vec{p_{in,k}}\right).

    Parameters
    ----------
    parametric_func : Callable
        The function to compute the parametric transformation :math:`P`.
        The function should take as input the input parameters of ``m`` problems
        (2D array with shape ``(m, n_parameters)``) and return the output parameters
        :math:`\mathbf{p}_{out}` as a 2D numpy array with shape ``(m, n_p_outputs)``,
        where the row ``a`` is :math:`P` applied to the row ``a`` of the input.

    jacobian_func : Optional[Callable] (default=None)
        The function to compute the Jacobian matrices of the parametric
        transformation :math:`P` with respect to the input parameters.
        The function should take as input the input parameters of ``m`` problems
        (2D array with shape ``(m, n_parameters)``) and return the Jacobian matrices
        as a 3D array with shape ``(m, n_p_outputs, n_parameters)``.
        Each element of the Jacobian is given by
        :math:`\mathbf{J}_{P,k,(j,l)}=\frac{\partial p_{out,k,j}}{\partial p_{in,k,l}}`
        where :math:`j=0,...,n_{p_o}-1}`
        and :math:`l=0,...,n_{p}-1}`.
        If not provided, ``finite_difference`` must be specified to
        numerically compute the Jacobian.

    finite_difference : Optional[str] (default=None)
        The finite difference method used to numerically compute the Jacobian
        when ``jacobian_func`` is not provided.
        If ``None``, the Jacobian is not computed numerically. In this case,
        ``jacobian_func`` must be provided.
        Available methods are ``"central"``, ``"forward"``, and ``"backward"``.

    Examples
    --------
    Optimize strictly positive output parameters through their logarithm, for all the
    problems of the batch:

    .. code-block:: python

        import numpy
        from pysolvegn import BatchParametrization

        def parametric_func(p_in):  # (m, n) -> (m, n)
            return numpy.exp(p_in)

        def jacobian_func(p_in):  # (m, n) -> (m, n, n)
            return numpy.exp(p_in)[:, :, None] * numpy.eye(p_in.shape[1])

        parametrization = BatchParametrization(parametric_func, jacobian_func)

    """

    __slots__ = [
        "_parametric_func",
        "_jacobian_func",
    ]

    def __init__(
        self,
        parametric_func: Callable = None,
        jacobian_func: Optional[Callable] = None,
        finite_difference: Optional[str] = None,
    ):
        if parametric_func is None:
            raise ValueError(
                "A BatchParametrization object must be defined by the parametric function."
            )

        if not callable(parametric_func):
            raise ValueError(
                f"parametric_func must be a callable function, got {type(parametric_func)}."
            )

        if jacobian_func is not None and not callable(jacobian_func):
            raise ValueError(
                f"jacobian_func must be a callable function, got {type(jacobian_func)}."
            )

        if (parametric_func is not None) and (
            jacobian_func is None and finite_difference is None
        ):
            raise ValueError(
                "Both parametric_func and jacobian_func must be provided together (or use finite_difference to numerically compute the jacobian)."
            )

        if jacobian_func is not None and finite_difference is not None:
            raise ValueError(
                "finite_difference must be None if jacobian_func is provided."
            )

        if finite_difference is not None:
            if not isinstance(finite_difference, str):
                raise TypeError("finite_difference must be a string.")
            finite_difference = finite_difference.lower()
            if finite_difference not in _IMPLEMENTED_FINITE_DIFFERENCE_METHODS:
                raise ValueError(
                    f"finite_difference must be one of {_IMPLEMENTED_FINITE_DIFFERENCE_METHODS}, got '{finite_difference}'."
                )

        self._parametric_func = parametric_func
        self._jacobian_func = jacobian_func

        # If parametric but no Jacobian is provided, build the numerical Jacobian function
        if self._parametric_func is not None and (
            self._jacobian_func is None and finite_difference is not None
        ):
            self._jacobian_func = self._build_numerical_jacobian(
                self._parametric_func, finite_difference
            )

    @staticmethod
    def _build_numerical_jacobian(
        parametric_func: Callable, finite_difference: str
    ) -> Callable:
        r"""
        Build the batched numerical Jacobian ``(m, n_parameters) -> (m, n_p_outputs, n_parameters)``
        of a parametric function that does not depend on the indices of the problems.

        :func:`pysolvegn.build_batch_numerical_jacobian` works with the
        ``(parameters, indices)`` convention of :class:`pysolvegn.BatchTerm`: the
        parametric function is wrapped to ignore the indices, and the indices
        are not exposed to the user.
        """
        numerical_jacobian = build_batch_numerical_jacobian(
            residual_func=lambda parameters, indices: parametric_func(parameters),
            method=finite_difference,
            epsilon=1e-8,
        )

        def jacobian_func(parameters: numpy.ndarray) -> numpy.ndarray:
            parameters = numpy.asarray(parameters, dtype=numpy.float64)
            indices = numpy.arange(parameters.shape[0])  # unused by parametric_func
            return numerical_jacobian(parameters, indices)

        return jacobian_func

    @property
    def parametric_func(self) -> Callable:
        r"""
        [Get] the batched parametric function :math:`P: \mathbb{R}^{n_p} \rightarrow \mathbb{R}^{n_{p_o}}` of the parametrization.

        The parametric function takes the input parameters of ``m`` problems as a 2D array
        with shape ``(m, n_parameters)`` and returns the output parameters as a 2D array with
        shape ``(m, n_p_outputs)``. The same transformation is applied to each row.

        .. note::

            The alias ``p_func`` is also available for ``parametric_func`` for convenience.

        Returns
        -------
        Callable
            The parametric function of the parametrization.
        """
        return self._parametric_func

    @property
    def p_func(self) -> Callable:
        return self.parametric_func

    @property
    def jacobian_func(self) -> Callable:
        r"""
        [Get] the batched Jacobian function :math:`\mathbf{J}_P: \mathbb{R}^{n_p} \rightarrow \mathbb{M}_{n_{p_o}, n_p}(\mathbb{R})` of the parametrization.

        The Jacobian function takes the input parameters of ``m`` problems as a 2D array
        with shape ``(m, n_parameters)`` and returns the Jacobian matrices as a 3D array with
        shape ``(m, n_p_outputs, n_parameters)``.

        .. note::

            The alias ``J_func`` is also available for ``jacobian_func`` for convenience.

        Returns
        -------
        Callable
            The Jacobian function of the parametrization.
        """
        return self._jacobian_func

    @property
    def J_func(self) -> Callable:
        return self.jacobian_func