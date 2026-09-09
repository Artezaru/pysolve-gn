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

from .derivation import build_numerical_jacobian
from .implemented_conf import (
    _IMPLEMENTED_FINITE_DIFFERENCE_METHODS,
)


class Parametrization(object):
    r"""
    A parametric transformation :math:`P` that
    maps input parameters :math:`\vec{p_{in}}` from an input parametric space
    :math:`\mathbb{R}^{n_{\mathrm{parameters}}}` to output parameters
    :math:`\vec{p_{out}}` in an output parametric space
    :math:`\mathbb{R}^{n_{\mathrm{p\_outputs}}}`.

    .. math::

        \vec{p_{out}} = \vec{p} = P(\vec{p_{in}})

    Considering the following least-square problem:

    .. math::

        \min_{\vec{p_{out}}} \frac{1}{2} \sum_{i} w_i \sum_j
        \rho_i\left(\| \mathbf{r}_{i,j}(\vec{p_{out}}) \|^2\right)

    it can be rewritten using the parametric transformation :math:`P` as:

    .. math::

        \min_{\vec{p_{in}}} \frac{1}{2} \sum_{i} w_i \sum_j
        \rho_i\left(
            \| \mathbf{r}_{i,j}(P(\vec{p_{in}})) \|^2
        \right)

    The parameters :math:`\vec{p_{in}}` are the parameters actually optimized
    by the Gauss-Newton algorithm, while :math:`\vec{p_{out}}` are the
    parameters passed to the residual functions.

    This class is used to store the parametric transformation :math:`P` and
    its first derivative:

    .. math::

        \mathbf{J}_P =
        \frac{\partial \vec{p_{out}}}
             {\partial \vec{p_{in}}}.

    Parameters
    ----------
    parametric_func : Callable
        The function to compute the parametric transformation :math:`P`.
        The function should take as input the parameters
        (array-like with shape ``(n_parameters,)``) and return the transformed
        parameters as a 1D numpy array with shape ``(n_p_outputs,)``,
        representing the output parameters
        :math:`\mathbf{p}_{out}`.

    jacobian_func : Optional[Callable] (default=None)
        The function to compute the Jacobian matrix of the parametric
        transformation :math:`P` with respect to the input parameters.
        The function should take as input the parameters
        (1D array-like with shape ``(n_parameters,)``) and return the Jacobian
        matrix as a 2D array-like with shape
        ``(n_p_outputs, n_parameters)``.
        Each element of the Jacobian is given by
        :math:`\mathbf{J}_{P,(j,l)}=\frac{\partial p_{out,j}}{\partial p_{in,l}}`
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
                "A Parametrization object must be defined by the parametric function."
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
            self._jacobian_func = build_numerical_jacobian(
                residual_func=self._parametric_func,
                method=finite_difference,
                epsilon=1e-8,
            )

    @property
    def parametric_func(self) -> Callable:
        r"""
        [Get] the parametric function :math:`\P: \mathbb{R}^{n_p} \rightarrow \mathbb{R}^{n_{p_o}}}` of the parametrization.

        The parametric function take parameters as a 1D array-like of shape ``(n_parameters,)`` and return the output parameters vector as a
        1D array-like with shape ``(n_p_outputs,)``.

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
        [Get] the Jacobian function :math:`\mathbf{J}_i: \mathbb{R}^{n_p} \rightarrow :math:`\mathbb{M}_{n_{p_o}, n_p}(\mathbb{R})`` of the parametrization.

        The Jacobian function take parameters as a 1D array-like of shape ``(n_parameters,)`` and return the Jacobian matrix as a
        2D array-like with shape ``(n_p_outputs, n_parameters)``.

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
