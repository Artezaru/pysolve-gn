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
        (array-like with shape ``(n_parameters,)``) and return the Jacobian
        matrix as a 2D numpy array with shape
        ``(n_p_outputs, n_parameters)``.

        Each row :math:`j` corresponds to the derivative of the output
        parameter :math:`p_{out,j}` with respect to the input parameters:

        .. math::

            \mathbf{J}_{P,j}
            =
            \frac{\partial p_{out,j}}
                 {\partial \mathbf{p}_{in}}.

        If not provided, the Jacobian will be computed numerically using
        finite differences.

    finite_difference : str (default="central")
        The finite difference method to use for computing the numerical
        Jacobian if the Jacobian function is not provided. Must be one of
        ``"central"``, ``"forward"``, or ``"backward"``.

    Notes
    -----

    - If ``jacobian_func`` is not provided, the Jacobian will be computed
      numerically using finite differences.

    Version
    -------

    - 0.3.0: Initial version of the :class:`Parametrization` class.

    """

    __slots__ = [
        "_parametric_func",
        "_jacobian_func",
        "_finite_difference",
        "_use_finite_difference_jacobian",
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

        if finite_difference is None:
            finite_difference = "central"
        if not isinstance(finite_difference, str):
            raise ValueError("finite_difference must be a string.")
        finite_difference = finite_difference.lower()
        if finite_difference not in _IMPLEMENTED_FINITE_DIFFERENCE_METHODS:
            raise ValueError(
                f"finite_difference must be one of {_IMPLEMENTED_FINITE_DIFFERENCE_METHODS}, got '{finite_difference}'."
            )

        self._parametric_func = parametric_func
        self._jacobian_func = jacobian_func
        self._finite_difference = finite_difference.lower()
        self._use_finite_difference_jacobian = False

        # If parametric function but no Jacobian is provided, build the numerical Jacobian function
        if self._parametric_func is not None and self._jacobian_func is None:
            self._jacobian_func = build_numerical_jacobian(
                parametric_func=self._parametric_func,
                method=self._finite_difference,
                epsilon=1e-8,
            )
            self._use_finite_difference_jacobian = True

    @property
    def parametric_func(self) -> Callable:
        r"""
        [Get] the parametric function of the parametrization.

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
        [Get] the Jacobian function of the parametrization.

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

    @property
    def finite_difference(self) -> str:
        r"""
        [Get] the finite difference method for numerical Jacobian computation of the parametrization.

        Returns
        -------
        str
            The current finite difference method for numerical Jacobian computation of the parametrization.
        """
        return self._finite_difference

    @property
    def use_finite_difference(self) -> bool:
        r"""
        [Get] whether the parametrization is using a finite difference Jacobian function.

        Returns
        -------
        bool
            True if the parametrization is using a finite difference Jacobian function, False otherwise.
        """
        return self._use_finite_difference_jacobian
