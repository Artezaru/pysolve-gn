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
from typing import Callable, Optional, Union
from numbers import Real

from .derivation import build_numerical_jacobian
from .implemented_conf import (
    _IMPLEMENTED_LOSS_FUNCTIONS,
    _IMPLEMENTED_FINITE_DIFFERENCE_METHODS,
)
from .loss_functions import get_rho_function_by_name


class Term(object):
    r"""
    A class representing a term in the least squares problem.

    .. seealso::

        For more details on the notations for the optimization problem,
        please refer to the mathematical section of the documentation.

    For the following definition, the :math:`i`-th term in the least squares
    problem is denoted as :math:`\mathbf{r}_i(\mathbf{p}) \in \mathbb{R}^{n_{r_i}}``.

    .. math::

        \min_{\mathbf{p}} \frac{1}{2} \sum_{i} w_i \sum_j \rho_i\left(\| \mathbf{r}_{i,j}(\mathbf{p}) \|^2\right)

    where :math:`i` indexes the different least squares terms (:class:`pysolvegn.Term`) and :math:`j`
    indexes the residuals within each term.
    Each term can be defined by two callable functions:

    1. The **Residual** and **Jacobian** functions

        The residual function :math:`\mathbf{r}_i(\mathbf{p})` to minimize
        and the associated Jacobian function
        :math:`\mathbf{J}_i(\mathbf{p}) = \frac{\partial \mathbf{r}_i}{\partial \mathbf{p}}`
        to compute the Jacobian matrix of the residuals with respect to the parameters.

        Then the solver will build the gradient :math:`\mathbf{g}(\mathbf{p})` and
        the Hessian of the least squares problem :math:`\mathbf{H}(\mathbf{p})` as:

        .. math::

            \mathbf{g} = w_i \tilde{\mathbf{J}}_i^T \tilde{\mathbf{r}}_i

        .. math::

            \mathbf{H} = w_i \tilde{\mathbf{J}}_i^T \tilde{\mathbf{J}}_i

        such that :math:`\mathbf{H} \Delta \mathbf{p} = -\mathbf{g}`.


    2. The **gradient** and **Hessian** functions

        The Hessian function :math:`\mathbf{H}_i(\mathbf{p})` to compute the Hessian of
        the least squares problem and the gradient function
        :math:`\mathbf{g}_i(\mathbf{p})` to compute the gradient of the
        least squares problem.

        Then the solver will use them directly without building them from the residuals
        and Jacobian, such that:

        .. math::

            \mathbf{g} = w_i \mathbf{g}_i(\mathbf{p})

        .. math::

            \mathbf{H} = w_i \mathbf{H}_i(\mathbf{p})

        Such that :math:`\mathbf{H} \Delta \mathbf{p} = -\mathbf{g}`.

    This class is used to represent each term in the least squares problem,
    and can be used to define the residual and Jacobian functions for each term,
    as well as the gradient and Hessian functions for each term,
    which can be used to build the least squares problem.

    .. important::

        The term must be define by (residual + jacobian) or (gradient + Hessian) not both.


    Parameters
    ----------
    residual_func : Optional[Callable] (default=None)
        The function to compute the residuals :math:`\mathbf{r}_i: \mathbb{R}^{n_p} \rightarrow \mathbb{R}^{n_{r_i}}`
        of the least squares problem. The function should take as inputs the parameters
        (1D array-like with shape ``(n_parameters,)``) and return the residuals as a
        1D array-like with shape ``(n_residuals,)`` representing each
        :math:`\mathbf{r}_{i,j}(\mathbf{p})` where :math:`j=0,...,n_{r_i}-1`.

    jacobian_func : Optional[Callable] (default=None)
        The function to compute the Jacobian matrix of the residuals with respect to the parameters.
        The function should take as inputs the parameters
        (1D array-like with shape ``(n_parameters,)``) and return the Jacobian matrix as a 2D array-like
        with shape ``(n_residuals, n_parameters)`` representing each
        :math:`\mathbf{J}_{i,(j,l)}(\mathbf{p}) = \frac{\partial \mathbf{r}_{i,j}}{\partial \mathbf{p}_l}` where :math:`j=0,...,n_{r_i}-1}`
        and :math:`l=0,...,n_{p}-1}`.
        Can be numericallty computed using ``finite_difference`` argument.

    gradient_func: Optional[Callable] (default=None)
        The function to compute the gradient :math:`\mathbf{g}_i(\mathbf{p}) = \mathbf{J}_i^T \mathbf{r}_i`
        of the least squares problem.
        The function should take as inputs the parameters
        (1D array-like with shape ``(n_parameters,)``) and return the gradient as a
        1D array-like with shape ``(n_parameters,)``.

    hessian_func: Optional[Callable] (default=None)
        The function to compute the Hessian :math:`\mathbf{H}_i(\mathbf{p}) = \mathbf{J}_i^T \mathbf{J}_i`
        of the least squares problem.
        The function should take as inputs the parameters
        (1D array-like with shape ``(n_parameters,)``) and return the Hessian matrix as a
        2D array-like with shape ``(n_parameters, n_parameters)``.

    cost_func: Optional[Callable] (default=None)
        The function to compute the cost of the term in the least squares problem.
        The function should take as inputs the parameters
        (1D array-like with shape ``(n_parameters,)``) and return the cost as a positive
        floating value independent of the ``weight``.
        If not provided, the default cost during optimization will be
        :math:`\frac{1}{2} \sum_j \rho_i(\| \mathbf{r}_{i,j}(\mathbf{p}_{out}) \|^2)`
        for ``"rJ"`` terms and ``0.0`` for ``"gH"`` terms.

    weight: Real (default=1.0)
        The weight of the term in the least squares problem. The weight will be
        applied to both the gradient and the Hessian matrix.

    loss: Optional[Union[str, Callable]] (default=None)
        The loss function to use for the term in the least squares problem (only for ``"rJ"`` terms !).
        The loss function will affect how the gradient and Hessian are computed
        from the residuals and Jacobian.
        If ``loss`` is a string, one of the predefined loss functions is used.
        Available loss functions are ``"linear"`` (:math:`\rho(x) = x`) [default for None],
        ``"cauchy"`` (:math:`\rho(x) = \log(1 + x)`),
        ``"arctan"`` (:math:`\rho(x) = \arctan(x)`), and
        ``"soft_l1"`` (:math:`\rho(x) = 2(\sqrt{1 + x} - 1)`).
        If ``loss`` is a callable, it is used as a custom loss function. The
        callable must take as input a 1D array-like containing the squared
        residuals and return three 1D array-like containing, respectively, the loss
        value, its first derivative, and its second derivative.
        When a custom loss function is provided, the ``loss`` property is set to
        ``"custom"`` and the corresponding callable is stored in the
        ``loss_func`` property.

    finite_difference: Optional[str] (default=None)
        The finite difference method used to numerically compute the Jacobian
        when ``jacobian_func`` is not provided.
        If ``None``, the Jacobian is not computed numerically. In this case,
        ``jacobian_func`` must be provided.
        Available methods are ``"central"``, ``"forward"``, and ``"backward"``.

    """

    __slots__ = [
        "_residual_func",
        "_jacobian_func",
        "_gradient_func",
        "_hessian_func",
        "_cost_func",
        "_weight",
        "_loss",
        "_loss_func",
    ]

    def __init__(
        self,
        *,
        residual_func: Optional[Callable] = None,
        jacobian_func: Optional[Callable] = None,
        gradient_func: Optional[Callable] = None,
        hessian_func: Optional[Callable] = None,
        cost_func: Optional[Callable] = None,
        weight: Real = 1.0,
        loss: Optional[Union[str, Callable]] = None,
        finite_difference: Optional[str] = None,
    ):
        if (residual_func is not None or jacobian_func is not None) and (
            gradient_func is not None or hessian_func is not None
        ):
            raise ValueError(
                "A Term object can be defined by either the residual and Jacobian "
                "functions, or the gradient and Hessian functions, but not both."
            )

        if (residual_func is None and jacobian_func is None) and (
            gradient_func is None and hessian_func is None
        ):
            raise ValueError(
                "A Term object must be defined by either the residual and Jacobian "
                "functions, or the gradient and Hessian functions."
            )

        if (gradient_func is None) != (hessian_func is None):
            raise ValueError(
                "Both gradient_func and hessian_func must be provided together."
            )

        if (residual_func is not None) and (
            jacobian_func is None and finite_difference is None
        ):
            raise ValueError(
                "Both residual_func and jacobian_func must be provided together (or use finite_difference to numerically compute the jacobian)."
            )

        if jacobian_func is not None and finite_difference is not None:
            raise ValueError(
                "finite_difference must be None if jacobian_func is provided."
            )

        if (residual_func is None) and (jacobian_func is not None):
            raise ValueError(
                "A term cannot have a Jacobian function without a residual function."
            )

        if residual_func is not None and not callable(residual_func):
            raise TypeError(
                f"residual_func must be a callable function, got {type(residual_func)}."
            )
        if jacobian_func is not None and not callable(jacobian_func):
            raise TypeError(
                f"jacobian_func must be a callable function, got {type(jacobian_func)}."
            )
        if gradient_func is not None and not callable(gradient_func):
            raise TypeError(
                f"gradient_func must be a callable function, got {type(gradient_func)}."
            )
        if hessian_func is not None and not callable(hessian_func):
            raise TypeError(
                f"hessian_func must be a callable function, got {type(hessian_func)}."
            )
        if cost_func is not None and not callable(cost_func):
            raise TypeError(
                f"cost_func must be a callable function, got {type(cost_func)}."
            )

        if not isinstance(weight, Real):
            raise TypeError("weight must be a real number.")
        if weight <= 0.0:
            raise ValueError("weight must be a positive number.")
        weight = float(weight)

        if (gradient_func is not None or hessian_func is not None) and loss is not None:
            raise ValueError(f"loss must be None for 'gH' terms.")

        if gradient_func is not None or hessian_func is not None:
            loss_name = None
            loss_func = None
        else:
            if loss is None:
                loss = "linear"

            if isinstance(loss, str):
                loss_name = loss.lower()
                if loss not in _IMPLEMENTED_LOSS_FUNCTIONS:
                    raise ValueError(
                        f"loss must be one of {_IMPLEMENTED_LOSS_FUNCTIONS}, got '{loss}'."
                    )
                loss_func = get_rho_function_by_name(loss_name)
            else:
                if not callable(loss):
                    raise TypeError(
                        f"loss must be a callable function, got {type(loss)}."
                    )
                loss_func = loss
                loss_name = "custom"

        if finite_difference is not None:
            if not isinstance(finite_difference, str):
                raise TypeError("finite_difference must be a string.")
            finite_difference = finite_difference.lower()
            if finite_difference not in _IMPLEMENTED_FINITE_DIFFERENCE_METHODS:
                raise ValueError(
                    f"finite_difference must be one of {_IMPLEMENTED_FINITE_DIFFERENCE_METHODS}, got '{finite_difference}'."
                )

        self._residual_func: Optional[Callable] = residual_func
        self._jacobian_func: Optional[Callable] = jacobian_func
        self._gradient_func: Optional[Callable] = gradient_func
        self._hessian_func: Optional[Callable] = hessian_func
        self._cost_func: Optional[Callable] = cost_func
        self._weight: float = weight
        self._loss: Optional[str] = loss_name
        self._loss_func: Optional[Callable] = loss_func

        # If residual but no Jacobian is provided, build the numerical Jacobian function
        if self._residual_func is not None and (
            self._jacobian_func is None and finite_difference is not None
        ):
            self._jacobian_func = build_numerical_jacobian(
                residual_func=self._residual_func,
                method=finite_difference,
                epsilon=1e-8,
            )

    @property
    def residual_func(self) -> Optional[Callable]:
        r"""
        [Get] the residual function :math:`\mathbf{r}_i: \mathbb{R}^{n_p} \rightarrow \mathbb{R}^{n_{r_i}}` of the term.

        The residual function take parameters as a 1D array-like of shape ``(n_parameters,)`` and return the residual vector as a
        1D array-like with shape ``(n_residuals,)``.

        .. note::

            The alias ``r_func`` is also available for ``residual_func`` for convenience.

        Returns
        -------
        Optional[Callable]
            The residual function of the term, or None if the term is ``"gH"``-type.
        """
        return self._residual_func

    @property
    def r_func(self) -> Optional[Callable]:
        return self.residual_func

    @property
    def jacobian_func(self) -> Optional[Callable]:
        r"""
        [Get] the Jacobian function :math:`\mathbf{J}_i: \mathbb{R}^{n_p} \rightarrow :math:`\mathbb{M}_{n_{r_i}, n_p}(\mathbb{R})`` of the term.

        The Jacobian function take parameters as a 1D array-like of shape ``(n_parameters,)`` and return the Jacobian matrix as a
        2D array-like with shape ``(n_residuals, n_parameters)``.

        .. note::

            The alias ``J_func`` is also available for ``jacobian_func`` for convenience.

        Returns
        -------
        Optional[Callable]
            The Jacobian function of the term, or None if the term is ``"gH"``-type.
        """
        return self._jacobian_func

    @property
    def J_func(self) -> Optional[Callable]:
        return self.jacobian_func

    @property
    def gradient_func(self) -> Optional[Callable]:
        r"""
        [Get] the gradient function :math:`\mathbf{g}_i: \mathbb{R}^{n_p} \rightarrow \mathbb{R}^{n_p}` of the term.

        The gradient function take parameters as a 1D array-like of shape ``(n_parameters,)`` and return the gradient vector as a
        1D array-like with shape ``(n_parameters,)``.

        .. note::

            The alias ``g_func`` is also available for ``gradient_func`` for convenience.

        Returns
        -------
        Optional[Callable]
            The gradient function of the term, or None if the term is ``"rJ"``-type.

        """
        return self._gradient_func

    @property
    def g_func(self) -> Optional[Callable]:
        return self.gradient_func

    @property
    def hessian_func(self) -> Optional[Callable]:
        r"""
        [Get] the Hessian function :math:`\mathbf{H}_i: \mathbb{R}^{n_p} \rightarrow :math:`\mathbb{M}_{n_p, n_p}(\mathbb{R})`` of the term.

        The Hessian function take parameters as a 1D array-like of shape ``(n_parameters,)`` and return the Hessian matrix as a
        2D array-like with shape ``(n_parameters, n_parameters)``.

        .. note::

            The alias ``H_func`` is also available for ``hessian_func`` for convenience.

        Returns
        -------
        Optional[Callable]
            The Hessian function of the term, or None if the term is ``"rJ"``-type.

        """
        return self._hessian_func

    @property
    def H_func(self) -> Optional[Callable]:
        return self.hessian_func

    @property
    def cost_func(self) -> Optional[Callable]:
        r"""
        [Get] the cost function :math:`C_i: \mathbb{R}^{n_p} \rightarrow \mathbb{R}`` of the term.

        The gradient function take parameters as a 1D array-like of shape ``(n_parameters,)`` and
        a number.

        .. note::

            The alias ``c_func`` is also available for ``cost_func`` for convenience.

        If None, the default cost during optimization will be
        :math:`\frac{1}{2} \sum_j \rho_i(\| \mathbf{r}_{i,j}(\mathbf{p}_{out}) \|^2)`
        for ``"rJ"`` terms and ``0.0`` for ``"gH"`` terms.

        Returns
        -------
        Optional[Callable]
            The cost function of the term, or None to use the default cost computation.

        """
        return self._cost_func

    @property
    def c_func(self) -> Optional[Callable]:
        return self.cost_func

    @property
    def weight(self) -> Real:
        r"""
        [Get/Set] the weight of the term in the least squares problem.

        Parameters
        ----------
        value : Optional[Real] (default=1.0)
            The new weight to set for the term. Must be a real number.

        Returns
        -------
        Real
            The current weight of the term in the least squares problem.
        """
        return self._weight

    @weight.setter
    def weight(self, value: Optional[Real]) -> None:
        if value is None:
            value = 1.0
        if not isinstance(value, Real):
            raise TypeError("weight must be a real number.")
        if value <= 0.0:
            raise ValueError("weight must be non-negative.")
        self._weight = float(value)

    @property
    def loss(self) -> Optional[str]:
        r"""
        [Get] the name of the loss function associated with the term in the least squares problem.

        .. note::

            If a custom loss function is used, ``'custom'`` is returned.

        Returns
        -------
        Optional[str]
            The current loss function of the term in the least squares problem or None for ``"gH"`` terms.
        """
        return self._loss

    @property
    def loss_func(self) -> Optional[Callable]:
        r"""
        [Get] the loss function :math:`\rho: \mathbb{R} \rightarrow \mathbb{R}`` of the term in the least squares problem.

        The loss function take squared-residuals as a 1D array-like of shape ``(n_parameters,)`` and
        return three 1D array-like:

        - The loss value with shape ``(n_parameters,)``.
        - The first derivative of the loss with shape ``(n_parameters,)``.
        - The second derivative of the loss with shape ``(n_parameters,)``.

        Returns
        -------
        Optional[Callable]
            The current loss function of the term in the least squares problem or None for ``"gH"`` terms.
        """
        return self._loss_func

    @property
    def type(self) -> str:
        r"""
        [Get] the type of the term, which can be either ``"rJ"`` for terms defined
        by residual and Jacobian functions, or ``"gH"`` for terms defined by second
        term and Hessian functions.

        Returns
        -------
        str
            The type of the term, either ``"rJ"`` or ``"gH"``.
        """
        if self._residual_func is not None and self._jacobian_func is not None:
            return "rJ"
        elif self._gradient_func is not None and self._hessian_func is not None:
            return "gH"
        else:
            raise ValueError(
                "Invalid Term: must be defined by either rJ or gH functions."
            )

    @classmethod
    def from_rJ(
        cls,
        residual_func: Optional[Callable] = None,
        jacobian_func: Optional[Callable] = None,
        *,
        weight: Optional[Real] = None,
        cost_func: Optional[Callable] = None,
        loss: Optional[str] = None,
        finite_difference: Optional[str] = None,
    ) -> Term:
        r"""
        Create a :class:`pysolvegn.Term` object from the residual and Jacobian functions.

        Parameters
        ----------
        residual_func : Optional[Callable] (default=None)
            The function to compute the residuals :math:`\mathbf{r}_i: \mathbb{R}^{n_p} \rightarrow \mathbb{R}^{n_{r_i}}`
            of the least squares problem. The function should take as inputs the parameters
            (1D array-like with shape ``(n_parameters,)``) and return the residuals as a
            1D array-like with shape ``(n_residuals,)`` representing each
            :math:`\mathbf{r}_{i,j}(\mathbf{p})` where :math:`j=0,...,n_{r_i}-1`.

        jacobian_func : Optional[Callable] (default=None)
            The function to compute the Jacobian matrix of the residuals with respect to the parameters.
            The function should take as inputs the parameters
            (1D array-like with shape ``(n_parameters,)``) and return the Jacobian matrix as a 2D array-like
            with shape ``(n_residuals, n_parameters)`` representing each
            :math:`\mathbf{J}_{i,(j,l)}(\mathbf{p}) = \frac{\partial \mathbf{r}_{i,j}}{\partial \mathbf{p}_l}` where :math:`j=0,...,n_{r_i}-1}`
            and :math:`l=0,...,n_{p}-1}`.
            Can be numericallty computed using ``finite_difference`` argument.

        cost_func: Optional[Callable] (default=None)
            The function to compute the cost of the term in the least squares problem.
            The function should take as inputs the parameters
            (1D array-like with shape ``(n_parameters,)``) and return the cost as a positive
            floating value independent of the ``weight``.
            If not provided, the default cost during optimization will be
            :math:`\frac{1}{2} \sum_j \rho_i(\| \mathbf{r}_{i,j}(\mathbf{p}_{out}) \|^2)`
            for ``"rJ"`` terms.

        weight: Real (default=1.0)
            The weight of the term in the least squares problem. The weight will be
            applied to both the gradient and the Hessian matrix.

        loss: Optional[Union[str, Callable]] (default=None)
            The loss function to use for the term in the least squares problem.
            The loss function will affect how the gradient and Hessian are computed
            from the residuals and Jacobian.
            If ``loss`` is a string, one of the predefined loss functions is used.
            Available loss functions are ``"linear"`` (:math:`\rho(x) = x`) [default for None],
            ``"cauchy"`` (:math:`\rho(x) = \log(1 + x)`),
            ``"arctan"`` (:math:`\rho(x) = \arctan(x)`), and
            ``"soft_l1"`` (:math:`\rho(x) = 2(\sqrt{1 + x} - 1)`).
            If ``loss`` is a callable, it is used as a custom loss function. The
            callable must take as input a 1D array-like containing the squared
            residuals and return three 1D array-like containing, respectively, the loss
            value, its first derivative, and its second derivative.
            When a custom loss function is provided, the ``loss`` property is set to
            ``"custom"`` and the corresponding callable is stored in the
            ``loss_func`` property.

        finite_difference: Optional[str] (default=None)
            The finite difference method used to numerically compute the Jacobian
            when ``jacobian_func`` is not provided.
            If ``None``, the Jacobian is not computed numerically. In this case,
            ``jacobian_func`` must be provided.
            Available methods are ``"central"``, ``"forward"``, and ``"backward"``.

        Returns
        -------
        Term
            A Term object defined by the given residual and Jacobian functions.

        """
        return cls(
            residual_func=residual_func,
            jacobian_func=jacobian_func,
            gradient_func=None,
            hessian_func=None,
            cost_func=cost_func,
            weight=weight,
            loss=loss,
            finite_difference=finite_difference,
        )

    @classmethod
    def from_gH(
        cls,
        gradient_func: Optional[Callable] = None,
        hessian_func: Optional[Callable] = None,
        *,
        weight: Optional[Real] = None,
        cost_func: Optional[Callable] = None,
    ) -> Term:
        r"""
        Create a :class:`pysolvegn.Term` object from the gradient and Hessian functions.

        Parameters
        ----------
        gradient_func: Optional[Callable] (default=None)
            The function to compute the gradient :math:`\mathbf{g}_i(\mathbf{p}) = \mathbf{J}_i^T \mathbf{r}_i`
            of the least squares problem.
            The function should take as inputs the parameters
            (1D array-like with shape ``(n_parameters,)``) and return the gradient as a
            1D array-like with shape ``(n_parameters,)``.

        hessian_func: Optional[Callable] (default=None)
            The function to compute the Hessian :math:`\mathbf{H}_i(\mathbf{p}) = \mathbf{J}_i^T \mathbf{J}_i`
            of the least squares problem.
            The function should take as inputs the parameters
            (1D array-like with shape ``(n_parameters,)``) and return the Hessian matrix as a
            2D array-like with shape ``(n_parameters, n_parameters)``.

        cost_func: Optional[Callable] (default=None)
            The function to compute the cost of the term in the least squares problem.
            The function should take as inputs the parameters
            (1D array-like with shape ``(n_parameters,)``) and return the cost as a positive
            floating value independent of the ``weight``.
            If not provided, the default cost during optimization will be ``0.0`` for ``"gH"`` terms.

        weight: Real (default=1.0)
            The weight of the term in the least squares problem. The weight will be
            applied to both the gradient and the Hessian matrix.


        Returns
        -------
        Term
            A Term object defined by the given gradient and Hessian functions.

        """
        return cls(
            residual_func=None,
            jacobian_func=None,
            gradient_func=gradient_func,
            hessian_func=hessian_func,
            weight=weight,
            loss="linear",
            finite_difference=None,
        )
