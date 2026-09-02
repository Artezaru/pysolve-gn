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
from numbers import Real

from .derivation import build_numerical_jacobian
from .implemented_conf import (
    _IMPLEMENTED_LOSS_FUNCTIONS,
    _IMPLEMENTED_FINITE_DIFFERENCE_METHODS,
)
from .rho_functions import get_rho_function_by_name


class Term(object):
    r"""
    A class representing a term in the least squares problem, which can be either a
    residual term or a regularization term.

    For the following definition, the :math:`i`-th term in the least squares
    problem is denoted as :math:`\mathbf{R}_i(\mathbf{p})`

    .. math::

        \min_{\mathbf{p}} \frac{1}{2} \sum_{i} w_i \sum_j \rho_i\left(\| \mathbf{r}_{i,j}(\mathbf{p}) \|^2\right)

    Each term can be defined by two callable functions:

    1. The **Residual** and **Jacobian** functions

        The residual function :math:`\mathbf{r}_i(\mathbf{p})` to minimize
        and the Jacobian function
        :math:`\mathbf{J}_i(\mathbf{p}) = \frac{\partial \mathbf{r}_i}{\partial \mathbf{p}}`
        to compute the Jacobian matrix of the residuals with respect to the parameters.

        Then the solver will build the second term :math:`\mathbf{g}(\mathbf{p})` and
        the Hessian of the least squares problem :math:`\mathbf{H}(\mathbf{p})` as:

        .. math::

            \mathbf{g} = w_i \tilde{\mathbf{J}}_i^T \tilde{\mathbf{r}}_i

        .. math::

            \mathbf{H} = w_i \tilde{\mathbf{J}}_i^T \tilde{\mathbf{J}}_i

        such that :math:`\mathbf{H} \Delta \mathbf{p} = -\mathbf{g}`.

        .. seealso::

            For more details on notations for the optimization problem and the algorithm,
            please refer to the mathematical section of the documentation.


    2. The **Second term** and **Hessian** functions

        The hessian function :math:`\mathbf{H}_i(\mathbf{p})` to compute the Hessian of
        the least squares problem and the second term function
        :math:`\mathbf{g}_i(\mathbf{p})` to compute the second term of the
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
    as well as the second term and Hessian functions for each term,
    which can be used to build the least squares problem.


    Parameters
    ----------
    residual_func : Optional[Callable] (default=None)
        The function to compute the residuals in the least squares problem.
        The function should take as inputs the parameters
        (array-like with shape ``(n_parameters,)``) and return the residuals as a
        1D numpy array with shape ``(n_residuals,)`` representing each
        :math:`\mathbf{r}_{j}(\mathbf{p})` where :math:`j=0,...,n_{\text{residuals}-1}`.

    jacobian_func : Optional[Callable] (default=None)
        The function to compute the Jacobian matrix of the residuals with respect to the parameters.
        The function should take as inputs the parameters
        (array-like with shape ``(n_parameters,)``) and return the Jacobian matrix as a 2D numpy array
        with shape ``(n_residuals, n_parameters)`` representing each
        :math:`\mathbf{J}_{j}(\mathbf{p})` where :math:`j=0,...,n_{\text{residuals}-1}`
        such that :math:`\mathbf{J}_j(\mathbf{p}) = \frac{\partial \mathbf{r}_j}{\partial \mathbf{p}}`.
        If not provided, the Jacobian will be computed numerically using finite differences.

    second_term_func: Optional[Callable] (default=None)
        The function to compute the second term :math:`\mathbf{g}_i(\mathbf{p})`
        of the least squares problem.
        The function should take as inputs the parameters
        (array-like with shape ``(n_parameters,)``) and return the second term as a
        1D numpy array with shape ``(n_parameters,)``.
        If provided, loss must be "linear".

    hessian_func: Optional[Callable] (default=None)
        The function to compute the Hessian :math:`\mathbf{H}_i(\mathbf{p})`
        of the least squares problem.
        The function should take as inputs the parameters
        (array-like with shape ``(n_parameters,)``) and return the Hessian as a
        2D numpy array with shape ``(n_parameters, n_parameters)``.
        If provided, loss must be "linear".

    weight: Real (default=1.0)
        The weight of the term in the least squares problem. The weight will be
        applied to both the residuals and the Jacobian matrix, as well as the
        second term and Hessian if provided.

    loss: str (default="linear")
        The loss function to use for the term in the least squares problem. The loss
        function will be applied to the residuals of the term, and will affect how
        the second term and Hessian are computed from the residuals and Jacobian.
        Available loss functions are "linear", "cauchy", "arctan", and "soft_l1".
        Must be "linear" if the term is defined by the second term and Hessian functions.

    finite_difference: str (default="central")
        The finite difference method to use for computing the numerical Jacobian if the
        Jacobian function is not provided. Must be one of ``"central"``, ``"forward"``, or ``"backward"``.


    Notes
    -----

    - Each term must be defined by either the residual and Jacobian functions, or the second term and Hessian functions, but not both.
    - Hessian and second term must be provided together.
    - If Residual is given without Jacobian, the jacobian will be computed numerically using finite differences.


    Version
    -------

    - 0.1.0: Initial version of the :class:`Term` class.
    - 0.3.0: Removing the possibility to update the finite difference method after initialization.

    """

    __slots__ = [
        "_residual_func",
        "_jacobian_func",
        "_second_term_func",
        "_hessian_func",
        "_weight",
        "_loss",
        "_finite_difference",
        "_use_finite_difference_jacobian",
    ]

    def __init__(
        self,
        residual_func: Optional[Callable] = None,
        jacobian_func: Optional[Callable] = None,
        second_term_func: Optional[Callable] = None,
        hessian_func: Optional[Callable] = None,
        weight: Optional[Real] = None,
        loss: Optional[str] = None,
        finite_difference: Optional[str] = None,
    ):
        if (residual_func is not None or jacobian_func is not None) and (
            second_term_func is not None or hessian_func is not None
        ):
            raise ValueError(
                "A Term object can be defined by either the residual and Jacobian "
                "functions, or the second term and Hessian functions, but not both."
            )

        if (residual_func is None and jacobian_func is None) and (
            second_term_func is None and hessian_func is None
        ):
            raise ValueError(
                "A Term object must be defined by either the residual and Jacobian "
                "functions, or the second term and Hessian functions."
            )

        if (second_term_func is None) != (hessian_func is None):
            raise ValueError(
                "Both second_term_func and hessian_func must be provided together."
            )

        if (residual_func is None) and (jacobian_func is not None):
            raise ValueError(
                "A term cannot have a Jacobian function without a residual function."
            )

        if residual_func is not None and not callable(residual_func):
            raise ValueError(
                f"residual_func must be a callable function, got {type(residual_func)}."
            )
        if jacobian_func is not None and not callable(jacobian_func):
            raise ValueError(
                f"jacobian_func must be a callable function, got {type(jacobian_func)}."
            )
        if second_term_func is not None and not callable(second_term_func):
            raise ValueError(
                f"second_term_func must be a callable function, got {type(second_term_func)}."
            )
        if hessian_func is not None and not callable(hessian_func):
            raise ValueError(
                f"hessian_func must be a callable function, got {type(hessian_func)}."
            )

        if weight is None:
            weight = 1.0
        if not isinstance(weight, Real):
            raise ValueError("weight must be a real number.")
        weight = float(weight)

        if loss is None:
            loss = "linear"
        if not isinstance(loss, str):
            raise ValueError("loss must be a string.")
        loss = loss.lower()
        if loss not in _IMPLEMENTED_LOSS_FUNCTIONS:
            raise ValueError(
                f"loss must be one of {_IMPLEMENTED_LOSS_FUNCTIONS}, got '{loss}'."
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

        self._residual_func = residual_func
        self._jacobian_func = jacobian_func
        self._second_term_func = second_term_func
        self._hessian_func = hessian_func
        self._weight = weight
        self._loss = loss.lower()
        self._finite_difference = finite_difference.lower()
        self._use_finite_difference_jacobian = False

        # If residual but no Jacobian is provided, build the numerical Jacobian function
        if self._residual_func is not None and self._jacobian_func is None:
            self._jacobian_func = build_numerical_jacobian(
                residual_func=self._residual_func,
                method=self._finite_difference,
                epsilon=1e-8,
            )
            self._use_finite_difference_jacobian = True

    @property
    def residual_func(self) -> Optional[Callable]:
        r"""
        [Get] the residual function of the term.

        .. note::

            The alias ``r_func`` is also available for ``residual_func`` for convenience.

        Returns
        -------
        Optional[Callable]
            The residual function of the term, or None if the term is defined by the second term
            and Hessian functions instead of the residual and Jacobian functions.
        """
        return self._residual_func

    @property
    def r_func(self) -> Optional[Callable]:
        return self.residual_func

    @property
    def jacobian_func(self) -> Optional[Callable]:
        r"""
        [Get] the Jacobian function of the term.

        .. note::

            The alias ``J_func`` is also available for ``jacobian_func`` for convenience.

        Returns
        -------
        Optional[Callable]
            The Jacobian function of the term, or None if the term is defined by the second term
            and Hessian functions instead of the residual and Jacobian functions.
        """
        return self._jacobian_func

    @property
    def J_func(self) -> Optional[Callable]:
        return self.jacobian_func

    @property
    def second_term_func(self) -> Optional[Callable]:
        r"""
        [Get] the second term function of the term.

        .. note::

            The alias ``g_func`` is also available for ``second_term_func`` for convenience.

        Returns
        -------
        Optional[Callable]
            The second term function of the term, or None if the term is defined by the residual
            and Jacobian functions instead of the second term and Hessian functions.

        """
        return self._second_term_func

    @property
    def g_func(self) -> Optional[Callable]:
        return self.second_term_func

    @property
    def hessian_func(self) -> Optional[Callable]:
        r"""
        [Get] the Hessian function of the term.

        .. note::

            The alias ``H_func`` is also available for ``hessian_func`` for convenience.

        Returns
        -------
        Optional[Callable]
            The Hessian function of the term, or None if the term is defined by the residual
            and Jacobian functions instead of the second term and Hessian functions.

        """
        return self._hessian_func

    @property
    def H_func(self) -> Optional[Callable]:
        return self.hessian_func

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
            raise ValueError("weight must be a real number.")
        if value < 0:
            raise ValueError("weight must be non-negative.")
        self._weight = float(value)

    @property
    def loss(self) -> str:
        r"""
        [Get/Set] the loss function of the term in the least squares problem.

        Parameters
        ----------
        value : Optional[str] (default="linear")
            The new loss function to set for the term. Must be a string and one of
            "linear", "cauchy", "arctan", or "soft_l1".


        Returns
        -------
        str
            The current loss function of the term in the least squares problem.
        """
        return self._loss

    @loss.setter
    def loss(self, value: Optional[str]) -> None:
        if value is None:
            value = "linear"
        if not isinstance(value, str):
            raise ValueError("loss must be a string.")
        value = value.lower()
        if value not in _IMPLEMENTED_LOSS_FUNCTIONS:
            raise ValueError(
                f"loss must be one of {_IMPLEMENTED_LOSS_FUNCTIONS}, got '{value}'."
            )
        self._loss = value.lower()

    @property
    def finite_difference(self) -> str:
        r"""
        [Get] the finite difference method for numerical Jacobian computation of the term.

        Returns
        -------
        str
            The current finite difference method for numerical Jacobian computation of the term.
        """
        return self._finite_difference

    @property
    def rho_function(self) -> Callable:
        r"""
        [Get] the rho function corresponding to the current loss function of the term.

        Returns
        -------
        Callable
            The rho function corresponding to the current loss function of the term.
        """
        return get_rho_function_by_name(self._loss)

    @property
    def type(self) -> str:
        r"""
        [Get] the type of the term, which can be either "rJ" for terms defined
        by residual and Jacobian functions, or "gH" for terms defined by second
        term and Hessian functions.

        Returns
        -------
        str
            The type of the term, either "rJ" or "gH".
        """
        if self._residual_func is not None and self._jacobian_func is not None:
            return "rJ"
        elif self._second_term_func is not None and self._hessian_func is not None:
            return "gH"
        else:
            raise ValueError(
                "Invalid Term: must be defined by either rJ or gH functions."
            )

    @property
    def use_finite_difference(self) -> bool:
        r"""
        [Get] whether the term is using a finite difference Jacobian function.

        Returns
        -------
        bool
            True if the term is using a finite difference Jacobian function, False otherwise.
        """
        return self._use_finite_difference_jacobian

    @classmethod
    def from_rJ(
        cls,
        residual_func: Optional[Callable] = None,
        jacobian_func: Optional[Callable] = None,
        weight: Optional[Real] = None,
        *,
        loss: Optional[str] = None,
        finite_difference: Optional[str] = None,
    ) -> Term:
        r"""
        Create a Term object from the residual and Jacobian functions.

        The term will be defined by the residual and Jacobian functions,
        and the second term and Hessian will be computed from the residuals and Jacobian.

        Parameters
        ----------
        residual_func : Optional[Callable] (default=None)
            The function to compute the residuals in the least squares problem.
            The function should take as inputs the parameters
            (array-like with shape ``(n_parameters,)``) and return the residuals as a
            1D numpy array with shape ``(n_residuals,)`` representing each
            :math:`\mathbf{R}_{j}(\mathbf{p})` where :math:`j=0,...,n_{\text{residuals}-1}`.

        jacobian_func : Optional[Callable] (default=None)
            The function to compute the Jacobian matrix of the residuals with respect to the parameters.
            The function should take as inputs the parameters
            (array-like with shape ``(n_parameters,)``) and return the Jacobian matrix as a 2D numpy array
            with shape ``(n_residuals, n_parameters)`` representing each
            :math:`\mathbf{J}_{j}(\mathbf{p})` where :math:`j=0,...,n_{\text{residuals}-1}`.

        weight: Real (default=1.0)
            The weight of the term in the least squares problem. The weight will be
            applied to both the residuals and the Jacobian matrix.

        loss: str (default="linear")
            The loss function to use for the term in the least squares problem. The loss
            function will be applied to the residuals of the term, and will affect how
            the second term and Hessian are computed from the residuals and Jacobian.
            Available loss functions are "linear", "cauchy", "arctan", and "soft_l1".

        finite_difference: str (default="central")
            The finite difference method to use for computing the numerical Jacobian if the
            Jacobian function is not provided. Must be one of ``"central"``, ``"forward"``, or ``"backward"``.


        Returns
        -------
        Term
            A Term object defined by the given residual and Jacobian functions, weight, loss, and
            finite difference method.

        """
        return cls(
            residual_func=residual_func,
            jacobian_func=jacobian_func,
            second_term_func=None,
            hessian_func=None,
            weight=weight,
            loss=loss,
            finite_difference=finite_difference,
        )

    @classmethod
    def from_gH(
        cls,
        second_term_func: Optional[Callable] = None,
        hessian_func: Optional[Callable] = None,
        weight: Optional[Real] = None,
    ) -> Term:
        r"""
        Create a Term object from the second term and Hessian functions.

        The term will be defined by the second term and Hessian functions,
        and the residual and Jacobian will not be used.

        Parameters
        ----------
        second_term_func: Optional[Callable] (default=None)
            The function to compute the second term of the least squares problem.
            The function should take as inputs the parameters
            (array-like with shape ``(n_parameters,)``) and return the second term as a
            1D numpy array with shape ``(n_parameters,)``.

        hessian_func: Optional[Callable] (default=None)
            The function to compute the Hessian of the least squares problem.
            The function should take as inputs the parameters
            (array-like with shape ``(n_parameters,)``) and return the Hessian as a
            2D numpy array with shape ``(n_parameters, n_parameters)``.

        weight: Real (default=1.0)
            The weight of the term in the least squares problem. The weight will be
            applied to both the second term and Hessian.

        Returns
        -------
        Term
            A Term object defined by the given second term and Hessian functions, weight, and loss.

        """
        return cls(
            residual_func=None,
            jacobian_func=None,
            second_term_func=second_term_func,
            hessian_func=hessian_func,
            weight=weight,
            loss="linear",
            finite_difference=None,
        )
