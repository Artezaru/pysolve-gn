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
from typing import Callable, Optional, Tuple, Union
from numbers import Real

import numpy
from numpy.typing import ArrayLike

from .derivation import build_batch_numerical_jacobian
from .implemented_conf import (
    _IMPLEMENTED_LOSS_FUNCTIONS,
    _IMPLEMENTED_FINITE_DIFFERENCE_METHODS,
)
from .loss_functions import (
    get_rho_function_by_name,
    scale_rho_function,
    _validate_loss_scale,
    _build_batch_tilde_R_and_tilde_J,
)


class BatchTerm(object):
    r"""
    A class representing a term in a batch of independent least squares problems.

    This is the batched counterpart of :class:`pysolvegn.Term`. Instead of a single
    parameter vector with shape ``(n_parameters,)``, the solver optimizes ``k``
    independent parameter vectors stored in an array with shape ``(k, n_parameters)``.
    For each problem :math:`k` of the batch:

    .. math::

        \min_{\mathbf{p}_k} \frac{1}{2} \sum_{i} w_{i,k} \sum_j \rho_i\left(\| \mathbf{r}_{i,j}(\mathbf{p}_k) \|^2\right)

    where :math:`i` indexes the different terms (:class:`pysolvegn.BatchTerm`), :math:`j`
    indexes the residuals within each term and :math:`w_{i,k}` is the weight of the term
    :math:`i` for the problem :math:`k`: either the same scalar weight for all the
    problems, or one weight per problem (vector weight with shape ``(k,)``).

    As for :class:`pysolvegn.Term`, a term is defined either by its **residual** and
    **Jacobian** functions (``"rJ"`` term) or by its **gradient** and **Hessian**
    functions (``"gH"`` term), but not both.

    .. important::

        **The number of evaluated problems** ``m`` **is NOT fixed.**

        All the callables take two arguments ``(parameters, indices)``:

        - ``parameters``: the parameters of the problems currently evaluated, with
          shape ``(m, n_parameters)`` where ``m`` is **any** integer ``1 <= m <= k``.
        - ``indices``: the indices of these problems in the full batch, with shape
          ``(m,)``. ``parameters[a]`` are the parameters of the problem ``indices[a]``.

        The solver removes the stopped problems at each iteration, so ``m`` changes
        from one call to the next (and the finite differences may also call the functions
        with any subset). The callables must therefore never assume a fixed ``m`` nor
        ``m == k``: all their outputs must have ``m`` as first dimension, and the data
        attached to each problem must be selected with ``indices``
        (e.g. ``observations[indices]``), never with a full-size array.
        

    Parameters
    ----------
    residual_func : Optional[Callable] (default=None)
        The function computing the residuals. Returns an array with shape
        ``(m, n_residuals)``.

    jacobian_func : Optional[Callable] (default=None)
        The function computing the Jacobian of the residuals with respect to the
        parameters. Returns an array with shape ``(m, n_residuals, n_parameters)``.
        Can be numerically computed using ``finite_difference`` argument.

    gradient_func : Optional[Callable] (default=None)
        The function computing the gradient :math:`\mathbf{g}_i = \mathbf{J}_i^T \mathbf{r}_i`.
        Returns an array with shape ``(m, n_parameters)``.

    hessian_func : Optional[Callable] (default=None)
        The function computing the Hessian :math:`\mathbf{H}_i = \mathbf{J}_i^T \mathbf{J}_i`.
        Returns an array with shape ``(m, n_parameters, n_parameters)``.

    cost_func : Optional[Callable] (default=None)
        The function computing the cost of the term for each problem, independent of
        the ``weight``. Returns a non-negative array with shape ``(m,)``.
        If not provided, the default cost is
        :math:`\frac{1}{2} \sum_j \rho_i(\| \mathbf{r}_{i,j}(\mathbf{p}_k) \|^2)`
        for ``"rJ"`` terms and ``0.0`` for ``"gH"`` terms.

    weight : Union[Real, ArrayLike] (default=1.0)
        The weight of the term, applied to both the gradient and the Hessian.

        - A scalar: the same strictly positive weight is used for all the problems.
        - A 1D array-like with shape ``(k,)``: ``weight[k]`` is the weight of the term
          for the problem ``k`` of the batch. The values must be finite and strictly
          positive.

    loss : Optional[Union[str, Callable]] (default=None)
        The loss function (only for ``"rJ"`` terms). Available loss functions are
        ``"linear"`` [default for None], ``"cauchy"``, ``"arctan"``, ``"soft_l1"``,
        ``"huber"`` and ``"tukey"`` (see :class:`pysolvegn.Term`).
        If ``loss`` is a callable, it must take the squared residuals with shape
        ``(m, n_residuals)`` and return three arrays with the same shape containing the
        loss value, its first derivative and its second derivative.
        The ``loss`` property is then ``"custom"``.

    loss_scale : Real (default=1.0)
        The soft threshold :math:`C` of the loss function (only for ``"rJ"`` terms),
        in the unit of the residuals. The loss function (predefined **or custom**) is
        replaced by :math:`\rho_C(x) = C^2 \rho(x / C^2)`, same as ``f_scale`` in
        ``scipy.optimize.least_squares``. The same scale is used for all the problems.
        No effect on the ``"linear"`` loss. Must be finite and strictly positive.

    finite_difference : Optional[str] (default=None)
        The finite difference method used to numerically compute the Jacobian when
        ``jacobian_func`` is not provided (see :func:`pysolvegn.build_batch_numerical_jacobian`).
        Available methods are ``"central"``, ``"forward"`` and ``"backward"``.

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
        "_loss_scale",
    ]

    def __init__(
        self,
        *,
        residual_func: Optional[Callable] = None,
        jacobian_func: Optional[Callable] = None,
        gradient_func: Optional[Callable] = None,
        hessian_func: Optional[Callable] = None,
        cost_func: Optional[Callable] = None,
        weight: Union[Real, ArrayLike] = 1.0,
        loss: Optional[Union[str, Callable]] = None,
        loss_scale: Real = 1.0,
        finite_difference: Optional[str] = None,
    ):
        if (residual_func is not None or jacobian_func is not None) and (
            gradient_func is not None or hessian_func is not None
        ):
            raise ValueError(
                "A BatchTerm object can be defined by either the residual and Jacobian "
                "functions, or the gradient and Hessian functions, but not both."
            )

        if (residual_func is None and jacobian_func is None) and (
            gradient_func is None and hessian_func is None
        ):
            raise ValueError(
                "A BatchTerm object must be defined by either the residual and Jacobian "
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

        for name, func in (
            ("residual_func", residual_func),
            ("jacobian_func", jacobian_func),
            ("gradient_func", gradient_func),
            ("hessian_func", hessian_func),
            ("cost_func", cost_func),
        ):
            if func is not None and not callable(func):
                raise TypeError(f"{name} must be a callable function, got {type(func)}.")

        if (gradient_func is not None or hessian_func is not None) and loss is not None:
            raise ValueError("loss must be None for 'gH' terms.")

        loss_scale = _validate_loss_scale(loss_scale)
        if (gradient_func is not None or hessian_func is not None) and loss_scale != 1.0:
            raise ValueError("loss_scale must be 1.0 for 'gH' terms.")

        if gradient_func is not None or hessian_func is not None:
            loss_name = None
            loss_func = None
        else:
            if loss is None:
                loss = "linear"

            if isinstance(loss, str):
                loss_name = loss.lower()
                if loss_name not in _IMPLEMENTED_LOSS_FUNCTIONS:
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

            # Generic scaling rho_C(x) = C^2 rho(x / C^2) (identity for "linear" or C = 1)
            if loss_name != "linear":
                loss_func = scale_rho_function(loss_func, loss_scale)

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
        self._weight: Union[float, numpy.ndarray] = self._validate_weight(weight)
        self._loss: Optional[str] = loss_name
        self._loss_func: Optional[Callable] = loss_func
        self._loss_scale: float = loss_scale

        # If residual but no Jacobian is provided, build the numerical Jacobian function
        if self._residual_func is not None and (
            self._jacobian_func is None and finite_difference is not None
        ):
            self._jacobian_func = build_batch_numerical_jacobian(
                residual_func=self._residual_func,
                method=finite_difference,
                epsilon=1e-8,
            )

    # ------------------------------------------------------------------
    # Weight
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_weight(value: Optional[Union[Real, ArrayLike]]) -> Union[float, numpy.ndarray]:
        if value is None:
            return 1.0

        if isinstance(value, Real):
            if not numpy.isfinite(value):
                raise ValueError("weight must be finite.")
            if value <= 0.0:
                raise ValueError("weight must be a positive number.")
            return float(value)

        try:
            array = numpy.array(value, dtype=numpy.float64)  # copy
        except (TypeError, ValueError) as error:
            raise TypeError(
                "weight must be a real number or a 1D array-like of real numbers."
            ) from error

        if array.ndim == 0:
            return BatchTerm._validate_weight(float(array))
        if array.ndim != 1:
            raise ValueError(
                f"A vector weight must be a 1D array-like with shape (k,), got shape {array.shape}."
            )
        if array.size == 0:
            raise ValueError("A vector weight must not be empty.")
        if not numpy.all(numpy.isfinite(array)):
            raise ValueError("A vector weight must contain only finite values.")
        if numpy.any(array <= 0.0):
            raise ValueError("A vector weight must contain only strictly positive values.")

        array.setflags(write=False)
        return array

    @property
    def weight(self) -> Union[float, numpy.ndarray]:
        r"""
        [Get/Set] the weight of the term.

        Either a strictly positive float shared by all the problems, or a read-only
        1D array with shape ``(k,)`` of strictly positive values (one weight per problem).

        Parameters
        ----------
        value : Optional[Union[Real, ArrayLike]] (default=1.0)
            The new weight. ``None`` resets it to ``1.0``.

        Returns
        -------
        Union[float, numpy.ndarray]
            The current weight of the term.
        """
        return self._weight

    @weight.setter
    def weight(self, value: Optional[Union[Real, ArrayLike]]) -> None:
        self._weight = self._validate_weight(value)

    @property
    def is_vector_weight(self) -> bool:
        r"""
        [Get] whether the weight is a vector (one value per problem).

        Returns
        -------
        bool
            ``True`` if the weight is a 1D array with shape ``(k,)``, ``False`` if it is a scalar.
        """
        return isinstance(self._weight, numpy.ndarray)

    @property
    def n_problems(self) -> Optional[int]:
        r"""
        [Get] the number of problems ``k`` imposed by a vector weight.

        Returns
        -------
        Optional[int]
            The length of the vector weight, or ``None`` for a scalar weight
            (compatible with any number of problems).
        """
        return self._weight.shape[0] if self.is_vector_weight else None

    def weight_at(self, indices: ArrayLike) -> numpy.ndarray:
        r"""
        Get the weights of the term for the given problems.

        Parameters
        ----------
        indices : ArrayLike
            Indices of the problems in the full batch, with shape ``(m,)``.

        Returns
        -------
        numpy.ndarray
            The weights with shape ``(m,)``.
        """
        indices = numpy.asarray(indices, dtype=numpy.intp)
        if self.is_vector_weight:
            return self._weight[indices]
        return numpy.full(indices.shape, self._weight, dtype=numpy.float64)

    # ------------------------------------------------------------------
    # Callables
    # ------------------------------------------------------------------
    @property
    def residual_func(self) -> Optional[Callable]:
        r"""
        [Get] the batched residual function ``(parameters (m, n_p), indices (m,)) -> (m, n_r)``.

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
        [Get] the batched Jacobian function ``(parameters (m, n_p), indices (m,)) -> (m, n_r, n_p)``.

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
        [Get] the batched gradient function ``(parameters (m, n_p), indices (m,)) -> (m, n_p)``.

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
        [Get] the batched Hessian function ``(parameters (m, n_p), indices (m,)) -> (m, n_p, n_p)``.

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
        [Get] the batched cost function ``(parameters (m, n_p), indices (m,)) -> (m,)``.

        .. note::

            The alias ``c_func`` is also available for ``cost_func`` for convenience.

        If None, the default cost is
        :math:`\frac{1}{2} \sum_j \rho_i(\| \mathbf{r}_{i,j}(\mathbf{p}_k) \|^2)`
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
    def loss(self) -> Optional[str]:
        r"""
        [Get] the name of the loss function of the term (``'custom'`` for a callable,
        None for ``"gH"`` terms).

        Returns
        -------
        Optional[str]
            The name of the loss function.
        """
        return self._loss

    @property
    def loss_func(self) -> Optional[Callable]:
        r"""
        [Get] the loss function of the term.

        The loss function takes the squared residuals with shape ``(m, n_residuals)`` and
        returns the loss value, its first derivative and its second derivative, each with
        shape ``(m, n_residuals)``.

        If ``loss_scale`` is not ``1.0``, this is the scaled loss function
        :math:`\rho_C(x) = C^2 \rho(x / C^2)`.

        Returns
        -------
        Optional[Callable]
            The loss function, or None for ``"gH"`` terms.
        """
        return self._loss_func

    @property
    def loss_scale(self) -> float:
        r"""
        [Get] the soft threshold :math:`C` of the loss function, in the unit of the
        residuals (always ``1.0`` for ``"gH"`` terms).

        Returns
        -------
        float
            The loss scale of the term.
        """
        return self._loss_scale

    @property
    def type(self) -> str:
        r"""
        [Get] the type of the term: ``"rJ"`` (residual and Jacobian functions) or
        ``"gH"`` (gradient and Hessian functions).

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
                "Invalid BatchTerm: must be defined by either rJ or gH functions."
            )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_inputs(
        parameters: ArrayLike, indices: ArrayLike
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        parameters = numpy.asarray(parameters, dtype=numpy.float64)
        indices = numpy.asarray(indices, dtype=numpy.intp)
        if parameters.ndim != 2:
            raise ValueError(
                f"parameters must be a 2D array with shape (m, n_parameters), got shape {parameters.shape}."
            )
        if indices.shape != (parameters.shape[0],):
            raise ValueError(
                f"indices must have shape ({parameters.shape[0]},), got {indices.shape}."
            )
        return parameters, indices

    def _evaluate_cost_func(
        self, parameters: numpy.ndarray, indices: numpy.ndarray
    ) -> numpy.ndarray:
        cost = numpy.asarray(self._cost_func(parameters, indices), dtype=numpy.float64)
        if cost.shape != (parameters.shape[0],):
            raise ValueError(
                f"cost_func must return an array with shape ({parameters.shape[0]},), got {cost.shape}."
            )
        return cost

    def _evaluate_residuals(
        self, parameters: numpy.ndarray, indices: numpy.ndarray
    ) -> numpy.ndarray:
        residuals = numpy.asarray(self._residual_func(parameters, indices), dtype=numpy.float64)
        if residuals.ndim != 2 or residuals.shape[0] != parameters.shape[0]:
            raise ValueError(
                f"residual_func must return an array with shape ({parameters.shape[0]}, n_residuals), got {residuals.shape}."
            )
        return residuals

    def _rho(self, squared_residuals: numpy.ndarray):
        if self._loss == "linear":
            return squared_residuals, 1.0, 0.0
        return self._loss_func(squared_residuals)

    def evaluate_cost(self, parameters: ArrayLike, indices: ArrayLike) -> numpy.ndarray:
        r"""
        Compute the weighted cost of the term for each problem, without computing the
        Jacobian (useful to test a trial step).

        Parameters
        ----------
        parameters : ArrayLike
            Parameters of the evaluated problems with shape ``(m, n_parameters)``.

        indices : ArrayLike
            Indices of the evaluated problems in the full batch with shape ``(m,)``.

        Returns
        -------
        numpy.ndarray
            The weighted cost :math:`w_{i,k} C_i(\mathbf{p}_k)` with shape ``(m,)``.
        """
        parameters, indices = self._prepare_inputs(parameters, indices)
        weights = self.weight_at(indices)

        if self._cost_func is not None:
            cost = self._evaluate_cost_func(parameters, indices)
        elif self.type == "rJ":
            residuals = self._evaluate_residuals(parameters, indices)
            rho = self._rho(residuals**2)[0]
            cost = 0.5 * numpy.sum(rho, axis=1)
        else:
            cost = numpy.zeros(parameters.shape[0], dtype=numpy.float64)

        return weights * cost

    def evaluate(
        self,
        parameters: ArrayLike,
        indices: ArrayLike,
        *,
        compute_cost: bool = True,
    ) -> Tuple[Optional[numpy.ndarray], numpy.ndarray, numpy.ndarray]:
        r"""
        Compute the weighted cost, gradient and Gauss-Newton Hessian of the term for
        each problem.

        For an ``"rJ"`` term, the robust loss is taken into account through the modified
        residuals and Jacobians :math:`\tilde{\mathbf{r}}` and :math:`\tilde{\mathbf{J}}`
        (same formulas as :class:`pysolvegn.Term`), and:

        .. math::

            \mathbf{g}_k = w_{i,k} \tilde{\mathbf{J}}_k^T \tilde{\mathbf{r}}_k
            \qquad
            \mathbf{H}_k = w_{i,k} \tilde{\mathbf{J}}_k^T \tilde{\mathbf{J}}_k

        For a ``"gH"`` term, the provided gradient and Hessian are multiplied by the weights.

        Parameters
        ----------
        parameters : ArrayLike
            Parameters of the evaluated problems with shape ``(m, n_parameters)``.

        indices : ArrayLike
            Indices of the evaluated problems in the full batch with shape ``(m,)``.

        compute_cost : bool, optional (default=True)
            If False, the cost is not computed and None is returned instead.

        Returns
        -------
        cost : Optional[numpy.ndarray]
            The weighted cost with shape ``(m,)``, or None if ``compute_cost`` is False.

        gradient : numpy.ndarray
            The weighted gradient with shape ``(m, n_parameters)``.

        hessian : numpy.ndarray
            The weighted Gauss-Newton Hessian with shape ``(m, n_parameters, n_parameters)``.
        """
        parameters, indices = self._prepare_inputs(parameters, indices)
        m, n_parameters = parameters.shape
        weights = self.weight_at(indices)
        cost = None

        if self.type == "rJ":
            residuals = self._evaluate_residuals(parameters, indices)
            jacobian = numpy.asarray(self._jacobian_func(parameters, indices), dtype=numpy.float64)
            if jacobian.shape != (m, residuals.shape[1], n_parameters):
                raise ValueError(
                    f"jacobian_func must return an array with shape "
                    f"({m}, {residuals.shape[1]}, {n_parameters}), got {jacobian.shape}."
                )

            rho, rho_prime, rho_double_prime = self._rho(residuals**2)
            if self._loss != "linear":
                residuals, jacobian = _build_batch_tilde_R_and_tilde_J(
                    residuals, jacobian, rho_prime, rho_double_prime
                )

            jacobian_t = jacobian.transpose(0, 2, 1)
            gradient = (jacobian_t @ residuals[..., None])[..., 0]
            hessian = jacobian_t @ jacobian

            if compute_cost and self._cost_func is None:
                cost = 0.5 * numpy.sum(rho, axis=1)

        else:  # gH
            gradient = numpy.asarray(self._gradient_func(parameters, indices), dtype=numpy.float64)
            hessian = numpy.asarray(self._hessian_func(parameters, indices), dtype=numpy.float64)
            if gradient.shape != (m, n_parameters):
                raise ValueError(
                    f"gradient_func must return an array with shape ({m}, {n_parameters}), got {gradient.shape}."
                )
            if hessian.shape != (m, n_parameters, n_parameters):
                raise ValueError(
                    f"hessian_func must return an array with shape "
                    f"({m}, {n_parameters}, {n_parameters}), got {hessian.shape}."
                )
            if compute_cost and self._cost_func is None:
                cost = numpy.zeros(m, dtype=numpy.float64)

        if compute_cost:
            if self._cost_func is not None:
                cost = self._evaluate_cost_func(parameters, indices)
            cost = weights * cost

        return cost, weights[:, None] * gradient, weights[:, None, None] * hessian

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_rJ(
        cls,
        residual_func: Optional[Callable] = None,
        jacobian_func: Optional[Callable] = None,
        *,
        cost_func: Optional[Callable] = None,
        weight: Union[Real, ArrayLike] = 1.0,
        loss: Optional[Union[str, Callable]] = None,
        loss_scale: Real = 1.0,
        finite_difference: Optional[str] = None,
    ) -> BatchTerm:
        r"""
        Create a :class:`pysolvegn.BatchTerm` object from the batched residual and
        Jacobian functions.

        See :class:`pysolvegn.BatchTerm` for the description of the parameters.

        Returns
        -------
        BatchTerm
            A BatchTerm object defined by the given residual and Jacobian functions.
        """
        return cls(
            residual_func=residual_func,
            jacobian_func=jacobian_func,
            gradient_func=None,
            hessian_func=None,
            cost_func=cost_func,
            weight=weight,
            loss=loss,
            loss_scale=loss_scale,
            finite_difference=finite_difference,
        )

    @classmethod
    def from_gH(
        cls,
        gradient_func: Optional[Callable] = None,
        hessian_func: Optional[Callable] = None,
        *,
        cost_func: Optional[Callable] = None,
        weight: Union[Real, ArrayLike] = 1.0,
    ) -> BatchTerm:
        r"""
        Create a :class:`pysolvegn.BatchTerm` object from the batched gradient and
        Hessian functions.

        See :class:`pysolvegn.BatchTerm` for the description of the parameters.

        Returns
        -------
        BatchTerm
            A BatchTerm object defined by the given gradient and Hessian functions.
        """
        return cls(
            residual_func=None,
            jacobian_func=None,
            gradient_func=gradient_func,
            hessian_func=hessian_func,
            cost_func=cost_func,
            weight=weight,
            loss=None,
            finite_difference=None,
        )