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
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy
import scipy.sparse

from .loss_functions import _build_tilde_R_and_tilde_J
from .term import Term
from .parametrization import Parametrization


@dataclass
class SystemState:
    r"""
    State of the least squares problem evaluated at given input parameters
    :math:`\mathbf{p}_{in}`.

    Attributes
    ----------
    term_parameters: numpy.ndarray
        The output parameters :math:`\mathbf{p}_{out} = P(\mathbf{p}_{in})`.

    residuals: List[Optional[numpy.ndarray]]
        The raw residuals :math:`\mathbf{r}_i(\mathbf{p}_{out})` of each term
        (None for ``gH`` terms).

    jacobians: List[Optional[Union[numpy.ndarray, scipy.sparse.spmatrix]]]
        The raw Jacobians :math:`\partial \mathbf{r}_i / \partial \mathbf{p}_{out}` of each
        term (None for ``gH`` terms).

    costs: Optional[List[float]]
        The cost of each term, without weight (None if the cost is not computed).

    cost: Optional[float]
        The total cost :math:`C = \sum_i w_i c_i` (None if the cost is not computed).

    hessian: Union[numpy.ndarray, scipy.sparse.spmatrix]
        The Gauss-Newton Hessian :math:`\mathbf{H}` with respect to :math:`\mathbf{p}_{in}`.

    second_term: numpy.ndarray
        The second term :math:`\mathbf{g}` with respect to :math:`\mathbf{p}_{in}`.
    """

    term_parameters: numpy.ndarray
    residuals: List[Optional[numpy.ndarray]]
    jacobians: List[Optional[Union[numpy.ndarray, scipy.sparse.spmatrix]]]
    costs: Optional[List[float]]
    cost: Optional[float]
    hessian: Union[numpy.ndarray, scipy.sparse.spmatrix]
    second_term: numpy.ndarray


@dataclass
class CostState:
    r"""
    Cost of the least squares problem evaluated at given output parameters
    :math:`\mathbf{p}_{out}` by :func:`_evaluate_cost`.

    It can be passed to :func:`_evaluate_system` at the same output parameters to
    reuse the residuals and the costs already computed (e.g. after an accepted
    Levenberg-Marquardt trial step).

    Attributes
    ----------
    cost: float
        The total cost :math:`C = \sum_i w_i c_i`.

    costs: List[float]
        The cost of each term, without weight.

    residuals: List[Optional[numpy.ndarray]]
        The raw residuals of each ``rJ`` term computed for the cost (None for ``gH``
        terms and for ``rJ`` terms with a ``cost_func``).
    """

    cost: float
    costs: List[float]
    residuals: List[Optional[numpy.ndarray]]


def _check_shape(name: str, array, expected: tuple) -> None:
    r"""
    Raise a ValueError if ``array.shape`` (numpy array or scipy sparse matrix)
    is not ``expected``.
    """
    shape = getattr(array, "shape", None)
    if shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {shape}.")


def _compute_term_parameters(
    parametrization: Optional[Parametrization],
    in_parameters: numpy.ndarray,
) -> numpy.ndarray:
    r"""
    Compute the output parameters :math:`\mathbf{p}_{out} = P(\mathbf{p}_{in})`
    (identity if no parametrization is provided).

    Returns
    -------
    numpy.ndarray
        The output parameters with shape ``(n_out,)``.
    """
    if parametrization is None:
        return in_parameters
    term_parameters = numpy.asarray(parametrization.p_func(in_parameters), dtype=numpy.float64)
    if term_parameters.ndim != 1:
        raise ValueError(
            f"The parametric function must return a 1D array, got shape {term_parameters.shape}."
        )
    return term_parameters


def _compute_jacobian_P(
    parametrization: Optional[Parametrization],
    in_parameters: numpy.ndarray,
    n_out: int,
) -> Optional[Union[numpy.ndarray, scipy.sparse.spmatrix]]:
    r"""
    Compute the Jacobian of the parametrization
    :math:`\mathbf{J}_P = \partial \mathbf{p}_{out} / \partial \mathbf{p}_{in}`
    and check its shape ``(n_out, n_in)``.

    Returns
    -------
    Optional[Union[numpy.ndarray, scipy.sparse.spmatrix]]
        The Jacobian with shape ``(n_out, n_in)``, or None if no parametrization is
        provided (identity: no chain rule to apply).
    """
    if parametrization is None:
        return None
    jacobian_P = parametrization.J_func(in_parameters)
    if not scipy.sparse.issparse(jacobian_P):
        jacobian_P = numpy.asarray(jacobian_P, dtype=numpy.float64)
    _check_shape(
        "The Jacobian of the parametrization", jacobian_P, (n_out, in_parameters.shape[0])
    )
    return jacobian_P


def _compute_rhos(term: Term, residuals: numpy.ndarray) -> tuple:
    r"""
    Compute :math:`\rho`, :math:`\rho'` and :math:`\rho''` of the squared residuals of a
    ``rJ`` term (shortcut for the linear loss).
    """
    if term.loss == "linear":
        return residuals**2, 1.0, 0.0
    return term.loss_func(residuals**2)


def _compute_term_cost(
    term: Term,
    term_parameters: numpy.ndarray,
    rho: Optional[numpy.ndarray],
) -> float:
    r"""
    Cost of one term, without weight: ``cost_func`` if provided, otherwise
    :math:`\frac{1}{2} \sum_j \rho(\|r_j\|^2)` for ``rJ`` terms and ``0.0`` for ``gH`` terms.
    """
    if term.c_func is not None:
        return float(term.c_func(term_parameters))
    if term.type == "rJ":
        return float(0.5 * numpy.sum(rho))
    if term.type == "gH":
        return 0.0
    raise ValueError(f"Unknown term type: {term.type}")


def _evaluate_cost(
    terms: Sequence[Term],
    term_parameters: numpy.ndarray,
) -> CostState:
    r"""
    Compute only the cost at the given output parameters :math:`\mathbf{p}_{out}`,
    without computing the Jacobians, the Hessian or the second term.

    Used to test a trial step (Levenberg-Marquardt): one call costs one evaluation of
    the residual functions of the ``rJ`` terms (and of the ``cost_func``). The returned
    :class:`CostState` can be passed to :func:`_evaluate_system` at the same output
    parameters to avoid evaluating the residuals twice.

    .. note::

        The output parameters are not checked: the caller must check them for NaN or
        Inf values before calling this function (see :func:`_compute_term_parameters`).
    """
    costs: List[float] = []
    residuals: List[Optional[numpy.ndarray]] = []
    for term in terms:
        r = None
        rho = None
        if term.type == "rJ" and term.c_func is None:
            r = numpy.asarray(term.r_func(term_parameters), dtype=numpy.float64)
            rho = _compute_rhos(term, r)[0]
        residuals.append(r)
        costs.append(_compute_term_cost(term, term_parameters, rho))

    cost = float(sum(term.weight * c for term, c in zip(terms, costs)))
    return CostState(cost=cost, costs=costs, residuals=residuals)


def _evaluate_system(
    terms: Sequence[Term],
    term_parameters: numpy.ndarray,
    jacobian_P: Optional[Union[numpy.ndarray, scipy.sparse.spmatrix]],
    *,
    compute_cost: bool,
    cost_state: Optional[CostState] = None,
) -> SystemState:
    r"""
    Evaluate the full system at the given output parameters (steps 3 to 5 of the loop):

    - ``rJ`` terms: residuals, Jacobians, robust loss (``r -> r~``, ``J -> J~``) and
      chain rule of the parametrization (``J~ -> J~ @ J_P``),
    - ``gH`` terms: gradient and Hessian with the chain rule
      (``H -> J_P.T H J_P``, ``g -> J_P.T g``),
    - assembly of ``H = sum(w_i H_i)`` and ``g = sum(w_i g_i)``,
    - costs of each term and total cost (if ``compute_cost``).

    The output parameters :math:`\mathbf{p}_{out}` and the Jacobian of the
    parametrization :math:`\mathbf{J}_P` are computed by the caller with
    :func:`_compute_term_parameters` and :func:`_compute_jacobian_P` (``jacobian_P`` is
    None if no parametrization is provided). The output parameters are not checked
    here: the caller must check them for NaN or Inf values before calling this function.

    The shapes returned by the user functions are checked against
    ``n_out = term_parameters.shape[0]``:

    - ``rJ`` terms: residuals ``(n_r,)`` and Jacobian ``(n_r, n_out)``,
    - ``gH`` terms: gradient ``(n_out,)`` and Hessian ``(n_out, n_out)``.

    Jacobians and Hessians can be numpy arrays or scipy sparse matrices.

    If ``cost_state`` is given (computed by :func:`_evaluate_cost` at the SAME output
    parameters), its residuals and costs are reused instead of being recomputed.

    Returns
    -------
    SystemState
        The evaluated system.
    """
    n_out = term_parameters.shape[0]

    residuals: List[Optional[numpy.ndarray]] = []
    jacobians: List[Optional[Union[numpy.ndarray, scipy.sparse.spmatrix]]] = []
    costs: Optional[List[float]] = [] if compute_cost else None
    hessian = 0.0
    second_term = 0.0

    for index, term in enumerate(terms):
        weight = term.weight

        if term.type == "rJ":
            if cost_state is not None and cost_state.residuals[index] is not None:
                r = cost_state.residuals[index]  # already computed at the same p_out
            else:
                r = numpy.asarray(term.r_func(term_parameters), dtype=numpy.float64)
            if r.ndim != 1:
                raise ValueError(
                    f"The residual function of the term {index} must return a 1D array, got shape {r.shape}."
                )
            J = term.J_func(term_parameters)
            if not scipy.sparse.issparse(J):
                J = numpy.asarray(J, dtype=numpy.float64)
            _check_shape(f"The Jacobian of the term {index}", J, (r.shape[0], n_out))
            rho, rho_prime, rho_double_prime = _compute_rhos(term, r)

            # Robust loss: r~, J~ (r and J are kept unchanged for the history)
            if term.loss != "linear":
                r_tilde, J_tilde = _build_tilde_R_and_tilde_J(r, J, rho_prime, rho_double_prime)
            else:
                r_tilde, J_tilde = r, J

            # Chain rule of the parametrization: J~ @ J_P
            if jacobian_P is not None:
                J_tilde = J_tilde @ jacobian_P

            hessian = hessian + weight * (J_tilde.T @ J_tilde)
            second_term = second_term + weight * (J_tilde.T @ r_tilde)
            residuals.append(r)
            jacobians.append(J)

        elif term.type == "gH":
            rho = None
            g = numpy.asarray(term.g_func(term_parameters), dtype=numpy.float64)
            _check_shape(f"The gradient of the term {index}", g, (n_out,))
            H = term.H_func(term_parameters)
            if not scipy.sparse.issparse(H):
                H = numpy.asarray(H, dtype=numpy.float64)
            _check_shape(f"The Hessian of the term {index}", H, (n_out, n_out))

            # Chain rule of the parametrization: J_P.T H J_P and J_P.T g
            if jacobian_P is not None:
                H = jacobian_P.T @ H @ jacobian_P
                g = jacobian_P.T @ g

            hessian = hessian + weight * H
            second_term = second_term + weight * g
            residuals.append(None)
            jacobians.append(None)

        else:
            raise ValueError(f"Unknown term type: {term.type}")

        if compute_cost:
            if cost_state is not None:
                costs.append(cost_state.costs[index])  # already computed at the same p_out
            else:
                costs.append(_compute_term_cost(term, term_parameters, rho))

    cost = None
    if compute_cost:
        cost = float(sum(term.weight * c for term, c in zip(terms, costs)))

    return SystemState(
        term_parameters=term_parameters,
        residuals=residuals,
        jacobians=jacobians,
        costs=costs,
        cost=cost,
        hessian=hessian,
        second_term=numpy.asarray(second_term, dtype=numpy.float64).ravel(),
    )