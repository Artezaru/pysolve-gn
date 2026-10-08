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
from typing import List, Optional, Sequence

import numpy

from .loss_functions import _build_batch_tilde_R_and_tilde_J
from .batch_term import BatchTerm
from .batch_parametrization import BatchParametrization


@dataclass
class BatchSystemState:
    r"""
    State of a batch of independent least squares problems evaluated at given input
    parameters :math:`\mathbf{p}_{in,k}` (batched counterpart of :class:`SystemState`).

    All the arrays have the number ``m`` of evaluated problems as first dimension.

    Attributes
    ----------
    term_parameters: numpy.ndarray
        The output parameters :math:`\mathbf{p}_{out,k} = P(\mathbf{p}_{in,k})` with shape
        ``(m, n_out)``.

    residuals: List[Optional[numpy.ndarray]]
        The raw residuals of each term with shape ``(m, n_r_i)`` (None for ``gH`` terms).

    jacobians: List[Optional[numpy.ndarray]]
        The raw Jacobians :math:`\partial \mathbf{r}_i / \partial \mathbf{p}_{out}` of each
        term with shape ``(m, n_r_i, n_out)`` (None for ``gH`` terms).

    costs: Optional[List[numpy.ndarray]]
        The cost of each term for each problem, without weight, with shape ``(m,)``
        (None if the cost is not computed).

    cost: Optional[numpy.ndarray]
        The total cost of each problem :math:`C_k = \sum_i w_{i,k} c_{i,k}` with shape
        ``(m,)`` (None if the cost is not computed).

    hessian: numpy.ndarray
        The Gauss-Newton Hessians :math:`\mathbf{H}_k` with respect to
        :math:`\mathbf{p}_{in,k}`, with shape ``(m, n_in, n_in)``.

    second_term: numpy.ndarray
        The second terms :math:`\mathbf{g}_k` with respect to :math:`\mathbf{p}_{in,k}`,
        with shape ``(m, n_in)``.
    """

    term_parameters: numpy.ndarray
    residuals: List[Optional[numpy.ndarray]]
    jacobians: List[Optional[numpy.ndarray]]
    costs: Optional[List[numpy.ndarray]]
    cost: Optional[numpy.ndarray]
    hessian: numpy.ndarray
    second_term: numpy.ndarray


@dataclass
class BatchCostState:
    r"""
    Cost of a batch of independent least squares problems evaluated at given output
    parameters :math:`\mathbf{p}_{out,k}` by :func:`_evaluate_batch_cost`
    (batched counterpart of :class:`CostState`).

    It can be passed to :func:`_evaluate_batch_system` at the same output parameters
    **and the same indices** to reuse the residuals and the costs already computed
    (e.g. after an accepted Levenberg-Marquardt trial step).

    Attributes
    ----------
    cost: numpy.ndarray
        The total cost of each problem :math:`C_k = \sum_i w_{i,k} c_{i,k}` with shape ``(m,)``.

    costs: List[numpy.ndarray]
        The cost of each term for each problem, without weight, with shape ``(m,)``.

    residuals: List[Optional[numpy.ndarray]]
        The raw residuals of each ``rJ`` term computed for the cost, with shape
        ``(m, n_r_i)`` (None for ``gH`` terms and for ``rJ`` terms with a ``cost_func``).
    """

    cost: numpy.ndarray
    costs: List[numpy.ndarray]
    residuals: List[Optional[numpy.ndarray]]

    def select(self, mask: numpy.ndarray) -> BatchCostState:
        r"""
        Extract the cost state of a subset of the problems.

        Useful when some problems are removed from the processing between the call to
        :func:`_evaluate_batch_cost` and the call to :func:`_evaluate_batch_system`
        (the ``indices`` passed to the latter must then be ``indices[mask]``).

        Parameters
        ----------
        mask : numpy.ndarray
            Boolean mask with shape ``(m,)`` or integer positions of the problems to keep.

        Returns
        -------
        BatchCostState
            The cost state of the selected problems.
        """
        return BatchCostState(
            cost=self.cost[mask],
            costs=[c[mask] for c in self.costs],
            residuals=[None if r is None else r[mask] for r in self.residuals],
        )


def _check_shape(name: str, array, expected: tuple) -> None:
    r"""
    Raise a ValueError if ``array.shape`` is not ``expected``.
    """
    shape = getattr(array, "shape", None)
    if shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {shape}.")


def _compute_batch_term_parameters(
    parametrization: Optional[BatchParametrization],
    in_parameters: numpy.ndarray,
) -> numpy.ndarray:
    r"""
    Compute the output parameters :math:`\mathbf{p}_{out,k} = P(\mathbf{p}_{in,k})` of
    the ``m`` evaluated problems (identity if no parametrization is provided).

    Parameters
    ----------
    parametrization : Optional[BatchParametrization]
        The batched parametrization, or None for the identity.

    in_parameters : numpy.ndarray
        The input parameters with shape ``(m, n_in)``. The same transformation is
        applied to each problem (no indices).

    Returns
    -------
    numpy.ndarray
        The output parameters with shape ``(m, n_out)``.
    """
    if parametrization is None:
        return in_parameters
    m = in_parameters.shape[0]
    term_parameters = numpy.asarray(
        parametrization.p_func(in_parameters), dtype=numpy.float64
    )
    if term_parameters.ndim != 2 or term_parameters.shape[0] != m:
        raise ValueError(
            f"The parametric function must return an array with shape ({m}, n_out), "
            f"got shape {term_parameters.shape}."
        )
    return term_parameters


def _compute_batch_jacobian_P(
    parametrization: Optional[BatchParametrization],
    in_parameters: numpy.ndarray,
    n_out: int,
) -> Optional[numpy.ndarray]:
    r"""
    Compute the Jacobians of the parametrization
    :math:`\mathbf{J}_{P,k} = \partial \mathbf{p}_{out,k} / \partial \mathbf{p}_{in,k}`
    of the ``m`` evaluated problems and check their shape ``(m, n_out, n_in)``.

    Returns
    -------
    Optional[numpy.ndarray]
        The Jacobians with shape ``(m, n_out, n_in)``, or None if no parametrization is
        provided (identity: no chain rule to apply).
    """
    if parametrization is None:
        return None
    m, n_in = in_parameters.shape
    jacobian_P = numpy.asarray(
        parametrization.J_func(in_parameters), dtype=numpy.float64
    )
    _check_shape("The Jacobian of the parametrization", jacobian_P, (m, n_out, n_in))
    return jacobian_P


def _compute_batch_rhos(term: BatchTerm, residuals: numpy.ndarray) -> tuple:
    r"""
    Compute :math:`\rho`, :math:`\rho'` and :math:`\rho''` of the squared residuals
    with shape ``(m, n_r)`` of a ``rJ`` term (shortcut for the linear loss).
    """
    if term.loss == "linear":
        return residuals**2, 1.0, 0.0
    return term.loss_func(residuals**2)


def _compute_batch_term_cost(
    term: BatchTerm,
    term_parameters: numpy.ndarray,
    indices: numpy.ndarray,
    rho: Optional[numpy.ndarray],
) -> numpy.ndarray:
    r"""
    Cost of one term for each evaluated problem, without weight, with shape ``(m,)``:
    ``cost_func`` if provided, otherwise :math:`\frac{1}{2} \sum_j \rho(\|r_{k,j}\|^2)`
    for ``rJ`` terms and ``0.0`` for ``gH`` terms.
    """
    m = term_parameters.shape[0]
    if term.c_func is not None:
        cost = numpy.asarray(term.c_func(term_parameters, indices), dtype=numpy.float64)
        _check_shape("The cost function of a term", cost, (m,))
        return cost
    if term.type == "rJ":
        return 0.5 * numpy.sum(rho, axis=1)
    if term.type == "gH":
        return numpy.zeros(m, dtype=numpy.float64)
    raise ValueError(f"Unknown term type: {term.type}")


def _evaluate_batch_residuals(
    term: BatchTerm,
    index: int,
    term_parameters: numpy.ndarray,
    indices: numpy.ndarray,
) -> numpy.ndarray:
    r"""
    Evaluate the residuals of a ``rJ`` term and check their shape ``(m, n_r)``.
    """
    m = term_parameters.shape[0]
    r = numpy.asarray(term.r_func(term_parameters, indices), dtype=numpy.float64)
    if r.ndim != 2 or r.shape[0] != m:
        raise ValueError(
            f"The residual function of the term {index} must return an array with shape "
            f"({m}, n_residuals), got shape {r.shape}."
        )
    return r


def _sum_weighted_costs(
    terms: Sequence[BatchTerm],
    costs: List[numpy.ndarray],
    indices: numpy.ndarray,
) -> numpy.ndarray:
    r"""
    Total cost of each problem :math:`C_k = \sum_i w_{i,k} c_{i,k}` with shape ``(m,)``.
    """
    cost = numpy.zeros(indices.shape[0], dtype=numpy.float64)
    for term, c in zip(terms, costs):
        cost = cost + term.weight_at(indices) * c
    return cost


def _evaluate_batch_cost(
    terms: Sequence[BatchTerm],
    term_parameters: numpy.ndarray,
    indices: numpy.ndarray,
) -> BatchCostState:
    r"""
    Compute only the cost of each evaluated problem at the given output parameters
    :math:`\mathbf{p}_{out,k}`, without computing the Jacobians, the Hessians or the
    second terms (batched counterpart of :func:`_evaluate_cost`).

    Used to test a trial step (Levenberg-Marquardt): one call costs one evaluation of
    the residual functions of the ``rJ`` terms (and of the ``cost_func``). The returned
    :class:`BatchCostState` can be passed to :func:`_evaluate_batch_system` at the same
    output parameters and indices (or after :meth:`BatchCostState.select`) to avoid
    evaluating the residuals twice.

    .. note::

        The output parameters are not checked: the caller must check them for NaN or
        Inf values before calling this function (see :func:`_compute_batch_term_parameters`).

    Parameters
    ----------
    terms : Sequence[BatchTerm]
        The terms of the problems.

    term_parameters : numpy.ndarray
        The output parameters with shape ``(m, n_out)``.

    indices : numpy.ndarray
        The indices of the evaluated problems in the full batch with shape ``(m,)``.

    Returns
    -------
    BatchCostState
        The cost of each evaluated problem.
    """
    costs: List[numpy.ndarray] = []
    residuals: List[Optional[numpy.ndarray]] = []
    for index, term in enumerate(terms):
        r = None
        rho = None
        if term.type == "rJ" and term.c_func is None:
            r = _evaluate_batch_residuals(term, index, term_parameters, indices)
            rho = _compute_batch_rhos(term, r)[0]
        residuals.append(r)
        costs.append(_compute_batch_term_cost(term, term_parameters, indices, rho))

    cost = _sum_weighted_costs(terms, costs, indices)
    return BatchCostState(cost=cost, costs=costs, residuals=residuals)


def _evaluate_batch_system(
    terms: Sequence[BatchTerm],
    term_parameters: numpy.ndarray,
    indices: numpy.ndarray,
    jacobian_P: Optional[numpy.ndarray],
    *,
    compute_cost: bool,
    cost_state: Optional[BatchCostState] = None,
) -> BatchSystemState:
    r"""
    Evaluate the ``m`` independent systems at the given output parameters (batched
    counterpart of :func:`_evaluate_system`):

    - ``rJ`` terms: residuals, Jacobians, robust loss (``r -> r~``, ``J -> J~``) and
      chain rule of the parametrization (``J~_k -> J~_k @ J_P,k``),
    - ``gH`` terms: gradients and Hessians with the chain rule
      (``H_k -> J_P,k.T H_k J_P,k``, ``g_k -> J_P,k.T g_k``),
    - assembly of ``H_k = sum(w_i,k H_i,k)`` and ``g_k = sum(w_i,k g_i,k)``,
    - costs of each term and total cost of each problem (if ``compute_cost``).

    The output parameters and the Jacobians of the parametrization are computed by the
    caller with :func:`_compute_batch_term_parameters` and :func:`_compute_batch_jacobian_P`
    (``jacobian_P`` is None if no parametrization is provided). The output parameters are
    not checked here: the caller must check them for NaN or Inf values before calling
    this function.

    The shapes returned by the user functions are checked against
    ``(m, n_out) = term_parameters.shape``:

    - ``rJ`` terms: residuals ``(m, n_r)`` and Jacobians ``(m, n_r, n_out)``,
    - ``gH`` terms: gradients ``(m, n_out)`` and Hessians ``(m, n_out, n_out)``.

    If ``cost_state`` is given (computed by :func:`_evaluate_batch_cost` at the SAME
    output parameters and indices), its residuals and costs are reused instead of being
    recomputed.

    Parameters
    ----------
    terms : Sequence[BatchTerm]
        The terms of the problems.

    term_parameters : numpy.ndarray
        The output parameters with shape ``(m, n_out)``.

    indices : numpy.ndarray
        The indices of the evaluated problems in the full batch with shape ``(m,)``.

    jacobian_P : Optional[numpy.ndarray]
        The Jacobians of the parametrization with shape ``(m, n_out, n_in)``, or None.

    compute_cost : bool
        If True, the costs are computed (or reused from ``cost_state``).

    cost_state : Optional[BatchCostState], optional (default=None)
        Cost state at the same output parameters and indices to reuse.

    Returns
    -------
    BatchSystemState
        The evaluated systems.
    """
    m, n_out = term_parameters.shape
    n_in = n_out if jacobian_P is None else jacobian_P.shape[2]

    if cost_state is not None:
        _check_shape("The cost of the cost_state", cost_state.cost, (m,))

    residuals: List[Optional[numpy.ndarray]] = []
    jacobians: List[Optional[numpy.ndarray]] = []
    costs: Optional[List[numpy.ndarray]] = [] if compute_cost else None
    hessian = numpy.zeros((m, n_in, n_in), dtype=numpy.float64)
    second_term = numpy.zeros((m, n_in), dtype=numpy.float64)

    for index, term in enumerate(terms):
        weight = term.weight_at(indices)  # (m,)

        if term.type == "rJ":
            if cost_state is not None and cost_state.residuals[index] is not None:
                r = cost_state.residuals[index]  # already computed at the same p_out
            else:
                r = _evaluate_batch_residuals(term, index, term_parameters, indices)
            J = numpy.asarray(term.J_func(term_parameters, indices), dtype=numpy.float64)
            _check_shape(f"The Jacobian of the term {index}", J, (m, r.shape[1], n_out))
            rho, rho_prime, rho_double_prime = _compute_batch_rhos(term, r)

            # Robust loss: r~, J~ (r and J are kept unchanged for the history)
            if term.loss != "linear":
                r_tilde, J_tilde = _build_batch_tilde_R_and_tilde_J(
                    r, J, rho_prime, rho_double_prime
                )
            else:
                r_tilde, J_tilde = r, J

            # Chain rule of the parametrization: J~_k @ J_P,k
            if jacobian_P is not None:
                J_tilde = J_tilde @ jacobian_P  # (m, n_r, n_in)

            J_tilde_T = J_tilde.transpose(0, 2, 1)  # (m, n_in, n_r)
            hessian += weight[:, None, None] * (J_tilde_T @ J_tilde)
            second_term += weight[:, None] * (J_tilde_T @ r_tilde[..., None])[..., 0]
            residuals.append(r)
            jacobians.append(J)

        elif term.type == "gH":
            rho = None
            g = numpy.asarray(term.g_func(term_parameters, indices), dtype=numpy.float64)
            _check_shape(f"The gradient of the term {index}", g, (m, n_out))
            H = numpy.asarray(term.H_func(term_parameters, indices), dtype=numpy.float64)
            _check_shape(f"The Hessian of the term {index}", H, (m, n_out, n_out))

            # Chain rule of the parametrization: J_P,k.T H_k J_P,k and J_P,k.T g_k
            if jacobian_P is not None:
                jacobian_P_T = jacobian_P.transpose(0, 2, 1)  # (m, n_in, n_out)
                H = jacobian_P_T @ H @ jacobian_P
                g = (jacobian_P_T @ g[..., None])[..., 0]

            hessian += weight[:, None, None] * H
            second_term += weight[:, None] * g
            residuals.append(None)
            jacobians.append(None)

        else:
            raise ValueError(f"Unknown term type: {term.type}")

        if compute_cost:
            if cost_state is not None:
                costs.append(cost_state.costs[index])  # already computed at the same p_out
            else:
                costs.append(_compute_batch_term_cost(term, term_parameters, indices, rho))

    cost = None
    if compute_cost:
        cost = _sum_weighted_costs(terms, costs, indices)

    return BatchSystemState(
        term_parameters=term_parameters,
        residuals=residuals,
        jacobians=jacobians,
        costs=costs,
        cost=cost,
        hessian=hessian,
        second_term=second_term,
    )