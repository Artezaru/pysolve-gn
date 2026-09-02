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

from typing import Optional, Sequence, Union, Tuple, Callable, Dict, List, Any
from numbers import Real, Integral
from numpy.typing import ArrayLike

import numpy
import scipy
import time

from .implemented_conf import (
    _IMPLEMENTED_HISTORY_DETAILS,
)

from .rho_functions import (
    _build_tilde_R_and_tilde_J,
)

from .term import Term
from .parametrization import Parametrization


def solve(
    terms: Sequence[Term],
    p0: ArrayLike,
    parametrization: Optional[Parametrization] = None,
    *,
    max_iteration: Optional[Integral] = None,
    max_time: Optional[Real] = None,
    ftol: Optional[Real] = None,
    xtol: Optional[Real] = None,
    gtol: Optional[Real] = None,
    atol: Optional[Real] = None,
    ptol: Optional[Real] = None,
    callback_func: Optional[Callable[[Dict], bool]] = None,
    update_func: Optional[Callable[[numpy.ndarray, numpy.ndarray], ArrayLike]] = None,
    verbosity: Integral = 0,
    naninf: bool = True,
    history: bool = False,
    history_details: Optional[Union[str, Sequence[str]]] = None,
) -> Tuple[numpy.ndarray, Optional[List[Dict]]]:
    r"""
    Function to solve the least squares problem using the Gauss-Newton method
    with robust cost functions.

    The function accepts multiple terms in the least squares problem,
    each with its own residual function, Jacobian function, weight, and loss function,
    solving the following optimization problem:

    .. math::

        \min_{\mathbf{p}_{in}} \frac{1}{2} \sum_{i} w_i \sum_j
        \rho_i\left(
            \left\|
            \mathbf{r}_{i,j}\left(P(\mathbf{p}_{in})\right)
            \right\|^2
        \right)

    where :math:`\mathbf{p}_{in}` represents the parameters actually optimized
    by the solver, :math:`P` is a parametric transformation such that:

    .. math::

        \mathbf{p}_{out} = P(\mathbf{p}_{in})

    represents the parameters passed to the residual and Jacobian functions
    of the terms.

    The input parameters :math:`\mathbf{p}_{in}` have shape
    ``(n_parameters,)``.

    If no parametrization is provided, the identity transformation is implicitly
    used:

    .. math::

        P(\mathbf{p}_{in}) = \mathbf{p}_{in}

    Here :math:`w_i` is a weight for each sub least squares problem, and
    :math:`\rho_i` is a robust cost function for each sub least squares problem.

    .. seealso::

        For more details on the notations for the optimization problem and the
        algorithm, please refer to the mathematical section of the documentation.

    The cost of each term defined by the residuals is estimated as:

    .. math::

        \frac{1}{2} \sum_j
        \rho_i\left(
            \left\|
            \mathbf{r}_{i,j}(P(\mathbf{p}_{in}))
            \right\|^2
        \right)

    .. important::

        For terms defined directly by a Hessian and gradient, only a local quadratic model
        of the cost is available. Since this model is defined up to an arbitrary constant,
        an absolute cost value cannot be uniquely determined. For reporting consistency,
        the cost contribution of these terms is set to 0.

    Parameters
    ----------
    terms: Sequence[Term]
        The list of terms defining the least squares problem.
        Each term should be an instance of the :class:`Term` class containing
        the residual function, Jacobian function, weight, and loss function
        defining the term.
        The residual and Jacobian functions of each term are evaluated using
        the output parameters :math:`\mathbf{p}_{out} = P(\mathbf{p}_{in})`.

    p0: ArrayLike
        The initial guess for the parameters optimized by the solver, with shape
        ``(n_parameters,)``.
        If a parametrization is provided, ``p0`` represents the initial input
        parameters :math:`\mathbf{p}_{in}` of the parametrization. The initial
        output parameters passed to the terms are obtained as
        :math:`\mathbf{p}_{out} = P(\mathbf{p}_{in})`.
        If no parametrization is provided, ``p0`` directly represents the
        parameters passed to the terms.
        The array will not be modified by this function. A copy of the parameters
        will be used for the optimization process.

    parametrization: Optional[Parametrization], optional (default=None)
        The parametrization defining the transformation from the parameters
        optimized by the solver to the parameters passed to the terms.
        If provided, the solver optimizes the input parameters
        :math:`\mathbf{p}_{in}` with shape ``(n_parameters,)`` of this parametrization.

    max_iteration: Optional[Integral], optional (default=None)
        Stop criterion by the number of iterations.
        The optimization process is stopped when the number of iterations
        exceeds ``max_iteration``. If None, no limit on the
        number of iterations is considered.

    max_time: Optional[Real], optional (default=None)
        Stop criterion by the elapsed time of optimization.
        The optimization process is stopped when the time elapsed since the
        beginning of the optimization exceeds ``max_time`` seconds. If None, no limit
        on the computation time is considered.

    ftol: Optional[Real], optional (default=None)
        Stop criterion by the change of the cost function value.
        The optimization process is stopped when ``dF < ftol * F`` where F is the cost
        function value and dF is the change of the cost function value between two
        iterations. If None, this criterion is not considered.

    xtol: Optional[Real], optional (default=None)
        Stop criterion by the change of the optimized parameters.
        The optimization process is stopped when
        ``||dp|| < xtol * (xtol + ||p||)``
        where p is the input parameters :math:`\mathbf{p}_{in}` and dp is the
        change of the input parameters between two iterations.
        If None, this criterion is not considered.

    gtol: Optional[Real], optional (default=None)
        Stop criterion by the optimality value.
        The optimization process is stopped when the optimality verifies
        ``norm(g, ord=numpy.inf) < gtol`` where g is the scaled second term.
        If None, this criterion is not considered.

    atol: Optional[Real], optional (default=None)
        Stop criterion by the absolute cost function value.
        The optimization process is stopped when ``F < atol`` where F is the cost
        function value.
        If None, this criterion is not considered.

    ptol: Optional[Real], optional (default=None)
        Stop criterion by the change of the optimized parameters.
        The optimization process is stopped when
        ``norm(dp, ord=numpy.inf) < ptol``
        where dp is the change of the input parameters :math:`\mathbf{p}_{in}`.
        If None, this criterion is not considered.

    callback_func: Optional[Callable[[Dict], bool]], optional (default=None)
        A callback function that is called at the end of each iteration of the
        optimization process. The callback function should take a dictionary
        similar to the one returned in the history of the optimization process
        as input. If the return value of the callback function is True (or None),
        the optimization process will continue. If the return value is False,
        the optimization process will be stopped.
        If None, no callback function is used.

    update_func: Optional[Callable[[numpy.ndarray, numpy.ndarray], ArrayLike]], optional (default=None)
        A function that is called at the end of each iteration of the optimization
        process to compute the parameters for the next iteration.
        The function should take the current input parameters
        :math:`\mathbf{p}_{in}` and update
        :math:`\Delta\mathbf{p}_{in}` as inputs and return the next input
        parameters with shape ``(n_parameters,)``.
        If None, the input parameters for the next iteration will be computed as
        :math:`\mathbf{p}_{in}^{k+1} = \mathbf{p}_{in}^{k} + \Delta\mathbf{p}_{in}``.

    verbosity: Integral, optional (default=0)
        The level of verbosity for logging the optimization process.
        0: No logging
        1: Log only the final results of the optimization process.
        2: Log the results at each iteration of the optimization process.
        3: Details logging for debugging purposes.

    naninf: bool, optional (default=True)
        If True, the optimization process will be stopped
        if any NaN or Inf value is encountered in the cost function value
        or parameters during the optimization process.
        If False, the optimization process will continue.

    history: bool, optional (default=False)
        If True, the function will also return a tuple containing the history of
        the optimization process.

    history_details: Optional[Union[str, Sequence[str]]], optional (default=None)
        The details to include in the history of the optimization process.
        Can be a single string or a sequence of strings. See the Notes section for
        the available details.
        By default, the history will include:
        ``"iteration"``, ``"elapsed_time"``,
        ``"parameters"``,
        ``"delta_parameters"``, ``"delta_cost"``, ``"cost"``, and
        ``"optimality"``.
        For a specified list of details, only the details in the list below
        will be included in the history.


    Returns
    -------
    parameters: numpy.ndarray
        The optimized input parameters :math:`\mathbf{p}_{in}` with shape
        ``(n_parameters,)``.
        If no parametrization is provided, these parameters are also the
        parameters passed directly to the terms.
        If a parametrization is provided, the corresponding output parameters
        :math:`\mathbf{p}_{out}` can be obtained by applying the parametrization
        :math:`\mathbf{p}_{out} = P(\mathbf{p}_{in})`.

    history: List[Dict], optional
        Only returned if ``history`` is True. A list containing the history of the
        optimization process. Each element of the list is a dictionary
        containing the keys described below.


    Notes
    -----
    The history contains the following keys (if requested in ``history_details``):

    - "iteration": Integer representing the iteration number.
    - "elapsed_time": Float representing the time elapsed since the beginning of the optimization process in seconds.
    - "parameters": Numpy array representing the input parameters
      :math:`\mathbf{p}_{in}` at the current iteration.
    - "delta_parameters": Numpy array representing the change of input parameters
      :math:`\mathbf{p}_{in}` between the current and previous iteration
      (only for iterations > 0).
    - "costs": List of floats representing the cost function value for each term
      in the least squares problem at the current iteration, computed as
      :math:`\frac{1}{2} \sum_j
      \rho_i(\| \mathbf{R}_{i,j}(\mathbf{p}_{out}) \|^2)`
      for each term :math:`i`.
    - "cost": Float representing the cost function value at the current iteration,
      computed as
      :math:`\frac{1}{2} \sum_i w_i \sum_j
      \rho_i(\| \mathbf{R}_{i,j}(\mathbf{p}_{out}) \|^2)`.
    - "delta_cost": Float representing the change of the cost function value
      between the current and previous iteration (only for iterations > 0).
    - "optimality": Float representing the optimality value at the current iteration
      computed as ``norm(b, ord=numpy.inf)`` where b is the scaled second term.
    - "residuals": A list of numpy arrays representing the residuals for each term
      in the least squares problem at the current iteration (only for rJ terms).
    - "jacobians": A list of numpy arrays representing the Jacobians for each term
      in the least squares problem at the current iteration (only for rJ terms).
    - "second_term": The scaled second term :math:`\mathbf{g}` of the linear system
      at the current iteration.
    - "hessian": The Hessian approximation :math:`\mathbf{H}` of the linear system
      at the current iteration.

    The step of each loop iteration is as follows:

    .. code-block:: text

        While NOT converged:
            1. Compute the output parameters ``p_out = P(p_in)``.
            2. Build the system ``H Δp_in = -g``.
            3. Call the 'callback' function and STOP if it returns False.
            4. Check the stopping criteria and STOP if any of them is satisfied.
            5. If CONTINUE, solve the linear system ``H Δp_in = -g``.
            6. Update the input parameters ``p_in = p_in + Δp_in`` (or using 'update_func' if provided).


    Version
    -------

    - 0.1.0: Initial version for the solver method.
    - 0.3.0: Adding the regularization functionality to the solver.

    """

    # Check the validity of the input arguments and raise appropriate errors if necessary.
    if not isinstance(terms, Sequence):
        raise ValueError("terms must be a sequence of Term objects.")
    if len(terms) == 0:
        raise ValueError("terms sequence cannot be empty.")
    for term in terms:
        if not isinstance(term, Term):
            raise ValueError(
                "All elements of terms must be instances of the Term class."
            )

    p0 = numpy.asarray(p0, dtype=numpy.float64)
    if p0.ndim != 1:
        raise ValueError(f"p0 must be a 1D array, got {p0.ndim} dimensions.")

    if parametrization is not None and not isinstance(parametrization, Parametrization):
        raise ValueError(
            "parametrization must be an instance of the Parametrization class."
        )

    if max_iteration is not None:
        if not isinstance(max_iteration, Integral):
            raise ValueError("max_iteration must be an integer.")
        max_iteration = int(max_iteration)
        if max_iteration < 0:
            raise ValueError("max_iteration must be a non-negative integer.")

    if max_time is not None:
        if not isinstance(max_time, Real):
            raise ValueError("max_time must be a real number.")
        max_time = float(max_time)
        if max_time < 0:
            raise ValueError("max_time must be a non-negative real number.")

    if ftol is not None:
        if not isinstance(ftol, Real):
            raise ValueError("ftol must be a real number.")
        ftol = float(ftol)
        if ftol <= 0:
            raise ValueError("ftol must be a positive real number.")

    if xtol is not None:
        if not isinstance(xtol, Real):
            raise ValueError("xtol must be a real number.")
        xtol = float(xtol)
        if xtol <= 0:
            raise ValueError("xtol must be a positive real number.")

    if gtol is not None:
        if not isinstance(gtol, Real):
            raise ValueError("gtol must be a real number.")
        gtol = float(gtol)
        if gtol <= 0:
            raise ValueError("gtol must be a positive real number.")

    if atol is not None:
        if not isinstance(atol, Real):
            raise ValueError("atol must be a real number.")
        atol = float(atol)
        if atol <= 0:
            raise ValueError("atol must be a positive real number.")

    if ptol is not None:
        if not isinstance(ptol, Real):
            raise ValueError("ptol must be a real number.")
        ptol = float(ptol)
        if ptol <= 0:
            raise ValueError("ptol must be a positive real number.")

    if all(
        criterion is None
        for criterion in [max_iteration, max_time, ftol, xtol, gtol, atol, ptol]
    ):
        raise ValueError(
            "At least one stopping criterion must be provided "
            "(max_iteration, max_time, ftol, xtol, gtol, atol, or ptol)."
        )

    if callback_func is not None and not callable(callback_func):
        raise ValueError("callback_func must be a callable function.")

    if update_func is not None and not callable(update_func):
        raise ValueError("update_func must be a callable function.")

    if not isinstance(verbosity, Integral):
        raise ValueError("verbosity must be an integer.")
    verbosity = int(verbosity)
    if verbosity < 0 or verbosity > 3:
        raise ValueError("verbosity must be an integer between 0 and 3 inclusive.")

    if not isinstance(naninf, bool):
        raise ValueError("naninf must be a boolean value.")
    naninf = bool(naninf)

    if not isinstance(history, bool):
        raise ValueError("history must be a boolean value.")
    history = bool(history)

    if history_details is None:
        history_details = [
            "iteration",
            "elapsed_time",
            "parameters",
            "delta_parameters",
            "delta_cost",
            "cost",
            "optimality",
        ]
    if isinstance(history_details, str):
        history_details = [history_details]
    if not isinstance(history_details, Sequence):
        raise ValueError("history_details must be a string or a sequence of strings.")
    for detail in history_details:
        if detail not in _IMPLEMENTED_HISTORY_DETAILS and detail != "all":
            raise ValueError(
                f"Invalid history detail: {detail}. Valid details are: {_IMPLEMENTED_HISTORY_DETAILS} and 'all'."
            )
    if "all" in history_details:
        history_details = list(_IMPLEMENTED_HISTORY_DETAILS)

    # -- Select Computation --
    _compute_history = history or callback_func is not None
    _compute_cost = (
        ftol is not None
        or atol is not None
        or verbosity >= 2
        or (
            _compute_history
            and any(detail in ["cost", "costs"] for detail in history_details)
        )
    )
    _compute_optimality = (
        gtol is not None
        or verbosity >= 2
        or (
            _compute_history
            and any(detail in ["optimality"] for detail in history_details)
        )
    )
    _compute_delta_norm = xtol is not None or ptol is not None or verbosity >= 2

    _compute_convergence_analysis = verbosity >= 3

    # -- Solver Implementation --
    _n_terms = len(terms)
    _n_parameters = p0.shape[0]
    _parameters = p0.copy()
    _out_parameters = None
    _iteration = 0
    _history = []
    _starting_time = time.time()
    _end_flag = False  # ! (end-flag is used to BREAK the optimization loop)
    _end_message = ""
    _delta_parameters = None
    _delta_norm = None
    _last_total_cost = None

    r_vectors: List[numpy.ndarray] = None
    J_matrices: List[numpy.ndarray] = None
    H_matrices: List[numpy.ndarray] = None
    g_vectors: List[numpy.ndarray] = None
    costs: List[float] = None
    total_cost: float = None
    optimality: float = None
    rhos_arrays: List[Tuple[numpy.ndarray]] = None
    Hessian = None
    second_term = None

    # Printing the header for the optimization process logging based on the verbosity level.
    _printed_detail = f""
    _printed_header = f""
    if verbosity >= 2:
        _printed_detail += (
            f"\nIndividual costs [rJ term]: C_i = 0.5 * ρ(||r_i||^2) "
            f"\nCost: C = sum(w_i * C_i) "
            f"\nStep norm: ||Δp|| "
            f"\nOptimality: ||g|| "
        )
        _printed_header += (
            f"\n{'Iteration':^10} {'Total time (s)':^15} {'Cost C':^15} {'ΔC':^15}"
            + f" {'||Δp||_2':^15} {'||g||_∞':^15}"
        )
    if verbosity >= 3:
        _printed_header += (
            f" {'||Δp||_∞':^15} {'||g||_2':^15} {'Cond(H)':^15} {'Trace(H)':^15}"
            + " ".join(
                [f"{'Cost C_' + str(i):^15}" for i in range(_n_terms)],
            )
        )
    if verbosity >= 2:
        print(_printed_detail)
        print(_printed_header)

    # Start the optimization loop
    while True:  # ! (ensure end-flag activation for term "break" statement)
        # Apply the parametrization
        if parametrization is not None:
            _out_parameters = parametrization.p_func(_parameters)
        else:
            _out_parameters = _parameters

        # Compute r, J, H, g for each term and the cost of each term if necessary
        r_vectors = []
        J_matrices = []
        rhos_arrays = []
        H_matrices = []
        g_vectors = []
        for term in terms:
            if term.type == "rJ":
                r_vectors.append(term.r_func(_out_parameters))
                J_matrices.append(term.J_func(_out_parameters))
                g_vectors.append(None)
                H_matrices.append(None)
                # compute rho rho' and rho'' for each residual of the term
                if term.loss == "linear":
                    rhos_arrays.append((r_vectors[-1] ** 2, 1.0, 0.0))
                else:
                    rhos_arrays.append(term.rho_function(r_vectors[-1]))

            elif term.type == "gH":
                r_vectors.append(None)
                J_matrices.append(None)
                rhos_arrays.append(None)
                g_vectors.append(term.g_func(_out_parameters))
                H_matrices.append(term.H_func(_out_parameters))

        # Build the modified residuals and Jacobian for each rJ term based on the robust cost function
        costs = []
        if _compute_cost:
            for i in range(_n_terms):
                if terms[i].type == "rJ":
                    costs.append(0.5 * numpy.sum(rhos_arrays[i][0]))
                elif terms[i].type == "gH":
                    costs.append(0)
                else:
                    raise ValueError(f"Unknown term type: {terms[i].type}")
            total_cost = sum(
                w * c for w, c in zip([term.weight for term in terms], costs)
            )

        for i in range(_n_terms):
            if terms[i].type == "rJ" and terms[i].loss != "linear":
                r_vectors[i], J_matrices[i] = _build_tilde_R_and_tilde_J(
                    r_vectors[i],
                    J_matrices[i],
                    rhos_arrays[i][1],
                    rhos_arrays[i][2],
                )

        # Build the chain rule for the parametrization if it exists
        if parametrization is not None:
            Jp = parametrization.J_func(_parameters)

            for i in range(_n_terms):
                if terms[i].type == "rJ":
                    J_matrices[i] = J_matrices[i] @ Jp
                elif terms[i].type == "gH":
                    H_matrices[i] = Jp.T @ H_matrices[i] @ Jp
                    g_vectors[i] = Jp.T @ g_vectors[i]

        # Assemble the Hessian and second term for the linear system
        Hessian = sum(
            w * (J.T @ J) if terms[i].type == "rJ" else w * H
            for i, (w, J, H) in enumerate(
                zip([term.weight for term in terms], J_matrices, H_matrices)
            )
        )
        second_term = sum(
            w * (J.T @ r) if terms[i].type == "rJ" else w * g
            for i, (w, J, r, g) in enumerate(
                zip(
                    [term.weight for term in terms],
                    J_matrices,
                    r_vectors,
                    g_vectors,
                )
            )
        )

        if _compute_optimality:
            optimality = numpy.linalg.norm(second_term, ord=numpy.inf)

        if _compute_convergence_analysis:
            optimality_2 = numpy.linalg.norm(second_term, ord=2)
            if scipy.sparse.issparse(Hessian):
                cond_Hessian = scipy.sparse.linalg.norm(
                    Hessian
                ) * scipy.sparse.linalg.norm(scipy.sparse.linalg.inv(Hessian))
                trace_Hessian = Hessian.diagonal().sum()
            else:
                cond_Hessian = numpy.linalg.cond(Hessian)
                trace_Hessian = numpy.trace(Hessian)

        _elapsed_time = time.time() - _starting_time
        # Update history
        if _compute_history:
            _h = {}
            if "iteration" in history_details:
                _h["iteration"] = _iteration
            if "parameters" in history_details:
                _h["parameters"] = _parameters.copy()
            if "delta_parameters" in history_details:
                if _delta_parameters is None:
                    _h["delta_parameters"] = None
                else:
                    _h["delta_parameters"] = _delta_parameters.copy()
            if "delta_cost" in history_details:
                if _last_total_cost is None:
                    _h["delta_cost"] = None
                else:
                    _h["delta_cost"] = total_cost - _last_total_cost
            if "elapsed_time" in history_details:
                _h["elapsed_time"] = _elapsed_time
            if "costs" in history_details:
                _h["costs"] = costs
            if "cost" in history_details:
                _h["cost"] = total_cost
            if "optimality" in history_details:
                _h["optimality"] = optimality
            if "residuals" in history_details:
                _h["residuals"] = [
                    r.copy() if r is not None else None for r in r_vectors
                ]
            if "jacobians" in history_details:
                _h["jacobians"] = [
                    J.copy() if J is not None else None for J in J_matrices
                ]
            if "second_term" in history_details:
                _h["second_term"] = second_term.copy()
            if "hessian" in history_details:
                _h["hessian"] = Hessian.copy()
            _history.append(_h)

        _printed_row = f""
        if verbosity >= 2:
            if _delta_parameters is None:
                strdp2 = f"{'':^15}"
                strdpinf = f"{'':^15}"
            else:
                strdp2 = f"{_delta_norm:^15.3e}"
                strdpinf = (
                    f"{numpy.linalg.norm(_delta_parameters, ord=numpy.inf):^15.3e}"
                )
            if _last_total_cost is None:
                dC_str = f"{'':^15}"
            else:
                dC = total_cost - _last_total_cost
                dC_str = f"{dC:^15.3e}"
            _printed_row += (
                f"{_iteration:^10} {_elapsed_time:^15.3e} {total_cost:^15.3e} {dC_str}"
                + f" {strdp2} {optimality:^15.3e}"
            )
        if verbosity >= 3:
            _printed_row += (
                f" {strdpinf} {optimality_2:^15.3e}"
                + f" {cond_Hessian:^15.3e} {trace_Hessian:^15.3e}"
                + " ".join([f"{c:^15.3e}" for c in costs])
            )
        if verbosity >= 2:
            print(_printed_row)

        # Convergence analysis and stopping criteria check
        if ftol is not None and _last_total_cost is not None:
            dF = abs(total_cost - _last_total_cost)
            if dF < ftol * total_cost:
                _end_flag = True
                _end_message += f"\n[ftol] Convergence achieved (df < ftol * F) : {dF} < {ftol * total_cost}."

        if atol is not None and total_cost < atol:
            _end_flag = True
            _end_message += (
                f"\n[atol] Convergence achieved (F < atol) : {total_cost} < {atol}."
            )

        if gtol is not None and optimality < gtol:
            _end_flag = True
            _end_message += f"\n[gtol] Convergence achieved (optimality < gtol) : {optimality} < {gtol}."

        if xtol is not None and _delta_norm is not None:
            _parameters_norm = numpy.linalg.norm(_parameters, ord=2)
            if _delta_norm < xtol * (xtol + _parameters_norm):
                _end_flag = True
                _end_message += f"\n[xtol] Convergence achieved (||Δp|| < xtol * (xtol + ||p||)) : {_delta_norm} < {xtol * (xtol + _parameters_norm)}."

        if ptol is not None and _delta_norm is not None and _delta_norm < ptol:
            _end_flag = True
            _end_message += f"\n[ptol] Convergence achieved (||Δp|| < ptol) : {_delta_norm} < {ptol}."

        if max_iteration is not None and _iteration >= max_iteration:
            _end_flag = True
            _end_message += f"\n[max_iterations] Maximum number of iterations reached: {max_iteration}."

        if max_time is not None and _elapsed_time >= max_time:
            _end_flag = True
            _end_message += (
                f"\n[max_time] Maximum computation time reached: {max_time} seconds."
            )

        if naninf:
            if total_cost is not None and (
                numpy.isnan(total_cost) or numpy.isinf(total_cost)
            ):
                _end_flag = True
                _end_message += f"\n[naninf] Optimization stopped due to NaN or Inf value in cost: {total_cost}."
            elif numpy.any(numpy.isnan(_parameters)) or numpy.any(
                numpy.isinf(_parameters)
            ):
                _end_flag = True
                _end_message += f"\n[naninf] Optimization stopped due to NaN or Inf value in parameters."

        if callback_func is not None:
            callback_result = callback_func(_history[-1])
            if callback_result is not None and not isinstance(callback_result, bool):
                raise ValueError("Callback function must return a boolean or None.")
            if callback_result is False:
                _end_flag = True
                _end_message += (
                    "\n[callback] Optimization stopped by callback function."
                )

        if _end_flag:
            if verbosity >= 1:
                print(_end_message)
            break

        if scipy.sparse.issparse(Hessian):
            _delta_parameters = scipy.sparse.linalg.spsolve(Hessian, -second_term)
        else:
            _delta_parameters = numpy.linalg.solve(Hessian, -second_term)

        # Update parameters
        if update_func is not None:
            _new_parameters = update_func(_parameters, _delta_parameters)
            _new_parameters = numpy.asarray(_new_parameters, dtype=numpy.float64)
            if not _new_parameters.ndim == 1 or _new_parameters.size != _n_parameters:
                raise ValueError(
                    f"update_func must return a 1D array with shape ({_n_parameters},)."
                )
            _delta_parameters = _new_parameters - _parameters
            _parameters = _new_parameters
        else:
            _parameters = _parameters + _delta_parameters

        if _compute_delta_norm:
            _delta_norm = numpy.linalg.norm(_delta_parameters, ord=2)

        if _compute_cost:
            _last_total_cost = total_cost

        _iteration += 1

    if history:
        return _parameters, _history
    else:
        return _parameters.copy()
