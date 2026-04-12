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

from .rho_functions import (
    linear_rho_at_R2,
    soft_l1_rho_at_R2,
    cauchy_rho_at_R2,
    arctan_rho_at_R2,
    _build_tilde_R_and_tilde_J,
)

from .study_optimization import study_jacobian


def solve_gauss_newton(
    residual_func: Union[Callable, Sequence[Callable]],
    jacobian_func: Union[Callable, Sequence[Callable]],
    parameters: ArrayLike,
    *,
    weight: Optional[Union[str, Sequence[str]]] = None,
    loss: Optional[Union[str, Sequence[str]]] = None,
    max_iterations: Optional[Integral] = None,
    max_time: Optional[Real] = None,
    ftol: Optional[Real] = None,
    xtol: Optional[Real] = None,
    gtol: Optional[Real] = None,
    atol: Optional[Real] = None,
    ptol: Optional[Real] = None,
    callback: Optional[Callable[[Dict], bool]] = None,
    update_func: Optional[Callable[[numpy.ndarray, numpy.ndarray], ArrayLike]] = None,
    verbosity: Integral = 0,
    history: bool = False,
) -> Tuple[numpy.ndarray, Tuple[Dict]]:
    r"""
    Solve the least squares problem using the Gauss-Newton method with robust cost functions.

    The function accepts multiple terms in the least squares problem,
    each with its own residual function, Jacobian function, weight, and loss function
    solving the following optimization problem

    .. math::

        \min_{\mathbf{p}} \frac{1}{2} \sum_{i} w_i \sum_j \rho_i\left(\| \mathbf{R}_{i,j}(\mathbf{p}) \|^2\right)


    where :math:`w_i` is a weight for each sub least squares problem,
    and :math:`\rho_i` is a robust cost function for each sub least squares problem.

    For more details on the optimization problem and the algorithm,
    please refer to the documentation.

    .. note::

        By nomenclature, we assume that the first sub least squares problem (i.e. :math:`i=0`)
        is the main least squares problem containing the data residuals, and the other sub
        least squares problems (i.e. :math:`i \geq 1`) are regularization terms.

    .. important::

        At least one of the stopping criteria (i.e. ``max_iterations``, ``max_time``,
        ``ftol``, ``xtol``, ``gtol``, ``atol``, ``ptol``)
        must be provided to ensure the optimization process will stop.

    Parameters
    ----------
    residual_func: Union[Callable, Sequence[Callable]]
        The function(s) to compute the residuals for each term in the least squares problem.
        Can be a single function or a sequence of functions.
        Each callable represent the `i`-th term :math:`\mathbf{R}_i(\mathbf{p})`
        in the least squares problem and should take as inputs the parameters
        (array-like with shape (n_parameters,)) and return the residuals as a
        1D numpy array with shape (n_residuals_i,) representing each
        :math:`\mathbf{R}_{i,j}(\mathbf{p})` where :math:`j=0,...,n_{\text{residuals}_i-1}`.

    jacobian_func: Union[Callable, Sequence[Callable]]
        The function(s) to compute the Jacobian matrices for each term in the
        least squares problem. Can be a single function or a sequence of functions.
        Each callable represent the `i`-th term :math:`\mathbf{J}_i(\mathbf{p})`
        in the least squares problem such as
        :math:`\mathbf{J}_i(\mathbf{p}) = \frac{\partial \mathbf{R}_i(\mathbf{p})}{\partial \mathbf{p}}`
        and should take as inputs the parameters (array-like with shape (n_parameters,))
        and return the Jacobian matrix as a 2D numpy array with shape
        (n_residuals_i, n_parameters) representing the Jacobian of each residual
        :math:`\mathbf{R}_{i,j}(\mathbf{p})` where :math:`j=0,...,n_{\text{residuals}_i-1}`.

    parameters: ArrayLike
        The starting parameters of the optimization with shape (n_parameters,).
        The array will not be modified by this function, a copy of the parameters will
        be used for the optimization process.

    weight: Optional[Union[Real, Sequence[Real]]], optional (default=None)
        The weight(s) for each term in the least squares problem.
        Can be a single value or a sequence of values.
        Each weight :math:`w_i` is a weight for the corresponding term in
        the least squares problem and should be a non-negative real number.
        If a single value is provided, it will be used for all terms in the
        least squares problem. If None is provided, it will be set to 1.0.

    loss: Optional[Union[str, Sequence[str]]], optional (default=None)
        The loss function(s) for each term in the least squares problem.
        Can be a single string or a sequence of strings.
        Each loss function :math:`\rho_i` is a robust cost function for the corresponding
        term in the least squares problem and should be one of the following strings:
        ["linear", "soft_l1", "cauchy", "arctan"].
        If a single string is provided, it will be used for all terms in the least
        squares problem. If None is provided, it will be set to "linear".

    max_iterations: Optional[Integral], optional (default=None)
        Stop criterion by the number of iterations.
        The optimization process is stopped when the number of iterations
        exceeds ``max_iterations``. If None, no limit on the
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
        Stop criterion by the change of the parameters.
        The optimization process is stopped when ``||dp|| < xtol * (xtol + ||p||)``
        where p is the parameters and dp is the change of the parameters
        between two iterations. If None, this criterion is not considered.

    gtol: Optional[Real], optional (default=None)
        Stop criterion by the optimality value.
        The optimization process is stopped when the optimality verifies
        ``norm(b, ord=numpy.inf) < gtol`` where b is the scaled second term.
        If None, this criterion is not considered.

    atol: Optional[Real], optional (default=None)
        Stop criterion by the absolute cost function value.
        The optimization process is stopped when ``F < atol`` where F is the cost
        function value.
        If None, this criterion is not considered.

    ptol: Optional[Real], optional (default=None)
        Stop criterion by the parameters value.
        The optimization process is stopped when ``||p|| < ptol`` where p is the
        parameters.
        If None, this criterion is not considered.

    callback: Optional[Callable[[Dict], bool]], optional (default=None)
        A callback function that is called at the end of each iteration of the
        optimization process. The callback function should take a dictionary
        similar to the one returned in the history of the optimization process
        as input. If the return value of the callback function is True (or None),
        the optimization process will continue. If the return value is False,
        the optimization process will be stopped.
        If None, no callback function is used.

    update_func: Optional[Callable[[numpy.ndarray, numpy.ndarray], ArrayLike]], optional (default=None)
        A function that is called at the end of each iteration of the optimization
        process to compute the parameters for the next iteration. The function
        should take the current parameters and update ``(p, dp)`` as inputs and return
        the parameters with shape (n_parameters,) to use for the next iteration.
        If None, the parameters for the next iteration will be computed as ``p + dp``
        where p is the current parameters and dp is the update computed by solving the linear system.

    verbosity : Integral, optional
        The level of verbosity for logging the optimization process.
        0: No logging (default)
        1: Log only the final results of the optimization process.
        2: Log the results at each iteration of the optimization process.
        3: Log detailed information about the optimization process.

    history: bool
        If True, the function will also return a tuple containing the history of
        the optimization process.


    Returns
    -------
    parameters: numpy.ndarray
        The parameters that minimize the least squares problem with shape (n_parameters,).

    history: Tuple[Dict], optional
        Only returned if ``history`` is True. A tuple containing the history of the
        optimization process. Each element of the tuple is a dictionary
        containing the keys described below.


    Notes
    -----
    The history contains the following keys:

    - "iteration": Integer representing the iteration number.
    - "costs": List of floats representing the cost function value for each term in the least squares problem at the current iteration compute as :math:`\frac{1}{2} \sum_j \rho_i\left(\| \mathbf{R}_{i,j}(\mathbf{p}) \|^2\right)` for each term :math:`i`.
    - "cost": Float representing the cost function value at the current iteration computed as :math:`\frac{1}{2} \sum_i w_i \sum_j \rho_i\left(\| \mathbf{R}_{i,j}(\mathbf{p}) \|^2\right)`.
    - "parameters": Numpy array representing the parameters at the current iteration.
    - "residuals": The residual of the data term (i.e. the first term in the least squares problem) at the current iteration.

    The optimization process is performed by iteratively solving a linearized
    version of the least squares problem at each iteration for different values
    of the regularization factor, and then selecting the optimal regularization
    factor based on the L-curve analysis.
    The step of each loop iteration is as follows:

    .. code-block:: text

        While NOT converged:
            1. Build the system M Δp = -b from 'residual_func', 'jacobian_func', 'loss' and 'weight'.
            2. Call the 'callback' function and STOP if it returns False.
            3. Check the stopping criteria and STOP if any of them is satisfied.
            4. If CONTINUE solve the linear system M Δp = -b for Δp.
            5. Update the parameters as p = p + Δp (or using 'update_func' if provided).


    Version
    -------

    - 0.0.1: Initial version.
    - 0.0.3: Added elapsed time to the optimization details.

    """
    if not isinstance(residual_func, Sequence):
        residual_func = [residual_func]
    if not isinstance(jacobian_func, Sequence):
        jacobian_func = [jacobian_func]

    if not len(residual_func) == len(jacobian_func):
        raise ValueError(
            "The length of residual_func and jacobian_func must be the same."
        )
    if not all(callable(func) for func in residual_func + jacobian_func):
        raise ValueError(
            "All elements in residual_func and jacobian_func must be callable."
        )

    if weight is None:
        weight = 1.0
    if not isinstance(weight, Sequence):
        weight = [weight]
    if len(weight) == 1:
        weight = weight * len(residual_func)
    if not len(weight) == len(residual_func):
        raise ValueError(
            "The length of weight must be the same as the length of residual_func and jacobian_func."
        )
    if not all(isinstance(w, Real) for w in weight):
        raise ValueError("All elements in weight must be real numbers.")
    weight = [float(w) for w in weight]  # Ensure weights are floats

    if loss is None:
        loss = "linear"
    if isinstance(loss, str):
        loss = [loss]
    loss = [l if l is not None else "linear" for l in loss]  # None -> "linear"
    if len(loss) == 1:
        loss = loss * len(residual_func)
    if not len(loss) == len(residual_func):
        raise ValueError(
            "The length of loss must be the same as the length of residual_func and jacobian_func."
        )
    if not all(isinstance(l, str) for l in loss):
        raise ValueError("All elements in loss must be strings.")
    valid_loss_functions = {"linear", "soft_l1", "cauchy", "arctan"}
    if not all(l in valid_loss_functions for l in loss):
        raise ValueError(f"All elements in loss must be one of {valid_loss_functions}.")

    parameters = numpy.asarray(parameters, dtype=numpy.float64)
    if not isinstance(parameters, numpy.ndarray) or parameters.ndim != 1:
        raise ValueError("Parameters must be a 1D numpy array.")

    if not isinstance(verbosity, Integral):
        raise TypeError("verbosity must be an integer.")
    if not (0 <= verbosity <= 3):
        raise ValueError("verbosity must be an integer between 0 and 3.")
    verbosity = int(verbosity)

    if max_iterations is not None:
        if not isinstance(max_iterations, Integral):
            raise TypeError("max_iterations must be an integer.")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer.")
        max_iterations = int(max_iterations)

    if max_time is not None:
        if not isinstance(max_time, Real):
            raise TypeError("max_time must be a real number.")
        if max_time <= 0:
            raise ValueError("max_time must be a positive real number.")
        max_time = float(max_time)

    if ftol is not None:
        if not isinstance(ftol, Real):
            raise TypeError("ftol must be a real number.")
        if ftol <= 0:
            raise ValueError("ftol must be a positive real number.")
        ftol = float(ftol)

    if xtol is not None:
        if not isinstance(xtol, Real):
            raise TypeError("xtol must be a real number.")
        if xtol <= 0:
            raise ValueError("xtol must be a positive real number.")
        xtol = float(xtol)

    if gtol is not None:
        if not isinstance(gtol, Real):
            raise TypeError("gtol must be a real number.")
        if gtol <= 0:
            raise ValueError("gtol must be a positive real number.")
        gtol = float(gtol)

    if atol is not None:
        if not isinstance(atol, Real):
            raise TypeError("atol must be a real number.")
        if atol <= 0:
            raise ValueError("atol must be a positive real number.")
        atol = float(atol)

    if ptol is not None:
        if not isinstance(ptol, Real):
            raise TypeError("ptol must be a real number.")
        if ptol <= 0:
            raise ValueError("ptol must be a positive real number.")
        ptol = float(ptol)

    if all(
        criterion is None
        for criterion in [max_iterations, max_time, ftol, xtol, gtol, atol, ptol]
    ):
        raise ValueError(
            "At least one stopping criterion must be provided (max_iterations, max_time, ftol, xtol, gtol, atol, or ptol)."
        )

    if not isinstance(history, bool):
        raise TypeError("history must be a boolean value.")
    history = bool(history)

    if callback is not None and not isinstance(callback, Callable):
        raise TypeError("callback must be a callable function or None.")

    if update_func is not None and not isinstance(update_func, Callable):
        raise TypeError("update_func must be a callable function or None.")

    # ----------- Process optimisation ------------------
    rho_func = []
    for l in loss:
        if l == "linear":
            rho_func.append(linear_rho_at_R2)
        elif l == "soft_l1":
            rho_func.append(soft_l1_rho_at_R2)
        elif l == "cauchy":
            rho_func.append(cauchy_rho_at_R2)
        elif l == "arctan":
            rho_func.append(arctan_rho_at_R2)
        else:
            raise ValueError(
                f"Invalid loss function '{l}'. Valid options are 'linear', 'soft_l1', 'cauchy', 'arctan'."
            )

    _n_terms = len(residual_func)
    _n_parameters = parameters.size
    _parameters = parameters.copy()
    _iteration = 0
    _history = []
    _end_flag = False  # ! (end-flag is used to BREAK the optimization loop)
    _end_message = ""
    _delta = None
    _last_total_cost = None

    _compute_history = history or callback is not None
    _compute_cost = (
        ftol is not None or atol is not None or verbosity >= 2 or _compute_history
    )
    _compute_optimality = gtol is not None or verbosity >= 2 or _compute_history

    # Cached values
    residual_arrays: List[numpy.ndarray] = None
    jacobian_matrices: List[numpy.ndarray] = None
    costs: List[float] = None
    total_cost: float = None
    optimality: float = None
    rhos_arrays: List[Tuple[numpy.ndarray]] = None
    M = None
    b = None

    if verbosity >= 3:
        study_jacobian(
            residual_func=residual_func,
            jacobian_func=jacobian_func,
            parameters=_parameters,
            weight=weight,
            loss=loss,
            title="Initial Jacobian Analysis Before Optimization",
        )

    if verbosity >= 2:
        detail = (
            f"\nIndividual costs: C_i = 0.5 * ρ(||R_i||^2) "
            f"\nCost: C = sum(w_i * C_i) "
            f"\nStep norm: ||Δp||_2 and ||Δp||_∞ "
            f"\nOptimality: ||J^T R||_∞"
        )
        header = (
            f"\n{'Iteration':^10} {'Cost C':^15}"
            + " ".join(
                [f"{'Cost C_' + str(i):^15}" for i in range(len(residual_func))],
            )
            + f" {'||Δp||_2':^15} {'||Δp||_∞':^15} {'Optimality':^15}"
            + f" {'Total Time (s)':^15}"
        )
        print(detail)
        print(header)

    _starting_time = time.time()
    while True:  # ! (ensure end-flag activation for term "break" statement)
        # Compute residuals, Jacobians, costs
        residual_arrays = [R_func(_parameters) for R_func in residual_func]
        jacobian_matrices = [J_func(_parameters) for J_func in jacobian_func]
        rhos_arrays = [r_func(R) for r_func, R in zip(rho_func, residual_arrays)]

        if _compute_cost:
            costs = [0.5 * numpy.sum(rho[0]) for rho in rhos_arrays]
            total_cost = sum(w * c for w, c in zip(weight, costs))

        # Build tilted residuals and Jacobian based on the robust cost function
        for i in range(_n_terms):
            residual_arrays[i], jacobian_matrices[i] = _build_tilde_R_and_tilde_J(
                residual_arrays[i],
                jacobian_matrices[i],
                rhos_arrays[i][1],
                rhos_arrays[i][2],
            )

        # Assemble the operator M and the second term b for the linear system M Δp = -b
        M = sum(w * J.T @ J for w, J in zip(weight, jacobian_matrices))
        b = sum(
            w * J.T @ R for w, J, R in zip(weight, jacobian_matrices, residual_arrays)
        )
        if _compute_optimality:
            optimality = numpy.linalg.norm(b, ord=numpy.inf)

        # Update history
        if _compute_history:
            _history.append(
                {
                    "iteration": _iteration,
                    "costs": costs,
                    "cost": total_cost,
                    "parameters": _parameters.copy(),
                    "residuals": residual_arrays[0].copy(),  # Data term residuals
                }
            )

        # Check for convergence before solving the linear system
        if verbosity >= 2:
            if _delta is None:
                strdp2 = f"{'':^15}"
                strdpinf = f"{'':^15}"
            else:
                strdp2 = f"{numpy.linalg.norm(_delta, ord=2):^15.3e}"
                strdpinf = f"{numpy.linalg.norm(_delta, ord=numpy.inf):^15.3e}"
            row = (
                f"{_iteration:^10} {total_cost:^15.3e}"
                + " ".join([f"{c:^15.3e}" for c in costs])
                + f" {strdp2} {strdpinf} {optimality:^15.3e}"
                + f" {time.time() - _starting_time:^15.3e}"
            )
            print(row)

        if _compute_cost and ftol is not None and _last_total_cost is not None:
            dF = abs(total_cost - _last_total_cost)
            if dF < ftol * total_cost:
                _end_flag = True
                _end_message += f"\n[ftol] Convergence achieved (df < ftol * F) : {dF} < {ftol * total_cost}."

        if _compute_cost and atol is not None and total_cost < atol:
            _end_flag = True
            _end_message += (
                f"\n[atol] Convergence achieved (F < atol) : {total_cost} < {atol}."
            )

        if _compute_optimality and gtol is not None and optimality < gtol:
            _end_flag = True
            _end_message += f"\n[gtol] Convergence achieved (optimality < gtol) : {optimality} < {gtol}."

        if ptol is not None and numpy.linalg.norm(_parameters, ord=2) < ptol:
            _end_flag = True
            _end_message += f"\n[ptol] Convergence achieved (||p|| < ptol) : {numpy.linalg.norm(_parameters, ord=2)} < {ptol}."

        if max_iterations is not None and _iteration >= max_iterations:
            _end_flag = True
            _end_message += f"\n[max_iterations] Maximum number of iterations reached: {max_iterations}."

        if max_time is not None and (time.time() - _starting_time) >= max_time:
            _end_flag = True
            _end_message += (
                f"\n[max_time] Maximum computation time reached: {max_time} seconds."
            )

        if callback is not None:
            callback_result = callback(_history[-1])
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

        # Solve the linear system M Δp = -b
        if not numpy.linalg.cond(M) < 1 / numpy.finfo(M.dtype).eps:
            raise numpy.linalg.LinAlgError(
                "The Jacobian matrix is singular or ill-conditioned."
                "Consider using a different robust cost function or adding regularization."
            )

        if scipy.sparse.issparse(M):
            _delta = scipy.sparse.linalg.spsolve(M, -b)
        else:
            _delta = numpy.linalg.solve(M, -b)

        # Update parameters
        if update_func is not None:
            _parameters = update_func(_parameters, _delta)
            _parameters = numpy.asarray(_parameters, dtype=numpy.float64)
            if not _parameters.ndim == 1 or _parameters.size != _n_parameters:
                raise ValueError(
                    f"update_func must return a 1D array with shape ({_n_parameters},)."
                )
        else:
            _parameters += _delta

        _last_total_cost = total_cost
        _iteration += 1

    if verbosity >= 3:
        study_jacobian(
            residual_func=residual_func,
            jacobian_func=jacobian_func,
            parameters=_parameters,
            weight=weight,
            loss=loss,
            title="Final Jacobian Analysis After Optimization",
        )

    if history:
        return _parameters, tuple(_history)
    else:
        return _parameters.copy()
