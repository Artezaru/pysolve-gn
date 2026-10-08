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

from typing import Optional, Sequence, Union, Tuple, Callable, Dict, List, Any, Mapping
from numbers import Real, Integral
from dataclasses import dataclass
from numpy.typing import ArrayLike

import collections
import time

import numpy

from .implemented_conf import (
    _IMPLEMENTED_BATCH_HISTORY_DETAILS,
    _IMPLEMENTED_DAMPINGS,
    _DEFAULT_BATCH_SOLVE_HISTORY,
    _STOP_CODES,
    _STOP_CONVERGENCE,
    _STOP_NANINF,
    _STOP_FAILURE,
)
from .solver import _stop_reasons, _validate_lm_conf

from .batch_evaluation import (
    BatchCostState,
    BatchSystemState,
    _compute_batch_term_parameters,
    _compute_batch_jacobian_P,
    _evaluate_batch_cost,
    _evaluate_batch_system,
)

from .batch_term import BatchTerm
from .batch_parametrization import BatchParametrization


@dataclass
class BatchSolveResult:
    r"""
    Result of the optimization performed by :func:`pysolvegn.solve_batch`.

    All the per-problem arrays have the number ``k`` of problems of the batch as first
    dimension. For each problem, the values describe the last iteration during which
    the problem was processed (frozen values).

    Attributes
    ----------
    parameters: numpy.ndarray
        The optimized input parameters :math:`\mathbf{p}_{in,k}` of each problem with
        shape ``(k, n_parameters)``.
        If no parametrization is provided, these parameters are also the
        parameters passed directly to the terms.
        If a parametrization is provided, the corresponding parameters passed to the
        terms are available in ``term_parameters``.

    history: List[Dict]
        The history of the optimization process. Each element of the list is a
        dictionary describing one iteration (see the Notes section of
        :func:`pysolvegn.solve_batch` for the available keys).
        Empty if ``history`` is False.

    success: numpy.ndarray
        Array of booleans with shape ``(k,)``. ``success[k]`` is True if the
        optimization of the problem ``k`` stopped because a convergence criterion
        (``ftol``, ``atol``, ``gtol``, ``xtol`` or ``ptol``) was satisfied.
        The value of each problem is determined with the following priority:

        1. Always False if the problem was stopped by a NaN or Inf value (in its
           initial parameters, its term parameters, its update or, with
           ``damping="lm"``, its cost), by a singular linear system, by the callback
           function or by a failure of the Levenberg-Marquardt step, even if a
           convergence criterion was satisfied at the same iteration.
        2. Otherwise True if a convergence criterion was satisfied, even if
           ``max_iteration`` or ``max_time`` was reached at the same iteration.
        3. Otherwise False (stopped by ``max_iteration`` or ``max_time``).

    stop_code: numpy.ndarray
        Array of integers with shape ``(k,)``: the reasons why each problem stopped,
        encoded as a bit mask (same bits as ``SolveResult.stop_code``, listed in the
        Notes section of :class:`pysolvegn.SolveResult`;
        several criteria can be triggered at the same iteration). Use :meth:`stopped_by`
        to test a criterion for all the problems, :meth:`reasons` to get the text of a
        problem and the property ``message`` for a summary.

    n_iterations: numpy.ndarray
        Array of integers with shape ``(k,)``: the number of iterations performed for
        each problem, i.e. the number of updates :math:`\Delta\mathbf{p}_{in,k}^{*}` applied
        to its parameters (0 if the problem stopped at its initial parameters).

    cost: Optional[numpy.ndarray]
        The cost function value :math:`C_k` of each problem at the returned parameters,
        with shape ``(k,)``. The cost of the successful problems is always available
        (computed at the end if no criterion, damping, callback, history or verbosity
        required it during the optimization). NaN for the problems whose cost is not
        available. None if no cost is available at all.

    optimality: numpy.ndarray
        The optimality ``norm(g_k, ord=numpy.inf)`` of each problem at the returned
        parameters, with shape ``(k,)``, where :math:`\mathbf{g}_k` is the scaled second
        term. NaN for the problems whose terms were not evaluated at the returned parameters.

    term_parameters: Optional[numpy.ndarray]
        The parameters passed to the terms, :math:`\mathbf{p}_{out,k} = P(\mathbf{p}_{in,k})`,
        of each problem at the returned parameters, with shape ``(k, n_p_outputs)`` (equal
        to ``parameters`` if no parametrization is provided). NaN for the problems that did
        not start (NaN or Inf value in ``p0``). None if no problem started.

    elapsed_time: float
        The total time of the optimization in seconds.

    n_rejected: numpy.ndarray
        Array of integers with shape ``(k,)``: the total number of rejected
        Levenberg-Marquardt trial steps of each problem (0 if ``damping`` is None).

    config: Dict[str, Any]
        The configuration requested to the solver: the stopping criteria
        (``"max_iteration"``, ``"max_time"``, ``"ftol"``, ``"xtol"``, ``"gtol"``,
        ``"atol"``, ``"ptol"``, None if not used), ``"damping"`` and ``"lm_conf"``
        (the complete Levenberg-Marquardt configuration, default values included).
    """

    parameters: numpy.ndarray
    history: List[Dict]
    success: numpy.ndarray
    stop_code: numpy.ndarray
    n_iterations: numpy.ndarray
    cost: Optional[numpy.ndarray]
    optimality: numpy.ndarray
    term_parameters: Optional[numpy.ndarray]
    elapsed_time: float
    n_rejected: numpy.ndarray
    config: Dict[str, Any]

    def stopped_by(self, name: str) -> numpy.ndarray:
        r"""
        Test, for each problem, whether a stopping criterion was triggered at its last
        iteration.

        Parameters
        ----------
        name : str
            The name of the criterion (see :attr:`pysolvegn.SolveResult.stop_code`), or
            ``"naninf"`` for any NaN or Inf value.

        Returns
        -------
        numpy.ndarray
            Array of booleans with shape ``(k,)``.
        """
        if name == "naninf":
            return (self.stop_code & _STOP_NANINF) != 0
        if name not in _STOP_CODES:
            raise ValueError(f"Unknown stopping criterion '{name}'. Valid names are {tuple(_STOP_CODES)} and 'naninf'.")
        return (self.stop_code & _STOP_CODES[name]) != 0

    def reasons(self, index: Integral) -> List[str]:
        r"""
        The reasons why the problem ``index`` stopped, one string per triggered
        criterion, each starting with its tag (e.g. ``"[xtol] ..."``). The texts give
        the thresholds requested to the solver (not the values reached).

        Parameters
        ----------
        index : Integral
            The index of the problem in the batch.

        Returns
        -------
        List[str]
            The reasons, in the order of the checks of the solver.
        """
        return _stop_reasons(int(self.stop_code[index]), self.config)

    @property
    def message(self) -> str:
        r"""
        [Get] A summary of the optimization: the number of successful problems and the number
        of problems stopped by each criterion (one per line).

        Returns
        -------
        str
            The summary.
        """
        lines = [f"{int(numpy.sum(self.success))}/{self.success.shape[0]} problems converged."]
        for name, bit in _STOP_CODES.items():
            count = int(numpy.count_nonzero(self.stop_code & bit))
            if count > 0:
                lines.append(f"[{name}] {count} problem(s).")
        return "\n".join(lines)


def _solve_batch_linear_systems(
    hessian: numpy.ndarray,
    second_term: numpy.ndarray,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    r"""
    Solve the ``m`` independent linear systems :math:`\mathbf{H}_k \Delta\mathbf{p}_k = -\mathbf{g}_k`.

    The systems are solved together. If at least one of them is singular
    (``numpy.linalg.LinAlgError``), they are solved one by one to identify the singular
    ones.

    Parameters
    ----------
    hessian : numpy.ndarray
        The Hessians with shape ``(m, n, n)``.

    second_term : numpy.ndarray
        The second terms with shape ``(m, n)``.

    Returns
    -------
    delta : numpy.ndarray
        The solutions with shape ``(m, n)`` (NaN rows for the singular systems).

    is_singular : numpy.ndarray
        Array of booleans with shape ``(m,)``, True for the singular systems.
    """
    m, n = second_term.shape
    is_singular = numpy.zeros(m, dtype=bool)
    if m == 0:
        return numpy.empty((0, n), dtype=numpy.float64), is_singular
    try:
        delta = numpy.linalg.solve(hessian, -second_term[..., None])[..., 0]
    except numpy.linalg.LinAlgError:
        delta = numpy.full((m, n), numpy.nan, dtype=numpy.float64)
        for a in range(m):
            try:
                delta[a] = numpy.linalg.solve(hessian[a], -second_term[a])
            except numpy.linalg.LinAlgError:
                is_singular[a] = True
    return delta, is_singular


def _finite_mean(values: Optional[numpy.ndarray]) -> float:
    r"""
    Mean over the finite values only (used for the display: a NaN/Inf problem must not
    hide the others).
    """
    if values is None:
        return numpy.nan
    values = numpy.asarray(values, dtype=numpy.float64)
    finite = values[numpy.isfinite(values)]
    return float(numpy.mean(finite)) if finite.size > 0 else numpy.nan


def solve_batch(
    terms: Union[BatchTerm, Sequence[BatchTerm]],
    p0: ArrayLike,
    parametrization: Optional[BatchParametrization] = None,
    *,
    max_iteration: Optional[Integral] = None,
    max_time: Optional[Real] = None,
    ftol: Optional[Real] = None,
    xtol: Optional[Real] = None,
    gtol: Optional[Real] = None,
    atol: Optional[Real] = None,
    ptol: Optional[Real] = None,
    callback_func: Optional[Callable[[Dict], Union[bool, ArrayLike]]] = None,
    update_func: Optional[Callable[[numpy.ndarray, numpy.ndarray, numpy.ndarray], ArrayLike]] = None,
    damping: Optional[str] = None,
    lm_conf: Optional[Mapping[str, Real]] = None,
    verbosity: Integral = 0,
    history: bool = False,
    history_details: Optional[Union[str, Sequence[str]]] = None,
    history_length: Optional[Integral] = None,
) -> BatchSolveResult:
    r"""
    Function to solve a batch of ``k`` independent least squares problems using the
    Gauss-Newton method with robust cost functions.

    This is the batched counterpart of :func:`pysolvegn.solve`. The parameters of the
    ``k`` problems are stored in an array with shape ``(k, n_parameters)`` and each
    problem :math:`k` solves:

    .. math::

        \min_{\mathbf{p}_{in,k}} \frac{1}{2} \sum_{i} w_{i,k} \sum_j \rho_i\left(\| \mathbf{r}_{i,j}\left(P(\mathbf{p}_{in,k}), k\right) \|^2\right)

    where each term :math:`i` is a :class:`pysolvegn.BatchTerm` with its own residual
    function, Jacobian function, weight :math:`w_{i,k}` (scalar or one value per problem)
    and loss function :math:`\rho_i`.

    Here :math:`\mathbf{p}_{in,k}` represents the parameters of the problem :math:`k`
    actually optimized by the solver, and :math:`P` is a parametric transformation
    (:class:`pysolvegn.BatchParametrization`), **the same for all the problems**, such that:

    .. math::

        \mathbf{p}_{out,k} = P(\mathbf{p}_{in,k})

    represents the parameters passed to the residual and Jacobian functions
    of each terms. If no parametrization is provided, the identity
    transformation is implicitly used:

    .. math::

        P(\mathbf{p}_{in,k}) = \mathbf{p}_{in,k}

    The stopping criteria are applied to **each problem independently**: a problem
    satisfying a criterion is removed from the processing and its parameters are frozen,
    while the other problems keep iterating. The terms are therefore only evaluated on
    the ``m <= k`` problems still in processing: ``m`` is NOT fixed and changes during the
    optimization (see :class:`pysolvegn.BatchTerm` for the ``(parameters, indices)``
    convention). Each problem gives exactly the same result as if it was solved alone
    with :func:`pysolvegn.solve`.

    .. seealso::

        For more details on the notations for the optimization problem,
        please refer to the mathematical section of the documentation.


    Parameters
    ----------
    terms: Union[BatchTerm, Sequence[BatchTerm]]
        The list of terms defining the least squares problems.
        Each term should be an instance of the :class:`BatchTerm` class containing
        the residual function, Jacobian function, weight, and loss function
        defining the term. A term with a vector weight must have a weight with
        shape ``(k,)``.
        The residual and Jacobian functions of each term are evaluated using
        the output parametric parameters :math:`\mathbf{p}_{out} = P(\mathbf{p}_{in})`
        of the problems in processing, with shape ``(m, n_p_outputs)``, and their indices.

    p0: ArrayLike
        The initial guess for the parameters optimized by the solver, with shape
        ``(k, n_parameters)``.
        If a parametrization is provided, ``p0`` represents the initial input
        parameters :math:`\mathbf{p}_{in}` of the parametrization. The initial
        output parameters passed to the terms are obtained as
        :math:`\mathbf{p}_{out} = P(\mathbf{p}_{in})`.
        If no parametrization is provided, ``p0`` directly represents the
        parameters passed to the terms.
        The array will not be modified by this function. A copy of the parameters
        will be used for the optimization process.

    parametrization: Optional[BatchParametrization], optional (default=None)
        The parametrization defining the transformation from the parameters
        optimized by the solver to the parameters passed to the terms (the same for
        all the problems).
        If provided, the solver optimizes the input parameters
        :math:`\mathbf{p}_{in}` with shape ``(k, n_parameters)`` of this parametrization.

    max_iteration: Optional[Integral], optional (default=None)
        Maximum number of optimization iterations.
        If provided, all the problems still in processing are stopped after at most
        ``max_iteration`` iterations. If None, no limit on the number of
        iterations is imposed.

    max_time: Optional[Real], optional (default=None)
        Stop criterion by the elapsed time of optimization.
        All the problems still in processing are stopped when the time elapsed since the
        beginning of the optimization exceeds ``max_time`` seconds. If None, no limit
        on the computation time is considered.

    ftol: Optional[Real], optional (default=None)
        Stop criterion by the change of the cost function value.
        A problem is stopped when ``0 <= dF < ftol * F`` where F is its
        cost function value and ``dF = F_previous - F`` is the decrease of its cost
        function value between two iterations (an increase of the cost never satisfies
        this criterion). If None, this criterion is not considered.

    xtol: Optional[Real], optional (default=None)
        Stop criterion by the change of the optimized parameters.
        A problem is stopped when
        ``||dp|| < xtol * (xtol + ||p||)``
        where p is its input parameters :math:`\mathbf{p}_{in,k}` and dp is the
        change of its input parameters between two iterations.
        If None, this criterion is not considered.

    gtol: Optional[Real], optional (default=None)
        Stop criterion by the optimality value.
        A problem is stopped when its optimality verifies
        ``norm(g, ord=numpy.inf) < gtol`` where :math:`g` is its scaled second term.
        If None, this criterion is not considered.

    atol: Optional[Real], optional (default=None)
        Stop criterion by the absolute cost function value.
        A problem is stopped when ``F < atol`` where F is its cost
        function value.
        If None, this criterion is not considered.

    ptol: Optional[Real], optional (default=None)
        Stop criterion by the change of the optimized parameters.
        A problem is stopped when
        ``norm(dp, ord=numpy.inf) < ptol``
        where dp is the change of its input parameters :math:`\mathbf{p}_{in,k}`.
        If None, this criterion is not considered.

    callback_func: Optional[Callable[[Dict], Union[bool, ArrayLike]]], optional (default=None)
        A function called at each iteration, after the built-in stopping criteria,
        to implement custom stopping criteria. It receives a dictionary with the keys
        ``"indices"`` (indices of the ``m`` problems in processing, shape ``(m,)``),
        ``"parameters"`` ``(m, n_parameters)``, ``"delta_parameters"``
        ``(m, n_parameters)`` (None at the first iteration), ``"cost"`` ``(m,)``,
        ``"second_term"`` ``(m, n_parameters)`` and ``"hessian"``
        ``(m, n_parameters, n_parameters)`` of the current iteration, and must return
        either a boolean (applied to all the problems in processing) or an array of
        booleans with shape ``(m,)``: True to continue, False to stop the problem
        (``success=False``).
        If None, no callback function is used.

        .. warning::

            ``"hessian"`` is not a copy: do not modify it in place (copy it first
            with ``hessian.copy()``), as it is used afterwards to compute the update.

    update_func: Optional[Callable[[numpy.ndarray, numpy.ndarray, numpy.ndarray], ArrayLike]], optional (default=None)
        A function that is called at the end of each iteration of the optimization
        process to modify the update of the parameters.
        The function should take (copies of) the current input parameters
        :math:`\mathbf{p}_{in}^{k}` and the Gauss-Newton updates
        :math:`\Delta\mathbf{p}_{in}` (solutions of :math:`\mathbf{H} \Delta\mathbf{p}_{in} = -\mathbf{g}`)
        of ``m`` problems, both with shape ``(m, n_parameters)``, and the indices of these
        problems with shape ``(m,)``, and return the updates
        :math:`\Delta\mathbf{p}_{in}^{*}` actually applied, with shape ``(m, n_parameters)``,
        such that :math:`\mathbf{p}_{in}^{k+1} = \mathbf{p}_{in}^{k} + \Delta\mathbf{p}_{in}^{*}`.
        The returned updates are the ones stored in the history (``"delta_parameters"``).
        With Levenberg-Marquardt (``damping`` not None), ``update_func`` is applied to
        each trial step before its cost is tested, so the applied update is the tested one.
        If None, the Gauss-Newton update is directly applied:
        :math:`\Delta\mathbf{p}_{in}^{*} = \Delta\mathbf{p}_{in}`.

    damping: Optional[str], optional (default=None)
        The method used to compute the update :math:`\Delta\mathbf{p}_{in}` of each problem:

        - None: Gauss-Newton, the update is the solution of
          :math:`\mathbf{H} \Delta\mathbf{p}_{in} = -\mathbf{g}` and is always applied.
        - ``"lm"``: Levenberg-Marquardt, the update is the solution of
          :math:`(\mathbf{H} + \lambda \mathbf{I}) \Delta\mathbf{p}_{in} = -\mathbf{g}`, with
          the initial value :math:`\lambda = 10^{-3} \max(\mathrm{diag}(\mathbf{H}))`.
        - ``"lm-diag"``: Levenberg-Marquardt with the Marquardt scaling, the update is the
          solution of :math:`(\mathbf{H} + \lambda \mathbf{D}) \Delta\mathbf{p}_{in} = -\mathbf{g}`
          where :math:`\mathbf{D} = \mathrm{diag}(\mathbf{H})` (with a floor of
          :math:`10^{-12} \max(\mathrm{diag}(\mathbf{H}))`), with the initial value
          :math:`\lambda = 10^{-3}`. Unlike ``"lm"``, the damping is insensitive to the
          scale of each parameter.

        Each problem has its own damping :math:`\lambda`. With ``"lm"`` and ``"lm-diag"``,
        a trial step is accepted only if it does not increase the cost of the problem:
        otherwise its :math:`\lambda` is multiplied by 10 and its system is solved again.
        After an accepted step, :math:`\lambda` is divided by 10. If no acceptable step is
        found after 50 consecutive rejections, the problem is stopped with
        ``success=False``. The values of :math:`\lambda` are available in the history
        (``"damping"``) and their mean is displayed with ``verbosity >= 2``.

        .. note::

            With Levenberg-Marquardt, the cost of every term must be computable to test
            the steps: all ``gH`` terms must define a ``cost_func``.

    lm_conf: Optional[Mapping[str, Real]], optional (default=None)
        Used only if ``damping`` is ``"lm"`` or ``"lm-diag"``.
        Dictionary to change the default settings of the Levenberg-Marquardt damping
        (``"initial_scale"``, ``"factor"``, ``"max_rejections"`` and ``"diag_floor"``,
        see :func:`pysolvegn.solve`). The missing keys keep their default value. The
        same settings are used for all the problems.

    verbosity: Integral, optional (default=0)
        The level of verbosity for logging the optimization process.
        0: No logging
        1: Log only the final results of the optimization process.
        2: Log the results at each iteration of the optimization process.
        3: Details logging for debugging purposes.
        The logged values are averaged over the problems in processing at the
        current iteration (the total cost of the batch has no meaning).

    history: bool, optional (default=False)
        If True, the history of the optimization process is stored in the
        ``history`` attribute of the returned :class:`BatchSolveResult`.
        See the Notes section for more details.

    history_details: Optional[Union[str, Sequence[str]]], optional (default=None)
        Used only if ``history`` is True.
        Specifies the details to include in the history of the optimization process.
        It can be either a single string or a sequence of strings. See the Notes
        section for the available details.
        If None, the history will include the following details by default:
        ``"iteration"``, ``"elapsed_time"``, ``"n_processing"``, ``"parameters"``,
        ``"delta_parameters"``, ``"delta_cost"``, ``"cost"``, and
        ``"optimality"``.

    history_length: Optional[Integral], optional (default=None)
        Used only if ``history`` is True.
        Controls the number of iterations included in the history.
        If None, all iterations are included in the output history.
        If a positive integer ``N`` is given, only the first ``N`` iterations
        are included in the output history (from 0 to :math:`N-1`).
        If zero is given, no iterations are included.
        If a negative integer ``M`` is given, only the last ``|M|`` iterations
        are included in the output history.
        If the requested history length is greater than the number of iterations
        performed, the history will contain fewer entries than requested.


    Returns
    -------
    result: BatchSolveResult
        The result of the optimization (see :class:`pysolvegn.BatchSolveResult`), with
        the following attributes:

        - ``parameters`` (numpy.ndarray): the optimized input parameters
          :math:`\mathbf{p}_{in}` with shape ``(k, n_parameters)``. If a parametrization
          is provided, the corresponding output parameters are obtained as
          :math:`\mathbf{p}_{out} = P(\mathbf{p}_{in})`.
        - ``history`` (List[Dict]): the history of the optimization process, one
          dictionary per iteration containing the keys described below
          (empty if ``history`` is False).
        - ``success`` (numpy.ndarray): array of booleans with shape ``(k,)``, True for
          each problem for which a convergence criterion (``ftol``, ``atol``, ``gtol``,
          ``xtol`` or ``ptol``) was satisfied and which was not stopped by a NaN or Inf
          value, a singular linear system, the callback function or a failure of the
          Levenberg-Marquardt step (see :class:`BatchSolveResult` for the priority rules).
        - ``stop_code`` (numpy.ndarray): the stopping criteria of each problem (bit
          mask). The methods ``result.stopped_by(name)``, ``result.reasons(k)`` and the property
          ``result.message`` describe them.
        - ``n_iterations`` (numpy.ndarray): the number of updates applied to each problem.
        - ``cost`` (Optional[numpy.ndarray]): the cost of each problem at the returned parameters.
        - ``optimality`` (numpy.ndarray): ``norm(g_k, ord=numpy.inf)`` of each problem at
          the returned parameters.
        - ``term_parameters`` (Optional[numpy.ndarray]): :math:`\mathbf{p}_{out}` at the
          returned parameters.
        - ``elapsed_time`` (float): the total time of the optimization in seconds.
        - ``n_rejected`` (numpy.ndarray): the number of rejected Levenberg-Marquardt
          trial steps of each problem.
        - ``config`` (Dict): the stopping criteria, ``damping`` and ``lm_conf`` requested.


    Notes
    -----

    The solver always checks for NaN or Inf values in the initial parameters ``p0``, the
    output parameters ``p_out`` and the update ``Δp_in*`` (and, with ``damping="lm"``, in
    the costs used to accept the steps) of each problem: the problem is then stopped with
    ``success=False`` and its last valid parameters are returned.

    The step of each loop iteration is as follows (each step is applied to the ``m``
    problems in processing, and "stop" means that the concerned problems are removed from
    the processing):

    .. code-block:: text

        0.  [STOP check] Stop the problems for which ``p0`` contains NaN or Inf values.
            If all the problems are stopped, return immediately.

        While at least one problem is in processing:
            1.  Compute the output parameters ``p_out = P(p_in)`` (``p_out = p_in`` if no parametrization).
            2.  [STOP check] Stop the problems for which ``p_out`` contains NaN or Inf values.
            3.  For ``rJ`` terms, compute ``r_i``, ``J_i`` and the cost ``c_i``, apply the robust
                loss (``r_i -> r~_i``, ``J_i -> J~_i``) and the chain rule ``J~_i -> J~_i @ J_P``,
                then build ``H_i = J~_i.T J~_i`` and ``g_i = J~_i.T r~_i``.
            4.  For ``gH`` terms, compute ``H_i``, ``g_i`` and the cost ``c_i`` (``0.0`` without
                ``cost_func``), then apply the chain rule ``H_i -> J_P.T H_i J_P`` and ``g_i -> J_P.T g_i``.
            5.  Assemble the m systems ``H Δp_in = -g`` with ``H = sum(w_i H_i)`` and
                ``g = sum(w_i g_i)``, and the costs ``C = sum(w_i c_i)``.
            6.  [STOP checks] Store the history, check the stopping criteria of each problem
                (``ftol``, ``atol``, ``gtol``, ``xtol``, ``ptol``, ``max_iteration`` and ``max_time``)
                and call the ``callback_func`` with the current state (``indices``, ``parameters``,
                ``delta_parameters``, ``cost``, ``second_term`` and ``hessian``).
            7.  Remove the stopped problems from the processing (their parameters are frozen,
                see :class:`BatchSolveResult` for the value of ``success``). Exit the loop if
                no problem remains.
            8.  For the remaining problems, compute the update ``Δp_in``:

                - ``damping=None``: solve ``H Δp_in = -g``. Stop the problems with a
                  singular system.
                - ``damping="lm"`` or ``"lm-diag"``: solve ``(H + λ D) Δp_in = -g`` (``D = I`` or
                  ``diag(H)``), apply ``update_func`` and compute the cost ``C_new`` at
                  ``p_in + Δp_in*``. While ``C_new > C``, multiply ``λ`` by 10 and solve again
                  (a trial step with a singular system or NaN/Inf output parameters is also
                  rejected). Then divide ``λ`` by 10. Stop the problems for which ``C`` or
                  ``C_new`` contains NaN or Inf values, or for which no step is accepted
                  after 50 rejections.
            9.  (Gauss-Newton only) Compute the applied update ``Δp_in* = update_func(p_in, Δp_in, indices)``
                (``Δp_in* = Δp_in`` if no ``update_func``).
            10. [STOP check] Stop the problems for which ``Δp_in*`` contains NaN or Inf values
                WITHOUT applying the update (their last valid ``p_in`` is returned).
            11. Update the input parameters ``p_in = p_in + Δp_in*`` of the remaining problems.

    The history contains the following keys (if requested in ``history_details``).
    All the per-problem arrays have a first dimension of size ``k`` (all the problems of
    the batch). For the problems not in processing at the current iteration, the values
    of the last iteration during which they were processed are kept (frozen values),
    except for ``"delta_parameters"``, ``"delta_cost"`` and ``"damping"`` which are set
    to NaN.

    - "iteration": Integer representing the iteration number.
    - "elapsed_time": Float representing the time elapsed since the beginning of the optimization process in seconds.
    - "n_processing": Integer representing the number of problems in processing at the current iteration.
    - "is_processing": Numpy array of booleans with shape ``(k,)``, True for the problems
      in processing at the current iteration.
    - "parameters": Numpy array with shape ``(k, n_parameters)`` representing the input
      parameters :math:`\mathbf{p}_{in}` at the current iteration.
    - "delta_parameters": Numpy array with shape ``(k, n_parameters)`` representing the
      updates :math:`\Delta\mathbf{p}_{in}^{*}` applied between the previous and the current
      iteration (after ``update_func`` if provided), None at the first iteration.
    - "costs": List of numpy arrays with shape ``(k,)`` representing the cost function
      value (without weight) of each term at the current iteration: the ``cost_func`` of
      the term if provided, otherwise
      :math:`\frac{1}{2} \sum_j \rho_i(\| \mathbf{r}_{i,j}(\mathbf{p}_{out,k}) \|^2)`
      for ``rJ`` terms and ``0.0`` for ``gH`` terms.
    - "cost": Numpy array with shape ``(k,)`` representing the cost function value of
      each problem at the current iteration, computed as
      :math:`\frac{1}{2} \sum_i w_{i,k} \sum_j
      \rho_i(\| \mathbf{r}_{i,j}(\mathbf{p}_{out,k}) \|^2)`.
    - "delta_cost": Numpy array with shape ``(k,)`` representing the change of the cost
      function value of each problem between the previous and the current iteration
      (``C - C_previous``), None at the first iteration.
    - "optimality": Numpy array with shape ``(k,)`` representing the optimality value of
      each problem at the current iteration computed as ``norm(g, ord=numpy.inf)`` where
      :math:`g` is the scaled second term.
    - "residuals": A list of numpy arrays with shape ``(k, n_residuals_i)`` representing the
      raw residuals of each term at the current iteration, before the robust loss
      modification (None for ``gH`` terms).
    - "jacobians": A list of numpy arrays with shape ``(k, n_residuals_i, n_p_outputs)``
      representing the raw Jacobians of each term at the current iteration, with respect
      to the output parameters (before the robust loss modification and the chain rule of
      the parametrization; None for ``gH`` terms).
    - "second_term": Numpy array with shape ``(k, n_parameters)`` representing the scaled
      second term :math:`\mathbf{g}` of each linear system at the current iteration.
    - "hessian": Numpy array with shape ``(k, n_parameters, n_parameters)`` representing the
      Hessian approximation :math:`\mathbf{H}` of each linear system at the current iteration.
    - "damping": Numpy array with shape ``(k,)`` representing the Levenberg-Marquardt
      damping :math:`\lambda` used for the accepted step leading to the current iteration
      (None at the first iteration and if ``damping`` is None).
    - "all": include all the keys.

    The cost of each term can be compute only for ``rJ`` terms and is
    defined by :

    .. math::

        \frac{1}{2} \sum_j
        \rho_i\left(
            \left\|
            \mathbf{r}_{i,j}(P(\mathbf{p}_{in,k}))
            \right\|^2
        \right)

    .. important::

        For terms defined directly by a Hessian and gradient (``gH`` terms), only a local quadratic model
        of the cost is available. Since this model is defined up to an arbitrary constant and submit to
        cumulative errors, an absolute cost value cannot be uniquely determined.
        For reporting consistency, the cost contribution of these terms is set to ``0.0`` by default.
        If the term contains a ``cost_func`` function, it will be used to compute the cost of
        the term.

    """

    # Check the validity of the input arguments and raise appropriate errors if necessary.
    if isinstance(terms, BatchTerm):
        terms = [terms]
    if not isinstance(terms, Sequence):
        raise TypeError("terms must be a sequence of BatchTerm objects.")
    if len(terms) == 0:
        raise ValueError("terms sequence cannot be empty.")
    for term in terms:
        if not isinstance(term, BatchTerm):
            raise TypeError(
                "All elements of terms must be instances of the BatchTerm class."
            )

    p0 = numpy.asarray(p0, dtype=numpy.float64)
    if p0.ndim != 2:
        raise ValueError(f"p0 must be a 2D array with shape (k, n_parameters), got {p0.ndim} dimensions.")
    if p0.shape[0] == 0 or p0.shape[1] == 0:
        raise ValueError(f"p0 must not be empty, got shape {p0.shape}.")

    for index, term in enumerate(terms):
        if term.n_problems is not None and term.n_problems != p0.shape[0]:
            raise ValueError(
                f"The vector weight of the term {index} has shape ({term.n_problems},) "
                f"but p0 contains {p0.shape[0]} problems."
            )

    if parametrization is not None and not isinstance(parametrization, BatchParametrization):
        raise TypeError(
            "parametrization must be an instance of the BatchParametrization class."
        )

    if max_iteration is not None:
        if not isinstance(max_iteration, Integral):
            raise TypeError("max_iteration must be an integer.")
        max_iteration = int(max_iteration)
        if max_iteration < 0:
            raise ValueError("max_iteration must be a non-negative integer.")

    if max_time is not None:
        if not isinstance(max_time, Real):
            raise TypeError("max_time must be a real number.")
        max_time = float(max_time)
        if max_time < 0:
            raise ValueError("max_time must be a non-negative real number.")

    if ftol is not None:
        if not isinstance(ftol, Real):
            raise TypeError("ftol must be a real number.")
        ftol = float(ftol)
        if ftol <= 0:
            raise ValueError("ftol must be a positive real number.")

    if xtol is not None:
        if not isinstance(xtol, Real):
            raise TypeError("xtol must be a real number.")
        xtol = float(xtol)
        if xtol <= 0:
            raise ValueError("xtol must be a positive real number.")

    if gtol is not None:
        if not isinstance(gtol, Real):
            raise TypeError("gtol must be a real number.")
        gtol = float(gtol)
        if gtol <= 0:
            raise ValueError("gtol must be a positive real number.")

    if atol is not None:
        if not isinstance(atol, Real):
            raise TypeError("atol must be a real number.")
        atol = float(atol)
        if atol <= 0:
            raise ValueError("atol must be a positive real number.")

    if ptol is not None:
        if not isinstance(ptol, Real):
            raise TypeError("ptol must be a real number.")
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
        raise TypeError("callback_func must be a callable function.")

    if update_func is not None and not callable(update_func):
        raise TypeError("update_func must be a callable function.")

    if not isinstance(verbosity, Integral):
        raise TypeError("verbosity must be an integer.")
    verbosity = int(verbosity)
    if verbosity < 0 or verbosity > 3:
        raise ValueError("verbosity must be an integer between 0 and 3 inclusive.")

    if damping is not None and not isinstance(damping, str):
        raise TypeError("damping must be None or a string.")
    if isinstance(damping, str):
        damping = damping.lower()
    if damping not in _IMPLEMENTED_DAMPINGS:
        raise ValueError(f"damping must be one of {_IMPLEMENTED_DAMPINGS}, got '{damping}'.")
    lm_conf = _validate_lm_conf(lm_conf)
    if damping in ("lm", "lm-diag"):
        for index, term in enumerate(terms):
            if term.type == "gH" and term.c_func is None:
                raise ValueError(
                    f"damping='{damping}' requires the cost of every term: the 'gH' term {index} "
                    "must define a cost_func."
                )

    if not isinstance(history, bool):
        raise TypeError("history must be a boolean value.")
    history = bool(history)

    if history_details is None:
        history_details = list(_DEFAULT_BATCH_SOLVE_HISTORY)
    if isinstance(history_details, str):
        history_details = [history_details]
    if not isinstance(history_details, Sequence):
        raise TypeError("history_details must be a string or a sequence of strings.")
    for detail in history_details:
        if detail not in _IMPLEMENTED_BATCH_HISTORY_DETAILS and detail != "all":
            raise ValueError(
                f"Invalid history detail: {detail}. Valid details are: {_IMPLEMENTED_BATCH_HISTORY_DETAILS} and 'all'."
            )
    if "all" in history_details:
        history_details = list(_IMPLEMENTED_BATCH_HISTORY_DETAILS)

    if history_length is not None and not isinstance(history_length, Integral):
        raise TypeError(
            "history_length must be None or a positive or negative integer."
        )
    if history_length is not None:
        history_length = int(history_length)

    # -- Select Computation --
    use_callback = callback_func is not None
    use_update = update_func is not None
    use_xtol = xtol is not None
    use_ptol = ptol is not None
    use_atol = atol is not None
    use_ftol = ftol is not None
    use_gtol = gtol is not None
    use_maxiter = max_iteration is not None
    use_maxtime = max_time is not None
    use_lm = damping in ("lm", "lm-diag")

    compute_history = history
    compute_cost = (
        use_ftol or use_atol or use_callback or use_lm
        or verbosity >= 2
        or (
            compute_history
            and any(detail in ["cost", "costs", "delta_cost"] for detail in history_details)
        )
    )
    compute_optimality = (
        use_gtol
        or verbosity >= 2
        or (
            compute_history
            and any(detail in ["optimality"] for detail in history_details)
        )
    )
    compute_delta_norm = use_xtol or use_ptol or verbosity >= 2
    compute_params_norm = use_xtol
    compute_conv_analysis = verbosity >= 3

    # ------ Solver Variables and functions
    # - constants
    n_terms = len(terms)
    n_problems, n_parameters = p0.shape
    config = {
        "max_iteration": max_iteration,
        "max_time": max_time,
        "ftol": ftol,
        "xtol": xtol,
        "gtol": gtol,
        "atol": atol,
        "ptol": ptol,
        "damping": damping,
        "lm_conf": lm_conf,
    }

    # - recomputed each loop (reset to None at the start of each iteration)
    #   (all these arrays are relative to the m problems in processing: first dimension m)
    term_parameters: Optional[numpy.ndarray] = None
    jacobian_P = None
    state: Optional[BatchSystemState] = None
    optimality: Optional[numpy.ndarray] = None
    optimality_2: Optional[numpy.ndarray] = None
    cond_Hessian: Optional[numpy.ndarray] = None
    trace_Hessian: Optional[numpy.ndarray] = None
    delta_norm: Optional[numpy.ndarray] = None
    delta_norm_inf: Optional[numpy.ndarray] = None
    parameters_norm: Optional[numpy.ndarray] = None
    end_flags: Optional[numpy.ndarray] = None  # ! (problems to remove from the processing)

    # - conserved between loops
    #   (all these arrays are relative to the k problems of the batch: first dimension k)
    active = numpy.arange(n_problems)  # indices of the m problems in processing
    stop_code = numpy.zeros(n_problems, dtype=numpy.int64)  # bit mask of the triggered criteria (see _STOP_CODES)
    parameters = p0.copy()
    delta_parameters = None  # (k, n_parameters), NaN for the problems not updated
    iteration = 0
    n_iterations = numpy.zeros(n_problems, dtype=numpy.int64)
    history_list = (
        collections.deque(maxlen=-history_length)  # keeps only the last |M| entries
        if history_length is not None and history_length < 0
        else []
    )
    last_total_cost = None  # (k,) costs of the previous iteration
    lm_lambda = numpy.full(n_problems, numpy.nan)  # Levenberg-Marquardt damping (initialized at the first iteration)
    last_damping = None  # (k,) λ used for the accepted step leading to the current iteration
    n_rejected = numpy.zeros(n_problems, dtype=numpy.int64)  # total number of rejected LM trial steps
    next_term_parameters = None  # p_out of the accepted LM trial steps (reused at the next iteration)
    next_cost_state: Optional[BatchCostState] = None  # costs of the accepted LM trial steps (reused)

    # - frozen values of the k problems (last iteration during which each problem was processed)
    frozen = {
        "term_parameters": None,
        "cost": numpy.full(n_problems, numpy.nan) if compute_cost else None,
        "costs": None,
        "optimality": numpy.full(n_problems, numpy.nan),
        "residuals": None,
        "jacobians": None,
        "second_term": None,
        "hessian": None,
    }

    # - functions
    has_naninf = lambda a: ~numpy.all(numpy.isfinite(a.reshape(a.shape[0], -1)), axis=1)  # per problem (row)
    is_first_iteration = lambda: iteration == 0

    def apply_update_func(delta: numpy.ndarray, positions: numpy.ndarray) -> numpy.ndarray:
        # Apply update_func to Gauss-Newton / Levenberg-Marquardt steps of the problems
        # active[positions] and check its shape
        new_delta = numpy.asarray(
            update_func(parameters[active[positions]].copy(), delta.copy(), active[positions].copy()),
            dtype=numpy.float64,
        )
        if new_delta.shape != delta.shape:
            raise ValueError(f"update_func must return an array with shape {delta.shape}.")
        return new_delta

    def stop_problems(mask: numpy.ndarray, name: str) -> None:
        # Stop the problems active[mask] (mask with shape (m,)) with the criterion `name`
        # (one vectorized operation, the text of the reasons is built on demand)
        stop_code[active[mask]] |= _STOP_CODES[name]
        n_iterations[active[mask]] = iteration
        end_flags[mask] = True

    def scatter_frozen(name: str, values: numpy.ndarray) -> None:
        # Store the values of the m problems in processing in the frozen array (k, ...)
        if frozen[name] is None:
            frozen[name] = numpy.full((n_problems,) + values.shape[1:], numpy.nan)
        frozen[name][active] = values

    # Check NaN or Inf values in the initial parameters before the first loop
    end_flags = numpy.zeros(n_problems, dtype=bool)
    stop_problems(has_naninf(parameters), "naninf_p0")
    active = active[~end_flags]
    if active.size == 0:
        result = BatchSolveResult(
            parameters=parameters,
            history=list(history_list),
            success=numpy.zeros(n_problems, dtype=bool),
            stop_code=stop_code,
            n_iterations=n_iterations,
            cost=None,
            optimality=frozen["optimality"],
            term_parameters=None,
            elapsed_time=0.0,
            n_rejected=n_rejected,
            config=config,
        )
        if verbosity >= 1:
            print(result.message)
        return result

    # Printing the header for the optimization process logging based on the verbosity level.
    printed_detail = f""
    printed_header = f""
    if verbosity >= 2:
        printed_detail += (
            f"\nIndividual costs [rJ term]: C_i = 0.5 * ρ(||r_i||^2) "
            f"\nCost: C = sum(w_i * C_i) "
            f"\nStep norm: ||Δp|| "
            f"\nOptimality: ||g|| "
            f"\nAll the values are averaged over the N problems in processing."
        )
        printed_header += (
            f"\n{'Iteration':^10} {'N processing':^15} {'Total time (s)':^15} {'Cost C':^15} {'ΔC':^15}"
            + f" {'||Δp||_2':^15} {'||g||_∞':^15}"
            + (f" {'λ':^15}" if use_lm else "")
        )
    if verbosity >= 3:
        printed_header += (
            f" {'||Δp||_∞':^15} {'||g||_2':^15} {'Cond(H)':^15} {'Trace(H)':^15}"
            + " ".join(
                [f"{'Cost C_' + str(i):^15}" for i in range(n_terms)],
            )
        )
    if verbosity >= 2:
        print(printed_detail)
        print(printed_header)

    # --------- Solver Implementation
    starting_time = time.perf_counter()  # start of the computation (after validation and header)
    while True:  # ! (ensure end-flag activation for term "break" statement)

        # 0. ----- Reset to default variables
        term_parameters = None
        jacobian_P = None
        state = None
        optimality = None
        optimality_2 = None
        cond_Hessian = None
        trace_Hessian = None
        delta_norm = None
        delta_norm_inf = None
        parameters_norm = None
        end_flags = numpy.zeros(active.shape[0], dtype=bool)

        # 1. ----- Apply the parametrization p_out = P(p_in)
        # (reuse p_out of the accepted Levenberg-Marquardt trial steps: same p_in)
        if next_term_parameters is not None:
            term_parameters = next_term_parameters
        else:
            term_parameters = _compute_batch_term_parameters(parametrization, parameters[active])

        # 2. ----- Check parametrization
        is_naninf = has_naninf(term_parameters)
        if numpy.any(is_naninf):
            stop_problems(is_naninf, "naninf_term_parameters")
            scatter_frozen("term_parameters", term_parameters)
            frozen["optimality"][active[is_naninf]] = numpy.nan
            if compute_cost:
                frozen["cost"][active[is_naninf]] = numpy.nan
            active = active[~is_naninf]
            term_parameters = term_parameters[~is_naninf]
            if next_cost_state is not None:
                next_cost_state = next_cost_state.select(~is_naninf)
            end_flags = numpy.zeros(active.shape[0], dtype=bool)
            if active.size == 0:
                break

        # 3-5. ----- Evaluate the systems at p_out: r, J, c, H, g (chain rule with J_P)
        jacobian_P = _compute_batch_jacobian_P(parametrization, parameters[active], term_parameters.shape[1])
        state = _evaluate_batch_system(
            terms,
            term_parameters,
            active,
            jacobian_P,
            compute_cost=compute_cost,
            cost_state=next_cost_state,  # reuse r and costs of the accepted LM trial steps
        )
        next_term_parameters = None
        next_cost_state = None

        # 6. ----- Stopping criterion and storing history
        elapsed_time = time.perf_counter() - starting_time

        # - precomputing
        if compute_optimality:
            optimality = numpy.linalg.norm(state.second_term, ord=numpy.inf, axis=1)

        if compute_conv_analysis:
            optimality_2 = numpy.linalg.norm(state.second_term, ord=2, axis=1)
            trace_Hessian = numpy.trace(state.hessian, axis1=1, axis2=2)
            with numpy.errstate(all="ignore"):
                cond_Hessian = numpy.linalg.cond(state.hessian)

        if compute_delta_norm and not is_first_iteration():
            delta_norm = numpy.linalg.norm(delta_parameters[active], ord=2, axis=1)
            delta_norm_inf = numpy.linalg.norm(delta_parameters[active], ord=numpy.inf, axis=1)

        if compute_params_norm:
            parameters_norm = numpy.linalg.norm(parameters[active], ord=2, axis=1)

        # - Update the frozen values of the problems in processing
        scatter_frozen("term_parameters", term_parameters)
        frozen["optimality"][active] = numpy.linalg.norm(state.second_term, ord=numpy.inf, axis=1)
        if compute_cost:
            frozen["cost"][active] = state.cost

        # - Update history
        if compute_history:

            if "costs" in history_details:
                if frozen["costs"] is None:
                    frozen["costs"] = [numpy.full(n_problems, numpy.nan) for _ in range(n_terms)]
                for index in range(n_terms):
                    frozen["costs"][index][active] = state.costs[index]
            if "residuals" in history_details:
                if frozen["residuals"] is None:
                    frozen["residuals"] = [
                        None if r is None else numpy.full((n_problems,) + r.shape[1:], numpy.nan)
                        for r in state.residuals
                    ]
                for index, r in enumerate(state.residuals):
                    if r is not None:
                        frozen["residuals"][index][active] = r
            if "jacobians" in history_details:
                if frozen["jacobians"] is None:
                    frozen["jacobians"] = [
                        None if J is None else numpy.full((n_problems,) + J.shape[1:], numpy.nan)
                        for J in state.jacobians
                    ]
                for index, J in enumerate(state.jacobians):
                    if J is not None:
                        frozen["jacobians"][index][active] = J
            if "second_term" in history_details:
                scatter_frozen("second_term", state.second_term)
            if "hessian" in history_details:
                scatter_frozen("hessian", state.hessian)

            h = {}
            if "iteration" in history_details:
                h["iteration"] = iteration
            if "n_processing" in history_details:
                h["n_processing"] = int(active.shape[0])
            if "is_processing" in history_details:
                h["is_processing"] = numpy.isin(numpy.arange(n_problems), active)
            if "parameters" in history_details:
                h["parameters"] = parameters.copy()
            if "delta_parameters" in history_details:
                if delta_parameters is None:
                    h["delta_parameters"] = None
                else:
                    h["delta_parameters"] = delta_parameters.copy()
            if "delta_cost" in history_details:
                if last_total_cost is None:
                    h["delta_cost"] = None
                else:
                    h["delta_cost"] = numpy.full(n_problems, numpy.nan)
                    h["delta_cost"][active] = state.cost - last_total_cost[active]
            if "elapsed_time" in history_details:
                h["elapsed_time"] = elapsed_time
            if "costs" in history_details:
                h["costs"] = [c.copy() for c in frozen["costs"]]
            if "cost" in history_details:
                h["cost"] = frozen["cost"].copy()
            if "optimality" in history_details:
                h["optimality"] = frozen["optimality"].copy()
            if "residuals" in history_details:
                h["residuals"] = [
                    r.copy() if r is not None else None for r in frozen["residuals"]
                ]
            if "jacobians" in history_details:
                h["jacobians"] = [
                    J.copy() if J is not None else None for J in frozen["jacobians"]
                ]
            if "second_term" in history_details:
                h["second_term"] = frozen["second_term"].copy()
            if "hessian" in history_details:
                h["hessian"] = frozen["hessian"].copy()
            if "damping" in history_details:
                h["damping"] = None if last_damping is None else last_damping.copy()

            if history_length is None or history_length < 0:
                history_list.append(h)  # (deque with maxlen if history_length < 0)
            elif iteration < history_length:
                # Case 0 -> never append
                history_list.append(h)

        # - Display the iteration (mean over the problems in processing)
        printed_row = f""
        if verbosity >= 2:
            if delta_parameters is None:
                strdp2 = f"{'':^15}"
                strdpinf = f"{'':^15}"
            else:
                strdp2 = f"{_finite_mean(delta_norm):^15.3e}"
                strdpinf = f"{_finite_mean(delta_norm_inf):^15.3e}"
            if last_total_cost is None:
                dC_str = f"{'':^15}"
            else:
                dC = state.cost - last_total_cost[active]
                dC_str = f"{_finite_mean(dC):^15.3e}"
            printed_row += (
                f"{iteration:^10} {active.shape[0]:^15} {elapsed_time:^15.3e} {_finite_mean(state.cost):^15.3e} {dC_str}"
                + f" {strdp2} {_finite_mean(optimality):^15.3e}"
            )
            if use_lm:
                printed_row += f" {'':^15}" if last_damping is None else f" {_finite_mean(last_damping[active]):^15.3e}"
        if verbosity >= 3:
            printed_row += (
                f" {strdpinf} {_finite_mean(optimality_2):^15.3e}"
                + f" {_finite_mean(cond_Hessian):^15.3e} {_finite_mean(trace_Hessian):^15.3e}"
                + " ".join([f"{_finite_mean(c):^15.3e}" for c in state.costs])
            )
        if verbosity >= 2:
            print(printed_row)

        # - Convergence analysis and stopping criteria check (each problem independently)
        if use_ftol and not is_first_iteration():
            dF = last_total_cost[active] - state.cost
            is_converged = (0 <= dF) & (dF < ftol * state.cost)
            stop_problems(is_converged, "ftol")

        if use_atol:
            is_converged = state.cost < atol
            stop_problems(is_converged, "atol")

        if use_gtol:
            is_converged = optimality < gtol
            stop_problems(is_converged, "gtol")

        if use_xtol and not is_first_iteration():
            is_converged = delta_norm < xtol * (xtol + parameters_norm)
            stop_problems(is_converged, "xtol")

        if use_ptol and not is_first_iteration():
            is_converged = delta_norm_inf < ptol
            stop_problems(is_converged, "ptol")

        if use_maxiter:
            if iteration >= max_iteration:
                stop_problems(numpy.ones(active.shape[0], dtype=bool), "max_iteration")

        if use_maxtime:
            if elapsed_time >= max_time:
                stop_problems(numpy.ones(active.shape[0], dtype=bool), "max_time")

        if use_callback:
            callback_state = {
                "indices": active.copy(),
                "parameters": parameters[active].copy(),
                "delta_parameters": None if is_first_iteration() else delta_parameters[active].copy(),
                "cost": state.cost.copy(),
                "second_term": state.second_term.copy(),
                "hessian": state.hessian,
            }
            callback_result = callback_func(callback_state)
            if isinstance(callback_result, (bool, numpy.bool_)):
                callback_result = numpy.full(active.shape[0], bool(callback_result))
            callback_result = numpy.asarray(callback_result)
            if callback_result.dtype != bool or callback_result.shape != (active.shape[0],):
                raise ValueError(
                    f"Callback function must return a boolean or an array of booleans with shape ({active.shape[0]},)."
                )
            stop_problems(~callback_result, "callback")

        # 7. ----- Remove the stopped problems from the processing (exit if no problem remains)
        if numpy.any(end_flags):
            keep = ~end_flags
            active = active[keep]
            term_parameters = term_parameters[keep]
            state = BatchSystemState(
                term_parameters=term_parameters,
                residuals=[None if r is None else r[keep] for r in state.residuals],
                jacobians=[None if J is None else J[keep] for J in state.jacobians],
                costs=None if state.costs is None else [c[keep] for c in state.costs],
                cost=None if state.cost is None else state.cost[keep],
                hessian=state.hessian[keep],
                second_term=state.second_term[keep],
            )
            end_flags = numpy.zeros(active.shape[0], dtype=bool)
        if active.size == 0:
            break

        # 8. ----- Compute the update Δp_in
        if not use_lm:
            # - Gauss-Newton: solve H Δp_in = -g
            active_delta, is_singular = _solve_batch_linear_systems(state.hessian, state.second_term)
            stop_problems(is_singular, "singular")

            # 9. ----- Compute the update parameters (Gauss-Newton only, inside the LM loop otherwise)
            if use_update and numpy.any(~end_flags):
                positions = numpy.flatnonzero(~end_flags)
                active_delta[positions] = apply_update_func(active_delta[positions], positions)

        else:
            # - Levenberg-Marquardt: solve (H + λ D) Δp_in = -g until the cost does not increase
            is_naninf = has_naninf(state.cost)
            stop_problems(is_naninf, "naninf_cost")

            # Damping matrices D: identity ("lm") or diag(H) with a floor ("lm-diag")
            m = active.shape[0]
            diagonal_H = numpy.diagonal(state.hessian, axis1=1, axis2=2)  # (m, n_parameters)
            max_diagonal_H = numpy.max(diagonal_H, axis=1)  # (m,)
            if damping == "lm":
                diagonal_D = numpy.ones((m, n_parameters))
            else:
                floor = numpy.where(max_diagonal_H > 0, lm_conf["diag_floor"] * max_diagonal_H, lm_conf["diag_floor"])
                diagonal_D = numpy.maximum(diagonal_H, floor[:, None])
            damping_matrix = diagonal_D[:, :, None] * numpy.eye(n_parameters)  # (m, n, n)

            is_new = numpy.isnan(lm_lambda[active])
            if numpy.any(is_new):
                if damping == "lm":
                    initial_lambda = lm_conf["initial_scale"] * max_diagonal_H
                    initial_lambda = numpy.where(initial_lambda > 0, initial_lambda, lm_conf["initial_scale"])
                else:
                    initial_lambda = numpy.full(m, lm_conf["initial_scale"])
                lm_lambda[active[is_new]] = initial_lambda[is_new]

            active_lambda = lm_lambda[active]  # (m,) modified in the trial loop
            active_delta = numpy.full((m, n_parameters), numpy.nan)
            new_cost = numpy.full(m, numpy.inf)
            nan_cost = numpy.zeros(m, dtype=bool)
            n_rejections = numpy.zeros(m, dtype=numpy.int64)
            trial_term_parameters = numpy.full((m, term_parameters.shape[1]), numpy.nan)
            trial_costs = [numpy.full(m, numpy.nan) for _ in range(n_terms)]
            trial_residuals: List[Optional[numpy.ndarray]] = [None] * n_terms
            is_pending = ~end_flags  # problems still looking for an acceptable step
            is_exhausted = numpy.zeros(m, dtype=bool)
            while numpy.any(is_pending):
                is_rejected = is_pending & (n_rejections > 0)
                active_lambda[is_rejected] *= lm_conf["factor"]
                is_exhausted |= is_pending & (n_rejections >= lm_conf["max_rejections"])
                is_pending &= ~is_exhausted
                if not numpy.any(is_pending):
                    break
                n_rejections[is_pending] += 1

                positions = numpy.flatnonzero(is_pending)
                trial_delta, is_singular = _solve_batch_linear_systems(
                    state.hessian[positions] + active_lambda[positions, None, None] * damping_matrix[positions],
                    state.second_term[positions],
                )
                positions = positions[~is_singular]  # rejected: larger λ
                trial_delta = trial_delta[~is_singular]

                # 9. ----- Compute the update parameters (applied to each trial step)
                if use_update and positions.size > 0:
                    trial_delta = apply_update_func(trial_delta, positions)

                trial_term = _compute_batch_term_parameters(parametrization, parameters[active[positions]] + trial_delta)
                is_valid = ~has_naninf(trial_term)  # rejected: larger λ
                positions = positions[is_valid]
                trial_delta = trial_delta[is_valid]
                trial_term = trial_term[is_valid]

                if positions.size > 0:
                    trial_cost_state = _evaluate_batch_cost(terms, trial_term, active[positions])
                    new_cost[positions] = trial_cost_state.cost
                    active_delta[positions] = trial_delta
                    trial_term_parameters[positions] = trial_term
                    for index in range(n_terms):
                        trial_costs[index][positions] = trial_cost_state.costs[index]
                        r = trial_cost_state.residuals[index]
                        if r is not None:
                            if trial_residuals[index] is None:
                                trial_residuals[index] = numpy.full((m,) + r.shape[1:], numpy.nan)
                            trial_residuals[index][positions] = r
                    nan_cost[positions] = has_naninf(trial_cost_state.cost)  # the step cannot be compared

                is_pending &= (new_cost > state.cost) & ~nan_cost

            is_accepted = ~end_flags & ~is_exhausted & ~nan_cost & (new_cost <= state.cost)
            is_tried = ~end_flags
            n_rejected[active[is_tried]] += numpy.where(
                is_accepted[is_tried], n_rejections[is_tried] - 1, n_rejections[is_tried]
            )

            stop_problems(nan_cost & ~end_flags, "naninf_trial_cost")

            stop_problems(is_tried & ~is_accepted & ~end_flags, "lm")

            # Accepted steps: reuse p_out and the costs at the next iteration (same p_in)
            if last_damping is None:
                last_damping = numpy.full(n_problems, numpy.nan)
            last_damping[:] = numpy.nan
            last_damping[active[is_accepted]] = active_lambda[is_accepted]
            lm_lambda[active[is_accepted]] = active_lambda[is_accepted] / lm_conf["factor"]
            next_term_parameters = trial_term_parameters
            next_cost_state = BatchCostState(
                cost=new_cost,
                costs=trial_costs,
                residuals=trial_residuals,
            )

        # 10. ----- Check update parameters
        is_naninf = has_naninf(active_delta) & ~end_flags
        stop_problems(is_naninf, "naninf_update")

        # 11. ----- Update the parameters of the remaining problems
        keep = ~end_flags
        if delta_parameters is None:
            delta_parameters = numpy.full((n_problems, n_parameters), numpy.nan)
        delta_parameters[:] = numpy.nan
        delta_parameters[active[keep]] = active_delta[keep]
        parameters[active[keep]] = parameters[active[keep]] + active_delta[keep]

        if compute_cost:
            if last_total_cost is None:
                last_total_cost = numpy.full(n_problems, numpy.nan)
            last_total_cost[active] = state.cost

        active = active[keep]
        if next_term_parameters is not None:
            next_term_parameters = next_term_parameters[keep]
            next_cost_state = next_cost_state.select(keep)

        iteration += 1

        if active.size == 0:
            break

    # ---- End of solver
    elapsed_time = time.perf_counter() - starting_time

    # success: a convergence criterion and no failure (see BatchSolveResult for the priority rules)
    success = ((stop_code & _STOP_CONVERGENCE) != 0) & ((stop_code & _STOP_FAILURE) == 0)

    # The returned parameters are always the last evaluated ones (the update is never
    # applied to a stopped problem), so the frozen values describe them.
    cost = frozen["cost"]
    if not compute_cost and numpy.any(success):
        # The cost was not required during the optimization: computed once for the
        # successful problems at their solution
        indices = numpy.flatnonzero(success)
        cost = numpy.full(n_problems, numpy.nan)
        cost[indices] = _evaluate_batch_cost(terms, frozen["term_parameters"][indices], indices).cost

    result = BatchSolveResult(
        parameters=parameters,
        history=list(history_list),
        success=success,
        stop_code=stop_code,
        n_iterations=n_iterations,
        cost=cost,
        optimality=frozen["optimality"],
        term_parameters=frozen["term_parameters"],
        elapsed_time=elapsed_time,
        n_rejected=n_rejected,
        config=config,
    )
    if verbosity >= 1:
        print(result.message)
    return result