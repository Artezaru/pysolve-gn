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

from typing import Optional, Sequence, Union, Callable, Dict, List, Any, Mapping
from numbers import Real, Integral
from dataclasses import dataclass
from numpy.typing import ArrayLike

import time
import warnings

import numpy
import scipy.sparse
import scipy.sparse.linalg

from .implemented_conf import (
    _IMPLEMENTED_HISTORY_DETAILS,
    _DEFAULT_SOLVE_HISTORY,
    _STOP_CODES,
    _STOP_CONVERGENCE,
    _STOP_NANINF,
    _STOP_FAILURE,
)
from .solver_common import (
    _HistoryRecorder,
    _IterationDisplay,
    _copy_or_none,
    _stop_reasons,
    _validate_solver_options,
    _validate_terms,
)

from .evaluation import (
    CostState,
    SystemState,
    _compute_term_parameters,
    _compute_jacobian_P,
    _evaluate_cost,
    _evaluate_system,
)

from .term import Term
from .parametrization import Parametrization

_MACHINE_EPSILON = float(numpy.finfo(numpy.float64).eps)


@dataclass
class SolveResult:
    r"""
    Result of the optimization performed by :func:`pysolvegn.solve`.

    Attributes
    ----------
    parameters: numpy.ndarray
        The optimized input parameters :math:`\mathbf{p}_{in}` with shape
        ``(n_parameters,)``.
        If no parametrization is provided, these parameters are also the
        parameters passed directly to the terms.
        If a parametrization is provided, the corresponding parameters passed to the
        terms are available in ``term_parameters``.

    history: List[Dict]
        The history of the optimization process. Each element of the list is a
        dictionary describing one iteration (see the Notes section of
        :func:`pysolvegn.solve` for the available keys).
        Empty if ``history`` is False.

    success: bool
        True if the optimization stopped because a convergence criterion
        (``ftol``, ``atol``, ``gtol``, ``xtol`` or ``ptol``) was satisfied.
        The value is determined with the following priority:

        1. Always False if the optimization was stopped by a NaN or Inf value (in the
           initial parameters, the term parameters, the update or, with
           ``damping="lm"``, the cost), by a singular linear system, by the callback
           function, by a failure of the Levenberg-Marquardt step or of the line search,
           even if a convergence criterion was satisfied at the same iteration.
        2. Otherwise True if a convergence criterion was satisfied, even if
           ``max_iteration`` or ``max_time`` was reached at the same iteration.
        3. Otherwise False (stopped by ``max_iteration`` or ``max_time``).

    stop_code: int
        The reasons why the optimization stopped, encoded as a bit mask: one bit per
        stopping criterion triggered at the last iteration (several criteria can be
        triggered at the same iteration). Use :meth:`stopped_by` to test a criterion,
        and the properties ``reasons`` or ``message`` to get a text description.
        The bits are listed in the Notes section.

    n_iterations: int
        The number of iterations performed, i.e. the number of updates
        :math:`\Delta\mathbf{p}_{in}^{*}` applied to the parameters (0 if the
        optimization stopped at the initial parameters).

    cost: Optional[float]
        The cost function value :math:`C` at the returned parameters.
        If the optimization succeeded (``success=True``), the cost is always available
        (computed at the end if no criterion, damping, callback, history or verbosity
        required it during the optimization). Otherwise, None if it was not computed
        or if the terms were not evaluated at the returned parameters (NaN or Inf value
        in the initial or term parameters).

    optimality: Optional[float]
        The optimality ``norm(g, ord=numpy.inf)`` at the returned parameters, where
        :math:`\mathbf{g}` is the scaled second term. None if the terms were not
        evaluated at the returned parameters.

    term_parameters: Optional[numpy.ndarray]
        The parameters passed to the terms, :math:`\mathbf{p}_{out} = P(\mathbf{p}_{in})`,
        at the returned parameters (equal to ``parameters`` if no parametrization is
        provided). None if the optimization did not start (NaN or Inf value in ``p0``).

    elapsed_time: float
        The total time of the optimization in seconds.

    n_rejected: int
        The total number of rejected trial steps of the Levenberg-Marquardt damping or
        of the line search (0 if ``damping`` is None and ``line_search`` is False).

    config: Dict[str, Any]
        The configuration requested to the solver: the stopping criteria
        (``"max_iteration"``, ``"max_time"``, ``"ftol"``, ``"xtol"``, ``"gtol"``,
        ``"atol"``, ``"ptol"``, None if not used), ``"damping"``, ``"lm_conf"``
        (the complete Levenberg-Marquardt configuration, default values included),
        ``"line_search"`` and ``"line_search_conf"`` (complete line search configuration).

    Notes
    -----
    Bits of ``stop_code`` (``stop_code`` is the sum of the bits of the criteria
    triggered at the last iteration):

    +------------------------------+-------+----------------------------------------------------+
    | Name                         | Bit   | Meaning                                            |
    +==============================+=======+====================================================+
    | ``"ftol"``                   | 1     | Relative decrease of the cost < ``ftol``           |
    +------------------------------+-------+----------------------------------------------------+
    | ``"atol"``                   | 2     | Cost < ``atol``                                    |
    +------------------------------+-------+----------------------------------------------------+
    | ``"gtol"``                   | 4     | Optimality < ``gtol``                              |
    +------------------------------+-------+----------------------------------------------------+
    | ``"xtol"``                   | 8     | Relative step < ``xtol``                           |
    +------------------------------+-------+----------------------------------------------------+
    | ``"ptol"``                   | 16    | Maximal absolute step < ``ptol``                   |
    +------------------------------+-------+----------------------------------------------------+
    | ``"max_iteration"``          | 32    | Maximum number of iterations reached               |
    +------------------------------+-------+----------------------------------------------------+
    | ``"max_time"``               | 64    | Maximum computation time reached                   |
    +------------------------------+-------+----------------------------------------------------+
    | ``"callback"``               | 128   | Stopped by the callback function                   |
    +------------------------------+-------+----------------------------------------------------+
    | ``"singular"``               | 256   | Singular linear system                             |
    +------------------------------+-------+----------------------------------------------------+
    | ``"lm"``                     | 512   | No acceptable Levenberg-Marquardt step             |
    +------------------------------+-------+----------------------------------------------------+
    | ``"naninf_p0"``              | 1024  | NaN or Inf value in ``p0``                         |
    +------------------------------+-------+----------------------------------------------------+
    | ``"naninf_term_parameters"`` | 2048  | NaN or Inf value in :math:`P(\mathbf{p}_{in})`     |
    +------------------------------+-------+----------------------------------------------------+
    | ``"naninf_cost"``            | 4096  | NaN or Inf value in the cost (Levenberg-Marquardt) |
    +------------------------------+-------+----------------------------------------------------+
    | ``"naninf_trial_cost"``      | 8192  | NaN or Inf value in the cost of a trial step       |
    +------------------------------+-------+----------------------------------------------------+
    | ``"naninf_update"``          | 16384 | NaN or Inf value in the update                     |
    +------------------------------+-------+----------------------------------------------------+
    | ``"line_search"``            | 32768 | No step satisfying the Armijo condition            |
    +------------------------------+-------+----------------------------------------------------+
    | ``"not_descent"``            | 65536 | Update not a descent direction (line search)       |
    +------------------------------+-------+----------------------------------------------------+

    For example, ``stop_code = 12 = 4 + 8`` means that ``gtol`` and ``xtol`` were both
    satisfied at the last iteration: ``result.stopped_by("gtol")`` and
    ``result.stopped_by("xtol")`` are True.
    """

    parameters: numpy.ndarray
    history: List[Dict]
    success: bool
    stop_code: int
    n_iterations: int
    cost: Optional[float]
    optimality: Optional[float]
    term_parameters: Optional[numpy.ndarray]
    elapsed_time: float
    n_rejected: int
    config: Dict[str, Any]

    def stopped_by(self, name: str) -> bool:
        r"""
        Test whether a stopping criterion was triggered at the last iteration.

        Parameters
        ----------
        name : str
            The name of the criterion (see ``stop_code``), or ``"naninf"`` for any
            NaN or Inf value.

        Returns
        -------
        bool
            True if the criterion was triggered.
        """
        if name == "naninf":
            return bool(self.stop_code & _STOP_NANINF)
        if name not in _STOP_CODES:
            raise ValueError(f"Unknown stopping criterion '{name}'. Valid names are {tuple(_STOP_CODES)} and 'naninf'.")
        return bool(self.stop_code & _STOP_CODES[name])

    @property
    def reasons(self) -> List[str]:
        r"""
        [Get] The reasons why the optimization stopped, one string per triggered criterion,
        each starting with its tag (e.g. ``"[xtol] ..."``). The texts give the
        thresholds requested to the solver (not the values reached).

        Returns
        -------
        List[str]
            The reasons, in the order of the checks of the solver.
        """
        return _stop_reasons(self.stop_code, self.config)

    @property
    def message(self) -> str:
        r"""
        [Get] A description of the reasons why the optimization stopped (one per line).

        Returns
        -------
        str
            The reasons joined with new lines.
        """
        return "\n".join(self.reasons)


def _solve_linear_system(
    hessian: Union[numpy.ndarray, scipy.sparse.spmatrix],
    second_term: numpy.ndarray,
) -> numpy.ndarray:
    r"""
    Solve the linear system :math:`\mathbf{H} \Delta\mathbf{p} = -\mathbf{g}` (dense or
    sparse).

    Raises ``numpy.linalg.LinAlgError`` if the system is singular, for dense AND sparse
    matrices (the sparse solver does not raise but returns NaN values with a
    ``MatrixRankWarning``: the warning is silenced and converted into the error).
    """
    if scipy.sparse.issparse(hessian):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", scipy.sparse.linalg.MatrixRankWarning)
            delta = scipy.sparse.linalg.spsolve(hessian, -second_term)
        if not numpy.all(numpy.isfinite(delta)):
            raise numpy.linalg.LinAlgError("Singular sparse matrix.")
        return delta
    return numpy.linalg.solve(hessian, -second_term)


def solve(
    terms: Union[Term, Sequence[Term]],
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
    damping: Optional[str] = None,
    lm_conf: Optional[Mapping[str, Real]] = None,
    line_search: bool = False,
    line_search_conf: Optional[Mapping[str, Real]] = None,
    verbosity: Integral = 0,
    history: bool = False,
    history_details: Optional[Union[str, Sequence[str]]] = None,
    history_length: Optional[Integral] = None,
) -> SolveResult:
    r"""
    Function to solve a least squares problem using the Gauss-Newton method
    with robust cost functions.

    The function accepts multiple terms in the least squares problem,
    each with its own residual function :math:`r_i`, Jacobian function
    :math:`J_i = \frac{\partial r_i}{\partial \mathbf{p}},
    weight :math:`w_i`, and loss function :math`\rho_i`,
    solving the following optimization problem:

    .. math::

        \min_{\mathbf{p}_{in}} \frac{1}{2} \sum_{i} w_i \sum_j \rho_i\left(\| \mathbf{r}_{i,j}\left(P(\mathbf{p}_{in})\right) \|^2\right)

    Here :math:`\mathbf{p}_{in}` represents the parameters actually optimized
    by the solver, and :math:`P` is a parametric transformation such that:

    .. math::

        \mathbf{p}_{out} = P(\mathbf{p}_{in})

    represents the parameters passed to the residual and Jacobian functions
    of each terms. The input parameters :math:`\mathbf{p}_{in}` have shape
    ``(n_parameters,)``. If no parametrization is provided, the identity
    transformation is implicitly used:

    .. math::

        P(\mathbf{p}_{in}) = \mathbf{p}_{in}

    .. seealso::

        For more details on the notations for the optimization problem,
        please refer to the mathematical section of the documentation.


    Parameters
    ----------
    terms: Union[Term, Sequence[Term]]
        The list of terms defining the least squares problem.
        Each term should be an instance of the :class:`Term` class containing
        the residual function, Jacobian function, weight, and loss function
        defining the term.
        The residual and Jacobian functions of each term are evaluated using
        the output parametric parameters :math:`\mathbf{p}_{out} = P(\mathbf{p}_{in})`.

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
        Maximum number of optimization iterations.
        If provided, the optimization process is stopped after at most
        ``max_iteration`` iterations. If None, no limit on the number of
        iterations is imposed.

    max_time: Optional[Real], optional (default=None)
        Stop criterion by the elapsed time of optimization.
        The optimization process is stopped when the time elapsed since the
        beginning of the optimization exceeds ``max_time`` seconds. If None, no limit
        on the computation time is considered. The time is checked after the evaluation
        of each iteration and between the trial steps of the Levenberg-Marquardt damping
        and of the line search (the trial step being tested is then not applied).

    ftol: Optional[Real], optional (default=None)
        Stop criterion by the change of the cost function value.
        The optimization process is stopped when ``0 <= dF < ftol * F`` where F is the
        cost function value and ``dF = F_previous - F`` is the decrease of the cost
        function value between two iterations (an increase of the cost never satisfies
        this criterion). If None, this criterion is not considered.
        Not allowed if a ``gH`` term has no ``cost_func`` (ValueError).

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
        ``norm(g, ord=numpy.inf) < gtol`` where :math:`g` is the scaled second term.
        If None, this criterion is not considered.

    atol: Optional[Real], optional (default=None)
        Stop criterion by the absolute cost function value.
        The optimization process is stopped when ``F < atol`` where F is the cost
        function value.
        If None, this criterion is not considered.
        Not allowed if a ``gH`` term has no ``cost_func`` (ValueError).

    ptol: Optional[Real], optional (default=None)
        Stop criterion by the change of the optimized parameters.
        The optimization process is stopped when
        ``norm(dp, ord=numpy.inf) < ptol``
        where dp is the change of the input parameters :math:`\mathbf{p}_{in}`.
        If None, this criterion is not considered.

    callback_func: Optional[Callable[[Dict], bool]], optional (default=None)
        A function called at each iteration, after the built-in stopping criteria,
        to implement custom stopping criteria. It receives a dictionary with the keys
        ``"parameters"``, ``"delta_parameters"`` (None at the first iteration),
        ``"cost"``, ``"second_term"`` and ``"hessian"`` of the current iteration, and
        must return a boolean: True to continue, False to stop the optimization
        (``success=False``).
        If None, no callback function is used.

        .. warning::

            ``"hessian"`` is not a copy: do not modify it in place (copy it first
            with ``hessian.copy()``), as it is used afterwards to compute the update.

    update_func: Optional[Callable[[numpy.ndarray, numpy.ndarray], ArrayLike]], optional (default=None)
        A function that is called at the end of each iteration of the optimization
        process to modify the update of the parameters.
        The function should take (copies of) the current input parameters
        :math:`\mathbf{p}_{in}^{k}` and the Gauss-Newton update
        :math:`\Delta\mathbf{p}_{in}` (solution of :math:`\mathbf{H} \Delta\mathbf{p}_{in} = -\mathbf{g}`),
        both with shape ``(n_parameters,)``, and return the update
        :math:`\Delta\mathbf{p}_{in}^{*}` actually applied, with shape ``(n_parameters,)``,
        such that :math:`\mathbf{p}_{in}^{k+1} = \mathbf{p}_{in}^{k} + \Delta\mathbf{p}_{in}^{*}`.
        The returned update is the one stored in the history (``"delta_parameters"``).
        With Levenberg-Marquardt (``damping`` not None), ``update_func`` is applied to
        each trial step before its cost is tested, so the applied update is the tested one.
        If None, the Gauss-Newton update is directly applied:
        :math:`\Delta\mathbf{p}_{in}^{*} = \Delta\mathbf{p}_{in}`.

    damping: Optional[str], optional (default=None)
        The method used to compute the update :math:`\Delta\mathbf{p}_{in}`:

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

        With ``"lm"`` and ``"lm-diag"``, a trial step is accepted only if it does not
        increase the cost beyond its rounding noise (:math:`C_{new} \leq C + \eta` with
        :math:`\eta = \text{noise} \times |C|`, see ``lm_conf``): otherwise :math:`\lambda` is multiplied by 10 and the system
        is solved again. After an accepted step, :math:`\lambda` is divided by 10 (down to
        :math:`\varepsilon_m` times its initial scale, below which :math:`\lambda \mathbf{D}`
        is lost in the rounding of :math:`\mathbf{H}`). If no
        acceptable step is found after 50 consecutive rejections, the optimization is
        stopped with ``success=False``. The value of :math:`\lambda` is available in the
        history (``"damping"``) and displayed with ``verbosity >= 2``. These default
        values can be changed with ``lm_conf``.

        .. note::

            With Levenberg-Marquardt, the cost of every term must be computable to test
            the steps: all ``gH`` terms must define a ``cost_func``.

    lm_conf: Optional[Mapping[str, Real]], optional (default=None)
        Used only if ``damping`` is ``"lm"`` or ``"lm-diag"``.
        Dictionary to change the default settings of the Levenberg-Marquardt damping.
        The missing keys keep their default value:

        - ``"initial_scale"`` (default ``1e-3``): initial value of :math:`\lambda`
          (``"lm-diag"``), or factor of :math:`\max(\mathrm{diag}(\mathbf{H}))` (``"lm"``).
        - ``"factor"`` (default ``10.0``, must be > 1): :math:`\lambda` is multiplied by
          this factor after a rejected step and divided by it after an accepted step.
        - ``"max_rejections"`` (default ``50``): maximum number of consecutive rejected
          steps before stopping with ``success=False``.
        - ``"diag_floor"`` (default ``1e-12``): (``"lm-diag"`` only) floor of
          :math:`\mathrm{diag}(\mathbf{H})` relative to its maximum.
        - ``"noise"`` (default ``1e-13``, ``>= 0``): relative rounding noise of the cost
          :math:`\eta / |C|` tolerated to accept a step (``0.0`` for the strict condition
          :math:`C_{new} \leq C`). At the machine precision the cost cannot decrease anymore,
          and this tolerance avoids rejecting steps (and increasing :math:`\lambda`) in vain.

        The complete configuration used is stored in ``result.config["lm_conf"]``.

    line_search: bool, optional (default=False)
        Used only if ``damping`` is None (Gauss-Newton). If False (default), the full
        Gauss-Newton step is always applied: :math:`\mathbf{H} \Delta\mathbf{p}_{in} = -\mathbf{g}`
        and :math:`\mathbf{p}_{in}^{k+1} = \mathbf{p}_{in}^{k} + \Delta\mathbf{p}_{in}^{*}`, nothing more.
        If True, an Armijo backtracking line search is applied along the update
        :math:`\Delta\mathbf{p}_{in}^{*}` (after ``update_func``): the step
        :math:`\alpha \Delta\mathbf{p}_{in}^{*}` is applied with the first
        :math:`\alpha = 1, \beta, \beta^2, ...` such that

        .. math::

            C(\mathbf{p}_{in} + \alpha \Delta\mathbf{p}_{in}^{*}) \leq C(\mathbf{p}_{in}) + c_1 \alpha \, \mathbf{g}^T \Delta\mathbf{p}_{in}^{*} + \eta

        where :math:`\eta = \text{noise} \times |C(\mathbf{p}_{in})|` is the tolerated rounding
        noise of the cost (``line_search_conf["noise"]``, default ``1e-13``): at the machine
        precision the cost cannot decrease anymore, and the steps that do not increase it
        beyond its rounding error are accepted at once instead of being shortened in vain.
        A trial step with NaN or Inf output parameters or cost is rejected (smaller
        :math:`\alpha`). Each trial step only evaluates the residuals (and the
        ``cost_func``): no Jacobian, Hessian or linear system. The residuals of the accepted
        step are reused at the next iteration, so a full step accepted at once
        (:math:`\alpha = 1`) costs exactly the same as the pure Gauss-Newton iteration.
        The optimization is stopped with ``success=False`` if the update is not a descent
        direction (:math:`\mathbf{g}^T \Delta\mathbf{p}_{in}^{*} > \eta`, stop code
        ``"not_descent"``, e.g. an indefinite Hessian of a ``gH`` term or an ``update_func``)
        or if no step satisfies the condition (stop code ``"line_search"``).
        The value of :math:`\alpha` is available in the history (``"step_length"``) and
        displayed with ``verbosity >= 2``.

        .. note::

            The line search requires the cost of every term: all ``gH`` terms must define
            a ``cost_func`` (ValueError otherwise). It cannot be combined with ``damping``
            (ValueError).

    line_search_conf: Optional[Mapping[str, Real]], optional (default=None)
        Used only if ``line_search`` is True. Dictionary to change the default settings of
        the line search. The missing keys keep their default value:

        - ``"c1"`` (default ``1e-4``, in ``(0, 1)``): the sufficient decrease constant :math:`c_1`.
        - ``"factor"`` (default ``0.5``, in ``(0, 1)``): the backtracking factor :math:`\beta`.
        - ``"max_backtracks"`` (default ``30``): maximum number of rejected trial steps in one
          iteration before stopping with ``success=False``.
        - ``"noise"`` (default ``1e-13``, ``>= 0``): relative rounding noise of the cost
          :math:`\eta / |C|` tolerated by the condition (``0.0`` for the strict condition).

        The complete configuration used is stored in ``result.config["line_search_conf"]``.

    verbosity: Integral, optional (default=0)
        The level of verbosity for logging the optimization process.
        0: No logging
        1: Log only the final results of the optimization process.
        2: Log the results at each iteration of the optimization process.
        3: Details logging for debugging purposes.

    history: bool, optional (default=False)
        If True, the history of the optimization process is stored in the
        ``history`` attribute of the returned :class:`SolveResult`.
        See the Notes section for more details.

    history_details: Optional[Union[str, Sequence[str]]], optional (default=None)
        Used only if ``history`` is True.
        Specifies the details to include in the history of the optimization process.
        It can be either a single string or a sequence of strings. See the Notes
        section for the available details.
        If None, the history will include the following details by default:
        ``"iteration"``, ``"elapsed_time"``, ``"parameters"``,
        ``"delta_parameters"``, ``"delta_cost"``, ``"cost"``, and
        ``"optimality"`` (see ``_DEFAULT_SOLVE_HISTORY``).

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
    result: SolveResult
        The result of the optimization (see :class:`pysolvegn.SolveResult`), with
        the following attributes:

        - ``parameters`` (numpy.ndarray): the optimized input parameters
          :math:`\mathbf{p}_{in}` with shape ``(n_parameters,)``. If a parametrization
          is provided, the corresponding output parameters are obtained as
          :math:`\mathbf{p}_{out} = P(\mathbf{p}_{in})`.
        - ``history`` (List[Dict]): the history of the optimization process, one
          dictionary per iteration containing the keys described below
          (empty if ``history`` is False).
        - ``success`` (bool): True if a convergence criterion (``ftol``, ``atol``,
          ``gtol``, ``xtol`` or ``ptol``) was satisfied and the optimization was not
          stopped by a NaN or Inf value, a singular linear system, the callback function
          or a failure of the Levenberg-Marquardt step or of the line search (see
          :class:`SolveResult` for the priority rules).
        - ``stop_code`` (int): the stopping criteria triggered at the last iteration
          (bit mask). The method ``result.stopped_by(name)`` and the properties
          ``result.reasons`` and ``result.message`` describe them.
        - ``n_iterations`` (int): the number of updates applied to the parameters.
        - ``cost`` (Optional[float]): the cost at the returned parameters (always
          available if ``success`` is True).
        - ``optimality`` (Optional[float]): ``norm(g, ord=numpy.inf)`` at the returned
          parameters.
        - ``term_parameters`` (Optional[numpy.ndarray]): :math:`\mathbf{p}_{out} = P(\mathbf{p}_{in})`
          at the returned parameters.
        - ``elapsed_time`` (float): the total time of the optimization in seconds.
        - ``n_rejected`` (int): the number of rejected trial steps (Levenberg-Marquardt or line search).
        - ``config`` (Dict): the stopping criteria, ``damping`` and ``lm_conf`` requested.


    Notes
    -----

    The solver always checks for NaN or Inf values in the initial parameters ``p0``, the
    output parameters ``p_out`` and the update ``Δp_in*`` (and, with ``damping="lm"``, in
    the costs used to accept the steps): the optimization is then stopped with
    ``success=False`` and the last valid parameters are returned.

    The step of each loop iteration is as follows:

    .. code-block:: text

        0.  [STOP check] If ``p0`` contains NaN or Inf values, return
            immediately with ``success=False``.

        While NOT stopped:
            1.  Compute the output parameters ``p_out = P(p_in)`` (``p_out = p_in`` if no parametrization).
            2.  [STOP check] If ``p_out`` contains NaN or Inf values,
                stop with ``success=False``.
            3.  For ``rJ`` terms, compute ``r_i``, ``J_i`` and the cost ``c_i``, apply the robust
                loss (``r_i -> r~_i``, ``J_i -> J~_i``) and the chain rule ``J~_i -> J~_i @ J_P``,
                then build ``H_i = J~_i.T J~_i`` and ``g_i = J~_i.T r~_i``.
            4.  For ``gH`` terms, compute ``H_i``, ``g_i`` and the cost ``c_i`` (``0.0`` without
                ``cost_func``), then apply the chain rule ``H_i -> J_P.T H_i J_P`` and ``g_i -> J_P.T g_i``.
            5.  Assemble the full system ``H Δp_in = -g`` with ``H = sum(w_i H_i)`` and
                ``g = sum(w_i g_i)``, and the cost ``C = sum(w_i c_i)``.
            6.  [STOP checks] Store the history, check the stopping criteria (``ftol``, ``atol``,
                ``gtol``, ``xtol``, ``ptol``, ``max_iteration`` and ``max_time``) and call the
                ``callback_func`` with the current state (``parameters``,
                ``delta_parameters``, ``cost``, ``second_term`` and ``hessian``).
            7.  If STOP, exit the loop and return the current ``p_in`` and the history
                (see :class:`SolveResult` for the value of ``success``).
            8.  If CONTINUE, compute the update ``Δp_in``:

                - ``damping=None``: solve ``H Δp_in = -g``. If the system is singular, stop
                  with ``success=False``.
                - ``damping="lm"`` or ``"lm-diag"``: solve ``(H + λ D) Δp_in = -g`` (``D = I`` or
                  ``diag(H)``), apply ``update_func`` and compute the cost ``C_new`` at
                  ``p_in + Δp_in*``. While ``C_new > C``, multiply ``λ`` by 10 and solve again
                  (a trial step with a singular system or NaN/Inf output parameters is also
                  rejected). Then divide ``λ`` by 10. Stop with ``success=False`` if ``C`` or
                  ``C_new`` contains NaN or Inf values, or if no step is accepted after 50
                  rejections.
            9.  (Gauss-Newton only) Compute the applied update ``Δp_in* = update_func(p_in, Δp_in)``
                (``Δp_in* = Δp_in`` if no ``update_func``).
                With ``line_search=True``: stop with ``success=False`` if ``g^T Δp_in* >= 0``,
                otherwise replace ``Δp_in*`` by ``α Δp_in*`` with the first
                ``α = 1, β, β^2, ...`` such that ``C(p_in + α Δp_in*) <= C + c1 α g^T Δp_in*``
                (only the cost is evaluated; stop with ``success=False`` if none is found).
            10. [STOP check] If ``Δp_in*`` contains NaN or Inf values, stop with
                ``success=False`` WITHOUT applying the update (the last valid ``p_in`` is returned).
            11. Update the input parameters ``p_in = p_in + Δp_in*``.
   
    The history contains the following keys (if requested in ``history_details``):

    - "iteration": Integer representing the iteration number.
    - "elapsed_time": Float representing the time elapsed since the beginning of the optimization process in seconds.
    - "parameters": Numpy array representing the input parameters
      :math:`\mathbf{p}_{in}` at the current iteration.
    - "delta_parameters": Numpy array representing the update
      :math:`\Delta\mathbf{p}_{in}^{*}` applied between the previous and the current
      iteration (after ``update_func`` if provided), None at the first iteration.
    - "costs": List of floats representing the cost function value (without weight)
      of each term at the current iteration: the ``cost_func`` of the term if provided,
      otherwise :math:`\frac{1}{2} \sum_j \rho_i(\| \mathbf{r}_{i,j}(\mathbf{p}_{out}) \|^2)`
      for ``rJ`` terms and ``0.0`` for ``gH`` terms.
    - "cost": Float representing the cost function value at the current iteration,
      computed as
      :math:`\frac{1}{2} \sum_i w_i \sum_j
      \rho_i(\| \mathbf{r}_{i,j}(\mathbf{p}_{out}) \|^2)`.
    - "delta_cost": Float representing the change of the cost function value
      between the previous and the current iteration (``C - C_previous``), None at the
      first iteration.
    - "optimality": Float representing the optimality value at the current iteration
      computed as ``norm(g, ord=numpy.inf)`` where :math:`g` is the scaled second term.
    - "residuals": A list of numpy arrays representing the raw residuals
      :math:`\mathbf{r}_i(\mathbf{p}_{out})` of each term at the current iteration, before
      the robust loss modification (None for ``gH`` terms).
    - "jacobians": A list of arrays representing the raw Jacobians
      :math:`\mathbf{J}_i = \partial \mathbf{r}_i / \partial \mathbf{p}_{out}` of each term at
      the current iteration, with respect to the output parameters (before the robust
      loss modification and the chain rule of the parametrization; None for ``gH`` terms).
    - "second_term": The scaled second term :math:`\mathbf{g}` of the linear system
      at the current iteration.
    - "hessian": The Hessian approximation :math:`\mathbf{H}` of the linear system
      at the current iteration.
    - "damping": Float representing the Levenberg-Marquardt damping :math:`\lambda`
      used for the accepted step leading to the current iteration (None at the first
      iteration and if ``damping`` is None).
    - "gain_ratio": Float representing the gain ratio of the accepted Levenberg-Marquardt
      step leading to the current iteration: the actual decrease of the cost divided by the
      decrease predicted by the Gauss-Newton model
      :math:`-\mathbf{g}^T \Delta\mathbf{p}_{in}^{*} - \frac{1}{2} \Delta\mathbf{p}_{in}^{*T} \mathbf{H} \Delta\mathbf{p}_{in}^{*}`
      (close to 1: the model is reliable; close to 0 or negative: the model is poor).
      NaN if the predicted decrease is not positive, None at the first iteration and if
      ``damping`` is None. Diagnostic only: it does not change the update of :math:`\lambda`.
    - "step_length": Float representing the step length :math:`\alpha` of the line
      search for the step leading to the current iteration (None at the first
      iteration and if ``line_search`` is False).
    - "all": include all the keys.

    The cost of each term can be compute only for ``rJ`` terms and is
    defined by :

    .. math::

        \frac{1}{2} \sum_j
        \rho_i\left(
            \left\|
            \mathbf{r}_{i,j}(P(\mathbf{p}_{in}))
            \right\|^2
        \right)

    .. important::

        For terms defined directly by a Hessian and gradient (``gH`` terms), only a local quadratic model
        of the cost is available. Since this model is defined up to an arbitrary constant and submit to
        cumulative errors, an absolute cost value cannot be uniquely determined.
        For reporting consistency, the cost contribution of these terms is set to ``0.0`` by default.
        If the term contains a ``cost_func`` function, it will be used to compute the cost of
        the term.

        A ``UserWarning`` is emitted for every call with a ``gH`` term without ``cost_func``
        (the reported cost does not include it), and the cost criteria ``atol`` and ``ftol``
        raise a ``ValueError`` in this case (with only such terms, ``atol`` would be satisfied
        at once without updating the parameters, and ``ftol`` never).

    """

    # Check the validity of the input arguments and raise appropriate errors if necessary.
    terms = _validate_terms(terms, Term)

    p0 = numpy.asarray(p0, dtype=numpy.float64)
    if p0.ndim != 1:
        raise ValueError(f"p0 must be a 1D array, got {p0.ndim} dimensions.")

    if parametrization is not None and not isinstance(parametrization, Parametrization):
        raise TypeError(
            "parametrization must be an instance of the Parametrization class."
        )

    options = _validate_solver_options(
        terms,
        max_iteration=max_iteration,
        max_time=max_time,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        atol=atol,
        ptol=ptol,
        callback_func=callback_func,
        update_func=update_func,
        damping=damping,
        lm_conf=lm_conf,
        line_search=line_search,
        line_search_conf=line_search_conf,
        verbosity=verbosity,
        history=history,
        history_details=history_details,
        history_length=history_length,
        implemented_history_details=_IMPLEMENTED_HISTORY_DETAILS,
        default_history_details=_DEFAULT_SOLVE_HISTORY,
    )
    max_iteration, max_time = options.max_iteration, options.max_time
    ftol, xtol, gtol, atol, ptol = options.ftol, options.xtol, options.gtol, options.atol, options.ptol
    damping, lm_conf, verbosity = options.damping, options.lm_conf, options.verbosity

    # ------ Solver Variables and functions
    # - constants
    n_terms = len(terms)
    n_parameters = p0.shape[0]
    config = options.config
    recorder = _HistoryRecorder(options)
    ls_conf = options.line_search_conf
    display = _IterationDisplay(verbosity, n_terms, options.use_lm, batch=False, use_line_search=options.use_line_search)

    # - recomputed each loop (reset to None at the start of each iteration)
    term_parameters: Optional[numpy.ndarray] = None
    jacobian_P = None
    state: Optional[SystemState] = None
    optimality: Optional[float] = None
    optimality_2: Optional[float] = None
    cond_Hessian: Optional[float] = None
    trace_Hessian: Optional[float] = None
    delta_norm: Optional[float] = None
    delta_norm_inf: Optional[float] = None
    parameters_norm: Optional[float] = None

    # - conserved between loops
    end_flag = False  # ! (end-flag is used to BREAK the optimization loop)
    stop_code = 0  # bit mask of the triggered stopping criteria (see _STOP_CODES)
    parameters = p0.copy()
    delta_parameters = None
    iteration = 0
    last_total_cost = None
    lm_lambda = None  # Levenberg-Marquardt damping (initialized at the first iteration)
    lm_lambda_floor = None  # minimum value of λ (initialized with λ)
    last_damping = None  # λ used for the accepted step leading to the current iteration
    last_gain_ratio = None  # gain ratio of the accepted LM step leading to the current iteration
    last_step_length = None  # α of the line search for the step leading to the current iteration
    n_rejected = 0  # total number of rejected Levenberg-Marquardt trial steps
    next_term_parameters = None  # p_out of the accepted LM trial step (reused at the next iteration)
    next_cost_state: Optional[CostState] = None  # cost of the accepted LM trial step (reused)

    # - functions
    has_naninf = lambda p: not numpy.all(numpy.isfinite(p))
    is_first_iteration = lambda: iteration == 0

    def apply_update_func(delta: numpy.ndarray) -> numpy.ndarray:
        # Apply update_func to a Gauss-Newton / Levenberg-Marquardt step and check its shape
        new_delta = numpy.asarray(update_func(parameters.copy(), delta.copy()), dtype=numpy.float64)
        if new_delta.ndim != 1 or new_delta.size != n_parameters:
            raise ValueError(f"update_func must return a 1D array with shape ({n_parameters},).")
        return new_delta

    # Check NaN or Inf values in the initial parameters before the first loop
    if has_naninf(parameters):
        stop_code |= _STOP_CODES["naninf_p0"]
        if verbosity >= 1:
            print("\n".join(_stop_reasons(stop_code, config)))
        return SolveResult(
            parameters=parameters,
            history=recorder.to_list(),
            success=False,
            stop_code=stop_code,
            n_iterations=0,
            cost=None,
            optimality=None,
            term_parameters=None,
            elapsed_time=0.0,
            n_rejected=0,
            config=config,
        )

    # Printing the header for the optimization process logging based on the verbosity level.
    display.print_header()

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

        # 1. ----- Apply the parametrization p_out = P(p_in)
        # (reuse p_out of the accepted Levenberg-Marquardt trial step: same p_in)
        if next_term_parameters is not None:
            term_parameters = next_term_parameters
        else:
            term_parameters = _compute_term_parameters(parametrization, parameters)

        # 2. ----- Check parametrization
        if has_naninf(term_parameters):
            end_flag = True
            stop_code |= _STOP_CODES["naninf_term_parameters"]
            break

        # 3-5. ----- Evaluate the system at p_out: r, J, c, H, g (chain rule with J_P)
        jacobian_P = _compute_jacobian_P(parametrization, parameters, term_parameters.shape[0])
        state = _evaluate_system(
            terms,
            term_parameters,
            jacobian_P,
            compute_cost=options.compute_cost,
            cost_state=next_cost_state,  # reuse r and costs of the accepted LM trial step
        )
        next_term_parameters = None
        next_cost_state = None

        # 6. ----- Stopping criterion and storing history
        elapsed_time = time.perf_counter() - starting_time

        # - precomputing
        if options.compute_optimality:
            optimality = float(numpy.linalg.norm(state.second_term, ord=numpy.inf))

        if options.compute_conv_analysis:
            optimality_2 = float(numpy.linalg.norm(state.second_term, ord=2))
            if scipy.sparse.issparse(state.hessian):
                trace_Hessian = float(state.hessian.diagonal().sum())
            else:
                trace_Hessian = float(numpy.trace(state.hessian))

            try:
                if scipy.sparse.issparse(state.hessian):
                    cond_Hessian = float(
                        scipy.sparse.linalg.norm(state.hessian)
                        * scipy.sparse.linalg.norm(scipy.sparse.linalg.inv(state.hessian))
                    )
                else:
                    cond_Hessian = float(numpy.linalg.cond(state.hessian))
            except Exception:
                cond_Hessian = numpy.nan

        if options.compute_delta_norm and not is_first_iteration():
            delta_norm = numpy.linalg.norm(delta_parameters, ord=2)
            delta_norm_inf = numpy.linalg.norm(delta_parameters, ord=numpy.inf)

        if options.compute_params_norm:
            parameters_norm = numpy.linalg.norm(parameters, ord=2)

        # - Update history (only the requested details are computed)
        recorder.record(iteration, {
            "iteration": lambda: iteration,
            "parameters": lambda: parameters.copy(),
            "delta_parameters": lambda: _copy_or_none(delta_parameters),
            "delta_cost": lambda: None if last_total_cost is None else state.cost - last_total_cost,
            "elapsed_time": lambda: elapsed_time,
            "costs": lambda: state.costs,
            "cost": lambda: state.cost,
            "optimality": lambda: optimality,
            "residuals": lambda: [_copy_or_none(r) for r in state.residuals],
            "jacobians": lambda: [_copy_or_none(J) for J in state.jacobians],
            "second_term": lambda: state.second_term.copy(),
            "hessian": lambda: state.hessian.copy(),
            "damping": lambda: last_damping,
            "gain_ratio": lambda: last_gain_ratio,
            "step_length": lambda: last_step_length,
        })

        # - Display the iteration
        display.print_row(
            iteration=iteration,
            elapsed_time=elapsed_time,
            cost=state.cost,
            delta_cost=None if last_total_cost is None else state.cost - last_total_cost,
            delta_norm=delta_norm,
            optimality=optimality,
            damping=last_damping,
            step_length=last_step_length,
            delta_norm_inf=delta_norm_inf,
            optimality_2=optimality_2,
            cond_hessian=cond_Hessian,
            trace_hessian=trace_Hessian,
            costs=state.costs,
        )

        # - Convergence analysis and stopping criteria check
        if options.use_ftol and not is_first_iteration():
            dF = last_total_cost - state.cost
            if 0 <= dF < ftol * state.cost:
                end_flag = True
                stop_code |= _STOP_CODES["ftol"]

        if options.use_atol:
            if state.cost < atol:
                end_flag = True
                stop_code |= _STOP_CODES["atol"]

        if options.use_gtol:
            if optimality < gtol:
                end_flag = True
                stop_code |= _STOP_CODES["gtol"]

        if options.use_xtol and not is_first_iteration():
            if delta_norm < xtol * (xtol + parameters_norm):
                end_flag = True
                stop_code |= _STOP_CODES["xtol"]

        if options.use_ptol and not is_first_iteration():
            if delta_norm_inf < ptol:
                end_flag = True
                stop_code |= _STOP_CODES["ptol"]

        if options.use_maxiter:
            if iteration >= max_iteration:
                end_flag = True
                stop_code |= _STOP_CODES["max_iteration"]

        if options.use_maxtime:
            if elapsed_time >= max_time:
                end_flag = True
                stop_code |= _STOP_CODES["max_time"]

        if options.use_callback:
            callback_state = {
                "parameters": parameters.copy(),
                "delta_parameters": None if is_first_iteration() else delta_parameters.copy(),
                "cost": state.cost,
                "second_term": state.second_term.copy(),
                "hessian": state.hessian,
            }
            callback_result = callback_func(callback_state)
            if not isinstance(callback_result, bool):
                raise ValueError("Callback function must return a boolean.")
            if not callback_result:
                end_flag = True
                stop_code |= _STOP_CODES["callback"]

        # 7. ----- Exit if any flag
        if end_flag:
            break

        # 8. ----- Compute the update Δp_in
        if not options.use_lm:
            # - Gauss-Newton: solve H Δp_in = -g
            try:
                delta_parameters = _solve_linear_system(state.hessian, state.second_term)
            except numpy.linalg.LinAlgError:
                end_flag = True
                stop_code |= _STOP_CODES["singular"]
                break

            # 9. ----- Compute the update parameters (Gauss-Newton only, inside the LM loop otherwise)
            if options.use_update:
                delta_parameters = apply_update_func(delta_parameters)

            # 9b. ----- Optional Armijo backtracking line search along Δp_in*
            #           (only the residuals / cost_func are evaluated at the trial steps)
            if options.use_line_search and not has_naninf(delta_parameters):
                if has_naninf(state.cost):
                    end_flag = True
                    stop_code |= _STOP_CODES["naninf_cost"]
                    break
                slope = float(numpy.dot(state.second_term, delta_parameters))  # g^T Δp_in*
                # Rounding noise of the cost: at the machine precision the cost cannot
                # decrease and the sign of g^T Δp_in* is meaningless (g ~ 0)
                noise = ls_conf["noise"] * abs(state.cost)
                if slope > noise:
                    end_flag = True
                    stop_code |= _STOP_CODES["not_descent"]
                    break

                step_length = 1.0
                accepted = False
                time_out = False
                for n_backtracks in range(ls_conf["max_backtracks"] + 1):
                    if n_backtracks > 0:
                        # max_time is also checked between the trial steps
                        if options.use_maxtime and time.perf_counter() - starting_time >= max_time:
                            time_out = True
                            break
                        step_length *= ls_conf["factor"]
                    trial_step = step_length * delta_parameters
                    trial_term_parameters = _compute_term_parameters(parametrization, parameters + trial_step)
                    if has_naninf(trial_term_parameters):
                        continue  # rejected: smaller α
                    trial_cost_state = _evaluate_cost(terms, trial_term_parameters)
                    if has_naninf(trial_cost_state.cost):
                        continue  # rejected: smaller α
                    if trial_cost_state.cost <= state.cost + ls_conf["c1"] * step_length * slope + noise:
                        accepted = True
                        break

                if time_out:
                    n_rejected += n_backtracks  # (trials 0 to n_backtracks - 1, all rejected)
                    end_flag = True
                    stop_code |= _STOP_CODES["max_time"]
                    break
                n_rejected += n_backtracks if accepted else n_backtracks + 1
                if not accepted:
                    end_flag = True
                    stop_code |= _STOP_CODES["line_search"]
                    break

                # Accepted step: reuse p_out and the cost at the next iteration (same p_in)
                delta_parameters = trial_step
                last_step_length = step_length
                next_term_parameters = trial_term_parameters
                next_cost_state = trial_cost_state

        else:
            # - Levenberg-Marquardt: solve (H + λ D) Δp_in = -g until the cost does not increase
            if has_naninf(state.cost):
                end_flag = True
                stop_code |= _STOP_CODES["naninf_cost"]
                break

            # Damping matrix D: identity ("lm") or diag(H) with a floor ("lm-diag")
            diagonal_H = numpy.asarray(state.hessian.diagonal(), dtype=numpy.float64)
            max_diagonal_H = float(numpy.max(diagonal_H)) if diagonal_H.size > 0 else 0.0
            if damping == "lm":
                diagonal_D = numpy.ones(n_parameters)
            else:
                floor = lm_conf["diag_floor"] * max_diagonal_H if max_diagonal_H > 0 else lm_conf["diag_floor"]
                diagonal_D = numpy.maximum(diagonal_H, floor)
            damping_matrix = (
                scipy.sparse.diags(diagonal_D, format="csc")
                if scipy.sparse.issparse(state.hessian)
                else numpy.diag(diagonal_D)
            )

            if lm_lambda is None:
                if damping == "lm":
                    lm_lambda = lm_conf["initial_scale"] * max_diagonal_H
                    lm_lambda = lm_lambda if lm_lambda > 0 else lm_conf["initial_scale"]
                else:
                    lm_lambda = lm_conf["initial_scale"]
                # Floor of λ: below machine precision × its scale, λ D is lost in the rounding of
                # H (and λ = 0 could never grow again by multiplication)
                lm_lambda_floor = _MACHINE_EPSILON * lm_lambda / lm_conf["initial_scale"]

            new_cost = numpy.inf
            nan_cost = False
            time_out = False
            n_rejections = 0
            trial_term_parameters = None
            trial_cost_state = None
            # A step is accepted if it does not increase the cost beyond its rounding noise
            lm_threshold = state.cost + lm_conf["noise"] * abs(state.cost)
            while new_cost > lm_threshold:
                if n_rejections > 0:
                    # max_time is also checked between the trial steps (not only between iterations)
                    if options.use_maxtime and time.perf_counter() - starting_time >= max_time:
                        time_out = True
                        break
                    lm_lambda *= lm_conf["factor"]
                if n_rejections >= lm_conf["max_rejections"]:
                    break
                n_rejections += 1

                try:
                    delta_parameters = _solve_linear_system(state.hessian + lm_lambda * damping_matrix, state.second_term)
                except numpy.linalg.LinAlgError:
                    continue  # rejected: larger λ

                # 9. ----- Compute the update parameters (applied to each trial step)
                if options.use_update:
                    delta_parameters = apply_update_func(delta_parameters)

                trial_term_parameters = _compute_term_parameters(parametrization, parameters + delta_parameters)
                if has_naninf(trial_term_parameters):
                    continue  # rejected: larger λ

                trial_cost_state = _evaluate_cost(terms, trial_term_parameters)
                new_cost = trial_cost_state.cost
                if has_naninf(new_cost):
                    nan_cost = True
                    break  # the step cannot be compared

            n_rejected += n_rejections - 1 if new_cost <= lm_threshold else n_rejections

            if time_out:
                # the step is not applied: the returned parameters are the last evaluated ones
                end_flag = True
                stop_code |= _STOP_CODES["max_time"]
                break

            if nan_cost:
                end_flag = True
                stop_code |= _STOP_CODES["naninf_trial_cost"]
                break

            if new_cost > lm_threshold:
                end_flag = True
                stop_code |= _STOP_CODES["lm"]
                break

            # Gain ratio: actual decrease / decrease predicted by the Gauss-Newton model
            # m(Δp) = C + g^T Δp + ½ Δp^T H Δp (valid for any Δp, also after update_func)
            predicted = -float(numpy.dot(state.second_term, delta_parameters)) - 0.5 * float(
                numpy.dot(delta_parameters, numpy.asarray(state.hessian @ delta_parameters).ravel())
            )
            last_gain_ratio = (state.cost - new_cost) / predicted if predicted > 0 else numpy.nan

            # Accepted step: reuse p_out and the cost at the next iteration (same p_in)
            last_damping = lm_lambda
            next_term_parameters = trial_term_parameters
            next_cost_state = trial_cost_state
            lm_lambda = max(lm_lambda / lm_conf["factor"], lm_lambda_floor)

        # 10. ----- Check update parameters
        if has_naninf(delta_parameters):
            end_flag = True
            stop_code |= _STOP_CODES["naninf_update"]
            break

        # 11. ----- Update the parameters
        parameters = parameters + delta_parameters

        if options.compute_cost:
            last_total_cost = state.cost

        iteration += 1

    # ---- End of solver
    elapsed_time = time.perf_counter() - starting_time

    # success: a convergence criterion and no failure (see SolveResult for the priority rules)
    success = bool(stop_code & _STOP_CONVERGENCE) and not (stop_code & _STOP_FAILURE)

    if verbosity >= 1:
        print("\n".join(_stop_reasons(stop_code, config)))

    # The returned parameters are always the last evaluated ones (the update is never
    # applied when the loop is stopped), so the final state describes them.
    cost = None if state is None else state.cost
    if success and cost is None:
        # The cost was not required during the optimization: computed once at the solution
        cost = _evaluate_cost(terms, term_parameters).cost

    return SolveResult(
        parameters=parameters,
        history=recorder.to_list(),
        success=success,
        stop_code=stop_code,
        n_iterations=iteration,
        cost=cost,
        optimality=None if state is None else float(numpy.linalg.norm(state.second_term, ord=numpy.inf)),
        term_parameters=term_parameters,
        elapsed_time=elapsed_time,
        n_rejected=n_rejected,
        config=config,
    )