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
import warnings

import numpy
import scipy.sparse
import scipy.sparse.linalg

from .implemented_conf import (
    _IMPLEMENTED_HISTORY_DETAILS,
    _IMPLEMENTED_DAMPINGS,
    _DEFAULT_SOLVE_HISTORY,
    _DEFAULT_LM_CONF,
    _STOP_CODES,
    _STOP_CONVERGENCE,
    _STOP_NANINF,
    _STOP_FAILURE,
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


# Text of each stopping reason (built from the stop code and the configuration of the
# solver: the thresholds are given, not the values reached at the last iteration).
_STOP_MESSAGES = {
    "ftol": "[ftol] Convergence achieved: 0 <= F_previous - F < ftol * F with ftol = {ftol}.",
    "atol": "[atol] Convergence achieved: F < atol with atol = {atol}.",
    "gtol": "[gtol] Convergence achieved: ||g||_inf < gtol with gtol = {gtol}.",
    "xtol": "[xtol] Convergence achieved: ||Δp|| < xtol * (xtol + ||p||) with xtol = {xtol}.",
    "ptol": "[ptol] Convergence achieved: ||Δp||_inf < ptol with ptol = {ptol}.",
    "max_iteration": "[max_iteration] Maximum number of iterations reached: {max_iteration}.",
    "max_time": "[max_time] Maximum computation time reached: {max_time} seconds.",
    "callback": "[callback] Optimization stopped by the callback function.",
    "singular": "[singular] Optimization stopped: the linear system H Δp_in = -g is singular.",
    "lm": "[lm] Optimization stopped: no step decreasing the cost found after {lm_max_rejections} rejections.",
    "naninf_p0": "[naninf] Optimization not started: NaN or Inf value in the initial parameters p0.",
    "naninf_term_parameters": "[naninf] Optimization stopped: NaN or Inf value in the term parameters p_out = P(p_in).",
    "naninf_cost": "[naninf] Optimization stopped: NaN or Inf value in the cost (the Levenberg-Marquardt steps cannot be compared).",
    "naninf_trial_cost": "[naninf] Optimization stopped: NaN or Inf value in the cost of a Levenberg-Marquardt trial step.",
    "naninf_update": "[naninf] Optimization stopped: NaN or Inf value in the parameter update Δp_in*.",
}


def _stop_reasons(stop_code: int, config: Dict[str, Any]) -> List[str]:
    r"""
    Build the text of the stopping reasons encoded in ``stop_code`` (one string per
    triggered bit, in the order of the checks of the solver).
    """
    values = dict(config)
    values["lm_max_rejections"] = config["lm_conf"]["max_rejections"]
    return [
        _STOP_MESSAGES[name].format(**values)
        for name, bit in _STOP_CODES.items()
        if stop_code & bit
    ]


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
           function or by a failure of the Levenberg-Marquardt step, even if a
           convergence criterion was satisfied at the same iteration.
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
        The total number of rejected Levenberg-Marquardt trial steps
        (0 if ``damping`` is None).

    config: Dict[str, Any]
        The configuration requested to the solver: the stopping criteria
        (``"max_iteration"``, ``"max_time"``, ``"ftol"``, ``"xtol"``, ``"gtol"``,
        ``"atol"``, ``"ptol"``, None if not used), ``"damping"`` and ``"lm_conf"``
        (the complete Levenberg-Marquardt configuration, default values included).

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


def _validate_lm_conf(lm_conf: Optional[Mapping[str, Real]]) -> Dict[str, Any]:
    r"""
    Complete the Levenberg-Marquardt configuration with the default values and check it.
    """
    if lm_conf is None:
        lm_conf = {}
    if not isinstance(lm_conf, Mapping):
        raise TypeError("lm_conf must be None or a dictionary.")
    unknown = set(lm_conf) - set(_DEFAULT_LM_CONF)
    if unknown:
        raise ValueError(f"Unknown lm_conf keys {sorted(unknown)}. Valid keys are {tuple(_DEFAULT_LM_CONF)}.")

    conf = dict(_DEFAULT_LM_CONF)
    conf.update(lm_conf)

    for key in ("initial_scale", "factor", "diag_floor"):
        if isinstance(conf[key], bool) or not isinstance(conf[key], Real):
            raise TypeError(f"lm_conf['{key}'] must be a real number.")
        conf[key] = float(conf[key])
        if not numpy.isfinite(conf[key]) or conf[key] <= 0:
            raise ValueError(f"lm_conf['{key}'] must be a finite strictly positive number.")
    if conf["factor"] <= 1.0:
        raise ValueError("lm_conf['factor'] must be strictly greater than 1.")
    if isinstance(conf["max_rejections"], bool) or not isinstance(conf["max_rejections"], Integral):
        raise TypeError("lm_conf['max_rejections'] must be an integer.")
    conf["max_rejections"] = int(conf["max_rejections"])
    if conf["max_rejections"] < 1:
        raise ValueError("lm_conf['max_rejections'] must be a strictly positive integer.")

    return conf


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
        on the computation time is considered.

    ftol: Optional[Real], optional (default=None)
        Stop criterion by the change of the cost function value.
        The optimization process is stopped when ``0 <= dF < ftol * F`` where F is the
        cost function value and ``dF = F_previous - F`` is the decrease of the cost
        function value between two iterations (an increase of the cost never satisfies
        this criterion). If None, this criterion is not considered.

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
        increase the cost: otherwise :math:`\lambda` is multiplied by 10 and the system
        is solved again. After an accepted step, :math:`\lambda` is divided by 10. If no
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

        The complete configuration used is stored in ``result.config["lm_conf"]``.

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
          or a failure of the Levenberg-Marquardt step (see :class:`SolveResult` for the
          priority rules).
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
        - ``n_rejected`` (int): the number of rejected Levenberg-Marquardt trial steps.
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

    """

    # Check the validity of the input arguments and raise appropriate errors if necessary.
    if isinstance(terms, Term):
        terms = [terms]
    if not isinstance(terms, Sequence):
        raise TypeError("terms must be a sequence of Term objects.")
    if len(terms) == 0:
        raise ValueError("terms sequence cannot be empty.")
    for term in terms:
        if not isinstance(term, Term):
            raise TypeError(
                "All elements of terms must be instances of the Term class."
            )

    p0 = numpy.asarray(p0, dtype=numpy.float64)
    if p0.ndim != 1:
        raise ValueError(f"p0 must be a 1D array, got {p0.ndim} dimensions.")

    if parametrization is not None and not isinstance(parametrization, Parametrization):
        raise TypeError(
            "parametrization must be an instance of the Parametrization class."
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
        history_details = list(_DEFAULT_SOLVE_HISTORY)
    if isinstance(history_details, str):
        history_details = [history_details]
    if not isinstance(history_details, Sequence):
        raise TypeError("history_details must be a string or a sequence of strings.")
    for detail in history_details:
        if detail not in _IMPLEMENTED_HISTORY_DETAILS and detail != "all":
            raise ValueError(
                f"Invalid history detail: {detail}. Valid details are: {_IMPLEMENTED_HISTORY_DETAILS} and 'all'."
            )
    if "all" in history_details:
        history_details = list(_IMPLEMENTED_HISTORY_DETAILS)

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
    n_parameters = p0.shape[0]
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
    history_list = (
        collections.deque(maxlen=-history_length)  # keeps only the last |M| entries
        if history_length is not None and history_length < 0
        else []
    )
    last_total_cost = None
    lm_lambda = None  # Levenberg-Marquardt damping (initialized at the first iteration)
    last_damping = None  # λ used for the accepted step leading to the current iteration
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
            history=list(history_list),
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
    printed_detail = f""
    printed_header = f""
    if verbosity >= 2:
        printed_detail += (
            f"\nIndividual costs [rJ term]: C_i = 0.5 * ρ(||r_i||^2) "
            f"\nCost: C = sum(w_i * C_i) "
            f"\nStep norm: ||Δp|| "
            f"\nOptimality: ||g|| "
        )
        printed_header += (
            f"\n{'Iteration':^10} {'Total time (s)':^15} {'Cost C':^15} {'ΔC':^15}"
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
            compute_cost=compute_cost,
            cost_state=next_cost_state,  # reuse r and costs of the accepted LM trial step
        )
        next_term_parameters = None
        next_cost_state = None

        # 6. ----- Stopping criterion and storing history
        elapsed_time = time.perf_counter() - starting_time

        # - precomputing
        if compute_optimality:
            optimality = float(numpy.linalg.norm(state.second_term, ord=numpy.inf))

        if compute_conv_analysis:
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

        if compute_delta_norm and not is_first_iteration():
            delta_norm = numpy.linalg.norm(delta_parameters, ord=2)
            delta_norm_inf = numpy.linalg.norm(delta_parameters, ord=numpy.inf)

        if compute_params_norm:
            parameters_norm = numpy.linalg.norm(parameters, ord=2)

        # - Update history
        if compute_history:

            h = {}
            if "iteration" in history_details:
                h["iteration"] = iteration
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
                    h["delta_cost"] = state.cost - last_total_cost
            if "elapsed_time" in history_details:
                h["elapsed_time"] = elapsed_time
            if "costs" in history_details:
                h["costs"] = state.costs
            if "cost" in history_details:
                h["cost"] = state.cost
            if "optimality" in history_details:
                h["optimality"] = optimality
            if "residuals" in history_details:
                h["residuals"] = [
                    r.copy() if r is not None else None for r in state.residuals
                ]
            if "jacobians" in history_details:
                h["jacobians"] = [
                    J.copy() if J is not None else None for J in state.jacobians
                ]
            if "second_term" in history_details:
                h["second_term"] = state.second_term.copy()
            if "hessian" in history_details:
                h["hessian"] = state.hessian.copy()
            if "damping" in history_details:
                h["damping"] = last_damping

            if history_length is None or history_length < 0:
                history_list.append(h)  # (deque with maxlen if history_length < 0)
            elif iteration < history_length:
                # Case 0 -> never append
                history_list.append(h)

        # - Display the iteration
        printed_row = f""
        if verbosity >= 2:
            if delta_parameters is None:
                strdp2 = f"{'':^15}"
                strdpinf = f"{'':^15}"
            else:
                strdp2 = f"{delta_norm:^15.3e}"
                strdpinf = f"{delta_norm_inf:^15.3e}"
            if last_total_cost is None:
                dC_str = f"{'':^15}"
            else:
                dC = state.cost - last_total_cost
                dC_str = f"{dC:^15.3e}"
            printed_row += (
                f"{iteration:^10} {elapsed_time:^15.3e} {state.cost:^15.3e} {dC_str}"
                + f" {strdp2} {optimality:^15.3e}"
            )
            if use_lm:
                printed_row += f" {'':^15}" if last_damping is None else f" {last_damping:^15.3e}"
        if verbosity >= 3:
            printed_row += (
                f" {strdpinf} {optimality_2:^15.3e}"
                + f" {cond_Hessian:^15.3e} {trace_Hessian:^15.3e}"
                + " ".join([f"{c:^15.3e}" for c in state.costs])
            )
        if verbosity >= 2:
            print(printed_row)

        # - Convergence analysis and stopping criteria check
        if use_ftol and not is_first_iteration():
            dF = last_total_cost - state.cost
            if 0 <= dF < ftol * state.cost:
                end_flag = True
                stop_code |= _STOP_CODES["ftol"]

        if use_atol:
            if state.cost < atol:
                end_flag = True
                stop_code |= _STOP_CODES["atol"]

        if use_gtol:
            if optimality < gtol:
                end_flag = True
                stop_code |= _STOP_CODES["gtol"]

        if use_xtol and not is_first_iteration():
            if delta_norm < xtol * (xtol + parameters_norm):
                end_flag = True
                stop_code |= _STOP_CODES["xtol"]

        if use_ptol and not is_first_iteration():
            if delta_norm_inf < ptol:
                end_flag = True
                stop_code |= _STOP_CODES["ptol"]

        if use_maxiter:
            if iteration >= max_iteration:
                end_flag = True
                stop_code |= _STOP_CODES["max_iteration"]

        if use_maxtime:
            if elapsed_time >= max_time:
                end_flag = True
                stop_code |= _STOP_CODES["max_time"]

        if use_callback:
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
        if not use_lm:
            # - Gauss-Newton: solve H Δp_in = -g
            try:
                delta_parameters = _solve_linear_system(state.hessian, state.second_term)
            except numpy.linalg.LinAlgError:
                end_flag = True
                stop_code |= _STOP_CODES["singular"]
                break

            # 9. ----- Compute the update parameters (Gauss-Newton only, inside the LM loop otherwise)
            if use_update:
                delta_parameters = apply_update_func(delta_parameters)

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

            new_cost = numpy.inf
            nan_cost = False
            n_rejections = 0
            trial_term_parameters = None
            trial_cost_state = None
            while new_cost > state.cost:
                if n_rejections > 0:
                    lm_lambda *= lm_conf["factor"]
                if n_rejections >= lm_conf["max_rejections"]:
                    break
                n_rejections += 1

                try:
                    delta_parameters = _solve_linear_system(state.hessian + lm_lambda * damping_matrix, state.second_term)
                except numpy.linalg.LinAlgError:
                    continue  # rejected: larger λ

                # 9. ----- Compute the update parameters (applied to each trial step)
                if use_update:
                    delta_parameters = apply_update_func(delta_parameters)

                trial_term_parameters = _compute_term_parameters(parametrization, parameters + delta_parameters)
                if has_naninf(trial_term_parameters):
                    continue  # rejected: larger λ

                trial_cost_state = _evaluate_cost(terms, trial_term_parameters)
                new_cost = trial_cost_state.cost
                if has_naninf(new_cost):
                    nan_cost = True
                    break  # the step cannot be compared

            n_rejected += n_rejections - 1 if new_cost <= state.cost else n_rejections

            if nan_cost:
                end_flag = True
                stop_code |= _STOP_CODES["naninf_trial_cost"]
                break

            if new_cost > state.cost:
                end_flag = True
                stop_code |= _STOP_CODES["lm"]
                break

            # Accepted step: reuse p_out and the cost at the next iteration (same p_in)
            last_damping = lm_lambda
            next_term_parameters = trial_term_parameters
            next_cost_state = trial_cost_state
            lm_lambda /= lm_conf["factor"]

        # 10. ----- Check update parameters
        if has_naninf(delta_parameters):
            end_flag = True
            stop_code |= _STOP_CODES["naninf_update"]
            break

        # 11. ----- Update the parameters
        parameters = parameters + delta_parameters

        if compute_cost:
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
        history=list(history_list),
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