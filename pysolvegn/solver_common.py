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

# Private tools shared by :func:`pysolvegn.solve` and :func:`pysolvegn.solve_batch`:
#
# - validation of the terms and of the options of the solvers (:func:`_validate_terms`,
#   :func:`_validate_solver_options`, :func:`_validate_lm_conf`),
# - text of the stopping reasons (:func:`_stop_reasons`),
# - storage of the history (:class:`_HistoryRecorder`),
# - display of the iterations (:class:`_IterationDisplay`),
# - shape checks of the user functions (:func:`_check_shape`).


import collections
import warnings
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import numpy

from .implemented_conf import (
    _IMPLEMENTED_DAMPINGS,
    _DEFAULT_LM_CONF,
    _DEFAULT_LINE_SEARCH_CONF,
    _STOP_CODES,
)


# ----------------------------------------------------------------------------
# Shape checks
# ----------------------------------------------------------------------------
def _check_shape(name: str, array, expected: tuple) -> None:
    r"""
    Raise a ValueError if ``array.shape`` (numpy array or scipy sparse matrix)
    is not ``expected``.
    """
    shape = getattr(array, "shape", None)
    if shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {shape}.")


# ----------------------------------------------------------------------------
# Stopping reasons
# ----------------------------------------------------------------------------
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
    "line_search": "[line_search] Optimization stopped: no step satisfying the Armijo condition found after {ls_max_backtracks} backtracks.",
    "not_descent": "[not_descent] Optimization stopped: the update is not a descent direction (g^T Δp_in* >= 0), the line search is impossible.",
}


def _stop_reasons(stop_code: int, config: Dict[str, Any]) -> List[str]:
    r"""
    Build the text of the stopping reasons encoded in ``stop_code`` (one string per
    triggered bit, in the order of the checks of the solver).
    """
    values = dict(config)
    values["lm_max_rejections"] = config["lm_conf"]["max_rejections"]
    values["ls_max_backtracks"] = config["line_search_conf"]["max_backtracks"]
    return [
        _STOP_MESSAGES[name].format(**values)
        for name, bit in _STOP_CODES.items()
        if stop_code & bit
    ]


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------
def _validate_terms(terms: Any, term_class: type) -> List[Any]:
    r"""
    Check that ``terms`` is a ``term_class`` instance or a non-empty sequence of
    ``term_class`` instances, and return it as a list.
    """
    name = term_class.__name__
    if isinstance(terms, term_class):
        terms = [terms]
    if not isinstance(terms, Sequence):
        raise TypeError(f"terms must be a sequence of {name} objects.")
    if len(terms) == 0:
        raise ValueError("terms sequence cannot be empty.")
    for term in terms:
        if not isinstance(term, term_class):
            raise TypeError(f"All elements of terms must be instances of the {name} class.")
    return list(terms)


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
    if isinstance(conf["noise"], bool) or not isinstance(conf["noise"], Real):
        raise TypeError("lm_conf['noise'] must be a real number.")
    conf["noise"] = float(conf["noise"])
    if not numpy.isfinite(conf["noise"]) or conf["noise"] < 0.0:
        raise ValueError("lm_conf['noise'] must be a finite non-negative number.")

    return conf


def _validate_line_search_conf(line_search_conf: Optional[Mapping[str, Real]]) -> Dict[str, Any]:
    r"""
    Complete the line search configuration with the default values and check it.
    """
    if line_search_conf is None:
        line_search_conf = {}
    if not isinstance(line_search_conf, Mapping):
        raise TypeError("line_search_conf must be None or a dictionary.")
    unknown = set(line_search_conf) - set(_DEFAULT_LINE_SEARCH_CONF)
    if unknown:
        raise ValueError(
            f"Unknown line_search_conf keys {sorted(unknown)}. Valid keys are {tuple(_DEFAULT_LINE_SEARCH_CONF)}."
        )

    conf = dict(_DEFAULT_LINE_SEARCH_CONF)
    conf.update(line_search_conf)

    for key in ("c1", "factor"):
        if isinstance(conf[key], bool) or not isinstance(conf[key], Real):
            raise TypeError(f"line_search_conf['{key}'] must be a real number.")
        conf[key] = float(conf[key])
        if not (0.0 < conf[key] < 1.0):
            raise ValueError(f"line_search_conf['{key}'] must be in the open interval (0, 1).")
    if isinstance(conf["max_backtracks"], bool) or not isinstance(conf["max_backtracks"], Integral):
        raise TypeError("line_search_conf['max_backtracks'] must be an integer.")
    conf["max_backtracks"] = int(conf["max_backtracks"])
    if conf["max_backtracks"] < 0:
        raise ValueError("line_search_conf['max_backtracks'] must be a non-negative integer.")
    if isinstance(conf["noise"], bool) or not isinstance(conf["noise"], Real):
        raise TypeError("line_search_conf['noise'] must be a real number.")
    conf["noise"] = float(conf["noise"])
    if not numpy.isfinite(conf["noise"]) or conf["noise"] < 0.0:
        raise ValueError("line_search_conf['noise'] must be a finite non-negative number.")

    return conf


def _validate_tolerance(name: str, value: Optional[Real], *, allow_zero: bool) -> Optional[float]:
    r"""
    Check an optional stopping threshold: None, or a real number ``> 0``
    (``>= 0`` if ``allow_zero``). Returns the value as a float.
    """
    if value is None:
        return None
    if not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    value = float(value)
    if allow_zero and value < 0:
        raise ValueError(f"{name} must be a non-negative real number.")
    if not allow_zero and value <= 0:
        raise ValueError(f"{name} must be a positive real number.")
    return value


@dataclass
class _SolverOptions:
    r"""
    Validated options of :func:`pysolvegn.solve` / :func:`pysolvegn.solve_batch` and the
    quantities to compute at each iteration (built by :func:`_validate_solver_options`).
    """

    # - requested options (validated)
    max_iteration: Optional[int]
    max_time: Optional[float]
    ftol: Optional[float]
    xtol: Optional[float]
    gtol: Optional[float]
    atol: Optional[float]
    ptol: Optional[float]
    callback_func: Optional[Callable]
    update_func: Optional[Callable]
    damping: Optional[str]
    lm_conf: Dict[str, Any]
    line_search: bool
    line_search_conf: Dict[str, Any]
    verbosity: int
    history: bool
    history_details: List[str]
    history_length: Optional[int]

    # - derived flags (filled in __post_init__)
    use_callback: bool = field(init=False)
    use_update: bool = field(init=False)
    use_xtol: bool = field(init=False)
    use_ptol: bool = field(init=False)
    use_atol: bool = field(init=False)
    use_ftol: bool = field(init=False)
    use_gtol: bool = field(init=False)
    use_maxiter: bool = field(init=False)
    use_maxtime: bool = field(init=False)
    use_lm: bool = field(init=False)
    use_line_search: bool = field(init=False)
    compute_cost: bool = field(init=False)
    compute_optimality: bool = field(init=False)
    compute_delta_norm: bool = field(init=False)
    compute_params_norm: bool = field(init=False)
    compute_conv_analysis: bool = field(init=False)

    def __post_init__(self):
        self.use_callback = self.callback_func is not None
        self.use_update = self.update_func is not None
        self.use_xtol = self.xtol is not None
        self.use_ptol = self.ptol is not None
        self.use_atol = self.atol is not None
        self.use_ftol = self.ftol is not None
        self.use_gtol = self.gtol is not None
        self.use_maxiter = self.max_iteration is not None
        self.use_maxtime = self.max_time is not None
        self.use_lm = self.damping in ("lm", "lm-diag")
        self.use_line_search = self.line_search

        self.compute_cost = (
            self.use_ftol or self.use_atol or self.use_callback or self.use_lm
            or self.use_line_search
            or self.verbosity >= 2
            or self.history_requires("cost", "costs", "delta_cost")
        )
        self.compute_optimality = (
            self.use_gtol or self.verbosity >= 2 or self.history_requires("optimality")
        )
        self.compute_delta_norm = self.use_xtol or self.use_ptol or self.verbosity >= 2
        self.compute_params_norm = self.use_xtol
        self.compute_conv_analysis = self.verbosity >= 3

    def history_requires(self, *details: str) -> bool:
        r"""True if the history is stored and contains at least one of ``details``."""
        return self.history and any(detail in self.history_details for detail in details)

    @property
    def config(self) -> Dict[str, Any]:
        r"""The configuration stored in the result (``result.config``)."""
        return {
            "max_iteration": self.max_iteration,
            "max_time": self.max_time,
            "ftol": self.ftol,
            "xtol": self.xtol,
            "gtol": self.gtol,
            "atol": self.atol,
            "ptol": self.ptol,
            "damping": self.damping,
            "lm_conf": self.lm_conf,
            "line_search": self.line_search,
            "line_search_conf": self.line_search_conf,
        }


def _validate_solver_options(
    terms: Sequence[Any],
    *,
    max_iteration: Optional[Integral],
    max_time: Optional[Real],
    ftol: Optional[Real],
    xtol: Optional[Real],
    gtol: Optional[Real],
    atol: Optional[Real],
    ptol: Optional[Real],
    callback_func: Optional[Callable],
    update_func: Optional[Callable],
    damping: Optional[str],
    lm_conf: Optional[Mapping[str, Real]],
    line_search: bool,
    line_search_conf: Optional[Mapping[str, Real]],
    verbosity: Integral,
    history: bool,
    history_details: Optional[Union[str, Sequence[str]]],
    history_length: Optional[Integral],
    implemented_history_details: Sequence[str],
    default_history_details: Sequence[str],
) -> _SolverOptions:
    r"""
    Check the options common to :func:`pysolvegn.solve` and :func:`pysolvegn.solve_batch`
    (everything except ``terms``, ``p0`` and ``parametrization``) and return them with
    the derived computation flags.

    ``terms`` (already validated) is only used to check that every ``gH`` term defines a
    ``cost_func`` with Levenberg-Marquardt damping.
    """
    # - stopping criteria
    if max_iteration is not None:
        if not isinstance(max_iteration, Integral):
            raise TypeError("max_iteration must be an integer.")
        max_iteration = int(max_iteration)
        if max_iteration < 0:
            raise ValueError("max_iteration must be a non-negative integer.")

    max_time = _validate_tolerance("max_time", max_time, allow_zero=True)
    ftol = _validate_tolerance("ftol", ftol, allow_zero=False)
    xtol = _validate_tolerance("xtol", xtol, allow_zero=False)
    gtol = _validate_tolerance("gtol", gtol, allow_zero=False)
    atol = _validate_tolerance("atol", atol, allow_zero=False)
    ptol = _validate_tolerance("ptol", ptol, allow_zero=False)

    # - 'gH' terms without cost_func: their cost is 0.0 by convention, so the total cost
    #   is wrong (always a warning) and the cost criteria are meaningless (error):
    #   with only such terms, atol is satisfied at once (C = 0 < atol, parameters never
    #   updated) and ftol never (0 <= dF < ftol * 0 is always False).
    no_cost_gH = [
        index for index, term in enumerate(terms) if term.type == "gH" and term.c_func is None
    ]
    if no_cost_gH:
        warnings.warn(
            f"The 'gH' term(s) {no_cost_gH} have no cost_func: their cost is set to 0.0, "
            "so the reported cost of the problem does not include them. Define a cost_func "
            "to get the true cost.",
            UserWarning,
            stacklevel=3,  # (user call of solve / solve_batch)
        )
        used = [name for name, value in (("atol", atol), ("ftol", ftol)) if value is not None]
        if used:
            raise ValueError(
                f"{' and '.join(used)} cannot be used when a 'gH' term has no cost_func "
                f"(term(s) {no_cost_gH}): the cost of these terms is unknown (0.0 by "
                "convention), so the cost criteria would be wrong. Define a cost_func for "
                "these terms or use another stopping criterion (gtol, xtol, ptol, "
                "max_iteration, max_time)."
            )

    if all(
        criterion is None
        for criterion in [max_iteration, max_time, ftol, xtol, gtol, atol, ptol]
    ):
        raise ValueError(
            "At least one stopping criterion must be provided "
            "(max_iteration, max_time, ftol, xtol, gtol, atol, or ptol)."
        )

    # - user functions
    if callback_func is not None and not callable(callback_func):
        raise TypeError("callback_func must be a callable function.")
    if update_func is not None and not callable(update_func):
        raise TypeError("update_func must be a callable function.")

    # - verbosity
    if not isinstance(verbosity, Integral):
        raise TypeError("verbosity must be an integer.")
    verbosity = int(verbosity)
    if verbosity < 0 or verbosity > 3:
        raise ValueError("verbosity must be an integer between 0 and 3 inclusive.")

    # - damping
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

    # - line search (Armijo backtracking, optional: the default is the pure Gauss-Newton step)
    if not isinstance(line_search, bool):
        raise TypeError("line_search must be a boolean value.")
    line_search_conf = _validate_line_search_conf(line_search_conf)
    if line_search:
        if damping is not None:
            raise ValueError(
                f"line_search=True cannot be combined with damping='{damping}': the "
                "Levenberg-Marquardt damping already controls the steps."
            )
        if no_cost_gH:
            raise ValueError(
                f"line_search=True requires the cost of every term: the 'gH' term(s) {no_cost_gH} "
                "must define a cost_func."
            )

    # - history
    if not isinstance(history, bool):
        raise TypeError("history must be a boolean value.")

    if history_details is None:
        history_details = list(default_history_details)
    if isinstance(history_details, str):
        history_details = [history_details]
    if not isinstance(history_details, Sequence):
        raise TypeError("history_details must be a string or a sequence of strings.")
    for detail in history_details:
        if detail not in implemented_history_details and detail != "all":
            raise ValueError(
                f"Invalid history detail: {detail}. Valid details are: {tuple(implemented_history_details)} and 'all'."
            )
    if "all" in history_details:
        history_details = list(implemented_history_details)
    history_details = list(history_details)

    if history_length is not None and not isinstance(history_length, Integral):
        raise TypeError(
            "history_length must be None or a positive or negative integer."
        )
    if history_length is not None:
        history_length = int(history_length)

    return _SolverOptions(
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
    )


# ----------------------------------------------------------------------------
# History
# ----------------------------------------------------------------------------
class _HistoryRecorder:
    r"""
    Storage of the history of a solver (``history``, ``history_details`` and
    ``history_length`` options).

    At each iteration, the solver calls :meth:`record` with a dictionary of getters
    ``{detail: callable}``: only the requested details are evaluated (and only if the
    iteration is kept), in the order of ``getters``.

    - ``history_length=None``: all the iterations are kept.
    - ``history_length=N >= 0``: only the iterations ``0`` to ``N - 1`` are kept.
    - ``history_length=-M < 0``: only the last ``M`` iterations are kept.
    """

    def __init__(self, options: _SolverOptions):
        self.enabled: bool = options.history
        self.details: List[str] = options.history_details
        self.length: Optional[int] = options.history_length
        self._entries = (
            collections.deque(maxlen=-self.length)  # keeps only the last |M| entries
            if self.length is not None and self.length < 0
            else []
        )

    def requires(self, detail: str) -> bool:
        r"""True if the history is stored and contains ``detail``."""
        return self.enabled and detail in self.details

    def keeps(self, iteration: int) -> bool:
        r"""True if the entry of ``iteration`` is stored."""
        if not self.enabled:
            return False
        return self.length is None or self.length < 0 or iteration < self.length

    def record(self, iteration: int, getters: Mapping[str, Callable[[], Any]]) -> None:
        r"""Store the requested details of ``iteration`` (no-op if not kept)."""
        if not self.keeps(iteration):
            return
        self._entries.append(
            {detail: getter() for detail, getter in getters.items() if detail in self.details}
        )

    def to_list(self) -> List[Dict]:
        r"""The stored entries as a list (``result.history``)."""
        return list(self._entries)


def _copy_or_none(value: Any) -> Any:
    r"""Copy of an array (None and scalars are returned as is)."""
    if value is None or not hasattr(value, "copy"):
        return value
    return value.copy()


# ----------------------------------------------------------------------------
# Display
# ----------------------------------------------------------------------------
class _IterationDisplay:
    r"""
    Display of the iterations of a solver (``verbosity >= 2``).

    For :func:`pysolvegn.solve_batch` (``batch=True``), an ``N processing`` column is
    added and the values given to :meth:`print_row` are means over the problems in
    processing.
    """

    _WIDTH = 15

    def __init__(self, verbosity: int, n_terms: int, use_lm: bool, batch: bool, use_line_search: bool = False):
        self.verbosity = verbosity
        self.n_terms = n_terms
        self.use_lm = use_lm
        self.batch = batch
        self.use_line_search = use_line_search

    def _cell(self, value: Optional[float]) -> str:
        width = self._WIDTH
        return f"{'':^{width}}" if value is None else f"{value:^{width}.3e}"

    def print_header(self) -> None:
        if self.verbosity < 2:
            return
        width = self._WIDTH
        detail = (
            "\nIndividual costs [rJ term]: C_i = 0.5 * ρ(||r_i||^2) "
            "\nCost: C = sum(w_i * C_i) "
            "\nStep norm: ||Δp|| "
            "\nOptimality: ||g|| "
        )
        if self.batch:
            detail += "\nAll the values are averaged over the N problems in processing."

        header = f"\n{'Iteration':^10}"
        if self.batch:
            header += f" {'N processing':^{width}}"
        header += (
            f" {'Total time (s)':^{width}} {'Cost C':^{width}} {'ΔC':^{width}}"
            f" {'||Δp||_2':^{width}} {'||g||_∞':^{width}}"
        )
        if self.use_lm:
            header += f" {'λ':^{width}}"
        if self.use_line_search:
            header += f" {'α':^{width}}"
        if self.verbosity >= 3:
            header += (
                f" {'||Δp||_∞':^{width}} {'||g||_2':^{width}} {'Cond(H)':^{width}} {'Trace(H)':^{width}}"
                + "".join(f" {'Cost C_' + str(i):^{width}}" for i in range(self.n_terms))
            )
        print(detail)
        print(header)

    def print_row(
        self,
        *,
        iteration: int,
        elapsed_time: float,
        cost: Optional[float],
        delta_cost: Optional[float],
        delta_norm: Optional[float],
        optimality: Optional[float],
        damping: Optional[float] = None,
        step_length: Optional[float] = None,
        n_processing: Optional[int] = None,
        delta_norm_inf: Optional[float] = None,
        optimality_2: Optional[float] = None,
        cond_hessian: Optional[float] = None,
        trace_hessian: Optional[float] = None,
        costs: Optional[Sequence[float]] = None,
    ) -> None:
        r"""Print one iteration (None values are displayed as blank cells)."""
        if self.verbosity < 2:
            return
        width = self._WIDTH
        row = f"{iteration:^10}"
        if self.batch:
            row += f" {n_processing:^{width}}"
        row += " " + " ".join(
            self._cell(value)
            for value in (elapsed_time, cost, delta_cost, delta_norm, optimality)
        )
        if self.use_lm:
            row += " " + self._cell(damping)
        if self.use_line_search:
            row += " " + self._cell(step_length)
        if self.verbosity >= 3:
            row += " " + " ".join(
                self._cell(value)
                for value in (delta_norm_inf, optimality_2, cond_hessian, trace_hessian)
            )
            row += "".join(" " + self._cell(c) for c in (costs or []))
        print(row)