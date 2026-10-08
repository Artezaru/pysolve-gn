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

_IMPLEMENTED_LOSS_FUNCTIONS = (
    "linear",
    "cauchy",
    "arctan",
    "soft_l1",
    "huber",
    "tukey",
)

_IMPLEMENTED_FINITE_DIFFERENCE_METHODS = (
    "central",
    "forward",
    "backward",
)

_IMPLEMENTED_HISTORY_DETAILS = (
    "iteration",
    "elapsed_time",
    "parameters",
    "delta_parameters",
    "cost",
    "delta_cost",
    "costs",
    "residuals",
    "jacobians",
    "optimality",
    "second_term",
    "hessian",
    "damping",
)

# Default history details of solve (history_details=None)
_DEFAULT_SOLVE_HISTORY = (
    "iteration",
    "elapsed_time",
    "parameters",
    "delta_parameters",
    "delta_cost",
    "cost",
    "optimality",
)

# Default history details of solve_batch (history_details=None)
_DEFAULT_BATCH_SOLVE_HISTORY = (
    "iteration",
    "elapsed_time",
    "n_processing",
    "parameters",
    "delta_parameters",
    "delta_cost",
    "cost",
    "optimality",
)

_IMPLEMENTED_BATCH_HISTORY_DETAILS = (
    "iteration",
    "elapsed_time",
    "n_processing",
    "is_processing",
    "parameters",
    "delta_parameters",
    "cost",
    "delta_cost",
    "costs",
    "residuals",
    "jacobians",
    "optimality",
    "second_term",
    "hessian",
    "damping",
)

_IMPLEMENTED_DAMPINGS = (
    None,
    "lm",
    "lm-diag",
)

# Levenberg-Marquardt settings (damping="lm" or "lm-diag")
_LM_INITIAL_SCALE = 1e-3  # initial lambda: _LM_INITIAL_SCALE * max(diag(H)) ("lm") or _LM_INITIAL_SCALE ("lm-diag")
_LM_FACTOR = 10.0  # lambda is multiplied (rejected step) or divided (accepted step) by this factor
_LM_MAX_REJECTIONS = 50  # maximum number of consecutive rejected steps in one iteration
_LM_DIAG_FLOOR = 1e-12  # (damping="lm-diag") floor of diag(H) relative to max(diag(H))

# Default Levenberg-Marquardt configuration (lm_conf argument of solve: missing keys use these values)
_DEFAULT_LM_CONF = {
    "initial_scale": _LM_INITIAL_SCALE,
    "factor": _LM_FACTOR,
    "max_rejections": _LM_MAX_REJECTIONS,
    "diag_floor": _LM_DIAG_FLOOR,
}

# Stop codes: one bit per stopping reason (SolveResult.stop_code is the OR of the triggered bits).
# The order of the dictionary is the order of the checks in the solver loop.
_STOP_CODES = {
    "ftol": 1 << 0,
    "atol": 1 << 1,
    "gtol": 1 << 2,
    "xtol": 1 << 3,
    "ptol": 1 << 4,
    "max_iteration": 1 << 5,
    "max_time": 1 << 6,
    "callback": 1 << 7,
    "singular": 1 << 8,
    "lm": 1 << 9,
    "naninf_p0": 1 << 10,
    "naninf_term_parameters": 1 << 11,
    "naninf_cost": 1 << 12,
    "naninf_trial_cost": 1 << 13,
    "naninf_update": 1 << 14,
}

# Groups of stop codes
_STOP_CONVERGENCE = (
    _STOP_CODES["ftol"] | _STOP_CODES["atol"] | _STOP_CODES["gtol"] | _STOP_CODES["xtol"] | _STOP_CODES["ptol"]
)
_STOP_NANINF = (
    _STOP_CODES["naninf_p0"]
    | _STOP_CODES["naninf_term_parameters"]
    | _STOP_CODES["naninf_cost"]
    | _STOP_CODES["naninf_trial_cost"]
    | _STOP_CODES["naninf_update"]
)
_STOP_FAILURE = _STOP_CODES["callback"] | _STOP_CODES["singular"] | _STOP_CODES["lm"] | _STOP_NANINF