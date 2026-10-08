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

from .__version__ import __version__

from .derivation import (
    build_numerical_jacobian,
    build_batch_numerical_jacobian,
)

# Single problem
from .term import Term
from .parametrization import Parametrization
from .solver import solve, SolveResult

# Batch of independent problems
from .batch_term import BatchTerm
from .batch_parametrization import BatchParametrization
from .batch_solver import solve_batch, BatchSolveResult

from .implemented_parametrizations import (
    build_affine_parametrization,
    build_fixed_parametrization,
    build_sigmoid_parametrization,
    build_positive_parametrization,
    build_batch_affine_parametrization,
    build_batch_fixed_parametrization,
    build_batch_sigmoid_parametrization,
    build_batch_positive_parametrization,
)

from .implemented_regularizations import (
    build_squared_regularization,
    build_soft_squared_regularization,
    build_absolute_regularization,
    build_batch_squared_regularization,
    build_batch_soft_squared_regularization,
    build_batch_absolute_regularization,
)

from .loss_functions import (
    linear_rho,
    soft_l1_rho,
    cauchy_rho,
    arctan_rho,
    huber_rho,
    tukey_rho,
    scale_rho_function,
    get_rho_function_by_name,
)

# Deprecated
from .study_optimization import study_optimization
from .L_curve import perform_Lcurve_analysis