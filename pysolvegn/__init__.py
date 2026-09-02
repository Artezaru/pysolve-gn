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

from .derivation import build_numerical_jacobian
from .term import Term
from .parametrization import Parametrization
from .solver import solve

from .implemented_parametrizations import (
    build_affine_parametrization,
    build_fixed_parametrization,
    build_sigmoid_parametrization,
    build_positive_parametrization,
)

from .implemented_regularizations import (
    build_squared_regularization,
    build_soft_squared_regularization,
)

# Deprecated
from .study_optimization import study_optimization
from .L_curve import perform_Lcurve_analysis
