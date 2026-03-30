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

__all__ = ["__version__"]

from .rho_functions import (
    linear_rho_at_R2,
    soft_l1_rho_at_R2,
    cauchy_rho_at_R2,
    arctan_rho_at_R2,
)

__all__.extend(
    [
        "linear_rho_at_R2",
        "soft_l1_rho_at_R2",
        "cauchy_rho_at_R2",
        "arctan_rho_at_R2",
    ]
)

from .solver import solve_gauss_newton

__all__.extend(["solve_gauss_newton"])

from .study_optimization import study_jacobian

__all__.extend(["study_jacobian"])

from .derivation import build_numerical_jacobian

__all__.extend(["build_numerical_jacobian"])

from .regularization import (
    build_squared_regularization,
    build_soft_squared_regularization,
)

__all__.extend(
    [
        "build_squared_regularization",
        "build_soft_squared_regularization",
    ]
)

from .L_curve import perform_Lcurve_analysis

__all__.extend(["perform_Lcurve_analysis"])
