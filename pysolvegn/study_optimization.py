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

from typing import Sequence, Optional

from numpy.typing import ArrayLike

import numpy
import scipy

from .term import Term
from .solver import solve
from .parametrization import Parametrization


def study_optimization(
    terms: Sequence[Term],
    p0: ArrayLike,
    parametrization: Optional[Parametrization] = None,
    *,
    title: str = "",
) -> None:
    r"""
    Study the Gauss-Newton Hessian built by the solver.

    This function is deliberately implemented on top of :func:`solve`.
    It does not rebuild the residuals, robust Jacobians, weights, losses,
    parametrization chain rule, or Hessian itself.

    The solver is executed with ``max_iteration=0`` and ``history=True``.
    Consequently, only the initial state is evaluated and no optimization
    step is performed.

    The analysis is therefore exactly consistent with the current solver
    implementation.

    The Hessian returned by the solver is:

    .. math::

        H =
        \sum_i w_i J_i^T J_i

    for ``rJ`` terms, and:

    .. math::

        H =
        \sum_i w_i H_i

    for ``gH`` terms.

    If a parametrization is provided to the solver through the terms'
    parameter functions, the Hessian is expressed in the corresponding
    input parameter space.

    The function displays:

    - the number of terms;
    - the number of equations and optimized parameters;
    - the individual and total costs;
    - the Hessian condition number;
    - the singular values of the Hessian;
    - the inverse singular values;
    - the estimated residual variance;
    - the parameter covariance and standard deviations;
    - the least observable parameter combinations;
    - the right singular vectors associated with the smallest singular values.

    Parameters
    ----------
    terms : Sequence[Term]
        Terms defining the least-squares problem.

    p0 : ArrayLike
        Initial input parameters passed to :func:`solve`.

    parametrization : Optional[Parametrization], optional
        Parametrization applied to the input parameters.

    title : str, optional
        Optional title printed before the analysis.

    Returns
    -------
    None
        This function only prints the analysis.

    Notes
    -----
    The function intentionally calls the solver with:

    ``max_iteration=0``

    This guarantees that the exact same implementation used during
    optimization is responsible for constructing the Hessian.

    The returned history contains:

    - ``parameters``
    - ``costs``
    - ``cost``
    - ``residuals``
    - ``second_term``
    - ``hessian``

    No optimization step is performed.

    """

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    if not isinstance(terms, Sequence):
        raise ValueError("terms must be a sequence of Term objects.")

    if len(terms) == 0:
        raise ValueError("terms sequence cannot be empty.")

    if not all(isinstance(term, Term) for term in terms):
        raise ValueError("All elements of terms must be instances of the Term class.")

    p0 = numpy.asarray(p0, dtype=numpy.float64)

    if p0.ndim != 1:
        raise ValueError(f"p0 must be a 1D array, got {p0.ndim} dimensions.")

    if parametrization is not None and not isinstance(parametrization, Parametrization):
        raise ValueError(
            "parametrization must be an instance of the Parametrization class or None."
        )

    if not isinstance(title, str):
        raise ValueError("title must be a string.")

    n_parameters = p0.size

    # ------------------------------------------------------------------
    # Ask the current solver to build the exact initial system.
    #
    # max_iteration=0 is intentional:
    # the solver evaluates iteration 0 and stops before solving
    # H * delta_parameters = -second_term.
    # ------------------------------------------------------------------

    _, history = solve(
        terms=terms,
        p0=p0,
        parametrization=parametrization,
        max_iteration=0,
        history=True,
        history_details=[
            "iteration",
            "parameters",
            "costs",
            "cost",
            "residuals",
            "second_term",
            "hessian",
        ],
    )

    if history is None or len(history) == 0:
        raise RuntimeError("The solver did not return the initial optimization state.")

    state = history[-1]

    # ------------------------------------------------------------------
    # Retrieve exactly what was built by the solver.
    # ------------------------------------------------------------------

    parameters = numpy.asarray(state["parameters"], dtype=numpy.float64)

    hessian = state["hessian"]

    costs = state["costs"]
    total_cost = state["cost"]

    residuals = state["residuals"]

    second_term = state["second_term"]

    # ------------------------------------------------------------------
    # Number of equations.
    #
    # Only rJ terms provide actual residual equations.
    # gH terms represent a local quadratic model and therefore do not
    # contribute to the residual degrees of freedom.
    # ------------------------------------------------------------------

    n_equations = 0

    for term, residual in zip(terms, residuals):

        if term.type == "rJ":

            if residual is None:
                raise RuntimeError(
                    "An rJ term returned no residual in the solver history."
                )

            n_equations += numpy.asarray(residual).size

    # ------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)

    if title:
        print(f"\n{title:^70}")
        print("-" * 70)

    # ------------------------------------------------------------------
    # Global information
    # ------------------------------------------------------------------

    print("\nSolver state")
    print("------------")

    print(
        "The following quantities are taken directly from the current "
        "Gauss-Newton solver."
    )

    print("\nNumber of terms       :", len(terms))
    print("Number of equations   :", n_equations)
    print("Number of parameters  :", n_parameters)

    print(f"Total cost            : " f"{total_cost:.6e}")

    print(
        f"||g||_inf             : "
        f"{numpy.linalg.norm(second_term, ord=numpy.inf):.6e}"
    )

    # ------------------------------------------------------------------
    # Individual term information
    # ------------------------------------------------------------------

    print("\nTerm information")
    print("----------------")

    print(
        f"{'Term':^10}"
        f"{'Type':^10}"
        f"{'Equations':^15}"
        f"{'Weight':^15}"
        f"{'Cost':^20}"
    )

    for i, (term, residual, cost) in enumerate(zip(terms, residuals, costs)):

        if term.type == "rJ":

            n_eq = numpy.asarray(residual).size

        else:

            n_eq = 0

        print(
            f"{i:^10}"
            f"{term.type:^10}"
            f"{n_eq:^15}"
            f"{term.weight:^15.6e}"
            f"{cost:^20.6e}"
        )

    # ------------------------------------------------------------------
    # Hessian conversion
    # ------------------------------------------------------------------

    if scipy.sparse.issparse(hessian):

        H = hessian.toarray()

    else:

        H = numpy.asarray(hessian, dtype=numpy.float64)

    if H.ndim != 2:
        raise RuntimeError(
            f"The solver Hessian must be a 2D matrix, got {H.ndim} dimensions."
        )

    if H.shape != (n_parameters, n_parameters):
        raise RuntimeError(
            "The Hessian shape is inconsistent with the number of "
            "optimized parameters: "
            f"H.shape={H.shape}, n_parameters={n_parameters}."
        )

    # ------------------------------------------------------------------
    # Hessian symmetry
    # ------------------------------------------------------------------
    #
    # Numerically, H should be symmetric because it is constructed as
    # J.T @ J or as a symmetric Hessian from a gH term.
    #
    # Symmetrizing here removes tiny numerical asymmetries without
    # changing the actual Gauss-Newton system in practice.
    # ------------------------------------------------------------------

    H = 0.5 * (H + H.T)

    # ------------------------------------------------------------------
    # Hessian analysis using SVD
    # ------------------------------------------------------------------

    print("\nHessian analysis")
    print("----------------")

    print("The Hessian below is the exact Gauss-Newton system built " "by the solver.")

    print("It is expressed in the input parameter space.")

    # ------------------------------------------------------------------
    # SVD
    #
    # H = U @ diag(S) @ Vt
    #
    # The columns of Vt.T are the right singular vectors.
    # These vectors represent directions in parameter space.
    # ------------------------------------------------------------------

    U, singular_values, Vt = numpy.linalg.svd(
        H,
        full_matrices=False,
    )

    if singular_values.size == 0:

        lambda_max = 0.0
        lambda_min = 0.0
        condition_number = float("inf")

    else:

        lambda_max = singular_values[0]

        non_zero = singular_values > 1e-12

        if numpy.any(non_zero):

            lambda_min = numpy.min(singular_values[non_zero])

            condition_number = (
                lambda_max / lambda_min if lambda_min > 0.0 else float("inf")
            )

        else:

            lambda_min = 0.0
            condition_number = float("inf")

    print(f"\nLargest singular value  : " f"{lambda_max:.6e}")

    print(f"Smallest singular value : " f"{lambda_min:.6e}")

    print(f"Condition number        : " f"{condition_number:.6e}")

    print(f"Trace(H)                : " f"{numpy.trace(H):.6e}")

    # ------------------------------------------------------------------
    # Singular values
    # ------------------------------------------------------------------

    print("\nSingular values of H")
    print("--------------------")

    print(f"{'Index':^10}" f"{'λ²':^20}" f"{'1/λ²':^20}")

    for i, value in enumerate(singular_values):

        if value > 1e-12:

            inverse = 1.0 / value

        else:

            inverse = float("inf")

        print(f"{i:^10}" f"{value:^20.6e}" f"{inverse:^20.6e}")

    # ------------------------------------------------------------------
    # Parameter sensitivity
    # ------------------------------------------------------------------

    print("\nParameter sensitivity")
    print("---------------------")

    degrees_of_freedom = n_equations - n_parameters

    print(f"Degrees of freedom    : " f"{degrees_of_freedom}")

    if degrees_of_freedom > 0:

        # The first term is assumed to represent the observational data,
        # consistently with the previous implementation.
        sigma2 = 2.0 * costs[0] / degrees_of_freedom

    else:

        sigma2 = float("inf")

    print(f"Estimated residual variance σ² : " f"{sigma2:.6e}")

    # ------------------------------------------------------------------
    # Covariance
    # ------------------------------------------------------------------
    #
    # Use the SVD rather than numpy.linalg.inv(H).
    #
    # This is numerically safer and naturally exposes poorly observable
    # directions.
    # ------------------------------------------------------------------

    covariance = None

    if numpy.isfinite(sigma2):

        if singular_values.size > 0:

            tolerance = (
                numpy.finfo(numpy.float64).eps * max(H.shape) * singular_values[0]
            )

            if numpy.all(singular_values > tolerance):

                covariance = sigma2 * Vt.T @ numpy.diag(1.0 / singular_values) @ Vt

            else:

                print(
                    "\nCovariance matrix cannot be computed "
                    "as a regular inverse because H is singular "
                    "or numerically rank deficient."
                )

        else:

            print("\nCovariance matrix cannot be computed " "because H is empty.")

    else:

        print(
            "\nCovariance matrix cannot be estimated "
            "because there are not enough degrees of freedom."
        )

    # ------------------------------------------------------------------
    # Parameter table
    # ------------------------------------------------------------------

    if covariance is not None:

        print(
            f"\n{'Parameter':^12}"
            f"{'Value':^20}"
            f"{'Variance':^20}"
            f"{'Std':^20}"
            f"{'Std / |Value|':^20}"
        )

        for i in range(n_parameters):

            variance = covariance[i, i]

            std = numpy.sqrt(max(variance, 0.0))

            if abs(parameters[i]) > 0.0:

                relative_std = std / abs(parameters[i])

            else:

                relative_std = float("inf")

            print(
                f"{i:^12}"
                f"{parameters[i]:^20.6e}"
                f"{variance:^20.6e}"
                f"{std:^20.6e}"
                f"{relative_std:^20.6e}"
            )

    # ------------------------------------------------------------------
    # Best and worst SVD parameter-space directions
    # ------------------------------------------------------------------

    print("\nSVD parameter-space directions")
    print("------------------------------")

    print("The right singular vectors of H describe directions in " "parameter space.")

    print(
        "Large singular values correspond to strongly observable "
        "directions, while small singular values correspond to "
        "weakly observable directions."
    )

    n_display = min(3, n_parameters // 2)

    print("\nBest observable directions")
    print("--------------------------")

    for mode in range(n_display):

        singular_value = singular_values[mode]
        vector = Vt[mode, :]

        print(f"\nMode {mode + 1}" f"  (λ² = {singular_value:.6e})")

        for j, coefficient in enumerate(vector):

            print(f"  p[{j}] : " f"{coefficient:+.6e}")

    print("\nWorst observable directions")
    print("---------------------------")

    for mode in range(1, n_display + 1):

        index = -mode

        singular_value = singular_values[index]
        vector = Vt[index, :]

        print(f"\nMode {mode}" f"  (λ² = {singular_value:.6e})")

        for j, coefficient in enumerate(vector):

            print(f"  p[{j}] : " f"{coefficient:+.6e}")

    print("\nRight singular vectors Vt")
    print("-------------------------")

    print("Each row of Vt is one parameter-space singular vector.")

    numpy.set_printoptions(
        precision=6,
        suppress=False,
    )

    print(Vt)

    # ------------------------------------------------------------------
    # End
    # ------------------------------------------------------------------

    print("\n" + "=" * 70 + "\n")
