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

from typing import Callable, Any
from numbers import Real, Integral
from numpy.typing import ArrayLike

import numpy
import scipy
import matplotlib.pyplot as plt

from .solver import solve_gauss_newton


def perform_Lcurve_analysis(
    residual_func: Callable,
    jacobian_func: Callable,
    residual_reg_func: Callable,
    jacobian_reg_func: Callable,
    parameters: ArrayLike,
    start_weight: Real,
    end_weight: Real,
    n_weights: Integral,
    *,
    n_labels: Integral = 10,
    loss: str = "linear",
    loss_reg: str = "linear",
    optimal: bool = False,
    **kwargs: Any,
) -> None:
    r"""
    Optimize the regularization factor using the L-curve method with the Gauss-Newton optimization method.

    The L-curve method is a graphical method for selecting the optimal regularization parameter
    in ill-posed problems. It consists in plotting the norm of the residuals against the norm
    of the regularization term for different values of the regularization parameter, and selecting
    the value that corresponds to the corner of the L-curve, which represents a good balance between
    fitting the data and regularizing the solution.

    The optimization is performed by minimizing the residuals between the transformed input points
    and the output points for each chain solving iteratively a linearized version of the problem
    at each iteration for different values of the regularization factor, and then selecting the
    optimal regularization factor based on the L-curve.


    Parameters
    ----------
    residual_func: Callable
        The function to compute the residuals for the data fitting term in the least squares problem.

    jacobian_func: Callable
        The function to compute the Jacobian matrix for the data fitting term in the least squares problem.

    residual_reg_func: Callable
        The function to compute the residuals for the regularization term in the least squares problem.

    jacobian_reg_func: Callable
        The function to compute the Jacobian matrix for the regularization term in the least squares problem.

    parameters: ArrayLike
        The current parameters of the optimization. Shape (n_parameters,).

    start_weight: Real
        The starting value of the regularization factor for the L-curve analysis.

    end_weight: Real
        The ending value of the regularization factor for the L-curve analysis.

    n_weights: Integral
        The number of regularization factors to evaluate between start_weight and end_weight.

    n_labels: Integral, optional
        The number of labels to display on the L-curve plot for the regularization factors.
        Default is 10.

    loss: str, optional
        The loss function to use for the data fitting term in the least squares problem. Default is "linear".

    loss_reg: str, optional
        The loss function to use for the regularization term in the least squares problem. Default is "linear".

    optimal: bool, optional
        If True, the function will compute the optimal regularization factor and parameters,
        and display them on the L-curve plot.

    **kwargs: Any
        Additional keyword arguments to pass to the optimization function for
        each regularization factor. This can include stopping criteria, verbosity level, logger, etc.


    Returns
    -------
    None
        The function does not return anything, but it prints the optimal regularization factor and parameters,
        and displays the L-curve plot.


    Notes
    -----
    To mathematically determine the L-curve’s corner, its curvature is derived,
    and the corner is defined as the point where the curvature is maximal.

    Lets note:

    .. math::

        \rho(\lambda) = ||R_{data}|| \quad \text{and} \quad \eta(\lambda) = ||R_{reg}||

    First we build the logarithmic L-curve such as:

    .. math::

        \hat{\rho}(\lambda) = \log(\rho(\lambda)) \quad \text{and} \quad \hat{\eta}(\lambda) = \log(\eta(\lambda))

    The L-curve is then defined as the parametric curve:

        \Gamma(\lambda) = (\hat{\rho}(\lambda)/2, \hat{\eta}(\lambda)/2)

    Then the curvature of the L-curve is defined as follows:

    .. math::

        \kappa(\lambda) = \frac{(\hat{\rho}/2)''(\hat{\eta}/2)' - (\hat{\rho}/2)'(\hat{\eta}/2)''}{((\hat{\rho}/2)'^2 + (\hat{\eta}/2)'^2)^{3/2}}

    .. math::

        \kappa(\lambda) = 2 \frac{\hat{\rho}''\hat{\eta}' - \hat{\rho}'\hat{\eta}''}{(\hat{\rho}'^2 + \hat{\eta}'^2)^{3/2}}

    """
    R_functions = [residual_func, residual_reg_func]
    J_functions = [jacobian_func, jacobian_reg_func]
    loss = [loss, loss_reg]

    weights = numpy.logspace(
        numpy.log10(start_weight), numpy.log10(end_weight), n_weights
    )

    verbosity = kwargs.get("verbosity", 0)

    results = []

    for index, weight in enumerate(weights):
        if verbosity > 0:
            print(
                f"\nEvaluating regularization weight: {weight:.3e} [{index + 1}/{n_weights}]"
            )

        out_parameters, history = solve_gauss_newton(
            residual_func=R_functions,
            jacobian_func=J_functions,
            parameters=parameters,
            weight=[1.0, weight],
            loss=loss,
            history=True,
            **kwargs,
        )

        cost_data = history[-1]["costs"][0]  # Cost of the data fitting term
        cost_reg = history[-1]["costs"][1]  # Cost of the regularization term
        results.append((weight, cost_data, cost_reg, out_parameters))

    weights, cost_data, cost_reg, parameters_list = zip(*results)

    rho_squares = numpy.array(cost_data)
    eta_squares = numpy.array(cost_reg)
    lambdas = numpy.array(weights)

    # --- Lcurve analysis to select the optimal factor2 ---
    rhos = numpy.sqrt(rho_squares)
    etas = numpy.sqrt(eta_squares)
    hat_rhos = numpy.log(rhos)
    hat_etas = numpy.log(etas)

    if optimal:
        rhos_prime = numpy.gradient(hat_rhos, lambdas)
        etas_prime = numpy.gradient(hat_etas, lambdas)
        rhos_double_prime = numpy.gradient(rhos_prime, lambdas)
        etas_double_prime = numpy.gradient(etas_prime, lambdas)

        kappas = (
            2
            * (rhos_double_prime * etas_prime - rhos_prime * etas_double_prime)
            / (rhos_prime**2 + etas_prime**2) ** 1.5
        )
        optimal_index = numpy.argmax(kappas)
        optimal_weight = weights[optimal_index]
        print(
            f"Optimal regularization weight (corner of L-curve): {optimal_weight:.3e}"
        )

    # Displaying the L-curve and the curvature
    plt.figure(figsize=(7, 5))
    plt.plot(hat_rhos / 2, hat_etas / 2, label="L-curve", marker="o", color="black")
    if optimal:
        plt.scatter(
            hat_rhos[optimal_index] / 2,
            hat_etas[optimal_index] / 2,
            color="red",
            label="Optimal Point",
            zorder=5,
        )
        plt.axvline(hat_rhos[optimal_index] / 2, color="red", linestyle="--")
        plt.axhline(hat_etas[optimal_index] / 2, color="red", linestyle="--")
    annotated_indices = range(0, len(weights), max(1, len(weights) // n_labels))
    for i in annotated_indices:
        plt.text(
            hat_rhos[i] / 2,
            hat_etas[i] / 2,
            f"{weights[i]:.1e}",
            fontsize=8,
            ha="right",
            va="bottom",
        )
    plt.xlabel(r"$\hat{\rho}(\lambda)/2$ = $\log(||R_{data}||)/2$")
    plt.ylabel(r"$\hat{\eta}(\lambda)/2$ = $\log(||R_{reg}||)/2$")
    plt.title("L-curve Analysis")
    plt.legend()
    plt.grid()
