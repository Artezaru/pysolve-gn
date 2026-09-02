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

from typing import Callable, Any, Optional
from numbers import Real, Integral
from numpy.typing import ArrayLike

import numpy
import scipy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .solver import solve
from .term import Term
from .parametrization import Parametrization


def perform_Lcurve_analysis(
    data_term: Term,
    reg_term: Term,
    p0: ArrayLike,
    reg_weights: ArrayLike,
    parametrization: Optional[Parametrization] = None,
    *,
    n_labels: Integral = 10,
    optimal: bool = False,
    logarithmic: bool = True,
    **kwargs: Any,
) -> None:
    r"""
    Analyze the regularization parameter using the L-curve method.

    The L-curve is a graphical method for selecting a regularization
    parameter in ill-posed or poorly conditioned least-squares problems.
    It consists in solving the regularized problem for a range of
    regularization weights and plotting the data-fitting norm against
    the regularization norm.

    For each regularization weight :math:`\lambda`, the function solves
    the corresponding regularized least-squares problem using
    :func:`pysolvegn.solve`.

    The regularization weight is varied over the values provided in
    ``reg_weights``. A separate Gauss-Newton optimization is performed
    for each value.

    If a parametrization is provided, the optimization is performed in
    the input parameter space of the parametrization, while the residual
    and Jacobian functions of the terms are evaluated using the
    corresponding output parameters.

    The parameter transformation is defined as:

    .. math::

        \mathbf{p}_{out} = P(\mathbf{p}_{in}).

    The resulting solutions are used to construct the L-curve, whose
    corner provides a heuristic estimate of a suitable regularization
    parameter.

    .. note::

        The L-curve is generally most informative when the regularization
        weights span several orders of magnitude. Providing logarithmically
        spaced values through ``reg_weights`` is therefore recommended.

    Parameters
    ----------
    data_term : Term
        The term representing the data-fitting part of the least-squares
        problem.
        This term is kept unchanged throughout the L-curve analysis.

    reg_term : Term
        The term representing the regularization part of the least-squares
        problem.
        Its weight is varied according to the values in ``reg_weights``.
        The term must be compatible with the :class:`Term` API used by
        :func:`pysolvegn.solve`.

    p0 : ArrayLike
        The initial input parameters for the optimization, with shape
        ``(n_parameters,)``.
        The same initial parameters are used for each regularization
        weight.
        If ``parametrization`` is ``None``, ``p0`` directly represents
        the parameters passed to the residual and Jacobian functions.
        If a parametrization is provided, ``p0`` represents the input
        parameters :math:`\mathbf{p}_{in}` of the parametrization.

    reg_weights : ArrayLike
        A one-dimensional array containing the regularization weights
        to evaluate for the L-curve analysis.
        Each value defines one regularized optimization problem and
        therefore one point of the L-curve.
        The values should normally be strictly positive.
        A logarithmically spaced range is recommended when
        ``logarithmic=True``.

    parametrization : Optional[Parametrization], optional (default=None)
        Parametrization defining the transformation from the input
        parameters optimized by :func:`pysolvegn.solve` to the output
        parameters passed to the residual and Jacobian functions of
        the terms.
        The same parametrization is used for every optimization
        performed during the L-curve analysis.

    n_labels : Integral, optional (default=10)
        Number of regularization-weight labels displayed on the L-curve
        plot.
        This controls only the number of labels displayed on the plot,
        not the number of regularization weights evaluated.

    optimal : bool, optional (default=False)
        If True, estimate the corner of the L-curve from its curvature.
        The estimated optimal regularization weight is displayed on the
        L-curve together with the corresponding solution.
        If False, the L-curve is displayed without selecting an optimal
        regularization weight.

    logarithmic : bool, optional (default=True)
        If True, display the regularization-weight axis using a
        logarithmic scale.
        This is recommended when ``reg_weights`` spans several orders
        of magnitude.

    **kwargs : Any
        Additional keyword arguments passed directly to
        :func:`pysolvegn.solve` for every regularization weight as convergence criterion.


    Returns
    -------
    None
        The function does not return the computed L-curve data or the
        optimized parameter vectors.
        It displays the L-curve and, when ``optimal=True``, reports the
        estimated optimal regularization weight and the corresponding
        solution.


    Notes
    -----
    For each regularization weight :math:`\lambda`, the function solves
    an independent regularized least-squares problem.

    The problem solved for a given :math:`\lambda` can be written as:

    .. math::

        \min_{\mathbf{p}_{in}}
        \frac{1}{2}
        \left[
            F_{data}
            \left(
                P(\mathbf{p}_{in})
            \right)
            +
            \lambda
            F_{reg}
            \left(
                P(\mathbf{p}_{in})
            \right)
        \right],

    where :math:`P` is the optional parametrization.

    The data and regularization terms therefore operate in the output
    parameter space, while the Gauss-Newton solver optimizes the input
    parameter space.

    For each regularization weight :math:`\lambda`, let

    .. math::

        \mathbf{p}_{in,\lambda}

    denote the optimized input parameter vector.

    If a parametrization is provided, the corresponding output
    parameters are:

    .. math::

        \mathbf{p}_{out,\lambda}
        =
        P(
            \mathbf{p}_{in,\lambda}
        ).

    To mathematically determine the L-curve’s corner, its curvature is derived,
    and the corner is defined as the point where the curvature is maximal.

    Lets note:

    .. math::

        \rho(\lambda) = \sqrt{\|R_{data}\|^2} \quad \text{and} \quad \eta(\lambda) = \sqrt{\|R_{reg}\|^2}

    First we build the logarithmic L-curve such as:

    .. math::

        \hat{\rho}(\lambda) = \log(\rho(\lambda)) \quad \text{and} \quad \hat{\eta}(\lambda) = \log(\eta(\lambda))


    The L-curve is then defined as the parametric curve:

    .. math::

        \Gamma(\lambda) = (\hat{\rho}(\lambda)/2, \hat{\eta}(\lambda)/2)

    Then the curvature of the L-curve is defined as follows:

    .. math::

        \kappa(\lambda) = \frac{(\hat{\rho}/2)''(\hat{\eta}/2)' - (\hat{\rho}/2)'(\hat{\eta}/2)''}{((\hat{\rho}/2)'^2 + (\hat{\eta}/2)'^2)^{3/2}}

    .. math::

        \kappa(\lambda) = 2 \frac{\hat{\rho}''\hat{\eta}' - \hat{\rho}'\hat{\eta}''}{(\hat{\rho}'^2 + \hat{\eta}'^2)^{3/2}}

    The optimal regularization factor is then defined as the value of :math:`\lambda`
    that maximizes the curvature :math:`\kappa(\lambda)`.

    .. warning::

        The L-curve method is a heuristic method for selecting the regularization parameter,
        and it may not always provide the optimal solution.


    Version
    -------

    - 0.0.1: Initial version.
    - 0.1.0: Refactored to use the new Term and solve API, added L-curve analysis and optimal regularization selection.
    - 0.3.0: Modifying function according to new nomenclature.

    """
    if not isinstance(data_term, Term) or not isinstance(reg_term, Term):
        raise ValueError("Both data_term and reg_term must be instances of Term.")
    if reg_term.type == "gH" or data_term.type == "gH":
        raise ValueError(
            "L-curve analysis is not applicable for gH terms. Must be rJ terms to compute the cost."
        )
    if parametrization is not None and not isinstance(parametrization, Parametrization):
        raise ValueError(
            "parametrization must be an instance of Parametrization or None."
        )

    if not isinstance(n_labels, Integral) or n_labels <= 0:
        raise ValueError("n_labels must be a positive integer.")
    if not isinstance(optimal, bool):
        raise ValueError("optimal must be a boolean value.")
    if not isinstance(logarithmic, bool):
        raise ValueError("logarithmic must be a boolean value.")

    p0 = numpy.asarray(p0, dtype=numpy.float64)
    if not p0.ndim == 1:
        raise ValueError("p0 must be a 1D array.")
    reg_weights = numpy.asarray(reg_weights, dtype=numpy.float64)
    if not reg_weights.ndim == 1:
        raise ValueError("reg_weights must be a 1D array.")

    verbosity = kwargs.get("verbosity", 0)
    _save_back_reg_weight = reg_term.weight
    results = []

    for index, weight in enumerate(reg_weights):
        if verbosity >= 1:
            print(
                f"\nEvaluating regularization weight: {weight:.3e} [{index + 1}/{len(reg_weights)}]"
            )

        reg_term.weight = weight

        out_parameters, history = solve(
            terms=[data_term, reg_term],
            p0=p0,
            parametrization=parametrization,
            history=True,
            history_details=["costs"],
            **kwargs,
        )

        cost_data = history[-1]["costs"][0]  # Cost of the data fitting term
        cost_reg = history[-1]["costs"][1]  # Cost of the regularization term
        results.append((weight, cost_data, cost_reg, out_parameters))

    reg_term.weight = _save_back_reg_weight

    weights, cost_data, cost_reg, parameters = zip(*results)

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
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(3, 4, figure=fig)
    ax_Lcurve = fig.add_subplot(gs[:, :3])

    ax_Lcurve.plot(
        hat_rhos / 2, hat_etas / 2, label="L-curve", marker="o", color="black"
    )
    if optimal:
        ax_Lcurve.scatter(
            hat_rhos[optimal_index] / 2,
            hat_etas[optimal_index] / 2,
            color="red",
            label="Optimal Point",
            zorder=5,
        )
        ax_Lcurve.axvline(hat_rhos[optimal_index] / 2, color="red", linestyle="--")
        ax_Lcurve.axhline(hat_etas[optimal_index] / 2, color="red", linestyle="--")
    annotated_indices = range(0, len(weights), max(1, len(weights) // n_labels))
    for i in annotated_indices:
        ax_Lcurve.text(
            hat_rhos[i] / 2,
            hat_etas[i] / 2,
            f"{weights[i]:.1e}",
            fontsize=8,
            ha="right",
            va="bottom",
        )
    ax_Lcurve.set_xlabel(r"$\hat{\rho}(\lambda)/2$ = $\log(||r_{data}||)/2$")
    ax_Lcurve.set_ylabel(r"$\hat{\eta}(\lambda)/2$ = $\log(||r_{reg}||)/2$")
    ax_Lcurve.set_title("L-curve Analysis")
    ax_Lcurve.legend()
    ax_Lcurve.grid()

    ax_lambda_rhos = fig.add_subplot(gs[0, 3])
    ax_lambda_rhos.plot(lambdas, rhos, marker="o", color="blue")
    if logarithmic:
        ax_lambda_rhos.set_xscale("log")
    if optimal:
        ax_lambda_rhos.axvline(optimal_weight, color="red", linestyle="--")
        ax_lambda_rhos.axhline(rhos[optimal_index], color="red", linestyle="--")
    ax_lambda_rhos.set_xlabel(r"Regularization Weight ($\lambda$)")
    ax_lambda_rhos.set_ylabel(r"$||r_{data}||$")
    ax_lambda_rhos.grid()

    ax_lambda_etas = fig.add_subplot(gs[1, 3])
    ax_lambda_etas.plot(lambdas, etas, marker="o", color="orange")
    if logarithmic:
        ax_lambda_etas.set_xscale("log")
    if optimal:
        ax_lambda_etas.axvline(optimal_weight, color="red", linestyle="--")
        ax_lambda_etas.axhline(etas[optimal_index], color="red", linestyle="--")
    ax_lambda_etas.set_xlabel(r"Regularization Weight ($\lambda$)")
    ax_lambda_etas.set_ylabel(r"$||r_{reg}||$")
    ax_lambda_etas.grid()

    if optimal:
        ax_lambda_kappas = fig.add_subplot(gs[2, 3])
        ax_lambda_kappas.plot(lambdas, kappas, marker="o", color="green")
        if logarithmic:
            ax_lambda_kappas.set_xscale("log")
        ax_lambda_kappas.axvline(optimal_weight, color="red", linestyle="--")
        ax_lambda_kappas.axhline(kappas[optimal_index], color="red", linestyle="--")
        ax_lambda_kappas.set_xlabel(r"Regularization Weight ($\lambda$)")
        ax_lambda_kappas.set_ylabel(r"Curvature ($\kappa$)")
        ax_lambda_kappas.grid()
    else:
        ax_lambda_norm_params = fig.add_subplot(gs[2, 3])
        ax_lambda_norm_params.plot(
            lambdas,
            [numpy.linalg.norm(p) for p in parameters],
            marker="o",
            color="purple",
        )
        if logarithmic:
            ax_lambda_norm_params.set_xscale("log")
        ax_lambda_norm_params.set_xlabel(r"Regularization Weight ($\lambda$)")
        ax_lambda_norm_params.set_ylabel(r"Norm of Parameters ($||p||$)")
        ax_lambda_norm_params.grid()

    plt.tight_layout()
    plt.show()
