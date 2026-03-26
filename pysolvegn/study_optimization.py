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

from typing import Union, Sequence, Callable, Optional
from numpy.typing import ArrayLike
from numbers import Real

import numpy
import scipy

from .rho_functions import (
    linear_rho_at_R2,
    soft_l1_rho_at_R2,
    cauchy_rho_at_R2,
    arctan_rho_at_R2,
    _build_tilde_R_and_tilde_J,
)


def study_jacobian(
    residual_func: Union[Callable, Sequence[Callable]],
    jacobian_func: Union[Callable, Sequence[Callable]],
    parameters: ArrayLike,
    *,
    weight: Optional[Union[str, Sequence[str]]] = None,
    loss: Optional[Union[str, Sequence[str]]] = None,
    title: str = "",
) -> None:
    r"""
    Study the Jacobian matrix of the least squares problem to analyze the observability
    of the parameters and the convergence properties of the optimization.

    This function computes the modified residuals and Jacobian using the robust cost
    function and then calls the internal function to analyze the Jacobian.

    Parameters
    ----------
    residual_func: Union[Callable, Sequence[Callable]]
        The function(s) to compute the residuals for each term in the least squares problem.
        Can be a single function or a sequence of functions.
        Each callable represent the `i`-th term :math:`\mathbf{R}_i(\mathbf{p})`
        in the least squares problem and should take as inputs the parameters
        (array-like with shape (n_parameters,)) and return the residuals as a
        1D numpy array with shape (n_residuals_i,) representing each
        :math:`\mathbf{R}_{i,j}(\mathbf{p})` where :math:`j=0,...,n_{\text{residuals}_i-1}`.

    jacobian_func: Union[Callable, Sequence[Callable]]
        The function(s) to compute the Jacobian matrices for each term in the
        least squares problem. Can be a single function or a sequence of functions.
        Each callable represent the `i`-th term :math:`\mathbf{J}_i(\mathbf{p})`
        in the least squares problem such as
        :math:`\mathbf{J}_i(\mathbf{p}) = \frac{\partial \mathbf{R}_i(\mathbf{p})}{\partial \mathbf{p}}`
        and should take as inputs the parameters (array-like with shape (n_parameters,))
        and return the Jacobian matrix as a 2D numpy array with shape
        (n_residuals_i, n_parameters) representing the Jacobian of each residual
        :math:`\mathbf{R}_{i,j}(\mathbf{p})` where :math:`j=0,...,n_{\text{residuals}_i-1}`.

    parameters: ArrayLike
        The parameters of the optimization step with shape (n_parameters,).

    weight: Optional[Union[Real, Sequence[Real]]], optional (default=None)
        The weight(s) for each term in the least squares problem.
        Can be a single value or a sequence of values.
        Each weight :math:`w_i` is a weight for the corresponding term in
        the least squares problem and should be a non-negative real number.
        If a single value is provided, it will be used for all terms in the
        least squares problem. If None is provided, it will be set to 1.0.

    loss: Optional[Union[str, Sequence[str]]], optional (default=None)
        The loss function(s) for each term in the least squares problem.
        Can be a single string or a sequence of strings.
        Each loss function :math:`\rho_i` is a robust cost function for the corresponding
        term in the least squares problem and should be one of the following strings:
        ["linear", "soft_l1", "cauchy", "arctan"].
        If a single string is provided, it will be used for all terms in the least
        squares problem. If None is provided, it will be set to "linear".

    title: str, optional
        An optional title for the analysis, which can be used for logging purposes.
        Default is an empty string.


    Returns
    -------
    None
        This function is used for analysis and does not return any value.


    Version
    -------
    0.0.1: Initial version.

    """
    if not isinstance(residual_func, Sequence):
        residual_func = [residual_func]
    if not isinstance(jacobian_func, Sequence):
        jacobian_func = [jacobian_func]

    if not len(residual_func) == len(jacobian_func):
        raise ValueError(
            "The length of residual_func and jacobian_func must be the same."
        )
    if not all(callable(func) for func in residual_func + jacobian_func):
        raise ValueError(
            "All elements in residual_func and jacobian_func must be callable."
        )

    if weight is None:
        weight = 1.0
    if not isinstance(weight, Sequence):
        weight = [weight]
    if len(weight) == 1:
        weight = weight * len(residual_func)
    if not len(weight) == len(residual_func):
        raise ValueError(
            "The length of weight must be the same as the length of residual_func and jacobian_func."
        )
    if not all(isinstance(w, Real) for w in weight):
        raise ValueError("All elements in weight must be real numbers.")
    weight = [float(w) for w in weight]  # Ensure weights are floats

    if loss is None:
        loss = "linear"
    if not isinstance(loss, Sequence):
        loss = [loss]
    loss = [l if l is not None else "linear" for l in loss]  # None -> "linear"
    if len(loss) == 1:
        loss = loss * len(residual_func)
    if not len(loss) == len(residual_func):
        raise ValueError(
            "The length of loss must be the same as the length of residual_func and jacobian_func."
        )
    if not all(isinstance(l, str) for l in loss):
        raise ValueError("All elements in loss must be strings.")
    valid_loss_functions = {"linear", "soft_l1", "cauchy", "arctan"}
    if not all(l in valid_loss_functions for l in loss):
        raise ValueError(f"All elements in loss must be one of {valid_loss_functions}.")

    parameters = numpy.asarray(parameters, dtype=numpy.float64)
    if not isinstance(parameters, numpy.ndarray) and parameters.ndim == 1:
        raise ValueError("Parameters must be a 1D numpy array.")

    if not isinstance(title, str):
        raise ValueError("Title must be a string.")

    # Prepare the robust cost functions for each term in the least squares problem
    rho_func = []
    for l in loss:
        if l == "linear":
            rho_func.append(linear_rho_at_R2)
        elif l == "soft_l1":
            rho_func.append(soft_l1_rho_at_R2)
        elif l == "cauchy":
            rho_func.append(cauchy_rho_at_R2)
        elif l == "arctan":
            rho_func.append(arctan_rho_at_R2)
        else:
            raise ValueError(
                f"Invalid loss function '{l}'. Valid options are 'linear', 'soft_l1', 'cauchy', 'arctan'."
            )

    _n_terms = len(residual_func)
    _n_parameters = parameters.size
    _parameters = parameters.copy()

    # Compute the residuals, Jacobian and robust cost function values for each term in the least squares problem
    residual_arrays = [R_func(_parameters) for R_func in residual_func]
    jacobian_matrices = [J_func(_parameters) for J_func in jacobian_func]
    rhos_arrays = [r_func(R) for r_func, R in zip(rho_func, residual_arrays)]

    # Cost computation
    costs = [0.5 * numpy.sum(rho[0]) for rho in rhos_arrays]
    total_cost = sum(w * c for w, c in zip(weight, costs))

    # Build tilted residuals and Jacobian based on the robust cost function
    for i in range(_n_terms):
        residual_arrays[i], jacobian_matrices[i] = _build_tilde_R_and_tilde_J(
            residual_arrays[i],
            jacobian_matrices[i],
            rhos_arrays[i][1],
            rhos_arrays[i][2],
        )

    # ----------- Title print ----------------
    print("\n" + "=" * 50)
    print("\n" + "-" * 50)
    print(f"{title:^50}")
    print("-" * 50 + "\n")

    # ----------- Global informations -----------
    print(f"Number of terms in the least squares problem: {_n_terms}")
    _n_equations = residual_arrays[0].size
    print(f"Number of equations in the first term (assuming data): {_n_equations}")
    print(f"Number of parameters: {_n_parameters}")
    print(f"\nTotal cost value C = ½Σ w*ρ(|R|²): {total_cost:.3e}")

    # ----------- Sub term contributions -----------
    header = f"\n{'LS Term':^10} {'Nequations':^12} {'Nparams':^10} {'Density (%)':^15} {'Loss ρ':^10} {'Weight w':^10} {'Cost ½ρ(|R|²)':^15} {'Cost (%)':^10}"
    print(header)
    for i, J in enumerate(jacobian_matrices):
        if scipy.sparse.issparse(J):
            n_residuals_i, n_parameters_i = J.shape
            density = 100.0 * J.nnz / (n_residuals_i * n_parameters_i)
        else:
            n_residuals_i, n_parameters_i = J.shape
            density = 100.0 * numpy.count_nonzero(J) / (n_residuals_i * n_parameters_i)
        cost_i = costs[i]
        weight_i = weight[i]
        loss_i = loss[i]
        cost_percent = int(
            round(
                100.0 * cost_i * weight_i / total_cost if total_cost > 0 else "inf", 0
            )
        )
        row = f"{i:^10} {n_residuals_i:^12} {n_parameters_i:^10} {density:^15.2f} {loss_i:^10} {weight_i:^10.2e} {cost_i:^15.3e} {cost_percent:^10.2f}"
        print(row)

    # ----------- Singular values analysis -----------
    M = sum(w * (J.T @ J) for w, J in zip(weight, jacobian_matrices))
    M = M.toarray() if scipy.sparse.issparse(M) else M
    U, S, Vt = numpy.linalg.svd(M, full_matrices=False)
    s_max = S[0]
    s_min = S[-1]
    cond = s_max / s_min if s_min > 1e-12 else float("inf")

    print(f"\nSingular values λ² (max/min) of Σ wJᵀJ : {s_max:.3e}/{s_min:.3e}")
    print(f"Condition number (max λ² / min λ²): {cond:.3e}")
    header = f"\n{'λ² Index':^10} {'λ² Value':^15} {'Var 1/λ²':^15}"
    print(header)
    for i, s in enumerate(S):
        var = 1.0 / s if s > 1e-12 else float("inf")  # Avoid division by zero
        print(f"{i:^10} {s:^15.3e} {var:^15.3e}")

    # ----------- Parameter sensitivity analysis -----------
    sigma_2 = (
        2 * total_cost / (_n_equations - _n_parameters)
        if _n_equations > _n_parameters
        else float("inf")
    )
    cov = (
        sigma_2 * numpy.linalg.inv(M) if _n_equations > _n_parameters else float("inf")
    )
    print(f"\nVt = (right singular vectors) of the combined Jacobian matrix Σ wJᵀJ:")
    print(f"Estimated residual variance σ² = 2C/(Neq-Np): {sigma_2:.3e}")

    indices = [0, -2, -1] if _n_parameters >= 3 else list(range(_n_parameters))
    header = f"\n{'Param Index':^15} {'Value P':^10} {'Var V=σ²M⁻¹':^15} {'Ratio √V/|P|':^15}"
    for i in indices:
        header += f" {'Vt[' + str(i) + ']':^15}"
    print(header)

    for j in range(_n_parameters):
        var = cov[j, j] if _n_equations > _n_parameters else float("inf")
        ratio = (
            numpy.sqrt(var) / abs(parameters[j]) if parameters[j] != 0 else float("inf")
        )
        row = f"{j:^15} {parameters[j]:^10.3e} {var:^15.3e} {ratio:^15.3e}"
        for i in indices:
            row += f" {Vt[i, j]:^15.3e}"
        print(row)

    # ----------- End of analysis -----------
    print("\n" + "=" * 50 + "\n")
