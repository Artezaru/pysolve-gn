# pysolve-gn

## Description

Robust Gauss-Newton Least Squares Solver.

**pysolve-gn** is a Python package designed to solve the generalized nonlinear least
squares problem using the Gauss-Newton method. The package provides efficient algorithms
for solving nonlinear optimization problems, making it suitable for a wide range of
applications in data fitting, machine learning, and scientific computing.

Main features:

- **Multiple terms**: combine data terms and regularization terms, each with its own weight,
  defined either by residuals and Jacobian (`rJ`) or by gradient and Hessian (`gH`).
- **Robust loss functions**: `"linear"`, `"soft_l1"`, `"huber"`, `"cauchy"`, `"arctan"`,
  `"tukey"` or a custom loss, with a soft threshold `loss_scale` expressed in the unit of
  the residuals.
- **Levenberg-Marquardt damping**: `damping="lm"` or `"lm-diag"` for difficult problems.
- **Parametrization**: optimize the problem in another parameter space `p_out = P(p_in)`.
- **Finite differences**: numerical Jacobians when the analytical ones are not available.
- **Batch solver**: solve thousands of independent problems at once with `solve_batch`,
  each problem converging at its own pace.

## Examples

```python
import numpy as np
import pysolvegn

np.random.seed(0)


# Define the model function
def model(params, x):
    a, b = params
    return a * np.exp(b * x)


# Generate synthetic data points
x_data = np.linspace(0, 3, 100)
true_params = [2.5, 0.5]  # True parameters for the curve: y = a * exp(b * x)
y_true = model(true_params, x_data)
y_data = y_true + 0.5 * np.random.normal(size=y_true.shape)  # Add noise to the data


# Define the residual function
def residual_func(params):
    return model(params, x_data) - y_data


# Define the Jacobian function
def jacobian_func(params):
    a, b = params
    J = np.zeros((len(x_data), len(params)))
    J[:, 0] = np.exp(b * x_data)  # Derivative with respect to a
    J[:, 1] = a * x_data * np.exp(b * x_data)  # Derivative with respect to b
    return J


data_term = pysolvegn.Term.from_rJ(
    residual_func=residual_func,
    jacobian_func=jacobian_func,
    loss="linear",
    weight=1.0,
)

initial_params = np.array([2.0, 0.4])

result = pysolvegn.solve(
    terms=data_term,
    p0=initial_params,
    max_iteration=10,
    xtol=1e-6,
    ftol=1e-6,
    verbosity=2,
)

print(result.success)     # True if a convergence criterion was satisfied
print(result.message)   # Why the optimization stopped
print(result.parameters)  # Fitted parameters [a, b]
```

To reduce the influence of outliers, use a robust loss function with a soft threshold
`loss_scale` close to the expected noise level (in the unit of the residuals), and
Levenberg-Marquardt damping:

```python
robust_term = pysolvegn.Term.from_rJ(
    residual_func=residual_func,
    jacobian_func=jacobian_func,
    loss="cauchy",
    loss_scale=1.0,
)

result = pysolvegn.solve(
    terms=robust_term,
    p0=initial_params,
    max_iteration=50,
    ftol=1e-8,
    damping="lm-diag",
)
```

More examples are available in the [online documentation](https://Artezaru.github.io/pysolve-gn).

## Authors

- Artezaru <artezaru.github@proton.me>

- **Git Plateform**: https://github.com/Artezaru/pysolve-gn.git
- **Online Documentation**: https://Artezaru.github.io/pysolve-gn

## Installation

Install with pip

```
pip install pysolve-gn
```

Or :

```
pip install git+https://github.com/Artezaru/pysolve-gn.git
```

Then import the package with **pysolvegn**

## License

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