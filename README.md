# pysolve-gn

## Description

Robust Gauss-Newton Least Squares Solver.

**pysolve-gn** is a Python package designed to solve the generalized nonlinear least 
squares problem using the Gauss-Newton method. The package provides efficient algorithms 
for solving nonlinear optimization problems, making it suitable for a wide range of 
applications in data fitting, machine learning, and scientific computing.

## Examples

```python
import numpy as np
import matplotlib.pyplot as plt
from pysolvegn import solve_gauss_newton

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
def residual_function(params, x, y):
    return model(params, x) - y


residual_func = lambda params: residual_function(params, x_data, y_data)


# Define the Jacobian function
def jacobian_function(params, x):
    a, b = params
    J = np.zeros((len(x), len(params)))
    J[:, 0] = np.exp(b * x)  # Derivative with respect to a
    J[:, 1] = a * x * np.exp(b * x)  # Derivative with respect to b
    return J


jacobian_func = lambda params: jacobian_function(params, x_data)

# Perform curve fitting using Gauss-Newton method
initial_params = [2.0, 0.4]

fitted_params = solve_gauss_newton(
    residual_func,
    jacobian_func,
    initial_params,
    max_iterations=10,
    xtol=1e-6,
    ftol=1e-6,
    verbosity=2,
    loss="linear",
)
```

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