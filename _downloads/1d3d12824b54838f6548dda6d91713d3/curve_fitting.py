"""
.. _sphx_glr__gallery_curve_fitting.py:

Curve fitting
=================

This example shows how to fit a curve to a set of data points using
pysolve-gn. We will use the Gauss-Newton method to minimize the
residuals between the observed data points and the curve defined by a
parametric function.

"""

# %%
# Create the residual and Jacobian functions
# -------------------------------------------
#
# First, we will define the model function, the residual function, and the
# Jacobian function. The model function defines the curve we want to fit, the
# residual function computes the difference between the observed data points and
# the model predictions, and the Jacobian function computes the derivatives of
# the residuals with respect to the parameters.

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

# %%
# Perform curve fitting
# ----------------------
#
# Now we can use the ``solve_gauss_newton`` function to perform the curve fitting.
# We will provide the residual function, the Jacobian function, and an initial
# guess for the parameters. The function will return the estimated parameters that
# best fit the data.

# Initial guess for the parameters
initial_params = [2.0, 0.4]

# Perform curve fitting using Gauss-Newton method
result = solve_gauss_newton(
    residual_func,
    jacobian_func,
    initial_params,
    max_iterations=10,
    xtol=1e-6,
    ftol=1e-6,
    verbosity=2,
    loss="linear",
)
print("Estimated parameters:", result)

y_retrieved = model(result, x_data)

# Display the results
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, label="Data Points", color="red", s=20)
plt.plot(x_data, y_true, label="True Curve", color="blue", linewidth=2)
plt.plot(
    x_data,
    y_retrieved,
    label="Fitted Curve",
    color="green",
    linestyle="--",
    linewidth=2,
)
plt.title("Curve Fitting using Gauss-Newton Method")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()
