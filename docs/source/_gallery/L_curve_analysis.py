"""
.. _sphx_glr__gallery_L_curve_analysis.py:

L-curve analysis
=================

This example shows how to perform L-curve analysis to estimate
the optimal regularization weight for an optimization problem.
We will use the Gauss-Newton method to solve a regularized
optimization problem for a range of regularization weights,
and then we will analyze the results to find the optimal weight.

"""

# %%
# Define the model function
# --------------------------
#
# First, we will define the model function that represents the curve we want to fit.
# In this example, we will use an exponential function defined as:
#
# .. math::
#
#     y = a \cdot e^{b \cdot x}
#
# We will also generate synthetic data points based on this model function, and we will
# add some noise and bias to the data to make the problem more realistic.

import numpy as np
import matplotlib.pyplot as plt
import pysolvegn


np.random.seed(0)


# Define the model function
def model(params, x):
    a, b = params
    return a * np.exp(b * x)


# Generate synthetic data points with lot of noise
x_data = np.linspace(0, 3, 100)
true_params = [2.5, 0.5]  # True parameters for the curve: y = a * exp(b * x)
y_true = model(true_params, x_data)
y_data = (
    y_true + 2.0 * np.random.normal(size=y_true.shape) + 0.2
)  # Add noise and bias to the data


# %%
# Create the residual and Jacobian functions
# -------------------------------------------
#
# Secondly, we will define the residual function, and the
# Jacobian function. The residual function computes the difference between the
# observed data points and the model predictions, and the Jacobian function computes
# the derivatives of the residuals with respect to the parameters.
#
# This equation is called a **term** in pysolve-gn, and it represents a single
# component of the optimization problem. See :class:`pysolvegn.Term` for more details
# on how to define terms in pysolve-gn.


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

# Build the regularization term
estimated_params = [2.2, 0.55]
estimated_stds = [0.5, 0.2]
estimated_trust = [0.1, 0.05]

residual_reg_func, jacobian_reg_func = pysolvegn.build_soft_squared_regularization(
    means=estimated_params, stds=estimated_stds, thresholds=estimated_trust
)

data_terms = pysolvegn.Term.from_rJ(
    residual_func=residual_func, jacobian_func=jacobian_func, loss="linear", weight=1.0
)
reg_term = pysolvegn.Term.from_rJ(
    residual_func=residual_reg_func,
    jacobian_func=jacobian_reg_func,
    loss="linear",
    weight=1.0,
)

# %%
# Perform L-curve analysis
# --------------------------
#
# Now we can perform L-curve analysis by solving the optimization problem for a range of
# regularization weights. We will use the ``solve`` function to solve the optimization
# problem for each weight, and we will store the results to analyze later.

weights = np.logspace(-2, 4, 50)  # Range of regularization weights to test

pysolvegn.perform_Lcurve_analysis(
    data_term=data_terms,
    reg_term=reg_term,
    x0=estimated_params,
    reg_weights=weights,
    max_iteration=10,
    xtol=1e-6,
    ftol=1e-6,
    verbosity=0,
    n_labels=20,
)
