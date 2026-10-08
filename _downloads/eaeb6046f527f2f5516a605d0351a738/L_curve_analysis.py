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
# Create the data and regularization terms
# -----------------------------------------
#
# Secondly, we will define the residual function, and the
# Jacobian function. The residual function computes the difference between the
# observed data points and the model predictions, and the Jacobian function computes
# the derivatives of the residuals with respect to the parameters.
#
# This equation is called a **term** in pysolve-gn, and it represents a single
# component of the optimization problem. See :class:`pysolvegn.Term` for more details
# on how to define terms in pysolve-gn.
#
# The regularization term is built with
# :func:`pysolvegn.build_soft_squared_regularization`: it is null when the parameters
# are within ``thresholds`` of the estimated values, and increases quadratically
# outside this interval.


# Define the residual function
def residual_function(params):
    return model(params, x_data) - y_data


# Define the Jacobian function
def jacobian_function(params):
    a, b = params
    J = np.zeros((len(x_data), len(params)))
    J[:, 0] = np.exp(b * x_data)  # Derivative with respect to a
    J[:, 1] = a * x_data * np.exp(b * x_data)  # Derivative with respect to b
    return J


data_term = pysolvegn.Term.from_rJ(
    residual_func=residual_function,
    jacobian_func=jacobian_function,
    loss="linear",
    weight=1.0,
)

# Build the regularization term
estimated_params = np.array([2.2, 0.55])
estimated_stds = np.array([0.5, 0.2])
estimated_trust = np.array([0.1, 0.05])

reg_term = pysolvegn.build_soft_squared_regularization(
    means=estimated_params,
    thresholds=estimated_trust,
    stds=estimated_stds,
    loss="linear",
    weight=1.0,
)

# %%
# Perform L-curve analysis
# --------------------------
#
# Now we can perform L-curve analysis with :func:`pysolvegn.perform_Lcurve_analysis`.
# For each regularization weight, the problem is solved with :func:`pysolvegn.solve`
# (the additional keyword arguments, here the stopping criteria, are passed to it),
# and the costs of the data term and of the regularization term at the solution are
# used to build the L-curve.
#
# With ``optimal=True``, the corner of the L-curve (maximal curvature) is estimated
# and displayed.

weights = np.logspace(-2, 4, 50)  # Range of regularization weights to test

pysolvegn.perform_Lcurve_analysis(
    data_term=data_term,
    reg_term=reg_term,
    p0=estimated_params,
    reg_weights=weights,
    n_labels=20,
    optimal=True,
    max_iteration=10,
    xtol=1e-6,
    ftol=1e-6,
    verbosity=0,
)